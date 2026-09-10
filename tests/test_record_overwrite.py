"""Hardware-free overwrite regression: run with the LeRobot Python environment."""

import hashlib
import io
import json
import pickle
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pyarrow.parquet as pq

from test_lerobot_v30 import RecordingWriter, check_target, frame, import_dataset
from lerobot_v30 import frame_features, hub_cleanup_marker, push_dataset, push_replacement, writer_main


def main():
    with patch("huggingface_hub.HfApi") as api:
        dataset = Mock(repo_id="local/overwrite", root=Path("unused"))
        push_replacement(dataset, True)
        assert api.return_value.upload_folder.call_args.kwargs["delete_patterns"] == [
            "data/**", "meta/episodes/**", "videos/**",
        ]
        dataset.push_to_hub.assert_called_once_with(private=True)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "dataset"
        config = dict(
            root=str(root), repo_id="local/overwrite", episode_index=None, fps=50,
            robot_type="aloha", camera_names=["cam_high"], include_velocity=True,
            include_effort=True, push_to_hub=False, private=False,
        )

        def record(index, value, length=3, finish=True):
            writer = RecordingWriter(sys.executable, dict(config, episode_index=index), frame(value))
            try:
                for _ in range(length):
                    writer.add_frame(frame(value, f"task {value}"))
                if finish:
                    writer.finish()
            finally:
                writer.close()

        def snapshot():
            return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).digest()
                    for p in root.rglob("*") if p.is_file()}

        # One writer packs episodes into shared files, unlike three separate resumptions.
        ds = import_dataset().create(
            repo_id=config["repo_id"], root=root, fps=50, robot_type="aloha",
            features=frame_features(frame(0)), use_videos=True, image_writer_threads=1,
        )
        try:
            for value in range(3):
                for _ in range(3):
                    ds.add_frame(frame(value, f"task {value}"))
                ds.save_episode()
        finally:
            ds.finalize()
        # Legacy episode metadata may lack quantiles even when stats.json has them.
        for path in (root / "meta/episodes").rglob("*.parquet"):
            table = pq.read_table(path)
            pq.write_table(table.drop([key for key in table.column_names if "/q" in key]), path)
        before = snapshot()
        for index in (-1, 4):
            try:
                check_target(dict(config, episode_index=index))
            except ValueError as exc:
                assert "out of range" in str(exc)
            else:
                raise AssertionError("Accepted invalid index")
        record(1, 10, finish=False)
        assert snapshot() == before, "Interrupted overwrite changed original files"
        record(1, 10, length=2)
        assert not list(root.parent.glob(".dataset.overwrite-*"))
        video_hashes = [hashlib.sha256(p.read_bytes()).digest() for p in (root / "videos").rglob("*.mp4")]
        for path, digest in before.items():
            if path.startswith("videos/"):
                assert digest in video_hashes, "Existing video was re-encoded"

        def verify(values):
            ds = import_dataset()(repo_id=config["repo_id"], root=root, video_backend="pyav")
            assert len(ds) == len(values)
            for i, value in enumerate(values):
                sample = ds[i]
                np.testing.assert_allclose(sample["observation.state"], value)
                np.testing.assert_allclose(sample["action"], value + 1)
                assert sample["task"] == f"task {value}"
                assert sample["observation.images.cam_high"][0].mean() > 0.6
                assert int(sample["index"]) == i
            info = json.loads((root / "meta/info.json").read_text())
            assert info["total_frames"] == len(values)
            stats = json.loads((root / "meta/stats.json").read_text())
            np.testing.assert_allclose(stats["observation.state"]["mean"], np.mean(values))
            np.testing.assert_allclose(stats["index"]["mean"], (len(values) - 1) / 2)
            for key in ("observation.state", "action"):
                for quantile in ("q01", "q99"):
                    expected = np.mean(values) + (key == "action")
                    np.testing.assert_allclose(stats[key][quantile], expected, atol=1e-5)
            return info

        assert verify([0]*3 + [10]*2 + [2]*3)["total_episodes"] == 3
        # Both boundary replacements and subsequent appends retain contiguous indices.
        record(0, 20, length=4)
        record(2, 30, length=1)
        record(3, 40, length=2)
        assert verify([20]*4 + [10]*2 + [30] + [40]*2)["total_episodes"] == 4
        record(None, 50, length=1)
        assert verify([20]*4 + [10]*2 + [30] + [40]*2 + [50])["total_episodes"] == 5
        # An offline replacement followed by appends still requires remote shard cleanup.
        marker = hub_cleanup_marker(root)
        assert marker.exists()
        with patch("huggingface_hub.HfApi") as api:
            uploaded = Mock(repo_id=config["repo_id"], root=root)
            api.return_value.upload_folder.side_effect = OSError("offline")
            try:
                push_dataset(uploaded)
            except OSError:
                pass
            else:
                raise AssertionError("Expected upload failure")
            assert marker.exists()
            api.return_value.upload_folder.side_effect = None
            push_dataset(uploaded)
            assert not marker.exists()
            assert api.return_value.upload_folder.call_args.kwargs["delete_patterns"] == [
                "data/**", "meta/episodes/**", "videos/**",
            ]
            calls = api.return_value.upload_folder.call_count
            push_dataset(uploaded)
            assert api.return_value.upload_folder.call_count == calls
        before = snapshot()
        def saved_capture(config, index):
            config["_save_started"] = True
            (Path(config["root"]) / "saved-capture").write_text("new capture")
        input_config = dict(config, episode_index=0)
        with patch.object(sys, "stdin", SimpleNamespace(buffer=io.BytesIO(pickle.dumps(input_config)))), \
                patch("lerobot_v30.record_dataset", side_effect=saved_capture), \
                patch("lerobot_v30.replace_appended_episode", side_effect=OSError("disk full")):
            try:
                writer_main()
            except OSError:
                pass
            else:
                raise AssertionError("Expected rebuild failure")
        assert snapshot() == before
        assert len(list(root.parent.glob(".dataset.overwrite-*/dataset/saved-capture"))) == 1
    print("OK: overwrite, interrupted capture, index bounds, video preservation, stats and append")


if __name__ == "__main__":
    main()
