"""Run with the LeRobot Python: python tests/test_convert_resume.py. No Hub writes."""

import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import convert_hdf5_to_lerobot as converter


def check_recovery_rows():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "meta/episodes/chunk-000").mkdir(parents=True)
        (root / "data/chunk-000").mkdir(parents=True)
        info_path = root / "meta/info.json"
        info_path.write_text(json.dumps({"total_episodes": 1, "total_frames": 2}))
        journal = root / "meta/hdf5_conversion_pending.json"
        journal.write_text(json.dumps({"base_count": 0, "sources": [], "pending_sources": ["episode_0.hdf5", "episode_1.hdf5"]}))
        sources = root / "meta/hdf5_sources.json"
        sources.write_text("[]\n")
        pq.write_table(pa.table({"episode_index": [0], "length": [2]}), root / "meta/episodes/chunk-000/file-000.parquet")
        path = root / "data/chunk-000/file-000.parquet"
        before = journal.read_bytes(), sources.read_bytes()
        for indices, episodes, frames in (
            ([0, 1, 2, 3], [0, 0, 1, 1], [0, 1, 0, 1]),  # Video failure after data write.
            ([0], [0], [0]),  # Missing frame.
            ([0, 0], [0, 0], [0, 1]),  # Duplicate global index.
            ([0, 1], [0, 1], [0, 1]),  # Wrong episode, same total row count.
            ([0, 1], [0, 0], [0, 0]),  # Duplicate frame within episode.
        ):
            pq.write_table(pa.table({"index": indices, "episode_index": episodes, "frame_index": frames}), path)
            data_before = path.read_bytes()
            try:
                converter.recover_conversion(root)
            except ValueError as exc:
                assert "Physical episode data" in str(exc)
            else:
                raise AssertionError("Accepted inconsistent physical rows")
            assert (journal.read_bytes(), sources.read_bytes()) == before
            assert path.read_bytes() == data_before
        # A failure on the very first episode must also retain orphan data and its journal.
        info_path.write_text(json.dumps({"total_episodes": 0, "total_frames": 0}))
        try:
            converter.recover_conversion(root)
        except ValueError:
            pass
        else:
            raise AssertionError("Accepted orphan rows with no committed episodes")
        assert (journal.read_bytes(), sources.read_bytes()) == before
        info_path.write_text(json.dumps({"total_episodes": 1, "total_frames": 2}))
        pq.write_table(pa.table({"index": [0, 1], "episode_index": [0, 0], "frame_index": [0, 1]}), path)
        converter.recover_conversion(root)
        assert not journal.exists()
        assert json.loads(sources.read_text()) == ["episode_0.hdf5"]


def main():
    check_recovery_rows()
    cls = converter.import_lerobot_dataset()
    with tempfile.TemporaryDirectory() as directory, patch.object(cls, "push_to_hub") as push:
        root = Path(directory)
        raw = root / "raw"
        raw.mkdir()
        output = root / "converted"

        def record(index):
            with h5py.File(raw / f"episode_{index}.hdf5", "w") as source:
                source["observations/qpos"] = np.full((2, 14), index, dtype=np.float32)
                source["action"] = np.full((2, 14), index + 1, dtype=np.float32)
                source["observations/images/cam_high"] = np.full((2, 64, 64, 3), 100, dtype=np.uint8)

        def convert(resume=False, **overrides):
            paths = sorted(raw.glob("episode_*.hdf5"))
            options = dict(
                episode_paths=paths, schema=converter.discover_schema(paths, False, False),
                output_root=output, repo_id="local/test", task="pick up", fps=50,
                robot_type="aloha", include_velocity=False, include_effort=False,
                no_videos=False, image_writer_threads=1, resume=resume,
            )
            return converter.convert_v30(**dict(options, **overrides))

        record(0)
        record(2)
        assert convert() == (2, 4)
        # Simulate output from the old converter, which did not track source names.
        (output / "meta/hdf5_sources.json").unlink()
        saved_files = {
            p: hashlib.sha256(p.read_bytes()).digest()
            for folder in ("data", "videos") for p in (output / folder).rglob("*") if p.is_file()
        }
        record(10)
        assert convert(True, task="put down") == (1, 2)
        for p, digest in saved_files.items():
            assert hashlib.sha256(p.read_bytes()).digest() == digest, f"Overwrote {p}"
        # A later filename sorts before episode_2, but source history still skips it correctly.
        record(11)
        assert convert(True) == (1, 2)
        info_path = output / "meta/info.json"
        before = info_path.read_bytes()
        calls = push.call_count
        assert convert(True) == (0, 0)
        assert push.call_count == calls
        assert info_path.read_bytes() == before
        # Direct recordings have no HDF5 source entry and must not block conversion.
        ds = cls.resume(repo_id="local/test", root=output, image_writer_threads=1)
        try:
            for _ in range(2):
                ds.add_frame({
                    "observation.state": np.full(14, 20, dtype=np.float32),
                    "action": np.full(14, 21, dtype=np.float32),
                    "observation.images.cam_high": np.full((64, 64, 3), 100, dtype=np.uint8),
                    "task": "direct recording",
                })
            ds.save_episode()
        finally:
            ds.finalize()
        record(12)
        assert convert(True) == (1, 2)
        assert json.loads(info_path.read_text())["total_episodes"] == 6
        # Retry an upload failure without converting or duplicating its saved episode.
        record(13)
        push.side_effect = OSError("upload unavailable")
        try:
            convert(True)
        except OSError:
            pass
        else:
            raise AssertionError("Expected upload failure")
        pending = output.with_name(f".{output.name}.conversion-upload-pending")
        assert pending.exists()
        push.side_effect = None
        calls = push.call_count
        assert convert(True) == (0, 0)
        assert push.call_count == calls + 1 and not pending.exists()
        assert json.loads(info_path.read_text())["total_episodes"] == 7
        calls = push.call_count
        assert convert(True) == (0, 0) and push.call_count == calls
        # Offline conversion leaves an upload pending, without contacting the Hub.
        record(14)
        assert convert(True, push_to_hub=False) == (1, 2)
        assert push.call_count == calls and pending.exists()
        assert convert(True, push_to_hub=False) == (0, 0)
        assert push.call_count == calls
        assert convert(True) == (0, 0) and push.call_count == calls + 1
        before = info_path.read_bytes()
        info = json.loads(before)
        assert info["total_episodes"] == 8 and info["total_frames"] == 16
        assert info["total_tasks"] == 3
        rows = pads.dataset(output / "data", format="parquet").to_table().sort_by("index")
        np.testing.assert_array_equal(
            np.asarray(rows["observation.state"].to_pylist())[:, 0], [0, 0, 2, 2, 10, 10, 11, 11, 20, 20, 12, 12, 13, 13, 14, 14],
        )
        for options in ({"fps": 25}, {"no_videos": True}):
            try:
                convert(True, **options)
            except ValueError:
                pass
            else:
                raise AssertionError(f"Accepted incompatible resume: {options}")
        try:
            convert()
        except FileExistsError:
            pass
        else:
            raise AssertionError("Existing root accepted without --resume")
        assert info_path.read_bytes() == before
        assert len(list(raw.glob("*.hdf5"))) == 7
        # Simulate termination after finalization but before source-history persistence.
        record(15)
        recover = converter.recover_conversion
        def interrupt_checkpoint(root):
            if (root / "meta/hdf5_conversion_pending.json").exists():
                raise OSError("interrupted before source checkpoint")
            return recover(root)
        with patch.object(converter, "recover_conversion", side_effect=interrupt_checkpoint):
            try:
                convert(True)
            except OSError:
                pass
            else:
                raise AssertionError("Expected checkpoint interruption")
        assert json.loads(info_path.read_text())["total_episodes"] == 9
        from lerobot_v30 import check_target
        try:
            check_target({"root": str(output)})
        except ValueError as exc:
            assert "Interrupted HDF5 conversion" in str(exc)
        else:
            raise AssertionError("Allowed recording before conversion reconciliation")
        assert convert(True) == (0, 0)
        assert json.loads(info_path.read_text())["total_episodes"] == 9
        assert "episode_15.hdf5" in json.loads((output / "meta/hdf5_sources.json").read_text())
        assert not (output / "meta/hdf5_conversion_pending.json").exists()
        before = info_path.read_bytes()

        # CLI upload-only resume after offline conversion deletes the HDF5 inputs.
        upload_raw, upload_root = root / "upload-raw", root / "upload-only"
        upload_raw.mkdir()
        shutil.copy2(raw / "episode_0.hdf5", upload_raw / "episode_0.hdf5")
        argv = ["convert", "--input-dir", str(upload_raw), "--output-root", str(upload_root),
                "--repo-id", "local/test", "--task", "pick up"]
        calls = push.call_count
        with patch.object(sys, "argv", argv + ["--no-push-to-hub", "--delete-source-hdf5"]):
            converter.main()
        assert not list(upload_raw.glob("*.hdf5")) and push.call_count == calls
        with patch.object(sys, "argv", argv + ["--resume"]):
            converter.main()
        assert push.call_count == calls + 1
        cleanup = converter.hub_cleanup_marker(output)
        cleanup.touch()
        with patch("lerobot_v30.push_replacement") as replace:
            assert convert(True) == (0, 0)
            replace.assert_called_once()
        assert not cleanup.exists()
        # Legacy adoption fails closed if the input order no longer matches saved episodes.
        (output / "meta/hdf5_sources.json").unlink()
        try:
            convert(True)
        except ValueError as exc:
            assert "requires all original HDF5 files" in str(exc)
        else:
            raise AssertionError("Skipped a mismatched legacy source")
        assert info_path.read_bytes() == before
    print("OK: legacy resume, append, source tracking, no-op, compatibility and preservation")


if __name__ == "__main__":
    main()
