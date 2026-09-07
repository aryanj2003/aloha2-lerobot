"""Run with a Python >=3.10 environment containing LeRobot v3; no robot needed."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "aloha_scripts"))
from lerobot_v30 import RecordingWriter, check_target, import_dataset


def frame(index, task="pick up the block"):
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:, :, 0] = 200  # Red: catches RGB/BGR reversal on decode.
    return {
        "observation.state": np.full(14, index, dtype=np.float32),
        "action": np.full(14, index + 1, dtype=np.float32),
        "observation.velocity": np.full(14, 2, dtype=np.float32),
        "observation.effort": np.full(14, 3, dtype=np.float32),
        "observation.images.cam_high": image,
        "task": task,
    }


def verify(root):
    info = json.loads((root / "meta/info.json").read_text())
    assert info["codebase_version"] == "v3.0"
    assert info["total_episodes"] == 2 and info["total_frames"] == 6
    assert info["features"]["observation.images.cam_high"]["dtype"] == "video"
    assert list((root / "meta/episodes").rglob("*.parquet"))
    assert list((root / "videos").rglob("*.mp4"))
    assert not list(root.rglob("*.hdf5"))
    dataset = import_dataset()(repo_id="local/test", root=root, video_backend="pyav")
    for idx in range(6):
        sample = dataset[idx]
        np.testing.assert_allclose(sample["observation.state"], idx % 3)
        np.testing.assert_allclose(sample["action"], idx % 3 + 1)
        np.testing.assert_allclose(sample["observation.velocity"], 2)
        np.testing.assert_allclose(sample["observation.effort"], 3)
        np.testing.assert_allclose(sample["timestamp"], (idx % 3) / 50, atol=1e-6)
        assert sample["task"] == ("pick up the block" if idx < 3 else "put down the block")
        image = sample["observation.images.cam_high"]
        assert image[0].mean() > 0.6 and image[2].mean() < 0.1


def main():
    with tempfile.TemporaryDirectory(prefix="aloha-v30-") as temp:
        temp = Path(temp)
        direct = temp / "direct"
        config = dict(
            root=str(direct), repo_id="local/test", episode_index=None, fps=50,
            robot_type="aloha", camera_names=["cam_high"], include_velocity=True,
            include_effort=True, push_to_hub=False, private=False,
        )
        # LeRobot creates the root before validating feature names.
        invalid = frame(0)
        invalid["observation.images.cam/high"] = invalid.pop("observation.images.cam_high")
        try:
            writer = RecordingWriter(sys.executable, dict(config, camera_names=["cam/high"]), invalid)
        except RuntimeError:
            pass
        else:
            writer.close()
            raise AssertionError("Invalid feature name was accepted")
        assert not direct.exists(), "Failed creation left a directory that blocks corrected retries"

        # Abandoning the first episode, even before its first frame, must permit retry.
        for send_frame in (False, True):
            writer = RecordingWriter(sys.executable, config, frame(0))
            assert (direct / "meta/info.json").exists()  # Creation precedes readiness.
            if send_frame:
                writer.add_frame(frame(0))
            writer.close()
            assert writer.process.returncode != 0
            assert not direct.exists()

        for episode, task in enumerate(("pick up the block", "put down the block")):
            writer = RecordingWriter(sys.executable, config, frame(0))
            try:
                assert writer.episode_index == episode
                for idx in range(3):
                    writer.add_frame(frame(idx, task))
                writer.finish()
            finally:
                writer.close()
        verify(direct)
        before = (direct / "meta/info.json").read_bytes()
        for overrides in ({"episode_index": 0}, {"fps": 25}, {"include_effort": False}):
            try:
                check_target(dict(config, **overrides))
            except ValueError:
                pass
            else:
                raise AssertionError(f"Unsafe append accepted: {overrides}")
        assert before == (direct / "meta/info.json").read_bytes()

        incompatible = frame(0)
        incompatible["observation.images.cam_high"] = np.zeros((32, 32, 3), dtype=np.uint8)
        try:
            writer = RecordingWriter(sys.executable, config, incompatible)
        except RuntimeError:
            pass
        else:
            writer.close()
            raise AssertionError("Writer reported readiness before rejecting incompatible image shape")
        assert before == (direct / "meta/info.json").read_bytes()

        blocked_parent = temp / "not-a-directory"
        blocked_parent.write_text("preserve me")
        try:
            writer = RecordingWriter(sys.executable, dict(config, root=str(blocked_parent / "dataset")), frame(0))
        except RuntimeError:
            pass
        else:
            writer.close()
            raise AssertionError("Writer reported readiness before dataset creation failed")
        assert blocked_parent.read_text() == "preserve me"

        # Closing without finish must not commit an interrupted episode.
        writer = RecordingWriter(sys.executable, config, frame(0))
        writer.add_frame(frame(0))
        writer.close()
        assert writer.process.returncode != 0
        verify(direct)

        raw = temp / "raw"
        raw.mkdir()
        with h5py.File(raw / "episode_0.hdf5", "w") as file:
            for source, target in (
                ("observation.state", "observations/qpos"), ("action", "action"),
                ("observation.velocity", "observations/qvel"),
                ("observation.effort", "observations/effort"),
                ("observation.images.cam_high", "observations/images/cam_high"),
            ):
                file[target] = np.stack([frame(i)[source] for i in range(3)])
        converted = temp / "converted"
        subprocess.run([
            sys.executable, str(ROOT / "tools/convert_hdf5_to_lerobot.py"),
            "--input-dir", str(raw), "--repo-id", "local/test", "--output-root", str(converted),
            "--task", "pick up the block", "--include-velocity", "--include-effort",
        ], check=True)
        # Append to converter output too: both producers must have compatible schemas.
        writer = RecordingWriter(sys.executable, dict(config, root=str(converted)), frame(0))
        try:
            for idx in range(3):
                writer.add_frame(frame(idx, "put down the block"))
            writer.finish()
        finally:
            writer.close()
        verify(converted)
        assert (raw / "episode_0.hdf5").exists()
        for options, version in ((["--no-videos"], "v3.0"), (["--format", "v2.1"], "v2.1")):
            output = temp / version
            subprocess.run([
                sys.executable, str(ROOT / "tools/convert_hdf5_to_lerobot.py"),
                "--input-dir", str(raw), "--repo-id", "local/test", "--output-root", str(output),
                "--task", "pick up the block", *options,
            ], check=True)
            info = json.loads((output / "meta/info.json").read_text())
            assert info["codebase_version"] == version and info["total_frames"] == 3
            if version == "v3.0":
                assert info["features"]["observation.images.cam_high"]["dtype"] == "image"
                assert not list(output.rglob("*.mp4"))
    print("OK: v3 recording, resume, interruption, conversion, task labels and video decoding")


if __name__ == "__main__":
    main()
