"""Official v3 writer, isolated from the ROS Python environment."""

import contextlib
import json
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

from lerobot_v21 import feature_names


def import_dataset():
    try:
        from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
    except ImportError as exc:
        raise RuntimeError(
            "LeRobot v3 requires Python >=3.10 with the official lerobot package. "
            "For ROS recording, select that environment with --lerobot-python."
        ) from exc
    if CODEBASE_VERSION != "v3.0":
        raise RuntimeError(f"Expected LeRobot v3.0, installed writer uses {CODEBASE_VERSION}")
    return LeRobotDataset


def frame_features(frame):
    features = {}
    for key, value in frame.items():
        if key == "task":
            continue
        camera = key.startswith("observation.images.")
        features[key] = {
            "dtype": "video" if camera else "float32",
            "shape": tuple(value.shape),
            "names": ["height", "width", "channel"] if camera else feature_names(value.shape[0]),
        }
    return features


def check_target(config):
    root = Path(config["root"])
    info_path = root / "meta" / "info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text())
        if info["codebase_version"] != "v3.0":
            raise ValueError("Use a new output root: existing dataset is not v3.0.")
        if info["fps"] != config["fps"] or info["robot_type"] != config["robot_type"]:
            raise ValueError("Existing dataset FPS or robot type differs from this recording.")
        expected_keys = {"observation.state", "action"}
        expected_keys.update("observation.images." + name for name in config["camera_names"])
        for field in ("velocity", "effort"):
            if config["include_" + field]:
                expected_keys.add("observation." + field)
        actual_keys = {key for key in info["features"] if key == "action" or key.startswith("observation.")}
        if actual_keys != expected_keys:
            raise ValueError("Existing dataset cameras or optional signals differ from this recording.")
        next_index = info["total_episodes"]
    else:
        if root.exists():
            raise ValueError("New dataset output root must not already exist.")
        next_index = 0
    if config["episode_index"] not in (None, next_index):
        raise ValueError(f"v3 recording is append-only; next episode index is {next_index}.")
    return next_index


def writer_main():
    # Pickle is used only on private pipes to our own child process, never for dataset input.
    source, reply = sys.stdin.buffer, sys.stdout.buffer
    with contextlib.redirect_stdout(sys.stderr):
        config = pickle.load(source)
        cls = import_dataset()
        next_index = check_target(config)
        root = Path(config["root"])
        new_dataset = not root.exists()
        features = config["features"]
        dataset = None
        save_started = False
        saved = False
        try:
            kwargs = dict(repo_id=config["repo_id"], root=root)
            if new_dataset:
                try:
                    dataset = cls.create(
                        **kwargs, fps=config["fps"], robot_type=config["robot_type"],
                        features=features, use_videos=True, image_writer_threads=4,
                    )
                except FileExistsError:
                    # Another writer may have created the root after check_target().
                    new_dataset = False
                    raise
            else:
                stored_features = json.loads((root / "meta/info.json").read_text())["features"]
                for key, feature in features.items():
                    stored = stored_features[key]
                    if tuple(stored["shape"]) != feature["shape"] or stored["dtype"] != feature["dtype"]:
                        raise ValueError(f"Frame schema differs for {key}")
                if hasattr(cls, "resume"):
                    dataset = cls.resume(**kwargs, image_writer_threads=4)
                else:
                    dataset = cls(**kwargs)
                    dataset.start_image_writer(num_threads=4)
            pickle.dump(next_index, reply, protocol=4)
            reply.flush()
            frame_count = 0
            while True:
                frame = pickle.load(source)
                if frame is None:
                    if not frame_count:
                        raise ValueError("No frames were recorded.")
                    # Once saving starts, retain files for recovery even if saving fails.
                    save_started = True
                    dataset.save_episode()
                    saved = True
                    break
                dataset.add_frame(frame)
                frame_count += 1
        finally:
            if dataset is not None:
                try:
                    if not saved:
                        dataset.clear_episode_buffer()
                finally:
                    dataset.finalize()
            if new_dataset and not save_started and root.is_dir():
                shutil.rmtree(root)
        if config["push_to_hub"]:
            dataset.push_to_hub(private=config["private"])


class RecordingWriter:
    """Stream frames through a bounded OS pipe to a modern Python writer."""

    def __init__(self, python, config, sample_frame):
        env = os.environ.copy()
        # ROS's PYTHONPATH contains extensions for Python 3.8.
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        self.process = subprocess.Popen(
            [python, str(Path(__file__).resolve())], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, env=env, start_new_session=True,
        )
        try:
            self.add_frame(dict(config, features=frame_features(sample_frame)))
            self.episode_index = pickle.load(self.process.stdout)
        except BaseException:
            self.close()
            raise RuntimeError("LeRobot writer failed to start; see its error above.")

    def add_frame(self, frame):
        pickle.dump(frame, self.process.stdin, protocol=4)
        self.process.stdin.flush()

    def finish(self):
        self.add_frame(None)
        self.close()
        if self.process.returncode:
            raise RuntimeError("LeRobot writer failed; episode was not successfully finalized.")

    def close(self):
        if not self.process.stdin.closed:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
        self.process.wait()
        self.process.stdout.close()


if __name__ == "__main__":
    writer_main()
