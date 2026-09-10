"""Official v3 writer, isolated from the ROS Python environment."""

import contextlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
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
    if (root / "meta/hdf5_conversion_pending.json").exists():
        raise ValueError("Interrupted HDF5 conversion: rerun the converter with --resume before recording.")
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
    index = config["episode_index"]
    if index is None:
        return next_index
    if not 0 <= index <= next_index:
        raise ValueError(f"Episode index {index} is out of range; use 0 through {next_index} (append).")
    return index


def replace_appended_episode(root, target):
    """Replace target with the just-appended episode, retaining all encoded videos."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.dataset as pads
    import pyarrow.parquet as pq
    from lerobot.datasets.compute_stats import aggregate_stats, get_feature_stats
    from lerobot.datasets.io_utils import write_stats
    from lerobot.utils.utils import unflatten_dict

    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    last = info["total_episodes"] - 1
    episode_files = sorted((root / "meta/episodes").rglob("*.parquet"))
    episode_schema = pa.unify_schemas([pq.read_schema(path) for path in episode_files])
    episodes = pads.dataset(episode_files, schema=episode_schema, format="parquet").to_table().to_pylist()
    episodes = {row["episode_index"]: row for row in episodes}
    data = pads.dataset(root / "data", format="parquet")
    rebuilt = root / "replacement-data"
    rebuilt.mkdir()
    metadata, stats = [], []
    offset = 0
    # ponytail: rewrite numeric data once; videos stay byte-identical. Optimize shards if this gets slow.
    for index in range(last):
        source_index = last if index == target else index
        row = episodes[source_index]
        table = data.to_table(filter=pads.field("episode_index") == source_index).sort_by("frame_index")
        length = len(table)
        for key, values in (("episode_index", np.full(length, index)), ("index", np.arange(offset, offset + length))):
            column = table.schema.get_field_index(key)
            table = table.set_column(column, table.schema.field(key), pa.array(values, type=table.schema.field(key).type))
        # Recompute numeric statistics, including quantiles absent from older exports.
        for key in table.column_names:
            if info["features"][key]["dtype"] in ("video", "image", "string"):
                continue
            values = np.asarray(table[key].to_pylist())
            row.update({f"stats/{key}/{stat}": value.tolist()
                        for stat, value in get_feature_stats(values, axis=0, keepdims=values.ndim == 1).items()})
        chunk, file_index = divmod(index, info["chunks_size"])
        path = rebuilt / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.parquet"
        path.parent.mkdir(exist_ok=True)
        pq.write_table(table, path)
        row.update({
            "episode_index": index, "data/chunk_index": chunk, "data/file_index": file_index,
            "dataset_from_index": offset, "dataset_to_index": offset + length,
            "meta/episodes/chunk_index": 0, "meta/episodes/file_index": 0,
        })
        metadata.append(row)
        stats.append(unflatten_dict({
            k: np.asarray(v) for k, v in row.items()
            if k.startswith("stats/") and v is not None
        })["stats"])
        offset += length
    # Resume allocates videos after the last logical episode's file. Keep that file last
    # physically too, otherwise a later append could overwrite an earlier replacement.
    rebuilt_videos = root / "replacement-videos"
    for key, feature in info["features"].items():
        if feature["dtype"] != "video":
            continue
        chunk_key, file_key = f"videos/{key}/chunk_index", f"videos/{key}/file_index"
        def video_path(row):
            return root / info["video_path"].format(video_key=key, chunk_index=row[chunk_key], file_index=row[file_key])
        references = {video_path(row): row["episode_index"] for row in metadata}
        paths = sorted((root / "videos" / key).rglob("*.mp4"))
        mapping = {}
        for number, path in enumerate(sorted(paths, key=lambda p: references.get(p, -1))):
            chunk, file_index = divmod(number, info["chunks_size"])
            destination = rebuilt_videos / key / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.mp4"
            destination.parent.mkdir(parents=True, exist_ok=True)
            path.rename(destination)
            mapping[path] = chunk, file_index
        for row in metadata:
            row[chunk_key], row[file_key] = mapping[video_path(row)]
    shutil.rmtree(root / "videos")
    rebuilt_videos.rename(root / "videos")
    shutil.rmtree(root / "data")
    rebuilt.rename(root / "data")
    shutil.rmtree(root / "meta/episodes")
    meta_path = root / "meta/episodes/chunk-000/file-000.parquet"
    meta_path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(metadata), meta_path)
    info.update(total_episodes=last, total_frames=offset, splits={"train": f"0:{last}"})
    info_path.write_text(json.dumps(info, indent=2) + "\n")
    write_stats(aggregate_stats(stats), root)


def writer_main():
    # Stage overwrites so any capture/save failure leaves the original dataset untouched.
    config = pickle.load(sys.stdin.buffer)
    root = Path(config["root"])
    if not root.exists():
        backups = list(root.parent.glob(f".{root.name}.overwrite-*/original"))
        if backups:
            raise RuntimeError(f"Interrupted dataset replacement: restore {backups[0]} to {root} before recording.")
    index = check_target(config)
    info_path = root / "meta/info.json"
    overwrite = info_path.exists() and index < json.loads(info_path.read_text())["total_episodes"]
    if not overwrite:
        return record_dataset(config, index)
    staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.overwrite-", dir=root.parent))
    candidate, backup = staging / "dataset", staging / "original"
    replacement_config = dict(config, root=str(candidate), push_to_hub=False)
    try:
        shutil.copytree(root, candidate)
        with contextlib.redirect_stdout(sys.stderr):
            print(f"Replacing episode {index}; original dataset stays intact until saving completes.")
        record_dataset(replacement_config, index)
        replace_appended_episode(candidate, index)
        # Open the result before publishing it to catch invalid metadata or frame indexing.
        cls = import_dataset()
        cls(repo_id=config["repo_id"], root=candidate, video_backend="pyav")
        hub_cleanup_marker(root).touch()
        root.rename(backup)
        try:
            candidate.rename(root)
        except BaseException:
            backup.rename(root)
            raise
        shutil.rmtree(backup)
        if config["push_to_hub"]:
            push_dataset(cls(repo_id=config["repo_id"], root=root), config["private"])
    finally:
        if backup.exists():
            print(f"Original dataset retained for recovery at {backup}", file=sys.stderr)
        elif replacement_config.get("_save_started") and candidate.exists():
            print(f"Replacement data retained for recovery at {candidate}", file=sys.stderr)
        else:
            shutil.rmtree(staging)


def hub_cleanup_marker(root):
    root = Path(root)
    return root.with_name(f".{root.name}.hub-cleanup-pending")


def push_dataset(dataset, private=False):
    marker = hub_cleanup_marker(dataset.root)
    if marker.exists():
        push_replacement(dataset, private)
        marker.unlink()
    else:
        dataset.push_to_hub(private=private)


def push_replacement(dataset, private):
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=dataset.repo_id, repo_type="dataset", private=private, exist_ok=True)
    # Remove obsolete shards in the same commit as their replacements. The regular
    # push then updates LeRobot's card/version tag and reuses the already uploaded files.
    api.upload_folder(
        repo_id=dataset.repo_id, repo_type="dataset", folder_path=dataset.root,
        delete_patterns=["data/**", "meta/episodes/**", "videos/**"], ignore_patterns=["images/"],
    )
    dataset.push_to_hub(private=private)


def record_dataset(config, next_index):
    # Pickle is used only on private pipes to our own child process, never for dataset input.
    source, reply = sys.stdin.buffer, sys.stdout.buffer
    with contextlib.redirect_stdout(sys.stderr):
        cls = import_dataset()
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
                    save_started = config["_save_started"] = True
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
            push_dataset(dataset, config["private"])


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
