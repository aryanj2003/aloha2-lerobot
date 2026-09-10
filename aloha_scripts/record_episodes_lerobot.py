#!/usr/bin/env python3
"""Record real ALOHA episodes directly to a local LeRobot v3.0 dataset."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from constants import DT, START_ARM_POSE, TASK_CONFIGS
from constants import MASTER_GRIPPER_JOINT_MID, PUPPET_GRIPPER_JOINT_CLOSE, PUPPET_GRIPPER_JOINT_OPEN
from interbotix_xs_modules.arm import InterbotixManipulatorXS
from lerobot_v21 import default_dataset_root
from lerobot_v30 import RecordingWriter, check_target
from real_env import get_action, make_real_env
from robot_utils import get_arm_gripper_positions
from robot_utils import move_arms, move_grippers, torque_off, torque_on


def opening_ceremony(master_bot_left, master_bot_right, puppet_bot_left, puppet_bot_right):
    """Move all 4 robots to a pose where it is easy to start demonstration."""
    puppet_bot_left.dxl.robot_reboot_motors("single", "gripper", True)
    puppet_bot_left.dxl.robot_set_operating_modes("group", "arm", "position", "time")
    puppet_bot_left.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    master_bot_left.dxl.robot_set_operating_modes("group", "arm", "position", "time")
    master_bot_left.dxl.robot_set_operating_modes("single", "gripper", "position")

    puppet_bot_right.dxl.robot_reboot_motors("single", "gripper", True)
    puppet_bot_right.dxl.robot_set_operating_modes("group", "arm", "position", "time")
    puppet_bot_right.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    master_bot_right.dxl.robot_set_operating_modes("group", "arm", "position", "time")
    master_bot_right.dxl.robot_set_operating_modes("single", "gripper", "position")

    torque_on(puppet_bot_left)
    torque_on(master_bot_left)
    torque_on(puppet_bot_right)
    torque_on(master_bot_right)

    start_arm_qpos = START_ARM_POSE[:6]
    move_arms([master_bot_left, puppet_bot_left, master_bot_right, puppet_bot_right], [start_arm_qpos] * 4, move_time=1.5)
    move_grippers(
        [master_bot_left, puppet_bot_left, master_bot_right, puppet_bot_right],
        [MASTER_GRIPPER_JOINT_MID, PUPPET_GRIPPER_JOINT_CLOSE] * 2,
        move_time=0.5,
    )

    master_bot_left.dxl.robot_torque_enable("single", "gripper", False)
    master_bot_right.dxl.robot_torque_enable("single", "gripper", False)
    print("Close the gripper to start")
    close_thresh = -0.78
    pressed = False
    while not pressed:
        gripper_pos_left = get_arm_gripper_positions(master_bot_left)
        gripper_pos_right = get_arm_gripper_positions(master_bot_right)
        if (gripper_pos_left < close_thresh) and (gripper_pos_right < close_thresh):
            pressed = True
        time.sleep(DT / 10)
    torque_off(master_bot_left)
    torque_off(master_bot_right)
    print("Started!")


def print_dt_diagnosis(actual_dt_history):
    actual_dt_history = np.array(actual_dt_history)
    get_action_time = actual_dt_history[:, 1] - actual_dt_history[:, 0]
    step_env_time = actual_dt_history[:, 2] - actual_dt_history[:, 1]
    total_time = actual_dt_history[:, 2] - actual_dt_history[:, 0]

    dt_mean = np.mean(total_time)
    freq_mean = 1 / dt_mean
    print(f"Avg freq: {freq_mean:.2f} Get action: {np.mean(get_action_time):.3f} Step env: {np.mean(step_env_time):.3f}")
    return freq_mean


def recording_frame(obs, action, camera_names, task, include_velocity, include_effort):
    frame = {
        "observation.state": np.asarray(obs["qpos"], dtype=np.float32),
        "action": np.asarray(action, dtype=np.float32),
        "task": task,
    }
    if include_velocity:
        frame["observation.velocity"] = np.asarray(obs["qvel"], dtype=np.float32)
    if include_effort:
        frame["observation.effort"] = np.asarray(obs["effort"], dtype=np.float32)
    for camera_name in camera_names:
        image = obs["images"][camera_name]
        if image is None:
            raise RuntimeError(f"No image received for camera {camera_name}")
        frame[f"observation.images.{camera_name}"] = image
    return frame


def capture_one_episode(
    max_timesteps,
    camera_names,
    writer_python,
    writer_config,
    task,
    include_velocity,
    include_effort,
):
    master_bot_left = InterbotixManipulatorXS(
        robot_model="wx250s",
        group_name="arm",
        gripper_name=None,
        robot_name="master_left",
        init_node=True,
    )
    master_bot_right = InterbotixManipulatorXS(
        robot_model="wx250s",
        group_name="arm",
        gripper_name=None,
        robot_name="master_right",
        init_node=False,
    )
    env = make_real_env(init_node=False, setup_robots=False)

    sample = recording_frame(
        env.get_observation(), get_action(master_bot_left, master_bot_right),
        camera_names, task, include_velocity, include_effort,
    )
    writer = RecordingWriter(writer_python, writer_config, sample)
    print(f"Episode index: {writer.episode_index}")
    try:
        opening_ceremony(master_bot_left, master_bot_right, env.puppet_bot_left, env.puppet_bot_right)

        ts = env.reset(fake=True)
        actual_dt_history = []
        try:
            for _ in tqdm(range(max_timesteps)):
                t0 = time.time()
                action = get_action(master_bot_left, master_bot_right)
                t1 = time.time()
                obs = ts.observation
                frame = recording_frame(obs, action, camera_names, task, include_velocity, include_effort)
                writer.add_frame(frame)
                ts = env.step(action)
                actual_dt_history.append([t0, t1, time.time()])
        finally:
            torque_on(master_bot_left)
            torque_on(master_bot_right)
            move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)

        print_dt_diagnosis(actual_dt_history)
        writer.finish()
        print(f"Saved LeRobot v3.0 episode {writer.episode_index} with {max_timesteps} frames.")
    finally:
        writer.close()


def resolve_dataset_root(args, task_config):
    if args.output_root:
        return Path(args.output_root).expanduser()
    if args.repo_id:
        return default_dataset_root(args.repo_id)
    return Path(task_config["dataset_dir"]).expanduser() / "lerobot"


def main(args):
    task_config = TASK_CONFIGS[args.task_name]
    dataset_root = resolve_dataset_root(args, task_config)
    max_timesteps = task_config["episode_len"]
    camera_names = task_config["camera_names"]
    task = args.task if args.task else args.task_name
    if args.fps <= 0 or max_timesteps <= 0:
        raise SystemExit("FPS and episode length must be positive.")
    if args.push_to_hub and not args.repo_id:
        raise SystemExit("--push-to-hub requires --repo-id.")

    print(f"LeRobot v3.0 dataset root: {dataset_root}")
    writer_config = {
        "root": str(dataset_root.resolve()),
        "repo_id": args.repo_id or f"local/{dataset_root.name}",
        "episode_index": args.episode_idx,
        "fps": args.fps,
        "robot_type": args.robot_type,
        "camera_names": camera_names,
        "include_velocity": args.include_velocity,
        "include_effort": args.include_effort,
        "push_to_hub": args.push_to_hub,
        "private": args.private,
    }
    try:
        check_target(writer_config)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    capture_one_episode(
        max_timesteps=max_timesteps,
        camera_names=camera_names,
        writer_python=args.lerobot_python,
        writer_config=writer_config,
        task=task,
        include_velocity=args.include_velocity,
        include_effort=args.include_effort,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", action="store", type=str, help="Task name from constants.py.", required=True)
    parser.add_argument("--episode_idx", action="store", type=int, help="Replace an existing episode, or append at the next index. Omit to append; gaps are rejected.", default=None)
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Hugging Face dataset repo id, e.g. <hf_user>/<dataset_name>. Defaults local root to HF_LEROBOT_HOME/<repo-id>, HF_HOME/lerobot/<repo-id>, or ~/.cache/huggingface/lerobot/<repo-id>.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Override local LeRobot dataset root. If omitted with --repo-id, uses the LeRobot cache path.",
    )
    parser.add_argument("--task", type=str, default=None, help="Natural-language task stored in LeRobot metadata.")
    parser.add_argument("--lerobot-python", default=os.environ.get("LEROBOT_PYTHON", sys.executable),
                        help="Python >=3.10 executable with LeRobot v3 installed (or set LEROBOT_PYTHON).")
    parser.add_argument("--fps", type=int, default=50, help="Dataset frame rate.")
    parser.add_argument("--robot-type", type=str, default="aloha", help="Robot type metadata.")
    parser.add_argument("--include-velocity", action="store_true", help="Store observation.velocity.")
    parser.add_argument("--include-effort", action="store_true", help="Store observation.effort.")
    parser.add_argument("--push-to-hub", action="store_true", help="Upload the local LeRobot dataset folder after recording.")
    parser.add_argument("--private", action="store_true", help="Create the Hugging Face dataset repo as private when pushing.")
    main(parser.parse_args())
