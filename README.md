# ALOHA: A Low-cost Open-source Hardware System for Bimanual Teleoperation

#### Project Website: https://tonyzhaozh.github.io/aloha/

This codebase contains implementation for teleoperation and data collection with the ALOHA hardware.
To build ALOHA, follow the [Hardware Assembly Tutorial](https://docs.google.com/document/d/1sgRZmpS7HMcZTPfGy3kAxDrqFMtNNzmK-yVtX5cKYME/edit?usp=sharing) and the quick start guide below.
To train imitation learning algorithms, you would also need to install [ACT](https://github.com/tonyzhaozh/act).

## Notes for This Fork

This is a modified fork of Tony Zhao's original ALOHA repository for upgraded lab hardware and LeRobot-compatible dataset export. Original project: https://github.com/tonyzhaozh/aloha

Some gripper-related constants and mode settings in this fork are calibrated for our upgraded lab hardware. Do not copy these gripper limits blindly to another ALOHA setup; re-measure the open/close positions and update the relevant constants and motor/config files for your own hardware.

### Repo Structure
- ``config``: a config for each robot, designating the port they should bind to, more details in quick start guide.
- ``launch``: a ROS launch file for all 4 cameras and all 4 robots.
- ``aloha_scripts``: python code for teleop and data collection

## Quick start guide

### Hardware selection 

We suggest using a "heavy-duty" computer if possible. 

*In particular, at least 6 USB3 ports are needed. 4 ports for robot connections and 2 ports for cameras.* We have seen cases that a machine was not able to stably connect to all 4 robot arms simultaneously over USB, especially when USB hubs are used.

### Software selection -- OS:

Currently tested and working configurations: 
- :white_check_mark: Ubuntu 18.04 + ROS 1 noetic
- :white_check_mark: Ubuntu 20.04 + ROS 1 noetic

Ongoing testing (compatibility effort underway):
- :construction: ROS 2

### Software installation - ROS:
1. Install ROS and interbotix software following https://docs.trossenrobotics.com/interbotix_xsarms_docs/
2. This will create the directory ``~/interbotix_ws`` which contains ``src``.
3. git clone this repo inside ``~/interbotix_ws/src``
4. ``source /opt/ros/noetic/setup.sh && source ~/interbotix_ws/devel/setup.sh``
5. ``sudo apt-get install ros-noetic-usb-cam && sudo apt-get install ros-noetic-cv-bridge``
6. run ``catkin_make`` inside ``~/interbotix_ws``, make sure the build is successful
7. go to ``~/interbotix_ws/src/interbotix_ros_toolboxes/interbotix_xs_toolbox/interbotix_xs_modules/src/interbotix_xs_modules/arm.py``, find function ``publish_positions``.
   Change ``self.T_sb = mr.FKinSpace(self.robot_des.M, self.robot_des.Slist, self.joint_commands)`` to ``self.T_sb = None``.
   This prevents the code from calculating FK at every step which delays teleoperation.
### Hardware installation:

The goal of this section is to run ``roslaunch aloha 4arms_teleop.launch``, which starts
communication with 4 robots and 4 cameras. It should work after finishing the following steps:

Step 1: Connect 4 robots to the computer via USB, and power on. *Do not use extension cable or usb hub*.
- To check if the robot is connected, install dynamixel wizard [here](https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_wizard2/)
- Dynamixel wizard is a very helpful debugging tool that connects to individual motors of the robot. It allows
things such as rebooting the motor (very useful!), torque on/off, and sending commands.
However, it has no knowledge about the kinematics of the robot, so be careful about collisions.
The robot *will* collapse if motors are torque off i.e. there is no automatically engaged brakes in joints.
- Open Dynamixel wizard, go into ``options`` and select:
  - Protocal 2.0
  - All ports
  - 1000000 bps
  - ID range from 0-10
- Note: repeat above everytime before you scan.
- Then hit ``Scan``. There should be 4 devices showing up, each with 9 motors.


- One issue that arises is the port each robot binds to can change over time, e.g. a robot that
is initially ``ttyUSB0`` might suddenly become ``ttyUSB5``. To resolve this, we bind each robot to a fixed symlink
port with the following mapping:
  - ``ttyDXL_master_right``: right master robot (master: the robot that the operator would be holding)
  - ``ttyDXL_puppet_right``: right puppet robot (puppet: the robot that performs the task)
  - ``ttyDXL_master_left``: left master robot
  - ``ttyDXL_puppet_left``: left puppet robot
- Take ``ttyDXL_master_right``: right master robot as an example:
  1. Find the port that the right master robot is currently binding to, e.g. ``ttyUSB0``
  2. run ``udevadm info --name=/dev/ttyUSB0 --attribute-walk | grep serial`` to obtain the serial number. Use the first one that shows up, the format should look similar to ``FT6S4DSP``.
  3. ``sudo vim /etc/udev/rules.d/99-fixed-interbotix-udev.rules`` and add the following line: 

         SUBSYSTEM=="tty", ATTRS{serial}=="<serial number here>", ENV{ID_MM_DEVICE_IGNORE}="1", ATTR{device/latency_timer}="1", SYMLINK+="ttyDXL_master_right"

  4. This will make sure the right master robot is *always* binding to ``ttyDXL_master_right``
  5. Repeat with the rest of 3 arms.
- To apply the changes, run ``sudo udevadm control --reload && sudo udevadm trigger``
- If successful, you should be able to find ``ttyDXL*`` in your ``/dev``

Step 2: Set max current for gripper motors
- Open Dynamixel Wizard, and select the wrist motor for puppet arms. The name of it should be ```[ID:009] XM430-W350```
- Tip: the LED on the base of robot will flash when it is talking to Dynamixel Wizard. This will help determine which robot is selected. 
- Find ``38 Current Limit``, enter ``200``, then hit ``save`` at the bottom.
- Repeat this for both puppet robots.
- This limits the max current through gripper motors, to prevent overloading errors.


Step 3: Setup 4 cameras
- You may use usb hub here, but *maximum 2 cameras per hub for reasonable latency*.
- To make sure all 4 cameras are binding to a consistent port, similar steps are needed.
- Cameras are by default binding to ``/dev/video{0, 1, 2...}``, while we want to have symlinks ``{CAM_RIGHT_WRIST, CAM_LEFT_WRIST, CAM_LOW, CAM_HIGH}``
- Take ``CAM_RIGHT_WRIST`` as an example, and let's say it is now binding to ``/dev/video0``. run ``udevadm info --name=/dev/video0 --attribute-walk | grep serial`` to obtain it's serial. Use the first one that shows up, the format should look similar to ``0E1A2B2F``.
- Then ``sudo vim /etc/udev/rules.d/99-fixed-interbotix-udev.rules`` and add the following line 

      SUBSYSTEM=="video4linux", ATTRS{serial}=="<serial number here>", ATTR{index}=="0", ATTRS{idProduct}=="085c", ATTR{device/latency_timer}="1", SYMLINK+="CAM_RIGHT_WRIST"

- Repeat this for ``{CAM_LEFT_WRIST, CAM_LOW, CAM_HIGH}`` in additional to ``CAM_RIGHT_WRIST``
- To apply the changes, run ``sudo udevadm control --reload && sudo udevadm trigger``
- If successful, you should be able to find ``{CAM_RIGHT_WRIST, CAM_LEFT_WRIST, CAM_LOW, CAM_HIGH}`` in your ``/dev``

At this point, have a new terminal
    
    conda deactivate # if conda shows up by default
    source /opt/ros/noetic/setup.sh && source ~/interbotix_ws/devel/setup.sh
    roslaunch aloha 4arms_teleop.launch

If no error message is showing up, the computer should be successfully connected to all 4 cameras and all 4 robots.

#### Trouble shooting
- Make sure Dynamixel Wizard is disconnected, and no app is using webcam's stream. It will prevent ROS from connecting to
these devices.

### Software installation - Conda:

Use the checked-in Conda environment file instead of installing the Python packages one by one. It reflects the working ``aloha`` environment on this machine, including the ROS recording dependencies and CUDA 11.8 PyTorch wheels.

    conda env create -f environment.yml
    conda activate aloha

If the environment already exists, update it with:

    conda env update -n aloha -f environment.yml

ROS Noetic and Interbotix packages are still installed through the ROS workspace steps above, not by ``environment.yml``.

LeRobot v3 requires Python 3.10 or newer, while the ROS ``aloha`` environment uses Python 3.8. Keep ROS in ``aloha`` and run the official dataset writer in a separate environment with LeRobot v3 installed. On this machine, ``aloha2-policy`` already provides it:

    export LEROBOT_PYTHON=/media/hong/home/conda-envs/aloha2-policy/bin/python

On another machine, set this to the Python executable in your LeRobot environment (for example one with ``lerobot==0.4.4`` installed). The recorder streams frames to that process over a local pipe; it does not write intermediate HDF5 files. You can also pass ``--lerobot-python /path/to/python`` directly. Conversion runs entirely in the LeRobot environment.


### Testing teleoperation

**Notice**: Before running the commands below, be sure to place all 4 robots in their sleep positions, and open master robot's gripper. 
All robots will rise to a height that is easy for teleoperation.

    # ROS terminal
    conda deactivate
    source /opt/ros/noetic/setup.sh && source ~/interbotix_ws/devel/setup.sh
    roslaunch aloha 4arms_teleop.launch
    
    # Right hand terminal
    conda activate aloha
    cd ~/interbotix_ws/src/aloha/aloha_scripts
    python3 one_side_teleop.py right
    
    # Left hand terminal
    conda activate aloha
    cd ~/interbotix_ws/src/aloha/aloha_scripts
    python3 one_side_teleop.py left

The teleoperation will start when the master side gripper is closed.

Before running ``sleep.py``, stop both ``one_side_teleop.py`` processes with ``Ctrl-Z``. The sleep script sends its own puppet arm commands; if a teleop process is still running, that arm can keep trying to follow the master arm while ``sleep.py`` is moving it to the sleep pose.


## Example Usages

To set up a new terminal, run:

    conda activate aloha
    cd ~/interbotix_ws/src/aloha/aloha_scripts


The ``one_side_teleop.py`` we ran is for testing teleoperation and has no data collection. To collect data for an episode, run:

    python3 record_episodes.py --task_name <task name> --episode_idx 0

This will store ``episode_<idx>.hdf5`` in the task's ``dataset_dir`` from ``constants.py``.
To change the save directory, episode length, camera list, or other task params, edit ``TASK_CONFIGS`` in ``constants.py`` directly.

To record directly to a local LeRobot v3.0 dataset instead of writing HDF5 first, set ``LEROBOT_PYTHON`` as above and run:

    python3 record_episodes_lerobot.py \
        --task_name <task name> \
        --repo-id <hf_user>/<dataset_name> \
        --task "<short natural-language task description>"

With ``--repo-id``, the local dataset is written under ``HF_LEROBOT_HOME`` if set, otherwise ``HF_HOME/lerobot`` if set, otherwise the standard LeRobot cache path ``~/.cache/huggingface/lerobot/<hf_user>/<dataset_name>``. To use a different local path, pass ``--output-root <local lerobot dataset root>``. To upload after recording, authenticate with Hugging Face first and add ``--push-to-hub``:

    python3 record_episodes_lerobot.py \
        --task_name <task name> \
        --repo-id <hf_user>/<dataset_name> \
        --task "<short natural-language task description>" \
        --push-to-hub

This direct path uses the official LeRobot API to write v3.0 Parquet data, MP4 videos, statistics, and episode/task metadata. Images are written asynchronously and videos are encoded when the episode is saved; finalization finishes before optional upload. No ``episode_<idx>.hdf5`` is created. Add ``--include-velocity --include-effort`` to retain those optional signals.

Omitting ``--episode_idx`` appends the next episode. With N saved episodes, ``--episode_idx 0`` through ``N-1`` replaces that episode, ``N`` appends, and negative indices or indices above N are rejected. A replacement receives the current ``--task`` label and may have a different length. Keep FPS, robot type, cameras, and optional signals consistent; existing v2.1 roots are rejected.

Overwrites are saved in a temporary sibling copy before replacing the dataset directory. Interrupted capture leaves the original intact. This needs space for a second dataset copy and rewrites numeric data and normalization statistics, including quantiles, to keep them consistent with the replacement. Existing videos are retained without re-encoding, including unused clips from replaced episodes. Once saving starts, failures retain the staged replacement and report its recovery path; a failed directory swap also retains any original backup. Avoid concurrent writers to the same dataset.

An overwrite also leaves a hidden sibling ``.<dataset-name>.hub-cleanup-pending`` marker. The next upload through either recorder or converter removes obsolete remote shards, even if the overwrite was performed offline and more episodes were appended afterward. The marker is removed only after that upload succeeds.

``--fps`` controls dataset timestamps and video playback rate; actual collection also includes robot and writer latency. Check the printed average frequency on the real setup. The writer initializes and checks the sample schema before the opening ceremony. An interrupted capture is not committed; earlier completed episodes remain available. If the first capture is abandoned before saving starts, its newly created dataset directory is removed so the next run can retry. Files are retained for recovery if saving has already started.

To delete a local direct LeRobot dataset, remove its local cache directory:

    rm -rf "$HF_LEROBOT_HOME/<hf_user>/<dataset_name>"

If ``HF_LEROBOT_HOME`` is not set, use the resolved local dataset path, such as ``~/.cache/huggingface/lerobot/<hf_user>/<dataset_name>``. If the dataset was pushed to Hugging Face Hub, local deletion does not delete the online dataset. Delete the Hub dataset separately:

    hf repo delete <hf_user>/<dataset_name> --repo-type dataset

To visualize the episode collected, run:

    python3 visualize_episodes.py --dataset_dir <data save dir> --episode_idx 0

For existing **v2.1** recordings, create a combined quick-look video with the legacy helper below. This helper does not support v3 shared video files; use a v3-capable LeRobot viewer for new recordings.

    python3 visualize_lerobot_episodes.py \
        --repo-id <hf_user>/<dataset_name> \
        --episode_idx 0

To replay the episode collected with real robot, run:

    python3 replay_episodes.py --dataset_dir <data save dir> --episode_idx 0

For existing **v2.1** recordings, stop both ``one_side_teleop.py`` processes before replaying with the legacy helper below. This helper does not support v3 datasets.

    python3 replay_lerobot_episodes.py \
        --repo-id <hf_user>/<dataset_name> \
        --episode_idx 0

To lower 4 robots before e.g. cutting off power, run:

    python3 sleep.py

## Dataset conversion to LeRobot

``record_episodes_lerobot.py`` now collects directly in **LeRobot v3.0**. The older ``record_episodes.py`` and ``auto_record.sh`` still collect HDF5; convert those existing recordings with ``tools/convert_hdf5_to_lerobot.py``, which also defaults to **v3.0**.

V3 stores multiple episodes in shared Parquet/MP4 files and tracks episode boundaries in Parquet metadata. Both new paths use the official writer for indexing, task labels, image/state statistics, encoding, and finalization. See the [official v3 documentation](https://huggingface.co/docs/lerobot/lerobot-dataset-v3).

Run from the repository root, with ``LEROBOT_PYTHON`` set to your LeRobot environment's Python executable:

    # Inspect raw recordings using the aloha environment
    python3 tools/inspect_aloha_hdf5.py <data save dir>

    # Convert to v3.0 (default); output-root must be a new directory
    "$LEROBOT_PYTHON" tools/convert_hdf5_to_lerobot.py \
        --input-dir <data save dir> \
        --repo-id <hf_user>/<dataset_name> \
        --output-root <new local lerobot output root> \
        --task "<short natural-language task description>" \
        --fps 50 \
        --robot-type aloha \
        --include-velocity --include-effort

    # Reopen and decode samples with the official loader
    "$LEROBOT_PYTHON" tools/validate_lerobot_dataset.py <hf_user>/<dataset_name> \
        --root <local lerobot output root> \
        --expected-fps 50

``--format v3.0`` is accepted explicitly. ``--format v2.1`` retains the legacy export for older trainers. ``--no-videos`` stores image features instead of MP4 for v3 only. Without ``--output-root``, conversion writes to the current directory under the repo ID with ``/`` replaced by ``_``.

Source HDF5 files are kept by default. ``--delete-source-hdf5`` deletes them only after conversion finalizes successfully; validate a representative batch and back up the output before using that option.

To append newly collected HDF5 episodes to a v3.0 export, rerun the conversion command with the same ``--output-root`` and ``--resume``. The converter skips filenames recorded in ``meta/hdf5_sources.json`` and appends unseen filenames; keep source filenames unique and do not reuse them for different recordings. Direct recordings can be appended between conversions without adding entries to this source history. Existing episodes retain their task labels, and ``--task`` applies to new episodes. FPS, robot type, cameras, optional signals and image/video format must match.

Conversion uploads to the Hub by default. Add ``--no-push-to-hub`` to save locally. A hidden sibling file, ``.<output-root-name>.conversion-upload-pending``, tracks deferred or failed uploads; rerunning with ``--resume`` retries the upload even when no new HDF5 files remain. After a successful upload, a resume with no new files makes no Hub calls. The regression checks disable or mock uploads.

Conversion journals planned source filenames in ``meta/hdf5_conversion_pending.json`` before saving. Resume reconciles that journal against finalized episode metadata and physical Parquet frame indices before atomically updating source history. Orphan, missing or duplicate rows stop recovery without changing the data or journal; inspect the retained dataset before retrying. Direct recording is blocked until the journal is reconciled. Upload-only resume reads the existing dataset schema, so it also works after ``--no-push-to-hub --delete-source-hdf5`` removed the inputs.

For an older export without source history, keep all original HDF5 files and give new episodes higher filename numbers. The first resume selects the lowest-numbered originals, restores the old converter's lexicographic order, and verifies their state/action arrays before adopting their filenames; mismatches stop conversion. For nonnumeric filenames, the originals must remain first in lexicographic order. Resume is not supported for v2.1 or together with ``--delete-source-hdf5``.

Hardware-free regression check (requires the LeRobot environment):

    "$LEROBOT_PYTHON" tests/test_lerobot_v30.py
