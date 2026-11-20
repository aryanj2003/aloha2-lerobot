"""Offline checks for tools/rope_overlay.py (no ROS master or camera needed)."""
import os
import sys
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
from rope_overlay import image_to_bgr, render, rope_contours  # noqa: E402


def fake_scene():
    img = np.zeros((480, 640, 3), np.uint8)  # black table
    cv2.polylines(img, [np.array([[100, 300], [250, 200], [400, 320], [550, 220]])], False, (255, 255, 255), 8)
    return img


def test_outline_finds_rope():
    assert len(rope_contours(fake_scene())) >= 1


def test_render_never_modifies_live_frame():
    live = fake_scene()
    before = live.copy()
    ref = np.full_like(live, 128)
    for mode in ("outline", "blend", "off"):
        render(live, ref, rope_contours(ref), mode, 9)
    assert np.array_equal(live, before)


def test_rgb8_message_decodes_to_bgr():
    rgb = np.zeros((2, 3, 3), np.uint8)
    rgb[..., 0] = 200  # red channel in RGB
    msg = SimpleNamespace(data=rgb.tobytes(), height=2, width=3, step=9, encoding="rgb8")
    assert image_to_bgr(msg)[0, 0].tolist() == [0, 0, 200]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
