#!/usr/bin/env python3
"""Live placement guide: show a camera topic with a saved reference drawn on top.

Display only. This node subscribes to one image topic and never publishes,
so it cannot change anything that record_episodes.py records.

Keys (click the window first):
  s      save the current raw frame as the reference
  e      switch view: outline -> blend -> off
  + / -  stronger / weaker overlay
  q      quit
"""
import argparse
import os
import threading

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image

MODES = ("outline", "blend", "off")


def image_to_bgr(msg):
    """Convert a sensor_msgs/Image to a BGR numpy array (no cv_bridge needed)."""
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    enc = msg.encoding.lower()
    if enc in ("rgb8", "bgr8"):
        img = buf.reshape(msg.height, msg.step)[:, : msg.width * 3].reshape(msg.height, msg.width, 3)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if enc == "rgb8" else img.copy()
    if enc == "mono8":
        img = buf.reshape(msg.height, msg.step)[:, : msg.width]
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if enc in ("yuv422", "uyvy"):
        img = buf.reshape(msg.height, msg.step)[:, : msg.width * 2].reshape(msg.height, msg.width, 2)
        return cv2.cvtColor(img, cv2.COLOR_YUV2BGR_UYVY)
    if enc in ("yuyv", "yuv422_yuy2"):
        img = buf.reshape(msg.height, msg.step)[:, : msg.width * 2].reshape(msg.height, msg.width, 2)
        return cv2.cvtColor(img, cv2.COLOR_YUV2BGR_YUYV)
    raise ValueError(f"unsupported image encoding: {msg.encoding}")


def rope_contours(ref_bgr, min_area=150):
    """Outline of the bright rope on the dark table (Otsu threshold)."""
    gray = cv2.GaussianBlur(cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) >= min_area]


def render(live, ref, contours, mode, strength):
    """Return a NEW image for display; `live` is never modified."""
    out = live.copy()
    if ref is None or mode == "off":
        return out
    if ref.shape != live.shape:
        ref = cv2.resize(ref, (live.shape[1], live.shape[0]))
    if mode == "blend":
        alpha = strength / 10.0
        return cv2.addWeighted(live, 1.0 - alpha, ref, alpha, 0)
    cv2.drawContours(out, contours, -1, (0, 255, 0), max(1, strength // 2))
    return out


class Overlay:
    def __init__(self, topic, ref_path):
        self.topic, self.ref_path = topic, ref_path
        self.lock = threading.Lock()
        self.frame = None
        self.mode, self.strength = "outline", 5
        self.ref = cv2.imread(ref_path) if os.path.exists(ref_path) else None
        self.contours = rope_contours(self.ref) if self.ref is not None else []
        rospy.Subscriber(topic, Image, self._on_image, queue_size=1, buff_size=2**24)

    def _on_image(self, msg):
        try:
            img = image_to_bgr(msg)
        except ValueError as e:
            rospy.logerr_throttle(5, str(e))
            return
        with self.lock:
            self.frame = img

    def save_reference(self, frame):
        os.makedirs(os.path.dirname(self.ref_path) or ".", exist_ok=True)
        tmp = self.ref_path + ".tmp.png"
        if not cv2.imwrite(tmp, frame):
            rospy.logerr(f"could not write {tmp}")
            return
        os.replace(tmp, self.ref_path)  # atomic: an old reference is never half-overwritten
        self.ref, self.contours = frame.copy(), rope_contours(frame)
        rospy.loginfo(f"saved reference -> {self.ref_path} ({len(self.contours)} outline pieces)")

    def run(self):
        win = f"rope overlay: {self.topic}"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        while not rospy.is_shutdown():
            with self.lock:
                frame = self.frame
            if frame is None:
                view = np.zeros((480, 640, 3), np.uint8)
                cv2.putText(view, f"waiting for {self.topic} ...", (20, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            else:
                view = render(frame, self.ref, self.contours, self.mode, self.strength)
                status = "no reference (press s)" if self.ref is None else f"{self.mode}  strength {self.strength}/9"
                cv2.putText(view, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow(win, view)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q") or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("s") and frame is not None:
                self.save_reference(frame)
            elif key == ord("e"):
                self.mode = MODES[(MODES.index(self.mode) + 1) % len(MODES)]
            elif key in (ord("+"), ord("=")):
                self.strength = min(9, self.strength + 1)
            elif key in (ord("-"), ord("_")):
                self.strength = max(1, self.strength - 1)
        cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--topic", default="/usb_cam_high/image_raw", help="sensor_msgs/Image topic")
    p.add_argument("--ref", default=os.path.expanduser("~/aloha_refs/rope_reference.png"),
                   help="reference image file (loaded if it exists, written on 's')")
    args = p.parse_args(rospy.myargv()[1:])
    rospy.init_node("rope_overlay", anonymous=True)
    Overlay(args.topic, args.ref).run()


if __name__ == "__main__":
    main()
