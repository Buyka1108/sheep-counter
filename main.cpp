import cv2
import time
import os
import threading
import queue
from datetime import datetime
from ultralytics import YOLO

# =========================
# CONFIG
# =========================

CONFIG = {
  "video_source": 0,
  "model_path": "yolo11n.pt",

  # Detection
  "imgsz": 320,
  "conf": 0.35,

  # Counting line
  "line_y": 0.60,

  # Performance
  "frame_skip": 1,
  "max_queue": 2,

  # Cleanup
  "track_timeout": 3.0,

  # Saving
  "save_snapshots": True,

  # Files
  "log_file": "logs/counts.csv"
}

os.makedirs("logs", exist_ok=True)
os.makedirs("snapshots", exist_ok=True)

# =========================
# SHEEP COUNTER
# =========================

class SheepCounter:

  def __init__(self):

    self.model = YOLO(CONFIG["model_path"])

    self.frame_queue = queue.Queue(maxsize=CONFIG["max_queue"])
    self.save_queue = queue.Queue()

    self.running = True

    self.counts = {
      "IN": 0,
      "OUT": 0
    }

    self.track_history = {}
    self.counted_ids = {}

    self.frame_id = 0

    # CSV init
    if not os.path.exists(CONFIG["log_file"]):
      with open(CONFIG["log_file"], "w") as f:
        f.write("timestamp,direction,track_id\n")

  # =========================
  # CAMERA THREAD
  # =========================

  def capture_loop(self):

    while self.running:

      cap = cv2.VideoCapture(CONFIG["video_source"])

      cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

      if not cap.isOpened():
        print("Camera failed. Retrying...")
        time.sleep(2)
        continue

      while self.running:

        ret, frame = cap.read()

        if not ret:
          print("Frame read failed. Reconnecting...")
          break

        # Drop old frames for low latency
        if self.frame_queue.full():
          try:
            self.frame_queue.get_nowait()
          except:
            pass

        self.frame_queue.put(frame)

      cap.release()

  # =========================
  # SNAPSHOT THREAD
  # =========================

  def snapshot_loop(self):

    while self.running:

      try:
        filename, frame = self.save_queue.get(timeout=1)
        cv2.imwrite(filename, frame)

      except:
        pass

  # =========================
  # CLEANUP OLD TRACKS
  # =========================

  def cleanup_tracks(self):

    now = time.time()

    expired = []

    for tid, data in self.track_history.items():

      if now - data["last_seen"] > CONFIG["track_timeout"]:
        expired.append(tid)

    for tid in expired:

      self.track_history.pop(tid, None)
      self.counted_ids.pop(tid, None)

  # =========================
  # LOG EVENT
  # =========================

  def log_event(self, direction, track_id, frame):

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(CONFIG["log_file"], "a") as f:
      f.write(f"{ts},{direction},{track_id}\n")

    if CONFIG["save_snapshots"]:

      filename = (
        f"snapshots/"
        f"{direction}_{track_id}_{int(time.time())}.jpg"
      )

      self.save_queue.put((filename, frame.copy()))

  # =========================
  # MAIN PROCESS
  # =========================

  def process(self):

    threading.Thread(
      target=self.capture_loop,
      daemon=True
    ).start()

    threading.Thread(
      target=self.snapshot_loop,
      daemon=True
    ).start()

    prev_time = time.time()

    print("Production Sheep Counter Started")

    while self.running:

      if self.frame_queue.empty():
        continue

      frame = self.frame_queue.get()

      self.frame_id += 1

      # Frame skip
      if self.frame_id % CONFIG["frame_skip"] != 0:
        continue

      h, w = frame.shape[:2]

      line_y = int(h * CONFIG["line_y"])

      # =========================
      # YOLO TRACKING
      # =========================

      results = self.model.track(
        frame,
        persist=True,
        classes=[18],
        imgsz=CONFIG["imgsz"],
        conf=CONFIG["conf"],
        tracker="bytetrack.yaml",
        verbose=False
      )

      boxes = results[0].boxes

      if boxes.id is not None:

        ids = boxes.id.int().cpu().tolist()
        coords = boxes.xyxy.cpu().numpy()

        now = time.time()

        for box, track_id in zip(coords, ids):

          x1, y1, x2, y2 = map(int, box)

          cy = int((y1 + y2) / 2)

          # =========================
          # TRACK HISTORY
          # =========================

          if track_id not in self.track_history:

            self.track_history[track_id] = {
              "y": cy,
              "last_seen": now
            }

          else:

            prev_y = self.track_history[track_id]["y"]

            # Crossing logic
            if track_id not in self.counted_ids:

              direction = None

              if prev_y < line_y <= cy:
                direction = "IN"

              elif prev_y > line_y >= cy:
                direction = "OUT"

              if direction:

                self.counts[direction] += 1

                self.counted_ids[track_id] = now

                self.log_event(
                  direction,
                  track_id,
                  frame
                )

            self.track_history[track_id]["y"] = cy
            self.track_history[track_id]["last_seen"] = now

          # =========================
          # DRAW
          # =========================

          cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
          )

          cv2.putText(
            frame,
            f"ID:{track_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
          )

      # Cleanup old tracks
      self.cleanup_tracks()

      # =========================
      # UI
      # =========================

      cv2.line(
        frame,
        (0, line_y),
        (w, line_y),
        (0, 0, 255),
        2
      )

      fps = 1 / (time.time() - prev_time)
      prev_time = time.time()

      text = (
        f"FPS:{fps:.1f} "
        f"IN:{self.counts['IN']} "
        f"OUT:{self.counts['OUT']}"
      )

      cv2.rectangle(
        frame,
        (0, 0),
        (420, 50),
        (0, 0, 0),
        -1
      )

      cv2.putText(
        frame,
        text,
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
      )

      cv2.imshow("Sheep Counter PRO", frame)

      key = cv2.waitKey(1)

      if key == ord("q"):
        self.running = False

    cv2.destroyAllWindows()

# =========================
# MAIN
# =========================

if __name__ == "__main__":

  SheepCounter().process()
