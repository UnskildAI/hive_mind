import os
import json
import time
import queue
import threading
import base64
import logging
from pathlib import Path
from datetime import datetime

class RobotDataLogger:
    """
    High-performance logger for robot data collection.
    Saves state and image data in chunked directories.
    """
    def __init__(self, config: dict):
        self.config = config.get("logging", {})
        self.enabled = self.config.get("enabled", False)
        self.log_dir = Path(self.config.get("log_dir", "data/logs"))
        self.chunk_size = self.config.get("chunk_size", 1000)
        self.save_images = self.config.get("save_images", True)
        
        self.logger = logging.getLogger("data_logger")
        
        if self.enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._q = queue.Queue()
            self._stop_event = threading.Event()
            self._thread = threading.Thread(target=self._worker, daemon=True)
            
            self._current_chunk_dir = None
            self._step_count = 0
            
            self._thread.start()
            self.logger.info(f"RobotDataLogger started: dir={self.log_dir}, chunk_size={self.chunk_size}")

    def log_step(self, perception, task, robot, action, image_base64: str = None):
        if not self.enabled:
            return
            
        # Put into queue to avoid blocking main thread
        data = {
            "timestamp": time.time(),
            "perception": perception.model_dump(),
            "task": task.model_dump() if task else None,
            "robot": robot.model_dump(),
            "action": action.model_dump(),
            "image_base64": image_base64
        }
        self._q.put(data)

    def _start_new_chunk(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._current_chunk_dir = self.log_dir / f"chunk_{timestamp}"
        self._current_chunk_dir.mkdir(parents=True, exist_ok=True)
        
        if self.save_images:
            (self._current_chunk_dir / "images").mkdir(exist_ok=True)
            
        self._step_count = 0
        self.logger.info(f"Started new chunk: {self._current_chunk_dir}")

    def _worker(self):
        while not self._stop_event.is_set():
            try:
                # Batch process if possible, but at least 1
                item = self._q.get(timeout=1.0)
                
                if self._current_chunk_dir is None or self._step_count >= self.chunk_size:
                    self._start_new_chunk()

                # 1. Save Image
                img_path = None
                if self.save_images and item["image_base64"]:
                    img_filename = f"step_{self._step_count:06d}.jpg"
                    img_path = f"images/{img_filename}"
                    img_data = base64.b64decode(item["image_base64"])
                    with open(self._current_chunk_dir / img_path, "wb") as f:
                        f.write(img_data)
                
                # 2. Save Step Data
                # Remove large image string from step log, reference local file
                item_for_log = item.copy()
                if img_path:
                    item_for_log["image_path"] = img_path
                    del item_for_log["image_base64"]

                log_file = self._current_chunk_dir / "steps.jsonl"
                with open(log_file, "a") as f:
                    f.write(json.dumps(item_for_log) + "\n")

                self._step_count += 1
                self._q.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"DataLogger worker error: {e}")

    def stop(self):
        if not self.enabled:
            return
        self._stop_event.set()
        self._thread.join(timeout=5.0)
        self.logger.info("DataLogger stopped")
