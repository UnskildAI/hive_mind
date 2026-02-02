import json
import os
from pathlib import Path
from PIL import Image
from typing import Iterator, Dict, Any

class ReplayLoader:
    """
    Utility for loading and iterating through robot datasets.
    """
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")

    def get_chunks(self) -> list:
        """Returns list of chunk directories sorted by name."""
        chunks = [d for d in self.dataset_path.iterdir() if d.is_dir() and d.name.startswith("chunk_")]
        return sorted(chunks)

    def iterate_steps(self, chunk_path: Path) -> Iterator[Dict[str, Any]]:
        """Iterates through steps in a single chunk."""
        log_file = chunk_path / "steps.jsonl"
        if not log_file.exists():
            return

        with open(log_file, "r") as f:
            for line in f:
                step_data = json.loads(line)
                
                # Attach image if referenced
                if "image_path" in step_data:
                    full_img_path = chunk_path / step_data["image_path"]
                    if full_img_path.exists():
                        step_data["image"] = Image.open(full_img_path)
                
                yield step_data

    def load_episode(self, chunk_name: str) -> list:
        """Loads a full chunk/episode into memory."""
        chunk_path = self.dataset_path / chunk_name
        return list(self.iterate_steps(chunk_path))

if __name__ == "__main__":
    # Example usage
    LOG_DIR = "/home/mecha/hive_mind/data/logs"
    try:
        loader = ReplayLoader(LOG_DIR)
        chunks = loader.get_chunks()
        if chunks:
            print(f"Found {len(chunks)} chunks.")
            for step in loader.iterate_steps(chunks[0]):
                print(f"Timestamp: {step['timestamp']}, Image size: {step.get('image').size if 'image' in step else 'N/A'}")
                break # Just show first
        else:
            print("No chunks found yet.")
    except Exception as e:
        print(f"Error: {e}")
