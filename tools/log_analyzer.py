import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict
from tools.metrics import (
    calculate_embedding_drift,
    calculate_subtask_stability,
    load_episode_from_chunk
)

class LogAnalyzer:
    """
    Analyze and compare robot execution logs.
    """
    def __init__(self, log_dir: str = "/home/mecha/hive_mind/data/logs"):
        self.log_dir = Path(log_dir)
        
    def list_chunks(self) -> List[Path]:
        """List all available chunks"""
        if not self.log_dir.exists():
            return []
        return sorted([d for d in self.log_dir.iterdir() if d.is_dir() and d.name.startswith("chunk_")])
    
    def analyze_chunk(self, chunk_path: Path) -> Dict:
        """Analyze a single chunk"""
        episode = load_episode_from_chunk(chunk_path)
        
        if not episode:
            return {}
        
        # Calculate metrics
        drift_metrics = calculate_embedding_drift(episode["embeddings"])
        stability_metrics = calculate_subtask_stability(episode["subtask_ids"])
        
        return {
            "chunk_name": episode["chunk_name"],
            "num_steps": episode["num_steps"],
            "drift": drift_metrics,
            "stability": stability_metrics
        }
    
    def compare_chunks(self, chunk_paths: List[Path], output_path: str = None):
        """
        Compare multiple chunks side-by-side.
        
        Args:
            chunk_paths: List of chunk directories to compare
            output_path: Optional path to save comparison report
        """
        results = []
        for chunk_path in chunk_paths:
            analysis = self.analyze_chunk(chunk_path)
            if analysis:
                results.append(analysis)
        
        # Generate comparison report
        report = {
            "timestamp": str(Path.ctime(chunk_paths[0])) if chunk_paths else "",
            "chunks_analyzed": len(results),
            "results": results,
            "summary": self._generate_summary(results)
        }
        
        if output_path:
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2)
        
        return report
    
    def _generate_summary(self, results: List[Dict]) -> Dict:
        """Generate aggregate statistics"""
        if not results:
            return {}
        
        mean_drift = np.mean([r["drift"]["mean_drift"] for r in results])
        mean_stability = np.mean([r["stability"]["stability_ratio"] for r in results])
        
        return {
            "avg_drift": mean_drift,
            "avg_stability_ratio": mean_stability,
            "total_steps": sum(r["num_steps"] for r in results)
        }
    
    def visualize_drift(self, chunk_path: Path, save_path: str = None):
        """
        Visualize embedding drift over time.
        
        Args:
            chunk_path: Path to chunk directory
            save_path: Optional path to save figure
        """
        episode = load_episode_from_chunk(chunk_path)
        
        if not episode or not episode["embeddings"]:
            print("No data to visualize")
            return
        
        embeddings = np.array(episode["embeddings"])
        
        # Calculate drift at each step
        drifts = [0.0]
        for i in range(1, len(embeddings)):
            drift = np.linalg.norm(embeddings[i] - embeddings[i-1])
            drifts.append(drift)
        
        # Plot
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(drifts)
        plt.xlabel("Step")
        plt.ylabel("Embedding Drift")
        plt.title(f"Drift Over Time - {episode['chunk_name']}")
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.hist(drifts[1:], bins=30)
        plt.xlabel("Drift Magnitude")
        plt.ylabel("Frequency")
        plt.title("Drift Distribution")
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
        else:
            plt.show()

if __name__ == "__main__":
    analyzer = LogAnalyzer()
    chunks = analyzer.list_chunks()
    
    if chunks:
        print(f"Found {len(chunks)} chunks")
        
        # Analyze latest chunk
        latest = chunks[-1]
        print(f"\nAnalyzing: {latest.name}")
        analysis = analyzer.analyze_chunk(latest)
        print(json.dumps(analysis, indent=2))
        
        # Visualize if multiple chunks
        if len(chunks) >= 2:
            print("\nComparing last 2 chunks...")
            report = analyzer.compare_chunks(chunks[-2:])
            print(json.dumps(report["summary"], indent=2))
    else:
        print("No chunks found. Run the robot first to generate data.")
