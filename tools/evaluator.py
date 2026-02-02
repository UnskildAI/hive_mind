import time
import requests
import json
from pathlib import Path
from typing import Dict, List, Optional
from tools.metrics import (
    calculate_task_success_rate,
    calculate_embedding_drift,
    calculate_subtask_stability,
    load_episode_from_chunk
)

class RobotEvaluator:
    """
    Orchestrate repeatable robot evaluation experiments.
    """
    def __init__(self, 
                 pipeline_url: str = "http://localhost:8000/run_step",
                 log_dir: str = "/home/mecha/hive_mind/data/logs"):
        self.pipeline_url = pipeline_url
        self.log_dir = Path(log_dir)
        self.results = []
        
    def run_episode(self, 
                    instruction: str,
                    duration_seconds: float = 30.0,
                    success_checker: Optional[callable] = None) -> Dict:
        """
        Run a single evaluation episode.
        
        Args:
            instruction: Task instruction
            duration_seconds: How long to run
            success_checker: Optional function to determine task success
            
        Returns:
            Episode results dictionary
        """
        print(f"Starting episode: '{instruction}' for {duration_seconds}s")
        
        start_time = time.time()
        steps = 0
        embeddings = []
        subtask_ids = []
        
        # Mock sensor data (in real scenario, this would come from ROS)
        mock_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
        mock_robot_state = {
            "joint_position": [0.0] * 6,
            "joint_velocities": [0.0] * 6,
            "gripper_state": 0.0
        }
        
        while (time.time() - start_time) < duration_seconds:
            try:
                payload = {
                    "image_base64": mock_image,
                    "camera_pose": [[0.0] * 6],
                    "instruction": instruction,
                    "robot_state": mock_robot_state
                }
                
                resp = requests.post(self.pipeline_url, json=payload, timeout=2.0)
                resp.raise_for_status()
                
                # Track metrics (would extract from logged data in real scenario)
                steps += 1
                time.sleep(0.05)  # 20 Hz
                
            except Exception as e:
                print(f"Step failed: {e}")
                break
        
        elapsed = time.time() - start_time
        
        # Load logged data for analysis
        chunks = sorted([d for d in self.log_dir.iterdir() if d.is_dir() and d.name.startswith("chunk_")])
        if chunks:
            latest_chunk = chunks[-1]
            episode_data = load_episode_from_chunk(latest_chunk)
            embeddings = episode_data.get("embeddings", [])
            subtask_ids = episode_data.get("subtask_ids", [])
        
        # Calculate metrics
        drift_metrics = calculate_embedding_drift(embeddings) if embeddings else {}
        stability_metrics = calculate_subtask_stability(subtask_ids) if subtask_ids else {}
        
        # Determine success
        success = success_checker() if success_checker else False
        
        result = {
            "instruction": instruction,
            "duration": elapsed,
            "steps": steps,
            "success": success,
            "drift": drift_metrics,
            "stability": stability_metrics
        }
        
        self.results.append(result)
        print(f"Episode complete: {steps} steps in {elapsed:.2f}s")
        
        return result
    
    def run_experiment(self, 
                       instructions: List[str],
                       episodes_per_instruction: int = 3,
                       duration_per_episode: float = 30.0) -> Dict:
        """
        Run a full experiment with multiple episodes.
        
        Args:
            instructions: List of task instructions to test
            episodes_per_instruction: Repetitions for reproducibility
            duration_per_episode: Duration of each episode
            
        Returns:
            Experiment summary
        """
        print(f"Starting experiment: {len(instructions)} tasks × {episodes_per_instruction} episodes")
        
        all_episodes = []
        
        for instruction in instructions:
            for episode_num in range(episodes_per_instruction):
                print(f"\n=== Task: {instruction} | Episode {episode_num + 1}/{episodes_per_instruction} ===")
                result = self.run_episode(instruction, duration_per_episode)
                all_episodes.append(result)
                
                # Brief pause between episodes
                time.sleep(2.0)
        
        # Generate summary
        summary = {
            "total_episodes": len(all_episodes),
            "success_rate": calculate_task_success_rate(all_episodes),
            "avg_drift": sum(ep["drift"].get("mean_drift", 0) for ep in all_episodes) / len(all_episodes) if all_episodes else 0,
            "avg_stability": sum(ep["stability"].get("stability_ratio", 0) for ep in all_episodes) / len(all_episodes) if all_episodes else 0,
            "episodes": all_episodes
        }
        
        # Save report
        report_path = self.log_dir / "evaluation_report.json"
        with open(report_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n=== Experiment Complete ===")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Avg Drift: {summary['avg_drift']:.4f}")
        print(f"Avg Stability Ratio: {summary['avg_stability']:.4f}")
        print(f"Report saved to: {report_path}")
        
        return summary

if __name__ == "__main__":
    evaluator = RobotEvaluator()
    
    # Example experiment
    tasks = [
        "pick up the red cube",
        "move to home position"
    ]
    
    summary = evaluator.run_experiment(
        instructions=tasks,
        episodes_per_instruction=2,
        duration_per_episode=10.0
    )
