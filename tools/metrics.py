import numpy as np
from typing import List, Dict, Any
from pathlib import Path
import json

def calculate_task_success_rate(episodes: List[Dict]) -> float:
    """
    Calculate task success rate from episode data.
    
    Args:
        episodes: List of episode dictionaries with 'success' field
        
    Returns:
        Success rate as percentage
    """
    if not episodes:
        return 0.0
    
    successes = sum(1 for ep in episodes if ep.get("success", False))
    return (successes / len(episodes)) * 100.0

def calculate_embedding_drift(embeddings: List[List[float]]) -> Dict[str, float]:
    """
    Measure drift in goal embeddings over time.
    
    Args:
        embeddings: Time series of goal embeddings
        
    Returns:
        Dictionary with drift metrics
    """
    if len(embeddings) < 2:
        return {"mean_drift": 0.0, "max_drift": 0.0, "std_drift": 0.0}
    
    embeddings_arr = np.array(embeddings)
    
    # Calculate consecutive differences
    diffs = np.linalg.norm(embeddings_arr[1:] - embeddings_arr[:-1], axis=1)
    
    return {
        "mean_drift": float(np.mean(diffs)),
        "max_drift": float(np.max(diffs)),
        "std_drift": float(np.std(diffs)),
        "total_drift": float(np.sum(diffs))
    }

def calculate_subtask_stability(subtask_ids: List[str]) -> Dict[str, Any]:
    """
    Measure subtask switching frequency.
    
    Args:
        subtask_ids: Time series of subtask IDs
        
    Returns:
        Dictionary with stability metrics
    """
    if len(subtask_ids) < 2:
        return {"switches": 0, "unique_subtasks": len(set(subtask_ids)), "stability_ratio": 1.0}
    
    # Count switches
    switches = sum(1 for i in range(1, len(subtask_ids)) if subtask_ids[i] != subtask_ids[i-1])
    unique = len(set(subtask_ids))
    
    # Stability ratio: lower is more stable
    stability_ratio = switches / (len(subtask_ids) - 1)
    
    return {
        "switches": switches,
        "unique_subtasks": unique,
        "stability_ratio": stability_ratio,
        "avg_duration": len(subtask_ids) / max(unique, 1)
    }

def calculate_recovery_metrics(perturbation_steps: List[int], recovery_steps: List[int]) -> Dict[str, float]:
    """
    Measure recovery ability after perturbations.
    
    Args:
        perturbation_steps: Step indices where perturbations occurred
        recovery_steps: Step indices where system recovered
        
    Returns:
        Recovery metrics
    """
    if not perturbation_steps or not recovery_steps:
        return {"mean_recovery_time": 0.0, "recovery_rate": 0.0}
    
    recovery_times = []
    for p_step in perturbation_steps:
        # Find next recovery after this perturbation
        later_recoveries = [r for r in recovery_steps if r > p_step]
        if later_recoveries:
            recovery_times.append(later_recoveries[0] - p_step)
    
    recovery_rate = len(recovery_times) / len(perturbation_steps)
    mean_time = np.mean(recovery_times) if recovery_times else 0.0
    
    return {
        "mean_recovery_time": float(mean_time),
        "recovery_rate": recovery_rate,
        "total_perturbations": len(perturbation_steps),
        "successful_recoveries": len(recovery_times)
    }

def load_episode_from_chunk(chunk_path: Path) -> Dict[str, Any]:
    """
    Load an episode from a data chunk directory.
    
    Args:
        chunk_path: Path to chunk directory
        
    Returns:
        Episode data dictionary
    """
    steps_file = chunk_path / "steps.jsonl"
    if not steps_file.exists():
        return {}
    
    steps = []
    with open(steps_file, "r") as f:
        for line in f:
            steps.append(json.loads(line))
    
    # Extract time series
    embeddings = [s["task"]["goal_embedding"] for s in steps if "task" in s and s["task"]]
    subtask_ids = [s["task"]["subtask_id"] for s in steps if "task" in s and s["task"]]
    
    return {
        "chunk_name": chunk_path.name,
        "num_steps": len(steps),
        "steps": steps,
        "embeddings": embeddings,
        "subtask_ids": subtask_ids
    }
