"""Utilities for saving and loading training artifacts."""

import pickle
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from loguru import logger

from .trajectory import Trajectory
from datasets import Dataset


def save_trajectories(
    trajectories: List[Trajectory],
    output_dir: str,
    prefix: str = "trajectories",
    iteration: Optional[int] = None,
) -> str:
    """Save trajectories to pickle file.
    
    Args:
        trajectories: List of trajectories to save
        output_dir: Output directory path
        prefix: File prefix (default: "trajectories")
        iteration: Optional iteration number to include in filename
        
    Returns:
        Path to saved file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create artifacts subdirectory
    artifacts_dir = output_path / "artifacts" / "trajectories"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if iteration is not None:
        filename = f"{prefix}_iter{iteration:04d}_{timestamp}.pkl"
    else:
        filename = f"{prefix}_{timestamp}.pkl"
    
    filepath = artifacts_dir / filename
    
    # Save trajectories
    logger.info(f"Saving {len(trajectories)} trajectories to {filepath}")
    with open(filepath, 'wb') as f:
        pickle.dump(trajectories, f)
    
    logger.success(f"Trajectories saved to {filepath}")
    
    # Also save metadata as JSON
    metadata = {
        "num_trajectories": len(trajectories),
        "timestamp": timestamp,
        "iteration": iteration,
        "total_steps": sum(t.episode_length for t in trajectories),
        "total_reward": sum(t.episode_reward for t in trajectories),
        "avg_reward": sum(t.episode_reward for t in trajectories) / len(trajectories) if trajectories else 0.0,
        "total_deliveries": sum(t.num_deliveries for t in trajectories),
    }
    
    metadata_path = artifacts_dir / f"{prefix}_iter{iteration:04d}_{timestamp}_metadata.json" if iteration else artifacts_dir / f"{prefix}_{timestamp}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return str(filepath)


def load_trajectories(filepath: str) -> List[Trajectory]:
    """Load trajectories from pickle file.
    
    Args:
        filepath: Path to pickle file
        
    Returns:
        List of trajectories
    """
    logger.info(f"Loading trajectories from {filepath}")
    with open(filepath, 'rb') as f:
        trajectories = pickle.load(f)
    
    logger.success(f"Loaded {len(trajectories)} trajectories")
    return trajectories


def save_dataset(
    dataset: Dataset,
    output_dir: str,
    prefix: str = "dataset",
    iteration: Optional[int] = None,
) -> str:
    """Save dataset to disk.
    
    Args:
        dataset: Dataset to save
        output_dir: Output directory path
        prefix: File prefix (default: "dataset")
        iteration: Optional iteration number to include in filename
        
    Returns:
        Path to saved dataset directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create artifacts subdirectory
    artifacts_dir = output_path / "artifacts" / "datasets"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate directory name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if iteration is not None:
        dirname = f"{prefix}_iter{iteration:04d}_{timestamp}"
    else:
        dirname = f"{prefix}_{timestamp}"
    
    dataset_dir = artifacts_dir / dirname
    
    # Save dataset
    logger.info(f"Saving dataset with {len(dataset)} examples to {dataset_dir}")
    dataset.save_to_disk(str(dataset_dir))
    
    logger.success(f"Dataset saved to {dataset_dir}")
    
    # Also save metadata
    metadata = {
        "num_examples": len(dataset),
        "timestamp": timestamp,
        "iteration": iteration,
        "features": list(dataset.features.keys()) if hasattr(dataset, 'features') else [],
    }
    
    metadata_path = dataset_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return str(dataset_dir)


def load_dataset(dataset_dir: str) -> Dataset:
    """Load dataset from disk.
    
    Args:
        dataset_dir: Path to dataset directory
        
    Returns:
        Loaded dataset
    """
    from datasets import load_from_disk
    
    logger.info(f"Loading dataset from {dataset_dir}")
    dataset = load_from_disk(dataset_dir)
    
    logger.success(f"Loaded dataset with {len(dataset)} examples")
    return dataset


def save_training_metadata(
    metadata: Dict[str, Any],
    output_dir: str,
    filename: str = "training_metadata.json",
) -> str:
    """Save training metadata to JSON file.
    
    Args:
        metadata: Dictionary with training metadata
        output_dir: Output directory path
        filename: Metadata filename
        
    Returns:
        Path to saved metadata file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    artifacts_dir = output_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    metadata_path = artifacts_dir / filename
    
    # Add timestamp if not present
    if "timestamp" not in metadata:
        metadata["timestamp"] = datetime.now().isoformat()
    
    logger.info(f"Saving training metadata to {metadata_path}")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.success(f"Metadata saved to {metadata_path}")
    return str(metadata_path)

