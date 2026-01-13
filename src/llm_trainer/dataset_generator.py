"""Dataset generator for GRPO training."""

import jax
import jax.numpy as jnp
from typing import List, Dict, Callable, Optional, Tuple
from datasets import Dataset
from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2
from loguru import logger

from .overcooked_llm_wrapper import OvercookedLLMWrapper
from .trajectory import Trajectory
from .reward_functions import (
    compute_combined_reward,
    compute_step_rewards,
)


def generate_trajectories(
    env: OvercookedV2,
    llm_generate_fn: Callable[[List[Dict[str, str]]], str],
    num_episodes: int,
    max_steps: int = 400,
    rng_key: Optional[jax.random.PRNGKey] = None,
    use_mlflow: bool = False,
    iteration: Optional[int] = None,
) -> List[Trajectory]:
    """Generate trajectories from environment using LLM.
    
    Args:
        env: OvercookedV2 environment
        llm_generate_fn: Function that generates LLM responses
        num_episodes: Number of episodes to generate
        max_steps: Maximum steps per episode
        rng_key: Random key for environment
        use_mlflow: Whether to log metrics to MLflow during generation
        iteration: Current iteration number (for MLflow step tracking)
        
    Returns:
        List of trajectories
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)
    
    logger.info(f"Starting trajectory generation: {num_episodes} episodes, max {max_steps} steps each")
    wrapper = OvercookedLLMWrapper(env, llm_generate_fn)
    trajectories = []
    
    # Initialize MLflow if needed
    if use_mlflow:
        try:
            import mlflow
        except ImportError:
            logger.warning("MLflow requested but not installed. Continuing without MLflow logging.")
            use_mlflow = False
            mlflow = None
    else:
        mlflow = None
    
    for episode in range(num_episodes):
        logger.info(f"Episode {episode + 1}/{num_episodes}: Starting...")
        rng_key, episode_key = jax.random.split(rng_key)
        trajectory = wrapper.collect_trajectory(episode_key, max_steps=max_steps)
        trajectories.append(trajectory)
        wrapper.reset_action_buffers()
        
        # Log episode statistics
        logger.success(
            f"Episode {episode + 1}/{num_episodes} completed: "
            f"length={trajectory.episode_length}, "
            f"reward={trajectory.episode_reward:.2f}, "
            f"deliveries={trajectory.num_deliveries}"
        )
        
        # Log metrics to MLflow in real-time
        if use_mlflow and mlflow is not None:
            try:
                # Calculate running averages
                avg_length = sum(t.episode_length for t in trajectories) / len(trajectories)
                avg_reward = sum(t.episode_reward for t in trajectories) / len(trajectories)
                total_deliveries = sum(t.num_deliveries for t in trajectories)
                avg_deliveries = total_deliveries / len(trajectories)
                
                # Calculate step number: (iteration * num_episodes) + episode
                # This allows tracking progress across iterations
                step = (iteration * num_episodes + episode + 1) if iteration is not None else (episode + 1)
                
                mlflow.log_metrics({
                    f"episode_generation/{episode + 1}/length": trajectory.episode_length,
                    f"episode_generation/{episode + 1}/reward": trajectory.episode_reward,
                    f"episode_generation/{episode + 1}/deliveries": trajectory.num_deliveries,
                    # Running averages
                    f"episode_generation/running_avg_length": avg_length,
                    f"episode_generation/running_avg_reward": avg_reward,
                    f"episode_generation/running_total_deliveries": total_deliveries,
                    f"episode_generation/running_avg_deliveries": avg_deliveries,
                    f"episode_generation/episodes_completed": len(trajectories),
                }, step=step)
                
                logger.debug(f"Episode {episode + 1} metrics logged to MLflow")
            except Exception as e:
                logger.warning(f"Failed to log episode metrics to MLflow: {e}")
    
    logger.info(f"Trajectory generation complete: {len(trajectories)} episodes collected")
    return trajectories


def trajectory_to_grpo_dataset(
    trajectories: List[Trajectory],
    env: OvercookedV2,
    reward_strategy: str = "episode_shared",
) -> Dataset:
    """Convert trajectories to GRPO-compatible dataset.
    
    Args:
        trajectories: List of trajectories
        env: Environment instance (for reward computation)
        reward_strategy: How to assign rewards
            - "episode_shared": All steps in episode get same episode reward
            - "step_individual": Each step gets individual reward
            
    Returns:
        Dataset with prompts, completions, and rewards
    """
    logger.info(f"Converting {len(trajectories)} trajectories to GRPO dataset (strategy: {reward_strategy})")
    examples = []
    
    for traj_idx, trajectory in enumerate(trajectories):
        # Compute episode-level reward
        episode_reward = compute_combined_reward(trajectory, env)
        logger.debug(f"Trajectory {traj_idx + 1}: {len(trajectory.steps)} steps, reward={episode_reward:.2f}")
        
        for step in trajectory.steps:
            # Agent 0 example
            step_reward_agent0 = episode_reward if reward_strategy == "episode_shared" else compute_step_rewards(step, env)["agent_0"]
            
            examples.append({
                "prompt": step.prompt_agent0,
                "completion": step.completion_agent0,
                "reward": float(step_reward_agent0),
                "agent_id": "agent_0",
                "episode": traj_idx,
                "step": step.step,
                "conversation_history": [
                    {
                        "from": msg.from_agent,
                        "to": msg.to_agent,
                        "message": msg.message,
                    }
                    for msg in step.conversation_history
                ],
            })
            
            # Agent 1 example
            step_reward_agent1 = episode_reward if reward_strategy == "episode_shared" else compute_step_rewards(step, env)["agent_1"]
            
            examples.append({
                "prompt": step.prompt_agent1,
                "completion": step.completion_agent1,
                "reward": float(step_reward_agent1),
                "agent_id": "agent_1",
                "episode": traj_idx,
                "step": step.step,
                "conversation_history": [
                    {
                        "from": msg.from_agent,
                        "to": msg.to_agent,
                        "message": msg.message,
                    }
                    for msg in step.conversation_history
                ],
            })
    
    logger.success(f"Dataset conversion complete: {len(examples)} examples created")
    return Dataset.from_list(examples)


def generate_grpo_dataset(
    env: OvercookedV2,
    llm_generate_fn: Callable[[List[Dict[str, str]]], str],
    num_episodes: int,
    max_steps: int = 400,
    reward_strategy: str = "episode_shared",
    rng_key: Optional[jax.random.PRNGKey] = None,
    save_artifacts: bool = True,
    output_dir: Optional[str] = None,
    iteration: Optional[int] = None,
    use_mlflow: bool = False,
) -> Tuple[Dataset, List[Trajectory]]:
    """Generate GRPO dataset from environment rollouts.
    
    Args:
        env: OvercookedV2 environment
        llm_generate_fn: Function that generates LLM responses
        num_episodes: Number of episodes to generate
        max_steps: Maximum steps per episode
        reward_strategy: Reward assignment strategy
        rng_key: Random key for environment
        save_artifacts: Whether to save trajectories and dataset to disk
        output_dir: Output directory for artifacts. All artifacts will be saved to {output_dir}/artifacts/
        iteration: Optional iteration number for artifact naming
        use_mlflow: Whether to log metrics to MLflow during generation
        
    Returns:
        Tuple of (GRPO-compatible dataset, list of trajectories)
    """
    # Generate trajectories
    trajectories = generate_trajectories(
        env=env,
        llm_generate_fn=llm_generate_fn,
        num_episodes=num_episodes,
        max_steps=max_steps,
        rng_key=rng_key,
        use_mlflow=use_mlflow,
        iteration=iteration,
    )
    
    # Save trajectories if requested
    if save_artifacts and output_dir:
        from .artifacts import save_trajectories
        save_trajectories(trajectories, output_dir, iteration=iteration)
    
    # Convert to dataset
    dataset = trajectory_to_grpo_dataset(
        trajectories=trajectories,
        env=env,
        reward_strategy=reward_strategy,
    )
    
    # Save dataset if requested
    if save_artifacts and output_dir:
        from .artifacts import save_dataset
        save_dataset(dataset, output_dir, iteration=iteration)
    
    return dataset, trajectories


def filter_dataset_by_reward(
    dataset: Dataset,
    min_reward: float = None,
    max_examples: int = None,
) -> Dataset:
    """Filter dataset by reward threshold or limit examples.
    
    Args:
        dataset: Input dataset
        min_reward: Minimum reward threshold
        max_examples: Maximum number of examples to keep
        
    Returns:
        Filtered dataset
    """
    if min_reward is not None:
        dataset = dataset.filter(lambda x: x["reward"] >= min_reward)
    
    if max_examples is not None and len(dataset) > max_examples:
        # Sort by reward and take top examples
        dataset = dataset.sort("reward", reverse=True)
        dataset = dataset.select(range(max_examples))
    
    return dataset

