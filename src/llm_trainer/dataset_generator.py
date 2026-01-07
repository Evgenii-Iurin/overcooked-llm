"""Dataset generator for GRPO training."""

import jax
import jax.numpy as jnp
from typing import List, Dict, Callable, Optional
from datasets import Dataset
from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2
from loguru import logger

from .overcooked_llm_wrapper import OvercookedLLMWrapper, Trajectory
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
) -> List[Trajectory]:
    """Generate trajectories from environment using LLM.
    
    Args:
        env: OvercookedV2 environment
        llm_generate_fn: Function that generates LLM responses
        num_episodes: Number of episodes to generate
        max_steps: Maximum steps per episode
        rng_key: Random key for environment
        
    Returns:
        List of trajectories
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)
    
    logger.info(f"Starting trajectory generation: {num_episodes} episodes, max {max_steps} steps each")
    wrapper = OvercookedLLMWrapper(env, llm_generate_fn)
    trajectories = []
    
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
) -> Dataset:
    """Generate GRPO dataset from environment rollouts.
    
    Args:
        env: OvercookedV2 environment
        llm_generate_fn: Function that generates LLM responses
        num_episodes: Number of episodes to generate
        max_steps: Maximum steps per episode
        reward_strategy: Reward assignment strategy
        rng_key: Random key for environment
        
    Returns:
        GRPO-compatible dataset
    """
    # Generate trajectories
    trajectories = generate_trajectories(
        env=env,
        llm_generate_fn=llm_generate_fn,
        num_episodes=num_episodes,
        max_steps=max_steps,
        rng_key=rng_key,
    )
    
    # Convert to dataset
    dataset = trajectory_to_grpo_dataset(
        trajectories=trajectories,
        env=env,
        reward_strategy=reward_strategy,
    )
    
    return dataset


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

