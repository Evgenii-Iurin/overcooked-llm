"""Reward functions for GRPO training."""

import re
from typing import List, Dict, Optional
from .communication import CommunicationMessage, ActionPlan, LLMResponse
from .trajectory import Trajectory, TrajectoryStep


def environment_reward_func(
    trajectory: Trajectory,
    **kwargs
) -> float:
    """Primary reward from environment performance.
    
    Args:
        trajectory: Complete trajectory from episode
        **kwargs: Additional arguments (ignored)
        
    Returns:
        Episode reward (sum of all step rewards)
    """
    return trajectory.episode_reward


def coordination_reward_func(
    trajectory: Trajectory,
    **kwargs
) -> float:
    """Reward for communication quality and coordination.
    
    Args:
        trajectory: Complete trajectory from episode
        **kwargs: Additional arguments (ignored)
        
    Returns:
        Coordination reward score
    """
    reward = 0.0
    
    # Check if agents communicate
    if trajectory.final_conversation_history:
        reward += 2.0  # Base reward for having communication
        
        # Check if agents reference each other
        messages = [msg.message.lower() for msg in trajectory.final_conversation_history]
        for i, msg in enumerate(messages):
            # Check for references to other agent
            if "agent" in msg or "you" in msg or "other" in msg:
                reward += 1.0
            
            # Check for coordination keywords
            coordination_keywords = [
                "help", "bring", "cook", "deliver", "ingredient",
                "pot", "plate", "together", "coordinate"
            ]
            if any(keyword in msg for keyword in coordination_keywords):
                reward += 0.5
    
    # Check plan complementarity
    for step in trajectory.steps:
        plan0 = step.llm_response_agent0.plan
        plan1 = step.llm_response_agent1.plan
        
        if plan0 and plan1:
            # Check if plans are different (complementary)
            if plan0.action != plan1.action:
                reward += 0.5
            
            # Check if plans mention different ingredients/locations
            if (plan0.target_location and plan1.target_location and
                plan0.target_location != plan1.target_location):
                reward += 0.5
    
    return reward


def plan_executability_reward_func(
    trajectory: Trajectory,
    env,
    **kwargs
) -> float:
    """Reward for plan quality and executability.
    
    Args:
        trajectory: Complete trajectory from episode
        env: Environment instance for validation
        **kwargs: Additional arguments (ignored)
        
    Returns:
        Plan quality reward score
    """
    reward = 0.0
    
    for step in trajectory.steps:
        # Check if plans are parseable (only reward if also valid)
        if step.llm_response_agent0.plan and step.plan_valid_agent0:
            reward += 1.0  # Parsing reward (only for valid plans)
        elif not step.llm_response_agent0.plan:
            reward -= 0.5  # Penalty for no plan
        
        if step.llm_response_agent1.plan and step.plan_valid_agent1:
            reward += 1.0  # Parsing reward (only for valid plans)
        elif not step.llm_response_agent1.plan:
            reward -= 0.5  # Penalty for no plan
        
        # Check if plans passed validation (additional reward on top of parsing)
        if step.plan_valid_agent0:
            reward += 1.5  # Validation passed
        if step.plan_valid_agent1:
            reward += 1.5
        
        # Check if plans were successfully executed by executor (highest reward)
        if step.plan_executed_agent0:
            reward += 2.0  # Successfully executed
        if step.plan_executed_agent1:
            reward += 2.0
        
        # Check if plans have valid actions
        valid_actions = [
            "cook_dish", "pickup_ingredient", "pickup_plate",
            "deliver_dish", "wait", "move_to"
        ]
        
        plan0 = step.llm_response_agent0.plan
        plan1 = step.llm_response_agent1.plan
        
        if plan0 and plan0.action in valid_actions:
            reward += 0.5
        if plan1 and plan1.action in valid_actions:
            reward += 0.5
        
        # Check if plans reference valid locations (basic check)
        if plan0 and plan0.target_location:
            x, y = plan0.target_location
            if 0 <= x < env.width and 0 <= y < env.height:
                reward += 0.5
        
        if plan1 and plan1.target_location:
            x, y = plan1.target_location
            if 0 <= x < env.width and 0 <= y < env.height:
                reward += 0.5
    
    return reward


def _completion_to_text(completion) -> str:
    """Extract plain text from one completion. Handles TRL GRPOTrainer formats.
    
    - str: returned as-is.
    - list of dicts (chat): concatenate 'content' or 'text' from each item.
    - list of str: join.
    - otherwise: str(completion).
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and len(completion) > 0:
        parts = []
        for x in completion:
            if isinstance(x, dict):
                parts.append(x.get("content", x.get("text", "")) or "")
            else:
                parts.append(str(x) if x is not None else "")
        return " ".join(parts)
    return str(completion) if completion is not None else ""


def format_reward_func(
    completions,
    prompts=None,
    completion_ids=None,
    **kwargs
) -> List[float]:
    """Reward for proper XML format in LLM responses.
    
    Compatible with TRL GRPOTrainer: receives prompts=, completions=, completion_ids=.
    Each completion can be a string or a list of chat dicts (e.g. [{"role":"assistant","content":"..."}]).
    Also supports internal usage: format_reward_func(list_of_strings).
    
    Returns:
        List of format rewards (one per completion)
    """
    if completions is None:
        return []
    
    rewards = []
    for completion in completions:
        text = _completion_to_text(completion)
        reward = 0.0
        
        # Check for communication tag
        if re.search(r'<communication>.*?</communication>', text, re.DOTALL):
            reward += 1.0
        
        # Check for plan tag
        if re.search(r'<plan>.*?</plan>', text, re.DOTALL):
            reward += 1.0
        
        # Check for all required sub-tags in communication
        comm_match = re.search(r'<communication>.*?</communication>', text, re.DOTALL)
        if comm_match:
            comm_text = comm_match.group(0)
            if '<to_agent>' in comm_text and '</to_agent>' in comm_text:
                reward += 0.5
            if '<message>' in comm_text and '</message>' in comm_text:
                reward += 0.5
        
        # Check for action tag in plan
        plan_match = re.search(r'<plan>.*?</plan>', text, re.DOTALL)
        if plan_match:
            plan_text = plan_match.group(0)
            if '<action>' in plan_text and '</action>' in plan_text:
                reward += 0.5
        
        rewards.append(reward)
    
    return rewards


def compute_combined_reward(
    trajectory: Trajectory,
    env,
    env_weight: float = 1.0,
    coord_weight: float = 0.3,
    plan_weight: float = 0.2,
    format_weight: float = 0.1,
) -> float:
    """Compute combined reward from all reward functions.
    
    Args:
        trajectory: Complete trajectory
        env: Environment instance
        env_weight: Weight for environment reward
        coord_weight: Weight for coordination reward
        plan_weight: Weight for plan executability reward
        format_weight: Weight for format reward
        
    Returns:
        Combined reward score
    """
    env_reward = environment_reward_func(trajectory)
    coord_reward = coordination_reward_func(trajectory)
    plan_reward = plan_executability_reward_func(trajectory, env)
    
    # Format reward (computed per completion)
    all_completions = []
    for step in trajectory.steps:
        all_completions.append(step.completion_agent0)
        all_completions.append(step.completion_agent1)
    
    format_rewards = format_reward_func(all_completions)
    format_reward = sum(format_rewards) / len(format_rewards) if format_rewards else 0.0
    
    combined = (
        env_weight * env_reward +
        coord_weight * coord_reward +
        plan_weight * plan_reward +
        format_weight * format_reward
    )
    
    return combined


def compute_step_rewards(
    step: TrajectoryStep,
    env,
) -> Dict[str, float]:
    """Compute rewards for a single step (for per-step reward assignment).
    
    Args:
        step: Single trajectory step
        env: Environment instance
        
    Returns:
        Dictionary with rewards for agent_0 and agent_1
    """
    rewards = {
        "agent_0": 0.0,
        "agent_1": 0.0,
    }
    
    # Environment reward (shared)
    step_reward = sum(step.rewards.values())
    rewards["agent_0"] += step_reward
    rewards["agent_1"] += step_reward
    
    # Format rewards (individual)
    format_rewards = format_reward_func([step.completion_agent0, step.completion_agent1])
    rewards["agent_0"] += format_rewards[0] * 0.1
    rewards["agent_1"] += format_rewards[1] * 0.1
    
    # Plan parsing (individual) - small reward for parsing
    # Only give parsing reward if plan is also valid (to avoid rewarding invalid plans)
    if step.llm_response_agent0.plan and step.plan_valid_agent0:
        rewards["agent_0"] += 0.1
    
    if step.llm_response_agent1.plan and step.plan_valid_agent1:
        rewards["agent_1"] += 0.1
    
    # Plan validation (individual) - medium reward for validation
    if step.plan_valid_agent0:
        rewards["agent_0"] += 0.3
    
    if step.plan_valid_agent1:
        rewards["agent_1"] += 0.3
    
    # Plan execution (individual) - highest reward for successful execution
    if step.plan_executed_agent0:
        rewards["agent_0"] += 0.5
    
    if step.plan_executed_agent1:
        rewards["agent_1"] += 0.5
    
    return rewards

