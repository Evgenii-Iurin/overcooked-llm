"""Evaluation metrics for GRPO training."""

import re
from typing import List, Dict, Optional
from dataclasses import dataclass
from .overcooked_llm_wrapper import Trajectory, TrajectoryStep
from .communication import CommunicationMessage


@dataclass
class EvaluationMetrics:
    """Evaluation metrics for a trajectory."""
    episode_reward: float
    episode_length: int
    num_deliveries: int
    communication_frequency: float  # Messages per step
    communication_quality: float  # 0-1 score
    plan_success_rate: float  # 0-1 score
    plan_complementarity: float  # 0-1 score
    information_asymmetry_handling: float  # 0-1 score


def evaluate_trajectory(trajectory: Trajectory) -> EvaluationMetrics:
    """Evaluate a single trajectory.
    
    Args:
        trajectory: Trajectory to evaluate
        
    Returns:
        EvaluationMetrics object
    """
    # Basic metrics
    episode_reward = trajectory.episode_reward
    episode_length = trajectory.episode_length
    num_deliveries = trajectory.num_deliveries
    
    # Communication frequency
    total_messages = len(trajectory.final_conversation_history)
    communication_frequency = total_messages / max(episode_length, 1)
    
    # Communication quality
    communication_quality = compute_communication_quality(trajectory)
    
    # Plan success rate
    plan_success_rate = compute_plan_success_rate(trajectory)
    
    # Plan complementarity
    plan_complementarity = compute_plan_complementarity(trajectory)
    
    # Information asymmetry handling
    info_asymmetry = compute_information_asymmetry_handling(trajectory)
    
    return EvaluationMetrics(
        episode_reward=episode_reward,
        episode_length=episode_length,
        num_deliveries=num_deliveries,
        communication_frequency=communication_frequency,
        communication_quality=communication_quality,
        plan_success_rate=plan_success_rate,
        plan_complementarity=plan_complementarity,
        information_asymmetry_handling=info_asymmetry,
    )


def compute_communication_quality(trajectory: Trajectory) -> float:
    """Compute communication quality score.
    
    Args:
        trajectory: Trajectory to evaluate
        
    Returns:
        Score between 0 and 1
    """
    if not trajectory.final_conversation_history:
        return 0.0
    
    score = 0.0
    messages = trajectory.final_conversation_history
    
    # Check for references to other agent
    for msg in messages:
        msg_lower = msg.message.lower()
        
        # References to other agent
        if any(word in msg_lower for word in ["agent", "you", "other", "partner"]):
            score += 0.2
        
        # Coordination keywords
        coord_keywords = [
            "help", "bring", "cook", "deliver", "ingredient",
            "pot", "plate", "together", "coordinate", "need"
        ]
        if any(keyword in msg_lower for keyword in coord_keywords):
            score += 0.2
        
        # Action-oriented language
        action_keywords = ["going", "will", "can", "should", "let's"]
        if any(keyword in msg_lower for keyword in action_keywords):
            score += 0.1
    
    # Check for message-response patterns
    if len(messages) >= 2:
        # Check if agents respond to each other
        for i in range(len(messages) - 1):
            if messages[i].to_agent != messages[i+1].from_agent:
                score += 0.1
    
    return min(score, 1.0)


def compute_plan_success_rate(trajectory: Trajectory) -> float:
    """Compute plan success rate (parseable and valid plans).
    
    Args:
        trajectory: Trajectory to evaluate
        
    Returns:
        Score between 0 and 1
    """
    if not trajectory.steps:
        return 0.0
    
    valid_plans = 0
    total_plans = 0
    
    valid_actions = [
        "cook_dish", "pickup_ingredient", "pickup_plate",
        "deliver_dish", "wait", "move_to"
    ]
    
    for step in trajectory.steps:
        # Check agent 0 plan
        if step.llm_response_agent0.plan:
            total_plans += 1
            if step.llm_response_agent0.plan.action in valid_actions:
                valid_plans += 1
        
        # Check agent 1 plan
        if step.llm_response_agent1.plan:
            total_plans += 1
            if step.llm_response_agent1.plan.action in valid_actions:
                valid_plans += 1
    
    if total_plans == 0:
        return 0.0
    
    return valid_plans / total_plans


def compute_plan_complementarity(trajectory: Trajectory) -> float:
    """Compute how well plans complement each other.
    
    Args:
        trajectory: Trajectory to evaluate
        
    Returns:
        Score between 0 and 1
    """
    if not trajectory.steps:
        return 0.0
    
    complementarity_scores = []
    
    for step in trajectory.steps:
        plan0 = step.llm_response_agent0.plan
        plan1 = step.llm_response_agent1.plan
        
        if not plan0 or not plan1:
            complementarity_scores.append(0.0)
            continue
        
        score = 0.0
        
        # Different actions are good (complementary)
        if plan0.action != plan1.action:
            score += 0.4
        
        # Different target locations are good
        if (plan0.target_location and plan1.target_location and
            plan0.target_location != plan1.target_location):
            score += 0.3
        
        # Different ingredients are good
        if (plan0.ingredients and plan1.ingredients and
            set(plan0.ingredients) != set(plan1.ingredients)):
            score += 0.3
        
        complementarity_scores.append(score)
    
    if not complementarity_scores:
        return 0.0
    
    return sum(complementarity_scores) / len(complementarity_scores)


def compute_information_asymmetry_handling(trajectory: Trajectory) -> float:
    """Compute how well agents handle partial information.
    
    Args:
        trajectory: Trajectory to evaluate
        
    Returns:
        Score between 0 and 1
    """
    if not trajectory.steps:
        return 0.0
    
    scores = []
    
    for step in trajectory.steps:
        score = 0.0
        
        # Check if agents mention what they see (acknowledging partial info)
        comm0 = step.llm_response_agent0.communication
        comm1 = step.llm_response_agent1.communication
        
        if comm0:
            msg0 = comm0.message.lower()
            # Mentions of visibility/observation
            if any(word in msg0 for word in ["see", "visible", "can't see", "don't see", "observe"]):
                score += 0.25
        
        if comm1:
            msg1 = comm1.message.lower()
            if any(word in msg1 for word in ["see", "visible", "can't see", "don't see", "observe"]):
                score += 0.25
        
        # Check if agents ask for information
        if comm0 and any(word in comm0.message.lower() for word in ["where", "what", "can you", "do you"]):
            score += 0.25
        
        if comm1 and any(word in comm1.message.lower() for word in ["where", "what", "can you", "do you"]):
            score += 0.25
        
        scores.append(min(score, 1.0))
    
    if not scores:
        return 0.0
    
    return sum(scores) / len(scores)


def evaluate_multiple_trajectories(trajectories: List[Trajectory]) -> Dict[str, float]:
    """Evaluate multiple trajectories and compute aggregate metrics.
    
    Args:
        trajectories: List of trajectories
        
    Returns:
        Dictionary with aggregate metrics
    """
    if not trajectories:
        return {}
    
    metrics_list = [evaluate_trajectory(traj) for traj in trajectories]
    
    return {
        "mean_episode_reward": sum(m.episode_reward for m in metrics_list) / len(metrics_list),
        "std_episode_reward": (
            sum((m.episode_reward - sum(m.episode_reward for m in metrics_list) / len(metrics_list))**2 
                for m in metrics_list) / len(metrics_list)
        ) ** 0.5,
        "mean_episode_length": sum(m.episode_length for m in metrics_list) / len(metrics_list),
        "mean_deliveries": sum(m.num_deliveries for m in metrics_list) / len(metrics_list),
        "mean_communication_frequency": sum(m.communication_frequency for m in metrics_list) / len(metrics_list),
        "mean_communication_quality": sum(m.communication_quality for m in metrics_list) / len(metrics_list),
        "mean_plan_success_rate": sum(m.plan_success_rate for m in metrics_list) / len(metrics_list),
        "mean_plan_complementarity": sum(m.plan_complementarity for m in metrics_list) / len(metrics_list),
        "mean_information_asymmetry": sum(m.information_asymmetry_handling for m in metrics_list) / len(metrics_list),
    }


def print_evaluation_report(metrics: Dict[str, float]):
    """Print formatted evaluation report.
    
    Args:
        metrics: Dictionary of metrics
    """
    print("\n" + "="*60)
    print("EVALUATION REPORT")
    print("="*60)
    print(f"Mean Episode Reward: {metrics.get('mean_episode_reward', 0):.2f} ± {metrics.get('std_episode_reward', 0):.2f}")
    print(f"Mean Episode Length: {metrics.get('mean_episode_length', 0):.1f}")
    print(f"Mean Deliveries: {metrics.get('mean_deliveries', 0):.2f}")
    print(f"\nCommunication Metrics:")
    print(f"  Frequency: {metrics.get('mean_communication_frequency', 0):.3f} messages/step")
    print(f"  Quality: {metrics.get('mean_communication_quality', 0):.3f}")
    print(f"\nPlanning Metrics:")
    print(f"  Success Rate: {metrics.get('mean_plan_success_rate', 0):.3f}")
    print(f"  Complementarity: {metrics.get('mean_plan_complementarity', 0):.3f}")
    print(f"  Info Asymmetry Handling: {metrics.get('mean_information_asymmetry', 0):.3f}")
    print("="*60 + "\n")

