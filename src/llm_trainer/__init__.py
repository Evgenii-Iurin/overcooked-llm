"""LLM Trainer for Overcooked environment using GRPO."""

from .observation_encoder import (
    encode_observation,
    encode_state_for_agent,
    encode_recipe,
    encode_inventory,
)
from .communication import (
    format_agent_prompt,
    parse_llm_response,
    update_conversation_history,
    CommunicationMessage,
    ActionPlan,
    LLMResponse,
    SYSTEM_PROMPT,
)
from .action_executor import ActionExecutor
from .overcooked_llm_wrapper import (
    OvercookedLLMWrapper,
    Trajectory,
    TrajectoryStep,
)
from .reward_functions import (
    environment_reward_func,
    coordination_reward_func,
    plan_executability_reward_func,
    format_reward_func,
    compute_combined_reward,
    compute_step_rewards,
)
from .dataset_generator import (
    generate_trajectories,
    trajectory_to_grpo_dataset,
    generate_grpo_dataset,
)
from .train_grpo import train_grpo_overcooked
from .evaluation import (
    evaluate_trajectory,
    evaluate_multiple_trajectories,
    EvaluationMetrics,
    print_evaluation_report,
)

__all__ = [
    # Observation encoding
    "encode_observation",
    "encode_state_for_agent",
    "encode_recipe",
    "encode_inventory",
    # Communication
    "format_agent_prompt",
    "parse_llm_response",
    "update_conversation_history",
    "CommunicationMessage",
    "ActionPlan",
    "LLMResponse",
    "SYSTEM_PROMPT",
    # Action execution
    "ActionExecutor",
    # Environment wrapper
    "OvercookedLLMWrapper",
    "Trajectory",
    "TrajectoryStep",
    # Rewards
    "environment_reward_func",
    "coordination_reward_func",
    "plan_executability_reward_func",
    "format_reward_func",
    "compute_combined_reward",
    "compute_step_rewards",
    # Dataset generation
    "generate_trajectories",
    "trajectory_to_grpo_dataset",
    "generate_grpo_dataset",
    # Training
    "train_grpo_overcooked",
    # Evaluation
    "evaluate_trajectory",
    "evaluate_multiple_trajectories",
    "EvaluationMetrics",
    "print_evaluation_report",
]

