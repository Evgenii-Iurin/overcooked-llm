"""Data structures for storing trajectory information."""

import jax.numpy as jnp
from typing import List, Dict
from dataclasses import dataclass

from .communication import CommunicationMessage, LLMResponse


@dataclass
class TrajectoryStep:
    """Single step in a trajectory."""
    step: int
    prompt_agent0: List[Dict[str, str]]
    completion_agent0: str
    prompt_agent1: List[Dict[str, str]]
    completion_agent1: str
    llm_response_agent0: LLMResponse
    llm_response_agent1: LLMResponse
    actions: Dict[str, int]
    rewards: Dict[str, float]
    dones: Dict[str, bool]
    conversation_history: List[CommunicationMessage]
    obs_agent0: jnp.ndarray
    obs_agent1: jnp.ndarray
    # Plan execution tracking
    plan_valid_agent0: bool = False  # Whether plan passed validation
    plan_executed_agent0: bool = False  # Whether plan was successfully executed by executor
    plan_valid_agent1: bool = False
    plan_executed_agent1: bool = False


@dataclass
class Trajectory:
    """Complete trajectory from an episode."""
    steps: List[TrajectoryStep]
    episode_reward: float
    episode_length: int
    num_deliveries: int
    final_conversation_history: List[CommunicationMessage]

