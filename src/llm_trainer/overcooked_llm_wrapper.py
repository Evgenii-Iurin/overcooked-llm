"""Environment wrapper that integrates LLM for communication and planning."""

import jax
import jax.numpy as jnp
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2, State
from jaxmarl.environments.overcooked_v2.common import Actions
from loguru import logger

from .observation_encoder import encode_state_for_agent
from .communication import (
    format_agent_prompt,
    parse_llm_response,
    update_conversation_history,
    CommunicationMessage,
    LLMResponse,
)
from .action_executor import ActionExecutor


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


@dataclass
class Trajectory:
    """Complete trajectory from an episode."""
    steps: List[TrajectoryStep]
    episode_reward: float
    episode_length: int
    num_deliveries: int
    final_conversation_history: List[CommunicationMessage]


class OvercookedLLMWrapper:
    """Wrapper that integrates LLM into OvercookedV2 environment."""
    
    def __init__(
        self,
        env: OvercookedV2,
        llm_generate_fn: Callable[[List[Dict[str, str]]], str],
    ):
        """Initialize wrapper.
        
        Args:
            env: OvercookedV2 environment instance
            llm_generate_fn: Function that takes a prompt (list of message dicts) and returns LLM response string
        """
        self.env = env
        self.llm_generate_fn = llm_generate_fn
        self.executor = ActionExecutor(env)
        
        # Action buffers for multi-step plan execution
        self.action_buffers = {f"agent_{i}": [] for i in range(env.num_agents)}
        self.current_plans = {f"agent_{i}": None for i in range(env.num_agents)}
    
    def step_with_llm(
        self,
        obs: Dict[str, jnp.ndarray],
        state: State,
        conversation_history: List[CommunicationMessage],
    ) -> Tuple[Dict[str, int], List[CommunicationMessage], Dict[str, LLMResponse]]:
        """Process one step with LLM communication and planning.
        
        Args:
            obs: Current observations for both agents
            state: Current environment state
            conversation_history: Current conversation history
            
        Returns:
            Tuple of (actions, updated_history, llm_responses)
        """
        # Process agent_0 first
        obs_text_agent0 = encode_state_for_agent(state, 0, self.env)
        prompt_agent0 = format_agent_prompt(obs_text_agent0, 0, conversation_history)
        response_text_agent0 = self.llm_generate_fn(prompt_agent0)
        llm_response_agent0 = parse_llm_response(response_text_agent0, "agent_0", "agent_1")
        
        # Update conversation history with agent_0's message
        updated_history = conversation_history
        if llm_response_agent0.communication:
            updated_history = update_conversation_history(
                updated_history,
                llm_response_agent0.communication
            )
        
        # Process agent_1 with updated history
        obs_text_agent1 = encode_state_for_agent(state, 1, self.env)
        prompt_agent1 = format_agent_prompt(obs_text_agent1, 1, updated_history)
        response_text_agent1 = self.llm_generate_fn(prompt_agent1)
        llm_response_agent1 = parse_llm_response(response_text_agent1, "agent_1", "agent_0")
        
        # Update conversation history with agent_1's message
        if llm_response_agent1.communication:
            updated_history = update_conversation_history(
                updated_history,
                llm_response_agent1.communication
            )
        
        # Execute plans to get actions
        actions = {}
        
        # Check if we have remaining actions in buffer
        for agent_id in range(self.env.num_agents):
            agent_key = f"agent_{agent_id}"
            
            # If buffer is empty or plan changed, generate new actions
            if (not self.action_buffers[agent_key] or 
                self.current_plans[agent_key] != (llm_response_agent0.plan if agent_id == 0 else llm_response_agent1.plan)):
                
                # Get plan for this agent
                plan = llm_response_agent0.plan if agent_id == 0 else llm_response_agent1.plan
                self.current_plans[agent_key] = plan
                
                # Generate action sequence
                action_sequence = self.executor.execute_plan(plan, state, agent_id)
                self.action_buffers[agent_key] = action_sequence
            
            # Get next action from buffer
            if self.action_buffers[agent_key]:
                actions[agent_key] = int(self.action_buffers[agent_key][0])
                self.action_buffers[agent_key] = self.action_buffers[agent_key][1:]
            else:
                actions[agent_key] = int(Actions.stay)
        
        llm_responses = {
            "agent_0": llm_response_agent0,
            "agent_1": llm_response_agent1,
        }
        
        return actions, updated_history, llm_responses
    
    def collect_trajectory(
        self,
        key: jax.random.PRNGKey,
        max_steps: int = 400,
        reset_key: Optional[jax.random.PRNGKey] = None,
    ) -> Trajectory:
        """Collect a complete trajectory from the environment.
        
        Args:
            key: Random key for environment steps
            max_steps: Maximum number of steps
            reset_key: Optional separate key for reset
            
        Returns:
            Trajectory object with all steps
        """
        # Reset environment
        if reset_key is None:
            reset_key, key = jax.random.split(key)
        
        obs, state = self.env.reset(reset_key)
        logger.debug(f"Environment reset: recipe={state.recipe}, agents at positions "
                    f"({state.agents.pos.x[0]}, {state.agents.pos.y[0]}), "
                    f"({state.agents.pos.x[1]}, {state.agents.pos.y[1]})")
        
        # Initialize conversation history and action buffers
        conversation_history = []
        self.action_buffers = {f"agent_{i}": [] for i in range(self.env.num_agents)}
        self.current_plans = {f"agent_{i}": None for i in range(self.env.num_agents)}
        
        steps = []
        episode_reward = 0.0
        num_deliveries = 0
        
        for step in range(max_steps):
            # Process with LLM
            actions, conversation_history, llm_responses = self.step_with_llm(
                obs, state, conversation_history
            )
            
            # Log communication messages if any
            if llm_responses["agent_0"].communication:
                msg = llm_responses["agent_0"].communication
                logger.debug(f"Step {step}: Agent 0 → Agent 1: {msg.message}")
            if llm_responses["agent_1"].communication:
                msg = llm_responses["agent_1"].communication
                logger.debug(f"Step {step}: Agent 1 → Agent 0: {msg.message}")
            
            # Log plans
            if llm_responses["agent_0"].plan:
                plan = llm_responses["agent_0"].plan
                logger.debug(f"Step {step}: Agent 0 plan: {plan.action} - {plan.description}")
            if llm_responses["agent_1"].plan:
                plan = llm_responses["agent_1"].plan
                logger.debug(f"Step {step}: Agent 1 plan: {plan.action} - {plan.description}")
            
            # Step environment
            key, step_key = jax.random.split(key)
            step_keys = jax.random.split(step_key, 1)  # For single env
            obs_new, state_new, rewards, dones, infos = self.env.step(
                step_keys[0], state, actions
            )
            
            # Track rewards and deliveries
            step_reward = sum(rewards.values())
            episode_reward += step_reward
            if state_new.new_correct_delivery:
                num_deliveries += 1
                logger.info(f"Step {step}: Correct delivery! Total deliveries: {num_deliveries}")
            
            logger.debug(f"Step {step}: actions={actions}, rewards={rewards}, total_reward={episode_reward:.2f}")
            
            # Store step
            step_data = TrajectoryStep(
                step=step,
                prompt_agent0=format_agent_prompt(
                    encode_state_for_agent(state, 0, self.env),
                    0,
                    conversation_history[:len(conversation_history)-2] if len(conversation_history) >= 2 else [],
                ),
                completion_agent0=llm_responses["agent_0"].raw_response,
                prompt_agent1=format_agent_prompt(
                    encode_state_for_agent(state, 1, self.env),
                    1,
                    conversation_history[:len(conversation_history)-1] if len(conversation_history) >= 1 else [],
                ),
                completion_agent1=llm_responses["agent_1"].raw_response,
                llm_response_agent0=llm_responses["agent_0"],
                llm_response_agent1=llm_responses["agent_1"],
                actions=actions,
                rewards=rewards,
                dones=dones,
                conversation_history=conversation_history.copy(),
                obs_agent0=obs["agent_0"],
                obs_agent1=obs["agent_1"],
            )
            steps.append(step_data)
            
            # Check if done
            if dones.get("__all__", False) or state_new.terminal:
                logger.debug(f"Episode terminated at step {step + 1}")
                break
            
            # Update for next iteration
            obs = obs_new
            state = state_new
        
        trajectory = Trajectory(
            steps=steps,
            episode_reward=episode_reward,
            episode_length=len(steps),
            num_deliveries=num_deliveries,
            final_conversation_history=conversation_history,
        )
        
        logger.debug(f"Trajectory collected: {len(steps)} steps, {len(conversation_history)} messages, "
                    f"{num_deliveries} deliveries, reward={episode_reward:.2f}")
        
        return trajectory
    
    def reset_action_buffers(self):
        """Reset action buffers (call after episode ends)."""
        self.action_buffers = {f"agent_{i}": [] for i in range(self.env.num_agents)}
        self.current_plans = {f"agent_{i}": None for i in range(self.env.num_agents)}

