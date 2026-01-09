"""Environment wrapper that integrates LLM for communication and planning."""

import jax
import jax.numpy as jnp
from typing import Dict, List, Tuple, Optional, Callable
from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2, State
from jaxmarl.environments.overcooked_v2.common import Actions
from rich.console import Console

from .observation_encoder import encode_state_for_agent
from .communication import (
    format_agent_prompt,
    parse_llm_response,
    update_conversation_history,
    validate_plan,
    CommunicationMessage,
    LLMResponse,
    ActionPlan,
)
from .action_executor import ActionExecutor
from .trajectory import Trajectory, TrajectoryStep
from .display import display_agent_context, display_step_summary

console = Console()


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
        # Store validation errors from previous step to include in next prompt
        self.previous_validation_errors = {f"agent_{i}": [] for i in range(env.num_agents)}
    
    
    def step_with_llm(
        self,
        obs: Dict[str, jnp.ndarray],
        state: State,
        conversation_history: List[CommunicationMessage],
        step: int = 0,
    ) -> Tuple[Dict[str, int], List[CommunicationMessage], Dict[str, LLMResponse], Dict[str, bool]]:
        """Process one step with LLM communication and planning.
        
        Args:
            obs: Current observations for both agents
            state: Current environment state
            conversation_history: Current conversation history
            step: Current step number (for display)
            
        Returns:
            Tuple of (actions, updated_history, llm_responses, plan_status)
            where plan_status contains validation and execution status for both agents
        """
        # Process agent_0 first (with previous validation errors if any)
        obs_text_agent0 = encode_state_for_agent(state, 0, self.env)
        
        prompt_agent0 = format_agent_prompt(
            obs_text_agent0, 
            0, 
            conversation_history,
            previous_validation_errors=self.previous_validation_errors["agent_0"]
        )
        response_text_agent0 = self.llm_generate_fn(prompt_agent0)
        llm_response_agent0 = parse_llm_response(response_text_agent0, "agent_0", "agent_1")
        
        # Update conversation history with agent_0's message
        updated_history = conversation_history
        if llm_response_agent0.communication:
            updated_history = update_conversation_history(
                updated_history,
                llm_response_agent0.communication
            )
        
        # Process agent_1 with updated history (with previous validation errors if any)
        obs_text_agent1 = encode_state_for_agent(state, 1, self.env)
        
        prompt_agent1 = format_agent_prompt(
            obs_text_agent1, 
            1, 
            updated_history,
            previous_validation_errors=self.previous_validation_errors["agent_1"]
        )
        response_text_agent1 = self.llm_generate_fn(prompt_agent1)
        llm_response_agent1 = parse_llm_response(response_text_agent1, "agent_1", "agent_0")
        
        # Update conversation history with agent_1's message
        if llm_response_agent1.communication:
            updated_history = update_conversation_history(
                updated_history,
                llm_response_agent1.communication
            )
        
        # Execute plans to get actions and validate them
        actions = {}
        validation_errors = {f"agent_{i}": [] for i in range(self.env.num_agents)}
        # Track plan validation and execution success for rewards
        plan_validation_status = {f"agent_{i}": False for i in range(self.env.num_agents)}
        plan_execution_status = {f"agent_{i}": False for i in range(self.env.num_agents)}
        
        # Check if we have remaining actions in buffer
        for agent_id in range(self.env.num_agents):
            agent_key = f"agent_{agent_id}"
            llm_response = llm_response_agent0 if agent_id == 0 else llm_response_agent1
            obs_text = obs_text_agent0 if agent_id == 0 else obs_text_agent1
            prompt = prompt_agent0 if agent_id == 0 else prompt_agent1
            
            # If buffer is empty or plan changed, generate new actions
            if (not self.action_buffers[agent_key] or 
                self.current_plans[agent_key] != llm_response.plan):
                
                # Get plan for this agent
                plan = llm_response.plan
                
                # Validate plan before execution
                agent_pos = (int(state.agents.pos.x[agent_id]), int(state.agents.pos.y[agent_id]))
                is_valid, error_msg = validate_plan(plan, self.env, agent_pos)
                validation_result = (is_valid, error_msg)
                
                # Track validation status
                plan_validation_status[agent_key] = is_valid
                
                if not is_valid:
                    # Store validation error for feedback in next step
                    validation_errors[agent_key].append(error_msg)
                    
                    # Add system message to conversation history for immediate feedback
                    system_msg = CommunicationMessage(
                        from_agent="system",
                        to_agent=agent_key,
                        message=f"Your plan was invalid: {error_msg}. Please provide a valid plan with all required fields in your next response."
                    )
                    updated_history.append(system_msg)
                    
                    # Use wait action for invalid plans
                    plan = None
                    executor_actions = [Actions.stay]
                    executor_info = {
                        "error": f"Plan validation failed: {error_msg}",
                        "plan_action": None,
                    }
                    plan_execution_status[agent_key] = False
                else:
                    # Clear validation errors if plan is valid
                    validation_errors[agent_key] = []
                    
                    # Generate action sequence
                    try:
                        executor_actions = self.executor.execute_plan(plan, state, agent_id)
                        # Plan executed successfully if we got non-empty action sequence
                        plan_execution_status[agent_key] = (
                            len(executor_actions) > 0 and 
                            executor_actions != [Actions.stay]
                        )
                        executor_info = {
                            "plan_action": plan.action,
                            "target_found": plan.target_location is not None if plan else None,
                            "path_length": len(executor_actions),
                        }
                    except Exception as e:
                        executor_actions = [Actions.stay]
                        executor_info = {
                            "error": f"Executor error: {str(e)}",
                            "plan_action": plan.action if plan else None,
                        }
                        plan_execution_status[agent_key] = False
                
                self.current_plans[agent_key] = plan
                self.action_buffers[agent_key] = executor_actions
                
                # Display agent context with rich
                display_agent_context(
                    agent_id=agent_id,
                    step=step,
                    obs_text=obs_text,
                    prompt=prompt,
                    raw_response=llm_response.raw_response,
                    parsed_response=llm_response,
                    plan=plan,
                    validation_result=validation_result,
                    executor_actions=executor_actions,
                    executor_info=executor_info,
                    previous_validation_errors=self.previous_validation_errors[agent_key],
                    env=self.env,
                )
            else:
                # Plan hasn't changed, use existing buffer
                # If we're using buffer, it means plan was already validated and executed
                executor_actions = self.action_buffers[agent_key]
                executor_info = {
                    "plan_action": self.current_plans[agent_key].action if self.current_plans[agent_key] else None,
                    "using_buffer": True,
                    "buffer_length": len(executor_actions),
                }
                # Keep previous validation/execution status when using buffer
                plan_validation_status[agent_key] = self.current_plans[agent_key] is not None
                plan_execution_status[agent_key] = (
                    self.current_plans[agent_key] is not None and
                    len(executor_actions) > 0
                )
            
            # Get next action from buffer
            if self.action_buffers[agent_key]:
                actions[agent_key] = int(self.action_buffers[agent_key][0])
                self.action_buffers[agent_key] = self.action_buffers[agent_key][1:]
            else:
                actions[agent_key] = int(Actions.stay)
        
        # Store validation errors for next step
        self.previous_validation_errors = validation_errors
        
        llm_responses = {
            "agent_0": llm_response_agent0,
            "agent_1": llm_response_agent1,
        }
        
        # Return plan execution status for reward computation
        plan_status = {
            "plan_valid_agent0": plan_validation_status["agent_0"],
            "plan_executed_agent0": plan_execution_status["agent_0"],
            "plan_valid_agent1": plan_validation_status["agent_1"],
            "plan_executed_agent1": plan_execution_status["agent_1"],
        }
        
        return actions, updated_history, llm_responses, plan_status
    
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
        
        # Initialize conversation history and action buffers
        conversation_history = []
        self.action_buffers = {f"agent_{i}": [] for i in range(self.env.num_agents)}
        self.current_plans = {f"agent_{i}": None for i in range(self.env.num_agents)}
        self.previous_validation_errors = {f"agent_{i}": [] for i in range(self.env.num_agents)}
        
        steps = []
        episode_reward = 0.0
        num_deliveries = 0
        
        for step in range(max_steps):
            console.print(f"\n[bold bright_white]{'='*80}[/bold bright_white]")
            console.print(f"[bold bright_white]Step {step}[/bold bright_white]\n")
            
            # Process with LLM
            actions, conversation_history, llm_responses, plan_status = self.step_with_llm(
                obs, state, conversation_history, step=step
            )
            
            # Step environment
            key, step_key = jax.random.split(key)
            step_keys = jax.random.split(step_key, 1)  # For single env
            obs_new, state_new, rewards, dones, infos = self.env.step(
                step_keys[0], state, actions
            )
            
            # Track rewards and deliveries
            step_reward = sum(rewards.values())
            episode_reward += step_reward
            
            # Display step summary with plan rewards
            display_step_summary(step, actions, rewards, episode_reward, plan_status, llm_responses)
            
            if state_new.new_correct_delivery:
                num_deliveries += 1
                console.print(f"[bold green]✓ Correct delivery! Total deliveries: {num_deliveries}[/bold green]")
            
            console.print()
            
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
                plan_valid_agent0=plan_status["plan_valid_agent0"],
                plan_executed_agent0=plan_status["plan_executed_agent0"],
                plan_valid_agent1=plan_status["plan_valid_agent1"],
                plan_executed_agent1=plan_status["plan_executed_agent1"],
            )
            steps.append(step_data)
            
            # Check if done
            if dones.get("__all__", False) or state_new.terminal:
                console.print(f"[bold yellow]Episode terminated at step {step + 1}[/bold yellow]")
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
        
        console.print(f"\n[bold green]Episode completed:[/bold green] {len(steps)} steps, "
                     f"{len(conversation_history)} messages, {num_deliveries} deliveries, "
                     f"reward={episode_reward:.2f}\n")
        
        return trajectory
    
    def reset_action_buffers(self):
        """Reset action buffers (call after episode ends)."""
        self.action_buffers = {f"agent_{i}": [] for i in range(self.env.num_agents)}
        self.current_plans = {f"agent_{i}": None for i in range(self.env.num_agents)}
        self.previous_validation_errors = {f"agent_{i}": [] for i in range(self.env.num_agents)}

