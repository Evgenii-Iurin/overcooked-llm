"""Action executor that translates high-level plans to discrete environment actions."""

import numpy as np
import jax.numpy as jnp
from typing import List, Tuple, Optional, Dict
from collections import deque
from jaxmarl.environments.overcooked_v2.common import Actions, Direction, StaticObject, DynamicObject
from jaxmarl.environments.overcooked_v2.overcooked import State
from .communication import ActionPlan


class ActionExecutor:
    """Executes high-level action plans by generating sequences of discrete actions."""
    
    def __init__(self, env):
        """Initialize executor with environment reference.
        
        Args:
            env: OvercookedV2 environment instance
        """
        self.env = env
        self.width = env.width
        self.height = env.height
        
        # Precompute move area (non-wall cells)
        if hasattr(env, 'layout'):
            self.move_area = (env.layout.static_objects == StaticObject.EMPTY).astype(bool)
        else:
            self.move_area = np.ones((env.height, env.width), dtype=bool)
    
    def find_path(
        self,
        start: Tuple[int, int],
        target: Tuple[int, int],
        max_steps: int = 100,
    ) -> List[int]:
        """Find path from start to target using BFS.
        
        Args:
            start: Starting position (x, y)
            target: Target position (x, y)
            max_steps: Maximum path length
            
        Returns:
            List of action indices (Actions enum values) to reach target
        """
        if start == target:
            return []
        
        # BFS to find path
        queue = deque([(start[0], start[1], [])])
        visited = {start}
        
        # Direction vectors: right, down, left, up
        dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        action_map = [Actions.right, Actions.down, Actions.left, Actions.up]
        
        while queue and len(queue[0][2]) < max_steps:
            x, y, path = queue.popleft()
            
            for i, (dx, dy) in enumerate(dirs):
                nx, ny = x + dx, y + dy
                
                if (nx, ny) == target:
                    return path + [action_map[i]]
                
                if (0 <= nx < self.width and 0 <= ny < self.height and
                    (nx, ny) not in visited and
                    self.move_area[ny, nx]):
                    visited.add((nx, ny))
                    queue.append((nx, ny, path + [action_map[i]]))
        
        # If no path found, return empty (will stay in place)
        return []
    
    def get_adjacent_position(
        self,
        pos: Tuple[int, int],
        direction: int,
    ) -> Tuple[int, int]:
        """Get position adjacent to current position in given direction.
        
        Args:
            pos: Current position (x, y)
            direction: Direction (0=right, 1=down, 2=left, 3=up)
            
        Returns:
            Adjacent position (x, y)
        """
        dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        dx, dy = dirs[direction]
        return (pos[0] + dx, pos[1] + dy)
    
    def find_nearest_object(
        self,
        pos: Tuple[int, int],
        obj_type: int,
        state: State,
    ) -> Optional[Tuple[int, int]]:
        """Find nearest object of given type from position.
        
        Args:
            pos: Starting position (x, y)
            obj_type: StaticObject type to find
            state: Current environment state
            
        Returns:
            Position of nearest object, or None if not found
        """
        grid = np.array(state.grid[:, :, 0])
        min_dist = float('inf')
        nearest = None
        
        for y in range(self.height):
            for x in range(self.width):
                if grid[y, x] == obj_type:
                    dist = abs(x - pos[0]) + abs(y - pos[1])
                    if dist < min_dist:
                        min_dist = dist
                        nearest = (x, y)
        
        return nearest
    
    def find_nearest_ingredient_pile(
        self,
        pos: Tuple[int, int],
        ingredient_idx: int,
        state: State,
    ) -> Optional[Tuple[int, int]]:
        """Find nearest ingredient pile of specific type.
        
        Args:
            pos: Starting position (x, y)
            ingredient_idx: Ingredient index to find
            state: Current environment state
            
        Returns:
            Position of nearest ingredient pile, or None if not found
        """
        grid = np.array(state.grid[:, :, 0])
        target_obj = StaticObject.INGREDIENT_PILE_BASE + ingredient_idx
        return self.find_nearest_object(pos, target_obj, state)
    
    def execute_plan(
        self,
        plan: ActionPlan,
        state: State,
        agent_id: int,
        max_action_sequence: int = 50,
    ) -> List[int]:
        """Execute a high-level plan and return sequence of discrete actions.
        
        Args:
            plan: Action plan to execute
            state: Current environment state
            agent_id: Agent ID (0 or 1)
            max_action_sequence: Maximum number of actions to generate
            
        Returns:
            List of action indices (Actions enum values)
        """
        if plan is None:
            return [Actions.stay]
        
        agent_pos = (int(state.agents.pos.x[agent_id]), int(state.agents.pos.y[agent_id]))
        agent_dir = int(state.agents.dir[agent_id])
        inventory = int(state.agents.inventory[agent_id])
        
        actions = []
        
        if plan.action == "wait":
            return [Actions.stay]
        
        elif plan.action == "move_to":
            if plan.target_location:
                path = self.find_path(agent_pos, plan.target_location, max_steps=max_action_sequence)
                actions.extend(path)
            else:
                actions.append(Actions.stay)
        
        elif plan.action == "pickup_ingredient":
            if plan.ingredients and len(plan.ingredients) > 0:
                # Find nearest ingredient pile of first needed ingredient
                ing_idx = plan.ingredients[0]
                target = self.find_nearest_ingredient_pile(agent_pos, ing_idx, state)
                
                if target:
                    # Move to ingredient pile
                    path = self.find_path(agent_pos, target, max_steps=max_action_sequence - 2)
                    actions.extend(path)
                    
                    # Face the pile and interact
                    if actions:
                        # Get direction to target
                        dx = target[0] - agent_pos[0]
                        dy = target[1] - agent_pos[1]
                        
                        # Simple facing logic - move adjacent then interact
                        if len(actions) < max_action_sequence - 1:
                            actions.append(Actions.interact)
                else:
                    # Ingredient not found, just wait
                    actions.append(Actions.stay)
            else:
                actions.append(Actions.stay)
        
        elif plan.action == "pickup_plate":
            # Find nearest plate pile
            target = self.find_nearest_object(agent_pos, StaticObject.PLATE_PILE, state)
            
            if target:
                path = self.find_path(agent_pos, target, max_steps=max_action_sequence - 2)
                actions.extend(path)
                
                if len(actions) < max_action_sequence - 1:
                    actions.append(Actions.interact)
            else:
                actions.append(Actions.stay)
        
        elif plan.action == "cook_dish":
            if plan.target_location:
                # Move to pot
                path = self.find_path(agent_pos, plan.target_location, max_steps=max_action_sequence - 10)
                actions.extend(path)
                
                # Add ingredients to pot (simplified - assumes agent has ingredients)
                # In reality, we'd need to check inventory and add ingredients one by one
                if len(actions) < max_action_sequence - 1:
                    # Face pot and interact to add ingredient
                    actions.append(Actions.interact)
            else:
                # Find nearest pot
                target = self.find_nearest_object(agent_pos, StaticObject.POT, state)
                if target:
                    path = self.find_path(agent_pos, target, max_steps=max_action_sequence - 2)
                    actions.extend(path)
                    if len(actions) < max_action_sequence - 1:
                        actions.append(Actions.interact)
                else:
                    actions.append(Actions.stay)
        
        elif plan.action == "deliver_dish":
            # Find nearest goal
            target = self.find_nearest_object(agent_pos, StaticObject.GOAL, state)
            
            if target:
                path = self.find_path(agent_pos, target, max_steps=max_action_sequence - 2)
                actions.extend(path)
                
                if len(actions) < max_action_sequence - 1:
                    actions.append(Actions.interact)
            else:
                actions.append(Actions.stay)
        
        else:
            # Unknown action, just wait
            actions.append(Actions.stay)
        
        # Limit action sequence length
        if len(actions) > max_action_sequence:
            actions = actions[:max_action_sequence]
        
        # If no actions generated, at least stay
        if not actions:
            actions = [Actions.stay]
        
        return actions
    
    def execute_plan_step(
        self,
        plan: ActionPlan,
        state: State,
        agent_id: int,
        current_action_buffer: List[int] = None,
    ) -> Tuple[int, List[int]]:
        """Execute one step of a plan, maintaining action buffer.
        
        This allows plans to be executed over multiple environment steps.
        
        Args:
            plan: Action plan to execute
            state: Current environment state
            agent_id: Agent ID (0 or 1)
            current_action_buffer: Remaining actions from previous call
            
        Returns:
            Tuple of (next_action, remaining_actions)
        """
        if current_action_buffer is None or len(current_action_buffer) == 0:
            # Generate new action sequence from plan
            current_action_buffer = self.execute_plan(plan, state, agent_id)
        
        if current_action_buffer:
            next_action = current_action_buffer[0]
            remaining = current_action_buffer[1:]
            return next_action, remaining
        else:
            return Actions.stay, []

