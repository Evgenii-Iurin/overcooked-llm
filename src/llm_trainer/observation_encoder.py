"""Observation encoder for converting OvercookedV2 observations to text descriptions."""

import jax.numpy as jnp
import numpy as np
from typing import Dict, Tuple
from jaxmarl.environments.overcooked_v2.common import (
    StaticObject,
    DynamicObject,
    MAX_INGREDIENTS,
)


def encode_recipe(recipe_encoding: int, num_ingredients: int) -> str:
    """Decode recipe encoding to ingredient list description.
    
    Args:
        recipe_encoding: Encoded recipe integer
        num_ingredients: Number of ingredients in the environment
        
    Returns:
        String description of recipe, e.g., "ingredient0, ingredient0, ingredient1"
    """
    ingredient_list = DynamicObject.get_ingredient_idx_list_jit(recipe_encoding)
    ingredients = [int(i) for i in ingredient_list if i != -1]
    
    if not ingredients:
        return "no ingredients specified"
    
    # Count occurrences
    ingredient_counts = {}
    for ing in ingredients:
        ingredient_counts[ing] = ingredient_counts.get(ing, 0) + 1
    
    # Format as readable string
    parts = []
    for ing_idx in sorted(ingredient_counts.keys()):
        count = ingredient_counts[ing_idx]
        if count == 1:
            parts.append(f"ingredient{ing_idx}")
        else:
            parts.append(f"{count}x ingredient{ing_idx}")
    
    return ", ".join(parts)


def encode_inventory(inventory: int) -> str:
    """Encode agent inventory to text description.
    
    Args:
        inventory: Inventory encoding from environment
        
    Returns:
        String description of what agent is holding
    """
    if inventory == 0:
        return "nothing"
    
    parts = []
    
    # Check for plate
    if inventory & DynamicObject.PLATE:
        parts.append("a plate")
    
    # Check for cooked status
    is_cooked = (inventory & DynamicObject.COOKED) != 0
    
    # Check for ingredients
    if DynamicObject.is_ingredient(inventory):
        ingredient_list = DynamicObject.get_ingredient_idx_list_jit(inventory)
        ingredients = [int(i) for i in ingredient_list if i != -1]
        if ingredients:
            ing_counts = {}
            for ing in ingredients:
                ing_counts[ing] = ing_counts.get(ing, 0) + 1
            ing_parts = []
            for ing_idx in sorted(ing_counts.keys()):
                count = ing_counts[ing_idx]
                if count == 1:
                    ing_parts.append(f"ingredient{ing_idx}")
                else:
                    ing_parts.append(f"{count}x ingredient{ing_idx}")
            ing_str = ", ".join(ing_parts)
            if is_cooked:
                parts.append(f"cooked dish with {ing_str}")
            else:
                parts.append(ing_str)
    
    if not parts:
        return "unknown item"
    
    return " and ".join(parts)


def encode_observation(
    obs: jnp.ndarray,
    agent_pos: Tuple[int, int],
    agent_dir: int,
    inventory: int,
    recipe_encoding: int,
    num_ingredients: int,
    agent_view_size: int = None,
) -> str:
    """Convert grid observation to natural language description.
    
    Args:
        obs: Grid observation (height x width x channels) or (view_size x view_size x channels)
        agent_pos: Agent position (x, y)
        agent_dir: Agent direction (0=UP, 1=DOWN, 2=RIGHT, 3=LEFT)
        inventory: Agent inventory encoding
        recipe_encoding: Current recipe encoding
        num_ingredients: Number of ingredients in environment
        agent_view_size: View size if using partial observability, None for full
        
    Returns:
        Text description of the observation
    """
    obs_np = np.array(obs)
    height, width = obs_np.shape[:2]
    
    # Direction names
    dir_names = ["up", "down", "right", "left"]
    direction = dir_names[agent_dir] if agent_dir < len(dir_names) else f"direction_{agent_dir}"
    
    # Inventory description
    inventory_desc = encode_inventory(int(inventory))
    
    # Recipe description
    recipe_desc = encode_recipe(int(recipe_encoding), num_ingredients)
    
    # Parse grid channels
    static_channel = obs_np[:, :, 0]
    dynamic_channel = obs_np[:, :, 1]
    extra_channel = obs_np[:, :, 2] if obs_np.shape[2] > 2 else np.zeros_like(static_channel)
    
    # Find objects in the observation
    objects = []
    
    # Find pots
    pot_positions = []
    for y in range(height):
        for x in range(width):
            if static_channel[y, x] == StaticObject.POT:
                pot_positions.append((x, y))
                # Check pot state
                pot_content = dynamic_channel[y, x]
                pot_timer = int(extra_channel[y, x]) if extra_channel is not None else 0
                
                if pot_content == 0:
                    pot_state = "empty"
                elif pot_content & DynamicObject.COOKED:
                    pot_state = "cooked and ready"
                elif pot_timer > 0:
                    pot_state = f"cooking (timer: {pot_timer})"
                else:
                    ing_count = DynamicObject.ingredient_count(pot_content)
                    pot_state = f"has {ing_count} ingredients"
                
                objects.append(f"pot at ({x}, {y}) - {pot_state}")
    
    # Find ingredient piles
    ingredient_piles = []
    for y in range(height):
        for x in range(width):
            if StaticObject.is_ingredient_pile(static_channel[y, x]):
                ing_idx = static_channel[y, x] - StaticObject.INGREDIENT_PILE_BASE
                ingredient_piles.append((ing_idx, x, y))
                objects.append(f"ingredient{ing_idx} pile at ({x}, {y})")
    
    # Find plate pile
    plate_piles = []
    for y in range(height):
        for x in range(width):
            if static_channel[y, x] == StaticObject.PLATE_PILE:
                plate_piles.append((x, y))
                objects.append(f"plate pile at ({x}, {y})")
    
    # Find goal
    goals = []
    for y in range(height):
        for x in range(width):
            if static_channel[y, x] == StaticObject.GOAL:
                goals.append((x, y))
                objects.append(f"goal (delivery location) at ({x}, {y})")
    
    # Find recipe indicator
    recipe_indicators = []
    for y in range(height):
        for x in range(width):
            if static_channel[y, x] == StaticObject.RECIPE_INDICATOR:
                recipe_indicators.append((x, y))
                objects.append(f"recipe indicator at ({x}, {y}) showing recipe: {recipe_desc}")
            elif static_channel[y, x] == StaticObject.BUTTON_RECIPE_INDICATOR:
                recipe_indicators.append((x, y))
                is_active = extra_channel[y, x] > 0 if extra_channel is not None else False
                status = "active" if is_active else "inactive"
                objects.append(f"button recipe indicator at ({x}, {y}) - {status}")
    
    # Build description
    parts = []
    
    # Agent state
    if agent_view_size is None:
        parts.append(f"Agent at position ({agent_pos[0]}, {agent_pos[1]}), facing {direction}")
    else:
        parts.append(f"Agent at center of view (view radius: {agent_view_size}), facing {direction}")
    
    parts.append(f"Holding: {inventory_desc}")
    parts.append(f"Current recipe requires: {recipe_desc}")
    
    # Visible objects
    if objects:
        parts.append("Visible objects:")
        for obj in objects:
            parts.append(f"  - {obj}")
    else:
        parts.append("No visible objects (only walls or empty space)")
    
    # Check for other agents (if visible)
    other_agents = []
    for y in range(height):
        for x in range(width):
            if static_channel[y, x] == StaticObject.AGENT:
                other_agents.append((x, y))
    
    if other_agents:
        parts.append(f"Other agent(s) visible at: {', '.join([f'({x}, {y})' for x, y in other_agents])}")
    
    return "\n".join(parts)


def encode_state_for_agent(
    state,
    agent_id: int,
    env,
    agent_view_size: int = None,
) -> str:
    """Encode full state information for a specific agent.
    
    Args:
        state: OvercookedV2 State object
        agent_id: Agent index (0 or 1)
        env: OvercookedV2 environment instance
        agent_view_size: View size if using partial observability
        
    Returns:
        Text description of agent's observation
    """
    # Get agent's observation
    obs_dict = env.get_obs(state)
    obs = obs_dict[f"agent_{agent_id}"]
    
    # Get agent state
    agent_pos = (int(state.agents.pos.x[agent_id]), int(state.agents.pos.y[agent_id]))
    agent_dir = int(state.agents.dir[agent_id])
    inventory = int(state.agents.inventory[agent_id])
    
    # Get recipe
    recipe_encoding = int(state.recipe)
    num_ingredients = env.layout.num_ingredients
    
    # Encode observation
    return encode_observation(
        obs=obs,
        agent_pos=agent_pos,
        agent_dir=agent_dir,
        inventory=inventory,
        recipe_encoding=recipe_encoding,
        num_ingredients=num_ingredients,
        agent_view_size=agent_view_size or env.agent_view_size,
    )

