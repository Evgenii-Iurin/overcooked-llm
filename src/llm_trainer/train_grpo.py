"""Main training script for GRPO on Overcooked environment."""

import jax
import jax.numpy as jnp
from typing import Callable, Optional
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from jaxmarl import make
from loguru import logger

from .dataset_generator import generate_grpo_dataset
from .reward_functions import format_reward_func
from .overcooked_llm_wrapper import OvercookedLLMWrapper
from jaxmarl.environments.overcooked_v2.common import StaticObject, DynamicObject


def print_environment_grid(env, state):
    """Print the environment grid with symbols.
    
    Args:
        env: OvercookedV2 environment
        state: Current state
    """
    import numpy as np
    
    height, width = env.height, env.width
    grid = np.array(state.grid)
    static_objects = grid[:, :, 0]
    dynamic_objects = grid[:, :, 1]
    
    # Create symbol grid
    symbols = [[' ' for _ in range(width)] for _ in range(height)]
    
    # Map static objects to symbols
    for y in range(height):
        for x in range(width):
            static_obj = int(static_objects[y, x])
            
            if static_obj == StaticObject.WALL:
                symbols[y][x] = 'W'
            elif static_obj == StaticObject.GOAL:
                symbols[y][x] = 'X'
            elif static_obj == StaticObject.POT:
                # Check if pot has contents
                pot_content = int(dynamic_objects[y, x])
                if pot_content & DynamicObject.COOKED:
                    symbols[y][x] = 'P'  # Cooked pot
                elif pot_content != 0:
                    symbols[y][x] = 'p'  # Pot with ingredients
                else:
                    symbols[y][x] = 'P'  # Empty pot
            elif static_obj == StaticObject.PLATE_PILE:
                symbols[y][x] = 'B'
            elif static_obj == StaticObject.RECIPE_INDICATOR:
                symbols[y][x] = 'R'
            elif static_obj == StaticObject.BUTTON_RECIPE_INDICATOR:
                symbols[y][x] = 'L'
            elif StaticObject.is_ingredient_pile(static_obj):
                ing_idx = static_obj - StaticObject.INGREDIENT_PILE_BASE
                symbols[y][x] = str(ing_idx)
            elif static_obj == StaticObject.EMPTY:
                # Check for dynamic objects on empty cells
                dyn_obj = int(dynamic_objects[y, x])
                if dyn_obj != 0:
                    if dyn_obj == DynamicObject.PLATE:
                        symbols[y][x] = 'b'  # Plate on counter
                    elif dyn_obj & DynamicObject.COOKED:
                        symbols[y][x] = 'D'  # Cooked dish
                    else:
                        # Check if it's an ingredient (using bit manipulation)
                        if (dyn_obj >> 2) != 0 and (dyn_obj & DynamicObject.PLATE) == 0:
                            symbols[y][x] = 'i'  # Ingredient on counter
                        else:
                            symbols[y][x] = ' '  # Unknown dynamic object
                else:
                    symbols[y][x] = ' '
    
    # Place agents
    for agent_id in range(env.num_agents):
        agent_x = int(state.agents.pos.x[agent_id])
        agent_y = int(state.agents.pos.y[agent_id])
        if 0 <= agent_x < width and 0 <= agent_y < height:
            # Use A for agent 0, a for agent 1, or numbers
            if agent_id == 0:
                symbols[agent_y][agent_x] = 'A'
            else:
                symbols[agent_y][agent_x] = 'a'
    
    # Print grid
    logger.info("Environment Grid:")
    logger.info("=" * (width + 2))
    for row in symbols:
        logger.info("|" + "".join(row) + "|")
    logger.info("=" * (width + 2))
    logger.info("Legend: W=Wall, X=Goal, P=Pot, p=Pot(cooking), B=Plate pile, b=Plate, "
                "0-9=Ingredient piles, i=Ingredient, D=Dish, R=Recipe indicator, "
                "L=Button indicator, A/a=Agents, ' '=Empty")


def create_llm_generate_fn(model, tokenizer):
    """Create LLM generation function compatible with wrapper.
    
    Args:
        model: HuggingFace model object (must be loaded, not a string)
        tokenizer: Tokenizer instance
        
    Returns:
        Function that takes prompt and returns completion string
    """
    # Use transformers pipeline for inference
    from transformers import pipeline
    
    # Disable gradient checkpointing for inference to avoid warnings
    # The model may have gradient checkpointing enabled from training (via GRPOTrainer),
    # but it's not needed during inference and causes "Gradients will be None" warnings
    # when called with torch.no_grad() or without requires_grad=True inputs.
    # Note: GRPOTrainer will re-enable it automatically when training starts.
    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()
    
    # Set model to eval mode for inference
    model.eval()
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
    )
    
    def generate_fn(prompt):
        # Ensure model is in eval mode and checkpointing is disabled during generation
        # (in case it was re-enabled by trainer between calls)
        model.eval()
        if hasattr(model, 'gradient_checkpointing_disable'):
            model.gradient_checkpointing_disable()
        
        messages = prompt
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        outputs = pipe(
            prompt_text,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
        )
        return outputs[0]["generated_text"][len(prompt_text):].strip()
    
    return generate_fn


def train_grpo_overcooked(
    model_name: str,
    env_name: str = "overcooked_v2",
    env_kwargs: Optional[dict] = None,
    num_episodes_per_iteration: int = 10,
    max_steps: int = 400,
    num_iterations: int = 100,
    output_dir: str = "outputs/overcooked_grpo",
    learning_rate: float = 5e-6,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    num_generations: int = 2,
    max_prompt_length: int = 512,
    max_completion_length: int = 256,
    rng_key: Optional[jax.random.PRNGKey] = None,
    use_mlflow: bool = False,
    save_checkpoint_every_n_episodes: Optional[int] = None,
    verbose: bool = False,
):
    """Train LLM using GRPO on Overcooked environment.
    
    Args:
        model_name: HuggingFace model name or path
        env_name: Environment name (default: "overcooked_v2")
        env_kwargs: Environment configuration
        num_episodes_per_iteration: Episodes to collect per training iteration
        max_steps: Maximum steps per episode
        num_iterations: Number of training iterations
        output_dir: Output directory for checkpoints, models, and all artifacts.
        learning_rate: Learning rate
        per_device_train_batch_size: Batch size per device
        gradient_accumulation_steps: Gradient accumulation steps
        num_generations: Number of generations per prompt
        max_prompt_length: Maximum prompt length
        max_completion_length: Maximum completion length
        rng_key: Random key for environment
        use_mlflow: Whether to log metrics to MLflow. If True, assumes MLflow is already initialized.
        save_checkpoint_every_n_episodes: Optional. If set, save trajectory checkpoint every N episodes
            during generation (e.g., 5 means save after episodes 5, 10, 15, ...). 
            Useful for long-running generation to prevent data loss on crashes.
        verbose: Whether to display rich output during generation (default: False for faster generation).
            Set to True only for debugging - significantly slows down execution.
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(42)
    
    if env_kwargs is None:
        env_kwargs = {
            "layout": "cramped_room",
            "max_steps": max_steps,
            "agent_view_size": None,  # Full observability for now
        }
    
    # Initialize environment
    logger.info(f"Initializing environment: {env_name}")
    env = make(env_name, **env_kwargs)
    logger.info(f"Environment created: {env_name}")
    logger.info(f"Number of agents: {env.num_agents}")
    logger.info(f"Grid size: {env.width}x{env.height}")
    logger.info(f"Max steps per episode: {max_steps}")
    
    # Print initial environment state (use separate key for visualization)
    viz_key = jax.random.PRNGKey(999)  # Fixed key for visualization
    obs, state = env.reset(viz_key)
    print_environment_grid(env, state)
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Left-padding for decoder-only models: correct generation when batching
    tokenizer.padding_side = "left"
    logger.info("Tokenizer loaded successfully")
    
    # Check if CUDA is available
    import torch
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    
    # GRPO configuration
    training_args = GRPOConfig(
        learning_rate=learning_rate,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,
        bf16=use_cuda and torch.cuda.is_bf16_supported(),  # Only use bf16 if GPU supports it
        fp16=use_cuda and not torch.cuda.is_bf16_supported(),  # Fallback to fp16 if bf16 not supported
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_generations=num_generations,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        num_train_epochs=1,
        max_steps=1,  # Required when dataloader has no length (e.g. tiny sanity-check datasets). Upper bound; training stops when epoch ends or this is reached.
        save_steps=100,
        max_grad_norm=0.1,
        report_to="tensorboard",
        output_dir=output_dir,
    )
    
    # Load model once - will be used for both inference and training
    from transformers import AutoModelForCausalLM
    
    if use_cuda:
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        # Force accelerate to use GPU
        import os
        os.environ["ACCELERATE_USE_CPU"] = "false"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        
        logger.info("Loading model on GPU...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cuda:0",  # Explicitly place on GPU 0
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            low_cpu_mem_usage=True,
        )
        logger.info(f"Model loaded. Device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'N/A'}")
        # Verify first parameter is on GPU
        first_param = next(model.parameters(), None)
        if first_param is not None:
            logger.info(f"Model parameters on device: {first_param.device}")
    else:
        logger.warning("CUDA not available, using CPU (training will be slow)")
        logger.info("Loading model on CPU...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        logger.info("Model loaded on CPU")
    
    # Create LLM generation function using the loaded model
    logger.info("Creating inference function...")
    llm_generate_fn = create_llm_generate_fn(model, tokenizer)
    
    # Initialize trainer with the same model instance
    logger.info("Initializing GRPOTrainer...")
    trainer = GRPOTrainer(
        model=model,  # Use the same model instance for training
        processing_class=tokenizer,
        reward_funcs=[format_reward_func],  # Format reward for proper XML structure
        args=training_args,
        train_dataset=None,  # Will be generated iteratively
    )
    
    # Verify device placement
    if use_cuda:
        # Check if model is on GPU
        if hasattr(trainer, 'model') and trainer.model is not None:
            # Try to get device info from model
            try:
                if hasattr(trainer.model, 'device'):
                    logger.info(f"Model device: {trainer.model.device}")
                elif hasattr(trainer.model, 'hf_device_map'):
                    logger.info(f"Model device map: {trainer.model.hf_device_map}")
                else:
                    # Try to check first parameter's device
                    first_param = next(trainer.model.parameters(), None)
                    if first_param is not None:
                        logger.info(f"Model parameters on device: {first_param.device}")
            except Exception as e:
                logger.warning(f"Could not determine model device: {e}")
    
    # Training loop
    logger.info("=" * 60)
    logger.info("Starting GRPO training")
    logger.info("=" * 60)
    logger.info(f"Total iterations: {num_iterations}")
    logger.info(f"Episodes per iteration: {num_episodes_per_iteration}")
    logger.info(f"Max steps per episode: {max_steps}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Artifacts (trajectories, datasets) will be saved to: {output_dir}/artifacts/")
    
    # Initialize MLflow if requested
    if use_mlflow:
        try:
            import mlflow
            logger.info("MLflow logging enabled")
            # Log hyperparameters at the start
            mlflow.log_params({
                "model_name": model_name,
                "env_name": env_name,
                "layout": env_kwargs.get("layout", "unknown") if env_kwargs else "unknown",
                "num_episodes_per_iteration": num_episodes_per_iteration,
                "max_steps": max_steps,
                "num_iterations": num_iterations,
                "learning_rate": learning_rate,
                "per_device_train_batch_size": per_device_train_batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "num_generations": num_generations,
                "max_prompt_length": max_prompt_length,
                "max_completion_length": max_completion_length,
            })
        except ImportError:
            logger.warning("MLflow requested but not installed. Continuing without MLflow logging.")
            use_mlflow = False
    else:
        mlflow = None
    
    for iteration in range(num_iterations):
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"Iteration {iteration + 1}/{num_iterations}")
        logger.info("=" * 60)
        
        # Generate trajectories with current model
        logger.info(f"Generating {num_episodes_per_iteration} episodes...")
        rng_key, data_key = jax.random.split(rng_key)
        
        dataset, trajectories = generate_grpo_dataset(
            env=env,
            llm_generate_fn=llm_generate_fn,
            num_episodes=num_episodes_per_iteration,
            max_steps=max_steps,
            reward_strategy="episode_shared",
            rng_key=data_key,
            save_artifacts=True,
            output_dir=output_dir,
            iteration=iteration,
            use_mlflow=use_mlflow,
            save_checkpoint_every_n_episodes=save_checkpoint_every_n_episodes,
            verbose=verbose,
        )
        
        # Calculate statistics
        avg_reward = sum(dataset['reward']) / len(dataset) if len(dataset) > 0 else 0.0
        max_reward = max(dataset['reward']) if len(dataset) > 0 else 0.0
        min_reward = min(dataset['reward']) if len(dataset) > 0 else 0.0
        std_reward = (sum((r - avg_reward) ** 2 for r in dataset['reward']) / len(dataset)) ** 0.5 if len(dataset) > 0 else 0.0
        
        # Calculate episode statistics from trajectories
        avg_episode_length = sum(t.episode_length for t in trajectories) / len(trajectories) if trajectories else 0.0
        avg_episode_reward = sum(t.episode_reward for t in trajectories) / len(trajectories) if trajectories else 0.0
        total_deliveries = sum(t.num_deliveries for t in trajectories) if trajectories else 0
        avg_deliveries = total_deliveries / len(trajectories) if trajectories else 0.0
        
        logger.success(f"Dataset generated: {len(dataset)} examples")
        logger.info(f"Reward statistics - Avg: {avg_reward:.2f}, Max: {max_reward:.2f}, Min: {min_reward:.2f}, Std: {std_reward:.2f}")
        logger.info(f"Episode statistics - Avg length: {avg_episode_length:.1f}, Avg reward: {avg_episode_reward:.2f}, Total deliveries: {total_deliveries}, Avg deliveries: {avg_deliveries:.2f}")
        
        # Log metrics to MLflow
        if use_mlflow:
            try:
                mlflow.log_metrics({
                    "iteration": iteration + 1,
                    "dataset_size": len(dataset),
                    "reward/avg": avg_reward,
                    "reward/max": max_reward,
                    "reward/min": min_reward,
                    "reward/std": std_reward,
                    "episode/length_avg": avg_episode_length,
                    "episode/reward_avg": avg_episode_reward,
                    "episode/deliveries_total": total_deliveries,
                    "episode/deliveries_avg": avg_deliveries,
                }, step=iteration + 1)
                
                logger.debug(f"Metrics logged to MLflow for iteration {iteration + 1}")
            except Exception as e:
                logger.warning(f"Failed to log metrics to MLflow: {e}")
        
        # Update trainer dataset
        trainer.train_dataset = dataset
        
        # Train for one epoch
        logger.info("Starting training step...")
        trainer.train()
        logger.success("Training step completed")
        
        # Update LLM generation function with updated model from trainer
        # The trainer's model is the same instance, but weights have been updated
        logger.debug("Updating LLM generation function with trained model")
        llm_generate_fn = create_llm_generate_fn(trainer.model, tokenizer)
        
        logger.success(f"Iteration {iteration + 1}/{num_iterations} complete")
    
    logger.info("")
    logger.success("=" * 60)
    logger.success("Training complete!")
    logger.success(f"Model saved to: {output_dir}")
    logger.success("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train LLM with GRPO on Overcooked")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                       help="Model name or path")
    parser.add_argument("--env", type=str, default="overcooked_v2",
                       help="Environment name")
    parser.add_argument("--layout", type=str, default="cramped_room",
                       help="Layout name")
    parser.add_argument("--num_episodes", type=int, default=10,
                       help="Episodes per iteration")
    parser.add_argument("--max_steps", type=int, default=400,
                       help="Maximum steps per episode. Use 1 for a quick training sanity check.")
    parser.add_argument("--num_iterations", type=int, default=100,
                       help="Number of training iterations")
    parser.add_argument("--output_dir", type=str, default="outputs/overcooked_grpo",
                       help="Output directory")
    parser.add_argument("--use_mlflow", action="store_true",
                       help="Log metrics to MLflow (assumes MLflow is already initialized)")
    parser.add_argument("--save_checkpoint_every_n_episodes", type=int, default=None,
                       help="Save trajectory checkpoint every N episodes during generation (e.g., 5). Useful for long-running generation.")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable rich display output (slows down execution significantly - use only for debugging)")
    
    args = parser.parse_args()
    
    train_grpo_overcooked(
        model_name=args.model,
        env_name=args.env,
        env_kwargs={"layout": args.layout, "max_steps": args.max_steps},
        num_episodes_per_iteration=args.num_episodes,
        max_steps=args.max_steps,
        num_iterations=args.num_iterations,
        output_dir=args.output_dir,
        use_mlflow=args.use_mlflow,
        save_checkpoint_every_n_episodes=args.save_checkpoint_every_n_episodes,
        verbose=args.verbose,
    )

