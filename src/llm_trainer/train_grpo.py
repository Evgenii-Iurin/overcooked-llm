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


def create_llm_generate_fn(model, tokenizer, vllm_engine=None):
    """Create LLM generation function compatible with wrapper.
    
    Args:
        model: HuggingFace model or model path
        tokenizer: Tokenizer instance
        vllm_engine: Optional vLLM engine for faster inference
        
    Returns:
        Function that takes prompt and returns completion string
    """
    if vllm_engine is not None:
        def generate_fn(prompt):
            # Use vLLM for generation
            messages = prompt
            # Convert to vLLM format
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            outputs = vllm_engine.generate([prompt_text], sampling_params={
                "temperature": 0.7,
                "max_tokens": 256,
            })
            return outputs[0].outputs[0].text
        return generate_fn
    else:
        # Use transformers pipeline
        from transformers import pipeline
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto",
        )
        
        def generate_fn(prompt):
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
    use_vllm: bool = True,
    learning_rate: float = 5e-6,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    num_generations: int = 2,
    max_prompt_length: int = 512,
    max_completion_length: int = 256,
    vllm_gpu_memory_utilization: float = 0.3,
    rng_key: Optional[jax.random.PRNGKey] = None,
):
    """Train LLM using GRPO on Overcooked environment.
    
    Args:
        model_name: HuggingFace model name or path
        env_name: Environment name (default: "overcooked_v2")
        env_kwargs: Environment configuration
        num_episodes_per_iteration: Episodes to collect per training iteration
        max_steps: Maximum steps per episode
        num_iterations: Number of training iterations
        output_dir: Output directory for checkpoints
        use_vllm: Whether to use vLLM for inference
        learning_rate: Learning rate
        per_device_train_batch_size: Batch size per device
        gradient_accumulation_steps: Gradient accumulation steps
        num_generations: Number of generations per prompt
        max_prompt_length: Maximum prompt length
        max_completion_length: Maximum completion length
        vllm_gpu_memory_utilization: vLLM GPU memory utilization
        rng_key: Random key for environment
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
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("Tokenizer loaded successfully")
    
    # Initialize vLLM engine if requested
    vllm_engine = None
    if use_vllm:
        try:
            logger.info("Initializing vLLM engine...")
            from vllm import LLM, SamplingParams
            vllm_engine = LLM(
                model=model_name,
                gpu_memory_utilization=vllm_gpu_memory_utilization,
            )
            logger.success("vLLM engine initialized successfully")
        except ImportError:
            logger.warning("vLLM not available, falling back to transformers")
            use_vllm = False
    
    # Create LLM generation function
    llm_generate_fn = create_llm_generate_fn(model_name, tokenizer, vllm_engine)
    
    # Check if CUDA is available
    import torch
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    
    # GRPO configuration
    training_args = GRPOConfig(
        use_vllm=use_vllm,
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
        save_steps=100,
        max_grad_norm=0.1,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        report_to="tensorboard",
        output_dir=output_dir,
    )
    
    if use_cuda:
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        # Force accelerate to use GPU
        import os
        os.environ["ACCELERATE_USE_CPU"] = "false"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        # Try to load model on GPU explicitly before passing to trainer
        from transformers import AutoModelForCausalLM
        logger.info("Loading model on GPU...")
        # Force GPU placement - use explicit device_map instead of "auto"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cuda:0",  # Explicitly place on GPU 0
            dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            low_cpu_mem_usage=True,
        )
        logger.info(f"Model loaded. Device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'N/A'}")
        # Verify first parameter is on GPU
        first_param = next(model.parameters(), None)
        if first_param is not None:
            logger.info(f"Model parameters on device: {first_param.device}")
    else:
        logger.warning("CUDA not available, using CPU (training will be slow)")
        model = model_name  # Pass string, let trainer load it
    
    # Initialize trainer
    trainer = GRPOTrainer(
        model=model if use_cuda else model_name,  # Pass pre-loaded model if GPU, otherwise string
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
    
    for iteration in range(num_iterations):
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"Iteration {iteration + 1}/{num_iterations}")
        logger.info("=" * 60)
        
        # Generate trajectories with current model
        logger.info(f"Generating {num_episodes_per_iteration} episodes...")
        rng_key, data_key = jax.random.split(rng_key)
        
        dataset = generate_grpo_dataset(
            env=env,
            llm_generate_fn=llm_generate_fn,
            num_episodes=num_episodes_per_iteration,
            max_steps=max_steps,
            reward_strategy="episode_shared",
            rng_key=data_key,
        )
        
        # Calculate statistics
        avg_reward = sum(dataset['reward']) / len(dataset) if len(dataset) > 0 else 0.0
        max_reward = max(dataset['reward']) if len(dataset) > 0 else 0.0
        min_reward = min(dataset['reward']) if len(dataset) > 0 else 0.0
        
        logger.success(f"Dataset generated: {len(dataset)} examples")
        logger.info(f"Reward statistics - Avg: {avg_reward:.2f}, Max: {max_reward:.2f}, Min: {min_reward:.2f}")
        
        # Update trainer dataset
        trainer.train_dataset = dataset
        
        # Train for one epoch
        logger.info("Starting training step...")
        trainer.train()
        logger.success("Training step completed")
        
        # Update LLM generation function with new model
        if hasattr(trainer, 'model'):
            # Reload model if needed
            logger.debug("Updating LLM generation function with trained model")
            llm_generate_fn = create_llm_generate_fn(
                trainer.model if hasattr(trainer.model, 'name_or_path') else model_name,
                tokenizer,
                vllm_engine
            )
        
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
    parser.add_argument("--num_iterations", type=int, default=100,
                       help="Number of training iterations")
    parser.add_argument("--output_dir", type=str, default="outputs/overcooked_grpo",
                       help="Output directory")
    parser.add_argument("--use_vllm", action="store_true",
                       help="Use vLLM for inference")
    
    args = parser.parse_args()
    
    train_grpo_overcooked(
        model_name=args.model,
        env_name=args.env,
        env_kwargs={"layout": args.layout, "max_steps": 400},
        num_episodes_per_iteration=args.num_episodes,
        num_iterations=args.num_iterations,
        output_dir=args.output_dir,
        use_vllm=args.use_vllm,
    )

