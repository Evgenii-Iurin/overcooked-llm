# Overcooked LLM Trainer

Training language models for multi-agent communication and coordination in the OvercookedV2 environment using GRPO (Group Relative Policy Optimization).

## Overview

This project trains a shared LLM to handle communication and high-level planning for two agents in the OvercookedV2 environment. The LLM processes each agent's observations sequentially (with conversation history) and generates communication messages and action plans, which are then executed by a separate action executor.

## System Requirements

- **OS**: Linux (tested on Ubuntu 20.04+)
- **Python**: 3.11 - 3.13
- **CUDA**: 12.1+ (for GPU support)
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **Memory**: At least 16GB RAM, 8GB+ VRAM for LLM training

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd overcooked-llm
```

### 2. Initialize JaxMARL Submodule

```bash
git submodule update --init --recursive
```

### 3. Set Up Conda Environment

Create and activate a conda environment:

```bash
conda create -n overcooked-llm python=3.11
conda activate overcooked-llm
```

### 4. Install CUDA and cuDNN

**Important**: PyTorch (required by TRL) needs cuDNN 8, not cuDNN 9. Install the correct version:

```bash
# Install cuDNN 8.9.2 (compatible with PyTorch 2.1.0+)
conda install -c conda-forge cudnn=8.9.2 -y

# Verify installation
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('cuDNN version:', torch.backends.cudnn.version() if torch.cuda.is_available() else 'N/A')"
```

Expected output:
```
PyTorch: 2.1.0+cu121
CUDA available: True
cuDNN version: 8902
```

**Note**: If you encounter `ImportError: libcudnn.so.8: cannot open shared object file`, it means cuDNN 8 is not installed. Make sure to install cuDNN 8.9.2 as shown above.

### 5. Install JAX with CUDA Support

After installing cuDNN, install JAX with CUDA 12 support:

```bash
# Install JAX with CUDA 12 support
pip install --upgrade "jax[cuda12_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Verify JAX can see GPU
python -c "import jax; print('JAX devices:', jax.devices())"
```

You should see your GPU listed (e.g., `[gpu:0]`).

### 6. Install Project Dependencies with Poetry

```bash
# Install Poetry if not already installed
curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
poetry install
```

### 7. Verify Installation

Run a quick test to verify everything is set up correctly:

```bash
python -c "
import jax
import torch
from jaxmarl import make

print('JAX version:', jax.__version__)
print('JAX devices:', jax.devices())
print('PyTorch version:', torch.__version__)
print('PyTorch CUDA available:', torch.cuda.is_available())

# Test environment
env = make('overcooked_v2', layout='cramped_room')
print('Environment created successfully!')
print('Number of agents:', env.num_agents)
"
```

## Quick Start

### Training

Train the LLM using GRPO on the Overcooked environment:

```bash
python -m llm_trainer.train_grpo \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --layout cramped_room \
    --num_episodes 10 \
    --num_iterations 100 \
    --output_dir outputs/overcooked_grpo \
    --use_vllm
```

### Command Line Arguments

- `--model`: HuggingFace model name or path (default: `Qwen/Qwen2.5-0.5B-Instruct`)
- `--env`: Environment name (default: `overcooked_v2`)
- `--layout`: Layout name (default: `cramped_room`)
- `--num_episodes`: Episodes per training iteration (default: `10`)
- `--num_iterations`: Number of training iterations (default: `100`)
- `--output_dir`: Output directory for checkpoints (default: `outputs/overcooked_grpo`)
- `--use_vllm`: Use vLLM for faster inference (requires vLLM installation)

## Project Structure

```
src/llm_trainer/
├── observation_encoder.py      # Convert grid observations to text
├── communication.py            # Agent communication protocol
├── action_executor.py          # Translate plans to discrete actions
├── overcooked_llm_wrapper.py  # Environment wrapper with LLM integration
├── reward_functions.py         # GRPO reward functions
├── dataset_generator.py        # Generate training datasets
├── train_grpo.py              # Main training script
├── evaluation.py              # Evaluation metrics
└── config/
    └── grpo_config.yaml       # Training configuration
```

## Architecture

The system uses a **shared LLM** that processes agents sequentially:

1. **Agent 0** sees: Agent 0's observation + conversation history → generates message to Agent 1 + plan for Agent 0
2. **Agent 1** sees: Agent 1's observation + updated conversation history → generates message to Agent 0 + plan for Agent 1
3. **Action Executor** translates high-level plans to discrete environment actions
4. **Environment** executes actions and provides rewards

This maintains realistic communication where agents don't have direct access to each other's observations.

## Troubleshooting

### cuDNN Version Mismatch

**Error**: `ImportError: libcudnn.so.8: cannot open shared object file`

**Solution**: Install cuDNN 8.9.2:
```bash
conda install -c conda-forge cudnn=8.9.2 -y
```

### JAX GPU Not Detected

**Error**: JAX falls back to CPU even with GPU available

**Solution**: Install JAX with CUDA support:
```bash
pip install --upgrade "jax[cuda12_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Poetry Installation Issues

If Poetry commands fail, ensure Poetry is in your PATH:
```bash
export PATH="$HOME/.local/bin:$PATH"
```

## Dependencies

Key dependencies (managed by Poetry):
- `jaxmarl`: Multi-agent RL environments (local submodule)
- `jax`: JAX with CUDA support
- `transformers`: HuggingFace transformers for LLM
- `trl`: TRL library for GRPO training
- `datasets`: HuggingFace datasets
- `torch`: PyTorch (installed via transformers/trl dependencies)

## License

[Add your license here]

## Citation

If you use this code, please cite:
- JaxMARL: [citation]
- GRPO: [citation]
- OvercookedV2: [citation]
