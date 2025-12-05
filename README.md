# CODE FOR IMPLICIT COORDINATION THROUGH ENVIRONMENT MODIFICATION: MULTI-AGENT RL APPROACH FOR ADAPTIVE DOOR PLACEMENT IN EVACUATION SCENARIOS

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository implements a multi-agent reinforcement learning system for dynamic door placement in evacuation scenarios. The framework enables environment-mediated coordination between a navigator agent learning escape paths and a door controller agent strategically positioning doors based on observed navigation patterns, without explicit communication protocols.

## Implementation Architecture

The system is built on Ray RLlib with PPO optimization, utilizing MiniWorld for 3D simulation and PettingZoo for multi-agent standardization. The modular design supports both parallel and sequential agent execution paradigms within a customizable multi-room environment.

## Core Components

### Training Framework

#### `train.py`
Primary training script coordinating simultaneous agent learning with comprehensive tracking capabilities.

#### `env_sarl.py` - Single-Agent Reference Implementation
Baseline single-agent environment for comparative analysis and framework validation.

### Environment Implementations

#### `escape_room_env.py` - AEC Multi-Agent Environment
Turn-based multi-agent implementation using PettingZoo's Agent-Environment-Cycle API.

#### `escape_room_env_parallel.py` - Parallel Multi-Agent Environment
Simultaneous multi-agent execution using PettingZoo's Parallel API for optimized training performance.

#### `miniworld_env.py` - Base 3D Environment
Foundation environment extending MiniWorld with specialized escape room functionality.


### Agent Architectures

#### `navigator_agent.py` - Navigation Agent
Pathfinding agent responsible for escape route optimization and spatial navigation.

#### `door_agent.py` - Door Controller Agent
Strategic door placement agent optimizing spatial configurations based on navigator performance.


### Support Infrastructure

#### `gymnasium_wrapper.py` - RLlib Integration Wrapper
Compatibility layer ensuring seamless integration with Ray RLlib training pipelines.

#### `door_position_tracker.py` - Metrics Tracker
Comprehensive tracking and analysis framework for training progression monitoring.

#### `visualization.py` - Analysis and Visualization Suite
Advanced visualization utilities for training analysis and result presentation.


### Evaluation Tools

#### `random_agent.py` - Test-Script
Comprehensive testing framework using random agents for environment validation and baseline establishment.

#### `save_traj_allpaths_par.py` - Evaluation Script
Post-training analysis tool for comprehensive trajectory evaluation and visualization generation.


```bash
# Core RL dependencies
pip install ray[rllib]>=2.0.0
pip install miniworld>=1.2.0
pip install pettingzoo>=1.22.0
pip install gymnasium>=0.29.0
pip install stable-baselines3>=2.0.0

# Analysis and visualization
pip install matplotlib>=3.5.0
pip install numpy>=1.21.0
pip install scipy>=1.7.0
pip install pandas>=1.3.0

# Optional: Experiment tracking
pip install wandb>=0.15.0
```

## Usage

### Multi-Agent Training
```bash
python train.py
```

### Environment Validation
```bash
python random_agent.py --num-episodes 50 --output-dir validation_results
```

### Trained Model Analysis
```bash
python save_traj_allpaths_par.py --checkpoint path/to/checkpoint --output_dir analysis_results
```


## Configuration

Environment and training parameters are centralized in `default_config.py`.

### Example Configuration

```python
config = {
    'world_width': 18.4,
    'world_depth': 6.7,
    'max_episode_steps': 500,
    'door_move_frequency': 1,
    'discrete_door_positions': 5,
    'navigator_reward_scales': {
        'reward_orientation_scale': 1.0,
        'reward_terminal_scale': 0.3,
        'wall_collision_scale': 1.0
    },
    'door_controller_reward_scales': {
        'reward_success_scale': 0.1,
        'reward_terminal_steps_scale': 2.0
    }
}
```


## Citation

tba.

## License

MIT License

Copyright (c) 2025

## Acknowledgments

This implementation builds upon:
- [Ray RLlib](https://docs.ray.io/en/latest/rllib/) for distributed reinforcement learning
- [MiniWorld](https://github.com/maximecb/gym-miniworld) for 3D environment simulation
- [PettingZoo](https://pettingzoo.farama.org/) for multi-agent environment standardization
