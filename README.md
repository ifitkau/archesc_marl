# CODE FOR IMPLICIT COORDINATION THROUGH ENVIRONMENT MODIFICATION: MULTI-AGENT RL APPROACH FOR ADAPTIVE DOOR PLACEMENT IN EVACUATION SCENARIOS

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository implements a multi-agent reinforcement learning system for dynamic door placement in evacuation scenarios. The framework enables environment-mediated coordination between a navigator agent learning escape paths and a door controller agent strategically positioning doors based on observed navigation patterns, without explicit communication protocols.

## Implementation Architecture

The system is built on Ray RLlib with PPO optimization, utilizing MiniWorld for 3D simulation and PettingZoo for multi-agent standardization. The modular design supports both parallel and sequential agent execution paradigms within a customizable multi-room environment.

## Core Components

### Training Framework

#### `train.py` - Multi-Agent Training Orchestrator
Primary training script coordinating simultaneous agent learning with comprehensive tracking capabilities.

**Key Features:**
- Ray RLlib integration with PPO algorithm configuration
- Custom callback system for door position and episode success monitoring
- Automated checkpoint management with iteration-based saving
- Weights & Biases integration for experiment tracking
- Persistent episode counting and performance metrics

#### `env_sarl.py` - Single-Agent Reference Implementation
Baseline single-agent environment for comparative analysis and framework validation.

**Functionality:**
- Custom MiniWorld environment extension with escape room scenarios
- Multi-component reward system (orientation, collision avoidance, progress tracking)
- LIDAR-based spatial perception with 5-directional sensing
- Comprehensive tracking callbacks for detailed behavioral analysis
- Wall interaction mechanics for dynamic door creation

### Environment Implementations

#### `escape_room_env.py` - AEC Multi-Agent Environment
Turn-based multi-agent implementation using PettingZoo's Agent-Environment-Cycle API.

**Core Features:**
- Sequential agent execution with proper turn management
- Interactive rendering with pause/resume functionality via spacebar
- Episode lifecycle management with termination condition handling
- Agent-specific observation and action space configuration

#### `escape_room_env_parallel.py` - Parallel Multi-Agent Environment
Simultaneous multi-agent execution using PettingZoo's Parallel API for optimized training performance.

**Optimizations:**
- Concurrent agent action processing for improved computational efficiency
- Dynamic door controller activation based on episode frequency parameters
- Coordinated reward distribution and state synchronization
- Optimized step counting logic for parallel environment paradigms

#### `miniworld_env.py` - Base 3D Environment
Foundation environment extending MiniWorld with specialized escape room functionality.

**Implementation Details:**
- Multi-room layout with controllable door positioning systems
- LIDAR-based obstacle detection and collision avoidance
- Door safe zone management preventing agent entrapment during door movement
- Spatial tracking and room categorization for navigation assistance

### Agent Architectures

#### `navigator_agent.py` - Navigation Agent
Pathfinding agent responsible for escape route optimization and spatial navigation.

**Configuration:**
- Discrete action space: turn left/right, move forward
- Rich observation space including LIDAR, door positions, spatial awareness
- Room-aware door visibility system (context-dependent observation)
- Multi-component reward structure encouraging efficient navigation

#### `door_agent.py` - Door Controller Agent
Strategic door placement agent optimizing spatial configurations based on navigator performance.

**Capabilities:**
- Multi-discrete action space for individual room door control
- Environment analysis and navigator behavior pattern recognition
- Reward system based on navigation efficiency and success metrics
- Room-specific door position optimization with performance tracking

### Support Infrastructure

#### `gymnasium_wrapper.py` - RLlib Integration Wrapper
Compatibility layer ensuring seamless integration with Ray RLlib training pipelines.

**Optimizations:**
- Pre-allocated data structures for memory efficiency during training
- Proper step counting logic handling AEC vs Parallel environment differences
- Robust error handling and agent termination management
- Support for both environment paradigms with automatic detection

#### `door_position_tracker.py` - Training Analytics System
Comprehensive tracking and analysis framework for training progression monitoring.

**Analytics Features:**
- Episode success/failure rate monitoring with statistical analysis
- Door position frequency tracking across training phases
- Navigator starting position correlation with outcome analysis
- JSON and CSV export functionality for external data processing

#### `visualization.py` - Analysis and Visualization Suite
Advanced visualization utilities for training analysis and result presentation.

**Visualization Types:**
- Agent trajectory visualization with room layout overlays
- Door position distribution analysis across training iterations
- Multi-agent reward plotting with separate scaling mechanisms
- Success rate analysis with trend identification and statistical smoothing

### Evaluation Tools

#### `random_agent.py` - Environment Validation Framework
Comprehensive testing framework using random agents for environment validation and baseline establishment.

**Testing Capabilities:**
- Random action testing with detailed reward component analysis
- Door vs terminal orientation behavior comparison
- Environment state validation and debugging utilities
- Interactive rendering with debugging controls

#### `save_traj_allpaths_par.py` - Trained Model Analysis
Post-training analysis tool for comprehensive trajectory evaluation and visualization generation.

**Analysis Features:**
- Trained model checkpoint loading and systematic evaluation
- Trajectory path generation with success/failure classification
- Room-specific door position effectiveness analysis
- JSON export for programmatic analysis and further processing

## Installation

```bash
git clone https://github.com/yourusername/multi-agent-door-placement.git
cd multi-agent-door-placement
pip install -r requirements.txt
```

### Required Dependencies

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

### Single-Agent Baseline Training
```bash
python env_sarl.py
```

## Configuration

Environment and training parameters are centralized in `default_config.py`, supporting comprehensive customization of:

- **World Geometry**: Room dimensions, door constraints, terminal location
- **Agent Behavior**: Starting positions, observation spaces, action spaces
- **Reward Systems**: Component weights for both navigator and door controller
- **Training Parameters**: Episode limits, learning rates, batch sizes

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

## Results and Analysis

The system generates comprehensive analysis outputs:

1. **Training Metrics**: Episode success rates, step count reduction, reward convergence
2. **Door Position Analysis**: Frequency distributions across training phases
3. **Path Visualizations**: Agent trajectories with successful/failed episode distinction
4. **Performance Tracking**: Statistical analysis and trend identification

## Research Context

This implementation supports investigation of environment-mediated coordination in multi-agent systems, where meaningful coordination strategies emerge through spatial modifications rather than explicit communication protocols. The framework demonstrates applications in:

- Adaptive architectural design optimization
- Emergency evacuation path planning
- Dynamic environment modification strategies
- Performance-based building design tools

## Citation

If you use this code in your research, please cite:

```bibtex
@article{multi_agent_door_placement_2024,
  title={Implicit Coordination Through Environment Modification: Multi-Agent RL Approach for Adaptive Door Placement in Evacuation Scenarios},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

## License

MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Acknowledgments

This implementation builds upon:
- [Ray RLlib](https://docs.ray.io/en/latest/rllib/) for distributed reinforcement learning
- [MiniWorld](https://github.com/maximecb/gym-miniworld) for 3D environment simulation
- [PettingZoo](https://pettingzoo.farama.org/) for multi-agent environment standardization
