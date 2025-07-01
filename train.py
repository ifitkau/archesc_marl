# Multi-Agent PPO Training Script for Escape Room Environment
# =============================================================
# This script trains two agents (Navigator and Door Controller) using Ray RLlib's PPO algorithm
# in a custom escape room environment. The agents work cooperatively to solve navigation tasks.

# Core dependencies for Ray RLlib multi-agent training
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from escape_room.config.default_config import get_default_config
from escape_room.envs.navigator_agent import NavigatorAgent
from escape_room.envs.door_agent import DoorControllerAgent
from escape_room.utils.door_position_tracker import DoorPositionTracker

# Import the environment wrapper that creates the multi-agent environment
from escape_room.utils.gymnasium_wrapper import env_creator

# Standard libraries for file operations, logging, and utilities
import os
from ray.tune.logger import DEFAULT_LOGGERS, UnifiedLogger
import datetime
from ray.tune.logger.tensorboardx import TBXLoggerCallback

import sys
import numpy as np
import random
import json
import shutil
import torch
import ray

# Configure logging to reduce noise during training
import logging
logging.getLogger("ray").setLevel(logging.ERROR)
logging.getLogger("ray.rllib").setLevel(logging.ERROR)

# Import and configure Ray RLlib framework utilities
import ray.rllib.utils.framework as framework_utils
framework_utils.try_import_tf(True)
framework_utils.try_import_torch(True)

# Episode Tracking System
# =======================
# These global variables track training progress across iterations

# Main episode tracker - stores comprehensive episode statistics
episode_tracker = {
    "total_episodes": 0,           # Total episodes completed across all iterations
    "episodes_per_iteration": {},  # Episodes completed in each training iteration
    "successful_episodes": 0       # Episodes where agents successfully completed the task
}

# Simplified episode stats for easy access and monitoring
episode_stats = {
    "total_episodes": 0,      # Total episodes completed
    "successful_episodes": 0, # Successfully completed episodes
    "last_updated": None      # Timestamp of last update
}

# Persistent counter for successful episodes (prevents loss during checkpointing)
successful_episodes_counter = 0

# Persistence Functions
# ====================
# These functions ensure episode counts are preserved across training interruptions

def save_persistent_count(count):
    """
    Save the successful episodes count both in memory and to disk.
    This prevents loss of progress if training is interrupted.
    
    Args:
        count (int): Number of successful episodes to save
    
    Returns:
        int: The saved count
    """
    global successful_episodes_counter
    successful_episodes_counter = count
    
    # Save to JSON file for persistence across restarts
    count_path = os.path.join(run_dir, "persistent_success_count.json")
    with open(count_path, 'w') as f:
        json.dump({"successful_episodes": count}, f)
    
    return count

def save_successful_episodes_backup(count):
    """
    Create a backup of successful episodes count in a separate file.
    This file is not touched by RLlib's internal processes.
    
    Args:
        count (int): Number of successful episodes to backup
    
    Returns:
        int: The backed up count
    """
    backup_path = os.path.join(run_dir, "successful_episodes_backup.json")
    with open(backup_path, 'w') as f:
        json.dump({"successful_episodes": count}, f)
    return count

def restore_successful_episodes_backup():
    """
    Restore successful episodes count from the backup file.
    Used to recover progress after training interruptions.
    
    Returns:
        int: Restored successful episodes count, or 0 if no backup exists
    """
    backup_path = os.path.join(run_dir, "successful_episodes_backup.json")
    if os.path.exists(backup_path):
        try:
            with open(backup_path, 'r') as f:
                data = json.load(f)
                return data.get("successful_episodes", 0)
        except Exception as e:
            print(f"Error restoring successful episodes backup: {e}")
    return 0

# Episode Counting and Statistics
# ==============================

def update_episode_count(iteration, result):
    """
    Update episode tracking with the latest training results.
    Extracts episode information from RLlib's training results and updates global trackers.
    
    Args:
        iteration (int): Current training iteration number
        result (dict): Training results from RLlib containing episode statistics
    
    Returns:
        int: Number of episodes completed in this iteration
    """
    global episode_tracker
    
    # Extract episode count from RLlib result dictionary
    # RLlib may report episode counts in different ways depending on version
    episode_count = 0
    
    if "episodes_this_iter" in result:
        episode_count = result["episodes_this_iter"]
    elif "env_runners" in result and "num_episodes" in result["env_runners"]:
        episode_count = result["env_runners"]["num_episodes"]
    
    # Record episodes for this specific iteration
    iter_key = str(iteration)
    if iter_key not in episode_tracker["episodes_per_iteration"]:
        episode_tracker["episodes_per_iteration"][iter_key] = episode_count
    else:
        episode_tracker["episodes_per_iteration"][iter_key] += episode_count
    
    # Update total episode count across all iterations
    episode_tracker["total_episodes"] = sum(episode_tracker["episodes_per_iteration"].values())
    
    # Read successful episodes count from DoorPositionTracker's statistics file
    # The DoorPositionTracker maintains detailed episode success tracking
    tracker_stats_path = os.path.join(run_dir, "episode_stats.json")
    if os.path.exists(tracker_stats_path):
        try:
            with open(tracker_stats_path, 'r') as f:
                tracker_stats = json.load(f)
                if "successful_episodes" in tracker_stats:
                    episode_tracker["successful_episodes"] = tracker_stats["successful_episodes"]
        except Exception as e:
            print(f"Error reading successful episodes from tracker stats: {e}")
    
    # Persist episode tracking data to disk
    tracker_path = os.path.join(run_dir, "direct_episode_counts.json")
    with open(tracker_path, "w") as f:
        json.dump(episode_tracker, f, indent=4)
    
    # Update simplified episode statistics
    update_episode_stats()
    
    return episode_count

def update_episode_stats():
    """
    Update and save simplified episode statistics to a separate file.
    This provides an easy-to-read summary of training progress.
    """
    global episode_stats
    
    # Update statistics with current values
    episode_stats["total_episodes"] = episode_tracker["total_episodes"]
    episode_stats["successful_episodes"] = episode_tracker["successful_episodes"]
    episode_stats["last_updated"] = datetime.datetime.now().isoformat()
    
    # Save simplified stats to JSON file
    stats_path = os.path.join(run_dir, "episode_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(episode_stats, f, indent=4)

# Experiment Naming and Directory Setup
# ====================================
# The run directory name includes timestamp and experiment identifiers
# Historical naming conventions are documented below for reference

# Historical experiment naming conventions (for reference):
# 1. Create a unique folder for each run // mr = more rooms // & L = left terminal and R = right terminal and M = middle terminal // effex = exponential based efficiency reward for door-agent // seqs = sequential door position selection // pq0307 = path quality ratio set / trc = terminal reward change // how many iterations ran it100 = 100 iterations
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# Create unique run directory with timestamp
run_dir = os.path.join(os.path.dirname(__file__), "ray_results", f"run_{timestamp}")
os.makedirs(run_dir, exist_ok=True)

# Backup Critical Files
# ====================
# Save copies of important source files to preserve the exact configuration used for this training run

# Save a copy of this training script
script_path = os.path.abspath(__file__)
script_backup_path = os.path.join(run_dir, "train_backup.py")
shutil.copy2(script_path, script_backup_path)
print(f"Saved a backup of the training script to: {script_backup_path}")

# Save a copy of the default configuration file
config_script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                 "escr_marl", "escape_room", "config", "default_config.py")
config_backup_path = os.path.join(run_dir, "default_config_backup.py")
shutil.copy2(config_script_path, config_backup_path)
print(f"Saved a backup of the default config to: {config_backup_path}")

# Save a copy of the door agent implementation
dooragent_script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                 "escr_marl", "escape_room", "envs", "door_agent.py")
dooragent_backup_path = os.path.join(run_dir, "default_dooragent_backup.py")
shutil.copy2(dooragent_script_path, dooragent_backup_path)
print(f"Saved a backup of the default config to: {dooragent_backup_path}")

# Save a copy of the navigator agent implementation
navigatoragent_script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                 "escr_marl", "escape_room", "envs", "navigator_agent.py")
navigatoragent_backup_path = os.path.join(run_dir, "default_navigatoragent_backup.py")
shutil.copy2(navigatoragent_script_path, navigatoragent_backup_path)
print(f"Saved a backup of the default config to: {navigatoragent_backup_path}")

# Random Seed Configuration
# ========================
# Set up random seed for reproducibility or randomness based on current approach

# Option 1: Fixed seed for reproducible results (currently commented out)
"""SEED = 42
print(f"Using fixed seed: {SEED}")"""

# Option 2: Random seed based on current time (currently active)
import time
SEED = int(time.time()) % (2**31-1)  # Ensure seed fits in 32-bit signed integer
print(f"Using random seed: {SEED}")

# Set random seeds for all relevant libraries to ensure reproducibility
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

# Initialize Ray with single CPU and local mode for debugging
ray.init(num_cpus=1, local_mode=True)

# GPU-specific seeding and configuration (if CUDA is available)
"""if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(SEED)"""

# Enhanced GPU determinism settings for CUDA >= 10.2
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    # Configure CUDA workspace for deterministic operations
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # Required for CUDA >= 10.2
    
# Additional determinism settings (currently commented out)
"""torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)"""

# Environment Configuration Setup
# ==============================
# Get default configuration and add the random seed
config_default = get_default_config()
config_default["seed"] = SEED

# Suppress deprecation warnings to reduce console noise during training
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Custom TensorBoard Logger
# ========================
# Filter TensorBoard logging to only show relevant metrics and reduce log size

class FilteredTBXLogger(TBXLoggerCallback):
    """
    Custom TensorBoard logger that filters out unnecessary metrics.
    Only logs essential training metrics to keep TensorBoard files manageable.
    """
    def log_trial_result(self, iteration, trial, result):
        filtered_result = {}
        
        # Define which metrics to keep in TensorBoard logs
        metrics_to_keep = [
            "ray/tune/env_runners/episode_len_mean",      # Average episode length
            "ray/tune/env_runners/episode_len_min",       # Minimum episode length
            "ray/tune/env_runners/agent_episode_returns_mean/door_controller",  # Door agent rewards
            "ray/tune/env_runners/agent_episode_returns_mean/navigator",        # Navigator rewards
            "ray/tune/env_runners/agent_steps/navigator"  # Navigator step count
        ]
        
        # Extract only the specified metrics from the full result dictionary
        for path in metrics_to_keep:
            parts = path.split('/')
            current = result
            # Navigate through nested dictionary structure
            for part in parts:
                if part in current:
                    current = current[part]
                else:
                    current = None
                    break
            
            # If metric exists, add it to filtered results
            if current is not None:
                # Reconstruct the nested dictionary structure for this metric
                nested_dict = {}
                temp = nested_dict
                for i, part in enumerate(parts[:-1]):
                    temp[part] = {}
                    temp = temp[part]
                temp[parts[-1]] = current
                
                # Merge with existing filtered results
                filtered_result.update(nested_dict)
        
        # Call parent method with filtered result to log to TensorBoard
        super().log_trial_result(iteration, trial, filtered_result)

# Environment and Agent Setup
# ==========================

# Load and configure the default environment settings
config_default = get_default_config()
config_default["seed"] = SEED
config_default["use_sequential_all_positions"] = False  # Use random door positions
config_default["parallel_env"] = True                   # Enable parallel environment execution
config_default["random_agent_placement"] = True        # Randomize agent starting positions

# Optional: Dynamic door movement frequency (currently commented out)
#config_default["door_move_frequency"] = 3

# Register the custom environment with Ray RLlib
register_env("escape_room", lambda config: env_creator(config))

# Create agent instances to extract observation and action spaces
nav_agent = NavigatorAgent(config=config_default)      # Agent responsible for navigation
door_agent = DoorControllerAgent(config=config_default) # Agent responsible for door control

# Create checkpoints directory for saving model weights
checkpoints_dir = os.path.join(run_dir, "checkpoints")
os.makedirs(checkpoints_dir, exist_ok=True)

# Initialize door position tracker for episode success monitoring
door_tracker = DoorPositionTracker(run_dir=run_dir, seed=SEED)

# PPO Algorithm Configuration
# ==========================
# Configure the Proximal Policy Optimization algorithm for multi-agent training

config = (
    PPOConfig()
    # Environment configuration
    .environment("escape_room", env_config={**config_default, "seed": SEED})
    # Use PyTorch as the deep learning framework
    .framework("torch")
    # Configure environment runners (workers that collect experience)
    .env_runners(num_env_runners=1,  # Number of parallel workers
                num_envs_per_env_runner=1)  # Environments per worker
    # Multi-agent configuration
    .multi_agent(
        policies={
            # Define separate policies for each agent type
            "navigator": (None, nav_agent.observation_space, nav_agent.action_space, {}),
            "door_controller": (None, door_agent.observation_space, door_agent.action_space, {})
        },
        # Map agent IDs to their corresponding policies
        policy_mapping_fn=lambda agent_id, episode, worker=None: agent_id
    )
    # Training hyperparameters
    .training(
        train_batch_size=12000,        # Total batch size for training
        num_epochs=20,                 # Number of epochs per training iteration
        lr=3e-4,                       # Learning rate
        entropy_coeff=0.01,            # Entropy coefficient for exploration
        shuffle_batch_per_epoch=False, # Whether to shuffle batches between epochs
        minibatch_size=128             # Size of mini-batches for gradient updates
    )
    # Resource allocation (using CPU only)
    .resources(num_gpus=0)
    # Enable Ray RLlib's new API stack for improved performance
    .api_stack(
        enable_rl_module_and_learner=True,        # New modular architecture
        enable_env_runner_and_connector_v2=True   # Improved environment integration
    )
    # Reporting and logging configuration
    .reporting(
        min_time_s_per_iteration = 5,                    # Minimum time per iteration
        metrics_num_episodes_for_smoothing = 5,         # Episodes for metric smoothing
        log_gradients = True,                            # Log gradient information
        keep_per_episode_custom_metrics=True,           # Preserve episode-level metrics
        metrics_episode_collection_timeout_s=60,        # Timeout for metric collection
    )
    # Add custom callback for door position tracking
    .callbacks(lambda logdir=run_dir: door_tracker)
    # Debugging and logging settings
    .debugging(
        seed=SEED,           # Random seed for reproducibility
        log_level="ERROR",   # Reduce log verbosity
    )
)

# Logger Configuration
# ===================
# Set up custom logging that includes our filtered TensorBoard logger

# Create custom logger list excluding default TensorBoard logger
custom_loggers = [logger for logger in DEFAULT_LOGGERS if not isinstance(logger, TBXLoggerCallback)]
# Add our custom filtered TensorBoard logger
custom_loggers.append(FilteredTBXLogger)

# Build the PPO trainer with custom logging configuration
trainer = config.build(
    logger_creator=lambda cfg: UnifiedLogger(
        cfg, 
        loggers=custom_loggers, 
        logdir=run_dir
    ),
)

# Training Loop
# =============
# Main training loop that runs for the specified number of iterations

num_iterations = 3000  # Total number of training iterations
# Optional: Iteration at which to change door movement frequency
#door_frequency_change_iteration = 100

for i in range(num_iterations):
    # Optional: Dynamic configuration change during training (currently commented out)
    """if i == door_frequency_change_iteration:
        print(f"Changing door move frequency from 3 to 1 at iteration {i+1}")
        # Get the current config from the trainer
        current_env_config = trainer.get_config().env_config
        # Update door move frequency
        current_env_config["door_move_frequency"] = 1
        # Update the config in the trainer
        trainer.get_config().environment_config["env_config"] = current_env_config
        # Reset all environments with the new config
        trainer.workers.foreach_worker(
            lambda worker: worker.foreach_env(
                lambda env: env.env.env.config.update({"door_move_frequency": 1})
            )
        )"""
    
    # Restore successful episodes count from backup before training
    # This ensures continuity if training was interrupted
    door_tracker.successful_episodes = restore_successful_episodes_backup()
    
    # Execute one training iteration
    result = trainer.train()
    
    # Save successful episodes count to backup after training
    save_successful_episodes_backup(door_tracker.successful_episodes)
    
    # Update episode tracking with results from this iteration
    current_episodes = update_episode_count(i+1, result)
    print(f"Iteration {i+1}: completed with {current_episodes} episodes (successful: {door_tracker.successful_episodes})")

    # Checkpoint Saving
    # ================
    # Save model checkpoints periodically for recovery and analysis
    if (i + 1) % 10 == 0:
        # Create checkpoint filename with iteration number
        checkpoint_filename = f"checkpoint_iter_{i+1}.pkl"
        checkpoint_full_path = os.path.join(checkpoints_dir, checkpoint_filename)
        
        try:
            # Save trainer state to checkpoint file
            checkpoint_path = trainer.save(checkpoint_full_path)
            print(f"Checkpoint saved at {checkpoint_path}")
            
            # Force update of episode statistics after checkpoint
            update_episode_stats()
        except Exception as e:
            print(f"Error saving checkpoint: {e}")

# Post-Training Analysis and Cleanup
# ==================================

# Restore and verify final episode counts
final_successful_count = restore_successful_episodes_backup()

def read_tracker_state():
    """
    Read the final state from the door position tracker.
    This provides the authoritative count of successful episodes.
    
    Returns:
        tuple: (successful_episodes, total_episodes) from tracker state
    """
    state_file = os.path.join(run_dir, 'tracker_state_direct.json')
    if os.path.exists(state_file):
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
                return state.get('successful_episodes', 0), state.get('total_episodes', 0)
        except Exception as e:
            print(f"Error reading tracker state: {e}")
    return 0, 0

# Read final statistics from tracker
successful_count, tracker_total = read_tracker_state()
print(f"Final successful episodes from tracker: {successful_count}")
print(f"Total episodes from tracker: {tracker_total}")
print(f"Total episodes from episode_tracker: {episode_tracker['total_episodes']}")

# Update episode tracker with most reliable information
episode_tracker["successful_episodes"] = successful_count
update_episode_stats()

# Training Summary
# ===============
# Print final training results and instructions for viewing logs

print(f"\nTraining completed.")
print(f"Total episodes: {episode_tracker['total_episodes']}")
print(f"Successful episodes: {final_successful_count}")
print(f"Checkpoints saved in: {checkpoints_dir}")
print(f"TensorBoard logs can be viewed by running:")
print(f"tensorboard --logdir={run_dir}")