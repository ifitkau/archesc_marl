# Trajectory Visualization Script for Trained Escape Room Agents
# ==============================================================
# This script loads trained RL models from checkpoints and generates comprehensive
# path trajectory visualizations and analysis reports. It supports both Parallel
# and AEC environments and provides detailed statistics about agent performance.

"""
Script to load a checkpoint from the escape room environment and save path trajectories
as PNG files similar to the example image.

Updated to work with both Parallel and AEC environments automatically.

Usage:
  python save_traj_allpaths.py --checkpoint PATH_TO_CHECKPOINT --output_dir OUTPUT_DIRECTORY --iteration ITERATION_NUMBER
"""

# Standard libraries for data processing, visualization, and file operations
import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import pathlib
import ray

# Ray RLlib imports for loading trained models
from ray.rllib.core.rl_module import RLModule

# Custom escape room environment configuration
from escape_room.config.default_config import get_default_config

def parse_args():
    """
    Parse command line arguments for trajectory visualization configuration.
    
    Returns:
        argparse.Namespace: Parsed arguments with visualization and analysis settings
    """
    parser = argparse.ArgumentParser(description='Save path trajectories from trained agents')
    
    # Model and checkpoint configuration
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help='Path to the checkpoint file containing trained model weights')
    parser.add_argument('--output_dir', type=str, default='trajectories', 
                        help='Directory to save path trajectories and analysis reports')
    parser.add_argument('--iteration', type=int, default=None, 
                        help='Iteration number to include in visualization titles')
    
    # Episode and seed configuration
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducible episode generation')
    parser.add_argument('--num_episodes', type=int, default=20, 
                        help='Number of episodes to run for trajectory collection')
    
    # Agent placement and environment settings
    parser.add_argument('--random_placement', action='store_true', 
                        help='Use random agent placement instead of fixed positions')
    parser.add_argument('--loop_positions', action='store_true', 
                        help='Loop through all starting positions defined in config')
    parser.add_argument('--terminal_location', type=str, default=None, 
                        help='Override terminal location (format: "x,y" e.g. "6.2,4.2")')
    
    # Door controller behavior configuration
    parser.add_argument('--door_move_frequency', type=int, default=1, 
                        help='Door move frequency (1 = every episode, default: 1)')
    parser.add_argument('--force_door_action', action='store_true', 
                        help='Force door controller to act in every episode')
    
    # Visualization and filtering options
    parser.add_argument('--max_steps', type=int, default=None, 
                        help='Only save paths with fewer than this many steps (default: no limit)')
    parser.add_argument('--include_all_paths', action='store_true',
                        help='Include all paths in visualization, not just successful ones')
    
    # Environment type forcing (for debugging)
    parser.add_argument('--force_parallel', action='store_true',
                        help='Force use of parallel environment')
    parser.add_argument('--force_aec', action='store_true',
                        help='Force use of AEC environment')
    
    return parser.parse_args()

def load_rl_modules(checkpoint_path):
    """
    Load the trained RL modules (neural networks) from a checkpoint directory.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file or directory
        
    Returns:
        dict: Dictionary containing loaded RL modules for each agent type
        
    Raises:
        Exception: If checkpoint loading fails or required modules are not found
    """
    print(f"Loading RL modules from: {checkpoint_path}")
    
    # Handle both file and directory checkpoint paths
    if checkpoint_path.endswith('.pkl') and not os.path.isdir(checkpoint_path):
        checkpoint_dir = os.path.dirname(checkpoint_path)
    else:
        checkpoint_dir = checkpoint_path
    
    # Navigate to the correct checkpoint directory structure
    # Ray RLlib stores checkpoints in a specific nested structure
    if not os.path.exists(os.path.join(checkpoint_dir, "learner_group")):
        # Navigate up one level if we're in a specific checkpoint folder
        if os.path.exists(os.path.join(os.path.dirname(checkpoint_dir), "learner_group")):
            checkpoint_dir = os.path.dirname(checkpoint_dir)
    
    # Load the RL modules from the checkpoint structure
    rl_modules = {}
    try:
        # Construct path to the RL module checkpoints
        learner_path = pathlib.Path(checkpoint_dir) / "learner_group" / "learner" / "rl_module"
        
        # Load navigator agent neural network
        rl_modules["navigator"] = RLModule.from_checkpoint(learner_path)["navigator"]
        print("Successfully loaded navigator RL module")
        
        # Load door controller agent neural network
        rl_modules["door_controller"] = RLModule.from_checkpoint(learner_path)["door_controller"]
        print("Successfully loaded door controller RL module")
    except Exception as e:
        print(f"Error loading RL modules: {e}")
        raise
    
    return rl_modules

def create_environment(config, args):
    """
    Create the appropriate environment (Parallel or AEC) based on configuration.
    
    Args:
        config (dict): Environment configuration dictionary
        args (argparse.Namespace): Command line arguments
        
    Returns:
        tuple: (environment_instance, is_parallel_flag)
    """
    # Determine environment type based on arguments and configuration
    if args.force_parallel:
        config["parallel_env"] = True
        use_parallel = True
    elif args.force_aec:
        config["parallel_env"] = False
        use_parallel = False
    else:
        # Use configuration setting or default to parallel
        use_parallel = config.get("parallel_env", True)
        config["parallel_env"] = use_parallel
    
    print(f"Using {'Parallel' if use_parallel else 'AEC'} environment")
    
    # Import and create the appropriate environment type
    if use_parallel:
        from escape_room.envs.escape_room_env_parallel import ParallelEscapeRoomEnv
        env = ParallelEscapeRoomEnv(config=config, render_mode=None, view="top")
    else:
        from escape_room.envs.escape_room_env import EscapeRoomEnv
        env = EscapeRoomEnv(config=config, render_mode=None, view="top")
    
    return env, use_parallel

def patch_door_controller(env, use_parallel=True):
    """
    Apply patches to the door controller to prevent common runtime errors.
    Fixes issues with missing navigator_start_pos that can cause crashes.
    
    Args:
        env: Environment instance containing the door controller
        use_parallel (bool): Whether using parallel environment
    """
    try:
        # Extract door controller from environment
        if use_parallel:
            door_controller = env.door_controller
        else:
            door_controller = env.door_controller
        
        # Fix navigator_start_pos issues in efficiency reward calculation
        if hasattr(door_controller, 'calculate_efficiency_reward'):
            original_calculate_efficiency_reward = door_controller.calculate_efficiency_reward
            
            def patched_calculate_efficiency_reward(miniworld_env):
                """
                Patched version that ensures navigator_start_pos is always available.
                This prevents NoneType errors during reward calculation.
                """
                # Ensure navigator_start_pos is properly initialized
                if not hasattr(miniworld_env, 'navigator_start_pos') or miniworld_env.navigator_start_pos is None:
                    # Set to agent's current position if missing
                    if hasattr(miniworld_env, 'agent'):
                        miniworld_env.navigator_start_pos = miniworld_env.agent.pos
                    else:
                        # Default fallback position if agent is also missing
                        miniworld_env.navigator_start_pos = np.array([1.0, 0, 1.0])
                    print("  Fixed missing navigator_start_pos")
                
                # Call the original method with fixed state
                return original_calculate_efficiency_reward(miniworld_env)
            
            # Replace the method with the patched version
            door_controller.calculate_efficiency_reward = patched_calculate_efficiency_reward
            print("Door controller patched to handle missing navigator_start_pos")
    except Exception as e:
        print(f"Warning: Failed to patch door controller: {e}")

def create_path_visualization(all_paths, success_statuses, config, filename, iteration=None, max_steps=None, include_all=False):
    """
    Create a clean path visualization showing agent trajectories.
    
    Args:
        all_paths (list): List of paths, where each path is a list of 3D position points
        success_statuses (list): Boolean list indicating success/failure for each path
        config (dict): Environment configuration for world boundaries
        filename (str): Output filename for the visualization
        iteration (int, optional): Training iteration number for the title
        max_steps (int, optional): Maximum steps filter criteria for title
        include_all (bool): Whether to show all paths or only successful ones
    """
    plt.figure(figsize=(10, 10))
    
    # Extract environment boundaries from configuration
    world_width = config["world_width"]
    world_depth = config["world_depth"]
    
    # Count and categorize paths for statistics
    successful_count = sum(success_statuses)
    unsuccessful_count = len(success_statuses) - successful_count
    
    # Plot each trajectory with different styling based on success
    for idx, (path, is_successful) in enumerate(zip(all_paths, success_statuses)):
        path_array = np.array(path)
        if len(path_array) > 0:
            # Extract x and z coordinates (y is height, not needed for top-down view)
            x_coords = path_array[:, 0]
            z_coords = path_array[:, 2]
            
            # Apply different visual styles for successful vs unsuccessful paths
            if is_successful:
                # Successful paths: light orange lines with higher visibility
                plt.plot(x_coords, z_coords, '-', color='#FFA500', linewidth=1.5, alpha=0.7)
                # Mark starting position with a dot
                plt.scatter(x_coords[0], z_coords[0], color='#FFA500', s=30, alpha=0.9)
            elif include_all:
                # Unsuccessful paths: light blue lines with lower opacity
                plt.plot(x_coords, z_coords, '-', color='#87CEFA', linewidth=1.0, alpha=0.4)
                # Mark starting position with smaller, less visible dot
                plt.scatter(x_coords[0], z_coords[0], color='#87CEFA', s=20, alpha=0.6)
    
    # Configure plot boundaries and orientation
    plt.xlim(0, world_width)
    plt.ylim(0, world_depth)
    
    # Invert y-axis to match environment's coordinate system
    plt.gca().invert_yaxis()
    
    # Remove axes for clean visualization
    plt.axis('off')
    
    # Generate informative title with statistics
    title = ""
    if include_all:
        title = f"All Paths - {successful_count} Successful, {unsuccessful_count} Unsuccessful"
    else:
        title = f"Successful Paths Only - {successful_count} Paths"
    
    # Add step limit information if applicable
    if max_steps is not None:
        title += f" (< {max_steps} steps)"
    
    # Add training iteration information if available
    if iteration is not None:
        title += f" - Iteration {iteration}"
    
    plt.title(title, fontsize=16)
    
    # Add legend if showing both path types
    if include_all and unsuccessful_count > 0:
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='#FFA500', lw=2, label='Successful'),
            Line2D([0], [0], color='#87CEFA', lw=2, label='Unsuccessful')
        ]
        plt.legend(handles=legend_elements, loc='lower right')
    
    # Save with high quality and transparent background
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()

def get_room_from_position(start_pos):
    """
    Determine which room a position belongs to based on x-coordinate boundaries.
    
    Args:
        start_pos (tuple): Position coordinates (x, z)
    
    Returns:
        str: Room identifier ('roomA', 'roomB', 'roomC', or 'unknown')
    """
    x, z = start_pos
    
    # Room boundaries based on environment configuration
    # These boundaries define the three rooms in the escape room
    if 0 <= x < 4.0:
        return 'roomA'      # First room (leftmost)
    elif 4.2 <= x < 8.2:
        return 'roomB'      # Middle room
    elif 8.4 <= x <= 12.4:
        return 'roomC'      # Third room (rightmost)
    else:
        # Position is outside known room boundaries
        return 'unknown'

def run_episodes_and_collect_paths(rl_modules, config, args):
    """
    Execute episodes with trained agents and collect trajectory data for analysis.
    
    Args:
        rl_modules (dict): Loaded neural network modules for each agent
        config (dict): Environment configuration
        args (argparse.Namespace): Command line arguments
        
    Returns:
        tuple: (all_paths_list, success_status_list) containing trajectory data
    """
    # Initialize Ray if not already running
    output_dir = args.output_dir
    if not ray.is_initialized():
        ray.init(num_cpus=1, local_mode=True, ignore_reinit_error=True)
    
    # Create environment instance
    env, use_parallel = create_environment(config, args)
    
    # Apply necessary patches for stability
    patch_door_controller(env, use_parallel)
    
    # Configure starting positions for systematic testing
    starting_positions = []
    if not args.random_placement and args.loop_positions:
        room_cats = config["room_categories"]
        # Get the highest probability room category (typically "room")
        first_cat = max(config["room_probabilities"], key=config["room_probabilities"].get)
        
        if first_cat in room_cats:
            starting_positions = room_cats[first_cat]
            print(f"Found {len(starting_positions)} starting positions to loop through:")
            for i, (pos, dir) in enumerate(starting_positions):
                # Determine which room this position belongs to
                room_name = get_room_from_position((pos[0], pos[2]))
                print(f"  {i}: {pos} (in {room_name})")

    # Apply terminal location override if specified
    if args.terminal_location:
        try:
            x, y = map(float, args.terminal_location.split(','))
            if use_parallel:
                env.world.terminal_location = [x, y]
            else:
                env.world.terminal_location = [x, y]
            print(f"Terminal location overridden to: {[x, y]}")
        except Exception as e:
            print(f"Error setting terminal location: {e}")
    
    # Initialize data collection containers
    all_paths = []              # All trajectory paths
    all_success_statuses = []   # Success/failure status for each episode
    all_episodes_data = []      # Comprehensive episode data
    successful_episodes_data = [] # Data for successful episodes only
    
    # Track door positions across all episodes
    all_door_positions = []
    
    # Main Episode Loop
    # ================
    # Run the specified number of episodes and collect trajectory data
    
    for episode_idx in range(args.num_episodes):
        print(f"\nRunning episode {episode_idx+1}/{args.num_episodes}")
        
        # Generate episode-specific seed for controlled randomness
        episode_seed = args.seed + episode_idx
        
        # Handle starting position selection
        custom_start_pos = None
        custom_start_dir = None
        
        if starting_positions:
            # Distribute episodes evenly across available starting positions
            pos_index = episode_idx % len(starting_positions)
            custom_start_pos, custom_start_dir = starting_positions[pos_index]
            room_name = get_room_from_position((custom_start_pos[0], custom_start_pos[2]))
            print(f"  Using starting position {pos_index}: {custom_start_pos} (in {room_name})")
        
        # Environment Reset and Initialization
        # ===================================
        try:
            # Reset environment with episode seed
            if use_parallel:
                observations, _ = env.reset(seed=episode_seed)
            else:
                observations, _ = env.reset(seed=episode_seed)
            
            # Apply custom starting position if specified
            if custom_start_pos is not None and custom_start_dir is not None:
                if hasattr(env, 'world') and hasattr(env.world, 'agent'):
                    env.world.place_entity(env.world.agent, pos=custom_start_pos, dir=custom_start_dir)
                    # Update navigator start position for reward calculations
                    env.world.navigator_start_pos = custom_start_pos.copy()
                    print(f"  Placed agent at position: {custom_start_pos}")
        except Exception as e:
            print(f"Error resetting environment: {e}")
            continue
        
        # Door Controller Configuration
        # ============================
        # Force door controller action if requested
        if args.force_door_action:
            if hasattr(env, 'door_can_act'):
                env.door_can_act = True
            if hasattr(env, 'world') and hasattr(env.world, 'door_can_act'):
                env.world.door_can_act = True
            print("  Forcing door controller to be able to act this episode")
        
        # Episode Data Initialization
        # ==========================
        current_path = []  # Path for this episode
        
        # Record initial door positions
        initial_door_positions = {}
        world_obj = env.world if hasattr(env, 'world') else env
        if hasattr(world_obj, "individual_door_positions"):
            initial_door_positions = world_obj.individual_door_positions.copy()
            print(f"  Initial door positions: {', '.join([f'{k}: {v:.2f}' for k, v in initial_door_positions.items()])}")
        elif hasattr(world_obj, "door_position"):
            # Handle single door position case
            initial_door_positions = {"roomA": world_obj.door_position, 
                                     "roomB": world_obj.door_position, 
                                     "roomC": world_obj.door_position}
            print(f"  Initial door position: {world_obj.door_position:.2f}")
        
        # Track door position changes during episode
        final_door_positions = initial_door_positions.copy()
            
        # Record starting position and initialize path
        start_position = None
        if hasattr(world_obj, 'agent'):
            # Ensure navigator start position is properly set
            if not hasattr(world_obj, 'navigator_start_pos') or world_obj.navigator_start_pos is None:
                world_obj.navigator_start_pos = world_obj.agent.pos.copy()
            current_path.append(list(world_obj.agent.pos))
            start_position = (world_obj.agent.pos[0], world_obj.agent.pos[2])
            print(f"  Starting position: ({start_position[0]:.2f}, {start_position[1]:.2f})")
            
        # Episode Execution Tracking
        episode_successful = False
        door_controller_acted = False
        
        try:
            # Episode Step Loop
            # ================
            # Execute episode steps with trained agents until completion
            
            done = False
            
            if use_parallel:
                # Parallel Environment Execution
                # =============================
                # All agents act simultaneously in each step
                
                while not done:
                    # Action Selection for All Agents
                    actions = {}
                    
                    for agent_id in observations:
                        if agent_id in rl_modules:
                            # Use trained neural network to select action
                            torch_obs = torch.FloatTensor([observations[agent_id]])
                            
                            # Forward pass through the neural network
                            fwd_outputs = rl_modules[agent_id].forward_inference({"obs": torch_obs})
                            action_dist_inputs = fwd_outputs["action_dist_inputs"]
                            action_dist_class = rl_modules[agent_id].get_inference_action_dist_cls()
                            action_dist = action_dist_class.from_logits(action_dist_inputs)
                            action = action_dist.sample()[0].cpu().numpy()
                            actions[agent_id] = action
                    
                    # Execute all actions simultaneously
                    observations, rewards, terminations, truncations, infos = env.step(actions)

                    # Track door controller actions and position changes
                    if 'door_controller' in infos and infos['door_controller'].get('door_acted', False):
                        door_controller_acted = True
                        # Log door action details when it actually occurs
                        if 'door_controller' in actions:
                            action_val = actions['door_controller'] 
                            if isinstance(action_val, (list, np.ndarray)) and len(action_val) >= 3:
                                print(f"  Door controller ACTED - {int(action_val[0])}, {int(action_val[1])}, {int(action_val[2])}")
                            else:
                                action_num = int(action_val[0]) if isinstance(action_val, (list, np.ndarray)) else int(action_val)
                                print(f"  Door controller ACTED: {action_num}")
                        
                        # Update final door positions with new values
                        if 'door_positions' in infos['door_controller']:
                            final_door_positions = infos['door_controller']['door_positions'].copy()
                    
                    # Check episode termination conditions
                    done = terminations.get('__all__', False) or truncations.get('__all__', False)
                    
                    # Check for successful episode completion
                    if 'navigator' in infos and infos['navigator'].get('is_successful', False):
                        episode_successful = True
                    
                    # Record navigator position for trajectory
                    if hasattr(world_obj, "agent"):
                        current_path.append(list(world_obj.agent.pos))
            else:
                # AEC Environment Execution
                # ========================
                # Agents act in turn-based fashion
                
                while not done:
                    # Get the current active agent
                    try:
                        agent_id = env.agent_selection
                    except:
                        print("Error getting agent_selection")
                        break
                    
                    # Handle terminated or truncated agents
                    try:
                        is_terminated = env.terminations.get(agent_id, False)
                        is_truncated = env.truncations.get(agent_id, False)
                        
                        if is_terminated or is_truncated:
                            try:
                                # Check if navigator reached goal (terminated but not truncated)
                                if agent_id == "navigator" and is_terminated and not is_truncated:
                                    episode_successful = True
                                    
                                env.step(None)  # Skip terminated agent
                                # Check if all agents are done
                                all_done = all(env.terminations.get(a, False) or env.truncations.get(a, False) 
                                            for a in env.possible_agents)
                                if all_done:
                                    done = True
                            except Exception as e:
                                print(f"Error in step with None: {e}")
                                done = True
                            continue
                    except Exception as e:
                        print(f"Error checking termination: {e}")
                        break
                    
                    # Get observation and compute action for active agent
                    try:
                        agent_obs = env.observe(agent_id)
                        
                        if agent_id in rl_modules:
                            # Use trained neural network for action selection
                            torch_obs = torch.FloatTensor([agent_obs])
                            
                            # Forward pass through neural network
                            fwd_outputs = rl_modules[agent_id].forward_inference({"obs": torch_obs})
                            action_dist_inputs = fwd_outputs["action_dist_inputs"]
                            action_dist_class = rl_modules[agent_id].get_inference_action_dist_cls()
                            action_dist = action_dist_class.from_logits(action_dist_inputs)
                            action = action_dist.sample()[0].cpu().numpy()
                        else:
                            # Fallback to random action if neural network not available
                            action = env.action_space(agent_id).sample()
                    except Exception as e:
                        print(f"Error computing action: {e}")
                        done = True
                        continue
                    
                    # Execute action and track results
                    try:
                        # Track door controller actions and position changes
                        if agent_id == "door_controller":
                            before_positions = None
                            
                            # Capture door positions before action
                            if hasattr(world_obj, "individual_door_positions"):
                                before_positions = world_obj.individual_door_positions.copy()
                            elif hasattr(world_obj, "door_position"):
                                before_positions = {"door": world_obj.door_position}
                                
                        # Execute the action
                        env.step(action)
                        
                        # Check for door position changes after action
                        if agent_id == "door_controller":
                            after_positions = None
                            
                            # Capture door positions after action
                            if hasattr(world_obj, "individual_door_positions"):
                                after_positions = world_obj.individual_door_positions.copy()
                                # Update final positions for analysis
                                final_door_positions = after_positions.copy()
                            elif hasattr(world_obj, "door_position"):
                                after_positions = {"door": world_obj.door_position}
                                # Update final positions for single door case
                                final_door_positions = {"roomA": world_obj.door_position, 
                                                      "roomB": world_obj.door_position, 
                                                      "roomC": world_obj.door_position}
                            
                            # Detect actual position changes
                            positions_changed = False
                            if before_positions and after_positions:
                                for key in before_positions:
                                    if key in after_positions and abs(before_positions[key] - after_positions[key]) > 0.001:
                                        positions_changed = True
                                        break
                            
                            if positions_changed:
                                door_controller_acted = True
                                print(f"  Door controller changed positions to: {', '.join([f'{k}: {v:.2f}' for k, v in after_positions.items()])}")
                        
                        # Record navigator position for trajectory
                        if agent_id == "navigator" and hasattr(world_obj, "agent"):
                            current_path.append(list(world_obj.agent.pos))
                            
                        # Check for episode success in environment info
                        if hasattr(env, 'infos'):
                            for a, info in env.infos.items():
                                if isinstance(info, dict) and info.get('is_successful', False):
                                    episode_successful = True
                            
                    except KeyError as e:
                        # Handle agent removal gracefully
                        print(f"KeyError in step: {e} - this may be normal if an agent was removed")
                        done = True
                    except Exception as e:
                        print(f"Error in step: {e}")
                        done = True
                    
                    # Check for overall episode completion
                    try:
                        all_done = all(env.terminations.get(a, False) or env.truncations.get(a, False) 
                                    for a in env.possible_agents)
                        if all_done:
                            done = True
                    except Exception as e:
                        print(f"Error checking overall termination: {e}")
                        done = True
        
        except Exception as e:
            print(f"Error during episode: {e}")
        
        # Episode Data Recording and Analysis
        # ==================================
        
        # Store door positions for analysis
        for room, pos in final_door_positions.items():
            all_door_positions.append((room, pos))
        
        # Calculate episode statistics
        num_steps = len(current_path)
        
        # Check if episode meets step limit criteria
        meets_step_criteria = True
        if args.max_steps is not None:
            meets_step_criteria = num_steps < args.max_steps
            
        # Create comprehensive episode data record
        episode_data = {
            "episode": episode_idx + 1,
            "door_positions": final_door_positions,
            "door_controller_acted": door_controller_acted,
            "start_position": start_position,
            "num_steps": num_steps,
            "is_successful": episode_successful,
            "meets_step_criteria": meets_step_criteria
        }
        all_episodes_data.append(episode_data)
        
        # Add to successful episodes data if successful
        if episode_successful:
            successful_episodes_data.append(episode_data)
            print(f"Episode {episode_idx+1} was successful with {num_steps} steps! Door controller acted: {door_controller_acted}")
        else:
            print(f"Episode {episode_idx+1} was unsuccessful with {num_steps} steps.")
        
        # Add path to visualization based on filtering criteria
        if (args.include_all_paths or episode_successful) and (meets_step_criteria or args.max_steps is None):
            # Include this path in the visualization
            all_paths.append(current_path)
            all_success_statuses.append(episode_successful)
            if episode_successful:
                print(f"  Adding successful path to visualization (steps: {num_steps}{' < max: '+str(args.max_steps) if args.max_steps else ''})")
            else:
                print(f"  Adding unsuccessful path to visualization (steps: {num_steps}{' < max: '+str(args.max_steps) if args.max_steps else ''})")
        else:
            # Path doesn't meet criteria for visualization
            if not meets_step_criteria:
                print(f"  Not adding path to visualization (steps: {num_steps} ≥ max: {args.max_steps})")
            elif not episode_successful and not args.include_all_paths:
                print(f"  Not adding unsuccessful path (include_all_paths=False)")
    
    # Room-Specific Analysis
    # =====================
    # Analyze door positions based on the room where episodes started
    
    # Dictionary to track door positions for successful episodes by starting room
    room_specific_successful_doors = {
        'roomA': [],
        'roomB': [],
        'roomC': []
    }
    
    # Data Export and Reporting
    # ========================
    # Save comprehensive episode data for further analysis
    
    if all_episodes_data:
        output_dir = args.output_dir
        all_episodes_file = os.path.join(output_dir, "all_episodes_data.txt")
        
        # Save detailed text report of all episodes
        with open(all_episodes_file, 'w') as f:
            f.write("All Episodes Data:\n")
            f.write("----------------\n\n")
            
            for data in all_episodes_data:
                f.write(f"Episode: {data['episode']}\n")
                f.write(f"  Door Positions:\n")
                for room, pos in data['door_positions'].items():
                    f.write(f"    {room}: {pos:.4f}\n")
                f.write(f"  Door Controller Acted: {data['door_controller_acted']}\n")
                f.write(f"  Start Position: ({data['start_position'][0]:.4f}, {data['start_position'][1]:.4f})\n")
                f.write(f"  Path Length: {data['num_steps']} steps\n")
                f.write(f"  Successful: {data['is_successful']}\n\n")
                
        print(f"Saved data for all {len(all_episodes_data)} episodes to {all_episodes_file}")
    
    # Export successful episodes data (backward compatibility)
    if successful_episodes_data:
        output_dir = args.output_dir
        door_positions_file = os.path.join(output_dir, "successful_episodes_data.txt")
        
        # Save detailed report of successful episodes only
        with open(door_positions_file, 'w') as f:
            f.write("Successful Episodes Data:\n")
            f.write("------------------------\n\n")
            
            for data in successful_episodes_data:
                f.write(f"Episode: {data['episode']}\n")
                f.write(f"  Door Positions:\n")
                for room, pos in data['door_positions'].items():
                    f.write(f"    {room}: {pos:.4f}\n")
                f.write(f"  Door Controller Acted: {data['door_controller_acted']}\n")
                f.write(f"  Start Position: ({data['start_position'][0]:.4f}, {data['start_position'][1]:.4f})\n")
                f.write(f"  Path Length: {data['num_steps']} steps\n\n")
                
        print(f"Saved door positions for {len(successful_episodes_data)} successful episodes to {door_positions_file}")
        
        # Generate door position frequency analysis
        door_summary_file = os.path.join(output_dir, "door_position_summary.txt")
        unique_door_positions = {}
        
        # Count occurrences of each door position by room
        for room, pos in all_door_positions:
            rounded_pos = round(pos, 1)  # Round to reduce noise from minor variations
            if room not in unique_door_positions:
                unique_door_positions[room] = {}
            if rounded_pos not in unique_door_positions[room]:
                unique_door_positions[room][rounded_pos] = 0
            unique_door_positions[room][rounded_pos] += 1
            
        # Save door position frequency summary
        with open(door_summary_file, 'w') as f:
            f.write("Door Position Summary:\n")
            f.write("---------------------\n\n")
            
            for room in sorted(unique_door_positions.keys()):
                f.write(f"{room} Door Positions:\n")
                for pos, count in sorted(unique_door_positions[room].items()):
                    f.write(f"  Position {pos:.1f}: {count} instances\n")
                f.write("\n")
                
        print(f"Saved door position summary to {door_summary_file}")
    
    # Close environment to free resources
    env.close()
    
    # Room-Specific Door Position Analysis
    # ===================================
    # Analyze door positions based on the room where successful episodes started
    
    for data in successful_episodes_data:
        # Get starting position coordinates
        start_pos = data['start_position']
        
        # Determine which room the episode started in
        start_room = get_room_from_position(start_pos)
        
        # Track door positions for the starting room only
        if start_room in room_specific_successful_doors:
            for room, pos in data['door_positions'].items():
                # Only record the door position for the room where the episode started
                if room == start_room:
                    room_specific_successful_doors[start_room].append(pos)
    
    # Generate room-specific analysis report
    room_summary_file = os.path.join(output_dir, "room_specific_door_positions.txt")
    with open(room_summary_file, 'w') as f:
        f.write("Room-Specific Successful Door Positions:\n")
        f.write("----------------------------------------\n\n")
        
        for room, positions in room_specific_successful_doors.items():
            f.write(f"{room} Successful Door Positions:\n")
            if positions:
                # Calculate statistical measures
                f.write(f"  Total successful episodes: {len(positions)}\n")
                f.write(f"  Mean door position: {np.mean(positions):.4f}\n")
                f.write(f"  Median door position: {np.median(positions):.4f}\n")
                f.write(f"  Min door position: {np.min(positions):.4f}\n")
                f.write(f"  Max door position: {np.max(positions):.4f}\n")
                
                # Generate frequency distribution
                unique_pos, counts = np.unique(np.round(positions, 1), return_counts=True)
                f.write("  Frequency Distribution:\n")
                for pos, count in zip(unique_pos, counts):
                    f.write(f"    Position {pos:.1f}: {count} instances\n")
            else:
                f.write("  No successful episodes in this room\n")
            f.write("\n")
    
    # Export machine-readable JSON version of room-specific data
    room_summary_json = os.path.join(output_dir, "room_specific_door_positions.json")
    with open(room_summary_json, 'w') as f:
        json.dump(room_specific_successful_doors, f, indent=2)
    
    # JSON Export for Programmatic Access
    # ==================================
    # Save episode data in JSON format for easier computational analysis
    
    try:
        # Export all episodes data as JSON
        all_episodes_json = os.path.join(output_dir, "all_episodes_data.json")
        with open(all_episodes_json, 'w') as f:
            json.dump(all_episodes_data, f, indent=2)
        print(f"Saved JSON version of all episodes data to {all_episodes_json}")
        
        # Export successful episodes data as JSON
        successful_data_json = os.path.join(output_dir, "successful_episodes_data.json")
        with open(successful_data_json, 'w') as f:
            json.dump(successful_episodes_data, f, indent=2)
        print(f"Saved JSON version of successful episodes data to {successful_data_json}")
    except Exception as e:
        print(f"Error creating JSON files: {e}")
    
    # Optional Visualization Generation
    # ================================
    # Generate additional plots if visualization module is available
    
    try:
        from escape_room.utils.visualization import plot_door_positions
        plot_door_positions(room_specific_successful_doors, 
                            filename=os.path.join(output_dir, "room_specific_door_distributions.png"))
    except ImportError:
        print("Visualization module not available. Skipping door position plot.")
    
    return all_paths, all_success_statuses


def main():
    """
    Main function that orchestrates the trajectory visualization process.
    Handles argument parsing, model loading, episode execution, and result generation.
    """
    # Parse command line arguments
    args = parse_args()
    
    # Random Seed Configuration
    # ========================
    # Set up randomization for episode variety while maintaining some reproducibility
    
    import time
    if args.seed is None:
        # If no seed provided, use current time for true randomness
        random_seed = int(time.time() * 1000) % (2**31-1)
        print(f"No seed provided, using time-based random seed: {random_seed}")
    else:
        # If seed provided, add time-based offset for variation between runs
        time_offset = int(time.time()) % 1000
        random_seed = (args.seed + time_offset) % (2**31-1)
        print(f"Base seed {args.seed} with time offset {time_offset}, final seed: {random_seed}")
    
    # Configure PyTorch and numpy for controlled randomness
    torch.use_deterministic_algorithms(False)  # Allow non-deterministic for more variety
    np.random.seed(random_seed)
    import random
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    
    # Update arguments with the computed random seed
    args.seed = random_seed
    
    # Environment Configuration
    # ========================
    # Load and customize environment configuration
    
    config = get_default_config()
    config["seed"] = args.seed
    
    # Apply configuration overrides based on arguments
    if args.random_placement:
        config["random_agent_placement"] = True
    else:
        # Force fixed placement for systematic testing
        config["random_agent_placement"] = False
        
    # Apply door movement frequency override
    if args.door_move_frequency is not None:
        config["door_move_frequency"] = args.door_move_frequency
    
    # Create output directory for results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Model Loading
    # ============
    # Load trained neural networks from checkpoint
    
    rl_modules = load_rl_modules(args.checkpoint)
    
    # Print configuration summary
    print("\nEnvironment Configuration:")
    print(f"  Environment type: {'Parallel' if config.get('parallel_env', True) else 'AEC'}")
    print(f"  Random agent placement: {config['random_agent_placement']}")
    print(f"  Door move frequency: Every {config['door_move_frequency']} episodes")
    print(f"  Force door action: {args.force_door_action}")
    print(f"  Terminal location: {config['terminal_location']}")
    
    # Episode Execution and Data Collection
    # ====================================
    # Run episodes with trained agents and collect trajectory data
    
    all_paths, all_success_statuses = run_episodes_and_collect_paths(rl_modules, config, args)
    
    # Iteration Number Extraction
    # ===========================
    # Attempt to extract training iteration from checkpoint path for labeling
    
    iteration = args.iteration
    if iteration is None and args.checkpoint:
        # Try to extract iteration number from checkpoint path
        try:
            import re
            # Look for common patterns like "checkpoint_iter_100.pkl"
            match = re.search(r'iter_(\d+)', args.checkpoint)
            if match:
                iteration = match.group(1)
            else:
                # Fallback: use directory name if it's a number
                basename = os.path.basename(os.path.dirname(args.checkpoint) if os.path.isfile(args.checkpoint) else args.checkpoint)
                if basename.isdigit():
                    iteration = basename
        except:
            pass
    
    # Visualization Generation
    # =======================
    # Create and save the trajectory visualization
    
    # Generate appropriate filename suffix based on configuration
    suffix = '_all' if args.include_all_paths else '_successful'
    output_file = os.path.join(args.output_dir, f"paths{suffix}_{'iter_'+str(iteration) if iteration else 'no_iter'}{f'_max{args.max_steps}steps' if args.max_steps else ''}.png")
    
    # Create the visualization
    create_path_visualization(all_paths, all_success_statuses, config, output_file, iteration, args.max_steps, args.include_all_paths)
    
    # Final Results Summary
    # ====================
    print(f"Successfully created path visualization at: {output_file}")
    print(f"Collected {len(all_paths)} path trajectories ({sum(all_success_statuses)} successful, {len(all_paths) - sum(all_success_statuses)} unsuccessful)")

# Script Entry Point
# =================
# Execute main function when script is run directly

if __name__ == "__main__":
    main()