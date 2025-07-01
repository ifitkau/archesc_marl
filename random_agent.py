# Random Agent Test Script for Escape Room Environment
# =====================================================
# This script tests the escape room environment by running random agents and provides
# comprehensive analysis of their performance, reward components, and behavioral patterns.
# It's useful for debugging, environment validation, and baseline performance measurement.

"""
Enhanced test script for running the escape room environment with random actions
Compatible with both AEC and Parallel environments, with improved debugging and reward tracking
"""

# Standard libraries for argument parsing, numerical operations, and system utilities
import argparse
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import defaultdict
import os

# Import custom escape room environment components
from escape_room.envs.escape_room_env import make_escape_room_env
from escape_room.config.default_config import get_default_config
from escape_room.utils.visualization import save_paths, create_color_matched_plot, plot_rewards, plot_success_rate, plot_door_positions

def parse_args():
    """
    Parse command line arguments for configuring the test run.
    
    Returns:
        argparse.Namespace: Parsed command line arguments with test configuration
    """
    parser = argparse.ArgumentParser(description="Run escape room environment with random actions")
    
    # Episode configuration
    parser.add_argument("--num-episodes", type=int, default=30,
                        help="Number of episodes to run")
    
    # Visualization and rendering options
    parser.add_argument("--render", action="store_true", default=True,
                        help="Render the environment")
    parser.add_argument("--delay", type=float, default=0.01,
                        help="Delay between steps for rendering (seconds)")
    
    # Output and logging configuration
    parser.add_argument("--output-dir", type=str, default="random_agent_results",
                        help="Directory to save results")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="Print detailed information")
    
    # Environment configuration
    parser.add_argument("--parallel", action="store_true", default=True,
                        help="Use parallel environment (default: True)")
    
    return parser.parse_args()

def decode_door_action(action, num_positions, position_min, position_max):
    """
    Convert a discrete door action to actual door positions.
    Handles both multi-discrete actions (one per room) and legacy single discrete actions.
    
    Args:
        action: Action array [roomA, roomB, roomC] or single integer
        num_positions: Number of discrete positions available
        position_min: Minimum door position value
        position_max: Maximum door position value
        
    Returns:
        dict: Dictionary mapping room names to decoded door positions
    """
    position_range = position_max - position_min
    
    # Handle multi-discrete action (array with action for each room)
    if isinstance(action, (list, np.ndarray)) and len(action) == 3:
        positions = {}
        room_names = ['roomA', 'roomB', 'roomC']
        
        for room_name, room_action in zip(room_names, action):
            if room_action == 0:
                # Action 0 means "keep current position"
                positions[room_name] = "keep"
            else:
                # Convert discrete action to continuous position
                # Subtract 1 to make it 0-indexed for calculation
                position_idx = room_action - 1
                positions[room_name] = position_min + (position_idx / (num_positions - 1)) * position_range
        
        return positions
    
    # Handle single discrete action (for backward compatibility)
    else:
        if action == 0:
            # Action 0 means keep all doors in current position
            return {"roomA": "keep", "roomB": "keep", "roomC": "keep"}
        else:
            # Legacy calculation: decode single action into three room actions
            room_names = ['roomA', 'roomB', 'roomC']
            room_actions = [
                (action - 1) // (num_positions ** 2),        # First room action
                ((action - 1) % (num_positions ** 2)) // num_positions,  # Second room action
                (action - 1) % num_positions                 # Third room action
            ]
            
            positions = {}
            for room_name, room_action in zip(room_names, room_actions):
                positions[room_name] = position_min + (room_action / (num_positions - 1)) * position_range
            
            return positions

# Environment Component Access Functions
# =====================================
# These functions help extract components from the environment's nested structure
# Different environment wrappers may nest components at different levels

def get_navigator_agent(env):
    """
    Extract the navigator agent from the environment structure.
    Tries multiple access paths to handle different environment wrapper configurations.
    
    Args:
        env: The environment instance
        
    Returns:
        Navigator agent object or None if not found
    """
    # Try various paths to find the navigator agent
    if hasattr(env, 'env') and hasattr(env.env, 'navigator'):
        return env.env.navigator
    elif hasattr(env, 'env') and hasattr(env.env, 'env') and hasattr(env.env.env, 'navigator'):
        return env.env.env.navigator
    elif hasattr(env, 'navigator'):
        return env.navigator
    return None

def get_world_from_env(env):
    """
    Extract the world object from the environment structure.
    The world object contains the physical simulation and agent states.
    
    Args:
        env: The environment instance
        
    Returns:
        World object or None if not found
    """
    # Try various paths to find the world object
    if hasattr(env, 'env') and hasattr(env.env, 'world'):
        return env.env.world
    elif hasattr(env, 'env') and hasattr(env.env, 'env') and hasattr(env.env.env, 'world'):
        return env.env.env.world
    elif hasattr(env, 'world'):
        return env.world
    return None

def get_door_controller(env):
    """
    Extract the door controller from the environment structure.
    The door controller manages door positions and states.
    
    Args:
        env: The environment instance
        
    Returns:
        Door controller object or None if not found
    """
    # Try various paths to find the door controller
    if hasattr(env, 'env') and hasattr(env.env, 'door_controller'):
        return env.env.door_controller
    elif hasattr(env, 'env') and hasattr(env.env, 'env') and hasattr(env.env.env, 'door_controller'):
        return env.env.env.door_controller
    elif hasattr(env, 'door_controller'):
        return env.door_controller
    return None

def run_random_agent():
    """
    Main function that runs the random agent test.
    Creates environment, runs episodes with random actions, and generates analysis reports.
    """
    args = parse_args()
    
    # Create output directory for results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed for reproducibility if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"Using random seed: {args.seed}")
    
    # Environment Configuration
    # ========================
    config = get_default_config()
    config['debug_mode'] = True  # Enable debug mode for detailed logging
    
    # Set environment to parallel or AEC mode based on argument
    config['parallel_env'] = args.parallel
    
    # Important: Let the environment handle door movement naturally
    # Don't override the door movement frequency - use what's in config
    print(f"Using door_move_frequency from config: {config['door_move_frequency']}")
    
    # Add the new reward scale for facing door (if not already present)
    config['navigator_reward_scales']['reward_facing_door_scale'] = 1.0
    
    # Print comprehensive configuration information
    print("Environment Configuration:")
    print(f"  Environment type: {'Parallel' if args.parallel else 'AEC'}")
    print(f"  Door move frequency: Every {config['door_move_frequency']} episodes")
    print(f"  Door position range: {config['door_position_min']} to {config['door_position_max']}")
    print(f"  Discrete door positions: {config['discrete_door_positions']}")
    print(f"  Max episode steps: {config['max_episode_steps']}")
    print(f"  Terminal location: {config['terminal_location']}")
    print(f"  Door controller reward scales: {config['door_controller_reward_scales']}")
    print(f"  Navigator reward scales: {config['navigator_reward_scales']}")
    
    # Create Environment
    # =================
    render_mode = "human" if args.render else None
    env = make_escape_room_env(config=config, render_mode=render_mode, view="top")
    
    # Check if we're using parallel environment
    is_parallel = config['parallel_env']
    print(f"Using {'Parallel' if is_parallel else 'AEC'} environment")
    
    # Rendering Setup
    # ==============
    # Ensure rendering is properly enabled and set up pause functionality
    if args.render:
        world = get_world_from_env(env)
        if world and hasattr(world, 'render_mode'):
            world.render_mode = "human"
            
        # Set up keyboard controls for pausing simulation
        if world and hasattr(world, 'window') and world.window is not None:
            from pyglet.window import key
            
            @world.window.event
            def on_key_press(symbol, modifiers):
                if symbol == key.SPACE:
                    world.paused = not world.paused
                    print(f"Simulation {'PAUSED' if world.paused else 'RESUMED'}")
            
            print("Press SPACE to pause/resume the simulation")
        elif world:
            # For cases where window isn't created yet, set up later
            world._setup_pause_controls = True
    
    # Data Tracking Initialization
    # ============================
    # Initialize variables to track performance metrics and analysis data
    
    # Success and performance tracking
    success_count = 0
    total_steps = []
    door_positions = defaultdict(list)  # Track door positions by room
    
    # Agent reward tracking
    agent_rewards = {"navigator": [], "door_controller": []}
    
    # Detailed reward component tracking
    reward_components = defaultdict(list)
    
    # Step-by-step reward tracking for detailed analysis
    door_reward_by_step = []      # Door facing rewards by step
    terminal_reward_by_step = []  # Terminal facing rewards by step
    distance_rewards_this_episode = []  # Distance-based rewards
    
    # Episode outcome tracking
    episode_successes = []
    
    # Spatial analysis tracking
    starting_positions = []        # Navigator starting positions
    door_position_by_episode = []  # Door positions for each episode
    starting_rooms = []           # Starting room for each episode

    # Main Episode Loop
    # ================
    # Run the specified number of episodes with random actions
    
    for episode in range(args.num_episodes):
        print(f"\n=== Episode {episode+1}/{args.num_episodes} ===")
        
        # Reset environment with episode-specific seed if provided
        episode_seed = args.seed + episode if args.seed is not None else None
        observations, info = env.reset(seed=episode_seed)
        
        # Log initial observations for debugging
        if args.verbose:
            print("Initial observations:")
            for agent, obs in observations.items():
                print(f"  {agent}: shape={obs.shape}")
        
        # Extract environment components for this episode
        world = get_world_from_env(env)
        door_controller = get_door_controller(env)
        navigator_agent = get_navigator_agent(env)
        
        # Navigator Starting Position Analysis
        # ===================================
        starting_pos = None
        current_room = None
        
        if world and hasattr(world, 'agent'):
            starting_pos = world.agent.pos
            starting_positions.append((starting_pos[0], starting_pos[2]))
            
            # Determine which room the navigator starts in
            if hasattr(world, '_get_current_room'):
                current_room = world._get_current_room()
                starting_rooms.append(current_room)
                
            if args.verbose:
                print(f"Navigator starting position: ({starting_pos[0]:.2f}, {starting_pos[2]:.2f}) in room {current_room}")
        
        # Door Controller State Analysis
        # =============================
        # Check if door controller can act this episode (based on door_move_frequency)
        door_can_act = False
        if hasattr(env.env, 'door_can_act'):
            door_can_act = env.env.door_can_act
        elif world and hasattr(world, 'door_can_act'):
            door_can_act = world.door_can_act
        
        print(f"Door can act this episode: {door_can_act} (episode {episode+1} % {config['door_move_frequency']} = {(episode+1) % config['door_move_frequency']})")
        
        # Let the environment handle door setup naturally - don't override it!
        if door_controller is not None:
            print("Door controller found - letting environment handle door setup")
            print("Current door positions:")
            for room, pos in door_controller.door_positions.items():
                print(f"  {room}: {pos:.4f}")
            
            # Store the starting room for debugging purposes only
            if current_room:
                door_controller.starting_room = current_room
                print(f"Navigator starting room: {current_room}")
        else:
            print("Warning: Door controller not found!")
        
        # Environment State Validation
        # ===========================
        print(f"Door can act this episode: {door_can_act}")
        if hasattr(world, 'portals'):
            print(f"Initial portals in world: {len(world.portals)}")
        else:
            print("No portals attribute found in world")
        
        # Episode-Level Tracking Initialization
        # ====================================
        episode_reward = {"navigator": 0, "door_controller": 0}
        step_count = 0
        terminated = False
        truncated = False
        episode_reward_components = defaultdict(float)
        
        # Track if door controller has acted this episode
        door_controller_acted = False
        door_agent_actions = {}
        
        # Reset reward tracking for this episode
        door_rewards_this_episode = []
        terminal_rewards_this_episode = []
        distance_rewards_this_episode = []

        if navigator_agent is None:
            print("Warning: Could not find navigator agent")
        
        # Initialize door facing reward tracking
        previous_distance_to_door = None
        
        # Episode Step Loop
        # ================
        # Handle parallel vs AEC environments differently due to different APIs
        
        while not (terminated or truncated):
            # Action Selection
            # ===============
            # Take random actions for all active agents
            actions = {}
            for agent in observations.keys():
                if is_parallel:
                    # For parallel environment, get action space from env
                    action_space = env.action_space(agent)
                else:
                    # For AEC environment, get action space from env
                    action_space = env.action_space(agent)
                actions[agent] = action_space.sample()
                
                # Save door controller actions for debugging
                if agent == "door_controller":
                    door_agent_actions = actions[agent]
            
            # Execute Actions
            # ==============
            # Different execution patterns for parallel vs AEC environments
            
            if is_parallel:
                # Parallel Environment Step
                # ========================
                # Pass all actions at once to the environment
                next_observations, rewards, terminations, truncations, infos = env.step(actions)
                
                # CRITICAL: Force visual update if doors moved
                # This ensures the visual representation matches the actual door state
                if 'door_controller' in actions and door_controller is not None:
                    # Check if door actually acted (moved) this step
                    door_info = infos.get('door_controller', {})
                    if door_info.get('door_acted', False):
                        print("Door moved - forcing visual update")
                        if world:
                            # Force regeneration of static geometry
                            if hasattr(world, '_gen_static_data'):
                                world._gen_static_data()
                            # Force re-rendering of static elements  
                            if hasattr(world, '_render_static'):
                                world._render_static()
                            # Clear window buffer for fresh render
                            if hasattr(world, 'window') and world.window is not None:
                                world.window.clear()
                        print("Visual update completed")

                # Check termination conditions for all agents
                terminated = terminations.get("__all__", False)
                truncated = truncations.get("__all__", False)
            else:
                # AEC Environment Step
                # ===================
                # Step through each agent individually (turn-based)
                
                # Get the current active agent
                current_agent = env.agent_selection if hasattr(env, 'agent_selection') else None
                
                if current_agent and current_agent in actions:
                    # Step with the action for the current agent
                    env.step(actions[current_agent])
                    
                    # CRITICAL: Force visual update if door controller acted
                    # Ensures visual consistency when doors move
                    if current_agent == 'door_controller':
                        door_info = env.infos.get('door_controller', {}) if hasattr(env, 'infos') else {}
                        if door_info.get('door_acted', False):
                            print("Door moved - forcing visual update")
                            if world:
                                # Force regeneration of static geometry
                                if hasattr(world, '_gen_static_data'):
                                    world._gen_static_data()
                                # Force re-rendering of static elements  
                                if hasattr(world, '_render_static'):
                                    world._render_static()
                                # Clear window buffer for fresh render
                                if hasattr(world, 'window') and world.window is not None:
                                    world.window.clear()
                            print("Visual update completed")
                    
                    # Collect updated state information for all agents
                    next_observations = {}
                    rewards = {}
                    terminations = {}
                    truncations = {}
                    infos = {}
                    
                    # Gather information from all agents
                    for agent in env.possible_agents:
                        if hasattr(env, 'terminations') and agent in env.terminations:
                            terminations[agent] = env.terminations[agent]
                        if hasattr(env, 'truncations') and agent in env.truncations:
                            truncations[agent] = env.truncations[agent]
                        if hasattr(env, 'rewards') and agent in env.rewards:
                            rewards[agent] = env.rewards[agent]
                        if hasattr(env, 'infos') and agent in env.infos:
                            infos[agent] = env.infos[agent]
                        
                        # Get observation for non-terminated agents
                        if not (terminations.get(agent, False) or truncations.get(agent, False)):
                            next_observations[agent] = env.observe(agent)
                    
                    # Check overall termination status
                    terminated = all(terminations.get(agent, False) for agent in env.possible_agents)
                    truncated = all(truncations.get(agent, False) for agent in env.possible_agents)
                else:
                    # No current agent or agent not in actions, break
                    break
            
            # Reward Analysis and Calculation
            # ==============================
            # Calculate detailed reward components for analysis
            
            if world is not None:
                agent = world.agent
                terminal_loc = world.terminal_location
                current_room = world._get_current_room() if hasattr(world, '_get_current_room') else None
                
                # Terminal Facing Reward Calculation
                # =================================
                # Calculate how well the agent is oriented toward the terminal
                
                # Vector from agent to terminal
                dx_terminal = terminal_loc[0] - agent.pos[0]
                dz_terminal = terminal_loc[1] - agent.pos[2]
                
                # Direction vector to terminal (with flipped z for correct orientation)
                terminal_direction_vector = np.array([dx_terminal, -dz_terminal])
                terminal_direction_length = np.linalg.norm(terminal_direction_vector)
                if terminal_direction_length > 0:
                    terminal_direction_vector = terminal_direction_vector / terminal_direction_length
                else:
                    terminal_direction_vector = np.array([1.0, 0.0])  # Default direction
                
                # Agent's forward direction vector
                agent_dir_vec = np.array([
                    np.cos(agent.dir),
                    np.sin(agent.dir)
                ])
                
                # Calculate dot product for terminal alignment
                terminal_dot_product = np.dot(terminal_direction_vector, agent_dir_vec)
                
                # Calculate terminal orientation reward
                terminal_orientation_reward = 0.0
                if terminal_dot_product < np.cos(np.pi / 6):  # Within 30 degrees
                    # Negative reward for not facing terminal (from original code)
                    terminal_orientation_reward = -0.1 * config['navigator_reward_scales']['reward_orientation_scale']
                
                # Track terminal reward for analysis
                terminal_rewards_this_episode.append(terminal_orientation_reward)
                
                # Door Facing Reward Calculation
                # =============================
                # Calculate how well the agent is oriented toward doors in the current room
                
                door_facing_reward = 0.0
                
                # Only calculate in rooms with doors
                if current_room in ['roomA', 'roomB', 'roomC']:
                    # Get door position for current room
                    door_pos = None
                    # Try different access paths to get door position
                    if door_controller is not None and hasattr(door_controller, 'door_positions'):
                        door_pos = door_controller.door_positions.get(current_room)
                    elif hasattr(world, 'individual_door_positions'):
                        door_pos = world.individual_door_positions.get(current_room)
                    elif hasattr(world, 'door_position'):
                        door_pos = world.door_position
                        
                    if door_pos is not None:
                        # Room boundaries for calculating absolute door positions
                        room_boundaries = {
                            'roomA': (0.0, 4.0),
                            'roomB': (4.2, 8.2),
                            'roomC': (8.4, 12.4)
                        }
                        
                        if current_room in room_boundaries:
                            room_min_x, _ = room_boundaries[current_room]
                            
                            # Calculate absolute door position in world coordinates
                            door_x = door_pos + room_min_x
                            door_z = 3.0  # All doors are at z=3.0
                            
                            # Calculate current distance to door
                            current_distance_to_door = np.sqrt(
                                (agent.pos[0] - door_x)**2 + 
                                (agent.pos[2] - door_z)**2
                            )
                            
                            # Calculate direction vector to door
                            door_direction_vector = np.array([
                                door_x - agent.pos[0],
                                -(door_z - agent.pos[2])  # FLIPPED z-component (same as terminal calc)
                            ])
                            
                            # Normalize door direction vector
                            door_direction_length = np.linalg.norm(door_direction_vector)
                            if door_direction_length > 0:
                                door_direction_vector = door_direction_vector / door_direction_length
                            else:
                                door_direction_vector = np.array([0.0, 1.0])  # Default if at same position
                            
                            # Calculate dot product between agent direction and door direction
                            door_dot_product = np.dot(door_direction_vector, agent_dir_vec)
                            
                            # Door approach reward (incremental reward for getting closer)
                            door_approach_reward = 0.0
                            if previous_distance_to_door is not None and current_distance_to_door < previous_distance_to_door:
                                door_approach_reward = 0.1 * config['navigator_reward_scales'].get('reward_door_approach_scale', 0.0)
                            
                            # Door facing reward (directional reward - currently commented out)
                            door_facing_reward = 0.0
                            #if door_dot_product < np.cos(np.pi / 9):  # Within 20 degrees
                                # Scale reward based on how closely aligned (1.0 is perfect)
                            #    door_facing_reward -= 0.1 * config['navigator_reward_scales'].get('reward_facing_door_scale', 1.0)
                            
                            # Detailed logging for debugging
                            if args.verbose and step_count % 1 == 0:
                                print(f"Step {step_count}: Room: {current_room}, Door at: ({door_x:.2f}, {door_z:.2f})")
                                print(f"  Agent position: ({agent.pos[0]:.2f}, {agent.pos[2]:.2f}), distance to door: {current_distance_to_door:.2f}")
                                print(f"  Door facing - dot product: {door_dot_product:.4f}, reward: {door_facing_reward:.4f}")
                                print(f"  Terminal facing - dot product: {terminal_dot_product:.4f}, reward: {terminal_orientation_reward:.4f}")
                            
                            # Store for next step comparison
                            previous_distance_to_door = current_distance_to_door
                            
                            # Track door facing reward
                            door_rewards_this_episode.append(door_facing_reward)
                else:
                    # No reward when not in a room with a door
                    door_rewards_this_episode.append(0.0)
                
                # Distance-Based Reward Calculation
                # ================================
                # Additional distance reward for terminal orientation
                
                reward_distance = 0.0
                if terminal_dot_product < np.cos(np.pi / 9):  # Within 20 degrees
                    # Reward for proper terminal orientation
                    reward_distance -= 0.1
                
                distance_rewards_this_episode.append(reward_distance)
                
                # Periodic detailed logging
                if args.verbose and step_count % 20 == 0:
                    print(f"Step {step_count}: Room: {current_room}")
                    print(f"  Agent position: ({agent.pos[0]:.2f}, {agent.pos[2]:.2f})")
                    print(f"  Terminal facing - dot product: {terminal_dot_product:.4f}, reward: {terminal_orientation_reward:.4f}")
                    print(f"  Terminal distance reward: {reward_distance:.4f}")
            
            # Update State
            # ===========
            observations = next_observations
            
            # Update cumulative reward tracking
            for agent, reward in rewards.items():
                episode_reward[agent] += reward
            
            # Rendering and Visualization
            # ==========================
            # Handle rendering with pause controls and visual updates
            
            if args.render:
                try:
                    # Set up pause controls if not already configured
                    if world and hasattr(world, '_setup_pause_controls') and world._setup_pause_controls:
                        if hasattr(world, 'window') and world.window is not None:
                            from pyglet.window import key
                            
                            @world.window.event
                            def on_key_press(symbol, modifiers):
                                if symbol == key.SPACE:
                                    world.paused = not world.paused
                                    print(f"Simulation {'PAUSED' if world.paused else 'RESUMED'}")
                            
                            world._setup_pause_controls = False
                            print("Pause controls set up - Press SPACE to pause/resume")
                    
                    # Render the environment normally - let the environment handle door updates
                    env.render()
                except Exception as e:
                    print(f"Render error: {e}")
                
                # Handle environment pausing during rendering
                if world and hasattr(world, 'paused'):
                    while world.paused:
                        try:
                            env.render()
                        except:
                            pass
                        time.sleep(0.01)
                
                # Add delay between steps for better visualization
                time.sleep(args.delay)
            
            step_count += 1
            
            # Periodic status logging
            if step_count % 20 == 0 and args.verbose and world is not None:
                if hasattr(world, 'agent'):
                    agent_pos = world.agent.pos
                    terminal_loc = world.terminal_location
                    distance = np.linalg.norm(
                        np.array([agent_pos[0], agent_pos[2]]) - 
                        np.array([terminal_loc[0], terminal_loc[1]])
                    )
                    print(f"Step {step_count}: Navigator at ({agent_pos[0]:.2f}, {agent_pos[2]:.2f}), "
                        f"distance to terminal: {distance:.2f}")
        
        # End of Episode Processing
        # ========================
        # Store tracking data and analyze episode outcome
        
        # Store reward tracking data for this episode
        door_reward_by_step.append(door_rewards_this_episode)
        terminal_reward_by_step.append(terminal_rewards_this_episode)
        
        # Episode Success Analysis
        # =======================
        success = False
        
        # Check if the episode was successful based on environment feedback
        if "navigator" in infos:
            success = infos["navigator"].get("is_successful", False)
        
        # Update success tracking
        if success:
            success_count += 1
            print(f"Episode {episode+1} SUCCESS ({step_count} steps)")
        else:
            print(f"Episode {episode+1} FAILURE ({step_count} steps)")
        
        episode_successes.append(success)
        total_steps.append(step_count)
        
        # Store reward data for analysis
        for agent, reward in episode_reward.items():
            agent_rewards[agent].append(reward)
            
        # Store detailed reward components
        for component, value in episode_reward_components.items():
            reward_components[component].append(value)
        
        # Print episode summary
        print(f"Episode rewards:")
        for agent, reward in episode_reward.items():
            print(f"  {agent}: {reward:.2f}")
        
        # Store door positions for this episode
        if door_controller is not None:
            final_door_positions = door_controller.door_positions.copy()
            for room, pos in final_door_positions.items():
                door_positions[room].append(pos)
            
            door_position_by_episode.append(final_door_positions)
    
    # Post-Episode Analysis and Reporting
    # ==================================
    # Calculate overall statistics and generate comprehensive reports
    
    print("\n=== Overall Statistics ===")
    print(f"Success rate: {success_count}/{args.num_episodes} ({100*success_count/args.num_episodes:.1f}%)")
    print(f"Average episode length: {np.mean(total_steps):.1f} steps")
    
    # Agent Performance Analysis
    for agent, rewards in agent_rewards.items():
        print(f"{agent} average reward: {np.mean(rewards):.2f}")
    
    # Detailed Reward Component Analysis
    print("\nDoor Controller Reward Component Statistics:")
    for component, values in reward_components.items():
        if values:  # Only print if we have values
            print(f"  {component}: avg={np.mean(values):.4f}, min={np.min(values):.4f}, max={np.max(values):.4f}")
    
    # Visualization and Report Generation
    # ==================================
    print("\nGenerating visualizations...")
    
    # Generate door vs terminal facing rewards comparison
    plot_door_vs_terminal_rewards(door_reward_by_step, terminal_reward_by_step, 
                                os.path.join(args.output_dir, "facing_rewards.png"))
    
    # Save navigation path visualizations
    world = get_world_from_env(env)
    if world and hasattr(world, 'successful_paths'):
        path_file = os.path.join(args.output_dir, "paths.png")
        save_paths(world, path_file)
        print(f"Saved path visualization to {path_file}")
        
        # Save color-matched path visualization with door positions
        color_matched_file = os.path.join(args.output_dir, "paths_color_matched.png")
        create_color_matched_plot(world, door_positions, color_matched_file)
        print(f"Saved color-matched path visualization to {color_matched_file}")
    
    # Generate reward analysis plots
    rewards_file = os.path.join(args.output_dir, "rewards.png")
    plot_rewards(agent_rewards, window=5, filename=rewards_file)
    print(f"Saved reward plot to {rewards_file}")
    
    # Generate success rate analysis
    success_file = os.path.join(args.output_dir, "success_rate.png")
    plot_success_rate(episode_successes, window=5, filename=success_file)
    print(f"Saved success rate plot to {success_file}")
    
    # Generate door position analysis
    door_file = os.path.join(args.output_dir, "door_positions.png")
    plot_door_positions(door_positions, filename=door_file)
    print(f"Saved door position plot to {door_file}")
    
    # Generate reward component analysis
    component_file = os.path.join(args.output_dir, "reward_components.png")
    plot_reward_components(reward_components, filename=component_file)
    print(f"Saved reward components plot to {component_file}")
    
    # Cleanup
    # =======
    env.close()
    
    print(f"\nResults saved to {args.output_dir}")

def plot_door_vs_terminal_rewards(door_rewards, terminal_rewards, filename=None):
    """
    Plot door facing rewards vs terminal facing rewards for detailed behavioral analysis.
    This visualization helps understand the balance between door-seeking and terminal-seeking behavior.
    
    Args:
        door_rewards: List of lists of door rewards by step for each episode
        terminal_rewards: List of lists of terminal rewards by step for each episode
        filename: Output filename for saving the plot
    """
    plt.figure(figsize=(12, 10))
    
    # Get maximum episode length for consistent x-axis scaling
    max_length = max([len(episode) for episode in door_rewards]) if door_rewards else 0
    
    # Plot the first 3 episodes (or fewer if less than 3 available)
    num_episodes = min(3, len(door_rewards))
    
    for i in range(num_episodes):
        plt.subplot(num_episodes, 1, i+1)
        
        episode_door_rewards = door_rewards[i]
        episode_terminal_rewards = terminal_rewards[i]
        
        # Ensure both reward lists have the same length for proper comparison
        min_length = min(len(episode_door_rewards), len(episode_terminal_rewards))
        episode_door_rewards = episode_door_rewards[:min_length]
        episode_terminal_rewards = episode_terminal_rewards[:min_length]
        
        # Create step indices for x-axis
        steps = range(len(episode_door_rewards))
        
        # Plot both reward types with different colors and labels
        plt.plot(steps, episode_door_rewards, 'b-', label='Door Facing Reward', linewidth=1.5)
        plt.plot(steps, episode_terminal_rewards, 'r-', label='Terminal Facing Reward', linewidth=1.5)
        
        plt.title(f'Episode {i+1} Facing Rewards Comparison')
        plt.xlabel('Step')
        plt.ylabel('Reward')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_reward_components(components, window=5, filename=None):
    """
    Plot detailed reward components over episodes with smoothing.
    Provides insight into how different reward mechanisms contribute to agent behavior.
    
    Args:
        components: Dict of component name to list of values across episodes
        window: Window size for smoothing (reduces noise in plots)
        filename: Output filename for saving the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Filter and categorize reward components for organized visualization
    alignment_components = {k: v for k, v in components.items() if 'alignment_reward' in k}
    approach_components = {k: v for k, v in components.items() if 'approach_reward' in k}
    efficiency_components = {k: v for k, v in components.items() if 'efficiency' in k}
    
    # Plot alignment reward components
    plt.subplot(3, 1, 1)
    for name, values in alignment_components.items():
        if values:  # Only plot if we have data
            # Apply smoothing to reduce noise
            smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
            plt.plot(smoothed, label=name, linewidth=1.5)
    plt.title('Alignment Rewards Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot approach reward components
    plt.subplot(3, 1, 2)
    for name, values in approach_components.items():
        if values:  # Only plot if we have data
            # Apply smoothing to reduce noise
            smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
            plt.plot(smoothed, label=name, linewidth=1.5)
    plt.title('Approach Rewards Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot efficiency reward components
    plt.subplot(3, 1, 3)
    for name, values in efficiency_components.items():
        if values:  # Only plot if we have data
            # Apply smoothing to reduce noise
            smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
            plt.plot(smoothed, label=name, linewidth=1.5)
    plt.title('Efficiency Rewards Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# Main Execution
# =============
# Entry point for running the script when called directly

if __name__ == "__main__":
    run_random_agent()