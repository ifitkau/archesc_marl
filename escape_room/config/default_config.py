import numpy as np

"""
Default configuration for the escape room environment

This configuration file defines all the parameters for a multi-agent escape room simulation
where a navigator agent must reach a terminal while a door controller agent manages door positions.
The environment consists of multiple rooms connected by doors that can be dynamically positioned.
"""

def get_default_config():
    """
    Returns the default configuration for the escape room environment
    
    This function provides all necessary parameters for setting up the escape room simulation,
    including world dimensions, door mechanics, agent positioning, reward systems, and training parameters.
    
    Returns:
        dict: Complete configuration dictionary with all environment parameters
    """
    return {
        # =====================================
        # WORLD GEOMETRY PROPERTIES
        # =====================================
        "world_width": 18.4,    # Total width of the environment (x-axis) in meters
        "world_depth": 6.7,     # Total depth of the environment (z-axis) in meters
        
        # =====================================
        # DOOR SYSTEM CONFIGURATION
        # =====================================
        "door_width": 1.0,                    # Width of doors in meters
        "door_position": 3.0,                 # Default/initial door position
        "door_position_min": 0.6,             # Minimum allowed door position (global constraint)
        "door_position_max": 5.4,             # Maximum allowed door position (global constraint)
        
        # Define valid door position ranges for each room
        "door_position_ranges": {
            "roomA": (0.6, 5.4),              # Valid door positions within room A boundaries
            "roomB": (0.6, 5.4),              # Valid door positions within room B boundaries  
            "roomC": (0.6, 5.4)               # Valid door positions within room C boundaries
        },
        
        # Door safety zones prevent agents from getting stuck when doors move
        "door_safe_zone_x_extension": 0.0,    # Safety buffer along door's length (x-axis) in meters
        "door_safe_zone_z_extension": 0.75,   # Safety buffer along door's width (z-axis) in meters
        "enable_door_safe_zones": True,       # Whether to enforce safety zones around doors
        "door_agent_start_episode": 1,        # Episode number when door controller agent becomes active
                
        # =====================================
        # TERMINAL (GOAL) CONFIGURATION  
        # =====================================
        "terminal_location": [18.4, 5.95],    # [x, z] coordinates of the goal terminal
        # Alternative terminal positions (commented out):
        #"terminal_location": [0.0, 5.95],    # Terminal at left edge
        #"terminal_location": [9.2, 6.7],     # Terminal at center-top

        # =====================================
        # EPISODE AND TRAINING PARAMETERS
        # =====================================
        "max_episode_steps": 500,             # Maximum steps allowed per episode before timeout

        "episodes_per_room": 3,               # Number of consecutive episodes in same room before switching        
        
        # Door controller behavior settings
        "door_move_frequency": 1,             # How often (in episodes) the door position can change
        "discrete_door_positions": 5,        # Number of discrete positions the door can occupy
        "fixed_position_episodes": 80,       # Number of episodes with fixed door positions (for initial training)

        # =====================================
        # SUCCESS REWARD SYSTEM
        # =====================================
        # Reward system that gives higher rewards for faster completion
        "success_reward_min": 500.0,         # Minimum reward for slow but successful completion
        "success_reward_max": 5000.0,        # Maximum reward for very fast completion
        "success_reward_midpoint": 100,      # Step count where reward transitions from high to low most steeply
        "success_reward_steepness": 0.015,   # Controls how quickly reward drops off with more steps
        
        # =====================================
        # NAVIGATOR AGENT STARTING POSITIONS
        # =====================================
        "room_categories": {
            # Starting positions and orientations for the navigator agent
            # Format: ([x, y, z], orientation_in_radians)
            "room": [
                # Room C starting positions (rightmost room)
                ([12.9, 0, 0.5], -np.pi/2),   # Position 1 in room C
                ([17.9, 0, 0.5], -np.pi/2),   # Position 2 in room C
                ([15.4, 0, 0.5], -np.pi/2),   # Position 3 in room C
                
                # Room B starting positions (middle room) 
                ([6.7, 0, 0.5], -np.pi/2),    # Position 1 in room B
                ([11.7, 0, 0.5], -np.pi/2),   # Position 2 in room B
                ([9.2, 0, 0.5], -np.pi/2),    # Position 3 in room B
                
                # Room A starting positions (leftmost room)
                ([0.5, 0, 0.5], -np.pi/2),    # Position 1 in room A
                ([5.5, 0, 0.5], -np.pi/2),    # Position 2 in room A
                ([3.0, 0, 0.5], -np.pi/2),    # Position 3 in room A
                
                # Additional commented positions for fine-tuning:
                #([11.4, 0, 1.5], -np.pi/2), #([10.4, 0, 0.5], -np.pi/2),
                #([5.2, 0, 0.5], -np.pi/2),  #([7.2, 0, 0.5], -np.pi/2),
                #([1.0, 0, 0.5], -np.pi/2),  #([3.0, 0, 0.5], -np.pi/2),
            ],

            # Hallway starting positions (corridor between rooms)
            "hallway": [
                ([9.2, 0, 5.95], -np.pi/2),   # Central hallway position
                # Additional hallway position (commented): ([4.1, 0, 3.4], -np.pi/2),
            ],
        },
        
        # Agent placement settings
        "use_sequential_all_positions": False,  # If True, cycle through positions sequentially; if False, random selection
        "random_agent_placement": True,         # Enable random placement of agents at episode start
        
        # =====================================
        # STARTING LOCATION PROBABILITIES
        # =====================================
        # Probability distribution for where navigator agent starts each episode
        "room_probabilities": {
            "room": 1.0,        # 100% chance to start in a room
            "hallway": 0.0,     # 0% chance to start in hallway
        },
         
        # =====================================
        # NAVIGATOR AGENT REWARD SCALES
        # =====================================
        # Fine-tune navigator behavior by scaling different reward components
        "navigator_reward_scales": {
            "reward_orientation_scale": 1.0,        # Reward for facing the right direction
            "reward_distance_scale": 0.0,           # Reward for getting closer to goal (disabled)
            "punishment_distance_scale": 0.0,       # Punishment for moving away from goal (disabled)
            "penalty_stagnation_scale": 1.0,        # Penalty for not moving/making progress
            "punishment_time_scale": 0.0,           # Punishment for taking too many steps (disabled)
            "reward_hallway_scale": 1.0,            # Reward for reaching hallway area
            "reward_terminal_scale": 0.3,           # Reward for approaching terminal
            "punishment_terminal_scale": 0.0,       # Punishment for moving away from terminal (disabled)
            "punishment_room_scale": 0.0,           # Punishment for staying in rooms (disabled)
            "wall_collision_scale": 1.0,            # Penalty for colliding with walls
            "reward_door_approach_scale": 0.0,      # Reward for approaching doors (disabled)
            "reward_approach_door_scale": 0.0,      # Alternative door approach reward (disabled)
        },
        
        # =====================================
        # DOOR CONTROLLER AGENT REWARD SCALES
        # =====================================
        # Configure door controller learning objectives and behavior
        "door_controller_reward_scales": {
            "reward_terminal_alignment_scale": 0.0,  # Terminal alignment reward (disabled - hardcoded to 10.0)
            "hallway_transition_reward": 0.01,       # Small reward when navigator enters hallway
            "reward_efficiency_scale": 0.0,          # Reward for helping navigator be efficient (disabled)
            "reward_door_placement_scale": 0.0,      # Reward for optimal door placement (disabled for unbiased learning)
            "reward_actual_steps_scale": 0.0,        # Reward based on navigator's step count (disabled)
            "reward_success_scale": 0.1,             # Reward when navigator successfully reaches terminal
            "reward_path_quality_scale": 0.0,        # Reward for enabling high-quality paths (disabled)
            "reward_terminal_steps_scale": 2.0,      # Reward based on steps taken to reach terminal
        },
        
        # =====================================
        # ENVIRONMENT EXECUTION SETTINGS
        # =====================================
        "parallel_env": True,    # Enable parallel environment execution for faster training
    }