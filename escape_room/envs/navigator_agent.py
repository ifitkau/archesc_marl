"""
Navigator agent implementation for the escape room environment

This module implements the navigator agent that must traverse through rooms to reach 
a terminal location. The navigator can move forward, turn left, and turn right while
navigating around obstacles and through doors controlled by the door controller agent.
The agent receives rewards for efficient navigation, proper orientation, and reaching
key milestones like the hallway and terminal.
"""
from logging import info
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time

class NavigatorAgent:
    """
    The navigator agent that attempts to reach the terminal location through strategic movement.
    
    This agent operates in a multi-room environment where it must:
    - Navigate from a starting position in one of the rooms
    - Find and pass through doors (controlled by door controller agent) 
    - Enter the central hallway
    - Reach the terminal location for episode success
    
    Key Features:
    - Discrete action space (turn right, turn left, move forward)
    - Rich observation space including lidar, door positions, and spatial awareness
    - Multi-component reward system encouraging efficient navigation
    - Collision detection and avoidance
    - Room-aware door visibility (can only see relevant doors)
    """
    
    def __init__(self, config=None, rng=None):
        """
        Initialize the navigator agent
        
        Args:
            config (dict): Configuration dictionary with environment parameters
            rng (np.random.RandomState): Random number generator for reproducible behavior
        """
        self.name = "navigator"
        
        # =====================================
        # OBSERVATION SPACE DEFINITION
        # =====================================
        # Define what the navigator agent can perceive about its environment
        self.observation_space = spaces.Box(
            low=np.array([
                # ===== SPATIAL CONTEXT =====
                0,                      # Room category (0-4: roomA, roomB, roomC, hallway, unknown)
                
                # ===== AGENT ORIENTATION =====
                -1, -1,                 # Agent's facing direction vector [x, z] normalized
                -1,                     # Alignment with optimal direction to terminal (-1 to 1)

                # ===== DOOR INFORMATION =====
                # For each room's door: [x_position, z_position, alignment_with_agent_direction]
                # Only visible doors (from current room) will have non-zero values
                0, 0,                   # Door A position [x, z] normalized to world coordinates
                -1,                     # Agent's alignment with direction to Door A
                0, 0,                   # Door B position [x, z] normalized to world coordinates
                -1,                     # Agent's alignment with direction to Door B
                0, 0,                   # Door C position [x, z] normalized to world coordinates
                -1,                     # Agent's alignment with direction to Door C

                # ===== SENSOR DATA =====
                0, 0, 0, 0, 0,          # LIDAR distance measurements (5 directions, normalized)
                
                # ===== PROGRESS TRACKING =====
                0,                      # Episode step count (normalized to max steps)
                0,                      # Stagnation counter (normalized, tracks lack of movement)
                
            ], dtype=np.float32),
            high=np.array([
                # Maximum values for all observation components
                4,                      # Maximum room category
                1, 1,                   # Maximum agent direction components
                1,                      # Maximum alignment value
                1, 1, 1,               # Maximum door A values
                1, 1, 1,               # Maximum door B values
                1, 1, 1,               # Maximum door C values
                1, 1, 1, 1, 1,         # Maximum LIDAR readings
                1, 1,                   # Maximum progress tracking values
            ], dtype=np.float32)
        )
        
        # =====================================
        # ACTION SPACE DEFINITION
        # =====================================
        # Define the actions the navigator can take
        # 0: Turn right, 1: Turn left, 2: Move forward
        self.action_space = spaces.Discrete(3)
        
        # =====================================
        # STATE TRACKING VARIABLES
        # =====================================
        # Track important state information across time steps
        self.previous_distance_to_terminal = None  # For distance-based reward calculation
        self.previous_distance_to_door = None      # For door approach reward calculation
        self.started_in_hallway = False            # Did agent start episode in hallway?
        self.has_reached_hallway = False           # Has agent reached hallway this episode?
        self.hallway_reward_given = False          # Prevent duplicate hallway rewards
        self.stagnant_steps = 0                    # Count of consecutive non-movement steps
        self.non_hallway_steps = 0                 # Steps taken outside hallway after reaching it

        # =====================================
        # RANDOM NUMBER GENERATION
        # =====================================
        # Initialize RNG for any stochastic behavior
        self.rng = rng if rng is not None else np.random.RandomState()

    def seed(self, seed):
        """
        Seed the agent's random number generator for reproducible behavior
        
        Args:
            seed (int): Random seed value. If None, uses current time.
            
        Returns:
            list: List containing the actual seed used
        """
        if seed is None:
            # Generate seed from current time if none provided
            seed = int(time.time()) % (2**31-1)
        self.rng = np.random.RandomState(seed)
        return [seed]
        
    def reset(self, miniworld_env, start_position=None, start_direction=None):
        """
        Reset the agent's state for a new episode
        
        Clears all tracking variables and optionally places the agent at a specific
        starting position and orientation. Called at the beginning of each episode.
        
        Args:
            miniworld_env: Reference to the MiniWorld environment
            start_position (np.array, optional): Starting position [x, y, z]
            start_direction (float, optional): Starting orientation in radians
            
        Returns:
            None
        """
        # =====================================
        # RESET STATE TRACKING
        # =====================================
        # Clear all episode-specific tracking variables
        self.previous_distance_to_terminal = None  # Reset distance tracking
        self.previous_distance_to_door = None      # Reset door approach tracking
        self.has_reached_hallway = False           # Reset hallway achievement
        self.hallway_reward_given = False          # Reset reward eligibility
        self.stagnant_steps = 0                    # Reset movement tracking
        self.non_hallway_steps = 0                 # Reset post-hallway movement
        self._terminal_reached = False             # Reset terminal achievement flag
        
        # =====================================
        # AGENT PLACEMENT
        # =====================================
        # Place the agent in the environment if position/direction specified
        if start_position is not None and start_direction is not None:
            miniworld_env.place_entity(
                miniworld_env.agent,
                pos=start_position,
                dir=start_direction
            )
    
    def get_observation(self, miniworld_env):
        """
        Construct the agent's observation vector from the current environment state
        
        Creates a comprehensive observation that includes spatial awareness, sensor data,
        door information (with room-based visibility), and progress tracking. Door positions
        are only visible when the agent is in the same room or in the hallway.
        
        Args:
            miniworld_env: Reference to the MiniWorld environment
            
        Returns:
            np.array: Normalized observation vector for the navigator agent
        """
        # =====================================
        # BASIC ENVIRONMENT REFERENCES
        # =====================================
        agent = miniworld_env.agent                    # Navigator agent reference
        terminal_loc = miniworld_env.terminal_location  # Goal location [x, z]
        world_width = miniworld_env.world_width         # Environment width
        world_depth = miniworld_env.world_depth         # Environment depth
        
        # Get current room to determine door visibility
        current_room = miniworld_env._get_current_room()
        
        # =====================================
        # DISTANCE AND PROGRESS TRACKING
        # =====================================
        # Calculate current straight-line distance to terminal
        distance_to_terminal = np.linalg.norm(
            np.array([agent.pos[0], agent.pos[2]]) - 
            np.array([terminal_loc[0], terminal_loc[1]])
        )
        
        # Store distance for reward calculation in process_action()
        if self.previous_distance_to_terminal is None:
            self.previous_distance_to_terminal = distance_to_terminal
        
        # =====================================
        # SENSOR DATA COLLECTION
        # =====================================
        # Get LIDAR measurements for obstacle detection
        lidar_measurements = miniworld_env.get_lidar_measurements()
        
        # Normalize LIDAR readings to [0,1] range for consistent observation space
        max_lidar_distance = 20.0  # Maximum possible LIDAR range
        normalized_lidar = [min(1.0, measurement / max_lidar_distance) for measurement in lidar_measurements]
        
        # =====================================
        # SPATIAL COORDINATE NORMALIZATION
        # =====================================
        # Normalize agent position to world coordinates
        norm_agent_pos = np.array([
            agent.pos[0] / world_width,   # X coordinate normalized
            0,                            # Skip Y coordinate (height not used)
            agent.pos[2] / world_depth    # Z coordinate normalized
        ])
        
        # Convert to [0,1] range for consistent observation space
        norm_agent_pos[0] = (norm_agent_pos[0] + 1) / 2  # X to [0,1]
        norm_agent_pos[2] = (norm_agent_pos[2] + 1) / 2  # Z to [0,1]

        # Normalize terminal position using same coordinate system
        norm_terminal_pos = np.array(terminal_loc) / np.array([world_width, world_depth])
        norm_terminal_pos[0] = (norm_terminal_pos[0] + 1) / 2  # X to [0,1]
        norm_terminal_pos[1] = (norm_terminal_pos[1] + 1) / 2  # Z to [0,1]

        # =====================================
        # DIRECTION VECTOR CALCULATIONS
        # =====================================
        # Calculate optimal direction from agent to terminal
        dx = terminal_loc[0] - agent.pos[0]  # X distance to terminal
        dz = terminal_loc[1] - agent.pos[2]  # Z distance to terminal
        
        # Create normalized direction vector to terminal (flip Z for correct orientation)
        direction_vector = np.array([dx, -dz])
        direction_length = np.linalg.norm(direction_vector)
        if direction_length > 0:
            direction_vector = direction_vector / direction_length
        else:
            direction_vector = np.array([1.0, 0.0])  # Default forward if at same position
        
        # Get agent's current facing direction (always normalized)
        agent_dir_vec = np.array([
            np.cos(agent.dir),   # X component of agent's facing direction
            np.sin(agent.dir)    # Z component of agent's facing direction
        ])
        
        # Calculate alignment between agent's direction and optimal direction to terminal
        dot_product = np.dot(direction_vector, agent_dir_vec)
        
        # Calculate signed angle difference for turning direction information
        angle_difference = np.arccos(np.clip(dot_product, -1.0, 1.0))
        
        # Determine turn direction using cross product
        cross_z = agent_dir_vec[0] * direction_vector[1] - agent_dir_vec[1] * direction_vector[0]
        if cross_z < 0:
            angle_difference = -angle_difference
        
        # Store for use in reward calculation
        self.last_angle_difference = angle_difference
        self.last_dot_product = dot_product
        
        # =====================================
        # ROOM CATEGORIZATION
        # =====================================
        # Convert current room to numerical category for observation
        try:
            room_category = miniworld_env._get_room_category(current_room)
        except Exception:
            room_category = 4  # Unknown room category
        
        # =====================================
        # DOOR VISIBILITY AND POSITIONING
        # =====================================
        # Room layout constants for coordinate transformation (CRITICAL: must match other agents)
        room_offsets = {
            'roomA': 0.0,    # Room A starts at world X=0.0
            'roomB': 6.2,    # Room B starts at world X=6.2  
            'roomC': 12.4    # Room C starts at world X=12.4
        }
        
        # All doors are at the same Z coordinate (entrance to hallway)
        door_z = 5.0
        norm_door_z = door_z / miniworld_env.world_depth  # Normalize Z coordinate
        
        # Initialize door position and alignment data structures
        door_positions_spatial = {
            'roomA': {'x': 0.0, 'z': 0.0},  # Default to invisible (0,0)
            'roomB': {'x': 0.0, 'z': 0.0},  # Default to invisible (0,0)
            'roomC': {'x': 0.0, 'z': 0.0}   # Default to invisible (0,0)
        }
        
        door_dot_products = {
            'roomA': 0.0,    # Default to no alignment
            'roomB': 0.0,    # Default to no alignment
            'roomC': 0.0     # Default to no alignment
        }
        
        # =====================================
        # ROOM-BASED DOOR VISIBILITY LOGIC
        # =====================================
        # Determine which doors the agent can see based on current location
        visible_rooms = []
        
        if current_room == 'roomD':  # Agent is in central hallway
            visible_rooms = ['roomA', 'roomB', 'roomC']  # Can see all doors from hallway
        elif current_room in ['roomA', 'roomB', 'roomC']:  # Agent is in a specific room
            visible_rooms = [current_room]  # Can only see the door of current room
        # If in unknown room, no doors are visible (remain at default 0,0)
        
        # =====================================
        # DOOR POSITION AND ALIGNMENT CALCULATION
        # =====================================
        # Calculate door positions and alignment for visible doors only
        for room in visible_rooms:
            # Set Z-coordinate for visible door
            door_positions_spatial[room]['z'] = norm_door_z
            
            # Get door position from environment tracking
            room_pos = None
            if hasattr(miniworld_env, 'individual_door_positions'):
                # Preferred: individual tracking for each room
                room_pos = miniworld_env.individual_door_positions.get(room, 0.0)
            elif hasattr(miniworld_env, 'door_position'):
                # Fallback: single door position (legacy compatibility)
                room_pos = miniworld_env.door_position
                
            if room_pos is not None:
                # Convert relative door position to absolute world coordinates
                door_x_actual = room_pos + room_offsets.get(room, 0.0)
                
                # Normalize door X position to world width for observation
                door_positions_spatial[room]['x'] = door_x_actual / world_width
                
                # =====================================
                # DOOR ALIGNMENT CALCULATION
                # =====================================
                # Calculate how well agent is aligned with direction to this door
                door_direction_vector = np.array([
                    door_x_actual - agent.pos[0],           # X distance to door
                    -(door_z - agent.pos[2])                # Z distance to door (flipped)
                ])
                
                # Normalize door direction vector
                door_direction_length = np.linalg.norm(door_direction_vector)
                if door_direction_length > 0:
                    door_direction_vector = door_direction_vector / door_direction_length
                    
                    # Calculate alignment (dot product) between agent direction and door direction
                    door_dot_products[room] = np.dot(door_direction_vector, agent_dir_vec)
                else:
                    # Agent is at the door position - consider perfectly aligned
                    door_dot_products[room] = 1.0

        # =====================================
        # OBSERVATION VECTOR CONSTRUCTION
        # =====================================
        # Assemble complete observation array with all state information
        obs = np.array([
            # ===== SPATIAL CONTEXT =====
            float(room_category),                       # Which room/area is agent in?
            
            # ===== AGENT ORIENTATION =====
            agent_dir_vec[0], agent_dir_vec[1],        # Agent's facing direction components
            dot_product,                               # Alignment with optimal direction to terminal
            
            # ===== DOOR INFORMATION =====
            # Room A door: position and alignment (only non-zero if visible)
            door_positions_spatial['roomA']['x'],       # Door A X position (normalized)
            door_positions_spatial['roomA']['z'],       # Door A Z position (normalized)
            door_dot_products['roomA'],                 # Agent alignment with Door A direction
            
            # Room B door: position and alignment (only non-zero if visible)
            door_positions_spatial['roomB']['x'],       # Door B X position (normalized)
            door_positions_spatial['roomB']['z'],       # Door B Z position (normalized)
            door_dot_products['roomB'],                 # Agent alignment with Door B direction
            
            # Room C door: position and alignment (only non-zero if visible)
            door_positions_spatial['roomC']['x'],       # Door C X position (normalized)
            door_positions_spatial['roomC']['z'],       # Door C Z position (normalized)
            door_dot_products['roomC'],                 # Agent alignment with Door C direction

            # ===== SENSOR DATA =====
            # LIDAR measurements in 5 directions (normalized to [0,1])
            normalized_lidar[0],                        # Right side distance
            normalized_lidar[1],                        # Left side distance
            normalized_lidar[2],                        # Forward distance
            normalized_lidar[3],                        # Forward-right diagonal distance
            normalized_lidar[4],                        # Forward-left diagonal distance

            # ===== PROGRESS TRACKING =====
            miniworld_env.step_count/500,               # Normalized step count (progress through episode)
            self.stagnant_steps/100,                    # Normalized stagnation counter (movement tracking)
        ], dtype=np.float32)
        
        # =====================================
        # SAFETY CHECKS
        # =====================================
        # Replace any NaN or infinite values with safe defaults
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0)
        
        return obs
    
    def process_action(self, miniworld_env, action):
        """
        Execute the agent's action and calculate resulting reward and state changes
        
        This method handles the core action-reward loop for the navigator agent:
        1. Executes the chosen action (turn/move) in the environment
        2. Tracks position changes and progress metrics
        3. Calculates multi-component rewards
        4. Checks for episode termination conditions
        5. Updates state tracking variables
        
        Args:
            miniworld_env: Reference to the MiniWorld environment
            action (int): Action to execute (0: right, 1: left, 2: forward)
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
                - observation: Updated state observation after action
                - reward: Calculated reward for this action
                - terminated: Whether episode should end (reached terminal)
                - truncated: Whether episode was cut short (always False)
                - info: Dictionary with additional episode information
        """
        # =====================================
        # INITIALIZE INFO DICTIONARY
        # =====================================
        # Track episode information for analysis and debugging
        info = {
            'distance_to_terminal': 0,     # Current distance to goal
            'is_in_hallway': False,        # Whether agent is in central hallway
            'steps': miniworld_env.step_count,  # Current step count
        }
        
        # =====================================
        # POSITION TRACKING FOR MOVEMENT DETECTION
        # =====================================
        # Store agent position before action to detect actual movement
        previous_agent_pos = np.array(miniworld_env.agent.pos)
        
        # Execute the chosen action in the MiniWorld environment
        miniworld_env.step_navigator(action)
        
        # Get new position after action execution
        current_agent_pos = np.array(miniworld_env.agent.pos)
        
        # Determine if agent actually moved (threshold to account for floating point precision)
        position_changed = np.linalg.norm(current_agent_pos - previous_agent_pos) > 0.1
        
        # =====================================
        # STAGNATION TRACKING
        # =====================================
        # Track consecutive steps without movement (important for anti-stagnation rewards)
        if position_changed:
            self.stagnant_steps = 0  # Reset counter when agent moves
        else:
            self.stagnant_steps += 1  # Increment when agent doesn't move
        
        # =====================================
        # SPATIAL STATE ANALYSIS
        # =====================================
        # Calculate current distance to terminal for reward/progress tracking
        distance_to_terminal = np.linalg.norm(
            np.array([miniworld_env.agent.pos[0], miniworld_env.agent.pos[2]]) - 
            np.array([miniworld_env.terminal_location[0], miniworld_env.terminal_location[1]])
        )
        
        # Determine current room and hallway status
        current_room = miniworld_env._get_current_room()
        is_in_hallway = current_room in ['roomD']  # roomD is the central hallway
        
        # =====================================
        # HALLWAY ACHIEVEMENT TRACKING
        # =====================================
        # Track hallway progress for reward calculations
        if is_in_hallway:
            self.non_hallway_steps = 0  # Reset counter when in hallway
            # Check for first-time hallway achievement
            if not self.has_reached_hallway and not self.started_in_hallway:
                self.has_reached_hallway = True
        else:
            self.non_hallway_steps += 1  # Count steps outside hallway
        
        # =====================================
        # TERMINAL DETECTION AND EPISODE TERMINATION
        # =====================================
        # Check if agent has reached the terminal area for episode success
        # Use agent's "nose" position (front edge) for more accurate detection
        agent_nose_pos = miniworld_env.agent.pos + miniworld_env.agent.dir_vec * miniworld_env.agent.radius
        
        # Terminal detection for RIGHT terminal configuration (default)
        # Terminal area is defined as a rectangular region around the terminal
        is_terminal_area = (
            5.45 <= agent_nose_pos[2] <= 6.45 and    # Z-coordinate range (depth)
            18.2 <= agent_nose_pos[0] <= 18.4 and    # X-coordinate range (width) - RIGHT side
            agent_nose_pos[1] <= 0.2                  # Y-coordinate constraint (height)
        )
        
        # Alternative terminal configurations (commented out):
        # LEFT terminal: 0 <= agent_nose_pos[0] <= 0.2
        # CENTER terminal: 8.7 <= agent_nose_pos[0] <= 9.7 with 6.5 <= agent_nose_pos[2] <= 6.7

        # Set episode termination flag
        terminated = is_terminal_area
        if terminated:
            # Mark episode as successful for other agents and analysis
            miniworld_env._episode_success = True
            info['terminate_all'] = True      # Signal to terminate all agents
            info['is_successful'] = True      # Mark as successful completion
        else:
            miniworld_env._episode_success = False

        # =====================================
        # INFO DICTIONARY UPDATE
        # =====================================
        # Add current state information to info dictionary
        info.update({
            'distance_to_terminal': distance_to_terminal,  # Current distance to goal
            'is_in_hallway': is_in_hallway,               # Current location status
        })
        
        # =====================================
        # REWARD CALCULATION
        # =====================================
        # Calculate comprehensive reward based on action outcomes
        reward = self._calculate_reward(
            miniworld_env,
            distance_to_terminal,
            position_changed,
            is_in_hallway,
            terminated,
            miniworld_env.step_count
        )
        
        # =====================================
        # STATE UPDATE FOR NEXT STEP
        # =====================================
        # Update distance tracking for reward calculation in next step
        if position_changed:
            self.previous_distance_to_terminal = distance_to_terminal
        
        # Track agent's path for analysis (used by door controller for path quality rewards)
        miniworld_env.current_path.append(list(miniworld_env.agent.pos))
        
        # =====================================
        # RETURN ENVIRONMENT STEP RESULT
        # =====================================
        # Return standard Gym environment tuple
        return self.get_observation(miniworld_env), reward, terminated, False, info
    
    def _calculate_boundary_based_collision_penalty(self, miniworld_env):
        """
        Calculate wall collision penalty using agent boundary distance rather than center distance
        
        This method provides more accurate collision detection by considering the agent's
        radius when calculating distance to walls. It prevents the agent from getting
        too close to walls by penalizing proximity based on the agent's actual boundary.
        
        Args:
            miniworld_env: Reference to the MiniWorld environment
            
        Returns:
            float: Collision penalty (negative value when too close to walls)
        """
        # =====================================
        # COLLISION DETECTION PARAMETERS
        # =====================================
        collision_threshold = 0.075    # Distance threshold for collision penalty
        agent_radius = miniworld_env.agent.radius  # Agent's physical radius (typically 0.25)
        
        # =====================================
        # LIDAR MEASUREMENTS FROM CENTER
        # =====================================
        # Get distance measurements from agent center to walls in all directions
        right_dist, left_dist, forward_dist, forward_right_dist, forward_left_dist = \
            miniworld_env.get_lidar_measurements_with_safe_zones()
        
        # =====================================
        # BOUNDARY DISTANCE CALCULATION
        # =====================================
        # Key insight: LIDAR measures from agent center to wall
        # But collision occurs when agent BOUNDARY touches wall
        # Therefore: boundary_distance = lidar_distance - agent_radius
        
        boundary_distances = {
            'right': max(0, right_dist - agent_radius),          # Right side boundary to wall
            'left': max(0, left_dist - agent_radius),            # Left side boundary to wall
            'forward': max(0, forward_dist - agent_radius),      # Front boundary to wall
            'forward_right': max(0, forward_right_dist - agent_radius),  # Front-right boundary to wall
            'forward_left': max(0, forward_left_dist - agent_radius)     # Front-left boundary to wall
        }
        
        # =====================================
        # COLLISION PENALTY CALCULATION
        # =====================================
        # Find the closest boundary distance to determine collision risk
        min_boundary_distance = min(boundary_distances.values())
        closest_direction = min(boundary_distances, key=boundary_distances.get)
        
        wall_collision_penalty = 0.0
        
        # Check if agent boundary is within collision threshold of any wall
        if min_boundary_distance < collision_threshold:
            # Calculate penalty based on how close the boundary is to collision
            # Penalty increases as boundary gets closer to wall
            wall_collision_penalty = -1 * (1 - (min_boundary_distance / collision_threshold))
            
            # Debug output for collision analysis (uncommented for production)
            # This shows the difference between center-based and boundary-based collision detection
        
        return wall_collision_penalty
    
    def _calculate_reward(self, miniworld_env, distance_to_terminal, position_changed, 
                          is_in_hallway, terminated, step_count):
        """
        Calculate comprehensive reward for the navigator agent
        
        This method implements a multi-component reward system that encourages:
        - Proper orientation toward goals (terminal and doors)
        - Efficient movement and progress
        - Reaching key milestones (hallway, terminal)
        - Avoiding walls and obstacles
        - Minimizing stagnation and time usage
        
        Args:
            miniworld_env: Reference to the MiniWorld environment
            distance_to_terminal (float): Current distance to terminal location
            position_changed (bool): Whether agent moved this step
            is_in_hallway (bool): Whether agent is currently in central hallway
            terminated (bool): Whether episode terminated successfully
            step_count (int): Current step number in episode
            
        Returns:
            float: Total calculated reward for this action
        """
        # =====================================
        # REWARD COMPONENT INITIALIZATION
        # =====================================
        # Initialize all reward components to zero for this step
        reward_orientation = 0          # Reward for facing correct direction
        reward_distance_terminal = 0    # Reward for getting closer to terminal
        punishment_distance_terminal = 0 # Punishment for getting farther from terminal
        penalty_stagnation = 0          # Penalty for not moving
        punishment_time = 0             # Penalty for each time step (efficiency pressure)
        reward_hallway = 0              # Reward for reaching/staying in hallway
        reward_terminal = 0             # Reward for reaching terminal
        punishment_terminal = 0         # Punishment for failure conditions
        punishment_room = 0             # Punishment for being in wrong locations
        wall_collision_penalty = 0      # Penalty for getting too close to walls
        reward_door_approach = 0        # Reward for approaching doors efficiently
        reward_approach_door = 0        # Reward for facing doors correctly
        
        # Get reward scaling parameters from configuration
        scales = miniworld_env.config["navigator_reward_scales"]
        
        # Environment references for calculations
        agent = miniworld_env.agent
        terminal_loc = miniworld_env.terminal_location

        # =====================================
        # OPTIMAL DIRECTION CALCULATION
        # =====================================
        # Calculate the optimal direction vector from agent to terminal
        direction_vector = np.array([
            terminal_loc[0] - agent.pos[0],              # X component of direction to terminal
            -(terminal_loc[1] - agent.pos[2])            # Z component (flipped for correct orientation)
        ])
        
        # Normalize direction vector for consistent calculations
        direction_length = np.linalg.norm(direction_vector)
        if direction_length > 0:
            direction_vector = direction_vector / direction_length
        
        # Agent's current facing direction
        agent_dir_vec = np.array([
            np.cos(agent.dir),   # X component of agent's facing direction
            np.sin(agent.dir)    # Z component of agent's facing direction
        ])
        
        # Calculate alignment between agent direction and optimal direction
        dot_product = np.dot(direction_vector, agent_dir_vec)

        # =====================================
        # DOOR APPROACH REWARD (PRE-HALLWAY)
        # =====================================
        # Reward for efficient door approach when agent hasn't reached hallway yet
        if not is_in_hallway and not self.has_reached_hallway:
            # Get current room to determine which door to approach
            current_room = miniworld_env._get_current_room()
            
            # Only calculate door approach rewards in rooms with doors
            if current_room in ['roomA', 'roomB', 'roomC']:
                # Get door position for current room from environment tracking
                door_pos = None
                if hasattr(miniworld_env, 'individual_door_positions'):
                    # Preferred: individual door position tracking
                    door_pos = miniworld_env.individual_door_positions.get(current_room)
                elif hasattr(miniworld_env, 'door_position'):
                    # Fallback: legacy single door position
                    door_pos = miniworld_env.door_position
                    
                if door_pos is not None:
                    # =====================================
                    # DOOR COORDINATE CALCULATION
                    # =====================================
                    # Room boundary definitions for coordinate transformation
                    room_boundaries = {
                        'roomA': (0.0, 6.0),      # Room A: left edge to right edge
                        'roomB': (6.2, 12.2),     # Room B: left edge to right edge
                        'roomC': (12.4, 18.4)     # Room C: left edge to right edge
                    }
                    
                    if current_room in room_boundaries:
                        room_min_x, _ = room_boundaries[current_room]
                        
                        # Convert relative door position to absolute world coordinates
                        door_x = door_pos + room_min_x  # Absolute X position of door
                        door_z = 5.0                    # All doors are at Z=5.0 (hallway entrance)
                        
                        # Calculate current distance from agent to door
                        current_distance_to_door = np.sqrt(
                            (miniworld_env.agent.pos[0] - door_x)**2 + 
                            (miniworld_env.agent.pos[2] - door_z)**2
                        )
                        
                        # =====================================
                        # DOOR DIRECTION ALIGNMENT REWARD
                        # =====================================
                        # Calculate direction vector from agent to door
                        door_direction_vector = np.array([
                            door_x - miniworld_env.agent.pos[0],           # X distance to door
                            -(door_z - miniworld_env.agent.pos[2])         # Z distance (flipped like terminal calc)
                        ])
                        
                        # Normalize door direction vector for consistent calculations
                        door_direction_length = np.linalg.norm(door_direction_vector)
                        if door_direction_length > 0:
                            door_direction_vector = door_direction_vector / door_direction_length
                        else:
                            door_direction_vector = np.array([0.0, 1.0])  # Default if at same position
                        
                        # Calculate alignment between agent direction and door direction
                        door_dot_product = np.dot(door_direction_vector, agent_dir_vec)
                        
                        # Reward agent for facing the door when close enough
                        if current_distance_to_door < 1.5:  # Within reasonable range of door
                            if door_dot_product > np.cos(np.pi / 9):  # Within 20 degrees of optimal direction
                                # Scale reward based on alignment quality (1.0 = perfect alignment)
                                reward_approach_door += 0.1
                        
                        # Update door distance tracking for future calculations
                        self.previous_distance_to_door = current_distance_to_door

        # =====================================
        # TERMINAL ORIENTATION REWARD/PENALTY
        # =====================================
        # Penalize agent for not facing toward the terminal
        if dot_product < np.cos(np.pi / 9):  # If not within 20 degrees of optimal direction
            reward_orientation -= 0.1  # Penalty for poor orientation
  
        # =====================================
        # TIME PRESSURE
        # =====================================
        # Apply constant time pressure to encourage efficiency
        punishment_time -= 0.5  # Small penalty per step to encourage faster completion
        
        # =====================================
        # STAGNATION PENALTY
        # =====================================
        # Large penalty for excessive stagnation (not moving for too long)
        if self.stagnant_steps >= 100:
            penalty_stagnation = -100  # Large penalty to break stagnation
            self.stagnant_steps = 0    # Reset counter after penalty
        
        # =====================================
        # HALLWAY REWARDS
        # =====================================
        # Reward system for reaching and staying in hallway
        if is_in_hallway:
            # Small constant reward for being in hallway (progress maintenance)
            reward_hallway += 0.05
            
            # Large one-time reward for first time reaching hallway
            if self.has_reached_hallway and not self.started_in_hallway and not self.hallway_reward_given:
                reward_hallway += 100      # Major milestone achievement
                self.hallway_reward_given = True  # Prevent duplicate rewards

        # =====================================
        # POST-HALLWAY MOVEMENT PENALTY
        # =====================================
        # Penalize leaving hallway after reaching it (discourages backtracking)
        if self.has_reached_hallway and not is_in_hallway:
            punishment_room -= 0.5  # Penalty for leaving hallway after achievement

        # =====================================
        # WALL COLLISION PENALTY
        # =====================================
        # Calculate penalty for getting too close to walls
        wall_collision_penalty = self._calculate_boundary_based_collision_penalty(miniworld_env)

        # =====================================
        # TERMINAL SUCCESS REWARD
        # =====================================
        # Large reward for successfully reaching the terminal
        if terminated:
            reward_terminal = 500  # Fixed success reward
            
            # Alternative reward systems (commented out):
            # - Step-based rewards that decrease with time
            # - Sigmoid-based rewards with configurable parameters
            # - Linear interpolation between min/max rewards based on step count

        # =====================================
        # TOTAL REWARD CALCULATION
        # =====================================
        # Apply configuration-based scaling to all reward components and sum them
        total_reward = (
            scales['reward_orientation_scale'] * reward_orientation +           # Orientation toward terminal
            scales['reward_distance_scale'] * reward_distance_terminal +        # Progress toward terminal
            scales['punishment_distance_scale'] * punishment_distance_terminal + # Regression penalty
            scales['penalty_stagnation_scale'] * penalty_stagnation +           # Anti-stagnation
            scales['punishment_time_scale'] * punishment_time +                 # Time efficiency pressure
            scales['reward_hallway_scale'] * reward_hallway +                   # Hallway achievement
            scales['reward_terminal_scale'] * reward_terminal +                 # Terminal success
            scales['punishment_terminal_scale'] * punishment_terminal +         # Terminal failure penalty
            scales['punishment_room_scale'] * punishment_room +                 # Location-based penalties
            scales['wall_collision_scale'] * wall_collision_penalty +           # Collision avoidance
            scales['reward_door_approach_scale'] * reward_door_approach +       # Door approach efficiency
            scales['reward_approach_door_scale'] * reward_approach_door         # Door orientation reward
        )
        
        return total_reward