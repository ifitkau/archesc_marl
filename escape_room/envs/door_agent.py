"""
Door controller agent implementation for the escape room environment

This module implements the door controller agent that dynamically positions doors
between rooms to facilitate or challenge the navigator agent's path to the terminal.
The door controller learns to optimize door placements based on various reward signals
including alignment with terminal position, navigation efficiency, and success rates.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math
from math import log
import time


class DoorControllerAgent:
    """
    The door controller agent controls the position of doors between rooms.
    
    This agent receives observations about the navigator's position, terminal location,
    and current door positions, then selects new door positions from a discrete set
    of options. The agent is rewarded for helping the navigator reach the terminal
    efficiently while learning strategic door placement.
    
    Key Features:
    - Multi-discrete action space (separate actions for each room's door)
    - Spatial awareness of environment geometry
    - Multiple reward components (alignment, efficiency, success)
    - Episode-based learning with success tracking
    """
    
    def __init__(self, config=None, rng=None, door_position=2.0, door_position_min=0.6, door_position_max=5.4):
        """
        Initialize the door controller agent
        
        Args:
            config (dict): Configuration dictionary with environment parameters
            rng (np.random.RandomState): Random number generator for reproducible behavior
            door_position (float): Initial door position for all rooms
            door_position_min (float): Minimum allowed door position within rooms
            door_position_max (float): Maximum allowed door position within rooms
        """
        self.name = "door_controller"
        
        # =====================================
        # DOOR POSITION MANAGEMENT
        # =====================================
        # Track door positions for each room independently (relative positions within each room)
        self.door_positions = {
            'roomA': door_position,    # Leftmost room door position
            'roomB': door_position,    # Middle room door position  
            'roomC': door_position     # Rightmost room door position
        }
        
        # Valid range for door positions within any room
        self.door_position_min = door_position_min  # Minimum relative position (usually 0.6m from room edge)
        self.door_position_max = door_position_max  # Maximum relative position (usually 5.4m from room edge)

        # =====================================
        # EPISODE STATE TRACKING
        # =====================================
        # Track navigator's progress through the environment
        self.navigator_has_reached_hallway = False  # Has navigator entered the central hallway?
        self.starting_room = None                   # Which room did navigator start in this episode?

        # Reward tracking flags - ensure each reward type is only given once per episode
        self.has_acted_this_episode = False         # Has door controller taken any action?
        self.alignment_reward_given = False         # Has alignment reward been calculated?
        self.end_rewards_given = False              # Have end-of-episode rewards been given?
        
        # =====================================
        # REWARD COMPONENT TRACKING
        # =====================================
        # Detailed breakdown of reward sources for analysis and debugging
        self.reward_components = {
            "roomA_alignment_reward": 0.0,      # Reward for roomA door alignment with terminal
            "roomB_alignment_reward": 0.0,      # Reward for roomB door alignment with terminal
            "roomC_alignment_reward": 0.0,      # Reward for roomC door alignment with terminal
            "door_placement_reward": 0.0,       # Reward for optimal door placement relative to navigator start
            "hallway_transition_reward": 0.0,   # Reward when navigator successfully enters hallway
            "actual_steps_reward": 0.0,         # Reward based on efficiency of navigator's path
            "path_quality_reward": 0.0,         # Reward for enabling straight-line paths
            "terminal_steps_reward": 0.0,       # Reward based on total steps to reach terminal
        }
        
        # =====================================
        # ACTION AND OBSERVATION SPACES
        # =====================================
        # Number of discrete positions each door can be placed at
        self.num_positions = 10 if config is None else config.get("discrete_door_positions", 10)
        
        # Observation space: Environment state that door controller can perceive
        self.observation_space = spaces.Box(
            low=np.array([
                # Navigator's current room category (0-4: roomA, roomB, roomC, hallway, unknown)
                0,
                # Navigator's normalized position in world coordinates
                0, 0,                             # [x, z] position normalized to [0,1]
                # All door positions as absolute world coordinates (normalized)
                0, 0,                             # Door A absolute position [x, z] 
                0, 0,                             # Door B absolute position [x, z]
                0, 0,                             # Door C absolute position [x, z]
                # Door controller action capability flag
                0,                                # Can door controller act this episode? (0/1)
                # Terminal (goal) information
                0, 0,                             # Terminal position [x, z] normalized
                # Navigator's orientation relative to terminal
                -1,                               # Dot product of navigator direction with direction to terminal
            ], dtype=np.float32),
            high=np.array([
                # Navigator's current room category
                4,                                # Maximum room category value
                # Navigator's normalized position  
                1, 1,                             # Maximum normalized coordinates
                # Door positions (normalized to world size)
                1, 1,                             # Door A maximum coordinates
                1, 1,                             # Door B maximum coordinates  
                1, 1,                             # Door C maximum coordinates
                # Door controller capability
                1,                                # Maximum capability flag value
                # Terminal position
                1, 1,                             # Maximum terminal coordinates
                # Navigator orientation
                1,                                # Maximum dot product value
            ], dtype=np.float32)
        )
        
        # Action space: Multi-discrete actions for each room's door
        # For each room: 0 = keep current position, 1-(num_positions) = move to specific position
        self.action_space = spaces.MultiDiscrete([
            self.num_positions + 1,  # Actions for Room A door (0=keep, 1-10=positions)
            self.num_positions + 1,  # Actions for Room B door (0=keep, 1-10=positions)
            self.num_positions + 1   # Actions for Room C door (0=keep, 1-10=positions)
        ])
        
        # =====================================
        # PERFORMANCE TRACKING
        # =====================================
        # Track success rates and path quality over recent episodes
        self.success_history = [0] * 5             # Binary success/failure for last 5 episodes
        
        # Path quality metrics for learning optimization
        self.last_path_quality = 0.0               # Path efficiency from most recent episode
        self.path_quality_history = []             # Historical path quality data
        self.avg_path_quality = 0.0                # Rolling average of path quality

        # =====================================
        # EXPECTED PERFORMANCE BASELINES
        # =====================================
        # Calculate expected step counts for each starting room to terminal
        # Used for normalizing efficiency rewards across different starting positions
        self.room_expected_steps = {}
        self.room_to_door_expected_steps = {}
        
        if config:
            # Get terminal location from config
            terminal_x, terminal_z = config.get("terminal_location", [18.4, 5.95])
            
            # Define approximate room center positions for distance calculations
            room_centers = {
                'roomA': [3.0, 0, 2.5],     # Center of leftmost room
                'roomB': [9.2, 0, 2.5],     # Center of middle room
                'roomC': [15.4, 0, 2.5],    # Center of rightmost room
            }
            
            # Calculate expected steps based on straight-line distance and typical movement speed
            for room, center in room_centers.items():
                # Euclidean distance from room center to terminal
                distance = np.sqrt((center[0] - terminal_x)**2 + (center[2] - terminal_z)**2)
                # Estimate steps needed (assuming ~0.15 units per step + overhead)
                self.room_expected_steps[room] = (distance / 0.15) + 10
            
            # Room offset positions for coordinate transformations
            room_offsets = {'roomA': 0.0, 'roomB': 9.2, 'roomC': 15.4}

        # =====================================
        # RANDOM NUMBER GENERATION
        # =====================================
        # Initialize random number generator for reproducible behavior
        self.rng = rng if rng is not None else np.random.RandomState()

    def calculate_path_quality_reward(self, miniworld_env):
        """
        Calculate reward based on how straight/efficient the navigator's path was to the door
        
        This method analyzes the navigator's actual path versus the optimal straight-line path
        from their starting position to the door they used. Higher efficiency (closer to straight-line)
        results in higher rewards, encouraging the door controller to place doors optimally.
        
        Args:
            miniworld_env: Reference to the MiniWorld environment containing path data
            
        Returns:
            float: Reward value based on path efficiency (0-10 scale)
        """
        # Only calculate if we have a recorded path and know which room was used
        if not hasattr(miniworld_env, 'current_path') or len(miniworld_env.current_path) < 2:
            return 0.0
        
        # Determine which room the navigator came from to reach the hallway
        from_room = self.starting_room
        if hasattr(miniworld_env, 'previous_room'):
            from_room = miniworld_env.previous_room
        
        # Only calculate for valid room transitions
        if from_room not in ['roomA', 'roomB', 'roomC']:
            return 0.0
        
        # Get the door position that was actually used
        door_pos = self.door_positions.get(from_room, 2.0)
        
        # Room boundary definitions for coordinate transformation
        room_boundaries = {
            'roomA': (0.0, 6.0),       # Left edge to right edge of room A
            'roomB': (6.2, 12.2),      # Left edge to right edge of room B  
            'roomC': (12.4, 18.4)      # Left edge to right edge of room C
        }
        
        # Convert relative door position to absolute world coordinates
        room_left, _ = room_boundaries.get(from_room, (0.0, 6.0))
        door_x = door_pos + room_left   # Absolute X coordinate of door
        door_z = 5.0                    # Doors are always at Z=5.0 (entrance to hallway)
        door_position = np.array([door_x, 0, door_z])
        
        # Get navigator's starting position (first point in recorded path)
        start_pos = np.array(miniworld_env.current_path[0])
        
        # Calculate optimal straight-line distance from start to door
        straight_line_dist = np.linalg.norm(start_pos - door_position)
        
        # Calculate actual distance traveled along the recorded path
        actual_path_dist = 0
        for i in range(1, len(miniworld_env.current_path)):
            actual_path_dist += np.linalg.norm(
                np.array(miniworld_env.current_path[i]) - 
                np.array(miniworld_env.current_path[i-1])
            )
        
        # Calculate path efficiency ratio (1.0 = perfect straight line, <1.0 = detours taken)
        if actual_path_dist > 0 and straight_line_dist > 0:
            path_efficiency = min(1.0, straight_line_dist / actual_path_dist)
            
            # Store in reward components for debugging/analysis
            self.reward_components["path_quality_reward"] = path_efficiency * 10.0
            
            return path_efficiency * 10.0  # Scale to meaningful reward range
        
        return 0.0

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
        
    def reset(self, miniworld_env):
        """
        Reset the agent state for a new episode
        
        This method is called at the beginning of each new episode to reset
        tracking variables and prepare for fresh learning. It maintains
        historical data while clearing episode-specific flags.
        
        Args:
            miniworld_env: Reference to the MiniWorld environment
        """
        # Update success history - shift left and add placeholder for new episode
        # Success/failure will be recorded at episode end
        self.success_history = self.success_history[1:] + [0]
        
        # Reset all reward tracking flags to ensure rewards are only given once per episode
        self.alignment_reward_given = False         # Terminal alignment reward eligibility
        self.door_placement_reward_given = False    # Door placement optimization reward eligibility
        self.hallway_reward_given = False           # Hallway transition reward eligibility
        self.actual_steps_reward_given = False      # Step efficiency reward eligibility
        self.end_rewards_given = False              # End-of-episode reward eligibility
        
        # Reset episode state tracking
        self.navigator_has_reached_hallway = False  # Navigator hasn't reached hallway yet
        
        # Clear all reward components for fresh tracking
        self.reward_components = {
            "roomA_alignment_reward": 0.0,      # Reset room A alignment tracking
            "roomB_alignment_reward": 0.0,      # Reset room B alignment tracking
            "roomC_alignment_reward": 0.0,      # Reset room C alignment tracking
            "door_placement_reward": 0.0,       # Reset door placement reward
            "hallway_transition_reward": 0.0,   # Reset hallway transition reward
            "actual_steps_reward": 0.0,         # Reset step efficiency reward
            "step_efficiency_reward": 0.0,      # Reset overall efficiency reward
            "path_quality_reward": 0.0,         # Reset path quality reward
            "terminal_steps_reward": 0.0,       # Reset terminal completion reward
        }
        
        # Record the starting room for this episode - used for targeted rewards
        self.starting_room = miniworld_env._get_current_room()

    def get_observation(self, miniworld_env):
        """
        Get the agent's observation from the environment
        
        Constructs a normalized observation vector containing information about:
        - Navigator's current position and room
        - All door positions in world coordinates  
        - Terminal location and navigator's orientation toward it
        - Door controller's action capability
        
        Args:
            miniworld_env: Reference to the MiniWorld environment
            
        Returns:
            np.array: Normalized observation vector for the door controller
        """
        # Get references to key environment components
        agent = miniworld_env.agent                    # Navigator agent
        terminal_loc = miniworld_env.terminal_location  # Goal position [x, z]
        world_width = miniworld_env.world_width         # Environment width
        world_depth = miniworld_env.world_depth         # Environment depth
        
        # =====================================
        # NAVIGATOR POSITION AND ORIENTATION
        # =====================================
        
        # Get navigator's current position and normalize to world coordinates
        navigator_pos = miniworld_env._normalize_position(agent.pos)
        
        # Calculate navigator's normalized position for observation
        norm_agent_pos = np.array([
            agent.pos[0] / world_width,   # X coordinate normalized
            0,                            # Skip Y coordinate (height not used)
            agent.pos[2] / world_depth    # Z coordinate normalized
        ])
        
        # Normalize coordinates to [0,1] range for consistent observation space
        norm_agent_pos[0] = (norm_agent_pos[0] + 1) / 2  # X to [0,1]
        norm_agent_pos[2] = (norm_agent_pos[2] + 1) / 2  # Z to [0,1]

        # =====================================
        # TERMINAL POSITION AND DIRECTION
        # =====================================
        
        # Normalize terminal position to same coordinate system
        norm_terminal_pos = np.array(terminal_loc) / np.array([world_width, world_depth])
        norm_terminal_pos[0] = (norm_terminal_pos[0] + 1) / 2  # X to [0,1]
        norm_terminal_pos[1] = (norm_terminal_pos[1] + 1) / 2  # Z to [0,1]

        # Calculate direction vectors for orientation analysis
        dx = terminal_loc[0] - agent.pos[0]  # X distance to terminal
        dz = terminal_loc[1] - agent.pos[2]  # Z distance to terminal
        
        # Direction vector from navigator to terminal (normalized)
        direction_vector = np.array([dx, -dz])  # Flip Z for correct orientation
        direction_length = np.linalg.norm(direction_vector)
        if direction_length > 0:
            direction_vector = direction_vector / direction_length
        else:
            direction_vector = np.array([1.0, 0.0])  # Default forward if at same position
        
        # Navigator's current facing direction (always normalized)
        agent_dir_vec = np.array([
            np.cos(agent.dir),
            np.sin(agent.dir)
        ])
        
        # Calculate alignment between navigator's direction and optimal direction to terminal
        # Dot product gives cosine of angle: 1.0 = perfectly aligned, -1.0 = opposite direction
        dot_product = np.dot(direction_vector, agent_dir_vec)

        # =====================================
        # ROOM CATEGORIZATION
        # =====================================
        
        # Determine navigator's current room and convert to numerical category
        current_room = miniworld_env._get_current_room() 
        room_category = miniworld_env._get_room_category(current_room)
        
        # =====================================
        # DOOR POSITION SPATIAL ENCODING
        # =====================================
        
        # Room layout constants for spatial coordinate transformation
        room_offsets = {
            'roomA': 0.0,    # Room A starts at world X=0
            'roomB': 6.2,    # Room B starts at world X=6.2
            'roomC': 12.4    # Room C starts at world X=12.4
        }
        
        # All doors are at the same Z coordinate (entrance to hallway)
        door_z = 5.0
        norm_door_z = door_z / miniworld_env.world_depth  # Normalize Z coordinate
        
        # Initialize spatial door position data structure
        door_positions_spatial = {
            'roomA': {'x': 0.0, 'z': norm_door_z},
            'roomB': {'x': 0.0, 'z': norm_door_z},
            'roomC': {'x': 0.0, 'z': norm_door_z}
        }
        
        # Convert relative door positions to absolute world coordinates
        for room, pos in self.door_positions.items():
            # Calculate absolute X position: relative position + room offset
            door_x_actual = pos + room_offsets.get(room, 0.0)
            
            # Normalize to world width for spatial awareness in observation
            door_positions_spatial[room]['x'] = door_x_actual / miniworld_env.world_width
        
        # =====================================
        # OBSERVATION VECTOR CONSTRUCTION
        # =====================================
        
        # Build complete observation array with all relevant state information
        obs = np.array([
            # Navigator's current location context
            float(room_category),                           # Which room/area is navigator in?
            
            # Navigator's spatial position (normalized to [0,1])
            norm_agent_pos[0],                             # X position in world
            norm_agent_pos[1],                             # Z position in world

            # All door positions as absolute world coordinates (for spatial reasoning)
            door_positions_spatial['roomA']['x'],          # Room A door X position
            door_positions_spatial['roomA']['z'],          # Room A door Z position
            door_positions_spatial['roomB']['x'],          # Room B door X position 
            door_positions_spatial['roomB']['z'],          # Room B door Z position
            door_positions_spatial['roomC']['x'],          # Room C door X position
            door_positions_spatial['roomC']['z'],          # Room C door Z position

            # Door controller capability flag
            1.0 if miniworld_env.door_can_act else 0.0,   # Can controller act this episode?
            
            # Goal information
            norm_terminal_pos[0],                          # Terminal X position
            norm_terminal_pos[1],                          # Terminal Z position

            # Navigator's orientation relative to optimal path
            dot_product,                                   # Alignment with direction to terminal
        ], dtype=np.float32)
        
        # Safety check: Replace any NaN or infinite values with safe defaults
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0)
        
        return obs
    
    def process_action(self, miniworld_env, action):
        """
        Process the agent's multi-discrete action in the environment
        
        Takes the door controller's action decisions and implements them by moving
        doors to new positions. Handles both individual room actions and applies
        any associated rewards for the door placement decisions.
        
        Args:
            miniworld_env: Reference to the MiniWorld environment
            action: Multi-discrete action array [roomA_action, roomB_action, roomC_action]
                   For each room: 0 = keep current position, 1-num_positions = move to specific position
        
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
                - observation: Updated state observation after action
                - reward: Immediate reward for this action  
                - terminated: Episode termination flag (always False for door controller)
                - truncated: Episode truncation flag (always False for door controller)
                - info: Dictionary with action details and reward breakdown
        """
        # =====================================
        # RANDOM NUMBER GENERATOR SAFETY CHECK
        # =====================================
        # Ensure we have a properly seeded RNG for any stochastic behavior
        if not hasattr(self, 'rng') or self.rng is None:
            # Create default RNG if missing (shouldn't happen in normal operation)
            self.rng = np.random.RandomState(42)
            print("Warning: Door agent RNG was not properly initialized")

        # =====================================
        # ACTION PARSING AND VALIDATION
        # =====================================
        room_names = ['roomA', 'roomB', 'roomC']
        
        # Handle multi-discrete action format
        if isinstance(action, (list, np.ndarray)) and len(action) == 3:
            roomA_action, roomB_action, roomC_action = action
            room_actions = [roomA_action, roomB_action, roomC_action]
        else:
            # Fallback for legacy single-action interface
            print(f"Warning: Received scalar action {action}, expected multi-discrete.")
            # Convert to 3 identical actions for all rooms
            room_actions = [action, action, action]
        
        # =====================================
        # DOOR POSITION UPDATES
        # =====================================
        # Process actions for each room independently
        new_door_positions = self.door_positions.copy()
        position_range = self.door_position_max - self.door_position_min
        self.current_navigator_room = miniworld_env._get_current_room()
        
        for i, (room_name, room_action) in enumerate(zip(room_names, room_actions)):
            if room_action == 0:
                # Action 0: Keep current door position for this room
                current_position = self.door_positions[room_name]
                # Still call move function to ensure proper rendering/synchronization
                self._move_specific_room_door(miniworld_env, room_name, current_position)
            else:
                # Actions 1-N: Move to specific position
                # Convert action to 0-based index for position calculation
                position_idx = room_action - 1
                
                # Calculate new relative position within room bounds
                new_position = self.door_position_min + (position_idx / (self.num_positions - 1)) * position_range
                new_door_positions[room_name] = new_position
                
                # Execute door movement in environment
                self._move_specific_room_door(miniworld_env, room_name, new_position)
        
        # Update internal door position tracking
        self.door_positions = new_door_positions
        self.door_positions = dict(sorted(self.door_positions.items()))  # Keep consistent ordering
        
        # =====================================
        # TERMINATION STATUS
        # =====================================
        # Door controller never terminates episodes (only navigator can do that)
        terminated = False
        truncated = False
        
        # =====================================
        # IMMEDIATE REWARD CALCULATION
        # =====================================
        total_reward = 0.0
        
        # Check if door controller is allowed to act in this episode
        door_can_act = False
        if hasattr(miniworld_env, 'door_can_act'):
            door_can_act = miniworld_env.door_can_act
        
        # Calculate door placement reward (only when door can act and hasn't been given yet)
        door_placement_reward = 0.0
        if door_can_act and not self.door_placement_reward_given:
            door_placement_reward = self.calculate_door_placement_reward(miniworld_env)
            self.door_placement_reward_given = True
            
            # Apply reward scaling from configuration
            if 'door_controller_reward_scales' in miniworld_env.config:
                scales = miniworld_env.config['door_controller_reward_scales']
                door_placement_reward *= scales.get('reward_door_placement_scale', 0.0)
            
            total_reward += door_placement_reward
        
        # =====================================
        # INFORMATION DICTIONARY
        # =====================================
        # Provide detailed information about the action and its effects
        info = {
            'door_positions': self.door_positions,                    # Current door positions after action
            'reward_components': self.reward_components.copy(),       # Breakdown of all reward sources
            'rewarded_door_room': self.current_navigator_room if hasattr(self, 'current_navigator_room') 
                                and self.current_navigator_room in ['roomA', 'roomB', 'roomC'] else None,
        }
        
        # Return standard Gym environment tuple
        return self.get_observation(miniworld_env), total_reward, False, False, info

    def calculate_alignment_reward(self, miniworld_env):
        """
        Calculate alignment reward based on door positions relative to terminal location
        
        This method rewards door placements that align strategically with the terminal position.
        If the terminal is on the left side of the environment, left door positions are rewarded.
        If the terminal is on the right side, right door positions are rewarded.
        If the terminal is in the middle, center door positions are rewarded.
        
        Args:
            miniworld_env: Reference to the MiniWorld environment
            
        Returns:
            float: Total alignment reward across all rooms
        """
        # Reset all alignment reward components to start fresh
        for key in self.reward_components:
            if key.endswith("_alignment_reward"):
                self.reward_components[key] = 0.0
        
        # =====================================
        # TERMINAL POSITION ANALYSIS
        # =====================================
        # Determine where the terminal is located relative to environment center
        terminal_x = miniworld_env.terminal_location[0]
        world_width = miniworld_env.world_width
        hallway_mid = world_width / 2
        
        # Categorize terminal position (affects optimal door placement strategy)
        terminal_position = "right" if terminal_x > hallway_mid + 3.0 else "left" if terminal_x < hallway_mid - 3.0 else "middle"
        
        # =====================================
        # ROOM-BY-ROOM ALIGNMENT CALCULATION
        # =====================================
        total_alignment_reward = 0.0
        
        # Process each room's door position independently
        for room_name in ['roomA', 'roomB', 'roomC']:
            # Get current door position for this room
            door_pos = self.door_positions.get(room_name, 3.0)
            min_pos = self.door_position_min  # Usually 0.6
            max_pos = self.door_position_max  # Usually 5.4
            
            # Normalize door position to [0,1] range for consistent reward calculation
            norm_pos = (door_pos - min_pos) / (max_pos - min_pos)  # 0 = leftmost, 1 = rightmost

            scale = 1.0  # Reward scaling factor
            
            # Calculate alignment reward based on terminal position strategy
            room_alignment_reward = 0.0
            if terminal_position == "left":
                # Terminal is on left: reward leftward door positions
                room_alignment_reward = 1.0 - norm_pos * scale  # Max reward for leftmost position
            elif terminal_position == "right":
                # Terminal is on right: reward rightward door positions  
                room_alignment_reward = norm_pos * scale        # Max reward for rightmost position
            else:  # middle
                # Terminal is in center: reward center door positions
                room_alignment_reward = (1.0 - 2.0 * abs(norm_pos - 0.5)) * scale  # Max reward for center position
                
            # Store individual room's alignment reward for detailed analysis
            self.reward_components[f"{room_name}_alignment_reward"] = room_alignment_reward
            
            # Add to total alignment reward
            total_alignment_reward += room_alignment_reward
        
        # Return average alignment reward across all rooms
        return total_alignment_reward
    
    def calculate_door_placement_reward(self, miniworld_env):
        """
        Calculate reward based on Euclidean distance from navigator start position to door
        
        This reward encourages placing doors close to where the navigator starts, making
        their initial path more efficient. Uses straight-line distance calculation.
        
        Args:
            miniworld_env: Reference to the MiniWorld environment
            
        Returns:
            float: Distance-based door placement reward
        """
        # Only calculate if we have navigator's starting position
        if not hasattr(miniworld_env, 'navigator_start_pos') or miniworld_env.navigator_start_pos is None:
            return 0.0
        
        start_pos = miniworld_env.navigator_start_pos
        current_room = self.starting_room
        
        # Only calculate for valid room starts
        if current_room not in ['roomA', 'roomB', 'roomC']:
            return 0.0
        
        # =====================================
        # COORDINATE SYSTEM MAPPING
        # =====================================
        # Room boundary definitions for coordinate transformation
        room_boundaries = {
            'roomA': (0.0, 6.0),       # Left and right edges of room A
            'roomB': (6.2, 12.2),      # Left and right edges of room B
            'roomC': (12.4, 18.4)      # Left and right edges of room C
        }
        
        # Get door position for the room navigator started in
        door_pos = self.door_positions.get(current_room, 3.0)
        
        # =====================================
        # ABSOLUTE DOOR POSITION CALCULATION
        # =====================================
        # Convert relative door position to absolute world coordinates
        room_left, _ = room_boundaries.get(current_room, (0.0, 6.0))
        door_x = door_pos + room_left   # Absolute X coordinate of door
        door_z = 5.0                    # All doors are at Z=5.0 (hallway entrance)
        
        # Calculate straight-line Euclidean distance from start to door
        distance = np.sqrt(
            (start_pos[0] - door_x)**2 + 
            (start_pos[2] - door_z)**2
        )
        
        # =====================================
        # DISTANCE-TO-REWARD CONVERSION
        # =====================================
        # Convert distance to reward (shorter distance = higher reward)
        max_reward = 10.0       # Maximum reward for optimal door placement
        min_distance = 1.0      # Theoretical minimum possible distance
        
        if distance <= min_distance:
            reward = max_reward
        else:
            # Inverse relationship: reward decreases as distance increases
            reward = max_reward * (min_distance / distance)
        
        # Store reward in tracking system for analysis
        self.reward_components["door_placement_reward"] = reward
        
        return reward
    
    def check_hallway_transition(self, miniworld_env):
        """
        Check if navigator has transitioned to hallway and calculate associated rewards
        
        This is a key reward checkpoint that triggers multiple reward calculations when
        the navigator successfully enters the central hallway from any room. It calculates:
        - Hallway transition reward (for reaching the hallway)
        - Step efficiency reward (for path efficiency)
        - Alignment reward (for strategic door placement)
        - Path quality reward (for enabling straight paths)
        
        Args:
            miniworld_env: Reference to the MiniWorld environment
            
        Returns:
            float: Total reward for hallway transition and associated achievements
        """
        # If rewards already given, skip to avoid duplicate rewards
        if self.hallway_reward_given:
            return 0.0
                
        current_room = miniworld_env._get_current_room()
        
        # =====================================
        # HALLWAY TRANSITION DETECTION
        # =====================================
        # Check if navigator has successfully reached the central hallway
        if current_room == 'roomD' and not self.navigator_has_reached_hallway:  # roomD is the hallway
            self.navigator_has_reached_hallway = True
            
            # Determine which room the navigator came from for targeted rewards
            from_room = self.starting_room
            
            # Use more accurate previous room tracking if available
            if hasattr(miniworld_env, 'previous_room'):
                from_room = miniworld_env.previous_room
            
            # =====================================
            # BASE HALLWAY TRANSITION REWARD
            # =====================================
            # Calculate primary reward for successfully reaching hallway
            base_reward = 100.0
            hallway_reward = base_reward
            
            # Apply configuration scaling for hallway transition reward
            if 'door_controller_reward_scales' in miniworld_env.config:
                scales = miniworld_env.config['door_controller_reward_scales']
                scale_value = scales.get('hallway_transition_reward', 0.01)
                hallway_reward = base_reward * scale_value
            
            # Store in reward component tracking
            self.reward_components["hallway_transition_reward"] = hallway_reward
            
            # =====================================
            # STEP EFFICIENCY REWARD
            # =====================================
            # Reward door controller for enabling efficient navigator paths
            steps_reward = 0.0
            if not self.actual_steps_reward_given:
                self.actual_steps_reward_given = True
                actual_steps = miniworld_env.step_count
                max_reward = 10.0
                optimal_steps = 30.0  # Expected optimal step count to hallway
                
                # Calculate efficiency-based reward
                if actual_steps <= optimal_steps:
                    steps_reward = max_reward
                else:
                    # Diminishing returns for longer paths
                    steps_reward = max_reward * (optimal_steps / actual_steps)
                    
                # Apply configuration scaling
                if 'door_controller_reward_scales' in miniworld_env.config:
                    scales = miniworld_env.config['door_controller_reward_scales']
                    steps_reward *= scales.get('reward_actual_steps_scale', 1.0)
                    
                self.reward_components["actual_steps_reward"] = steps_reward
                
            # =====================================
            # STRATEGIC ALIGNMENT REWARD
            # =====================================
            # Reward alignment between door used and terminal position
            alignment_reward = 0.0
            if not self.alignment_reward_given and from_room in ['roomA', 'roomB', 'roomC']:
                # Clear previous alignment rewards
                for key in self.reward_components:
                    if key.endswith("_alignment_reward"):
                        self.reward_components[key] = 0.0
                
                # Analyze terminal position for strategic door placement
                terminal_x = miniworld_env.terminal_location[0]
                world_width = miniworld_env.world_width
                hallway_mid = world_width / 2
                
                # Determine optimal door strategy based on terminal location
                terminal_position = "right" if terminal_x > hallway_mid + 3.0 else "left" if terminal_x < hallway_mid - 3.0 else "middle"
                
                # Get door position for the room that was actually used
                door_pos = self.door_positions.get(from_room, 3.0)
                min_pos = self.door_position_min  # Usually 0.6
                max_pos = self.door_position_max  # Usually 5.4
                norm_pos = (door_pos - min_pos) / (max_pos - min_pos)  # Normalize to [0,1]

                scale = 1.0  # Alignment reward scaling factor
                
                # Calculate strategic alignment reward based on terminal position
                room_alignment_reward = 0.0
                if terminal_position == "left":
                    # Terminal on left: reward leftward door positions
                    room_alignment_reward = 1.0 - norm_pos * scale
                elif terminal_position == "right":
                    # Terminal on right: reward rightward door positions
                    room_alignment_reward = norm_pos * scale
                else:  # middle
                    # Terminal in center: reward center door positions
                    room_alignment_reward = (1.0 - 2.0 * abs(norm_pos - 0.5)) * scale
                    
                # Store alignment reward for the room that was used
                self.reward_components[f"{from_room}_alignment_reward"] = room_alignment_reward
                alignment_reward = room_alignment_reward
                
                # Apply configuration scaling for alignment reward
                if 'door_controller_reward_scales' in miniworld_env.config:
                    scales = miniworld_env.config['door_controller_reward_scales']
                    alignment_scale = scales.get('reward_terminal_alignment_scale', 0.0)
                    alignment_reward *= alignment_scale
                
                self.alignment_reward_given = True
            
            # =====================================
            # PATH QUALITY REWARD
            # =====================================
            # Reward door controller for enabling high-quality (straight) paths
            path_quality_reward = 0.0
            if not hasattr(self, 'path_quality_given') or not self.path_quality_given:
                path_quality_reward = self.calculate_path_quality_reward(miniworld_env)
                self.path_quality_given = True
                
                # Apply configuration scaling for path quality
                if 'door_controller_reward_scales' in miniworld_env.config:
                    scales = miniworld_env.config['door_controller_reward_scales']
                    path_quality_reward *= scales.get('reward_path_quality_scale', 0.0)
            
            # Set flag to prevent duplicate reward calculation
            self.hallway_reward_given = True
            
            # Calculate total combined reward
            total_reward = hallway_reward + steps_reward + alignment_reward + path_quality_reward
            
            return total_reward
        
        return 0.0
    
    def calculate_hallway_transition_reward(self, miniworld_env):
        """
        Provide a one-time reward when navigator reaches hallway
        
        This is a simpler version of hallway reward calculation that only provides
        the base transition reward without additional efficiency or alignment bonuses.
        
        Args:
            miniworld_env: Reference to the MiniWorld environment
            
        Returns:
            float: Base hallway transition reward
        """
        # Prevent duplicate reward calculation
        if self.hallway_reward_given:
            return 0.0
                
        # Mark reward as given to ensure single calculation per episode
        self.hallway_reward_given = True
        
        # Calculate base reward for reaching hallway
        base_reward = 100.0
        hallway_reward = base_reward
        
        # Apply configuration scaling
        if 'door_controller_reward_scales' in miniworld_env.config:
            scales = miniworld_env.config['door_controller_reward_scales']
            scale_value = scales.get('hallway_transition_reward', 0.5)
            hallway_reward = base_reward * scale_value
        
        # Store in reward component tracking
        self.reward_components["hallway_transition_reward"] = hallway_reward
        
        return hallway_reward
    
    def calculate_actual_steps_reward(self, miniworld_env):
        """
        Calculate reward based on step efficiency for reaching hallway
        
        Compares the navigator's actual step count to reach the hallway against
        the expected optimal step count, rewarding more efficient paths.
        
        Args:
            miniworld_env: Reference to the MiniWorld environment
            
        Returns:
            float: Step efficiency reward based on path optimization
        """
        # Prevent duplicate reward calculation
        if self.actual_steps_reward_given:
            return 0.0
        
        self.actual_steps_reward_given = True
        
        # Get actual steps taken by navigator
        actual_steps = miniworld_env.step_count
        
        # =====================================
        # EFFICIENCY CALCULATION
        # =====================================
        # Calculate expected vs actual step efficiency
        room_center = self.room_centers[self.starting_room]
        door_x = self.door_positions[self.starting_room] + self.room_offsets[self.starting_room]

        if self.starting_room in room_center:
            door_z = 5.0  # Standard door Z coordinate
            
            # Calculate straight-line distance from room center to door
            distance_to_door = np.sqrt((room_center[0] - door_x)**2 + (room_center[2] - door_z)**2)
            expected_steps = distance_to_door / 0.15  # Assume ~0.15 units per step
            
            # Calculate efficiency ratio
            efficiency = expected_steps / actual_steps
            
            # =====================================
            # EFFICIENCY-TO-REWARD MAPPING
            # =====================================
            # Convert efficiency ratio to reward value
            min_reward = 0.0        # Minimum reward for very inefficient paths
            max_reward = 10.0       # Maximum reward for optimal paths
            min_efficiency = 0.1    # Below 10% efficiency gets minimum reward
            max_efficiency = 1.0    # 100%+ efficiency gets maximum reward
            
            if efficiency >= max_efficiency:
                reward = max_reward
            elif efficiency <= min_efficiency:
                reward = min_reward
            else:
                # Linear interpolation between min and max rewards
                reward = min_reward + (max_reward - min_reward) * ((efficiency - min_efficiency) / (max_efficiency - min_efficiency))
        else:
            reward = 0.0  # No reward if starting room data unavailable
        
        # Apply configuration scaling
        if 'door_controller_reward_scales' in miniworld_env.config:
            scales = miniworld_env.config['door_controller_reward_scales']
            reward *= scales.get('reward_actual_steps_scale', 1.0)
        
        # Store in reward component tracking
        self.reward_components["actual_steps_reward"] = reward
        
        return reward

    def _move_door(self, miniworld_env, new_position):
        """
        Legacy method for moving a single door (kept for backward compatibility)
        
        This method moves the first available door to a new position. It's primarily
        used for single-door environments or as a fallback for older code.
        
        Args:
            miniworld_env: Reference to the MiniWorld environment
            new_position (float): New door position in world coordinates
        """
        # Reset all portal connections to start fresh
        miniworld_env.portals = []
        
        door_position = float(new_position)
        
        # Find the first available room (excluding hallway)
        first_top_room = next((room for room in miniworld_env.rooms if room != miniworld_env.roomD), None)
        
        if first_top_room is None:
            print("Warning: No top room found for door connection")
            return
        
        # Create new door connection with specified width
        door_width = miniworld_env.door_width
        miniworld_env.connect_rooms(
            first_top_room, 
            miniworld_env.roomD, 
            min_x=door_position - door_width/2, 
            max_x=door_position + door_width/2
        )
        
        # Update environment door position tracking
        miniworld_env.door_position = door_position
        for room_name in self.door_positions:
            self.door_positions[room_name] = door_position

        # Debug output if enabled
        if hasattr(miniworld_env, 'debug_mode') and miniworld_env.debug_mode:
            print(f"Door moved to position {door_position:.8f}")

    def _move_specific_room_door(self, miniworld_env, room_name, new_position):
        """
        Move door for a specific room with improved synchronization
        
        This method handles the complex process of moving a door for a specific room,
        including coordinate transformation, boundary checking, portal management,
        and state synchronization across all environment components.
        
        Args:
            miniworld_env: Reference to the MiniWorld environment
            room_name (str): Name of the room ('roomA', 'roomB', or 'roomC')
            new_position (float): New relative door position within the room
        """
        try:
            # =====================================
            # ROOM AND HALLWAY REFERENCES
            # =====================================
            # Get references to the specific room and central hallway
            top_room = getattr(miniworld_env, room_name)    # The room to move door for
            hallway = miniworld_env.roomD                   # Central hallway (constant)
            
            # =====================================
            # COORDINATE SYSTEM CONSTANTS
            # =====================================
            # Get room-specific configuration constraints
            room_ranges = miniworld_env.config.get("door_position_ranges", {})
            
            # Room layout constants - CRITICAL: Must be consistent across all code
            room_offsets = {
                'roomA': 0.0,    # Room A starts at world X=0.0
                'roomB': 6.2,    # Room B starts at world X=6.2
                'roomC': 12.4    # Room C starts at world X=12.4
            }
            
            # Standard room widths (distance from left edge to right edge)
            room_widths = {
                'roomA': 6.0,    # Room A width
                'roomB': 6.0,    # Room B width
                'roomC': 6.0     # Room C width
            }
            
            # =====================================
            # POSITION VALIDATION AND CLAMPING
            # =====================================
            # Get layout parameters for this specific room
            room_offset = room_offsets.get(room_name, 0.0)
            room_width = room_widths.get(room_name, 6.0)
            
            # Apply room-specific position constraints from configuration
            if room_name in room_ranges:
                min_pos, max_pos = room_ranges[room_name]
                # Clamp to the configured valid range for this room
                relative_position = max(min_pos, min(max_pos, new_position))
            else:
                # Fallback: Use 10-90% of room width as safe range
                min_pos = 0.1 * room_width
                max_pos = 0.9 * room_width
                relative_position = max(min_pos, min(max_pos, new_position))
            
            # Convert from relative position (within room) to absolute world coordinates
            absolute_position = room_offset + relative_position
            
            # =====================================
            # FINAL BOUNDARY SAFETY CHECK
            # =====================================
            # Ensure door fits within actual room boundaries with safety margins
            margin = miniworld_env.door_width/2 + 0.1  # Half door width plus safety buffer
            min_absolute = top_room.min_x + margin      # Leftmost safe position
            max_absolute = top_room.max_x - margin      # Rightmost safe position
            
            # Final clamping to physical room boundaries
            adjusted_position = max(min_absolute, min(max_absolute, absolute_position))
            
            # =====================================
            # STATE SYNCHRONIZATION
            # =====================================
            # CRITICAL: Store positions for all environment components to maintain consistency
            
            # 1. Store absolute positions for visualization and rendering
            if not hasattr(miniworld_env, 'absolute_door_positions'):
                miniworld_env.absolute_door_positions = {}
            miniworld_env.absolute_door_positions[room_name] = adjusted_position
            
            # 2. Store relative positions in door controller for decision making
            self.door_positions[room_name] = relative_position  # Use the clamped relative position
            self.door_positions = dict(sorted(self.door_positions.items()))  # Maintain consistent ordering
            
            # 3. Update miniworld_env individual door tracking with relative positions
            if not hasattr(miniworld_env, 'individual_door_positions'):
                miniworld_env.individual_door_positions = {}
            miniworld_env.individual_door_positions[room_name] = relative_position
            
            # 4. Backward compatibility: Update legacy door_position for roomA
            if room_name == 'roomA':
                miniworld_env.door_position = relative_position
            
            # =====================================
            # PORTAL CONNECTION UPDATE
            # =====================================
            # Remove existing door connection between this room and hallway
            miniworld_env.disconnect_rooms(top_room, hallway)
            
            # Create new door connection at the adjusted position
            door_width = miniworld_env.door_width
            miniworld_env.connect_rooms(
                top_room, 
                hallway, 
                min_x=adjusted_position - door_width/2,  # Left edge of door
                max_x=adjusted_position + door_width/2   # Right edge of door
            )
            
            # =====================================
            # RENDERING AND VISUALIZATION UPDATE
            # =====================================
            # Force regeneration of static visual data for updated door positions
            if hasattr(miniworld_env, '_gen_static_data'):
                miniworld_env._gen_static_data()
                
            # Force re-rendering in debug mode for immediate visual feedback
            if hasattr(miniworld_env, 'debug_mode') and miniworld_env.debug_mode and hasattr(miniworld_env, '_render_static'):
                miniworld_env._render_static()

            # =====================================
            # DEBUG OUTPUT
            # =====================================
            # Provide detailed logging for troubleshooting door movement
            if hasattr(miniworld_env, 'debug_mode') and miniworld_env.debug_mode:
                print(f"  Moving {room_name} door: input={new_position:.2f}, relative={relative_position:.2f}, absolute={absolute_position:.2f}, adjusted={adjusted_position:.2f}")
                
        except Exception as e:
            # Error handling with full traceback for debugging
            print(f"Error moving door for {room_name}: {e}")
            import traceback
            traceback.print_exc()  # Print full traceback for debugging
    
    def give_end_of_episode_rewards(self, miniworld_env):
        """
        Calculate and provide comprehensive end-of-episode rewards
        
        This method is called when a navigator episode completes (either successfully
        or unsuccessfully) and calculates various reward components based on the
        overall episode performance. It only executes once per episode to prevent
        reward duplication.
        
        Args:
            miniworld_env: Reference to the MiniWorld environment
            
        Returns:
            float: Total combined end-of-episode reward
        """
        # =====================================
        # DUPLICATE REWARD PREVENTION
        # =====================================
        # Only give rewards once per episode to prevent exploitation
        if self.end_rewards_given:
            return 0.0
        
        # Mark rewards as given FIRST to prevent any recursion or duplicate calls
        self.end_rewards_given = True
        
        # =====================================
        # EPISODE SUCCESS DETECTION
        # =====================================
        # Determine if navigator successfully reached the terminal
        navigator_success = False
        if hasattr(miniworld_env, '_episode_success'):
            navigator_success = miniworld_env._episode_success
        elif hasattr(miniworld_env, '_terminal_reached'):
            navigator_success = miniworld_env._terminal_reached
        
        # =====================================
        # STEP EFFICIENCY BONUS
        # =====================================
        # Special bonus for very fast completion (under 100 steps)
        step_efficiency_reward = 0.0
        if navigator_success and hasattr(miniworld_env, 'step_count') and miniworld_env.step_count < 100:
            step_efficiency_reward = 100.0  # Large bonus for exceptional performance
        
        self.reward_components["step_efficiency_reward"] = step_efficiency_reward
        
        # =====================================
        # SUCCESS REWARD CALCULATION
        # =====================================
        # Base reward for successful episode completion
        success_reward = 0.0
        terminal_steps_success_reward = 0.0
        
        if navigator_success:
            success_reward = 10  # Base success reward
            
            # =====================================
            # ROOM-NORMALIZED EFFICIENCY REWARD
            # =====================================
            # Additional reward based on step efficiency normalized by starting room
            if hasattr(miniworld_env, 'step_count') and self.starting_room in self.room_expected_steps:
                steps_taken = miniworld_env.step_count
                expected_steps = self.room_expected_steps[self.starting_room]
                
                # Calculate efficiency ratio (expected / actual)
                efficiency = expected_steps / steps_taken

                # Define reward scaling parameters
                min_reward = 0.0        # Minimum reward for poor efficiency
                max_reward = 10.0       # Maximum reward for perfect efficiency
                min_efficiency = 0.1    # Below 10% efficiency gets minimum reward
                max_efficiency = 1.0    # 100%+ efficiency gets maximum reward

                # Calculate efficiency-based reward
                if efficiency >= max_efficiency:
                    terminal_steps_reward = max_reward
                elif efficiency <= min_efficiency:
                    terminal_steps_reward = min_reward
                else:
                    # Linear interpolation between min and max
                    terminal_steps_reward = min_reward + (max_reward - min_reward) * ((efficiency - min_efficiency) / (max_efficiency - min_efficiency))
                
                # Apply configuration scaling
                if 'door_controller_reward_scales' in miniworld_env.config:
                    scales = miniworld_env.config['door_controller_reward_scales']
                    terminal_steps_reward *= scales.get('reward_terminal_steps_scale', 2.0)
                
                # Store in reward components for analysis
                self.reward_components["terminal_steps_reward"] = terminal_steps_reward
                
                # Add to total success reward
                terminal_steps_success_reward += terminal_steps_reward

        # =====================================
        # EPISODE RESULT RECORDING
        # =====================================
        # Update success history for future decision making
        self.record_episode_result(navigator_success)

        # =====================================
        # REWARD SCALING APPLICATION
        # =====================================
        # Apply configuration-based scaling to all reward components
        if 'door_controller_reward_scales' in miniworld_env.config:
            scales = miniworld_env.config['door_controller_reward_scales']
            success_reward *= scales.get('reward_success_scale', 0.1)
            step_efficiency_reward *= scales.get('reward_efficiency_scale', 0.1)

        # =====================================
        # HALLWAY PENALTY FOR FAILURE
        # =====================================
        # Penalize door controller if navigator never reached hallway
        hallway_reward = 0.0
        if not self.navigator_has_reached_hallway:
            hallway_reward = -10  # Penalty for not enabling hallway access
            self.reward_components["hallway_transition_reward"] = hallway_reward

        # =====================================
        # TOTAL REWARD CALCULATION
        # =====================================
        # Sum all reward components for final episode reward
        total_reward = hallway_reward + success_reward + terminal_steps_success_reward + step_efficiency_reward
        
        return total_reward
    
    def record_episode_result(self, success):
        """
        Record the result of the current episode in success history
        
        Updates the rolling success history buffer that tracks performance
        over the most recent episodes. This data can be used for adaptive
        learning strategies or performance analysis.
        
        Args:
            success (bool): Whether the episode was completed successfully
            
        Returns:
            None
        """
        # Update the most recent episode result in success history
        # (The placeholder 0 was added during reset, now we update it with actual result)
        self.success_history[-1] = 1.0 if success else 0.0