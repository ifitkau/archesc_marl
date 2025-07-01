"""
MiniWorld environment for the escape room scenario

This module implements the core 3D environment for the multi-agent escape room simulation.
It extends MiniWorld to create a multi-room layout with controllable doors, LIDAR sensing,
collision detection, and comprehensive agent tracking. The environment serves as the
foundation for both navigator and door controller agents to interact.

Key Features:
- Multi-room layout (3 main rooms + central hallway)
- Dynamic door positioning with individual room control
- LIDAR-based obstacle detection and collision avoidance
- Safe zone management around doors
- Comprehensive agent tracking and state management
- Room-based navigation and spatial awareness
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from miniworld.miniworld import MiniWorldEnv, Room
from miniworld.entity import Agent, Box
import time

class EscapeRoomBaseEnv(MiniWorldEnv):
    """
    Base MiniWorld environment for the escape room scenario
    
    This class implements the core 3D environment functionality without multi-agent
    specific code. It provides the physical world, room layout, door management,
    and sensing capabilities that both agents rely on.
    
    World Layout:
    - Room A (leftmost): X=[0, 6.0], Z=[0, 5.0]
    - Room B (middle): X=[6.2, 12.2], Z=[0, 5.0] 
    - Room C (rightmost): X=[12.4, 18.4], Z=[0, 5.0]
    - Hallway (roomD): X=[0, 18.4], Z=[5.2, 6.7]
    - Doors connect each room to the hallway at Z=5.0
    """
    
    def __init__(self, config=None, render_mode=None, view="top"):
        """
        Initialize the escape room environment
        
        Sets up the world geometry, door system, tracking variables, and rendering.
        Inherits from MiniWorldEnv to provide 3D physics and visualization.
        
        Args:
            config (dict): Configuration dictionary with world parameters
            render_mode (str): Rendering mode ("human", "rgb_array", etc.)
            view (str): Camera view angle ("top", "agent", etc.)
        """
        # =====================================
        # CONFIGURATION SETUP
        # =====================================
        # Load default configuration if none provided
        if config is None:
            from escape_room.config.default_config import get_default_config
            config = get_default_config()
        
        self.config = config
        self.view = view
        self.episode_count = 0

        # =====================================
        # DOOR SAFE ZONE CONFIGURATION
        # =====================================
        # Safe zones prevent agents from getting stuck when doors move
        self.door_safe_zone_x_extension = config.get("door_safe_zone_x_extension", 0.0)    # Extension along door width
        self.door_safe_zone_z_extension = config.get("door_safe_zone_z_extension", 0.0)    # Extension along door depth
        self.enable_door_safe_zones = config.get("enable_door_safe_zones", True)           # Enable/disable safe zones
        
        # =====================================
        # WORLD GEOMETRY PARAMETERS
        # =====================================
        # Extract core world dimensions and constraints
        self.world_width = config["world_width"]        # Total environment width (18.4m)
        self.world_depth = config["world_depth"]        # Total environment depth (6.7m)
        self.door_width = config["door_width"]          # Width of doors between rooms (1.0m)
        
        # =====================================
        # DOOR POSITION MANAGEMENT
        # =====================================
        # Legacy single door position (maintained for backward compatibility)
        self.door_position = config["door_position"]
        
        # Individual door position tracking for each room (preferred system)
        self.individual_door_positions = {
            'roomA': self.config["door_position"],      # Door position for leftmost room
            'roomB': self.config["door_position"],      # Door position for middle room
            'roomC': self.config["door_position"]       # Door position for rightmost room
        }

        # Door position constraints (relative to room coordinates)
        self.door_position_min = config["door_position_min"]  # Minimum door position (0.6m)
        self.door_position_max = config["door_position_max"]  # Maximum door position (5.4m)
        
        # =====================================
        # GOAL AND EPISODE CONFIGURATION
        # =====================================
        self.terminal_location = config["terminal_location"]  # Goal position [x, z]
        max_episode_steps = config["max_episode_steps"]       # Maximum steps per episode
        
        # =====================================
        # STATE TRACKING INITIALIZATION
        # =====================================
        # Track door positions over time for analysis
        self.door_positions = []
        
        # Path tracking for navigator agent
        self.current_path = []              # Current episode path
        self.successful_paths = []          # Archive of successful paths
        
        # Navigator state tracking
        self.navigator_start_pos = None             # Starting position of navigator
        self.navigator_previous_distance = None     # Previous distance to terminal (for reward calculation)
        
        # Episode progress tracking
        self.has_reached_hallway = False            # Has navigator reached central hallway?
        self.hallway_reward_given = False           # Prevent duplicate hallway rewards
        self.started_in_hallway = False             # Did navigator start in hallway?
        
        # Rendering and UI state
        self.window_created = False                 # Track window creation status
        self.paused = False                         # Pause state for interactive debugging
        
        # =====================================
        # MINIWORLD INITIALIZATION
        # =====================================
        # Initialize parent MiniWorldEnv with configuration parameters
        super().__init__(
            max_episode_steps=config["max_episode_steps"], 
            render_mode=render_mode, 
            view=view
        )
    
    def _gen_world(self):
        """
        Generate the 3D world with rooms, doors, and connections
        
        Creates the physical environment layout including:
        - Four rooms (3 main rooms + central hallway)
        - Initial door connections between rooms
        - Agent placement and configuration
        - Static world data generation
        """
        # =====================================
        # AGENT CONFIGURATION
        # =====================================
        # Set agent physical properties
        self.agent.radius = 0.25  # Agent collision radius (affects navigation and collision detection)
        
        # =====================================
        # ROOM CREATION
        # =====================================
        # Create central hallway (roomD) - connects all other rooms
        self.roomD = self.add_rect_room(
            min_x=0, max_x=self.world_width,   # Full width of environment
            min_z=5.2, max_z=self.world_depth  # Top portion of environment
        )
        
        # Create main rooms in sequence from left to right
        # Room A (leftmost room)
        self.roomA = self.add_rect_room(
            min_x=0, max_x=6.0,      # Left portion: 0 to 6.0
            min_z=0, max_z=5         # Bottom portion: 0 to 5.0
        )
        
        # Room B (middle room)
        self.roomB = self.add_rect_room(
            min_x=6.2, max_x=12.2,   # Middle portion: 6.2 to 12.2 (gap for walls)
            min_z=0, max_z=5         # Bottom portion: 0 to 5.0
        )
        
        # Room C (rightmost room)
        self.roomC = self.add_rect_room(
            min_x=12.4, max_x=18.4,  # Right portion: 12.4 to 18.4 (gap for walls)
            min_z=0, max_z=5         # Bottom portion: 0 to 5.0
        )
        
        # =====================================
        # ROOM TRACKING AND MANAGEMENT
        # =====================================
        # Store rooms list for iteration and management
        self.rooms = [self.roomA, self.roomB, self.roomC, self.roomD]
        
        # =====================================
        # INITIAL AGENT PLACEMENT
        # =====================================
        # Place agent with default position (will be overridden in reset())
        self.place_entity(self.agent, pos=[1, 0, 1], dir=-np.pi/2)
        
        # =====================================
        # WORLD DATA GENERATION
        # =====================================
        # Generate static visual and collision data for rendering
        self._gen_static_data()

    def disconnect_rooms(self, room1, room2):
        """
        Robust method to remove portal connections between two rooms
        
        This method safely removes door connections to allow for dynamic door
        repositioning. It handles cases where portals may not exist and provides
        logging for debugging door movement issues.
        
        Args:
            room1: First room to disconnect
            room2: Second room to disconnect
        """
        # Store original portal count for logging
        original_portal_count = len(self.portals)
        
        # Remove all portals connecting these two rooms (bidirectional)
        self.portals = [
            portal for portal in self.portals 
            if not ((portal.room_from == room1 and portal.room_to == room2) or 
                    (portal.room_from == room2 and portal.room_to == room1))
        ]
        
        # Calculate and log number of portals removed for debugging
        portals_removed = original_portal_count - len(self.portals)
    
    def move_door(self, new_position):
        """
        Legacy method to move all doors to the same position
        
        This method maintains backward compatibility with older code that
        expected a single door position. It updates all individual door
        positions to the same value.
        
        Args:
            new_position (float): New position for all doors
            
        Note: For individual door control, use move_specific_door() instead
        """
        # Update all individual door positions to the same value
        for room_name in self.individual_door_positions:
            self.individual_door_positions[room_name] = new_position
            self.move_specific_door(room_name, new_position)
        
        # Update legacy door position reference for backward compatibility
        self.door_position = new_position

    def move_specific_door(self, room_name, new_position):
        """
        Move the door for a specific room with proper coordinate handling
        
        This method handles the complex process of repositioning a door including:
        - Coordinate transformation from relative to absolute positions
        - Boundary checking and position clamping
        - Portal disconnection and reconnection
        - State synchronization across all tracking systems
        
        Args:
            room_name (str): Name of the room ("roomA", "roomB", or "roomC")
            new_position (float): New door position relative to room coordinates
        """
        try:
            # =====================================
            # ROOM AND HALLWAY REFERENCES
            # =====================================
            # Get references to the specific room and central hallway
            top_room = getattr(self, room_name)    # The room whose door we're moving
            hallway = self.roomD                   # Central hallway (constant connection target)
            
            # =====================================
            # COORDINATE SYSTEM CONFIGURATION
            # =====================================
            # Get room-specific position constraints from configuration
            room_ranges = self.config.get("door_position_ranges", {})
            
            # CRITICAL: Room layout constants must be consistent across all code
            room_offsets = {
                'roomA': 0.0,    # Room A starts at world X=0.0
                'roomB': 6.2,    # Room B starts at world X=6.2
                'roomC': 12.4    # Room C starts at world X=12.4
            }
            
            # Standard room dimensions
            room_widths = {
                'roomA': 6.0,    # Room A width: 6.0 units
                'roomB': 6.0,    # Room B width: 6.0 units  
                'roomC': 6.0     # Room C width: 6.0 units
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
                # Fallback: Use 10-90% of room width as safe positioning range
                min_pos = 0.1 * room_width
                max_pos = 0.9 * room_width
                relative_position = max(min_pos, min(max_pos, new_position))
            
            # =====================================
            # COORDINATE TRANSFORMATION
            # =====================================
            # Convert from relative position (within room) to absolute world coordinates
            absolute_position = room_offset + relative_position
            
            # =====================================
            # FINAL BOUNDARY SAFETY CHECK
            # =====================================
            # Ensure door fits within actual room boundaries with safety margins
            margin = self.door_width/2 + 0.1      # Half door width plus safety buffer
            min_absolute = top_room.min_x + margin  # Leftmost safe position in room
            max_absolute = top_room.max_x - margin  # Rightmost safe position in room
            
            # Apply final clamping to physical room boundaries
            adjusted_position = max(min_absolute, min(max_absolute, absolute_position))
            
            # =====================================
            # STATE SYNCHRONIZATION
            # =====================================
            # Store absolute positions for visualization and debugging
            if not hasattr(self, 'absolute_door_positions'):
                self.absolute_door_positions = {}
            self.absolute_door_positions[room_name] = adjusted_position
            
            # =====================================
            # PORTAL CONNECTION UPDATE
            # =====================================
            # Remove existing door connection between this room and hallway
            self.disconnect_rooms(top_room, hallway)
            
            # Create new door connection at the adjusted position
            door_width = self.door_width
            self.connect_rooms(
                top_room, 
                hallway, 
                min_x=adjusted_position - door_width/2,  # Left edge of door opening
                max_x=adjusted_position + door_width/2   # Right edge of door opening
            )
            
            # =====================================
            # INTERNAL STATE UPDATES
            # =====================================
            # Store the relative position in tracking dictionary (IMPORTANT: relative, not absolute)
            self.individual_door_positions[room_name] = relative_position

            # Update legacy door position for backward compatibility (if roomA)
            if room_name == 'roomA':
                self.door_position = relative_position
            
            # =====================================
            # VISUALIZATION UPDATE
            # =====================================
            # Force regeneration of visual data for updated door positions
            self._gen_static_data()
            if hasattr(self, 'debug_mode') and self.debug_mode:
                self._render_static()
                
        except Exception as e:
            # Comprehensive error handling with detailed error information
            print(f"Error moving door for {room_name}: {e}")
    
    def _get_current_room(self):
        """
        Determine which room the agent is currently located in
        
        Uses the agent's position to determine room occupancy with boundary
        tolerance to handle floating-point precision and edge cases.
        
        Returns:
            str: Room identifier ("roomA", "roomB", "roomC", "roomD", or "unknown")
        """
        agent_pos = self.agent.pos
        boundary_tolerance = 0.2  # Tolerance for room boundary detection
        
        # Check each room in the environment
        for room in self.rooms:
            # Check if agent position is within room boundaries (with tolerance)
            if ((room.min_x - boundary_tolerance <= agent_pos[0] <= room.max_x + boundary_tolerance) and 
                (room.min_z - boundary_tolerance <= agent_pos[2] <= room.max_z + boundary_tolerance)):
                
                # Find the room's name by matching with class attributes
                for attr_name, attr_value in vars(self).items():
                    if attr_name.startswith('room') and attr_value == room:
                        return attr_name
        
        # Fallback: Use previous room if tracking is available
        if hasattr(self, 'previous_room'):
            return self.previous_room
                
        # Final fallback for edge cases
        return "unknown"
    
    def _get_room_category(self, room_name):
        """
        Convert room names to numerical categories for agent observations
        
        Maps string room identifiers to integer categories that agents can
        use in their observation spaces for room-aware decision making.
        
        Args:
            room_name (str): Room identifier string
            
        Returns:
            int: Room category number
                0: Hallway (roomD)
                1: Room A (leftmost)
                2: Room B (middle)
                3: Room C (rightmost)
                4: Unknown/invalid room
        """
        # Define room category mappings
        hallway_rooms = ['roomD']                    # Central hallway
        top_rooms = ['roomA', 'roomB', 'roomC']      # Main rooms
    
        if room_name in hallway_rooms:
            return 0  # Hallway category
        elif room_name in top_rooms:
            # Assign specific categories based on room position
            return {'roomA': 1, 'roomB': 2, 'roomC': 3}.get(room_name, 4)
        return 4  # Unknown room category
    
    def _normalize_position(self, position):
        """
        Normalize a position vector to [0, 1] range for consistent observations
        
        Converts world coordinates to normalized coordinates that agents can
        use consistently regardless of world size. Handles both 2D and 3D positions.
        
        Args:
            position: Position vector to normalize (2D [x,z] or 3D [x,y,z])
            
        Returns:
            np.array: Normalized position vector [0,1] range
        """
        try:
            if len(position) == 2:
                # Handle 2D position [x, z]
                return np.clip(np.array([
                    position[0] / self.world_width,   # Normalize X coordinate
                    position[1] / self.world_depth    # Normalize Z coordinate
                ]), 0, 1)
            else:
                # Handle 3D position [x, y, z] - extract X and Z
                return np.clip(np.array([
                    position[0] / self.world_width,   # Normalize X coordinate
                    position[2] / self.world_depth    # Normalize Z coordinate (skip Y)
                ]), 0, 1)
        except (IndexError, ZeroDivisionError, TypeError) as e:
            # Handle error cases gracefully
            print(f"Warning: Error in position normalization: {e}")
            return np.array([0.0, 0.0])
    
    def _normalize_terminal_distance(self, distance):
        """
        Normalize distance to terminal for reward calculations
        
        Converts distance to a normalized value where higher numbers indicate
        closer proximity to the terminal (better for reward calculations).
        
        Args:
            distance (float): Raw distance to terminal
            
        Returns:
            float: Normalized distance (0.0 = far, 1.0 = at terminal)
        """
        # Calculate maximum possible distance (diagonal of world)
        max_dist = np.sqrt(self.world_width**2 + self.world_depth**2)
        
        # Invert distance so closer = higher value
        return (max_dist - distance) / max_dist
    
    def get_wall_distance(self, pos, direction, max_distance=10):
        """
        Cast a ray from position in given direction and return distance to nearest wall
        
        This method implements raycasting for LIDAR-like distance measurements.
        It steps along the given direction until it hits a wall (leaves valid room space).
        
        Args:
            pos: Starting position for raycast
            direction: Direction vector for raycast
            max_distance: Maximum distance to check before giving up
            
        Returns:
            float: Distance to nearest wall in given direction
        """
        step_size = 0.1               # Resolution of raycast steps
        current_pos = np.array(pos)   # Current position along ray
        
        # Step along the ray until we hit a wall or reach max distance
        for i in range(int(max_distance / step_size)):
            current_pos = current_pos + direction * step_size
            
            # Check if current position is inside any valid room
            in_room = False
            for room in self.rooms:
                if (room.min_x <= current_pos[0] <= room.max_x and
                    room.min_z <= current_pos[2] <= room.max_z):
                    in_room = True
                    break
            
            # If not in any room, we've hit a wall
            if not in_room:
                return i * step_size
                
        # Reached maximum distance without hitting wall
        return max_distance
    
    def get_lidar_measurements(self):
        """
        Get LIDAR-like distance measurements in multiple directions from agent
        
        Provides the agent with distance sensors for obstacle detection and
        navigation. Measures distances to walls in 5 key directions relative
        to the agent's current orientation.
        
        Returns:
            tuple: (right_dist, left_dist, forward_dist, forward_right_dist, forward_left_dist)
                All distances in world units, representing distance to nearest wall
        """
        # =====================================
        # DIRECTION VECTOR CALCULATION
        # =====================================
        # Agent's current forward direction (primary orientation)
        forward_dir = self.agent.dir_vec
        
        # Calculate perpendicular directions relative to agent's orientation
        # Right direction: 90 degrees clockwise from forward
        right_dir = np.array([-forward_dir[2], 0, forward_dir[0]])
        
        # Left direction: 90 degrees counterclockwise from forward  
        left_dir = np.array([forward_dir[2], 0, -forward_dir[0]])
        
        # =====================================
        # DIAGONAL DIRECTION CALCULATION
        # =====================================
        # Calculate diagonal directions by combining forward with left/right
        # Forward-right diagonal: 45 degrees between forward and right
        forward_right_dir = forward_dir + right_dir
        forward_right_dir = forward_right_dir / np.linalg.norm(forward_right_dir)
        
        # Forward-left diagonal: 45 degrees between forward and left
        forward_left_dir = forward_dir + left_dir
        forward_left_dir = forward_left_dir / np.linalg.norm(forward_left_dir)
        
        # =====================================
        # DISTANCE MEASUREMENTS
        # =====================================
        # Cast rays in all 5 directions to measure distances to walls
        forward_dist = self.get_wall_distance(self.agent.pos, forward_dir)
        left_dist = self.get_wall_distance(self.agent.pos, left_dir)
        right_dist = self.get_wall_distance(self.agent.pos, right_dir)
        forward_right_dist = self.get_wall_distance(self.agent.pos, forward_right_dir)
        forward_left_dist = self.get_wall_distance(self.agent.pos, forward_left_dir)
        
        return right_dist, left_dist, forward_dist, forward_right_dist, forward_left_dist
    
    def is_in_door_safe_zone(self, agent_pos):
        """
        Check if agent is within a rectangular safe zone around any door
        
        Safe zones prevent agents from getting stuck when doors are moved during
        an episode. They create protected areas around doors where collision
        detection is modified to allow free movement.
        
        Args:
            agent_pos: Agent's current position [x, y, z]
            
        Returns:
            tuple: (is_in_safe_zone, room_name)
                - is_in_safe_zone (bool): Whether agent is in any safe zone
                - room_name (str): Which room's door safe zone (if any)
        """
        # Skip safe zone checks if disabled in configuration
        if not self.enable_door_safe_zones:
            return False
            
        # =====================================
        # ROOM BOUNDARY DEFINITIONS
        # =====================================
        # Room layout constants for coordinate calculations
        room_boundaries = {
            'roomA': (0.0, 6.0),      # Room A: left and right boundaries
            'roomB': (6.2, 12.2),     # Room B: left and right boundaries
            'roomC': (12.4, 18.4)     # Room C: left and right boundaries
        }
        
        # =====================================
        # DOOR SAFE ZONE CHECKING
        # =====================================
        # Check each door for safe zone overlap
        for room_name, (room_min_x, room_max_x) in room_boundaries.items():
            # Get door position for this room
            door_pos = None
            if hasattr(self, 'individual_door_positions'):
                # Preferred: individual door position tracking
                door_pos = self.individual_door_positions.get(room_name)
            elif hasattr(self, 'door_position'):
                # Fallback: legacy single door position
                door_pos = self.door_position
                
            if door_pos is not None:
                # =====================================
                # SAFE ZONE BOUNDARY CALCULATION
                # =====================================
                # Calculate absolute door position in world coordinates
                door_x = door_pos + room_min_x  # Relative position + room offset
                door_z = 5.0                    # All doors are at Z=5.0 (room-hallway boundary)
                
                # Get door dimensions
                door_width = self.door_width  # Typically 1.0 unit
                
                # Calculate safe zone boundaries with extensions
                # X-axis: door center ± (door_width/2 + extension)
                safe_zone_x_min = door_x - (door_width/2 + self.door_safe_zone_x_extension)
                safe_zone_x_max = door_x + (door_width/2 + self.door_safe_zone_x_extension)
                
                # Z-axis: door position ± extension (doors span room-hallway boundary)
                safe_zone_z_min = door_z - self.door_safe_zone_z_extension
                safe_zone_z_max = door_z + self.door_safe_zone_z_extension
                
                # =====================================
                # AGENT POSITION CHECK
                # =====================================
                # Check if agent is within this door's safe zone
                if (safe_zone_x_min <= agent_pos[0] <= safe_zone_x_max and
                    safe_zone_z_min <= agent_pos[2] <= safe_zone_z_max):
                    return True, room_name  # Agent is in this door's safe zone
                    
        return False, None  # Agent is not in any safe zone
    
    def get_wall_distance_with_safe_zones(self, pos, direction, max_distance=10):
        """
        Modified wall distance calculation that accounts for door safe zones
        
        This version of raycasting ignores walls when the ray passes through
        door safe zones, allowing agents to plan paths through doors even
        when the door position changes.
        
        Args:
            pos: Starting position for raycast
            direction: Direction vector for raycast  
            max_distance: Maximum distance to check
            
        Returns:
            float: Distance to nearest wall (ignoring safe zones)
        """
        step_size = 0.1               # Resolution of raycast steps
        current_pos = np.array(pos)   # Current position along ray
        
        # Step along the ray until we hit a wall or reach max distance
        for i in range(int(max_distance / step_size)):
            current_pos = current_pos + direction * step_size
            
            # Check if current position is in a door safe zone
            in_safe_zone, _ = self.is_in_door_safe_zone(current_pos)
            if in_safe_zone:
                continue  # Skip wall collision check in safe zones
            
            # Check if position is inside any valid room (original logic)
            in_room = False
            for room in self.rooms:
                if (room.min_x <= current_pos[0] <= room.max_x and
                    room.min_z <= current_pos[2] <= room.max_z):
                    in_room = True
                    break
            
            # If not in any room and not in safe zone, we've hit a wall
            if not in_room:
                return i * step_size
                
        # Reached maximum distance without hitting wall
        return max_distance
    
    def get_lidar_measurements_with_safe_zones(self):
        """
        LIDAR measurements that account for door safe zones
        
        This version of LIDAR uses safe-zone-aware raycasting to provide
        more accurate distance measurements that account for door openings
        and safe zones around doors.
        
        Returns:
            tuple: (right_dist, left_dist, forward_dist, forward_right_dist, forward_left_dist)
                All distances accounting for safe zones around doors
        """
        # =====================================
        # DIRECTION VECTOR CALCULATION
        # =====================================
        # Agent's current forward direction
        forward_dir = self.agent.dir_vec
        
        # Calculate perpendicular directions relative to agent
        right_dir = np.array([-forward_dir[2], 0, forward_dir[0]])      # 90° clockwise
        left_dir = np.array([forward_dir[2], 0, -forward_dir[0]])       # 90° counterclockwise
        
        # Calculate diagonal directions
        forward_right_dir = forward_dir + right_dir
        forward_right_dir = forward_right_dir / np.linalg.norm(forward_right_dir)
        
        forward_left_dir = forward_dir + left_dir
        forward_left_dir = forward_left_dir / np.linalg.norm(forward_left_dir)
        
        # =====================================
        # SAFE-ZONE-AWARE DISTANCE MEASUREMENTS
        # =====================================
        # Use safe-zone-aware raycasting for all directions
        forward_dist = self.get_wall_distance_with_safe_zones(self.agent.pos, forward_dir)
        left_dist = self.get_wall_distance_with_safe_zones(self.agent.pos, left_dir)
        right_dist = self.get_wall_distance_with_safe_zones(self.agent.pos, right_dir)
        forward_right_dist = self.get_wall_distance_with_safe_zones(self.agent.pos, forward_right_dir)
        forward_left_dist = self.get_wall_distance_with_safe_zones(self.agent.pos, forward_left_dir)
        
        return right_dist, left_dist, forward_dist, forward_right_dist, forward_left_dist
    
    def debug_visualize_safe_zones(self):
        """
        Debug method to print safe zone coordinates for troubleshooting
        
        Outputs detailed information about door safe zone boundaries for
        debugging door movement, safe zone configuration, and collision issues.
        Useful for understanding why agents can or cannot move in certain areas.
        """
        if self.enable_door_safe_zones:
            print("\n=== Door Safe Zones Debug Information ===")
            
            # Room boundary definitions for calculations
            room_boundaries = {
                'roomA': (0.0, 6.0),      # Room A boundaries
                'roomB': (6.2, 12.2),     # Room B boundaries  
                'roomC': (12.4, 18.4)     # Room C boundaries
            }
            
            # Print safe zone details for each room
            for room_name, (room_min_x, room_max_x) in room_boundaries.items():
                # Get door position for this room
                door_pos = None
                if hasattr(self, 'individual_door_positions'):
                    door_pos = self.individual_door_positions.get(room_name)
                elif hasattr(self, 'door_position'):
                    door_pos = self.door_position
                    
                if door_pos is not None:
                    # Calculate door coordinates and safe zone boundaries
                    door_x = door_pos + room_min_x
                    door_z = 5.0
                    door_width = self.door_width
                    
                    # Calculate safe zone boundaries
                    safe_zone_x_min = door_x - (door_width/2 + self.door_safe_zone_x_extension)
                    safe_zone_x_max = door_x + (door_width/2 + self.door_safe_zone_x_extension)
                    safe_zone_z_min = door_z - self.door_safe_zone_z_extension
                    safe_zone_z_max = door_z + self.door_safe_zone_z_extension
                    
                    # Print detailed information
                    print(f"{room_name}: Door at ({door_x:.2f}, {door_z:.2f})")
                    print(f"  Safe Zone: X[{safe_zone_x_min:.2f}, {safe_zone_x_max:.2f}] Z[{safe_zone_z_min:.2f}, {safe_zone_z_max:.2f}]")
    
    def seed(self, seed=None):
        """
        Seed the environment's random number generator for reproducible behavior
        
        Initializes the random number generator used for agent placement, door
        positioning, and other stochastic elements of the environment.
        
        Args:
            seed (int, optional): Random seed value. If None, uses current time.
            
        Returns:
            list: List containing the actual seed used
        """
        # Generate seed from current time if none provided
        if seed is None:
            seed = int(time.time()) % (2**31-1)
            
        # Initialize NumPy random state for consistent behavior
        self.np_random = np.random.RandomState(seed)
        
        # Initialize position tracking indices for agent placement
        self.position_indices = {cat: 0 for cat in self.config["room_categories"].keys()}
        
        return [seed]
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment for a new episode
        
        This method handles complete environment reset including:
        - Random number generator seeding
        - Agent placement using room-based cycling system
        - Portal/door connection clearing and reset
        - State tracking variable initialization
        - Episode counter management
        
        Args:
            seed (int, optional): Random seed for episode
            options (dict, optional): Additional reset options
            
        Returns:
            tuple: (observation, info) - Empty observation since multi-agent handles observations separately
        """
        # =====================================
        # RANDOM NUMBER GENERATOR SETUP
        # =====================================
        # Set seed if provided for reproducible episodes
        if seed is not None:
            self.seed(seed)
        
        # Initialize position tracking if not already present
        if not hasattr(self, 'position_indices'):
            self.position_indices = {cat: 0 for cat in self.config["room_categories"].keys()}
                
        # Reset position indices when seed is provided (fresh start)
        if seed is not None:
            self.position_indices = {cat: 0 for cat in self.config["room_categories"].keys()}
            self.global_position_index = 0
        elif not hasattr(self, 'np_random'):
            # If no seed but RNG is missing, create with default seed
            self.seed(0)
            self.position_indices = {cat: 0 for cat in self.config["room_categories"].keys()}
            self.global_position_index = 0
                
        # Initialize global position index if not present
        if not hasattr(self, 'global_position_index'):
            self.global_position_index = 0

        # =====================================
        # EPISODE MANAGEMENT
        # =====================================
        # Increment episode counter for tracking and analysis
        if not hasattr(self, 'episode_count'):
            self.episode_count = 0
        self.episode_count += 1
        
        # =====================================
        # PORTAL SYSTEM RESET
        # =====================================
        # Clear all existing door connections to start fresh
        self.portals = []
        
        # Call parent class reset to handle MiniWorld-specific resets
        super().reset(seed=seed)

        # =====================================
        # AGENT PLACEMENT SYSTEM
        # =====================================
        # Initialize placement tracking if needed
        if not hasattr(self, 'placement_incrementer'):
            self.placement_incrementer = 0
        
        # Get configuration parameters for placement strategy
        episodes_per_room = self.config.get("episodes_per_room", 3)
        
        # =====================================
        # PLACEMENT STRATEGY SELECTION
        # =====================================
        # Determine whether to use fixed positions or random placement
        # Strategy varies based on episodes_per_room configuration:
        # - Even numbers (2,4,6...): 50% fixed / 50% random
        # - Odd numbers (3,5,7...): 67% fixed / 33% random
        if episodes_per_room % 2 == 0:  # Even episodes per room
            use_fixed_position = self.np_random.random() < 0.5  # 50% chance of fixed position
        else:  # Odd episodes per room
            use_fixed_position = self.np_random.random() < 2/3  # 67% chance of fixed position

        # =====================================
        # ROOM CYCLING SYSTEM
        # =====================================
        # Cycle through rooms (A, B, C) to ensure balanced training
        rooms = [self.roomA, self.roomB, self.roomC]
        current_room_index = (self.placement_incrementer // episodes_per_room) % len(rooms)
        current_room = rooms[current_room_index]
        
        # =====================================
        # AGENT PLACEMENT EXECUTION
        # =====================================
        if use_fixed_position:
            # ===== FIXED POSITION PLACEMENT =====
            # Use predefined positions from configuration
            room_positions = self.config["room_categories"]["room"]
            
            # Filter positions that belong to the current room
            room_specific_positions = []
            for pos, dir in room_positions:
                # Check if position is within current room's boundaries
                if current_room.min_x <= pos[0] <= current_room.max_x and current_room.min_z <= pos[2] <= current_room.max_z:
                    room_specific_positions.append((pos, dir))
            
            if room_specific_positions:
                # Randomly select one of the valid fixed positions for this room
                pos_index = int(self.np_random.random() * len(room_specific_positions))
                start_pos, start_dir = room_specific_positions[pos_index]
                
                # Place agent at the selected fixed position
                self.place_entity(self.agent, pos=start_pos, dir=start_dir)
                self.navigator_start_pos = start_pos
            else:
                # Fallback: Use random placement if no fixed positions available
                self.place_entity(self.agent, room=current_room, dir=self.np_random.uniform(-np.pi, np.pi))
                self.navigator_start_pos = self.agent.pos
        else:
            # ===== RANDOM POSITION PLACEMENT =====
            # Place agent randomly within the current room
            random_dir = self.np_random.uniform(-np.pi, np.pi)
            self.place_entity(self.agent, room=current_room, dir=random_dir)
            self.navigator_start_pos = self.agent.pos
        
        # Increment placement counter for next episode
        self.placement_incrementer += 1
        
        # =====================================
        # STATE TRACKING RESET
        # =====================================
        # Reset all episode-specific tracking variables
        self.has_reached_hallway = False            # Navigator hasn't reached hallway yet
        self.hallway_reward_given = False           # Hallway reward not given yet
        self.current_path = []                      # Clear path tracking
        
        # Determine starting room and set initial flags
        current_room = self._get_current_room()
        self.started_in_hallway = current_room in ['roomD']  # Did agent start in hallway?
        self.previous_room = current_room                    # Track room transitions
        self._episode_success = False                        # Episode not completed yet
        self._terminal_reached = False                       # Terminal not reached yet
        
        # Special handling for hallway starts
        if self.started_in_hallway:
            self.has_reached_hallway = True     # Already in hallway
            self.hallway_reward_given = True    # Don't give hallway reward
        
        # =====================================
        # DISTANCE TRACKING INITIALIZATION
        # =====================================
        # Calculate initial distance to terminal for progress tracking
        distance_to_terminal = np.linalg.norm(
            np.array([self.agent.pos[0], self.agent.pos[2]]) - 
            np.array([self.terminal_location[0], self.terminal_location[1]])
        )
        self.navigator_previous_distance = distance_to_terminal
        
        # Return empty observation (multi-agent wrapper handles individual observations)
        return None, {}

    def step_navigator(self, action):
        """
        Execute a single step for the navigator agent
        
        Translates high-level navigator actions into MiniWorld actions and
        executes them in the physics simulation. Handles action mapping,
        distance tracking, and path recording.
        
        Args:
            action (int): Navigator action to execute
                0: Turn right
                1: Turn left  
                2: Move forward
        """
        # =====================================
        # PRE-ACTION STATE TRACKING
        # =====================================
        # Store current distance before moving for reward calculations
        current_distance = np.linalg.norm(
            np.array([self.agent.pos[0], self.agent.pos[2]]) - 
            np.array([self.terminal_location[0], self.terminal_location[1]])
        )
        
        # =====================================
        # ACTION MAPPING AND EXECUTION
        # =====================================
        # Convert navigator action space to MiniWorld action space
        if action == 0:  # Turn right
            miniworld_action = self.actions.turn_right
        elif action == 1:  # Turn left
            miniworld_action = self.actions.turn_left
        elif action == 2:  # Move forward
            miniworld_action = self.actions.move_forward
        else:
            raise ValueError(f"Invalid navigator action: {action}")
        
        # Execute the action in the MiniWorld physics simulation
        super().step(miniworld_action)
        
        # =====================================
        # POST-ACTION STATE UPDATE
        # =====================================
        # Update distance tracking for next reward calculation
        self.navigator_previous_distance = current_distance
        
        # Record current position in path for analysis and door controller rewards
        self.current_path.append(list(self.agent.pos))
    
    def render(self):
        """
        Render the environment if in human mode
        
        Provides visual output for debugging and human observation.
        Only renders when render_mode is set to "human" to avoid
        unnecessary computation during training.
        
        Returns:
            Rendered frame if render_mode="human", None otherwise
        """
        # Only render if specifically requested for human viewing
        if self.render_mode == "human":
            return super().render()
        return None
    
    def record_door_position(self, door_position):
        """
        Record a door position for analysis and debugging
        
        Maintains both internal tracking and external file logging of door
        positions for analysis of door controller behavior and training progress.
        
        Args:
            door_position (float): Door position to record
        """
        # =====================================
        # INTERNAL TRACKING
        # =====================================
        # Add to internal list for runtime access
        self.door_positions.append(door_position)
        
        # =====================================
        # EXTERNAL FILE LOGGING
        # =====================================
        # Save directly to file for persistent analysis
        import os
        file_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
            "door_positions_direct.txt"
        )
        with open(file_path, "a") as f:
            f.write(f"{door_position:.4f}\n")