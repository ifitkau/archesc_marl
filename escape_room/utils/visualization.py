"""
Enhanced visualization utilities for the escape room environment with improved path rendering
and separate scaling for rewards

This module provides comprehensive visualization tools for analyzing escape room environment
training data, including:
- Agent path visualization with room layouts and door positions
- Reward plotting with separate scaling for different agent types
- Success rate analysis over training episodes
- Door position distribution analysis
- Color-coded path matching to door positions

Key Features:
- Supports both individual and batch path visualization
- Handles multiple room layouts with proper wall and door rendering
- Provides both raw and smoothed reward plots
- Color-matches paths to their corresponding door positions
- Validates paths to prevent visualization artifacts through walls
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cmap

def save_paths(env, filename="paths.png", show_door=True, path_alpha=0.6, door_alpha=0.6, line_width=2):
    """
    Save a visualization of all paths with improved wall and door representation
    
    This function creates a comprehensive visualization of agent paths through the escape room
    environment, including room layouts, walls, doors, and the terminal location. All paths
    are rendered in the same color for consistency.
    
    Args:
        env: Environment or world instance containing the escape room data
        filename (str): Output filename for the saved visualization (default: "paths.png")
        show_door (bool): Whether to show door positions as rectangles (default: True)
        path_alpha (float): Transparency level for path lines (0.0-1.0, default: 0.6)
        door_alpha (float): Transparency level for door rectangles (0.0-1.0, default: 0.6)
        line_width (int): Width of path lines in pixels (default: 2)
    
    Returns:
        None: Saves the visualization to the specified filename
        
    Environment Requirements:
        - env.world or env must have 'rooms' attribute with room boundaries
        - Should have 'successful_paths' containing agent path data
        - Should have 'terminal_location' for goal position
        - May have 'individual_door_positions' or 'door_positions' for door data
    """
    # Create a large figure for detailed visualization
    plt.figure(figsize=(12, 10))
    
    # Get base environment - handle different environment wrapper types
    base_env = env.world if hasattr(env, 'world') else env
    
    # Plot outer walls for all rooms based on room boundaries
    if hasattr(base_env, 'rooms'):
        # Calculate overall environment boundaries from individual room boundaries
        min_x = min(room.min_x for room in base_env.rooms)
        max_x = max(room.max_x for room in base_env.rooms)
        min_z = min(room.min_z for room in base_env.rooms)
        max_z = max(room.max_z for room in base_env.rooms)
        
        # Draw outer perimeter walls (environment boundary)
        plt.plot([min_x, max_x, max_x, min_x, min_x],
                [min_z, min_z, max_z, max_z, min_z],
                color='black', linewidth=2)
        
        # Draw horizontal walls separating rooms from hallway
        # These create the hallway corridor that connects all rooms
        plt.plot([min_x, max_x], [3.0, 3.0], color='black', linewidth=2)  # Bottom of hallway
        plt.plot([min_x, max_x], [3.2, 3.2], color='black', linewidth=2)  # Top of hallway
        
        # Draw vertical walls separating individual rooms
        # Room A | Room B | Room C layout
        plt.plot([4.0, 4.0], [0, 3.0], color='black', linewidth=2)    # Left wall of Room B
        plt.plot([4.2, 4.2], [0, 3.0], color='black', linewidth=2)    # Right wall of Room A  
        plt.plot([8.2, 8.2], [0, 3.0], color='black', linewidth=2)    # Left wall of Room C
        plt.plot([8.4, 8.4], [0, 3.0], color='black', linewidth=2)    # Right wall of Room B
        
        # Add room labels for clarity
        plt.text(2.0, 1.5, "Room A", fontsize=12, ha='center')
        plt.text(6.2, 1.5, "Room B", fontsize=12, ha='center')
        plt.text(10.4, 1.5, "Room C", fontsize=12, ha='center')
        plt.text(6.2, 3.6, "Hallway", fontsize=12, ha='center')
    
    # Visualize terminal area (goal location) where agents must reach
    if hasattr(base_env, 'terminal_location'):
        term_x, term_z = base_env.terminal_location
        terminal_width = 0.8   # Width of terminal area
        terminal_height = 0.2  # Height of terminal area
        
        # Create terminal area rectangle with light green background
        terminal_area = plt.Rectangle(
            (term_x - terminal_width/2, term_z - terminal_height/2), 
            terminal_width, terminal_height, 
            color='lightgreen', alpha=0.3, label='Terminal Area'
        )
        plt.gca().add_patch(terminal_area)
    
    # Use consistent purple color for all paths
    path_color = 'purple'
    door_positions = {}  # Dictionary to store door position data
    
    # Plot door positions for all rooms if requested
    if show_door:
        # Get door width from environment or use default
        door_width = base_env.door_width if hasattr(base_env, 'door_width') else 1.0
        
        # Handle environments with individual door positions for each room
        if hasattr(base_env, 'individual_door_positions'):
            for room_name, pos in base_env.individual_door_positions.items():
                # Apply room offset to convert local door position to global coordinates
                offset = 0.0
                if room_name == 'roomB':
                    offset = 4.2  # Room B starts at x=4.2
                elif room_name == 'roomC':
                    offset = 8.4  # Room C starts at x=8.4
                
                door_pos = pos + offset  # Global door position
                door_positions[room_name] = door_pos
                
                # Draw only doors that are within the environment boundaries
                if door_pos >= min_x and door_pos <= max_x:
                    # Use rectangle to represent door opening in the wall
                    door_rect = plt.Rectangle(
                        (door_pos - door_width/2, 3.0),  # Position at hallway entrance
                        door_width, 0.2,                 # Width and height of door opening
                        color='red', alpha=door_alpha,
                        label=f"Door {room_name[-1]}" if room_name == 'roomA' else None
                    )
                    plt.gca().add_patch(door_rect)
                    
                    # Add door label showing room and local position
                    plt.text(door_pos, 2.9, f"{room_name[-1]}: {pos:.1f}", 
                            fontsize=9, ha='center', va='top', color='red')
        
        # Handle legacy environments with a single door position array
        elif hasattr(base_env, 'door_position') and hasattr(base_env, 'door_positions') and base_env.door_positions:
            for i, door_pos in enumerate(base_env.door_positions):
                # Use rectangle for door area
                door_rect = plt.Rectangle(
                    (door_pos - door_width/2, 3.0), 
                    door_width, 0.2,
                    color='red', alpha=door_alpha,
                    label="Door" if i == 0 else None  # Only label first door for legend
                )
                plt.gca().add_patch(door_rect)
                
                # Add door label showing global position
                plt.text(door_pos, 2.9, f"{door_pos:.1f}", 
                        fontsize=9, ha='center', va='top', color='red')
    
    # Plot agent paths - all use same color for consistency
    if hasattr(base_env, 'successful_paths') and base_env.successful_paths:
        for idx, path in enumerate(base_env.successful_paths):
            path_array = np.array(path)
            if path_array.size > 0:
                # Handle different path data formats
                if path_array.ndim == 1:
                    path_array = path_array.reshape(-1, 3)  # Reshape to (n_points, 3)
                
                # Validate path to ensure it doesn't go through walls
                valid_path = _validate_path_points(path_array, base_env)
                
                # Plot path as continuous line (NO DOTS for cleaner visualization)
                plt.plot(valid_path[:, 0], valid_path[:, 2], 
                        color=path_color, alpha=path_alpha, linewidth=line_width, 
                        label="Path" if idx == 0 else None)  # Only label first path for legend
                
                # Add starting position marker to show where agent began
                plt.scatter(valid_path[0, 0], valid_path[0, 2], 
                            color=path_color, s=100, marker='o', alpha=path_alpha)
                
                # Draw solid line to terminal if the path didn't reach it exactly
                last_point = valid_path[-1]
                terminal_distance = np.linalg.norm(
                    np.array([last_point[0], last_point[2]]) - 
                    np.array([base_env.terminal_location[0], base_env.terminal_location[1]])
                )
                if terminal_distance > 0.2:  # If more than 0.2 units away from terminal
                    plt.plot([last_point[0], base_env.terminal_location[0]], 
                            [last_point[2], base_env.terminal_location[1]], 
                            color=path_color, linewidth=line_width, alpha=path_alpha)
    
    # Plot terminal location on top of other elements for visibility
    if hasattr(base_env, 'terminal_location'):
        term_x, term_z = base_env.terminal_location
        plt.scatter(term_x, term_z, color='green', s=300, marker='*', 
                   label='Terminal', zorder=10)  # zorder=10 ensures it's on top
    
    # Set axis limits and labels for proper viewing
    plt.xlim(-0.5, base_env.world_width + 0.5)
    plt.ylim(-0.5, base_env.world_depth + 0.5)
    plt.gca().invert_yaxis()  # Invert Y-axis to match typical room layout convention
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Z', fontsize=12)
    plt.title('Agent Paths with Door Positions', fontsize=14)
    
    # Create custom legend with consistent styling
    legend_elements = [
        plt.Rectangle((0,0), 1, 1, color='lightgreen', alpha=0.3, label='Terminal Area'),
        plt.Line2D([0], [0], color=path_color, alpha=path_alpha, lw=2, label='Path'),
        plt.Rectangle((0,0), 1, 1, color='red', alpha=door_alpha, label='Door'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='green', 
                  markersize=15, label='Terminal')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Add grid for better visual reference
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Save the visualization with high quality
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()  # Close figure to free memory

def create_color_matched_plot(base_env, door_positions, filename="color_matched.png", path_alpha=0.6, door_alpha=0.6, line_width=2):
    """
    Create a visualization with paths color-matched to their corresponding doors
    
    This function creates an advanced visualization where each agent path is colored
    to match the door it uses, making it easy to see the relationship between
    starting positions, door choices, and path outcomes.
    
    Args:
        base_env: Environment or world instance containing the escape room data
        door_positions: List of door positions or dict of door positions by room
        filename (str): Output filename for the saved visualization (default: "color_matched.png")
        path_alpha (float): Transparency level for path lines (0.0-1.0, default: 0.6)
        door_alpha (float): Transparency level for door rectangles (0.0-1.0, default: 0.6)
        line_width (int): Width of path lines in pixels (default: 2)
    
    Returns:
        None: Saves the color-matched visualization to the specified filename
        
    Features:
        - Each path gets a unique color based on its associated door
        - Doors are colored to match their corresponding paths
        - Starting positions are marked and labeled
        - Proper room boundary validation for paths
    """
    # Create a large figure for detailed visualization
    plt.figure(figsize=(12, 10))
    
    # Get base environment - handle different environment wrapper types
    if not hasattr(base_env, 'world_width'):
        base_env = base_env.world if hasattr(base_env, 'world') else base_env
    
    # Plot outer walls for all rooms (same as save_paths function)
    if hasattr(base_env, 'rooms'):
        # Calculate overall environment boundaries
        min_x = min(room.min_x for room in base_env.rooms)
        max_x = max(room.max_x for room in base_env.rooms)
        min_z = min(room.min_z for room in base_env.rooms)
        max_z = max(room.max_z for room in base_env.rooms)
        
        # Draw environment structure
        # Outer perimeter walls
        plt.plot([min_x, max_x, max_x, min_x, min_x],
                [min_z, min_z, max_z, max_z, min_z],
                color='black', linewidth=2)
        
        # Horizontal walls creating hallway
        plt.plot([min_x, max_x], [3.0, 3.0], color='black', linewidth=2)
        plt.plot([min_x, max_x], [3.2, 3.2], color='black', linewidth=2)
        
        # Vertical walls separating rooms
        plt.plot([4.0, 4.0], [0, 3.0], color='black', linewidth=2)
        plt.plot([4.2, 4.2], [0, 3.0], color='black', linewidth=2)
        plt.plot([8.2, 8.2], [0, 3.0], color='black', linewidth=2)
        plt.plot([8.4, 8.4], [0, 3.0], color='black', linewidth=2)
        
        # Add room labels
        plt.text(2.0, 1.5, "Room A", fontsize=12, ha='center')
        plt.text(6.2, 1.5, "Room B", fontsize=12, ha='center')
        plt.text(10.4, 1.5, "Room C", fontsize=12, ha='center')
        plt.text(6.2, 3.6, "Hallway", fontsize=12, ha='center')
    
    # Draw terminal area
    if hasattr(base_env, 'terminal_location'):
        term_x, term_z = base_env.terminal_location
        terminal_width = 0.8
        terminal_height = 0.2
        
        # Create terminal area rectangle
        terminal_area = plt.Rectangle(
            (term_x - terminal_width/2, term_z - terminal_height/2), 
            terminal_width, terminal_height, 
            color='lightgreen', alpha=0.3, label='Terminal Area'
        )
        plt.gca().add_patch(terminal_area)
    
    # Use colormap for better color distinction between different door/path combinations
    colormap = plt.cm.get_cmap('tab10', 10)  # Get 10 distinct colors
    
    # Room offsets for calculating global door positions from local positions
    door_offset_map = {'roomA': 0.0, 'roomB': 4.2, 'roomC': 8.4}
    
    # Pre-calculate door width for drawing
    door_width = base_env.door_width if hasattr(base_env, 'door_width') else 1.0
    
    # Room boundaries for path validation and room detection
    room_boundaries = {
        'roomA': (0.0, 4.0),      # x_min, x_max for Room A
        'roomB': (4.2, 8.2),     # x_min, x_max for Room B
        'roomC': (8.4, 12.4),    # x_min, x_max for Room C
        'hallway': (0.0, 12.4)   # spans all X coordinates
    }
    
    # Create mapping for door positions to colors for consistent coloring
    door_color_map = {}
    
    def get_room_for_position(x, z):
        """
        Determine which room a position belongs to based on coordinates
        
        Args:
            x (float): X coordinate
            z (float): Z coordinate
            
        Returns:
            str or None: Room name ('roomA', 'roomB', 'roomC', 'hallway') or None
        """
        if z > 3.2:  # In hallway (above room level)
            return 'hallway'
        elif 0 <= x < 4.0:
            return 'roomA'
        elif 4.2 <= x < 8.2:
            return 'roomB'
        elif 8.4 <= x <= 12.4:
            return 'roomC'
        return None  # Not in any valid room or hallway
    
    def get_color_for_door(room, pos):
        """
        Get a consistent color for a room/door position combination
        
        Args:
            room (str): Room identifier
            pos (float): Door position (rounded to 1 decimal)
            
        Returns:
            color: Matplotlib color for the door/path combination
        """
        key = (room, round(float(pos), 1))  # Create unique key
        if key not in door_color_map:
            # Assign next available color from colormap
            door_color_map[key] = colormap(len(door_color_map) % 10)
        return door_color_map[key]
    
    # Track which doors we've already drawn to avoid duplicates
    drawn_doors = set()
    
    # Plot paths with colors matching their associated doors
    if hasattr(base_env, 'successful_paths') and base_env.successful_paths:
        for idx, path in enumerate(base_env.successful_paths):
            path_array = np.array(path)
            if path_array.size > 0:
                # Handle different path data formats
                if path_array.ndim == 1:
                    path_array = path_array.reshape(-1, 3)
                
                # Validate path to ensure it doesn't go through walls
                valid_path = _validate_path_points(path_array, base_env)
                
                # Determine starting room from first path point
                start_x, _, start_z = valid_path[0]
                start_room = get_room_for_position(start_x, start_z)
                
                # Get door position for this room/path
                room_key = f"room{start_room[-1]}" if start_room != 'hallway' else 'roomA'
                door_pos = None
                
                # Determine door position from individual door positions if available
                if hasattr(base_env, 'individual_door_positions') and room_key in base_env.individual_door_positions:
                    door_pos = base_env.individual_door_positions[room_key]
                # Fall back to door positions array if individual positions not available
                elif hasattr(base_env, 'door_positions') and idx < len(base_env.door_positions):
                    door_pos = base_env.door_positions[idx]
                else:
                    # Use default door position if no position found
                    door_pos = 2.0  # Default door position
                
                # Round position for consistent color mapping
                door_pos_rounded = round(float(door_pos), 1)
                
                # Get color for this room/door combination
                color = get_color_for_door(room_key, door_pos_rounded)
                
                # Draw door if not already drawn
                door_key = (room_key, door_pos_rounded)
                if door_key not in drawn_doors:
                    drawn_doors.add(door_key)
                    
                    # Calculate actual door position (with room offset for global coordinates)
                    actual_door_pos = door_pos + door_offset_map.get(room_key, 0.0)
                    
                    # Draw door rectangle
                    door_rect = plt.Rectangle(
                        (actual_door_pos - door_width/2, 3.0), 
                        door_width, 0.2,
                        color=color, alpha=door_alpha
                    )
                    plt.gca().add_patch(door_rect)
                    
                    # Add door label showing room and local position
                    plt.text(actual_door_pos, 2.9, f"{room_key[-1]}: {door_pos_rounded:.1f}", 
                            fontsize=9, ha='center', va='top', color=color)
                
                # Plot path with matching color
                plt.plot(valid_path[:, 0], valid_path[:, 2], 
                        color=color, alpha=path_alpha, linewidth=line_width)
                
                # Add starting position marker with path number
                plt.scatter(valid_path[0, 0], valid_path[0, 2], 
                            color=color, s=100, marker='o', alpha=0.7)
                plt.text(valid_path[0, 0]+0.1, valid_path[0, 2]+0.1, 
                         f"Start {idx+1}", fontsize=8, color=color)
                
                # Draw line to terminal if path didn't reach it exactly
                last_point = valid_path[-1]
                terminal_distance = np.linalg.norm(
                    np.array([last_point[0], last_point[2]]) - 
                    np.array([base_env.terminal_location[0], base_env.terminal_location[1]])
                )
                if terminal_distance > 0.2:
                    plt.plot([last_point[0], base_env.terminal_location[0]], 
                            [last_point[2], base_env.terminal_location[1]], 
                            color=color, linewidth=line_width, alpha=path_alpha)
    
    # Plot terminal location on top
    if hasattr(base_env, 'terminal_location'):
        term_x, term_z = base_env.terminal_location
        plt.scatter(term_x, term_z, color='darkgreen', s=300, marker='*', 
                   label='Terminal', zorder=10)
    
    # Set axis limits and labels
    plt.xlim(-0.5, base_env.world_width + 0.5)
    plt.ylim(-0.5, base_env.world_depth + 0.5)
    plt.gca().invert_yaxis()
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Z', fontsize=12)
    plt.title('Agent Paths with Door Positions', fontsize=14)
    
    # Create comprehensive legend with unique entries
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    
    # Add entries for each door/path combination
    for (room_key, pos), color in door_color_map.items():
        if room_key in ['roomA', 'roomB', 'roomC']:
            label = f"Path from {room_key[-1]} (door at {pos:.1f})"
            # Add entry to legend handles
            by_label[label] = plt.Line2D([0], [0], color=color, alpha=path_alpha, lw=2)
    
    # Add standard legend entries
    by_label['Terminal Area'] = plt.Rectangle((0,0), 1, 1, color='lightgreen', alpha=0.3)
    by_label['Terminal'] = plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='darkgreen', markersize=15)
    
    plt.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=9)
    
    # Add grid for better visual reference
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Save the visualization
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def _validate_path_points(path_array, base_env):
    """
    Validate and fix path points to ensure they don't go through walls
    
    This function processes agent paths to ensure they follow valid routes through
    the environment, preventing visualization artifacts where paths appear to go
    through walls or teleport between rooms without using doors.
    
    Args:
        path_array (np.array): Array of path points with shape (n_points, 3) [x, y, z]
        base_env: Environment or world instance containing room and door information
        
    Returns:
        np.array: Corrected path points that follow valid routes
        
    Validation Rules:
        - Paths must stay within room boundaries
        - Transitions between rooms must go through doors
        - Invalid segments are corrected with door waypoints
    """
    # If path is empty, return it unchanged
    if len(path_array) == 0:
        return path_array
    
    valid_path = [path_array[0]]  # Start with first point (always valid)
    
    # Define room boundaries (x_min, x_max, z_min, z_max) with small gaps to avoid wall clipping
    room_bounds = {
        'roomA': (0.0, 3.9, 0.0, 2.9),      # Room A boundaries
        'roomB': (4.3, 8.1, 0.0, 2.9),      # Room B boundaries  
        'roomC': (8.5, 12.4, 0.0, 2.9),     # Room C boundaries
        'hallway': (0.0, 12.4, 3.3, 4.2)    # Hallway boundaries
    }
    
    # Get door information for validation
    door_width = base_env.door_width if hasattr(base_env, 'door_width') else 1.0
    doors = []  # List of door boundaries
    
    # Add doors from individual door positions if available
    if hasattr(base_env, 'individual_door_positions'):
        for room, pos in base_env.individual_door_positions.items():
            # Calculate room offset for global coordinates
            offset = 0.0
            if room == 'roomB':
                offset = 4.2
            elif room == 'roomC':
                offset = 8.4
            
            # Add door as boundary rectangle (x_min, x_max, z_min, z_max)
            door_pos = pos + offset
            doors.append((door_pos - door_width/2, door_pos + door_width/2, 2.9, 3.3))
    
    def is_in_room(x, z, room):
        """Check if a position is within a specific room's boundaries"""
        bounds = room_bounds[room]
        return bounds[0] <= x <= bounds[1] and bounds[2] <= z <= bounds[3]
    
    def is_in_door(x, z):
        """Check if a position is within any door opening"""
        for door in doors:
            if door[0] <= x <= door[1] and door[2] <= z <= door[3]:
                return True
        return False
    
    # Process each path segment for validation
    for i in range(1, len(path_array)):
        prev_x, prev_y, prev_z = valid_path[-1]
        curr_x, curr_y, curr_z = path_array[i]
        
        # Determine which rooms the points are in
        prev_room = None
        curr_room = None
        
        # Find which room each point belongs to
        for room in room_bounds:
            if is_in_room(prev_x, prev_z, room):
                prev_room = room
            if is_in_room(curr_x, curr_z, room):
                curr_room = room
        
        # Add the current point if it's in a valid room
        if curr_room is not None:
            # If points are in different rooms, ensure they go through a door
            if prev_room != curr_room and not is_in_door(curr_x, curr_z) and not is_in_door(prev_x, prev_z):
                # Find the nearest door that connects these rooms
                for door in doors:
                    # Simple approximation - check if the door is between the rooms
                    door_x_center = (door[0] + door[1]) / 2
                    if ((prev_x <= door_x_center <= curr_x) or (curr_x <= door_x_center <= prev_x)):
                        # Add door entry point as waypoint
                        door_entry_z = 3.0  # Z coordinate at room entrance
                        valid_path.append([door_x_center, prev_y, door_entry_z])
                        break
            
            # Add the current point
            valid_path.append([curr_x, curr_y, curr_z])
    
    return np.array(valid_path)

def plot_rewards(rewards, window=10, filename="rewards.png", scale_rewards=True):
    """
    Plot and save the rewards over episodes with separate axes for each agent
    
    This function creates comprehensive reward visualizations that handle different
    agent types with potentially different reward scales. It provides both raw
    and smoothed reward plots, plus optional normalized plots for comparison.
    
    Args:
        rewards (dict): Dictionary mapping agent names to lists of episode rewards
                       Format: {'navigator': [r1, r2, ...], 'door_controller': [r1, r2, ...]}
        window (int): Window size for moving average smoothing (default: 10)
        filename (str): Output filename for the main reward plot (default: "rewards.png")
        scale_rewards (bool): Whether to create a second normalized plot (default: True)
        
    Returns:
        None: Saves reward plots to specified filename(s)
        
    Features:
        - Separate y-axes for different agent types (navigator vs door_controller)
        - Both raw and smoothed reward plots for trend analysis
        - Optional normalized plot for direct comparison between agents
        - Color-coded plots (blue for navigator, red for door_controller)
    """
    # Create a figure with two subplots with separate y-axes for different reward scales
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot each agent's rewards on appropriate subplot
    for agent, agent_rewards in rewards.items():
        # Convert to numpy array for easier mathematical operations
        rewards_array = np.array(agent_rewards)
        
        # Create episodes array for x-axis
        episodes = np.arange(1, len(rewards_array) + 1)
        
        # Calculate smoothed rewards using moving average
        if len(rewards_array) >= window:
            # Use convolution for efficient moving average calculation
            smoothed = np.convolve(rewards_array, np.ones(window)/window, mode='valid')
            smooth_episodes = episodes[window-1:]  # Adjust episodes for smoothed data
        else:
            # If not enough data for smoothing, use raw data
            smoothed = rewards_array
            smooth_episodes = episodes
        
        # Plot on respective axes based on agent type
        if agent == "navigator":
            # Navigator gets top subplot with blue coloring
            ax1.plot(smooth_episodes, smoothed, 'b-', linewidth=2, label=f"{agent} (smoothed)")
            ax1.plot(episodes, rewards_array, 'b-', alpha=0.3, linewidth=1, label=f"{agent} (raw)")
            ax1.set_ylabel('Navigator Reward', color='b', fontsize=12)
            ax1.tick_params(axis='y', labelcolor='b')
            ax1.grid(True, alpha=0.3)
            ax1.set_title('Navigator Rewards per Episode', fontsize=14)
        else:  # door_controller or other agents
            # Door controller gets bottom subplot with red coloring
            ax2.plot(smooth_episodes, smoothed, 'r-', linewidth=2, label=f"{agent} (smoothed)")
            ax2.plot(episodes, rewards_array, 'r-', alpha=0.3, linewidth=1, label=f"{agent} (raw)")
            ax2.set_ylabel('Door Controller Reward', color='r', fontsize=12)
            ax2.tick_params(axis='y', labelcolor='r')
            ax2.grid(True, alpha=0.3)
            ax2.set_title('Door Controller Rewards per Episode', fontsize=14)
    
    # Add legends to both subplots
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')
    
    # Set common x-axis label (only on bottom subplot)
    ax2.set_xlabel('Episode', fontsize=12)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save separate-axis plot
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a second plot with normalized rewards if requested
    if scale_rewards:
        plt.figure(figsize=(12, 6))
        
        # Track min/max values for normalization across all episodes
        navigator_min = float('inf')
        navigator_max = float('-inf')
        door_min = float('inf')
        door_max = float('-inf')
        
        # Find min/max values across all episodes for each agent type
        for agent, agent_rewards in rewards.items():
            rewards_array = np.array(agent_rewards)
            if len(rewards_array) > 0:
                if agent == "navigator":
                    navigator_min = min(navigator_min, np.min(rewards_array))
                    navigator_max = max(navigator_max, np.max(rewards_array))
                else:  # door_controller or other agents
                    door_min = min(door_min, np.min(rewards_array))
                    door_max = max(door_max, np.max(rewards_array))
        
        # Plot normalized rewards for each agent
        for agent, agent_rewards in rewards.items():
            rewards_array = np.array(agent_rewards)
            if len(rewards_array) > 0:
                # Calculate smoothed rewards
                if len(rewards_array) >= window:
                    smoothed = np.convolve(rewards_array, np.ones(window)/window, mode='valid')
                    episodes = np.arange(window, len(rewards_array) + 1)
                else:
                    smoothed = rewards_array
                    episodes = np.arange(1, len(rewards_array) + 1)
                
                # Normalize to [0, 1] range for direct comparison
                if agent == "navigator" and navigator_max > navigator_min:
                    normalized = (smoothed - navigator_min) / (navigator_max - navigator_min)
                    plt.plot(episodes, normalized, 'b-', linewidth=2, label=f"{agent} (normalized)")
                elif agent != "navigator" and door_max > door_min:
                    normalized = (smoothed - door_min) / (door_max - door_min)
                    plt.plot(episodes, normalized, 'r-', linewidth=2, label=f"{agent} (normalized)")
        
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Normalized Reward (0-1 scale)', fontsize=12)
        plt.title('Normalized Agent Rewards for Comparison', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save normalized plot with modified filename
        norm_filename = filename.replace('.png', '_normalized.png')
        plt.savefig(norm_filename, dpi=300, bbox_inches='tight')
        plt.close()

def plot_success_rate(successes, window=10, filename="success_rate.png"):
    """
    Plot and save the success rate over episodes
    
    This function creates a comprehensive success rate analysis showing both
    short-term trends (moving average) and long-term progress (cumulative rate).
    It helps visualize training progress and convergence behavior.
    
    Args:
        successes (list): List of booleans indicating success (True) or failure (False) for each episode
        window (int): Window size for moving average calculation (default: 10)
        filename (str): Output filename for the success rate plot (default: "success_rate.png")
        
    Returns:
        None: Saves success rate plot to specified filename
        
    Features:
        - Moving average success rate for trend analysis
        - Cumulative success rate for overall progress tracking
        - Raw success/failure data points for episode-by-episode view
        - Grid and proper scaling for clear visualization
    """
    plt.figure(figsize=(12, 6))
    
    # Convert boolean success indicators to integers (1 for success, 0 for failure)
    success_values = [1 if s else 0 for s in successes]
    
    # Calculate moving average success rate (windowed success rate)
    if len(success_values) >= window:
        # Use convolution for efficient moving average
        success_rate = np.convolve(success_values, np.ones(window)/window, mode='valid')
        episodes = range(window-1, len(success_values))  # Adjust episode range for windowed data
        plt.plot(episodes, success_rate, 'g-', linewidth=2, label=f'Success Rate (window={window})')
    
    # Calculate cumulative success rate (overall success rate from start)
    cumulative_successes = np.cumsum(success_values)  # Running sum of successes
    cumulative_rate = cumulative_successes / np.arange(1, len(success_values) + 1)  # Divide by episode number
    plt.plot(range(len(success_values)), cumulative_rate, 'b--', linewidth=1.5, 
             label='Cumulative Success Rate')
    
    # Plot raw data with lower alpha for individual episode visibility
    plt.scatter(range(len(success_values)), success_values, alpha=0.5, s=10, c='gray', label='Raw Success (0/1)')
    
    # Set labels and formatting
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Success Rate', fontsize=12)
    plt.title('Success Rate over Episodes', fontsize=14)
    plt.ylim(-0.05, 1.05)  # Slightly extend y-axis for better visibility
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save the plot
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_door_positions(positions, filename="door_positions.png"):
    """
    Plot the distribution of door positions
    
    This function creates histograms showing how door positions are distributed
    across training episodes. It can handle both single door position lists and
    multi-room door position dictionaries.
    
    Args:
        positions: Door position data in one of two formats:
                  - List: [pos1, pos2, ...] for single door analysis
                  - Dict: {'roomA': [pos1, pos2, ...], 'roomB': [...], ...} for multi-room analysis
        filename (str): Output filename for the door position plot (default: "door_positions.png")
        
    Returns:
        None: Saves door position distribution plot to specified filename
        
    Features:
        - Handles both single and multi-room door position data
        - Creates separate subplots for each room when using dictionary input
        - Proper binning for histogram clarity
        - Room-specific color coding for multi-room plots
    """
    # Handle different input formats
    if isinstance(positions, dict):
        # Multi-room format: {'roomA': [pos1, pos2, ...], 'roomB': [...], ...}
        
        # Create subplots for each room
        fig, axes = plt.subplots(len(positions), 1, figsize=(12, 4 * len(positions)))
        
        # If only one room, make axes a list for consistent indexing
        if len(positions) == 1:
            axes = [axes]
        
        # Plot histogram for each room
        for i, (room, room_positions) in enumerate(sorted(positions.items())):
            # Create histogram with room-specific color
            axes[i].hist(room_positions, bins=20, alpha=0.7, color=f'C{i}')
            axes[i].set_xlabel('Door Position', fontsize=12)
            axes[i].set_ylabel('Frequency', fontsize=12)
            axes[i].set_title(f'Distribution of Door Positions for {room}', fontsize=14)
            axes[i].grid(True, alpha=0.3)
    else:
        # Single list format: [pos1, pos2, ...] (simple list of door positions)
        plt.figure(figsize=(12, 6))
        
        # Create single histogram
        plt.hist(positions, bins=20, alpha=0.7)
        plt.xlabel('Door Position', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Distribution of Door Positions', fontsize=14)
        plt.grid(True, alpha=0.3)
    
    # Adjust layout to prevent subplot overlap
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()  # Close figure to free memory