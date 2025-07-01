import os
import datetime
import traceback
import re
import json
import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks
import time

class DoorPositionTracker(DefaultCallbacks):
    """
    Door position tracker with navigator starting position tracking
    
    This class extends Ray RLLib's DefaultCallbacks to track door positions and navigator
    starting positions during reinforcement learning training episodes. It's designed for
    escape room environments where agents (navigators) must interact with doors in different rooms.
    
    Key Features:
    - Tracks door positions for rooms A, B, and C across training episodes
    - Records navigator agent starting positions for each episode  
    - Monitors episode success/failure rates and step counts
    - Saves tracking data to multiple file formats for analysis
    - Handles both parallel and AEC (Agent Environment Cycle) environments
    - Provides comprehensive debugging and logging capabilities
    """
    
    def __init__(self, 
                 debug_mode=True, 
                 run_dir=None,
                 debug_file_name='door_tracker_debug.txt',
                 positions_file_name='door_positions.txt',
                 summary_file_name='door_positions_summary.txt',
                 episode_stats_file='episode_stats.json',
                 direct_positions_file_name='door_positions_direct.txt',
                 seed=None):
        """
        Initialize DoorPositionTracker with configurable file paths
        
        Args:
            debug_mode (bool): Enable detailed debug logging
            run_dir (str): Directory to save all tracking files. If None, creates timestamped directory
            debug_file_name (str): Name for debug log file
            positions_file_name (str): Name for main positions tracking file
            summary_file_name (str): Name for episode summary file
            episode_stats_file (str): Name for JSON statistics file
            direct_positions_file_name (str): Name for direct format positions file
            seed (int): Random seed for reproducible tracking (optional)
        """
        super().__init__()
        
        # Core tracking data structures
        self.door_positions = []  # List of door position dictionaries for each episode
        self.navigator_start_positions = []  # List of navigator starting positions for each episode
        self.debug_mode = debug_mode
        self.seed_value = seed
        
        # Episode tracking counters
        self.total_episodes = 0  # Total number of episodes processed
        self.successful_episodes = 0  # Number of episodes that ended successfully (not truncated)
        
        # Step counting for episodes - tracks how many steps each episode took
        self.current_episode_step_count = 0  # Steps in the current ongoing episode
        self.episode_step_counts = {}  # Dictionary mapping episode IDs to their step counts
        
        # Store current episode navigator start position (temporary storage during episode)
        self.current_episode_nav_start = None
        
        # Configurable file names for different output formats
        self.debug_file_name = debug_file_name
        self.positions_file_name = positions_file_name
        self.summary_file_name = summary_file_name
        self.episode_stats_file = episode_stats_file
        self.direct_positions_file_name = direct_positions_file_name
        
        # Determine output directory - create timestamped directory if none provided
        if run_dir is None:
            # Fallback to a default directory if no run_dir is provided
            run_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                "door_tracking", 
                datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            )
        
        # Create full output directory structure
        self.output_dir = run_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Track the source of direct positions file (for reference)
        self.direct_positions_source = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
            self.direct_positions_file_name
        )
        
        # Create a new empty direct positions file in the output directory
        self._create_new_direct_positions_file()
        
        # Always try to load previous stats, regardless of seed (for resuming runs)
        self._load_previous_stats()
        
        # Log seed information if provided
        if seed is not None:
            # Log that we're using a seeded run, but don't reset counters
            self.debug_log(f"Using seeded run with seed: {seed}")
        
        # Initialize episode stats file with current state
        self._write_episode_stats()
        
        print(f"DoorPositionTracker initialized in {run_dir}")

    def seed(self, seed=None):
        """
        Seed the tracker's random number generator without resetting counters
        
        This method allows for reproducible tracking behavior while preserving
        existing episode counts and statistics.
        
        Args:
            seed (int): Random seed value. If None, generates seed from current time
            
        Returns:
            list: List containing the seed value used
        """
        if seed is None:
            seed = int(time.time()) % (2**31-1)
        self.seed_value = seed
        self.debug_log(f"DoorPositionTracker seeded with: {seed}")
        return [seed]
        
    def _create_new_direct_positions_file(self):
        """
        Create a new empty direct positions file in the output directory
        
        This file will store position data in a simple key:value format
        that's easy to parse for external analysis tools.
        """
        try:
            destination = os.path.join(self.output_dir, self.direct_positions_file_name)
            open(destination, 'w').close()  # Create empty file
            print(f"Created new direct positions file at {destination}")
        except Exception as e:
            print(f"Error creating new direct positions file: {e}")
    
    def _load_previous_stats(self):
        """
        Try to load previous stats if file exists to prevent reset
        
        This allows the tracker to resume counting from where it left off
        if training is restarted or continued from a checkpoint.
        """
        stats_path = os.path.join(self.output_dir, self.episode_stats_file)
        if os.path.exists(stats_path):
            try:
                with open(stats_path, 'r') as f:
                    stats = json.load(f)
                    
                # Restore important tracking variables from previous run
                if 'total_episodes' in stats:
                    self.total_episodes = stats['total_episodes']
                    
                if 'successful_episodes' in stats:
                    self.successful_episodes = stats['successful_episodes']
                
                self.debug_log(f"Loaded previous stats: {self.total_episodes} episodes")
                
            except Exception as e:
                self.debug_log(f"Error loading previous stats: {e}")
                print(f"Error loading previous stats: {e}")
    
    def _write_episode_stats(self):
        """
        Write simplified episode statistics to JSON file
        
        Creates a JSON file with basic episode statistics that can be easily
        read by analysis scripts or monitoring tools.
        """
        stats_path = os.path.join(self.output_dir, self.episode_stats_file)
        
        # Prepare simplified stats dictionary
        stats = {
            "total_episodes": self.total_episodes,
            "successful_episodes": self.successful_episodes,
            "last_updated": datetime.datetime.now().isoformat(),
        }
        
        # Write to file with error handling
        try:
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=4)
            self.debug_log(f"Successfully wrote episode stats to {stats_path}")
        except Exception as e:
            self.debug_log(f"Error writing episode stats: {e}")
            print(f"Error writing episode stats: {e}")
    
    def debug_log(self, message):
        """
        Log debug information to debug file with timestamp
        
        Args:
            message (str): Debug message to log
            
        Only writes to file if debug_mode is enabled. Each message is
        timestamped for easier debugging and analysis.
        """
        if self.debug_mode:
            debug_file_path = os.path.join(self.output_dir, self.debug_file_name)
            try:
                with open(debug_file_path, 'a') as f:
                    timestamp = datetime.datetime.now().isoformat()
                    f.write(f"[{timestamp}] {message}\n")
            except Exception as e:
                print(f"Error writing to debug log: {e}")
    
    def _extract_navigator_start_pos(self, env_obj):
        """
        Helper function to extract navigator start position from an environment object
        
        This method tries multiple different ways to access the navigator's starting
        position since different environment wrappers and configurations may store
        this information in different locations.
        
        Args:
            env_obj: Environment object to search for navigator position
            
        Returns:
            list/tuple/None: Navigator starting position [x, y, z] or None if not found
        """
        if env_obj is None:
            return None
            
        # Try various access patterns to find navigator_start_pos
        # First, direct access to navigator_start_pos attribute
        if hasattr(env_obj, 'navigator_start_pos'):
            return env_obj.navigator_start_pos
            
        # For EscapeRoomBaseEnv in miniworld_env.py - check world attribute
        if hasattr(env_obj, 'world') and hasattr(env_obj.world, 'navigator_start_pos'):
            return env_obj.world.navigator_start_pos
            
        # For EscapeRoomEnv or ParallelEscapeRoomEnv - nested env access
        if hasattr(env_obj, 'env') and hasattr(env_obj.env, 'world') and hasattr(env_obj.env.world, 'navigator_start_pos'):
            return env_obj.env.world.navigator_start_pos
            
        # For EscapeRoomGymWrapper - double nested env access
        if hasattr(env_obj, 'env') and hasattr(env_obj.env, 'env') and hasattr(env_obj.env.env, 'world') and hasattr(env_obj.env.env.world, 'navigator_start_pos'):
            return env_obj.env.env.world.navigator_start_pos
            
        # For agent object directly - check agent position
        if hasattr(env_obj, 'agent') and hasattr(env_obj.agent, 'pos'):
            return env_obj.agent.pos
            
        # Try to directly access the current_path if it exists (path planning environments)
        if hasattr(env_obj, 'current_path') and env_obj.current_path and len(env_obj.current_path) > 0:
            return env_obj.current_path[0]  # First position in path is usually start
            
        # For world object in case env_obj is already the world
        if hasattr(env_obj, 'current_path') and env_obj.current_path and len(env_obj.current_path) > 0:
            return env_obj.current_path[0]  # First position in path
            
        return None
        
    def _find_navigator_start_pos_in_env(self, env):
        """
        Recursive helper to find navigator start position by traversing environment attributes
        
        This method recursively searches through the environment object hierarchy
        to find the navigator's starting position, handling complex nested environment
        structures common in Ray RLLib.
        
        Args:
            env: Environment object to search recursively
            
        Returns:
            list/tuple/None: Navigator starting position or None if not found
        """
        if env is None:
            return None
            
        # Try direct access using our helper function first
        nav_pos = self._extract_navigator_start_pos(env)
        if nav_pos is not None:
            return nav_pos
            
        # Try recursively accessing different parts of the environment
        # Check all attributes that might lead to the world or agent
        for attr_name in ['env', 'world', 'base_env', 'vector_env']:
            if hasattr(env, attr_name):
                attr_value = getattr(env, attr_name)
                
                # Skip None values or circular references (already checked objects)
                if attr_value is None or attr_value is env:
                    continue
                    
                # Try recursive access on this attribute
                nav_pos = self._find_navigator_start_pos_in_env(attr_value)
                if nav_pos is not None:
                    return nav_pos
                    
        # If the env has envs (like in a vector environment with multiple parallel envs)
        if hasattr(env, 'envs'):
            for sub_env in env.envs:
                nav_pos = self._find_navigator_start_pos_in_env(sub_env)
                if nav_pos is not None:
                    return nav_pos
                    
        return None
    
    def _find_navigator_start_pos_in_episode(self, episode):
        """
        Try to extract navigator start position from episode object
        
        When the environment objects don't contain the position information,
        this method attempts to extract it from the episode's observation
        or info data.
        
        Args:
            episode: RLLib episode object containing observations and info
            
        Returns:
            list/None: Navigator starting position or None if not found
        """
        if episode is None:
            return None
            
        # Try to extract from the episode's observations or infos
        try:
            # First check episode.user_data which might have custom info
            if hasattr(episode, 'user_data') and isinstance(episode.user_data, dict):
                user_data = episode.user_data
                if 'navigator_start_pos' in user_data:
                    return user_data['navigator_start_pos']
                    
            # Check episode's observations (first observation might have agent position)
            if hasattr(episode, 'observations'):
                for agent, obs_list in episode.observations.items():
                    if agent == 'navigator' and len(obs_list) > 0:
                        # Navigator's observation may include position info
                        first_obs = obs_list[0]
                        if isinstance(first_obs, np.ndarray) and len(first_obs) >= 2:
                            # This is approximate - depends on observation structure
                            # For many environments, the agent position is in the first few elements
                            # Format might be [x, z] or normalized values
                            pos_x = first_obs[0]
                            pos_z = first_obs[1]
                            return [float(pos_x), 0.0, float(pos_z)]  # Assume y=0 for 2D environments
                            
            # Check first info dictionary for position keys
            if hasattr(episode, 'infos'):
                for agent, info_list in episode.infos.items():
                    if len(info_list) > 0 and isinstance(info_list[0], dict):
                        first_info = info_list[0]
                        # Try various keys that might contain position
                        for key in ['pos', 'position', 'agent_pos', 'start_pos', 'navigator_start_pos']:
                            if key in first_info:
                                return first_info[key]
        except Exception as e:
            self.debug_log(f"Error extracting from episode: {e}")
            
        return None
    
    def on_episode_start(self, *, worker=None, base_env=None, policies=None, episode=None, env_runner=None, env=None, **kwargs):
        """
        Track when an episode starts and capture navigator's starting position
        
        This callback is called at the beginning of each training episode.
        It attempts to capture the navigator's starting position from various
        sources and initializes episode tracking variables.
        
        Args:
            worker: RLLib worker object
            base_env: Base environment object
            policies: Policy objects
            episode: Episode object
            env_runner: Environment runner object  
            env: Environment object
            **kwargs: Additional keyword arguments
        """
        try:
            self.debug_log(f"Episode {self.total_episodes + 1} started")
            
            # IMPORTANT: Reset step counter for this episode
            self.current_episode_step_count = 0
            
            # Store the episode ID if available, otherwise use episode counter
            if episode and hasattr(episode, 'episode_id'):
                self.current_episode_id = episode.episode_id
            else:
                self.current_episode_id = self.total_episodes + 1
            
            # Clear this episode's step count in tracking dictionary
            self.episode_step_counts[self.current_episode_id] = 0
            
            # Try to find navigator starting position from multiple sources
            # Start with our best known location: directly in mini_world env
            if base_env is not None:
                self.debug_log("Checking base_env for navigator start pos")
                self.current_episode_nav_start = self._find_navigator_start_pos_in_env(base_env)
                if self.current_episode_nav_start is not None:
                    self.debug_log(f"Found navigator start position in base_env: {self.current_episode_nav_start}")
                    return
                    
            if env is not None:
                self.debug_log("Checking env for navigator start pos")
                self.current_episode_nav_start = self._find_navigator_start_pos_in_env(env)
                if self.current_episode_nav_start is not None:
                    self.debug_log(f"Found navigator start position in env: {self.current_episode_nav_start}")
                    return
                    
            if env_runner is not None:
                self.debug_log("Checking env_runner for navigator start pos")
                self.current_episode_nav_start = self._find_navigator_start_pos_in_env(env_runner)
                if self.current_episode_nav_start is not None:
                    self.debug_log(f"Found navigator start position in env_runner: {self.current_episode_nav_start}")
                    return
                    
            if episode is not None:
                self.debug_log("Checking episode for navigator start pos")
                self.current_episode_nav_start = self._find_navigator_start_pos_in_episode(episode)
                if self.current_episode_nav_start is not None:
                    self.debug_log(f"Found navigator start position in episode: {self.current_episode_nav_start}")
                    return
                    
            # If we reach here, we couldn't find the navigator start position
            self.debug_log("Could not find navigator start position at episode start")
            self.current_episode_nav_start = None

            # Reset step counter for this episode (redundant but ensuring it's reset)
            self.current_episode_step_count = 0
            
            # Store the episode ID if available (redundant but ensuring it's set)
            if episode and hasattr(episode, 'episode_id'):
                self.current_episode_id = episode.episode_id
            else:
                self.current_episode_id = self.total_episodes + 1
            
            self.episode_step_counts[self.current_episode_id] = 0
                
        except Exception as e:
            self.debug_log(f"Error in on_episode_start: {e}")
            print(f"Error in on_episode_start: {e}")
            traceback.print_exc()

    """
    COMMENTED OUT VERSION - Simple step counting approach
    
    def on_episode_step(self, *, worker=None, base_env=None, policies=None, episode=None, env_runner=None, env=None, **kwargs):
        # This is a simpler version that just counts raw environment steps
        try:
            # Simply increment the counter for any step
            self.current_episode_step_count += 1
            
            # Store in episode dictionary
            if hasattr(self, 'current_episode_id'):
                self.episode_step_counts[self.current_episode_id] = self.current_episode_step_count
            
            # Log for debugging every 10 steps
            if self.current_episode_step_count % 10 == 0:
                self.debug_log(f"Episode step count: {self.current_episode_step_count}")
        except Exception as e:
            self.debug_log(f"Error in on_episode_step: {e}")
    """
        
    def on_episode_step(self, *, worker=None, base_env=None, policies=None, episode=None, env_runner=None, env=None, **kwargs):
        """
        Track steps for each episode, counting navigator steps correctly in both parallel and AEC environments
        
        This method handles the complexity of step counting in different environment types:
        - Parallel environments: All agents act simultaneously, so each env step = 1 navigator step
        - AEC environments: Agents take turns, so every 2 env steps = 1 navigator step
        
        Args:
            worker: RLLib worker object
            base_env: Base environment object  
            policies: Policy objects
            episode: Episode object
            env_runner: Environment runner object
            env: Environment object
            **kwargs: Additional keyword arguments
        """
        try:
            # Simply increment step counter for raw environment steps
            self.current_episode_step_count += 1
            
            # Get navigator steps directly from environment if possible
            navigator_steps = None
            is_parallel = False
            
            # First check if we can determine the environment type (parallel vs AEC)
            if base_env and hasattr(base_env, 'envs') and base_env.envs:
                env_obj = base_env.envs[0]
                
                # Check different paths for the is_parallel attribute
                if hasattr(env_obj, 'is_parallel'):
                    is_parallel = env_obj.is_parallel
                elif hasattr(env_obj, 'env') and hasattr(env_obj.env, 'is_parallel'):
                    is_parallel = env_obj.env.is_parallel
                
                # Additional check for env type - check class name for 'Parallel' substring
                if not is_parallel:
                    # Check class names for 'Parallel' substring
                    if 'Parallel' in str(type(env_obj).__name__):
                        is_parallel = True
                    elif hasattr(env_obj, 'env') and 'Parallel' in str(type(env_obj.env).__name__):
                        is_parallel = True
                    
                    # Check config directly if present
                    if hasattr(env_obj, 'config') and env_obj.config.get('parallel_env', False):
                        is_parallel = True
                    elif hasattr(env_obj, 'env') and hasattr(env_obj.env, 'config') and env_obj.env.config.get('parallel_env', False):
                        is_parallel = True
            
            # Try to get navigator steps directly from environment
            if base_env and hasattr(base_env, 'envs') and base_env.envs:
                env_obj = base_env.envs[0]
                
                # Try different paths to find navigator_steps counter
                if hasattr(env_obj, 'navigator_steps'):
                    navigator_steps = env_obj.navigator_steps
                elif hasattr(env_obj, 'env') and hasattr(env_obj.env, 'navigator_steps'):
                    navigator_steps = env_obj.env.navigator_steps
                elif hasattr(env_obj, 'steps'):
                    navigator_steps = env_obj.steps
                
                # Look for step count in episode info
                if hasattr(episode, 'last_info') and episode.last_info:
                    for agent, info in episode.last_info.items():
                        if isinstance(info, dict) and 'steps' in info:
                            navigator_steps = max(navigator_steps or 0, info['steps'])
            
            # If we still can't get navigator steps, make our best guess based on env type
            if navigator_steps is None:
                # Force parallel mode for testing (uncomment to force parallel counting)
                # is_parallel = True
                
                if is_parallel:
                    # For parallel environments, take the raw step count
                    navigator_steps = self.current_episode_step_count
                else:
                    # For AEC environments, normalize by dividing by 2 (one agent step per two env steps)
                    navigator_steps = (self.current_episode_step_count + 1) // 2
            
            # Store in episode dictionary for later retrieval
            if hasattr(self, 'current_episode_id'):
                self.episode_step_counts[self.current_episode_id] = navigator_steps
            
            # Log for debugging every 10 navigator steps
            if navigator_steps % 10 == 0:
                self.debug_log(f"Navigator step count: {navigator_steps}, Env type: {'parallel' if is_parallel else 'AEC'}")
        except Exception as e:
            self.debug_log(f"Error in on_episode_step: {e}")
            print(f"Error in on_episode_step: {e}")

    def write_tracker_state(self):
        """
        Write tracker state to a file that won't be affected by RLlib
        
        This creates a separate state file that's independent of RLLib's
        checkpoint system, useful for monitoring progress externally.
        """
        try:
            state_file = os.path.join(self.output_dir, 'tracker_state_direct.json')
            with open(state_file, 'w') as f:
                json.dump({
                    'total_episodes': self.total_episodes,
                    'successful_episodes': self.successful_episodes,
                    'timestamp': datetime.datetime.now().isoformat()
                }, f, indent=4)
        except Exception as e:
            print(f"Error writing tracker state: {e}")

    def _write_episode_stats_with_steps(self, steps, success):
        """
        Write enhanced episode statistics to JSON file including step count data
        
        This method creates comprehensive statistics including:
        - Step count distributions for successful/failed/truncated episodes
        - Average step counts by episode outcome
        - Historical data (keeps last 100 episodes of each type)
        
        Args:
            steps (int): Number of steps taken in the episode
            success (bool): Whether the episode was successful
        """
        stats_path = os.path.join(self.output_dir, self.episode_stats_file)
        
        # Load existing stats if available to preserve historical data
        existing_stats = {}
        if os.path.exists(stats_path):
            try:
                with open(stats_path, 'r') as f:
                    existing_stats = json.load(f)
            except Exception as e:
                self.debug_log(f"Error reading existing stats: {e}")
        
        # Initialize step tracking data structure if not present
        if 'step_data' not in existing_stats:
            existing_stats['step_data'] = {
                'total_steps': 0,
                'successful_episodes_steps': [],
                'failed_episodes_steps': [],
                'truncated_episodes_steps': [],  # New list for truncated episodes
                'avg_steps_successful': 0,
                'avg_steps_failed': 0
            }
        
        # Update step tracking data with current episode
        step_data = existing_stats['step_data']
        step_data['total_steps'] += steps
        
        # Determine if this is a truncated episode based on step count and max steps
        max_steps = 500  # Default maximum steps per episode
        if hasattr(self, 'config') and 'max_episode_steps' in self.config:
            max_steps = self.config['max_episode_steps']
        
        is_truncated = (steps >= max_steps)
        
        # Categorize episode by outcome and store step count
        if success:
            step_data['successful_episodes_steps'].append(steps)
            # Keep only the last 100 successful episodes for memory efficiency
            if len(step_data['successful_episodes_steps']) > 100:
                step_data['successful_episodes_steps'] = step_data['successful_episodes_steps'][-100:]
        elif is_truncated:
            # If truncated, add to both failed and truncated lists
            step_data['failed_episodes_steps'].append(steps)
            step_data['truncated_episodes_steps'].append(steps)
            # Keep only the last 100 truncated episodes
            if len(step_data['truncated_episodes_steps']) > 100:
                step_data['truncated_episodes_steps'] = step_data['truncated_episodes_steps'][-100:]
            # Also keep failed list to 100 entries
            if len(step_data['failed_episodes_steps']) > 100:
                step_data['failed_episodes_steps'] = step_data['failed_episodes_steps'][-100:]
        else:
            step_data['failed_episodes_steps'].append(steps)
            # Keep only the last 100 failed episodes for memory efficiency
            if len(step_data['failed_episodes_steps']) > 100:
                step_data['failed_episodes_steps'] = step_data['failed_episodes_steps'][-100:]
        
        # Calculate running averages for different episode types
        if step_data['successful_episodes_steps']:
            step_data['avg_steps_successful'] = sum(step_data['successful_episodes_steps']) / len(step_data['successful_episodes_steps'])
        if step_data['failed_episodes_steps']:
            step_data['avg_steps_failed'] = sum(step_data['failed_episodes_steps']) / len(step_data['failed_episodes_steps'])
        
        # Add average for truncated episodes
        if 'truncated_episodes_steps' in step_data and step_data['truncated_episodes_steps']:
            step_data['avg_steps_truncated'] = sum(step_data['truncated_episodes_steps']) / len(step_data['truncated_episodes_steps'])
        
        # Prepare enhanced stats dictionary with all tracking data
        stats = {
            "total_episodes": self.total_episodes,
            "successful_episodes": self.successful_episodes,
            "last_updated": datetime.datetime.now().isoformat(),
            "step_data": step_data
        }
        
        # Write to file with error handling
        try:
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=4)
            self.debug_log(f"Successfully wrote enhanced episode stats with step data to {stats_path}")
        except Exception as e:
            self.debug_log(f"Error writing enhanced episode stats: {e}")
            print(f"Error writing enhanced episode stats: {e}")
    
    def on_episode_end(self, *, worker=None, base_env=None, policies=None, episode=None, 
      env_runner=None, env=None, **kwargs):
        """
        Track door positions and navigator starting position at the end of each episode
        
        This is the main data collection method that:
        1. Extracts door positions from episode info
        2. Determines episode success/failure
        3. Records step counts
        4. Writes all data to multiple output files
        5. Updates statistics
        
        Args:
            worker: RLLib worker object
            base_env: Base environment object
            policies: Policy objects  
            episode: Episode object containing all episode data
            env_runner: Environment runner object
            env: Environment object
            **kwargs: Additional keyword arguments
        """
        
        try:
            # Initialize variables at the start
            door_positions = {}  # Will hold door positions for rooms A, B, C
            episode_success = False
            episode_steps = 500  # Default maximum steps if we can't determine actual count
            
            # Increment episode counter
            self.total_episodes += 1
            
            # Extract door positions from episode info
            # This searches through all agents' info to find door_positions data
            if episode:
                try:
                    # Use get_infos() method to get episode information
                    infos = episode.get_infos()
                    
                    # Search through agents to find door positions
                    for agent, info_list in infos.items():
                        if info_list and len(info_list) > 0:
                            # Search backwards through the info list to find door_positions
                            # (most recent info is more likely to have final door positions)
                            for i in range(len(info_list) - 1, -1, -1):
                                info_entry = info_list[i]
                                
                                if isinstance(info_entry, dict) and 'door_positions' in info_entry:
                                    door_positions = info_entry['door_positions']
                                    
                                    # Convert numpy values to regular Python floats for JSON serialization
                                    if door_positions:
                                        door_positions = {
                                            room: float(pos) for room, pos in door_positions.items()
                                        }
                                    break
                            
                            # If we found door_positions, stop searching other agents
                            if door_positions:
                                break
                                
                except Exception as e:
                    print(f"Error extracting door positions: {e}")
            
            # Determine episode success and step count
            if episode:
                # Episode is successful if it terminated naturally (not truncated due to max steps)
                episode_success = episode.is_terminated and not episode.is_truncated
                
                # Get step count from our tracking dictionary
                if hasattr(self, 'current_episode_id') and self.current_episode_id in self.episode_step_counts:
                    episode_steps = self.episode_step_counts[self.current_episode_id]
                else:
                    # Fallback to raw step count if tracking failed
                    episode_steps = self.current_episode_step_count
                
                # If truncated, use max steps (episode hit step limit)
                if episode.is_truncated:
                    episode_steps = 500  # Or get from config if available
            
            # Update success counter
            if episode_success:
                self.successful_episodes += 1
            
            # Format navigator start position for output
            nav_start_str = "Unknown"
            if self.current_episode_nav_start is not None:
                if isinstance(self.current_episode_nav_start, (list, tuple, np.ndarray)) and len(self.current_episode_nav_start) >= 3:
                    nav_start_str = f"{self.current_episode_nav_start[0]:.2f},{self.current_episode_nav_start[1]:.2f},{self.current_episode_nav_start[2]:.2f}"
                else:
                    nav_start_str = str(self.current_episode_nav_start)
            
            # Write door positions to files (only if we found them)
            if door_positions:
                # Store in internal lists for later analysis
                self.door_positions.append(door_positions)
                self.navigator_start_positions.append(self.current_episode_nav_start)
                
                # Write to main positions file (human-readable format)
                positions_file_path = os.path.join(self.output_dir, self.positions_file_name)
                with open(positions_file_path, 'a') as f:
                    line = f"Episode {self.total_episodes} | Nav Start: {nav_start_str} | " \
                        f"roomA:{door_positions.get('roomA', 0.0):.4f}, " \
                        f"roomB:{door_positions.get('roomB', 0.0):.4f}, " \
                        f"roomC:{door_positions.get('roomC', 0.0):.4f} | " \
                        f"Success: {1 if episode_success else 0} | " \
                        f"Steps: {episode_steps}\n"
                    f.write(line)
                
                # Write to direct positions file (key:value format for easy parsing)
                direct_file_path = os.path.join(self.output_dir, self.direct_positions_file_name)
                with open(direct_file_path, 'a') as f:
                    f.write(f"nav_start:{nav_start_str}\n")
                    f.write(f"roomA:{door_positions.get('roomA', 0.0):.4f}\n")
                    f.write(f"roomB:{door_positions.get('roomB', 0.0):.4f}\n")
                    f.write(f"roomC:{door_positions.get('roomC', 0.0):.4f}\n")
                    f.write(f"success:{1 if episode_success else 0}\n")
                    f.write(f"steps:{episode_steps}\n")
                
                # Log success
                self.debug_log(f"Episode {self.total_episodes}: Door positions recorded: {door_positions}")
            else:
                # No door positions found - log for debugging
                self.debug_log(f"Episode {self.total_episodes}: No door positions found")
            
            # Reset for next episode
            self.current_episode_nav_start = None
            
            # Update all tracking files
            self.write_tracker_state()
            self._write_episode_stats_with_steps(episode_steps, episode_success)
            
        except Exception as e:
            self.debug_log(f"Error in on_episode_end: {e}")
            print(f"Error in on_episode_end: {e}")
            import traceback
            traceback.print_exc()
    
    def on_train_result(self, *, algorithm, result, **kwargs):
        """
        Save summary periodically during training
        
        This callback is called periodically during training (after certain
        numbers of episodes or time intervals). It writes comprehensive
        summary files and maintains statistics.
        
        Args:
            algorithm: RLLib algorithm object
            result: Training result dictionary
            **kwargs: Additional keyword arguments
        """
        try:
            # Log the current state for debugging
            self.debug_log(f"Total tracked door positions: {len(self.door_positions)}")
            self.debug_log(f"Total episodes completed: {self.total_episodes}")
            
            # Save all positions captured so far to summary file
            if self.door_positions:
                summary_file_path = os.path.join(self.output_dir, self.summary_file_name)
                with open(summary_file_path, 'w') as f:
                    # Write header information
                    f.write(f"Total position sets: {len(self.door_positions)}\n")
                    f.write(f"Total episodes trained: {self.total_episodes}\n")
                    f.write(f"Total successful episodes: {self.successful_episodes}\n\n")
                    f.write("Format: navigator_start, roomA, roomB, roomC\n")
                    
                    # Write each recorded position set
                    for i in range(len(self.door_positions)):
                        pos_dict = self.door_positions[i]
                        nav_start = "Unknown"
                        
                        # Format navigator start position if available
                        if i < len(self.navigator_start_positions) and self.navigator_start_positions[i] is not None:
                            nav_pos = self.navigator_start_positions[i]
                            if isinstance(nav_pos, (list, tuple, np.ndarray)) and len(nav_pos) >= 3:
                                nav_start = f"({nav_pos[0]:.2f},{nav_pos[1]:.2f},{nav_pos[2]:.2f})"
                            else:
                                nav_start = str(nav_pos)
                                
                        f.write(f"Pos {i+1}: Nav Start: {nav_start}, " 
                               f"roomA:{pos_dict.get('roomA', 0.0):.4f}, "
                               f"roomB:{pos_dict.get('roomB', 0.0):.4f}, "
                               f"roomC:{pos_dict.get('roomC', 0.0):.4f}\n")
            
            # Update episode stats file
            self._write_episode_stats()

            # Memory management: limit the size of episode step tracking dictionary
            if len(self.episode_step_counts) > 100:
                # Only keep most recent 50 episodes to prevent memory bloat
                oldest_keys = sorted(list(self.episode_step_counts.keys()))[:50]
                for key in oldest_keys:
                    self.episode_step_counts.pop(key, None)
                    
        except Exception as e:
            self.debug_log(f"Error in on_train_result: {e}")
            print(f"Error in on_train_result: {str(e)}")
            traceback.print_exc()