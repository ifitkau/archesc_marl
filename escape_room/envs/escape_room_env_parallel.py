"""
PettingZoo environment for the escape room using Parallel API

This module implements the multi-agent environment wrapper that coordinates the navigator
and door controller agents using PettingZoo's Parallel API. Unlike sequential (AEC) APIs,
this implementation ensures all agents act simultaneously at each environment step.

Key Features:
- Simultaneous agent action execution
- Proper step counting for parallel environments
- Dynamic door controller activation based on episode timing
- Comprehensive reward coordination between agents
- Episode management with success/failure handling
- Seed management for reproducible training

This implementation ensures:
1. Each environment step corresponds to all agents acting simultaneously
2. Step counting properly differentiates between AEC and parallel modes  
3. Max episode steps is based on environment steps, not individual agent steps
4. Door controller rewards are properly coordinated with navigator progress
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import ParallelEnv
from escape_room.envs.miniworld_env import EscapeRoomBaseEnv
from escape_room.envs.navigator_agent import NavigatorAgent
from escape_room.envs.door_agent import DoorControllerAgent


class ParallelEscapeRoomEnv(ParallelEnv):
    """
    Multi-agent escape room environment using PettingZoo's Parallel API
    
    This environment coordinates two agents in a shared 3D world:
    - Navigator: Navigates from start position to terminal (acts every step)
    - Door Controller: Controls door positions strategically (acts periodically)
    
    The parallel API ensures both agents observe and act simultaneously, creating
    a truly interactive multi-agent environment where the door controller must
    anticipate navigator behavior and the navigator must adapt to changing doors.
    
    Agent Interaction:
    - Navigator receives rewards for efficient navigation and goal achievement
    - Door controller receives rewards for enabling navigator success
    - Both agents share the same physical environment and door positions
    - Episode terminates when navigator reaches goal or time limit is exceeded
    """
    
    # PettingZoo metadata for environment capabilities
    metadata = {
        "render_modes": ["human", "rgb_array"], 
        "render_fps": 30, 
        "is_parallelizable": True
    }
    
    def __init__(self, config=None, render_mode=None, view="top"):
        """
        Initialize the multi-agent escape room environment
        
        Sets up the 3D world, creates both agents, configures observation/action spaces,
        and initializes all tracking variables for episode management.
        
        Args:
            config (dict): Configuration dictionary with environment parameters
            render_mode (str): Rendering mode ("human", "rgb_array", or None)
            view (str): Camera view for rendering ("top", "agent", etc.)
        """
        super().__init__()
        
        # =====================================
        # CONFIGURATION SETUP
        # =====================================
        # Load default configuration if none provided
        if config is None:
            from escape_room.config.default_config import get_default_config
            config = get_default_config()
        
        self.config = config
        self.render_mode = render_mode
        
        # =====================================
        # ENVIRONMENT STATE INITIALIZATION
        # =====================================
        # UI and debugging state
        self.paused = False                     # Pause state for interactive debugging
        self.window_created = False             # Track rendering window creation
        self.debug_mode = True if config and config.get('debug_mode') else False  # Debug output control
        
        # =====================================
        # WORLD AND AGENT CREATION
        # =====================================
        # Create the base 3D environment (rooms, doors, physics)
        self.world = EscapeRoomBaseEnv(config=config, render_mode=render_mode, view=view)
        
        # Create navigator agent (pathfinding and navigation)
        self.navigator = NavigatorAgent(config=config)
        
        # Create door controller agent (strategic door placement)
        self.door_controller = DoorControllerAgent(
            config=config, 
            door_position=config["door_position"],              # Initial door position
            door_position_min=config["door_position_min"],      # Minimum allowed position
            door_position_max=config["door_position_max"]       # Maximum allowed position
        )
        
        # Track current door position for reference
        self.current_door_position = self.door_controller.door_positions.get('roomA', 2.0)

        # =====================================
        # PETTINGZOO AGENT SETUP
        # =====================================
        # Define all possible agents in the environment
        self.possible_agents = ["navigator", "door_controller"]
        self.agents = self.possible_agents[:]  # Currently active agents
        
        # Define action spaces for each agent
        self._action_spaces = {
            "navigator": self.navigator.action_space,           # Discrete: turn left/right, move forward
            "door_controller": self.door_controller.action_space # Multi-discrete: door positions for each room
        }
        
        # Define observation spaces for each agent
        self._observation_spaces = {
            "navigator": self.navigator.observation_space,     # Spatial awareness, LIDAR, door positions
            "door_controller": self.door_controller.observation_space  # Navigator state, door positions, terminal info
        }
        
        # =====================================
        # EPISODE AND STEP TRACKING
        # =====================================
        # Episode management
        self.episode_count = 0                  # Total episodes run
        
        # Step counting (CRITICAL for parallel environments)
        self.steps = 0                          # Total environment steps
        self.navigator_steps = 0                # Steps where navigator acted
        self.parallel_env_steps = 0             # Parallel-specific step counter
        
        # =====================================
        # DOOR CONTROLLER STATE MANAGEMENT
        # =====================================
        # Track door controller activity and permissions
        self.door_has_ever_acted = False        # Has door controller ever taken an action?
        self.door_acted_this_episode = False    # Has door controller acted in current episode?
        self.door_can_act = False               # Is door controller allowed to act this episode?

        # =====================================
        # REWARD AND STATE TRACKING
        # =====================================
        # Track cumulative rewards for each agent (important for parallel environments)
        self._cumulative_rewards = {agent: 0.0 for agent in self.possible_agents}
        
        # Episode termination state
        self.terminations = {agent: False for agent in self.possible_agents}   # Agent-specific termination
        self.truncations = {agent: False for agent in self.possible_agents}    # Agent-specific truncation
        self.infos = {agent: {} for agent in self.possible_agents}             # Agent-specific info
    
    # =====================================
    # PETTINGZOO PROPERTY INTERFACES
    # =====================================
    
    @property
    def observation_spaces(self):
        """Return observation spaces for all agents"""
        return self._observation_spaces
    
    @property
    def action_spaces(self):
        """Return action spaces for all agents"""
        return self._action_spaces
    
    def observation_space(self, agent):
        """Return observation space for specific agent"""
        return self._observation_spaces[agent]
    
    def action_space(self, agent):
        """Return action space for specific agent"""
        return self._action_spaces[agent]
    
    def observe(self, agent):
        """
        Get observation for a specific agent
        
        Args:
            agent (str): Agent name ("navigator" or "door_controller")
            
        Returns:
            np.array: Agent's observation of the current environment state
        """
        if agent == "navigator":
            return self.navigator.get_observation(self.world)
        elif agent == "door_controller":
            return self.door_controller.get_observation(self.world)
        return None
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment for a new episode
        
        This method handles complete environment reset including:
        - Episode counter management and step counter reset
        - Random number generator seeding for reproducibility
        - World state reset and agent initialization
        - Door controller activation logic based on episode timing
        - Initial door position setup (default positions for first episode)
        
        Args:
            seed (int, optional): Random seed for reproducible episodes
            options (dict, optional): Additional reset options
            
        Returns:
            tuple: (observations, infos) - Initial observations and info for all agents
        """
        # =====================================
        # EPISODE INITIALIZATION
        # =====================================
        # Reset debug and state flags
        self._printed_door_skipped = False
        
        # Increment episode counter for tracking
        self.episode_count += 1
        
        # CRITICAL: Reset step counters for proper parallel environment behavior
        self.steps = 0                          # Total environment steps
        self.navigator_steps = 0                # Navigator-specific steps
        
        # =====================================
        # MAX EPISODE STEPS VALIDATION
        # =====================================
        # Ensure the base environment has correct step limits
        if hasattr(self.world, 'max_episode_steps'):
            requested_max_steps = self.config.get("max_episode_steps", 500)
            if self.world.max_episode_steps != requested_max_steps:
                print(f"Correcting max_episode_steps from {self.world.max_episode_steps} to {requested_max_steps}")
                self.world.max_episode_steps = requested_max_steps
        
        # =====================================
        # RANDOM NUMBER GENERATOR SEEDING
        # =====================================
        # Set seeds for reproducible behavior across all components
        if seed is not None:
            self.seed(seed)
            # Create derived seeds for each agent to ensure independence
            nav_seed = (seed * 12345) % (2**31-1)
            door_seed = (seed * 54321) % (2**31-1)
            self.navigator.seed(nav_seed)
            self.door_controller.seed(door_seed)
        
        # =====================================
        # WORLD AND AGENT RESET
        # =====================================
        # Synchronize episode count with world
        self.world.episode_count = self.episode_count
        
        # Reset world state
        self.world.portals = []                 # Clear all door connections
        self.world.reset(seed=seed)             # Reset world geometry and agent placement
        self.world.step_count = 0               # Reset world's internal step counter
        self.end_rewards_given = False          # Reset end-of-episode reward flags
        
        # Reset both agents to initial state
        self.navigator.reset(self.world)
        self.door_controller.reset(self.world)

        # =====================================
        # DOOR CONTROLLER ACTIVATION LOGIC
        # =====================================
        # Get configuration parameters for door controller behavior
        door_move_frequency = self.config.get("door_move_frequency", 1)         # How often door can move
        episodes_per_room = self.config.get("episodes_per_room", 3)             # Episodes per room cycle
        door_agent_start_episode = self.config.get("door_agent_start_episode", 1)  # When door agent starts

        # =====================================
        # FIRST EPISODE SPECIAL HANDLING
        # =====================================
        if self.episode_count == 1:
            # On the very first episode, set doors to default position from config
            # This ensures doors exist and are properly positioned from the beginning
            default_door_pos = self.config.get("door_position", 3.0)
            print(f"First episode: Initializing all door positions to {default_door_pos}")
            
            # Initialize door positions for all rooms
            if not hasattr(self, 'initial_door_positions'):
                self.initial_door_positions = {}
                for room_name in ['roomA', 'roomB', 'roomC']:
                    # Use the default door position from configuration
                    self.initial_door_positions[room_name] = default_door_pos
                    
                    # Move the door to this position in the 3D world
                    self.door_controller._move_specific_room_door(self.world, room_name, default_door_pos)
                    self.door_controller.door_positions[room_name] = default_door_pos
            
            # Store these as the current door positions for reference
            self.current_door_positions = self.initial_door_positions.copy()
            
            # Don't allow door controller to act on first episode
            self.door_can_act = False
        else:
            # =====================================
            # SUBSEQUENT EPISODE DOOR LOGIC
            # =====================================
            # For episodes after the first, use standard door controller activation logic
            
            # Determine if door controller can act based on episode timing
            self.door_can_act = (self.episode_count > door_agent_start_episode) and \
                               ((self.episode_count - 1) % door_move_frequency == 0)
            
            # If door cannot act, maintain existing door positions
            if not self.door_can_act:
                # Choose which positions to use based on door agent status
                if self.episode_count <= door_agent_start_episode:
                    # Use initial default positions until door agent starts
                    positions_to_use = self.initial_door_positions
                else:
                    # Use current positions maintained by door controller
                    positions_to_use = self.door_controller.door_positions
                    
                # Apply the chosen door positions to the 3D world
                for room_name, pos in positions_to_use.items():
                    if hasattr(self.door_controller, '_move_specific_room_door'):
                        self.door_controller._move_specific_room_door(self.world, room_name, pos)
        
        # =====================================
        # DOOR CONTROLLER STATE SETUP
        # =====================================
        # Configure door controller permissions and tracking
        self.world.door_can_act = self.door_can_act        # Set world-level permission flag
        self.door_acted_this_episode = False               # Reset action tracking
        self.door_has_ever_acted = False                   # Reset lifetime action tracking
        
        # Reset agent list to include all possible agents
        self.agents = self.possible_agents[:]
            
        # =====================================
        # TERMINATION STATE RESET
        # =====================================
        # CRITICAL: Ensure all termination flags are reset for new episode
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        
        # Reset world-level success tracking
        if hasattr(self.world, '_episode_success'):
            self.world._episode_success = False
        if hasattr(self.world, '_terminal_reached'):
            self.world._terminal_reached = False
        
        # Initialize info dictionaries
        self.infos = {agent: {} for agent in self.possible_agents}

        # =====================================
        # VISUAL UPDATE AND RENDERING
        # =====================================
        # Force visual update to ensure doors are properly rendered
        if hasattr(self.world, '_gen_static_data'):
            self.world._gen_static_data()
        if hasattr(self.world, '_render_static'):
            self.world._render_static()
        
        # =====================================
        # REWARD AND OBSERVATION INITIALIZATION
        # =====================================
        # Reset reward tracking for all agents
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.rewards = {agent: 0.0 for agent in self.agents}
        
        # Reset termination and truncation tracking
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        # Get initial observations for all agents
        self.observations = {agent: self.observe(agent) for agent in self.agents}
        
        # Prepare return values
        observations = {agent: self.observe(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos

    def step(self, actions):
        """
        Execute one environment step with actions from all agents
        
        This method coordinates simultaneous action execution for both agents:
        1. Processes navigator movement and calculates rewards
        2. Handles door controller actions (if permitted)
        3. Manages episode termination conditions
        4. Coordinates reward sharing between agents
        5. Updates environment state and observations
        
        Args:
            actions (dict): Dictionary mapping agent names to their chosen actions
                
        Returns:
            tuple: (observations, rewards, terminations, truncations, infos)
                - observations: Current observations for each agent
                - rewards: Immediate rewards for each agent
                - terminations: Episode termination flags
                - truncations: Episode truncation flags (timeout)
                - infos: Additional information for each agent
        """
        # =====================================
        # STEP COUNTER MANAGEMENT
        # =====================================
        self.steps += 1  # Increment total environment steps
        
        # Identify currently active (non-terminated) agents
        active_agents = [agent for agent in self.agents 
                        if not (hasattr(self, 'terminations') and self.terminations.get(agent, False))]
        
        # Count navigator steps for episode limit checking
        if "navigator" in actions:
            self.navigator_steps += 1
        
        # =====================================
        # EPISODE TIMEOUT HANDLING
        # =====================================
        # Check if maximum episode steps reached
        max_steps = self.config.get("max_episode_steps", 500)
        if self.navigator_steps >= max_steps:
            # Get final observations before episode ends
            observations = {agent: self.observe(agent) for agent in active_agents}
            
            # ===== TIMEOUT PUNISHMENT CALCULATION =====
            # Calculate timeout punishment based on configuration
            punishment_terminal = -100  # Base punishment for timeout
            nav_reward_scales = self.config.get("navigator_reward_scales", {})
            punishment_terminal_scale = nav_reward_scales.get("punishment_terminal_scale", 0.0)
            
            # Apply scaling to timeout punishment
            timeout_reward = punishment_terminal * punishment_terminal_scale
            
            # ===== REWARD DISTRIBUTION =====
            # Apply timeout punishment to navigator, no reward for door controller during timeout
            rewards = {}
            for agent in active_agents:
                rewards[agent] = timeout_reward if agent == "navigator" else 0.0
            
            # ===== DOOR CONTROLLER END-OF-EPISODE HANDLING =====
            # Record failed episode and calculate final rewards for door controller
            if "door_controller" in active_agents:
                self.door_controller.record_episode_result(False)  # Timeout = failure
                # Get end-of-episode rewards for timeout case
                end_rewards = self.door_controller.give_end_of_episode_rewards(self.world)
                rewards["door_controller"] = end_rewards
            
            # ===== TERMINATION FLAGS =====
            # Set truncation (timeout) for all agents, not termination (success)
            terminations = {agent: False for agent in active_agents}
            truncations = {agent: True for agent in active_agents}
            infos = {agent: {"truncated_by_max_steps": True, "navigator_steps": self.navigator_steps} for agent in active_agents}
            
            # Add door controller specific timeout information
            if "door_controller" in active_agents:
                infos["door_controller"].update({
                    "reward_components": self.door_controller.reward_components.copy(),
                    "success_rewards": rewards["door_controller"],
                    "episode_success": False,
                    "env_type": "parallel"
                })
            
            # Set PettingZoo "__all__" flags for environment-wide status
            terminations["__all__"] = False if active_agents else True
            truncations["__all__"] = True
            
            return observations, rewards, terminations, truncations, infos
        
        # =====================================
        # RESPONSE DICTIONARY INITIALIZATION
        # =====================================
        # Initialize response dictionaries for all active agents
        observations = {}
        rewards = {agent: 0.0 for agent in active_agents}
        terminations = {agent: False for agent in active_agents}
        truncations = {agent: False for agent in active_agents}
        infos = {agent: {} for agent in active_agents}
        
        # =====================================
        # NAVIGATOR ACTION PROCESSING
        # =====================================
        # Process navigator action first (movement is primary)
        if "navigator" in actions and "navigator" in active_agents:
            navigator_action = actions["navigator"]
            
            # Execute navigator action and get immediate results
            navigator_obs, navigator_reward, navigator_terminated, navigator_truncated, navigator_info = \
                self.navigator.process_action(self.world, navigator_action)
            
            # Store navigator results
            observations["navigator"] = navigator_obs
            rewards["navigator"] = navigator_reward
            terminations["navigator"] = navigator_terminated
            truncations["navigator"] = navigator_truncated
            infos["navigator"] = navigator_info

            # =====================================
            # DOOR CONTROLLER HALLWAY REWARD COORDINATION
            # =====================================
            # IMPORTANT: Check for hallway transition after navigator moves
            # This allows door controller to receive rewards for enabling navigator progress
            if "door_controller" in active_agents:
                # Check if navigator reached hallway and calculate door controller rewards
                hallway_rewards = self.door_controller.check_hallway_transition(self.world)
                if hallway_rewards > 0:
                    rewards["door_controller"] += hallway_rewards
                    
                    # Add reward breakdown to door controller info
                    if "door_controller" not in infos:
                        infos["door_controller"] = {}
                    infos["door_controller"]["reward_components"] = self.door_controller.reward_components.copy()
                    infos["door_controller"]["hallway_reward"] = hallway_rewards
            
            # =====================================
            # EPISODE TERMINATION HANDLING
            # =====================================
            # Handle episode ending due to navigator reaching goal or other termination
            if navigator_terminated or navigator_truncated or navigator_info.get('terminate_all', False):
                # Determine episode success status
                episode_success = navigator_terminated and not navigator_truncated
                
                # Set success flag in world for other components to access
                if hasattr(self.world, '_episode_success'):
                    self.world._episode_success = episode_success
                
                # ===== DOOR CONTROLLER END-OF-EPISODE PROCESSING =====
                if "door_controller" in active_agents:
                    # Record episode result (success or failure)
                    self.door_controller.record_episode_result(episode_success)
                    
                    # Calculate final rewards based on episode outcome
                    end_rewards = self.door_controller.give_end_of_episode_rewards(self.world)
                    
                    # Add end-of-episode rewards to accumulated door controller rewards
                    rewards["door_controller"] += end_rewards
                    
                    # Provide comprehensive info about door controller performance
                    infos["door_controller"] = {
                        "reward_components": self.door_controller.reward_components.copy(),
                        "success_rewards": end_rewards,
                        "episode_success": episode_success,
                        "env_type": "parallel"
                    }
                    
                    # ===== SUCCESS DATA RECORDING =====
                    # Record successful paths and door positions for analysis
                    if episode_success:
                        if hasattr(self.world, "current_path"):
                            self.world.successful_paths.append(self.world.current_path.copy())
                        if hasattr(self.world, "door_position"):
                            self.world.door_positions.append(self.world.door_position)
                
                # ===== TERMINATE ALL AGENTS =====
                # When navigator terminates, terminate all other agents
                for agent in active_agents:
                    if agent != "navigator":  # Navigator termination already set
                        terminations[agent] = True
                        truncations[agent] = navigator_truncated
                        
                        # Ensure terminated agents have observations (required by PettingZoo)
                        if agent not in observations:
                            # For terminated agents, provide zero observation
                            obs_space = self.observation_space(agent)
                            observations[agent] = np.zeros(obs_space.shape, dtype=obs_space.dtype)
                            if agent not in infos:
                                infos[agent] = {"terminated_by_navigator": True, "env_type": "parallel"}
        
        # =====================================
        # ADDITIONAL DOOR CONTROLLER REWARD CHECK
        # =====================================
        # Check for hallway rewards even if door controller didn't act
        if "door_controller" in active_agents:
            hallway_rewards = self.door_controller.check_hallway_transition(self.world)
            if hallway_rewards > 0:
                rewards["door_controller"] += hallway_rewards
                
                # Add reward information to door controller info
                if "door_controller" not in infos:
                    infos["door_controller"] = {}
                infos["door_controller"]["reward_components"] = self.door_controller.reward_components.copy()
                infos["door_controller"]["hallway_reward"] = hallway_rewards
                
        # =====================================
        # DOOR CONTROLLER ACTION PROCESSING
        # =====================================
        # Process door controller action if navigator hasn't terminated the episode
        if "door_controller" in actions and "door_controller" in active_agents:
            door_action = actions["door_controller"]
            
            # ===== CHECK DOOR CONTROLLER PERMISSIONS =====
            # Door controller can only act on specific episodes based on configuration
            if self.door_can_act and not self.door_acted_this_episode:
                # ===== EXECUTE DOOR CONTROLLER ACTION =====
                door_obs, door_act_reward, door_terminated, door_truncated, door_info = \
                    self.door_controller.process_action(self.world, door_action)
                
                # Store door controller results
                observations["door_controller"] = door_obs
                
                # Add door placement reward to accumulated door controller rewards
                rewards["door_controller"] += door_act_reward
                
                terminations["door_controller"] = door_terminated
                truncations["door_controller"] = door_truncated
                
                # Update door controller info with action details
                infos["door_controller"] = door_info
                infos["door_controller"]["door_acted"] = True
                
                # Mark door controller as having acted this episode
                self.door_acted_this_episode = True
                self.current_door_position = door_info.get("door_position", self.current_door_position)
            else:
                # ===== DOOR CONTROLLER CANNOT ACT =====
                # Door controller skipped this episode due to timing restrictions
                door_obs = self.door_controller.get_observation(self.world)
                
                # Store observation even though no action taken
                observations["door_controller"] = door_obs
                
                # Update info to indicate skipped action
                if "door_controller" not in infos:
                    infos["door_controller"] = {}
                    
                infos["door_controller"]["skipped"] = True
                infos["door_controller"]["reward_components"] = self.door_controller.reward_components.copy()

        # =====================================
        # VISUAL AND DEBUG UPDATES
        # =====================================
        # Update visualization if in debug mode
        if hasattr(self.world, 'debug_mode') and self.world.debug_mode:
            self.world._gen_static_data()
            self.world._render_static()
        
        # =====================================
        # MISSING OBSERVATION HANDLING
        # =====================================
        # Ensure all active agents have observations and info
        for agent in active_agents:
            if agent not in observations:
                observations[agent] = self.observe(agent)
            if agent not in infos:
                infos[agent] = {}
        
        # =====================================
        # COMPREHENSIVE INFO UPDATES
        # =====================================
        # Add door positions to every agent's info for coordination
        for agent in active_agents:
            if agent not in infos:
                infos[agent] = {}
            # Provide current door positions to all agents
            infos[agent]["door_positions"] = self.door_controller.door_positions.copy()
        
        # =====================================
        # AGENT LIST UPDATE
        # =====================================
        # Update active agents list to exclude terminated/truncated agents
        self.agents = [agent for agent in active_agents 
                    if not (terminations.get(agent, False) or truncations.get(agent, False))]
        
        # =====================================
        # PETTINGZOO "__ALL__" FLAGS
        # =====================================
        # Set environment-wide termination/truncation flags for PettingZoo
        all_terminated = all(terminations.get(agent, False) for agent in active_agents) if active_agents else True
        all_truncated = all(truncations.get(agent, False) for agent in active_agents) if active_agents else False
        
        terminations["__all__"] = all_terminated
        truncations["__all__"] = all_truncated
        
        # =====================================
        # DEBUG INFORMATION
        # =====================================
        # Add step counter to info dictionaries for debugging and analysis
        for agent in infos:
            infos[agent]["wrapper_env_step_count"] = self.steps
        
        return observations, rewards, terminations, truncations, infos
    
    def render(self):
        """
        Render the environment for visualization
        
        Delegates rendering to the underlying MiniWorld environment.
        
        Returns:
            Rendered frame (if render_mode is enabled)
        """
        return self.world.render()
    
    def close(self):
        """
        Clean up and close the environment
        
        Properly shuts down the MiniWorld environment and releases resources.
        """
        self.world.close()
    
    def seed(self, seed=None):
        """
        Seed all random number generators for reproducible behavior
        
        Sets seeds for the environment and all agents to ensure reproducible
        training and evaluation across different runs.
        
        Args:
            seed (int, optional): Master seed for the environment
            
        Returns:
            list: List containing the seed used
        """
        # Set environment-level random number generator
        self.np_random = np.random.RandomState(seed)
        
        # Seed the base MiniWorld environment
        if hasattr(self, 'world'):
            self.world.seed(seed)
        
        # Create derived seeds for each agent to ensure independence
        nav_seed = (seed * 12345) % (2**31-1) if seed is not None else None
        door_seed = (seed * 54321) % (2**31-1) if seed is not None else None
        
        # Seed navigator agent
        if hasattr(self, 'navigator'):
            self.navigator.seed(nav_seed)
        
        # Seed door controller agent
        if hasattr(self, 'door_controller'):
            self.door_controller.seed(door_seed)
            
        return [seed]