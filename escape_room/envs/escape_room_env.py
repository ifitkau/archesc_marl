"""
PettingZoo environment for the escape room using AEC API

This module implements the multi-agent environment wrapper using PettingZoo's Agent-Environment-Cycle (AEC) API.
Unlike parallel APIs where all agents act simultaneously, AEC processes agents sequentially in turns.
This implementation provides turn-based coordination between the navigator and door controller agents.

Key Features:
- Sequential agent execution (agent turns)
- Turn-based reward accumulation and state management
- Dynamic door controller activation based on episode frequency
- Comprehensive episode lifecycle management
- Interactive rendering with pause/resume functionality
- Seeded random number generation for reproducible training

AEC vs Parallel API:
- AEC: Agents act one at a time in sequence (turn-based)
- Parallel: All agents act simultaneously each step
- AEC is better for games where turn order matters
- Parallel is better for real-time coordination scenarios
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils import wrappers
from escape_room.envs.miniworld_env import EscapeRoomBaseEnv
from escape_room.envs.navigator_agent import NavigatorAgent
from escape_room.envs.door_agent import DoorControllerAgent
from pyglet.window import key
import time

class EscapeRoomEnv(AECEnv):
    """
    Multi-agent escape room environment using PettingZoo's AEC (Agent-Environment-Cycle) API
    
    This environment coordinates two agents in sequential turns within a shared 3D world:
    - Navigator: Navigates from start position to terminal (acts every turn when selected)
    - Door Controller: Controls door positions strategically (acts periodically based on episode frequency)
    
    The AEC API processes agents sequentially, where each agent observes, acts, and receives
    rewards in turn. This creates a turn-based interaction where the door controller can
    observe navigator behavior before making door adjustments.
    
    Turn Sequence:
    1. Navigator observes environment and takes movement action
    2. Door controller observes navigator state and environment
    3. Door controller takes door positioning action (if permitted)
    4. Cycle repeats until episode termination
    
    Episode Flow:
    - Episodes terminate when navigator reaches goal or time limit exceeded
    - Door controller only acts on specific episodes (based on move frequency)
    - Both agents receive coordinated rewards based on episode outcomes
    """
    
    # PettingZoo metadata for environment capabilities
    metadata = {
        "render_modes": ["human", "rgb_array"], 
        "render_fps": 30, 
        "is_parallelizable": True
    }
    
    def __init__(self, config=None, render_mode=None, view="top"):
        """
        Initialize the AEC multi-agent escape room environment
        
        Sets up the 3D world, creates both agents, configures the agent selection system,
        and initializes all tracking variables for turn-based episode management.
        
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
        # UI AND DEBUG STATE
        # =====================================
        self.paused = False                     # Interactive pause state for debugging
        self.window_created = False             # Track rendering window creation
        
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
        
        # =====================================
        # DOOR POSITION TRACKING
        # =====================================
        # Store current door positions for each room (critical for coordination)
        self.current_door_positions = {
            'roomA': config["door_position"],   # Door position for leftmost room
            'roomB': config["door_position"],   # Door position for middle room
            'roomC': config["door_position"]    # Door position for rightmost room
        }
        
        # Track door controller activity state
        self.door_has_ever_acted = False        # Has door controller ever taken an action?
        
        # =====================================
        # PETTINGZOO AGENT SETUP
        # =====================================
        # Define all possible agents in the environment
        self.possible_agents = ["navigator", "door_controller"]
        
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
        # EPISODE MANAGEMENT
        # =====================================
        self.episode_count = 0                  # Total episodes run (for door move frequency)
        
        # =====================================
        # AEC-SPECIFIC STATE MANAGEMENT
        # =====================================
        # Agent selection system for turn-based execution
        self._agent_selector = None             # PettingZoo agent selector for turn management
        self.agent_selection = None             # Currently selected agent
        
        # AEC state tracking dictionaries (will be initialized in reset)
        self.rewards = None                     # Current step rewards for each agent
        self.terminations = None                # Termination flags for each agent
        self.truncations = None                 # Truncation flags for each agent  
        self.infos = None                       # Info dictionaries for each agent
        self._cumulative_rewards = None         # Accumulated rewards over episode
        self.observations = None                # Current observations for each agent
    
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
    
    def observe(self, agent):
        """
        Get observation for the specified agent
        
        Args:
            agent (str): Agent name ("navigator" or "door_controller")
            
        Returns:
            np.array: Agent's observation of the current environment state
        """
        if agent == "navigator":
            return self.navigator.get_observation(self.world)
        elif agent == "door_controller":
            return self.door_controller.get_observation(self.world)
        else:
            return None
        
    def action_space(self, agent):
        """
        Return the action space for the specified agent
        
        Args:
            agent (str): Agent name
            
        Returns:
            gymnasium.Space: Action space for the specified agent
        """
        return self._action_spaces[agent]
    
    def reset(self, *, seed=None, options=None):
        """
        Reset the environment for a new episode
        
        This method handles complete environment reset including:
        - Episode counter management and agent selector initialization
        - Random number generator seeding for reproducible episodes
        - World state reset and agent initialization
        - Door controller activation logic based on episode frequency
        - AEC-specific state dictionary initialization
        
        Args:
            seed (int, optional): Random seed for reproducible episodes
            options (dict, optional): Additional reset options
            
        Returns:
            tuple: (observations, infos) - Initial observations and info for all agents
        """
        # =====================================
        # EPISODE INITIALIZATION
        # =====================================
        # Reset debug flags
        self._printed_door_skipped = False
        
        # Increment episode counter for door frequency logic
        self.episode_count += 1
        
        # =====================================
        # RANDOM NUMBER GENERATOR SEEDING
        # =====================================
        # Set seed if provided for reproducible behavior
        if seed is not None:
            self.seed(seed)
            # Reset door positions to default when seeding (ensures consistent starting state)
            self.current_door_positions = {
                'roomA': self.config["door_position"],
                'roomB': self.config["door_position"],
                'roomC': self.config["door_position"]
            }
        
        # Initialize step counter for episode limit tracking
        self.steps = 0
        
        # =====================================
        # WORLD AND AGENT RESET
        # =====================================
        # Synchronize episode count with world
        self.world.episode_count = self.episode_count
        
        # Reset world state
        self.world.portals = []                 # Clear all existing door connections
        self.world.reset(seed=seed)             # Reset world geometry and agent placement
        self.world.step_count = 0               # Reset world's internal step counter
        
        # Reset both agents to initial state
        self.navigator.reset(self.world)
        self.door_controller.reset(self.world)
        
        # =====================================
        # DOOR CONTROLLER ACTIVATION LOGIC
        # =====================================
        # Get configuration parameters for door controller behavior
        door_move_frequency = self.config.get("door_move_frequency", 6)
        
        # Determine if door controller can act in this episode
        # Door acts every N episodes based on move frequency
        self.door_can_act = (self.episode_count % door_move_frequency == 0)
        self.world.door_can_act = self.door_can_act        # Set world-level permission flag
        self.door_acted_this_episode = False               # Reset action tracking

        # =====================================
        # DOOR POSITION INITIALIZATION
        # =====================================
        # If door cannot act this episode, set doors to current positions
        if not self.door_can_act:
            # Move all doors to their current tracked positions
            for room_name, door_pos in self.current_door_positions.items():
                self.door_controller._move_specific_room_door(self.world, room_name, door_pos)
        
        # =====================================
        # AEC AGENT SELECTOR INITIALIZATION
        # =====================================
        # Initialize agents list and agent selector for turn-based execution
        self.agents = self.possible_agents[:]   # All agents are active initially
        self._agent_selector = agent_selector(self.agents)
        
        # Seed the agent selector if seeding capability exists
        if hasattr(self._agent_selector, 'seed'):
            self._agent_selector.seed(seed)
        
        # Reset agent selector and get first agent
        self.agent_selection = self._agent_selector.reset()
        
        # =====================================
        # AEC STATE DICTIONARIES INITIALIZATION
        # =====================================
        # Initialize all AEC tracking dictionaries
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}    # Accumulated rewards over episode
        self.rewards = {agent: 0.0 for agent in self.agents}                # Current step rewards
        self.terminations = {agent: False for agent in self.agents}         # Episode termination flags
        self.truncations = {agent: False for agent in self.agents}          # Episode truncation flags
        self.infos = {agent: {} for agent in self.agents}                   # Additional info per agent
        self.observations = {agent: self.observe(agent) for agent in self.agents}  # Current observations

        # Prepare return values for PettingZoo interface
        observations = {agent: obs for agent, obs in self.observations.items()}
        infos = {agent: {} for agent in self.agents}

        return observations, infos

    def step(self, action):
        """
        Execute one turn of the AEC cycle with the current agent's action
        
        This method handles the core AEC turn processing:
        1. Validates that the current agent can act (not terminated/truncated)
        2. Executes the agent's chosen action
        3. Calculates immediate and coordinated rewards
        4. Checks for episode termination conditions
        5. Advances to the next agent in the turn sequence
        
        Args:
            action: Action chosen by the currently selected agent
        """
        # =====================================
        # AGENT STATE VALIDATION
        # =====================================
        # Handle terminated/truncated agents first
        current_agent = self.agent_selection
    
        if self.terminations[current_agent] or self.truncations[current_agent]:
            # Agent is already terminated/truncated, execute dead step and return
            self._was_dead_step(action)
            return
        
        agent = self.agent_selection
        
        # =====================================
        # STEP COUNTER MANAGEMENT
        # =====================================
        # Increment step counter when navigator acts (navigator drives episode progress)
        if agent == "navigator":
            self.steps += 1
        
        # =====================================
        # EPISODE TIMEOUT HANDLING
        # =====================================
        # Check if maximum episode steps reached
        if self.steps >= self.config["max_episode_steps"]:
            # Set truncation for all agents (timeout condition)
            for a in self.agents:
                self.truncations[a] = True

            # Update final observations for all agents
            for a in self.agents:
                self.observations[a] = self.observe(a)
                
            # Return immediately to prevent further processing
            return

        # =====================================
        # NAVIGATOR ACTION PROCESSING
        # =====================================
        if agent == "navigator":
            # Execute navigator action and get immediate results
            obs, rew, term, trunc, info = self.navigator.process_action(self.world, action)
            
            # ===== EPISODE TERMINATION HANDLING =====
            # Check if episode should end due to navigator reaching goal
            if term or info.get('terminate_all', False):
                # End the episode for all agents
                for a in self.agents:
                    self.terminations[a] = True
                    info['is_successful'] = True  # Mark as successful completion
                
                # ===== DOOR CONTROLLER END-OF-EPISODE REWARDS =====
                # For successful navigation, give door controller final rewards
                if term and not trunc:
                    self.door_controller.record_episode_result(True)  # Record success
                    
                    # Calculate and apply end-of-episode rewards for door controller
                    end_rewards = self.door_controller.give_end_of_episode_rewards(self.world)
                    self.rewards["door_controller"] += end_rewards
                    
                    # Add detailed reward information to info
                    info['reward_components'] = self.door_controller.reward_components.copy()
                    info['door_positions'] = self.current_door_positions
                    
                    # ===== SUCCESS DATA RECORDING =====
                    # Store successful path and door positions for analysis
                    if hasattr(self.world, "current_path"):
                        self.world.successful_paths.append(self.world.current_path.copy())
                    if hasattr(self.world, "door_position"):
                        self.world.door_positions.append(self.world.door_position)
                        
        # =====================================
        # DOOR CONTROLLER ACTION PROCESSING
        # =====================================
        elif agent == "door_controller":
            # ===== CHECK NAVIGATOR STATUS =====
            # If navigator is done, door controller is also done
            if self.terminations.get("navigator", False) or self.truncations.get("navigator", False):
                # Navigator finished, terminate door controller
                self.terminations[agent] = True
                obs = self.door_controller.get_observation(self.world)
                
                # Calculate any remaining rewards (hallway transition, etc.)
                rew = self.door_controller.check_hallway_transition(self.world)
                self.rewards[agent] += rew
                
                term = True
                trunc = False
                info = {
                    "skipped": True, 
                    "reward_components": self.door_controller.reward_components.copy()
                }
            
            # ===== DOOR CONTROLLER CAN ACT =====
            elif self.door_can_act and not self.door_acted_this_episode:
                # Door controller is permitted to act and hasn't acted yet this episode
                obs, rew, term, trunc, info = self.door_controller.process_action(self.world, action)
                self.rewards[agent] += rew
                
                # Update tracked door positions from door controller's new positions
                self.current_door_positions = self.door_controller.door_positions.copy()
                
                # Mark door controller as having acted this episode
                self.door_acted_this_episode = True
                info["door_acted"] = True
                info['door_positions'] = self.current_door_positions

            # ===== DOOR CONTROLLER CANNOT ACT =====
            else:
                # Door controller skipped this turn (frequency restrictions or already acted)
                obs = self.door_controller.get_observation(self.world)
                
                # Calculate any passive rewards (hallway transition monitoring)
                rew = self.door_controller.check_hallway_transition(self.world)
                self.rewards[agent] += rew  # Ensure reward is applied
                
                term = False
                trunc = False
                info = {
                    "skipped": True, 
                    "reward_components": self.door_controller.reward_components.copy()
                }

        # =====================================
        # VISUAL UPDATE AND DEBUGGING
        # =====================================
        # Update visualization if debug mode is enabled
        if not hasattr(self.world, 'debug_mode') or self.world.debug_mode is True:
            self.world._gen_static_data()       # Regenerate static visual data
            self.world._render_static()         # Update rendering
        
        # =====================================
        # AEC STATE UPDATE
        # =====================================
        # Update all AEC tracking dictionaries with current agent's results
        self.observations[current_agent] = obs     # Store new observation
        self.rewards[current_agent] = rew          # Store immediate reward
        self.terminations[current_agent] = term    # Store termination status
        self.truncations[current_agent] = trunc    # Store truncation status
        self.infos[current_agent] = info           # Store additional info

        # Accumulate rewards for the current agent
        self._cumulative_rewards[current_agent] += rew
        
        # =====================================
        # AGENT SELECTION ADVANCEMENT
        # =====================================
        # Advance to next agent in turn sequence if current agent is still active
        if not (self.terminations[agent] or self.truncations[agent]):
            self.agent_selection = self._agent_selector.next()
    
    def observe(self, agent):
        """
        Get the observation for an agent
        
        Args:
            agent (str): Agent name ("navigator" or "door_controller")
            
        Returns:
            np.array: Observation for the specified agent
        """
        if agent == "navigator":
            return self.navigator.get_observation(self.world)
        elif agent == "door_controller":
            return self.door_controller.get_observation(self.world)
        return None
    
    def render(self):
        """
        Render the environment with interactive controls
        
        Provides visual output for debugging and human observation, including
        interactive pause/resume functionality via spacebar key press.
        
        Returns:
            Rendered frame if render_mode is enabled
        """
        # =====================================
        # INITIAL RENDERING
        # =====================================
        # First render to ensure window is created
        result = self.world.render()
        
        # =====================================
        # INTERACTIVE CONTROLS SETUP
        # =====================================
        # Set up keyboard controls once window is created
        if not self.window_created and self.render_mode == "human" and \
           hasattr(self.world, 'window') and self.world.window is not None:
            
            from pyglet.window import key
            
            @self.world.window.event
            def on_key_press(symbol, modifiers):
                """Handle keyboard input for interactive controls"""
                if symbol == key.SPACE:
                    # Toggle pause/resume with spacebar
                    self.world.paused = not self.world.paused
            
            self.window_created = True
        
        # =====================================
        # WINDOW UPDATE
        # =====================================
        # Force window update for interactive responsiveness
        if hasattr(self.world, 'window') and self.world.window is not None:
            self.world.window.switch_to()       # Make window active
            self.world.window.dispatch_events() # Process input events
        
        return result
    
    def close(self):
        """
        Clean up and close the environment
        
        Properly shuts down the MiniWorld environment and releases resources.
        """
        self.world.close()

    def observation_space(self, agent):
        """
        Return the observation space for the specified agent
        
        Args:
            agent (str): Agent name ("navigator" or "door_controller")
            
        Returns:
            gymnasium.Space: Observation space for the specified agent
        """
        if agent == "navigator":
            return self.navigator.observation_space
        elif agent == "door_controller":
            return self.door_controller.observation_space
        return None

    def action_space(self, agent):
        """
        Return the action space for the specified agent
        
        Args:
            agent (str): Agent name ("navigator" or "door_controller")
            
        Returns:
            gymnasium.Space: Action space for the specified agent
        """
        if agent == "navigator":
            return self.navigator.action_space
        elif agent == "door_controller":
            return self.door_controller.action_space
        return None
    
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
        # Generate seed from current time if none provided
        if seed is None:
            seed = int(time.time()) % (2**31-1)
            
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


def make_escape_room_env(config=None, render_mode=None, view="top"):
    """
    Factory function to create an escape room environment
    
    This function serves as the main entry point for creating escape room environments.
    It automatically selects between AEC and Parallel implementations based on
    configuration and provides appropriate wrappers for different use cases.
    
    Args:
        config (dict, optional): Configuration dictionary with environment parameters
        render_mode (str, optional): Rendering mode ("human", "rgb_array", or None)
        view (str, optional): Camera view perspective ("top", "agent", etc.)
        
    Returns:
        Environment instance: Wrapped environment ready for training or evaluation
    """
    # =====================================
    # CONFIGURATION SETUP
    # =====================================
    # Load default configuration if none provided
    if config is None:
        from escape_room.config.default_config import get_default_config
        config = get_default_config()
    
    # Enable debug mode for development and testing
    config['debug_mode'] = True
    
    # =====================================
    # ENVIRONMENT TYPE SELECTION
    # =====================================
    # Determine whether to use parallel or AEC implementation
    use_parallel = config.get("parallel_env", False)
    
    print(f"Creating {'Parallel' if use_parallel else 'AEC'} environment with max_steps={config['max_episode_steps']}")
    
    # =====================================
    # ENVIRONMENT CREATION
    # =====================================
    # Create the appropriate wrapper - it will determine the correct environment type internally
    from escape_room.utils.gymnasium_wrapper import EscapeRoomGymWrapper
    return EscapeRoomGymWrapper(config=config)