import gymnasium as gym
import numpy as np
from typing import Dict, Any, Tuple, Optional
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import logging
import random
import time

# Set up logging for debugging and monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EscapeRoomWrapper")

class EscapeRoomGymWrapper(MultiAgentEnv):
    """
    Gymnasium wrapper for the EscapeRoomEnv to ensure RLlib compatibility
    
    This wrapper bridges the gap between custom PettingZoo-style escape room environments
    and Ray RLLib's multi-agent training framework. It handles both AEC (Agent Environment Cycle)
    and Parallel environment types with correct step counting logic.
    
    Key Features:
    - Supports both AEC and Parallel multi-agent environments
    - Optimized performance with reused data structures and reduced copying
    - Proper step counting for different environment types
    - Robust error handling and debugging capabilities
    - Compatible with Ray RLLib's MultiAgentEnv interface
    - Handles agent termination and truncation correctly
    
    Optimizations included:
    1. Reused data structures to minimize memory allocation
    2. Reduced object copying for better performance
    3. Optimized seeding strategy for reproducibility
    4. Pre-allocated observations for terminated agents
    """
    
    def __init__(self, config=None):
        """
        Initialize the Gymnasium-compatible environment wrapper
        
        This constructor sets up the wrapper to work with either AEC or Parallel
        versions of the escape room environment, configures action/observation spaces,
        and initializes optimization structures.
        
        Args:
            config (dict): Configuration dictionary containing:
                - parallel_env (bool): Whether to use Parallel or AEC environment
                - debug_mode (bool): Enable debug logging
                - Other environment-specific parameters
        """
        # Enable debug mode based on configuration
        self.debug = True if config and config.get('debug_mode') else False
        
        # Determine whether to use Parallel or AEC environment
        # Parallel: All agents act simultaneously each step
        # AEC: Agents take turns acting (Agent Environment Cycle)
        self.is_parallel = config.get("parallel_env", False)
        
        # Create the appropriate environment type based on configuration
        if self.is_parallel:
            from escape_room.envs.escape_room_env_parallel import ParallelEscapeRoomEnv
            self.env = ParallelEscapeRoomEnv(config=config, render_mode=None, view="top")
        else:
            from escape_room.envs.escape_room_env import EscapeRoomEnv
            self.env = EscapeRoomEnv(config=config, render_mode=None, view="top")
        
        # Define action and observation spaces for all agents
        # These are required by RLLib to understand what actions agents can take
        # and what observations they receive
        self.action_spaces = {
            agent: self.env.action_space(agent) 
            for agent in self.env.possible_agents
        }

        self.observation_spaces = {
            agent: self.env.observation_space(agent) 
            for agent in self.env.possible_agents
        }

        # Required for Ray RLlib compatibility - list of all possible agents
        self.possible_agents = self.env.possible_agents
        
        # Initialize agents as an empty list - will be populated during reset
        # This tracks which agents are currently active (not terminated)
        self.agents = []
        
        # Track episode done state to prevent unnecessary step calls
        self._episode_done = False
        
        # Episode counters for debugging and monitoring
        self.episode_count = 0
        
        # Environment step count (counts raw environment steps, not agent steps)
        # Important: This is different from agent step count in AEC environments
        self.env_step_count = 0
        
        # OPTIMIZATION 1: Pre-allocate dictionaries for step returns
        # This avoids creating new dictionaries every step, improving performance
        self._observations = {agent: None for agent in self.possible_agents}
        self._rewards = {agent: 0.0 for agent in self.possible_agents}
        self._terminations = {agent: False for agent in self.possible_agents}
        self._terminations['__all__'] = False  # Special key for RLLib
        self._truncations = {agent: False for agent in self.possible_agents}
        self._truncations['__all__'] = False   # Special key for RLLib
        self._infos = {agent: {} for agent in self.possible_agents}
        
        # Pre-allocate zero observations for terminated agents
        # When agents terminate, they need observations but shouldn't affect training
        self._zero_observations = {}
        for agent in self.possible_agents:
            shape = self.observation_space(agent).shape
            self._zero_observations[agent] = np.zeros(shape, dtype=self.observation_space(agent).dtype)
        
        # Call parent MultiAgentEnv constructor for RLLib compatibility
        super().__init__()
    
    def get_observation_space(self, agent):
        """
        Get observation space for a specific agent
        
        Args:
            agent (str): Agent identifier
            
        Returns:
            gym.Space: Observation space for the agent
        """
        return self.observation_spaces[agent]
    
    def get_action_space(self, agent):
        """
        Get action space for a specific agent
        
        Args:
            agent (str): Agent identifier
            
        Returns:
            gym.Space: Action space for the agent
        """
        return self.action_spaces[agent]
    
    def reset(
        self, 
        *, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset the environment to start a new episode
        
        This method resets the underlying environment, handles seeding for reproducibility,
        and initializes all tracking variables for the new episode.
        
        Args:
            seed (int, optional): Random seed for reproducibility. If None, generates random seed
            options (dict, optional): Additional options for reset (environment-specific)
        
        Returns:
            tuple: (observations, infos)
                - observations (dict): Initial observations for each agent
                - infos (dict): Additional information for each agent
        """
        # Increment episode counter for debugging and monitoring
        self.episode_count += 1
        self.env_step_count = 0
        
        # OPTIMIZATION 3: Simplified seeding strategy
        # Generate a random seed if none provided to ensure variety
        if seed is None:
            seed = int(time.time()) % (2**31-1)  # Use current time as seed source
        
        # Seed the environment once and let it handle component seeding
        # This ensures all random number generators are properly seeded
        if hasattr(self.env, 'seed'):
            self.env.seed(seed)
        
        # Reset environment with the seed
        observations = self.env.reset(seed=seed, options=options)
        
        # Handle different return types between AEC and Parallel environments
        # Some environments return (observations, infos), others just observations
        infos = {}
        if isinstance(observations, tuple):  # AEC returns (observations, infos)
            observations, infos = observations
        
        # OPTIMIZATION 2: Use direct reference instead of copying
        # Track which agents are currently active (have observations)
        self.agents = list(observations.keys())
        
        # Reset episode done state for new episode
        self._episode_done = False
        
        # Reset pre-allocated dictionaries to clean state
        for agent in self.possible_agents:
            self._rewards[agent] = 0.0
            self._terminations[agent] = False
            self._truncations[agent] = False
            self._infos[agent] = {}
        
        # Reset special RLLib keys
        self._terminations['__all__'] = False
        self._truncations['__all__'] = False
        
        # Get agent to start with (only for AEC environments)
        # AEC environments have a specific agent that acts each step
        if not self.is_parallel and hasattr(self.env, '_agent_selector'):
            self._agent_selector = self.env._agent_selector
            self.agent_selection = self.env.agent_selection
        
        return observations, infos
    
    def step(self, action: Dict[str, Any]):
        """
        Step the environment using the given action
        
        This is the core method that executes actions and returns the results.
        It handles both Parallel and AEC environments differently:
        - Parallel: All agents act simultaneously
        - AEC: One agent acts per step, cycling through agents
        
        Args:
            action (dict): Dictionary mapping agent_id to action
                          Format: {agent_id: action_value, ...}
            
        Returns:
            tuple: (observations, rewards, terminations, truncations, infos)
                - observations (dict): New observations for each agent
                - rewards (dict): Rewards received by each agent
                - terminations (dict): Whether each agent terminated (reached goal)
                - truncations (dict): Whether each agent was truncated (hit time limit)
                - infos (dict): Additional information for each agent
        """
        # Increment environment step counter for tracking
        self.env_step_count += 1
        
        # If episode is already done, return final result without processing
        if self._episode_done:
            return self._get_final_step_result()
        
        # OPTIMIZATION 2: Use direct reference instead of copying
        # Get list of currently active agents (not terminated)
        active_agents = self.agents  # No copy needed for performance
        
        # Filter actions to only include active agents
        # This prevents sending actions for terminated agents
        active_actions = {}
        for agent in action:
            if agent in active_agents:
                active_actions[agent] = action[agent]
        
        # Clear previous step results to avoid stale data
        for agent in self.possible_agents:
            self._rewards[agent] = 0.0
            self._terminations[agent] = False
            self._truncations[agent] = False
            self._infos[agent] = {}
        
        # Different handling for Parallel vs AEC environments
        if self.is_parallel:
            # PARALLEL ENVIRONMENT HANDLING
            # All agents act simultaneously, environment returns results for all
            
            # For Parallel environment, pass only actions for active agents
            observations, rewards, terminations, truncations, infos = self.env.step(active_actions)
            
            # Update episode done tracking
            self._episode_done = terminations.get('__all__', False) or truncations.get('__all__', False)
            
            # OPTIMIZATION 1: Reuse data structures instead of creating new ones
            for agent in observations:
                self._observations[agent] = observations[agent]
                self._rewards[agent] = rewards.get(agent, 0.0)
                self._terminations[agent] = terminations.get(agent, False)
                self._truncations[agent] = truncations.get(agent, False)
                self._infos[agent] = infos.get(agent, {})
            
            # Copy special __all__ keys required by RLLib
            self._terminations['__all__'] = terminations.get('__all__', False)
            self._truncations['__all__'] = truncations.get('__all__', False)
            
            # Update active agents list - critical for preventing terminated agent actions
            # Remove agents that have terminated or been truncated
            self.agents = [
                a for a in self.possible_agents 
                if a in observations and not (terminations.get(a, False) or truncations.get(a, False))
            ]
            
        else:  
            # AEC ENVIRONMENT HANDLING
            # Agents take turns acting, only one agent acts per environment step
            
            # Check if current agent is active before sending action
            current_agent = self.env.agent_selection if hasattr(self.env, 'agent_selection') else None
            
            if current_agent in active_agents and current_agent in active_actions:
                # Execute step with action for current agent
                self.env.step(active_actions[current_agent])
            elif current_agent in active_agents:
                # No action for current agent but it's still active, use default action
                self.env.step(0)  # Default action (usually "do nothing")
            
            # Get list of agents that were active before this step
            pre_step_agents = active_agents
            
            # Process results for all agents that were active
            for agent in pre_step_agents:
                # Get termination status from environment
                agent_terminated = self.env.terminations.get(agent, False) if hasattr(self.env, 'terminations') else False
                agent_truncated = self.env.truncations.get(agent, False) if hasattr(self.env, 'truncations') else False
                
                self._terminations[agent] = agent_terminated
                self._truncations[agent] = agent_truncated
                
                # Only include reward and info for agents that were active
                if hasattr(self.env, 'rewards') and agent in self.env.rewards:
                    self._rewards[agent] = self.env.rewards[agent]
                
                if hasattr(self.env, 'infos') and agent in self.env.infos:
                    self._infos[agent] = self.env.infos[agent]
                
                # Get observation only for non-terminated agents
                if not (agent_terminated or agent_truncated):
                    try:
                        self._observations[agent] = self.env.observe(agent)
                    except Exception as e:
                        logger.error(f"Error observing agent {agent}: {e}")
                        # Set as terminated due to error to prevent infinite loops
                        self._terminations[agent] = True
                        self._observations[agent] = self._zero_observations[agent]
                        self._infos[agent]["error"] = str(e)
                else:
                    # For terminated agents, use pre-allocated zero observation
                    # This ensures observations have correct shape but don't affect training
                    self._observations[agent] = self._zero_observations[agent]
            
            # Update active agents list for next step
            # Remove agents that terminated or were truncated this step
            self.agents = [
                a for a in pre_step_agents 
                if not (self._terminations.get(a, False) or self._truncations.get(a, False))
            ]
            
            # Add __all__ keys for RLlib compatibility
            # Episode is done when all agents are terminated or truncated
            has_active_agents = len(pre_step_agents) > 0
            self._terminations['__all__'] = all(self._terminations.get(a, False) for a in pre_step_agents) if has_active_agents else True
            self._truncations['__all__'] = all(self._truncations.get(a, False) for a in pre_step_agents) if has_active_agents else False
            
            # Update episode done state
            self._episode_done = self._terminations['__all__'] or self._truncations['__all__']
        
        # Add environment step counter to info dictionaries for debugging
        # This helps track performance and step counting across different environment types
        for agent in self._infos:
            if agent != '__all__':  # Skip the special RLLib key
                self._infos[agent]["wrapper_env_step_count"] = self.env_step_count
                self._infos[agent]["env_type"] = "parallel" if self.is_parallel else "aec"
        
        return self._observations, self._rewards, self._terminations, self._truncations, self._infos

    def _get_final_step_result(self):
        """
        Return a final step result when the episode is done
        
        This method is called when the episode has already ended but step() is called again.
        It returns empty observations/rewards but maintains proper termination flags
        to signal RLLib that the episode is complete.
        
        Returns:
            tuple: (empty_observations, empty_rewards, terminations, truncations, empty_infos)
        """
        # When episode is done, return empty observations/rewards but keep __all__ flags
        empty_observations = {}
        empty_rewards = {}
        
        # Reuse termination dictionaries, reset individual agent flags
        for agent in self.possible_agents:
            self._terminations[agent] = False
            self._truncations[agent] = False
        
        # Set special flags to indicate episode completion
        self._terminations['__all__'] = True   # Episode is terminated
        self._truncations['__all__'] = False   # Not truncated (clean termination)
        
        empty_infos = {}
        
        return empty_observations, empty_rewards, self._terminations, self._truncations, empty_infos
    
    def render(self):
        """
        Render the environment
        
        Delegates rendering to the underlying environment. Useful for visualization
        during debugging or demonstration.
        
        Returns:
            Rendered output from the underlying environment
        """
        return self.env.render()
    
    def close(self):
        """
        Close the environment and clean up resources
        
        Should be called when done with the environment to properly clean up
        any resources (graphics, file handles, etc.)
        """
        self.env.close()
    
    def observation_space(self, agent):
        """
        Returns the observation space for the specified agent
        
        Args:
            agent (str): Agent identifier
            
        Returns:
            gym.Space: Observation space defining the format and bounds of observations
        """
        return self.observation_spaces.get(agent)
    
    def action_space(self, agent):
        """
        Returns the action space for the specified agent
        
        Args:
            agent (str): Agent identifier
            
        Returns:
            gym.Space: Action space defining valid actions the agent can take
        """
        return self.action_spaces.get(agent)
    
    def observe(self, agent):
        """
        Get observation for the specified agent
        
        This method allows direct observation of a specific agent's state,
        useful for debugging or manual agent control.
        
        Args:
            agent (str): Agent identifier
            
        Returns:
            np.ndarray or None: Current observation for the agent, or None if invalid agent
        """
        if agent in self.env.possible_agents:
            return self.env.observe(agent)
        return None


def env_creator(config=None):
    """
    Environment creator function for Ray RLlib
    
    This function is used by Ray RLLib to create environment instances.
    It serves as a factory function that can be registered with RLLib
    and called to create new environment instances for training.
    
    Args:
        config (dict, optional): Configuration dictionary containing environment parameters.
                               If None, creates empty config with debug mode enabled.
    
    Returns:
        EscapeRoomGymWrapper: Configured Gymnasium-compatible environment instance
        
    Usage:
        # Register with RLLib
        tune.register_env("escape_room", env_creator)
        
        # Use in RLLib training
        config = {
            "env": "escape_room",
            "env_config": {"parallel_env": True, "debug_mode": True}
        }
    """
    # Enable debug mode in config by default for better monitoring
    if config is None:
        config = {}
    config['debug_mode'] = True
    
    # Create and return the wrapper
    return EscapeRoomGymWrapper(config=config)