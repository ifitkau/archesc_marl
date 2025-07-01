#!/usr/bin/env python3
"""
Improved Multi-Agent RL Visualization Script

This script creates comprehensive visualizations for multi-agent reinforcement learning
training data, specifically designed to handle the full range of returns including
negative values commonly found in RL environments.

Key Features:
- Dual visualization modes: single-axis and dual-axis comparisons
- Moving average trend analysis for noise reduction
- Statistical analysis and correlation computation
- Handles negative returns and varying scales between agents
- Publication-quality output with colorblind-friendly palettes
- Flexible data loading from JSON format
- Command-line interface for batch processing

Designed for Multi-Agent Scenarios:
- Navigator-Door Controller environments
- Cooperative/competitive agent interactions
- Agents with different reward scales and ranges
- Long training runs with noisy reward signals

Usage Examples:
  # Basic comparison with default settings
  python vis_rltrainingmultirun.py --navigator-reward nav_data.json --door-reward door_data.json
  
  # Custom moving average and dual-axis only
  python vis_rltrainingmultirun.py --navigator-reward nav.json --door-reward door.json 
                                   --window-size 50 --plot-type dual
  
  # Publication-ready output with custom directory
  python vis_rltrainingmultirun.py --navigator-reward data/nav.json --door-reward data/door.json 
                                   --output-dir figures/ --window-size 20

Date: July 2025
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from datetime import datetime

# ============================================================================
# OPTIONAL AESTHETIC ENHANCEMENT IMPORTS
# ============================================================================

# Try to import seaborn for publication-quality aesthetics
# This is optional - the script works without it but looks better with it
try:
    import seaborn as sns
    # Configure seaborn for clean, professional-looking plots
    sns.set_theme(style="whitegrid")  # Clean background with subtle grid
    sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})  # Publication sizing
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Note: Install seaborn for better plot aesthetics: pip install seaborn")

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# Moving average window sizes for trend analysis
# Multiple windows can be specified for different levels of smoothing
MOVING_AVERAGE_WINDOWS = [10]  # Default: light smoothing for trend visibility

# Output directory for generated visualizations
OUTPUT_DIR = "multi_agent_figures"

# Image resolution for publication-quality output
DPI = 300

# Agent-specific color palette (colorblind-friendly)
# Based on scientific visualization best practices
COLORBLIND_PALETTE = {
    "navigator": "#0072B2",        # Blue - associated with navigation/movement
    "door_controller": "#D55E00",  # Red/Orange - associated with control/action
}

# ============================================================================
# AGENT DATA MANAGEMENT CLASS
# ============================================================================

class AgentData:
    """
    Class to store and process training data from a single RL agent
    
    This class handles the complete data pipeline for a single agent:
    - Loading training data from various file formats
    - Computing statistical metrics (mean, std, min, max, etc.)
    - Calculating moving averages for trend analysis
    - Preparing data for visualization
    
    Designed specifically for RL training data where:
    - Returns can be positive or negative
    - Data may be noisy and require smoothing
    - Statistical analysis is crucial for performance assessment
    
    Attributes:
        file_path (str): Path to the source data file
        agent_name (str): Technical identifier for the agent
        label (str): Human-readable display name
        color (str): Matplotlib color for consistent visualization
        iterations (numpy.ndarray): Training iteration/episode numbers
        values (numpy.ndarray): Return/reward values
        mean, median, min, max, std (float): Basic statistical metrics
        moving_averages (dict): Computed moving averages for different window sizes
    """
    
    def __init__(self, file_path, agent_name):
        """
        Initialize AgentData with data loading and preprocessing
        
        Args:
            file_path (str): Path to JSON file containing training data
            agent_name (str): Agent identifier (e.g., "navigator", "door_controller")
        """
        self.file_path = file_path
        self.agent_name = agent_name
        
        # Create human-readable label from technical name
        # Converts "door_controller" -> "Door Controller"
        self.label = agent_name.replace('_', ' ').title()
        
        # Assign agent-specific color or default to black
        self.color = COLORBLIND_PALETTE.get(agent_name, "#000000")
        
        # ====================================================================
        # DATA LOADING AND BASIC PROCESSING
        # ====================================================================
        
        # Load raw training data from file
        self.iterations, self.values = self.load_data()
        
        # Calculate fundamental statistical metrics for performance assessment
        self.mean = np.mean(self.values)      # Average performance
        self.median = np.median(self.values)  # Robust central tendency
        self.min = np.min(self.values)        # Worst performance
        self.max = np.max(self.values)        # Best performance
        self.std = np.std(self.values)        # Performance variability
        
        # ====================================================================
        # MOVING AVERAGE COMPUTATION
        # ====================================================================
        
        # Calculate moving averages for trend analysis
        # Moving averages help identify learning trends by reducing noise
        self.moving_averages = {}
        for window in MOVING_AVERAGE_WINDOWS:
            # Only calculate if we have enough data points
            if len(self.values) >= window:
                ma = self.moving_average(window)
                self.moving_averages[window] = ma
    
    def load_data(self):
        """
        Load training data from file with format auto-detection
        
        Currently supports JSON format with the structure:
        [[timestamp, iteration, return_value], [timestamp, iteration, return_value], ...]
        
        This format is common in RL logging systems where each entry represents
        one training episode or evaluation point.
        
        Returns:
            tuple: (iterations, values) as numpy arrays
            
        Raises:
            ValueError: If file format is unsupported or data structure is invalid
            
        Note:
            The function extracts columns 1 (iterations) and 2 (values) from the
            nested list structure, skipping column 0 (timestamps) which are not
            needed for visualization.
        """
        # Determine file format from extension
        file_ext = os.path.splitext(self.file_path)[1].lower()
        
        # ====================================================================
        # JSON FORMAT HANDLING
        # ====================================================================
        if file_ext == '.json':
            with open(self.file_path, 'r') as f:
                data = json.load(f)
            
            # Handle nested list format: [[timestamp, iteration, value], ...]
            if isinstance(data, list):
                iterations = []
                values = []
                
                # Extract iteration and value columns from each entry
                for entry in data:
                    if isinstance(entry, list) and len(entry) >= 3:
                        # entry[0] = timestamp (ignored for visualization)
                        # entry[1] = iteration/episode number
                        # entry[2] = return/reward value
                        iterations.append(entry[1])
                        values.append(entry[2])
                
                return np.array(iterations), np.array(values)
        
        # If no supported format was found, provide helpful error message
        raise ValueError(f"Could not parse data from {self.file_path}. "
                        f"Supported format: JSON with [[timestamp, iteration, value]] structure")
    
    def moving_average(self, window_size):
        """
        Calculate moving average for trend analysis and noise reduction
        
        Moving averages are crucial in RL visualization because:
        - Raw returns are often noisy due to exploration and environment stochasticity
        - Trends are more visible when noise is reduced
        - Learning progress assessment requires smooth trend lines
        
        Args:
            window_size (int): Number of consecutive points to average
                              Larger windows = smoother curves, less responsiveness
                              Smaller windows = more responsive, more noise
        
        Returns:
            numpy.ndarray: Moving average values (shorter than input due to windowing)
                          Returns empty array if insufficient data points
        
        Note:
            Uses simple moving average (SMA) with equal weights for all points
            in the window. The output array is shorter than the input by
            (window_size - 1) points due to the windowing operation.
        """
        # Check if we have enough data points for the specified window
        if len(self.values) < window_size:
            return np.array([])
        
        # Calculate simple moving average using convolution
        # This is efficient and handles edge cases well
        ma = np.convolve(self.values, np.ones(window_size)/window_size, mode='valid')
        return ma
    
    def print_statistics(self):
        """
        Print comprehensive statistics for this agent's performance
        
        Provides a detailed statistical summary useful for:
        - Quick performance assessment
        - Identifying potential issues (e.g., extremely high variance)
        - Comparing agents quantitatively
        - Documenting experimental results
        
        Output includes:
        - Data quality metrics (number of points, iteration range)
        - Central tendency measures (mean, median)
        - Variability measures (standard deviation, min/max range)
        """
        print(f"\n===== {self.label} Return Statistics =====")
        print(f"Data points: {len(self.values)}")
        print(f"Overall mean: {self.mean:.2f}")
        print(f"Overall median: {self.median:.2f}")
        print(f"Overall min: {self.min:.2f}")
        print(f"Overall max: {self.max:.2f}")
        print(f"Overall std dev: {self.std:.2f}")
        print(f"Iteration range: {self.iterations[0]:.0f} to {self.iterations[-1]:.0f}")

# ============================================================================
# SINGLE-AXIS VISUALIZATION FUNCTION
# ============================================================================

def create_multi_agent_plot(agents, output_path, show_moving_average=True, window_size=10):
    """
    Create a single-axis plot comparing multiple agents' performance
    
    This visualization is ideal when:
    - Agents have similar return scales
    - Direct comparison of performance levels is desired
    - You want to see relative performance clearly
    - Statistical relationships (correlation) are of interest
    
    The plot includes:
    - Raw data with transparency to show variability
    - Moving average trends for clear learning progression
    - Horizontal mean lines for quick performance reference
    - Statistical correlation analysis for two-agent scenarios
    
    Args:
        agents (list): List of AgentData objects to visualize
        output_path (str): File path for saving the plot
        show_moving_average (bool): Whether to plot smoothed trend lines
        window_size (int): Moving average window size
        
    Returns:
        None (saves plot to file and prints statistics)
        
    Example:
        agents = [nav_agent, door_agent]
        create_multi_agent_plot(agents, "comparison.png", window_size=20)
    """
    # ========================================================================
    # FIGURE SETUP AND CONFIGURATION
    # ========================================================================
    
    # Create figure with wide aspect ratio suitable for time series data
    plt.figure(figsize=(16, 8))
    
    # ========================================================================
    # DATA PLOTTING FOR EACH AGENT
    # ========================================================================
    
    # Plot each agent's training data
    for agent in agents:
        # Plot raw data with high transparency to show underlying variability
        # This helps visualize the noise level and data density
        plt.plot(agent.iterations, agent.values, 
                color=agent.color, alpha=0.3, linewidth=1.0, 
                label=f"{agent.label} (raw)")
        
        # Plot moving average trend line if requested and available
        if show_moving_average and window_size in agent.moving_averages:
            ma = agent.moving_averages[window_size]
            # Align moving average with original iterations
            # Moving average starts from iteration (window_size-1) due to windowing
            ma_iterations = agent.iterations[window_size-1:]
            
            plt.plot(ma_iterations, ma, 
                    color=agent.color, linewidth=3.0, 
                    label=f"{agent.label} (MA={window_size})")
        
        # Add horizontal reference line for agent's overall mean performance
        # This provides quick visual reference for average performance level
        plt.axhline(agent.mean, linestyle='--', color=agent.color, 
                   alpha=0.7, linewidth=1.5,
                   label=f"{agent.label} Mean: {agent.mean:.1f}")
    
    # ========================================================================
    # PLOT FORMATTING AND STYLING
    # ========================================================================
    
    # Set axis labels with appropriate font sizes
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Return', fontsize=14)
    plt.title('Comparison of Agent Returns', fontsize=16)
    
    # Add subtle grid for easier data reading
    plt.grid(True, alpha=0.3, linestyle=':')
    
    # Format x-axis with thousands separators for readability
    # Large iteration numbers are common in RL training
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{x:,.0f}"))
    
    # Position legend to avoid obscuring data
    plt.legend(loc='lower right', fontsize=10)
    
    # Optimize layout to prevent label cutoff
    plt.tight_layout()
    
    # ========================================================================
    # FILE OUTPUT AND STATISTICAL ANALYSIS
    # ========================================================================
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save high-quality figure
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"Saved multi-agent plot to {output_path}")
    
    # Calculate and report correlation for two-agent scenarios
    # Correlation helps understand if agents learn complementary or similar behaviors
    if len(agents) == 2:
        # Find overlapping data range for fair correlation calculation
        min_len = min(len(agents[0].values), len(agents[1].values))
        corr = np.corrcoef(agents[0].values[:min_len], agents[1].values[:min_len])[0, 1]
        print(f"Correlation between {agents[0].label} and {agents[1].label}: {corr:.4f}")
    
    # Free memory by closing the figure
    plt.close()

# ============================================================================
# DUAL-AXIS VISUALIZATION FUNCTION
# ============================================================================

def create_dual_axis_plot(agents, output_path, show_moving_average=True, window_size=10):
    """
    Create a dual-axis plot for agents with significantly different return scales
    
    This visualization is essential when:
    - Agents have vastly different return ranges (e.g., one agent: 0-100, another: 0-10000)
    - Both agents' trends need to be visible simultaneously
    - Scale differences would make single-axis plots unreadable
    - Independent axis scaling is required for meaningful comparison
    
    Features:
    - Left y-axis for first agent (typically navigator)
    - Right y-axis for second agent (typically door controller)
    - Color-coded axes matching agent colors
    - Combined legend for both datasets
    - Moving average trends for both agents
    
    Args:
        agents (list): List of exactly 2 AgentData objects
        output_path (str): File path for saving the plot
        show_moving_average (bool): Whether to plot smoothed trend lines
        window_size (int): Moving average window size
        
    Raises:
        ValueError: If not exactly 2 agents are provided
        
    Returns:
        None (saves plot to file)
        
    Example:
        # For agents with very different scales
        agents = [navigator_agent, door_controller_agent]
        create_dual_axis_plot(agents, "dual_comparison.png")
    """
    # ========================================================================
    # INPUT VALIDATION
    # ========================================================================
    
    # Dual-axis plots require exactly 2 agents for meaningful comparison
    if len(agents) != 2:
        raise ValueError("Dual axis plot requires exactly 2 agents")
    
    # ========================================================================
    # DUAL-AXIS FIGURE SETUP
    # ========================================================================
    
    # Create main figure and primary axis
    fig, ax1 = plt.subplots(figsize=(16, 8))
    
    # Create secondary y-axis sharing the same x-axis
    ax2 = ax1.twinx()
    
    # ========================================================================
    # FIRST AGENT PLOTTING (LEFT AXIS)
    # ========================================================================
    
    agent1 = agents[0]
    
    # Plot raw data with transparency on left axis
    ax1.plot(agent1.iterations, agent1.values, 
            color=agent1.color, alpha=0.3, linewidth=0.5, 
            label=f"{agent1.label} (raw)")
    
    # Plot moving average trend if available
    if show_moving_average and window_size in agent1.moving_averages:
        ma = agent1.moving_averages[window_size]
        ma_iterations = agent1.iterations[window_size-1:]
        ax1.plot(ma_iterations, ma, 
                color=agent1.color, linewidth=2.0, 
                label=f"{agent1.label} (MA={window_size})")
    
    # ========================================================================
    # SECOND AGENT PLOTTING (RIGHT AXIS)
    # ========================================================================
    
    agent2 = agents[1]
    
    # Plot raw data with transparency on right axis
    ax2.plot(agent2.iterations, agent2.values, 
            color=agent2.color, alpha=0.3, linewidth=0.5, 
            label=f"{agent2.label} (raw)")
    
    # Plot moving average trend if available
    if show_moving_average and window_size in agent2.moving_averages:
        ma = agent2.moving_averages[window_size]
        ma_iterations = agent2.iterations[window_size-1:]
        ax2.plot(ma_iterations, ma, 
                color=agent2.color, linewidth=2.0, 
                label=f"{agent2.label} (MA={window_size})")
    
    # ========================================================================
    # DUAL-AXIS FORMATTING AND STYLING
    # ========================================================================
    
    # Set axis labels with color coding to match respective agents
    ax1.set_xlabel('Iteration', fontsize=14)
    ax1.set_ylabel(f'{agent1.label} Return', fontsize=14, color=agent1.color)
    ax2.set_ylabel(f'{agent2.label} Return', fontsize=14, color=agent2.color)
    
    # Color-code tick labels to match their respective y-axes
    # This helps users immediately identify which scale belongs to which agent
    ax1.tick_params(axis='y', labelcolor=agent1.color)
    ax2.tick_params(axis='y', labelcolor=agent2.color)
    
    # Add grid only on primary axis to avoid visual clutter
    ax1.grid(True, alpha=0.3, linestyle=':')
    
    # Set descriptive title
    plt.title('Multi-Agent Returns Comparison (Dual Axis)', fontsize=16)
    
    # ========================================================================
    # LEGEND MANAGEMENT FOR DUAL AXES
    # ========================================================================
    
    # Combine legends from both axes into a single, unified legend
    # This prevents having two separate legends which can be confusing
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=10)
    
    # ========================================================================
    # LAYOUT OPTIMIZATION AND EXPORT
    # ========================================================================
    
    # Optimize layout for dual-axis setup
    plt.tight_layout()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save high-quality figure
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"Saved dual-axis plot to {output_path}")
    
    # Free memory
    plt.close()

# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

def main():
    """
    Main function providing command-line interface for the RL visualization tool
    
    This function handles:
    - Command-line argument parsing and validation
    - Agent data loading and preprocessing
    - Statistical analysis and reporting
    - Visualization generation based on user preferences
    - File output management
    
    The CLI supports flexible visualization options:
    - Single-axis plots for similar-scale agents
    - Dual-axis plots for different-scale agents
    - Both plot types for comprehensive analysis
    - Custom moving average windows
    - Flexible output directory management
    """
    # ========================================================================
    # ARGUMENT PARSER CONFIGURATION
    # ========================================================================
    
    parser = argparse.ArgumentParser(
        description='Multi-Agent RL Visualization Tool',
        epilog="""
Examples:
  # Basic visualization with default settings
  %(prog)s --navigator-reward nav_data.json --door-reward door_data.json
  
  # Custom moving average and dual-axis only
  %(prog)s --navigator-reward nav.json --door-reward door.json --window-size 50 --plot-type dual
  
  # Publication output with custom directory
  %(prog)s --navigator-reward data/nav.json --door-reward data/door.json --output-dir figures/
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # ========================================================================
    # REQUIRED INPUT ARGUMENTS
    # ========================================================================
    
    # Agent data files - specifically designed for navigator-door controller scenarios
    parser.add_argument('--navigator-reward', type=str, required=True,
                        help='JSON file containing navigator agent reward/return data. '
                             'Format: [[timestamp, iteration, return_value], ...]')
    parser.add_argument('--door-reward', type=str, required=True,
                        help='JSON file containing door controller agent reward/return data. '
                             'Format: [[timestamp, iteration, return_value], ...]')
    
    # ========================================================================
    # OPTIONAL CUSTOMIZATION ARGUMENTS
    # ========================================================================
    
    # Analysis parameters
    parser.add_argument('--window-size', type=int, default=10, 
                        help='Moving average window size for trend smoothing (default: 10). '
                             'Larger values create smoother trends but reduce responsiveness.')
    
    # Output configuration
    parser.add_argument('--output', type=str, default=None,
                        help='Specific output file path. If not provided, uses automatic naming '
                             'based on plot type and parameters.')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR,
                        help=f'Output directory for generated plots (default: {OUTPUT_DIR}). '
                             'Created automatically if it does not exist.')
    
    # Visualization options
    parser.add_argument('--plot-type', type=str, default='both',
                        choices=['single', 'dual', 'both'],
                        help='Type of visualization to generate:\n'
                             '  single: Single-axis plot (good for similar scales)\n'
                             '  dual: Dual-axis plot (good for different scales)\n'
                             '  both: Generate both types (default)')
    
    # Parse command-line arguments
    args = parser.parse_args()
    
    # ========================================================================
    # DATA LOADING AND VALIDATION
    # ========================================================================
    
    agents = []
    
    # Load navigator agent data
    try:
        navigator_agent = AgentData(args.navigator_reward, "navigator")
        agents.append(navigator_agent)
    except Exception as e:
        print(f"Error loading navigator data from {args.navigator_reward}: {e}")
        return 1
    
    # Load door controller agent data
    try:
        door_agent = AgentData(args.door_reward, "door_controller")
        agents.append(door_agent)
    except Exception as e:
        print(f"Error loading door controller data from {args.door_reward}: {e}")
        return 1
    
    # ========================================================================
    # STATISTICAL ANALYSIS AND REPORTING
    # ========================================================================
    
    # Print comprehensive statistics for each agent
    print("=" * 60)
    print("MULTI-AGENT RL TRAINING ANALYSIS")
    print("=" * 60)
    
    for agent in agents:
        agent.print_statistics()
    
    # Print comparative analysis
    print(f"\n===== Comparative Analysis =====")
    if len(agents) == 2:
        nav_agent, door_agent = agents[0], agents[1]
        print(f"Performance ratio (Nav/Door): {nav_agent.mean / door_agent.mean:.3f}")
        print(f"Variability ratio (Nav/Door): {nav_agent.std / door_agent.std:.3f}")
        
        # Check if scales are significantly different (useful for plot type recommendation)
        scale_ratio = max(nav_agent.max, door_agent.max) / min(nav_agent.max, door_agent.max)
        if scale_ratio > 10:
            print(f"Note: Large scale difference detected (ratio: {scale_ratio:.1f}). "
                  f"Dual-axis plot recommended.")
    
    # ========================================================================
    # VISUALIZATION GENERATION
    # ========================================================================
    
    print(f"\nGenerating visualizations...")
    
    # Generate single-axis plot if requested
    if args.plot_type in ['single', 'both']:
        output_path = args.output or os.path.join(
            args.output_dir, 
            f"multi_agent_returns_MA{args.window_size}.png"
        )
        
        try:
            create_multi_agent_plot(agents, output_path, window_size=args.window_size)
        except Exception as e:
            print(f"Error creating single-axis plot: {e}")
    
    # Generate dual-axis plot if requested
    if args.plot_type in ['dual', 'both']:
        output_path = args.output or os.path.join(
            args.output_dir, 
            f"multi_agent_returns_dual_axis_MA{args.window_size}.png"
        )
        
        try:
            create_dual_axis_plot(agents, output_path, window_size=args.window_size)
        except Exception as e:
            print(f"Error creating dual-axis plot: {e}")
    
    print(f"\nVisualization generation completed!")
    return 0

# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    """
    Entry point for command-line execution
    
    Example usage scenarios:
    
    1. Quick comparison of two agents:
       python vis_rltrainingmultirun.py --navigator-reward nav.json --door-reward door.json
    
    2. Publication-quality plots with custom smoothing:
       python vis_rltrainingmultirun.py --navigator-reward data/nav.json --door-reward data/door.json 
                                        --window-size 50 --output-dir publication_figures/
    
    3. Dual-axis only for agents with very different scales:
       python vis_rltrainingmultirun.py --navigator-reward nav.json --door-reward door.json 
                                        --plot-type dual --window-size 20
    
    4. Custom output file naming:
       python vis_rltrainingmultirun.py --navigator-reward nav.json --door-reward door.json 
                                        --output experiment_1_comparison.png --plot-type single
    """
    exit_code = main()
    exit(exit_code)