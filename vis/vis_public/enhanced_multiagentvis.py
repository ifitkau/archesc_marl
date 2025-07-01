#!/usr/bin/env python3
"""
Combined Moving Average and Segment Analysis Visualization

This script creates a unified visualization that combines moving average trend lines
with segment-by-segment analysis for multi-agent reinforcement learning data.

The visualization provides two main view types:
1. Standard unified view - Shows all agents on the same scale
2. Dual-axis unified view - Shows exactly two agents with different y-axis scales

Key Features:
- Moving average trend analysis with configurable window sizes
- Automatic data segmentation for phase-based analysis
- Colorblind-friendly palette
- Statistical annotations (means, trends)
- High-quality output suitable for publications


Date: July 2025
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.ndimage import gaussian_filter1d
from matplotlib.gridspec import GridSpec

# Import the AgentData class from the main script
# Uncomment the line below if you have the main enhanced_multi_agent_vis module
# from enhanced_multi_agent_vis import AgentData

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# Moving average window size - controls smoothing of trend lines
WINDOW_SIZE = 10

# Number of training segments to analyze - divides training into equal phases
SEGMENT_COUNT = 4

# Gaussian smoothing parameter for segment plots - higher values = more smoothing
SMOOTHING_SIGMA = 2

# Output image resolution - 300 DPI suitable for publications
DPI = 300

# Colorblind-friendly color palette using distinct, accessible colors
# Based on Paul Tol's colorblind-safe palette recommendations
COLORBLIND_PALETTE = [
    "#0072B2",  # Blue - primary agent color
    "#D55E00",  # Red/Orange - secondary agent color
    "#009E73",  # Green - tertiary agent color
    "#CC79A7",  # Pink - quaternary agent color
]

# ============================================================================
# MAIN VISUALIZATION FUNCTIONS
# ============================================================================

def create_unified_visualization(agents, output_path, window_size=WINDOW_SIZE, num_segments=SEGMENT_COUNT):
    """
    Create a unified visualization that combines moving average trends with segment analysis
    
    This function creates a two-panel visualization:
    - Top panel: Moving average trends for all agents with segment boundaries
    - Bottom panel: Individual segment analysis with statistics
    
    Args:
        agents: List of AgentData objects containing training data
                Each agent should have: steps, values, label, color (optional)
        output_path: Path to save the output PNG file
        window_size: Window size for moving average calculation (default: 10)
        num_segments: Number of segments to divide training into (default: 4)
    
    Returns:
        None (saves visualization to file)
    
    Example:
        agents = [agent1, agent2, agent3]  # List of AgentData objects
        create_unified_visualization(agents, "output/training_analysis.png")
    """
    # ========================================================================
    # DATA PREPARATION
    # ========================================================================
    
    # Calculate segment analysis for each agent if not already done
    # This divides the training data into equal time segments for comparison
    for agent in agents:
        if len(agent.segments) != num_segments:
            agent.segments = agent.segment_analysis(num_segments)
    
    # Calculate segment boundaries using the first agent as reference
    # All agents should have similar training lengths for meaningful comparison
    reference_agent = agents[0]
    segment_boundaries = [segment['start_step'] for segment in reference_agent.segments[1:]]
    
    # ========================================================================
    # FIGURE LAYOUT SETUP
    # ========================================================================
    
    # Create main figure with specified size (width=14, height=10 inches)
    fig = plt.figure(figsize=(14, 10))
    
    # Create hierarchical grid layout:
    # - Top section (height ratio 1): Moving average plot
    # - Bottom section (height ratio 2): Segment analysis plots
    outer_grid = GridSpec(2, 1, height_ratios=[1, 2], hspace=0.3)
    
    # Create the main moving average plot in the top section
    ax_ma = fig.add_subplot(outer_grid[0])
    
    # Create horizontal grid for segment plots in the bottom section
    # Each segment gets its own subplot for detailed analysis
    segment_grid = GridSpec(1, num_segments, figure=fig, 
                           left=outer_grid[1].left, right=outer_grid[1].right,
                           top=outer_grid[1].top, bottom=outer_grid[1].bottom,
                           wspace=0.2)
    
    # Create individual subplot axes for each segment
    segment_axes = [fig.add_subplot(segment_grid[0, i]) for i in range(num_segments)]
    
    # ========================================================================
    # COLOR ASSIGNMENT
    # ========================================================================
    
    # Assign colors to agents if not already specified
    # Uses colorblind-friendly palette with cycling for >4 agents
    for i, agent in enumerate(agents):
        if agent.color is None:
            agent.color = COLORBLIND_PALETTE[i % len(COLORBLIND_PALETTE)]
    
    # ========================================================================
    # TOP PANEL: MOVING AVERAGE TRENDS
    # ========================================================================
    
    # Plot moving average trends for each agent
    for agent in agents:
        # Check if moving average has been calculated for this window size
        if window_size in agent.moving_averages:
            ma = agent.moving_averages[window_size]
            # Moving average steps start from (window_size-1) due to initial window requirement
            ma_steps = agent.steps[window_size-1:]
            
            # Plot trend line with thick line for visibility
            ax_ma.plot(ma_steps, ma, color=agent.color, linewidth=2.5, 
                      label=f"{agent.label} (MA={window_size})")
    
    # Add vertical lines to mark segment boundaries
    # These help visualize where training phases change
    for i, boundary in enumerate(segment_boundaries):
        ax_ma.axvline(boundary, color='gray', linestyle='--', alpha=0.4)
        # Add rotated text labels at boundary lines
        ax_ma.text(boundary, ax_ma.get_ylim()[1] * 0.9, f"Segment {i+1}|{i+2}", 
                  rotation=90, ha='right', va='top', fontsize=8,
                  bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    # Add segment labels in the center of each segment
    # This provides clear identification of training phases
    segment_midpoints = []
    for i, agent in enumerate(agents[0:1]):  # Just use first agent to avoid duplicate labels
        for j, segment in enumerate(agent.segments):
            # Calculate midpoint of each segment
            midpoint = (segment['start_step'] + segment['end_step']) / 2
            segment_midpoints.append(midpoint)
            
            # Add centered segment label at the top of the plot
            ax_ma.text(midpoint, ax_ma.get_ylim()[1] * 0.98, f"Segment {j+1}", 
                      ha='center', va='top', fontsize=10, color='gray',
                      bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    # Format the moving average plot
    ax_ma.set_title('Agent Returns with Moving Average and Segments', fontsize=14)
    ax_ma.set_ylabel('Return Value', fontsize=12)
    ax_ma.grid(True, alpha=0.3, linestyle=':')  # Light grid for readability
    ax_ma.legend(loc='upper right')
    
    # ========================================================================
    # BOTTOM PANEL: SEGMENT-BY-SEGMENT ANALYSIS
    # ========================================================================
    
    # Create individual plots for each training segment
    for i, ax in enumerate(segment_axes):
        # Set title for each segment subplot
        ax.set_title(f'Segment {i+1}', fontsize=12)
        
        # Plot data for each agent within this segment
        for agent in agents:
            segment = agent.segments[i]
            
            # Extract segment-specific data using precomputed indices
            start_idx = segment['start_idx']
            end_idx = segment['end_idx']
            
            segment_steps = agent.steps[start_idx:end_idx]
            segment_values = agent.values[start_idx:end_idx]
            
            # Apply Gaussian smoothing if segment is long enough
            # Prevents over-smoothing of short segments
            if len(segment_values) > SMOOTHING_SIGMA * 3:
                segment_smoothed = gaussian_filter1d(segment_values, sigma=SMOOTHING_SIGMA)
                ax.plot(segment_steps, segment_smoothed, color=agent.color, linewidth=2.0)
            else:
                # Plot raw data for short segments
                ax.plot(segment_steps, segment_values, color=agent.color, linewidth=2.0)
            
            # Add horizontal line showing segment mean performance
            # This provides quick visual reference for average performance
            ax.axhline(segment['mean'], linestyle='--', color=agent.color, alpha=0.5)
            
            # Add text annotation with segment statistics
            # Positioned in top-right corner with agent-specific color
            y_pos = 0.95 #- (0.08 * agents.index(agent))  # Stagger if multiple agents
            ax.text(0.98, y_pos,
                  f"{agent.label}: {segment['mean']:.1f}",
                  transform=ax.transAxes, va='top', ha='right', fontsize=9, color=agent.color,
                  bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        # Set y-axis label only for the leftmost subplot to avoid redundancy
        if i == 0:
            ax.set_ylabel('Return', fontsize=10)
        
        # Add subtle grid for readability
        ax.grid(True, alpha=0.3, linestyle=':')
        
        # Format x-axis with thousands separators for large step numbers
        # Makes training step numbers more readable (e.g., "10,000" instead of "10000")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{x:,.0f}"))
    
    # ========================================================================
    # FINAL FORMATTING AND EXPORT
    # ========================================================================
    
    # Add common x-axis label for all segment plots
    # Positioned at bottom center of the figure
    fig.text(0.5, 0.05, 'Training Step (Iteration)', ha='center', fontsize=12)
    
    # Adjust layout to prevent overlapping elements
    # rect parameter leaves space for the x-axis label
    plt.tight_layout(rect=[0, 0.07, 1, 0.98])
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save high-quality figure
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"Saved unified visualization to {output_path}")
    
    # Close figure to free memory
    plt.close(fig)


def create_dual_axis_unified_visualization(agents, output_path, window_size=WINDOW_SIZE, num_segments=SEGMENT_COUNT):
    """
    Create a unified visualization with dual y-axes for two agents with different scales
    
    This function is specifically designed for comparing two agents whose performance
    metrics are on vastly different scales (e.g., one agent gets returns 0-100, 
    another gets returns 0-10000). Each agent gets its own y-axis scale.
    
    Layout:
    - Top panel: Moving average trends with dual y-axes
    - Bottom panel: Segment analysis with dual y-axes for each segment
    
    Args:
        agents: List of exactly two AgentData objects
                Must contain exactly 2 agents for dual-axis comparison
        output_path: File path to save the visualization
        window_size: Moving average window size (default: 10)
        num_segments: Number of training segments (default: 4)
    
    Raises:
        ValueError: If agents list doesn't contain exactly 2 agents
    
    Returns:
        None (saves visualization to file)
    
    Example:
        # Compare two agents with different return scales
        agents = [low_scale_agent, high_scale_agent]
        create_dual_axis_unified_visualization(agents, "dual_comparison.png")
    """
    # ========================================================================
    # INPUT VALIDATION
    # ========================================================================
    
    # Ensure exactly 2 agents for dual-axis comparison
    if len(agents) != 2:
        raise ValueError("This visualization requires exactly 2 agents")
    
    # ========================================================================
    # DATA PREPARATION
    # ========================================================================
    
    # Calculate segment analysis for each agent if not already computed
    for agent in agents:
        if len(agent.segments) != num_segments:
            agent.segments = agent.segment_analysis(num_segments)
    
    # Use first agent as reference for segment timing
    # Both agents should have similar training duration
    reference_agent = agents[0]
    segment_boundaries = [segment['start_step'] for segment in reference_agent.segments[1:]]
    
    # ========================================================================
    # FIGURE LAYOUT WITH DUAL AXES
    # ========================================================================
    
    # Create main figure
    fig = plt.figure(figsize=(14, 10))
    
    # Create hierarchical layout (same as single-axis version)
    outer_grid = GridSpec(2, 1, height_ratios=[1, 2], hspace=0.3)
    
    # Create dual-axis setup for moving average plot
    ax_ma_left = fig.add_subplot(outer_grid[0])  # Left y-axis for agent 1
    ax_ma_right = ax_ma_left.twinx()             # Right y-axis for agent 2
    
    # Create segment subplot grid
    segment_grid = GridSpec(1, num_segments, figure=fig, 
                           left=outer_grid[1].left, right=outer_grid[1].right,
                           top=outer_grid[1].top, bottom=outer_grid[1].bottom,
                           wspace=0.2)
    
    # Create dual-axis pairs for each segment subplot
    segment_axes = []
    for i in range(num_segments):
        ax_left = fig.add_subplot(segment_grid[0, i])   # Left axis for agent 1
        ax_right = ax_left.twinx()                      # Right axis for agent 2
        segment_axes.append((ax_left, ax_right))
    
    # ========================================================================
    # COLOR ASSIGNMENT FOR DUAL AXES
    # ========================================================================
    
    # Assign distinct colors to each agent
    # Use first two colors from colorblind-friendly palette
    if agents[0].color is None:
        agents[0].color = COLORBLIND_PALETTE[0]  # Blue for first agent
    if agents[1].color is None:
        agents[1].color = COLORBLIND_PALETTE[1]  # Orange for second agent
    
    # ========================================================================
    # TOP PANEL: DUAL-AXIS MOVING AVERAGE PLOT
    # ========================================================================
    
    # Plot first agent on left y-axis
    agent = agents[0]
    if window_size in agent.moving_averages:
        ma = agent.moving_averages[window_size]
        ma_steps = agent.steps[window_size-1:]
        
        ax_ma_left.plot(ma_steps, ma, color=agent.color, linewidth=2.5, 
                      label=f"{agent.label} (MA={window_size})")
    
    # Plot second agent on right y-axis
    agent = agents[1]
    if window_size in agent.moving_averages:
        ma = agent.moving_averages[window_size]
        ma_steps = agent.steps[window_size-1:]
        
        ax_ma_right.plot(ma_steps, ma, color=agent.color, linewidth=2.5, 
                       label=f"{agent.label} (MA={window_size})")
    
    # Add segment boundary markers (only on left axis to avoid duplication)
    for boundary in segment_boundaries:
        ax_ma_left.axvline(boundary, color='gray', linestyle='--', alpha=0.4)
    
    # Add segment labels in the middle of each segment
    for i, segment in enumerate(reference_agent.segments):
        midpoint = (segment['start_step'] + segment['end_step']) / 2
        
        # Add segment label (positioned on left axis)
        ax_ma_left.text(midpoint, ax_ma_left.get_ylim()[1] * 0.9, f"Segment {i+1}", 
                      ha='center', va='top', fontsize=10, color='gray',
                      bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    # ========================================================================
    # FORMAT DUAL-AXIS MOVING AVERAGE PLOT
    # ========================================================================
    
    # Set title and axis labels with agent-specific colors
    ax_ma_left.set_title('Multi-Agent Returns with Moving Average and Segments', fontsize=14)
    ax_ma_left.set_ylabel(f'{agents[0].label} Return', fontsize=12, color=agents[0].color)
    ax_ma_right.set_ylabel(f'{agents[1].label} Return', fontsize=12, color=agents[1].color)
    
    # Color-code the tick labels to match their respective agents
    ax_ma_left.tick_params(axis='y', labelcolor=agents[0].color)
    ax_ma_right.tick_params(axis='y', labelcolor=agents[1].color)
    
    # Add grid (only on left axis to avoid visual clutter)
    ax_ma_left.grid(True, alpha=0.3, linestyle=':')
    
    # Combine legends from both axes into a single legend
    lines_left, labels_left = ax_ma_left.get_legend_handles_labels()
    lines_right, labels_right = ax_ma_right.get_legend_handles_labels()
    ax_ma_left.legend(lines_left + lines_right, labels_left + labels_right, loc='upper right')
    
    # ========================================================================
    # BOTTOM PANEL: DUAL-AXIS SEGMENT ANALYSIS
    # ========================================================================
    
    # Process each segment with dual axes for detailed comparison
    for i, (ax_left, ax_right) in enumerate(segment_axes):
        # Set segment title
        ax_left.set_title(f'Segment {i+1}', fontsize=12)
        
        # ====================================================================
        # PLOT FIRST AGENT ON LEFT AXIS
        # ====================================================================
        agent = agents[0]
        segment = agent.segments[i]
        
        # Extract segment data
        start_idx = segment['start_idx']
        end_idx = segment['end_idx']
        
        segment_steps = agent.steps[start_idx:end_idx]
        segment_values = agent.values[start_idx:end_idx]
        
        # Apply smoothing if segment is long enough
        if len(segment_values) > SMOOTHING_SIGMA * 3:
            segment_smoothed = gaussian_filter1d(segment_values, sigma=SMOOTHING_SIGMA)
            ax_left.plot(segment_steps, segment_smoothed, color=agent.color, linewidth=2.0)
        else:
            ax_left.plot(segment_steps, segment_values, color=agent.color, linewidth=2.0)
        
        # Add mean line and statistics annotation
        ax_left.axhline(segment['mean'], linestyle='--', color=agent.color, alpha=0.5)
        
        # Position annotation on left side for left-axis agent
        ax_left.text(0.02, 0.95,
                  f"{agent.label}: {segment['mean']:.1f}",
                  transform=ax_left.transAxes, va='top', ha='left', fontsize=9, color=agent.color,
                  bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        # ====================================================================
        # PLOT SECOND AGENT ON RIGHT AXIS
        # ====================================================================
        agent = agents[1]
        segment = agent.segments[i]
        
        # Extract segment data
        start_idx = segment['start_idx']
        end_idx = segment['end_idx']
        
        segment_steps = agent.steps[start_idx:end_idx]
        segment_values = agent.values[start_idx:end_idx]
        
        # Apply smoothing if segment is long enough
        if len(segment_values) > SMOOTHING_SIGMA * 3:
            segment_smoothed = gaussian_filter1d(segment_values, sigma=SMOOTHING_SIGMA)
            ax_right.plot(segment_steps, segment_smoothed, color=agent.color, linewidth=2.0)
        else:
            ax_right.plot(segment_steps, segment_values, color=agent.color, linewidth=2.0)
        
        # Add mean line and statistics annotation
        ax_right.axhline(segment['mean'], linestyle='--', color=agent.color, alpha=0.5)
        
        # Position annotation on right side for right-axis agent
        ax_right.text(0.98, 0.95,
                   f"{agent.label}: {segment['mean']:.1f}",
                   transform=ax_right.transAxes, va='top', ha='right', fontsize=9, color=agent.color,
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        # ====================================================================
        # FORMAT DUAL-AXIS SEGMENT PLOTS
        # ====================================================================
        
        # Color-code y-axis tick labels to match agents
        ax_left.tick_params(axis='y', labelcolor=agents[0].color)
        ax_right.tick_params(axis='y', labelcolor=agents[1].color)
        
        # Set y-axis labels only for leftmost subplot pair
        if i == 0:
            ax_left.set_ylabel(f'{agents[0].label}', fontsize=10, color=agents[0].color)
            ax_right.set_ylabel(f'{agents[1].label}', fontsize=10, color=agents[1].color)
        
        # Add grid (only on left axis)
        ax_left.grid(True, alpha=0.3, linestyle=':')
        
        # Format x-axis with thousands separators
        ax_left.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{x:,.0f}"))
    
    # ========================================================================
    # FINAL FORMATTING AND EXPORT
    # ========================================================================
    
    # Add common x-axis label
    fig.text(0.5, 0.05, 'Training Step (Iteration)', ha='center', fontsize=12)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.07, 1, 0.98])
    
    # Create output directory and save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"Saved dual-axis unified visualization to {output_path}")
    
    # Free memory
    plt.close(fig)


# ============================================================================
# EXAMPLE USAGE AND TESTING
# ============================================================================

# Example usage (if this script is run directly)
if __name__ == "__main__":
    """
    Example usage section - uncomment and modify for your data
    
    This section demonstrates how to use the visualization functions
    with your own AgentData objects.
    
    Prerequisites:
    1. You need AgentData objects with the following attributes:
       - steps: numpy array of training step numbers
       - values: numpy array of return/reward values
       - label: string identifier for the agent
       - color: matplotlib color (optional, will be auto-assigned)
       - moving_averages: dict with window_size as keys
       - segments: list of segment analysis results
    
    2. Agent data should be preprocessed with:
       - Moving average calculation
       - Segment analysis computation
    
    Example code:
    
    # Load your agent data (replace with your data loading code)
    agent1 = AgentData(steps=steps1, values=values1, label="PPO Agent")
    agent2 = AgentData(steps=steps2, values=values2, label="SAC Agent")
    
    # Calculate moving averages and segments
    agent1.calculate_moving_averages([10, 50, 100])
    agent2.calculate_moving_averages([10, 50, 100])
    
    # Create visualizations
    agents = [agent1, agent2]
    
    # Standard unified visualization
    create_unified_visualization(
        agents=agents,
        output_path="output/training_comparison.png",
        window_size=10,
        num_segments=4
    )
    
    # Dual-axis visualization (for agents with different scales)
    create_dual_axis_unified_visualization(
        agents=agents,
        output_path="output/dual_axis_comparison.png",
        window_size=10,
        num_segments=4
    )
    """
    # This would be where you'd load your agent data and call the visualization functions
    print("Enhanced Multi-Agent Visualization Module")
    print("========================================")
    print("This module provides two main visualization functions:")
    print("1. create_unified_visualization() - for standard multi-agent comparison")
    print("2. create_dual_axis_unified_visualization() - for two agents with different scales")
    print("\nSee the docstrings and example usage section for implementation details.")
    pass