"""
Enhanced Door Position Analysis & Visualization Script

This script analyzes and creates publication-quality visualizations of door positions
from RL training runs. It supports comparing door positions across multiple runs
and visualizing patterns in door controller agent behavior.

Includes improved visualizations with discrete door position markers and
percentage-based episode segmentation for more intuitive analysis.

Usage:
  python enhanced_visdoor.py door_positions1.txt --labels "Run 1" --start-episode 2000 --end-episode 3000

Date: July 2025
"""

import os
import sys
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime
import shutil
from collections import defaultdict
import json

# Try to import seaborn for better aesthetics
try:
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Note: Install seaborn for better plot aesthetics: pip install seaborn")

# Configuration
OUTPUT_DIR = "door_position_figures"  # Output directory
DPI = 300                   # Output DPI

# Color palettes (colorblind-friendly)
COLORBLIND_PALETTE = [
    "#0072B2",  # Blue
    "#D55E00",  # Red/Orange
    "#009E73",  # Green
    "#CC79A7",  # Pink
    "#F0E442",  # Yellow
    "#56B4E9",  # Light Blue
    "#E69F00",  # Orange
    "#000000"   # Black
]

# Room colors for consistency
ROOM_COLORS = {
    'roomA': "#0072B2",  # Blue
    'roomB': "#D55E00",  # Red/Orange  
    'roomC': "#009E73"   # Green
}

class DoorPositionData:
    """Class to store and analyze door position data from a single run"""
    
    def __init__(self, file_path, label=None, start_episode=None, end_episode=None, exclude_zeros=False):
        """
        Initialize door position data from file
        
        Args:
            file_path: Path to door position data file
            label: Label for this run
            start_episode: First episode to include (inclusive)
            end_episode: Last episode to include (inclusive)
            exclude_zeros: Whether to exclude 0.0 values from data (treat as missing data)
        """
        self.file_path = file_path
        self.label = label or os.path.splitext(os.path.basename(file_path))[0]
        self.start_episode = start_episode
        self.end_episode = end_episode
        self.exclude_zeros = exclude_zeros
        
        # Try to extract config settings from related files
        self.config = self._load_config()
        
        # Parse door position data
        self.episodes = []
        self.positions = {'roomA': [], 'roomB': [], 'roomC': []}
        self.load_data()
        
        # Calculate statistics
        self.stats = self.calculate_statistics()
        
        # Analyze distribution
        self.distributions = self.analyze_distribution()
        
        # Analyze transitions
        self.transitions = self.analyze_transitions()
        
        # Analyze segments
        self.segments = self.split_into_segments()
    
    def _load_config(self):
        """Attempt to load config data from related files"""
        config = {
            'discrete_door_positions': 5,  # Default value
            'door_position_min': 0.6,
            'door_position_max': 3.4,
            'terminal_location': [18.4, 5.95]  # Default
        }
        
        # Try to find config in same directory as data file
        dir_path = os.path.dirname(self.file_path)
        
        # Common config filenames
        config_files = [
            'config.json', 
            'default_config.py',
            'experiment_config.json'
        ]
        
        for filename in config_files:
            file_path = os.path.join(dir_path, filename)
            if os.path.exists(file_path):
                try:
                    # For JSON files
                    if filename.endswith('.json'):
                        with open(file_path, 'r') as f:
                            loaded_config = json.load(f)
                            # Update config with loaded values
                            for key in ['discrete_door_positions', 'door_position_min', 'door_position_max', 'terminal_location']:
                                if key in loaded_config:
                                    config[key] = loaded_config[key]
                    # For Python config files (basic parsing)
                    elif filename.endswith('.py'):
                        with open(file_path, 'r') as f:
                            content = f.read()
                            # Try to extract discrete_door_positions
                            match = re.search(r'discrete_door_positions"\s*:\s*(\d+)', content)
                            if match:
                                config['discrete_door_positions'] = int(match.group(1))
                            # Try to extract terminal_location
                            match = re.search(r'terminal_location"\s*:\s*\[([\d.]+),\s*([\d.]+)\]', content)
                            if match:
                                config['terminal_location'] = [float(match.group(1)), float(match.group(2))]
                except Exception as e:
                    print(f"Warning: Could not load config from {file_path}: {e}")
        
        # Try to infer terminal position from label
        if self.label:
            label_lower = self.label.lower()
            if "_r" in label_lower or "right" in label_lower:
                config['terminal_location'] = [12.4, 3.7]  # Right terminal
            elif "_l" in label_lower or "left" in label_lower:
                config['terminal_location'] = [0.0, 3.7]   # Left terminal
            elif "_m" in label_lower or "_c" in label_lower or "middle" in label_lower or "center" in label_lower:
                config['terminal_location'] = [6.2, 3.7]   # Middle terminal
        
        return config
    
    def load_data(self):
        """Load door position data from file with episode filtering"""
        pattern = re.compile(r'Episode (\d+) \|(?:.*?\|)? roomA:([\d.]+), roomB:([\d.]+), roomC:([\d.]+)')
        
        try:
            with open(self.file_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                match = pattern.match(line.strip())
                if match:
                    episode = int(match.group(1))
                    
                    # Skip episodes outside the specified range
                    if self.start_episode is not None and episode < self.start_episode:
                        continue
                    if self.end_episode is not None and episode > self.end_episode:
                        continue
                    
                    # Parse door positions
                    roomA_pos = float(match.group(2))
                    roomB_pos = float(match.group(3))
                    roomC_pos = float(match.group(4))
                    
                    # Skip the episode if excluding zeros and all positions are zero
                    if self.exclude_zeros and roomA_pos == 0.0 and roomB_pos == 0.0 and roomC_pos == 0.0:
                        continue
                    
                    self.episodes.append(episode)
                    
                    # Replace zeros with NaN if exclude_zeros is True
                    if self.exclude_zeros:
                        self.positions['roomA'].append(roomA_pos if roomA_pos > 0.0 else float('nan'))
                        self.positions['roomB'].append(roomB_pos if roomB_pos > 0.0 else float('nan'))
                        self.positions['roomC'].append(roomC_pos if roomC_pos > 0.0 else float('nan'))
                    else:
                        self.positions['roomA'].append(roomA_pos)
                        self.positions['roomB'].append(roomB_pos)
                        self.positions['roomC'].append(roomC_pos)
            
            # Log info with zero handling
            zero_info = "excluding zeros" if self.exclude_zeros else "including zeros"
            print(f"{self.label}: Loaded {len(self.episodes)} episodes ({zero_info})")
        
        except Exception as e:
            print(f"Error loading data from {self.file_path}: {e}")
            raise
    
    def calculate_statistics(self):
        """Calculate basic statistics for door positions"""
        stats = {}
        for room in self.positions:
            # Filter out NaN values if any
            valid_positions = [pos for pos in self.positions[room] if not (isinstance(pos, float) and np.isnan(pos))]
            
            if not valid_positions:
                continue
                
            positions = np.array(valid_positions)
            stats[room] = {
                'mean': np.mean(positions),
                'median': np.median(positions),
                'std': np.std(positions),
                'min': np.min(positions),
                'max': np.max(positions),
                'range': np.max(positions) - np.min(positions)
            }
        
        return stats
    
    def analyze_distribution(self, bins=20):
        """Analyze the distribution of door positions"""
        distributions = {}
        for room in self.positions:
            if not self.positions[room]:
                continue
                
            positions = np.array(self.positions[room])
            hist, bin_edges = np.histogram(positions, bins=bins, range=(0.6, 3.4))
            distributions[room] = {
                'histogram': hist,
                'bin_edges': bin_edges,
                'bin_centers': (bin_edges[:-1] + bin_edges[1:]) / 2,
                'modes': self._find_modes(positions, bin_edges)
            }
        
        return distributions
    
    def _find_modes(self, data, bin_edges, min_count=5):
        """Find modes in histogram data"""
        hist, _ = np.histogram(data, bins=bin_edges)
        modes = []
        for i, count in enumerate(hist):
            if count >= min_count:
                bin_center = (bin_edges[i] + bin_edges[i+1]) / 2
                modes.append({
                    'center': bin_center,
                    'count': count,
                    'frequency': count / len(data)
                })
        
        # Sort modes by count (descending)
        modes.sort(key=lambda x: x['count'], reverse=True)
        return modes
    
    def analyze_transitions(self):
        transitions = {}
        for room in self.positions:
            # Filter out NaN values
            valid_indices = []
            valid_positions = []
            
            for i, pos in enumerate(self.positions[room]):
                if not (isinstance(pos, float) and np.isnan(pos)):
                    valid_indices.append(i)
                    valid_positions.append(pos)
            
            if len(valid_positions) < 2:
                # Not enough valid positions for transitions
                transitions[room] = {
                    'changes': np.array([]),
                    'mean_change': 0,
                    'max_change': 0,
                    'positive_changes': 0,
                    'negative_changes': 0,
                    'no_changes': 0
                }
                continue
            
            # Calculate transitions between consecutive VALID positions
            positions = np.array(valid_positions)
            changes = np.diff(positions)
            
            # Record transitions
            transitions[room] = {
                'changes': changes,
                'mean_change': np.mean(np.abs(changes)) if len(changes) > 0 else 0,
                'max_change': np.max(np.abs(changes)) if len(changes) > 0 else 0,
                'positive_changes': np.sum(changes > 0),
                'negative_changes': np.sum(changes < 0),
                'no_changes': np.sum(changes == 0),
            }
        
        return transitions
    
    def split_into_segments(self, num_segments=4):
        segments = []
        
        if not self.episodes:
            return segments
            
        # Calculate segment size
        segment_size = max(1, len(self.episodes) // num_segments)
        
        for i in range(num_segments):
            start_idx = i * segment_size
            end_idx = min(len(self.episodes), (i + 1) * segment_size)
            
            # Skip segments with no data
            if start_idx >= len(self.episodes):
                continue
                
            segment_stats = {}
            for room in self.positions:
                # Get positions for this segment
                room_positions = self.positions[room][start_idx:end_idx]
                
                # Filter out NaN values 
                valid_positions = [pos for pos in room_positions 
                                if not (isinstance(pos, float) and np.isnan(pos))]
                
                if valid_positions:  # Only add stats if we have valid data
                    segment_stats[room] = {
                        'mean': np.mean(valid_positions),
                        'std': np.std(valid_positions),
                        'min': np.min(valid_positions),
                        'max': np.max(valid_positions)
                    }
            
            segments.append({
                'start_episode': self.episodes[start_idx],
                'end_episode': self.episodes[end_idx-1] if end_idx <= len(self.episodes) else self.episodes[-1],
                'stats': segment_stats
            })
        
        return segments
    
    def print_statistics(self):
        """Print statistics about door positions"""
        episode_range = f"Episodes {self.episodes[0]}-{self.episodes[-1]}" if self.episodes else "No episodes"
        print(f"\n===== Door Position Statistics for {self.label} ({episode_range}) =====")
        print(f"Total episodes: {len(self.episodes)}")
        
        for room in sorted(self.stats.keys()):
            print(f"\n{room}:")
            print(f"  Mean position: {self.stats[room]['mean']:.4f}")
            print(f"  Median position: {self.stats[room]['median']:.4f}")
            print(f"  Standard deviation: {self.stats[room]['std']:.4f}")
            print(f"  Range: {self.stats[room]['min']:.4f} - {self.stats[room]['max']:.4f} (span: {self.stats[room]['range']:.4f})")
            
            # Print mode information
            if self.distributions[room]['modes']:
                print("\n  Position modes (most frequent positions):")
                for i, mode in enumerate(self.distributions[room]['modes'][:3]):  # Top 3 modes
                    print(f"    Mode {i+1}: {mode['center']:.4f} (frequency: {mode['frequency']:.2%})")
            
            # Print transition information
            print("\n  Position changes:")
            print(f"    Mean absolute change: {self.transitions[room]['mean_change']:.4f}")
            print(f"    Max absolute change: {self.transitions[room]['max_change']:.4f}")
            print(f"    Direction: {self.transitions[room]['positive_changes']} increases, "
                  f"{self.transitions[room]['negative_changes']} decreases, "
                  f"{self.transitions[room]['no_changes']} no change")
        
        # Print segment information
        print("\nSegment Analysis:")
        for i, segment in enumerate(self.segments):
            print(f"\n  Segment {i+1} (Episodes {segment['start_episode']}-{segment['end_episode']}):")
            for room in sorted(segment['stats'].keys()):
                print(f"    {room}: {segment['stats'][room]['mean']:.4f} ± {segment['stats'][room]['std']:.4f}")


def create_position_timeline(runs, output_path, room='roomA'):
    """
    Create a timeline plot of door positions for one room across runs
    with improved discrete position markers
    
    Args:
        runs: List of DoorPositionData objects
        output_path: Path to save the output file
        room: Room to visualize ('roomA', 'roomB', or 'roomC')
    """
    plt.figure(figsize=(10, 6))
    
    # Get number of discrete door positions from first run with data
    # Replace with more robust version that also respects min/max positions:
    num_door_positions = 5  # Default to 5
    door_min = 0.6  # Default min
    door_max = 3.4  # Default max

    for run in runs:
        if hasattr(run, 'config'):
            if 'discrete_door_positions' in run.config:
                num_door_positions = run.config['discrete_door_positions']
            if 'door_position_min' in run.config:
                door_min = run.config['door_position_min']
            if 'door_position_max' in run.config:
                door_max = run.config['door_position_max']
            break
    door_positions = []
    
    if num_door_positions > 1:
        # Create evenly spaced door positions
        for i in range(num_door_positions):
            position = door_min + i * ((door_max - door_min) / (num_door_positions - 1))
            door_positions.append(round(position, 2))
    else:
        door_positions = [2.0]  # Default middle position
    
    # Plot door positions for each run
    for i, run in enumerate(runs):
        color = COLORBLIND_PALETTE[i % len(COLORBLIND_PALETTE)]
        
        if len(run.episodes) > 0 and len(run.positions[room]) > 0:
            # Calculate moving average for smoother visualization
            window_size = min(20, len(run.positions[room]) // 5)
            if window_size > 1:
                # Use convolution for moving average
                kernel = np.ones(window_size) / window_size
                smoothed = np.convolve(run.positions[room], kernel, mode='valid')
                # Pad to match original length
                pad_size = len(run.positions[room]) - len(smoothed)
                smoothed = np.pad(smoothed, (0, pad_size), 'edge')
            else:
                smoothed = run.positions[room]
            
            # Plot raw data with low alpha
            plt.plot(run.episodes, run.positions[room], 'o', color=color, alpha=0.15, 
                    markersize=3, label=None)
            
            # Plot smoothed line
            plt.plot(run.episodes, smoothed, '-', color=color, linewidth=2, 
                    label=f"{run.label}")
            
            # Add horizontal line for mean
            plt.axhline(run.stats[room]['mean'], color=color, linestyle='--', alpha=0.5)
    
    # Set labels and title
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel(f'{room} Door Position', fontsize=12)
    
    # Add episode range to title if any run has filtered episodes
    filtered_runs = any(run.start_episode is not None or run.end_episode is not None for run in runs)
    if filtered_runs:
        # Find min and max episodes across all runs
        all_episodes = [ep for run in runs for ep in run.episodes]
        min_ep = min(all_episodes) if all_episodes else 0
        max_ep = max(all_episodes) if all_episodes else 0
        plt.title(f'{room} Door Position (Episodes {min_ep}-{max_ep})', fontsize=14)
    else:
        plt.title(f'{room} Door Position Over Episodes', fontsize=14)
    
    # Set y-axis limits to standard door range with a bit of padding
    plt.ylim(0.4, 3.6)
    
    # Add grid
    plt.grid(True, alpha=0.3, linestyle=':')
    
    # Add legend
    plt.legend(loc='best')
    
    # Add horizontal lines for discrete door positions
    for pos in door_positions:
        plt.axhline(pos, color='gray', linestyle='--', alpha=0.5)
        plt.text(plt.xlim()[0] - 5, pos, f"{pos}", va='center', ha='right', fontsize=9,
               bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.8))
    
    # Annotate standard positions if not already included
    standard_positions = {
        0.6: "Min",
        2.0: "Middle",
        3.4: "Max"
    }
    
    for pos, label in standard_positions.items():
        if pos not in door_positions:
            plt.axhline(pos, color='black', linestyle=':', alpha=0.3)
        plt.text(plt.xlim()[1] * 1.01, pos, label, va='center', fontsize=9)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()


def create_boxplot_grid(runs, output_path):
    """
    Create a grid of box plots for easy comparison across rooms with reduced whitespace
    and improved discrete position markers
    
    Args:
        runs: List of DoorPositionData objects
        output_path: Path to save the output file
    """
    # Check if we have any valid data to plot
    has_data = False
    for run in runs:
        if any(len(run.positions[room]) > 0 for room in ['roomA', 'roomB', 'roomC']):
            has_data = True
            break
    
    if not has_data:
        print("Warning: No data available for boxplot grid. Skipping this visualization.")
        plt.figure(figsize=(3, 4))
        plt.text(0.5, 0.5, "No data available for visualization", 
                ha='center', va='center', fontsize=10)
        plt.savefig(output_path, dpi=DPI)
        plt.close()
        return
    
    # Get number of discrete door positions from first run
    # Replace with more robust version that also respects min/max positions:
    num_door_positions = 5  # Default to 5
    door_min = 0.6  # Default min
    door_max = 3.4  # Default max

    for run in runs:
        if hasattr(run, 'config'):
            if 'discrete_door_positions' in run.config:
                num_door_positions = run.config['discrete_door_positions']
            if 'door_position_min' in run.config:
                door_min = run.config['door_position_min']
            if 'door_position_max' in run.config:
                door_max = run.config['door_position_max']
            break
    door_positions = []
    
    if num_door_positions > 1:
        # Create evenly spaced door positions
        for i in range(num_door_positions):
            position = door_min + i * ((door_max - door_min) / (num_door_positions - 1))
            door_positions.append(round(position, 2))
    else:
        door_positions = [2.0]  # Default middle position
        
    # Create figure with subplots - one row per run
    fig, axes = plt.subplots(len(runs), 1, figsize=(5, 4 * len(runs)), sharex=True)
    
    # Handle single run case
    if len(runs) == 1:
        axes = [axes]
    
    # For each run, create a box plot of all rooms
    for run_idx, (run, ax) in enumerate(zip(runs, axes)):
        # Prepare data for box plots
        data = []
        positions = []
        colors = []
        
        # Inside the data preparation loop:
        for room_idx, room in enumerate(['roomA', 'roomB', 'roomC']):
            if room in run.positions and len(run.positions[room]) > 0:
                # Filter out NaN values
                valid_positions = [pos for pos in run.positions[room] 
                                if not (isinstance(pos, float) and np.isnan(pos))]
                
                if valid_positions:  # Only add if we have valid data
                    data.append(valid_positions)
                    positions.append(room_idx)
                    colors.append(ROOM_COLORS[room])
        
        # Skip if no data for this run
        if not data or not positions:
            ax.text(0.5, 0.5, f"No data for {run.label}", 
                   ha='center', va='center', fontsize=10, transform=ax.transAxes)
            continue
            
        # Create box plot
        bp = ax.boxplot(data, positions=positions, patch_artist=True, 
                     widths=0.6, showfliers=False, showmeans=True)
        
        # Customize box plot appearance
        for i, box in enumerate(bp['boxes']):
            box.set(facecolor=colors[i], alpha=0.7, linewidth=1.5)
        
        for i, whisker in enumerate(bp['whiskers']):
            whisker.set(linewidth=1.5, color=colors[i//2])
        
        for i, cap in enumerate(bp['caps']):
            cap.set(linewidth=1.5, color=colors[i//2])
        
        for i, median in enumerate(bp['medians']):
            median.set(linewidth=2, color='black')
        
        for i, flier in enumerate(bp['fliers']):
            flier.set(marker='o', markerfacecolor=colors[i], alpha=0.5, 
                    markersize=4, markeredgecolor='none')
        
        for i, mean in enumerate(bp['means']):
            mean.set(marker='D', markerfacecolor='black', markersize=6)
        
        # Set title with terminal position info if available
        run_title = f"Door Positions - {run.label}"
        if hasattr(run, 'config') and 'terminal_location' in run.config:
            terminal_loc = run.config['terminal_location']
            if terminal_loc[0] < 4.0:
                run_title += " - Terminal Left"
            elif terminal_loc[0] > 8.0:
                run_title += " - Terminal Right"
            else:
                run_title += " - Terminal Center"
                
        # Add episode range if applicable
        if run.start_episode is not None or run.end_episode is not None:
            if run.episodes:
                run_title += f" (Episodes {run.episodes[0]}-{run.episodes[-1]})"
        
        ax.set_title(run_title, fontsize=12)
        ax.set_ylabel('Door Position', fontsize=10)
        
        # Set x-axis ticks and labels
        ax.set_xticks(positions)
        ax.set_xticklabels(['Room A', 'Room B', 'Room C'], fontsize=10)
        
        # Set y-axis limits to standard door range
        ax.set_ylim(0.4, 3.6)
        
        # Add horizontal lines for discrete door positions
        for pos in door_positions:
            ax.axhline(pos, color='gray', linestyle='--', alpha=0.5)
            ax.text(-0.5, pos, f"{pos}", va='center', fontsize=8)
            
        # Add horizontal lines for key positions if not already included
        for pos, label in [(0.6, "Min"), (2.0, "Middle"), (3.4, "Max")]:
            if pos not in door_positions:
                ax.axhline(pos, color='black', linestyle=':', alpha=0.3)
            ax.text(ax.get_xlim()[1] + 0.2, pos, label, va='center', fontsize=8)
        
        # Add grid
        ax.grid(axis='y', alpha=0.3, linestyle=':')
        
        # Add statistics directly on plot
        for i, room_data in enumerate(data):
            # Filter out NaN values again just to be sure
            valid_data = [pos for pos in room_data 
                        if not (isinstance(pos, float) and np.isnan(pos))]
            
            if valid_data:
                # Calculate statistics
                mean = np.mean(valid_data)
                median = np.median(valid_data)
                
                # Add text at bottom of plot
                ax.text(positions[i], 0.6, f"Mean: {mean:.2f}\nMedian: {median:.2f}", 
                    ha='center', va='bottom', fontsize=9, 
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    # Adjust layout with minimal padding
    plt.tight_layout()
    
    # Reduce spacing between subplots
    plt.subplots_adjust(hspace=0.3, top=0.95)
    
    # Save figure
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)

def create_segment_heatmap(runs, output_path, num_segments=5, bins=10):
    """
    Create a heatmap showing the frequency of door positions across training segments
    with improved discrete position markers
    
    Args:
        runs: List of DoorPositionData objects
        output_path: Path to save the output file
        num_segments: Number of segments to divide the data into
        bins: Number of bins for position histogram
    """
    # Check if we have any valid data
    has_data = False
    for run in runs:
        if run.episodes:
            has_data = True
            break
    
    if not has_data:
        print("Warning: No data available for segment heatmap. Skipping this visualization.")
        plt.figure(figsize=(10, 10))
        plt.text(0.5, 0.5, "No data available for visualization", 
                ha='center', va='center', fontsize=14)
        plt.savefig(output_path, dpi=DPI)
        plt.close()
        return
    
    # Get number of discrete door positions from first run
   # Replace with more robust version that also respects min/max positions:
    num_door_positions = 5  # Default to 5
    door_min = 0.6  # Default min
    door_max = 3.4  # Default max

    for run in runs:
        if hasattr(run, 'config'):
            if 'discrete_door_positions' in run.config:
                num_door_positions = run.config['discrete_door_positions']
            if 'door_position_min' in run.config:
                door_min = run.config['door_position_min']
            if 'door_position_max' in run.config:
                door_max = run.config['door_position_max']
            break
    door_positions = []
    
    if num_door_positions > 1:
        # Create evenly spaced door positions
        for i in range(num_door_positions):
            position = door_min + i * ((door_max - door_min) / (num_door_positions - 1))
            door_positions.append(round(position, 2))
    else:
        door_positions = [2.0]  # Default middle position
    
    # Create figure with subplots - one row per room
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    
    room_names = {'roomA': 'Room A', 'roomB': 'Room B', 'roomC': 'Room C'}
    rooms = ['roomA', 'roomB', 'roomC']
    
    for run in runs:
        # Create segments
        if not run.episodes:
            continue
            
        max_episode = max(run.episodes)
        step = max(100, max_episode // num_segments)
        
        # Define bin edges for door positions - make them match the discrete positions
        bin_edges = np.linspace(0.5, 3.5, bins+1)
        
        # Process each room
        for room_idx, room in enumerate(rooms):
            ax = axes[room_idx]
            
            # Create a 2D histogram: segments x positions
            heatmap_data = np.zeros((num_segments, bins))
            segment_labels = []
            
            # Process each segment
            for seg_idx in range(num_segments):
                start = seg_idx * step + 1
                end = min((seg_idx + 1) * step, max_episode)
                
                # Find episodes in this range
                indices = [i for i, ep in enumerate(run.episodes) 
                        if start <= ep <= end]
                
                # Add this inside the segment processing loop:
                if indices:
                    # Extract positions for this segment
                    seg_positions = [run.positions[room][i] for i in indices]
                    
                    # Filter out NaN values
                    valid_positions = [pos for pos in seg_positions 
                                    if not (isinstance(pos, float) and np.isnan(pos))]
                    
                    # Create histogram only if we have valid data
                    if valid_positions:
                        hist, _ = np.histogram(valid_positions, bins=bin_edges, density=True)
                        heatmap_data[seg_idx] = hist
            
            # Check if we have any data in the heatmap
            if np.any(heatmap_data):
                # Create heatmap
                im = ax.imshow(heatmap_data.T, aspect='auto', origin='lower', 
                            extent=[0, num_segments, 0.5, 3.5],
                            cmap='viridis', interpolation='nearest')
                
                # Add colorbar
                cbar = fig.colorbar(im, ax=ax)
                cbar.set_label('Normalized Frequency')
                
                # Set title and labels
                ax.set_title(f"{room_names[room]} Door Position Frequency Across Training", fontsize=12)
                ax.set_ylabel("Door Position", fontsize=10)
                
                # Add horizontal lines and labels for discrete door positions
                for pos in door_positions:
                    ax.axhline(pos, color='white', linestyle=':', alpha=0.8, linewidth=1.0)
                    ax.text(-0.5, pos, f"{pos}", va='center', fontsize=8, color='white',
                        bbox=dict(boxstyle='round', facecolor='black', alpha=0.6, pad=0.1))
                
                # Set y-ticks to match door positions
                ax.set_yticks(door_positions)
                ax.set_yticklabels([f"{y:.1f}" for y in door_positions])
                
                # Set x-ticks
                ax.set_xticks(np.arange(0.5, num_segments, 1))
                ax.set_xticklabels(segment_labels, rotation=30, ha='right', fontsize=9)
            else:
                # If no data for this room
                ax.text(0.5, 0.5, f"No data for {room_names[room]}", 
                    ha='center', va='center', fontsize=12, transform=ax.transAxes)
    
    # Add overall title
    if runs and runs[0].label:
        # Add terminal info if available
        terminal_info = ""
        if hasattr(runs[0], 'config') and 'terminal_location' in runs[0].config:
            terminal_loc = runs[0].config['terminal_location']
            if terminal_loc[0] < 4.0:
                terminal_info = "- Terminal Left"
            elif terminal_loc[0] > 8.0:
                terminal_info = "- Terminal Right"
            else:
                terminal_info = "- Terminal Center"
        
        plt.suptitle(f"Door Position Distribution Changes During Training - {runs[0].label} {terminal_info}", 
                    fontsize=14, y=0.98)
    else:
        plt.suptitle("Door Position Distribution Changes During Training", 
                    fontsize=14, y=0.98)
    
    # Set x-label for bottom subplot
    axes[-1].set_xlabel("Episode Range", fontsize=10)
    
    # Tight layout with minimal spacing
    plt.tight_layout(pad=0.5)
    plt.subplots_adjust(hspace=0.25, top=0.95)
    
    # Save figure
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)

def create_summary_table(runs, output_path):
    """
    Create a summary table image comparing door position statistics across runs
    
    Args:
        runs: List of DoorPositionData objects
        output_path: Path to save the output file
    """
    # Check if we have any valid data
    has_data = False
    for run in runs:
        if run.episodes:
            has_data = True
            break
    
    if not has_data:
        print("Warning: No data available for summary table. Skipping this visualization.")
        plt.figure(figsize=(12, 6))
        plt.text(0.5, 0.5, "No data available for summary table", 
                ha='center', va='center', fontsize=14)
        plt.savefig(output_path, dpi=DPI)
        plt.close()
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, len(runs) * 2 + 6))
    
    # Hide axes
    ax.axis('off')
    
    # Create table data
    column_labels = [
        'Run', 'Room', 'Mean', 'Median', 'Min', 'Max', 'Std Dev', 'Primary Mode'
    ]
    
    # Prepare table data
    table_data = []
    
    for run in runs:
        # Skip runs with no episodes
        if not run.episodes:
            continue
            
        # Add episode range info to run label if filtered
        run_label = run.label
        if run.start_episode is not None or run.end_episode is not None:
            if run.episodes:
                run_label += f" (Eps {run.episodes[0]}-{run.episodes[-1]})"
        
        for room in ['roomA', 'roomB', 'roomC']:
            if room in run.stats:
                # Get the primary mode if available
                primary_mode = "N/A"
                if room in run.distributions and run.distributions[room]['modes']:
                    primary_mode = f"{run.distributions[room]['modes'][0]['center']:.2f}"
                
                # Add row to table data
                table_data.append([
                    run_label,
                    room,
                    f"{run.stats[room]['mean']:.2f}",
                    f"{run.stats[room]['median']:.2f}",
                    f"{run.stats[room]['min']:.2f}",
                    f"{run.stats[room]['max']:.2f}",
                    f"{run.stats[room]['std']:.2f}",
                    primary_mode
                ])
    
    # If no data after filtering runs with episodes
    if not table_data:
        print("Warning: No data available for summary table after filtering. Skipping this visualization.")
        plt.text(0.5, 0.5, "No data available for summary table", 
                ha='center', va='center', fontsize=14)
        plt.savefig(output_path, dpi=DPI)
        plt.close()
        return
    
    # Create the table
    table = ax.table(
        cellText=table_data,
        colLabels=column_labels,
        loc='center',
        cellLoc='center',
        colColours=['#f2f2f2'] * len(column_labels),
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    
    # Color cells by room
    for i, row_data in enumerate(table_data):
        room = row_data[1]
        if room in ROOM_COLORS:
            color = ROOM_COLORS[room]
            # Lighten the color for better readability
            light_color = f"{color}20"  # Add 20% alpha
            
            # Color the room cell
            cell = table[i+1, 1]  # +1 for header row
            cell.set_facecolor(light_color)
    
    # Add title - include episode range if filtered
    title = 'Door Position Statistics Across Runs'
    filtered_runs = any(run.start_episode is not None or run.end_episode is not None for run in runs)
    if filtered_runs:
        all_episodes = [ep for run in runs for ep in run.episodes]
        if all_episodes:
            min_ep = min(all_episodes)
            max_ep = max(all_episodes)
            title += f' (Episodes {min_ep}-{max_ep})'
    plt.title(title, fontsize=16, pad=20)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)

def create_percentage_segment_comparison(runs, output_path, segment_pct=25):
    # Add parameter to docstring:
    """
    Args:
        runs: List of DoorPositionData objects
        output_path: Path to save the output file
        segment_pct: Percentage increment for segments (e.g., 25 for 0-25%, 25-50%, etc. or 10 for 0-10%, 10-20%, etc.)
    """
    # Check if we have any valid data
    has_data = False
    for run in runs:
        if run.episodes:
            has_data = True
            break
    
    if not has_data:
        print("Warning: No data available for percentage segment comparison. Skipping this visualization.")
        plt.figure(figsize=(5, 6.5))
        plt.text(0.5, 0.5, "No data available for visualization", 
                ha='center', va='center', fontsize=10)
        plt.savefig(output_path, dpi=DPI)
        plt.close()
        return
    
    # Create figure with subplots - one row per room
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    
    # Define room names for labeling
    room_names = {'roomA': 'Room A', 'roomB': 'Room B', 'roomC': 'Room C'}
    
    # Get number of discrete door positions from first run
    # Replace with more robust version that also respects min/max positions:
    num_door_positions = 5  # Default to 5
    door_min = 0.6  # Default min
    door_max = 3.4  # Default max

    for run in runs:
        if hasattr(run, 'config'):
            if 'discrete_door_positions' in run.config:
                num_door_positions = run.config['discrete_door_positions']
            if 'door_position_min' in run.config:
                door_min = run.config['door_position_min']
            if 'door_position_max' in run.config:
                door_max = run.config['door_position_max']
            break
    door_positions = []
    
    if num_door_positions > 1:
        # Create evenly spaced door positions
        for i in range(num_door_positions):
            position = door_min + i * ((door_max - door_min) / (num_door_positions - 1))
            door_positions.append(round(position, 2))
    else:
        door_positions = [2.0]  # Default middle position
    
    # Define percentage segments
    #segments = [(0, 25), (25, 50), (50, 75), (75, 100)]
    #segment_labels = ["0-25%", "25-50%", "50-75%", "75-100%"]
    segments = []
    segment_labels = []

    # Generate segments and labels based on segment_pct
    for start_pct in range(0, 100, segment_pct):
        end_pct = min(start_pct + segment_pct, 100)
        segments.append((start_pct, end_pct))
        segment_labels.append(f"{start_pct}-{end_pct}%")

    for run in runs:
        if not run.episodes:
            continue
        
        # Process each room
        for room_idx, room in enumerate(['roomA', 'roomB', 'roomC']):
            ax = axes[room_idx]
            
            if len(run.positions[room]) == 0:
                continue
                
            # Prepare data for boxplots
            data = []
            positions = []
            colors = []
            placement_rates = []  # Store placement rates for each segment
            segment_ep_ranges = []  # Store episode ranges for each segment
            
            # Get total number of episodes
            total_episodes = len(run.episodes)
            
            # Process each percentage segment
            for i, (start_pct, end_pct) in enumerate(segments):
                start_idx = int(total_episodes * start_pct / 100)
                end_idx = int(total_episodes * end_pct / 100)
                
                # Ensure we have at least one episode in this segment
                if start_idx == end_idx:
                    if end_idx < total_episodes:
                        end_idx += 1
                    elif start_idx > 0:
                        start_idx -= 1
                
                # Extract actual episode range
                if start_idx < total_episodes and end_idx <= total_episodes and start_idx < end_idx:
                    start_ep = run.episodes[start_idx]
                    end_ep = run.episodes[min(end_idx-1, total_episodes-1)]
                    
                    # Get all positions for this segment (including zeros)
                    segment_positions = run.positions[room][start_idx:end_idx]
                    
                    # Count non-zero positions
                    non_zero_count = sum(1 for pos in segment_positions if pos > 0.0)
                    
                    # Calculate placement rate for this segment
                    placement_rate = non_zero_count / len(segment_positions) if segment_positions else 0
                    placement_rates.append(placement_rate)
                    
                    # Filter out zero/NaN values for statistics
                    valid_positions = [pos for pos in segment_positions 
                                    if not (isinstance(pos, float) and np.isnan(pos)) and pos > 0.0]
                                        
                    if valid_positions:  # Only add if we have valid data
                        data.append(valid_positions)
                        positions.append(i)
                        colors.append(ROOM_COLORS[room])
                        segment_ep_ranges.append(f"({start_ep}-{end_ep})")
                        segment_labels[i] = f"{start_pct}-{end_pct}%\n({start_ep}-{end_ep})"
            
            # If we have data for this room, create the boxplot
            if data and positions:
                # Create boxplot
                bp = ax.boxplot(data, positions=positions, patch_artist=True,
                             widths=0.6, showfliers=False, showmeans=True)
                
                # Style the boxplots
                for i, box in enumerate(bp['boxes']):
                    box.set(facecolor=ROOM_COLORS[room], alpha=0.7, linewidth=1.5)
                
                for i, whisker in enumerate(bp['whiskers']):
                    whisker.set(linewidth=1.5, color=ROOM_COLORS[room])
                
                for i, cap in enumerate(bp['caps']):
                    cap.set(linewidth=1.5, color=ROOM_COLORS[room])
                
                for i, median in enumerate(bp['medians']):
                    median.set(linewidth=2, color='black')
                
                for i, mean in enumerate(bp['means']):
                    mean.set(marker='D', markerfacecolor='black', markersize=6)
                
                # Set title and labels
                ax.set_title(f"{room_names[room]} Door Positions Across Episodes", fontsize=12)
                ax.set_ylabel("Door Position", fontsize=10)
                
                # Set y-axis limits
                ax.set_ylim(0.4, 3.6)
                
                # Add horizontal lines for discrete door positions
                for pos in door_positions:
                    ax.axhline(pos, color='gray', linestyle='--', alpha=0.5)
                    ax.text(-0.5, pos, f"{pos}", va='center', fontsize=8)
                
                # Add standard reference lines if not already included
                for pos, label in [(0.6, "Min"), (2.0, "Middle"), (3.4, "Max")]:
                    if pos not in door_positions:
                        ax.axhline(pos, color='black', linestyle=':', alpha=0.3)
                    ax.text(ax.get_xlim()[1] + 0.3, pos, label, va='center', fontsize=8)
                
                # Add grid
                ax.grid(True, axis='y', alpha=0.3, linestyle=':')
                
                # Set x-ticks with percentage labels
                ax.set_xticks(positions)
                ax.set_xticklabels([segment_labels[i] for i in positions], rotation=0, fontsize=9)
                
                # Add mean values and placement rate for each segment
                for i, seg_data in enumerate(data):
                    mean = np.mean(seg_data)
                    median = np.median(seg_data)
                    
                    # Use the corresponding placement rate
                    placement_pct = placement_rates[i] * 100
                    
                    ax.text(positions[i], 0.5, 
                            f"Mean: {mean:.2f}\nMedian: {median:.2f}\nPlacement: {placement_pct:.0f}%", 
                            ha='center', va='bottom', fontsize=8,
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
            else:
                # If no data for this room
                ax.text(0.5, 0.5, f"No data for {room_names[room]}", 
                       ha='center', va='center', fontsize=10, transform=ax.transAxes)
    
    # Add overall title with terminal position info if available
    if runs and runs[0].label:
        terminal_info = ""
        if hasattr(runs[0], 'config') and 'terminal_location' in runs[0].config:
            terminal_loc = runs[0].config['terminal_location']
            if terminal_loc[0] < 4.0:
                terminal_info = " - Terminal Left"
            elif terminal_loc[0] > 8.0:
                terminal_info = " - Terminal Right"
            else:
                terminal_info = " - Terminal Center"
                
        plt.suptitle(f"Door Position Distributions Across Training Segments ({segment_pct}% increments) - {runs[0].label}{terminal_info}", 
            fontsize=14, y=0.98)
    else:
        plt.suptitle("Door Position Distributions Across Training Segments", 
                    fontsize=14, y=0.98)
    
    # Set x-label for bottom subplot
    axes[-1].set_xlabel("Episode Range", fontsize=11)
    
    # Adjust the layout to prevent overlapping
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, top=0.92)
    
    # Save figure
    plt.savefig(output_path, dpi=DPI)
    plt.close(fig)

def create_segment_comparison_boxplots(runs, output_path, num_segments=4):
    """
    Create boxplots showing door position distributions across different training segments
    with improved layout and discrete position markers
    
    Args:
        runs: List of DoorPositionData objects
        output_path: Path to save the output file
        num_segments: Number of segments to divide the data into
    """
    # Check if we have any valid data
    has_data = False
    for run in runs:
        if run.episodes:
            has_data = True
            break
    
    if not has_data:
        print("Warning: No data available for segment comparison boxplots. Skipping this visualization.")
        plt.figure(figsize=(4, 5))
        plt.text(0.5, 0.5, "No data available for visualization", 
                ha='center', va='center', fontsize=10)
        plt.savefig(output_path, dpi=DPI)
        plt.close()
        return
    
    # Create figure with subplots - one row per room
    fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
    
    room_names = {'roomA': 'Room A', 'roomB': 'Room B', 'roomC': 'Room C'}
    rooms = ['roomA', 'roomB', 'roomC']
    
    # Get number of discrete door positions from first run
    # Replace with more robust version that also respects min/max positions:
    num_door_positions = 5  # Default to 5
    door_min = 0.6  # Default min
    door_max = 3.4  # Default max

    for run in runs:
        if hasattr(run, 'config'):
            if 'discrete_door_positions' in run.config:
                num_door_positions = run.config['discrete_door_positions']
            if 'door_position_min' in run.config:
                door_min = run.config['door_position_min']
            if 'door_position_max' in run.config:
                door_max = run.config['door_position_max']
            break
    door_positions = []
    
    if num_door_positions > 1:
        # Create evenly spaced door positions
        for i in range(num_door_positions):
            position = door_min + i * ((door_max - door_min) / (num_door_positions - 1))
            door_positions.append(round(position, 2))
    else:
        door_positions = [2.0]  # Default middle position
    
    for run in runs:
        # Create segments
        if not run.episodes:
            continue
            
        max_episode = max(run.episodes)
        step = max(100, max_episode // num_segments)
        
        # Define segment ranges
        segment_ranges = []
        for i in range(0, max_episode, step):
            end = min(i + step, max_episode)
            segment_ranges.append((i+1, end))
            
        # Create data for each segment
        for room_idx, room in enumerate(rooms):
            ax = axes[room_idx]
            
            data = []
            positions = []
            colors = []
            labels = []
            
            # Add this inside the room data collection loop:
            for seg_idx, (start, end) in enumerate(segment_ranges):
                indices = [i for i, ep in enumerate(run.episodes) 
                        if start <= ep <= end]
                
                if indices:
                    # Extract and filter positions
                    seg_positions = [run.positions[room][i] for i in indices]
                    valid_positions = [pos for pos in seg_positions 
                                    if not (isinstance(pos, float) and np.isnan(pos))]
                    
                    if valid_positions:  # Only add if we have valid data
                        data.append(valid_positions)
                        positions.append(seg_idx)
                        colors.append(ROOM_COLORS[room])
                        labels.append(f"{start}-{end}")
            
            if data:
                # Create boxplot for this room with segment data
                bp = ax.boxplot(data, positions=positions, patch_artist=True,
                             widths=0.6, showfliers=False, showmeans=True)
                
                # Style the boxplots
                for i, box in enumerate(bp['boxes']):
                    # Use a consistent color for each room
                    box.set(facecolor=colors[0], alpha=0.7, linewidth=1.5)
                
                for i, whisker in enumerate(bp['whiskers']):
                    whisker.set(linewidth=1.5, color=colors[0])
                
                for i, cap in enumerate(bp['caps']):
                    cap.set(linewidth=1.5, color=colors[0])
                
                for i, median in enumerate(bp['medians']):
                    median.set(linewidth=2, color='black')
                
                for i, mean in enumerate(bp['means']):
                    mean.set(marker='D', markerfacecolor='black', markersize=6)
                
                # Set title and labels
                ax.set_title(f"{room_names[room]} Door Positions", fontsize=12)
                ax.set_ylabel("Door Position", fontsize=10)
                
                # Set y-axis limits with a bit more padding
                ax.set_ylim(0.5, 3.6)
                
                # Add horizontal lines for discrete door positions
                for pos in door_positions:
                    ax.axhline(pos, color='gray', linestyle='--', alpha=0.5)
                    ax.text(-0.5, pos, f"{pos}", va='center', fontsize=8)
                
                # Also add the standard reference lines
                for pos, label in [(0.6, "Min"), (2.0, "Middle"), (3.4, "Max")]:
                    if pos not in door_positions:
                        ax.axhline(pos, color='black', linestyle=':', alpha=0.3)
                    ax.text(ax.get_xlim()[1] + 0.3, pos, label, va='center', fontsize=8)
                
                # Add grid
                ax.grid(True, axis='y', alpha=0.3, linestyle=':')
                
                # Set x-ticks and better labels
                ax.set_xticks(positions)
                ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=10)
                
                # Add mean values for each segment
                for i, seg_data in enumerate(data):
                    mean = np.mean(seg_data)
                    ax.text(positions[i], 0.6, f"Mean: {mean:.2f}", 
                           ha='center', va='bottom', fontsize=9,
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
            else:
                # If no data for this room
                ax.text(0.5, 0.5, f"No data for {room_names[room]}", 
                       ha='center', va='center', fontsize=10, transform=ax.transAxes)
    
    # Add overall title
    if runs and runs[0].label:
        # Add terminal position info if available
        terminal_info = ""
        if hasattr(runs[0], 'config') and 'terminal_location' in runs[0].config:
            terminal_loc = runs[0].config['terminal_location']
            if terminal_loc[0] < 4.0:
                terminal_info = "- Terminal Left"
            elif terminal_loc[0] > 8.0:
                terminal_info = "- Terminal Right"
            else:
                terminal_info = "- Terminal Center"
                
        plt.suptitle(f"Door Position Distributions Across Training Segments - {runs[0].label} {terminal_info}", 
                    fontsize=14, y=0.98)
    else:
        plt.suptitle("Door Position Distributions Across Training Segments", 
                    fontsize=14, y=0.98)
    
    # Set x-label for bottom subplot
    axes[-1].set_xlabel("Episode Range", fontsize=11)
    
    # Adjust the layout to prevent overlapping
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, top=0.92)
    
    # Save figure
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)

def create_mode_analysis_plot(runs, output_path):
    """
    Create refined door position visualization with improved layout and position markers
    
    Args:
        runs: List of DoorPositionData objects
        output_path: Path to save the output file
    """
    # Check if we have any valid data
    has_data = False
    for run in runs:
        if run.episodes:
            has_data = True
            break
    
    if not has_data:
        print("Warning: No data available for mode analysis plot. Skipping this visualization.")
        plt.figure(figsize=(4, 4))
        plt.text(0.5, 0.5, "No data available for visualization", 
                ha='center', va='center', fontsize=10)
        plt.savefig(output_path, dpi=DPI)
        plt.close()
        return
    
    # Get number of discrete door positions from first run
    # Replace with more robust version that also respects min/max positions:
    num_door_positions = 5  # Default to 5
    door_min = 0.6  # Default min
    door_max = 3.4  # Default max

    for run in runs:
        if hasattr(run, 'config'):
            if 'discrete_door_positions' in run.config:
                num_door_positions = run.config['discrete_door_positions']
            if 'door_position_min' in run.config:
                door_min = run.config['door_position_min']
            if 'door_position_max' in run.config:
                door_max = run.config['door_position_max']
            break
    door_positions = []
    
    if num_door_positions > 1:
        # Create evenly spaced door positions
        for i in range(num_door_positions):
            position = door_min + i * ((door_max - door_min) / (num_door_positions - 1))
            door_positions.append(round(position, 2))
    
    # Set consistent fonts for academic publication
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 8,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
    })
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
    
    # Room labels for titles
    room_labels = {
        'roomA': 'Room A',
        'roomB': 'Room B',
        'roomC': 'Room C'
    }
    
    # Experiment explanation mappings
    exp_explanations = {
        'Run L': 'Terminal Left',
        'Run C': 'Terminal Center',
        'Run R': 'Terminal Right'
    }
    
    # Plot each room
    for i, (room, ax) in enumerate(zip(['roomA', 'roomB', 'roomC'], axes)):
        # Check if any run has data for this room
        room_has_data = False
        for run in runs:
            if run.episodes and room in run.distributions and 'modes' in run.distributions[room] and run.distributions[room]['modes']:
                room_has_data = True
                break
        
        if not room_has_data:
            ax.text(0.5, 0.5, f"No mode data for {room_labels[room]}", 
                   ha='center', va='center', fontsize=10, transform=ax.transAxes)
            continue
            
        # Clear previous bars for accurate positioning
        bar_positions = {}
        
        # First pass - determine x-positions for all experiments
        x_start = 0
        for j, run in enumerate(runs):
            if run.episodes:  # Only consider runs with episodes
                # Set position for this experiment
                bar_positions[run.label] = x_start
                x_start += 1.5  # Large spacing between experiments
        
        # Second pass - plot all bars with proper positioning
        for j, run in enumerate(runs):
            if not run.episodes:  # Skip runs without episodes
                continue
                
            x_pos = bar_positions.get(run.label)
            if x_pos is None:  # Skip if no position was assigned
                continue
                
            color = ROOM_COLORS[room]
            
            # Check if modes data exists for this room
            if room not in run.distributions or 'modes' not in run.distributions[room]:
                continue
                
            modes = run.distributions[room]['modes']
            if not modes:
                continue
            
            # Plot up to 3 modes for each experiment with decreasing alpha
            for k, mode in enumerate(modes[:3]):
                # Bar attributes
                width = 0.3  # Width of each bar
                alpha = 0.9 - k * 0.2  # Decreasing alpha for secondary/tertiary modes
                x = x_pos + k * (width + 0.05)  # Slight offset for each mode
                
                # Calculate percentage for display
                percentage = int(mode['frequency'] * 100)
                
                # Plot the bar
                bar = ax.bar(x, mode['frequency'], width=width, color=color, alpha=alpha,
                           edgecolor='black', linewidth=0.5)
                
                # Add door position value inside bar
                ax.text(x, mode['frequency']/2, f"{mode['center']:.2f}", 
                       ha='center', va='center', fontsize=9, fontweight='bold', 
                       color='white')
                
                # Add percentage at top of bar
                ax.text(x, mode['frequency'] + 0.01, f"{percentage}%", 
                       ha='center', va='bottom', fontsize=8)
        
        # Set title and labels with improved styling
        ax.set_title(f"{room_labels[room]} Door Position Distribution", fontsize=12)
        ax.set_ylabel('Proportion of Episodes', fontsize=9)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        
        # Set y-axis limit to 25% as requested
        ax.set_ylim(0, 0.25)  # Slightly higher to fit percentage labels
        
        # Add markers for discrete door positions
        if door_positions:
            for pos in door_positions:
                # Instead of using transparent axvline, we'll just add a text label
                # This avoids the color error while still achieving the visual outcome
                ax.text(-0.5, pos/3.4 * 0.25, f"{pos}", va='center', ha='right', fontsize=8,
                       bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.7))
        
        # Adjust x-ticks to center of each experiment group
        x_ticks = []
        x_labels = []
        for run in runs:
            if run.label in bar_positions:
                x_ticks.append(bar_positions[run.label] + 0.3)
                x_labels.append(run.label)
        
        if x_ticks:
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels, fontsize=9)
        
        # Add subtle background color for readability
        ax.set_facecolor('#f8f8f8')
        
        # Enhanced grid
        ax.grid(True, axis='y', alpha=0.3, linestyle='-', zorder=0)
        
        # Ensure proper x-axis limits with padding
        if bar_positions:
            x_min = min(bar_positions.values()) - 0.5
            x_max = max(bar_positions.values()) + 1.5
            ax.set_xlim(x_min, x_max)
    
    # Create comprehensive legend for top of figure
    # First for experiments
    exp_handles = []
    exp_labels = []
    
    # Add experiment explanations
    for run in runs:
        if run.episodes:  # Only include runs with episodes
            exp_handles.append(plt.Rectangle((0,0), 1, 1, color=ROOM_COLORS['roomA']))
            # Try to infer terminal position from label or config
            explanation = "Unknown"
            if hasattr(run, 'config') and 'terminal_location' in run.config:
                terminal_loc = run.config['terminal_location']
                if terminal_loc[0] < 4.0:
                    explanation = "Terminal Left"
                elif terminal_loc[0] > 8.0:
                    explanation = "Terminal Right"
                else:
                    explanation = "Terminal Center"
            else:
                explanation = exp_explanations.get(run.label, run.label)
            exp_labels.append(f"{run.label} = {explanation}")
    
    # Create the legend at the top - experiment explanations
    if exp_handles and exp_labels:
        exp_legend = fig.legend(exp_handles, exp_labels, 
                               loc='upper center', 
                               bbox_to_anchor=(0.5, 0.98), 
                               ncol=len(exp_handles),
                               frameon=True)
    
    # Add overall title - include episode range if filtered
    title = 'Door Position Preferences by Room and Experiment'
    filtered_runs = any(run.start_episode is not None or run.end_episode is not None for run in runs)
    if filtered_runs:
        all_episodes = [ep for run in runs for ep in run.episodes]
        if all_episodes:
            min_ep = min(all_episodes)
            max_ep = max(all_episodes)
            title += f' (Episodes {min_ep}-{max_ep})'
            
    fig.suptitle(title, fontsize=12, y=1.02, fontweight='bold')
    
    # Add explanation text
    explanation = (f"Values inside bars show door positions (1-{num_door_positions}), where: {door_min} = closest to room start, {door_max} = farthest\n"
                  "Bar height represents the proportion of episodes where the agent preferred this specific door position")
    
    # Add explanation box at bottom
    fig.text(0.5, 0.01, explanation, ha='center', va='center', fontsize=8,
           bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def create_boxplot_grid(runs, output_path):
    """
    Create a grid of box plots for easy comparison across rooms with reduced whitespace
    and improved discrete position markers
    
    Args:
        runs: List of DoorPositionData objects
        output_path: Path to save the output file
    """
    # Check if we have any valid data to plot
    has_data = False
    for run in runs:
        if any(len(run.positions[room]) > 0 for room in ['roomA', 'roomB', 'roomC']):
            has_data = True
            break
    
    if not has_data:
        print("Warning: No data available for boxplot grid. Skipping this visualization.")
        plt.figure(figsize=(3, 4))
        plt.text(0.5, 0.5, "No data available for visualization", 
                ha='center', va='center', fontsize=10)
        plt.savefig(output_path, dpi=DPI)
        plt.close()
        return
    
    # Get number of discrete door positions from first run
    num_door_positions = 5  # Default to 5
    door_min = 0.6
    door_max = 3.4
    
    if runs and hasattr(runs[0], 'config'):
        if 'discrete_door_positions' in runs[0].config:
            num_door_positions = runs[0].config['discrete_door_positions']
        if 'door_position_min' in runs[0].config:
            door_min = runs[0].config['door_position_min']
        if 'door_position_max' in runs[0].config:
            door_max = runs[0].config['door_position_max']
    
    # Calculate the actual door position values
    door_positions = []
    
    if num_door_positions > 1:
        # Create evenly spaced door positions
        for i in range(num_door_positions):
            position = door_min + i * ((door_max - door_min) / (num_door_positions - 1))
            door_positions.append(round(position, 2))
    else:
        door_positions = [2.0]  # Default middle position
    
    # Create figure with subplots - one row per run
    fig, axes = plt.subplots(len(runs), 1, figsize=(5, 4 * len(runs)), sharex=True)
    
    # Handle single run case
    if len(runs) == 1:
        axes = [axes]
    
    # For each run, create a box plot of all rooms
    for run_idx, (run, ax) in enumerate(zip(runs, axes)):
        print(f"Processing run {run_idx+1}/{len(runs)}: {run.label}")
        
        # Prepare data for box plots
        data = []
        positions = []
        colors = []
        labels = []
        
        # Inside the data preparation loop:
        for room_idx, room in enumerate(['roomA', 'roomB', 'roomC']):
            if room in run.positions and len(run.positions[room]) > 0:
                # Filter out NaN values and zeros if exclude_zeros is True
                valid_positions = []
                for pos in run.positions[room]:
                    # Skip NaN values
                    if isinstance(pos, float) and np.isnan(pos):
                        continue
                    # Skip zeros if exclude_zeros is True
                    if run.exclude_zeros and pos == 0.0:
                        continue
                    valid_positions.append(pos)
                
                print(f"  {room}: Found {len(valid_positions)} valid positions")
                
                if valid_positions:  # Only add if we have valid data
                    data.append(valid_positions)
                    positions.append(room_idx)
                    colors.append(ROOM_COLORS[room])
                    labels.append(room)
        
        # Skip if no data for this run
        if not data or not positions:
            ax.text(0.5, 0.5, f"No data for {run.label}", 
                   ha='center', va='center', fontsize=10, transform=ax.transAxes)
            continue
        
        print(f"  Creating boxplot with {len(data)} rooms of data")
        
        try:
            # Create box plot
            bp = ax.boxplot(data, positions=positions, patch_artist=True, 
                         widths=0.6, showfliers=False, showmeans=True)
            
            # Customize box plot appearance
            for i, box in enumerate(bp['boxes']):
                box.set(facecolor=colors[i], alpha=0.7, linewidth=1.5)
            
            for i, whisker in enumerate(bp['whiskers']):
                whisker.set(linewidth=1.5, color=colors[i//2])
            
            for i, cap in enumerate(bp['caps']):
                cap.set(linewidth=1.5, color=colors[i//2])
            
            for i, median in enumerate(bp['medians']):
                median.set(linewidth=2, color='black')
            
            for i, mean in enumerate(bp['means']):
                mean.set(marker='D', markerfacecolor='black', markersize=6)
            
            # Set title with terminal position info if available
            run_title = f"Door Positions - {run.label}"
            if hasattr(run, 'config') and 'terminal_location' in run.config:
                terminal_loc = run.config['terminal_location']
                if terminal_loc[0] < 4.0:
                    run_title += " - Terminal Left"
                elif terminal_loc[0] > 8.0:
                    run_title += " - Terminal Right"
                else:
                    run_title += " - Terminal Center"
                    
            # Add episode range if applicable
            if run.start_episode is not None or run.end_episode is not None:
                if run.episodes:
                    run_title += f" (Episodes {run.episodes[0]}-{run.episodes[-1]})"
            
            ax.set_title(run_title, fontsize=12)
            ax.set_ylabel('Door Position', fontsize=10)
            
            # Set x-axis ticks and labels
            ax.set_xticks(positions)
            ax.set_xticklabels(['Room A', 'Room B', 'Room C'], fontsize=10)
            
            # Set y-axis limits to standard door range with some padding
            ax.set_ylim(0.4, 3.6)
            
            # Add horizontal lines for discrete door positions
            for pos in door_positions:
                ax.axhline(pos, color='gray', linestyle='--', alpha=0.5)
                ax.text(-0.5, pos, f"{pos}", va='center', fontsize=8)
                
            # Add horizontal lines for key positions if not already included
            for pos, label in [(0.6, "Min"), (2.0, "Middle"), (3.4, "Max")]:
                if pos not in door_positions:
                    ax.axhline(pos, color='black', linestyle=':', alpha=0.3)
                ax.text(ax.get_xlim()[1] + 0.2, pos, label, va='center', fontsize=8)
            
            # Add grid
            ax.grid(axis='y', alpha=0.3, linestyle=':')
            
            # Add statistics directly on plot
            for i, room_data in enumerate(data):
                if room_data:  # Only calculate stats if we have data
                    # Calculate statistics
                    mean = np.mean(room_data)
                    median = np.median(room_data)
                    
                    # Add text at bottom of plot
                    ax.text(positions[i], 0.6, f"Mean: {mean:.2f}\nMedian: {median:.2f}", 
                        ha='center', va='bottom', fontsize=9, 
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
        
        except Exception as e:
            print(f"Error creating boxplot for {run.label}: {e}")
            ax.text(0.5, 0.5, f"Error creating boxplot: {str(e)}", 
                   ha='center', va='center', fontsize=10, transform=ax.transAxes)
    
    # Adjust layout with minimal padding
    plt.tight_layout()
    
    # Reduce spacing between subplots
    plt.subplots_adjust(hspace=0.3, top=0.95)
    
    # Save figure
    try:
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
        print(f"Successfully saved boxplot grid to {output_path}")
    except Exception as e:
        print(f"Error saving boxplot grid to {output_path}: {e}")
    finally:
        plt.close(fig)

def create_segment_trend_plot(runs, output_path):
    """
    Create a plot showing trends in door positions across segments
    with improved discrete position markers
    
    Args:
        runs: List of DoorPositionData objects
        output_path: Path to save the output file
    """
    # Determine number of runs
    n_runs = len(runs)
    
    if n_runs == 0:
        return
    
    # Get number of discrete door positions from first run with data
    num_door_positions = 5  # Default to 5
    for run in runs:
        if hasattr(run, 'config') and 'discrete_door_positions' in run.config:
            num_door_positions = run.config['discrete_door_positions']
            break
    
    # Calculate the actual door position values
    door_min = 0.6
    door_max = 3.4
    door_positions = []
    
    if num_door_positions > 1:
        # Create evenly spaced door positions
        for i in range(num_door_positions):
            position = door_min + i * ((door_max - door_min) / (num_door_positions - 1))
            door_positions.append(round(position, 2))
    else:
        door_positions = [2.0]  # Default middle position
    
    # Create figure with subplots for each room
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    
    # Plot each room
    for i, (room, ax) in enumerate(zip(['roomA', 'roomB', 'roomC'], axes)):
        # Plot each run
        for j, run in enumerate(runs):
            color = COLORBLIND_PALETTE[j % len(COLORBLIND_PALETTE)]
            
            # Extract segment means for this room
            x = []
            y = []
            
            # Add this inside the segment processing loop:
            for segment in run.segments:
                if room in segment['stats']:
                    # Use the midpoint of the episode range as x
                    mid_episode = (segment['start_episode'] + segment['end_episode']) / 2
                    mean_value = segment['stats'][room]['mean']
                    
                    # Only add points with valid mean values
                    if not (isinstance(mean_value, float) and np.isnan(mean_value)):
                        x.append(mid_episode)
                        y.append(mean_value)
            
            if x and y:
                # Plot line connecting segment means
                ax.plot(x, y, 'o-', color=color, linewidth=2, markersize=8,
                       label=f"{run.label}")
        
        # Set title and labels
        ax.set_title(f"{room} Position Trend", fontsize=12)
        ax.set_ylabel('Door Position', fontsize=10)
        
        # Set y-axis limits to standard door range
        ax.set_ylim(0.4, 3.6)
        
        # Add horizontal lines for discrete door positions
        for pos in door_positions:
            ax.axhline(pos, color='gray', linestyle='--', alpha=0.5)
            ax.text(ax.get_xlim()[0] - 100, pos, f"{pos}", va='center', ha='right', fontsize=8,
                  bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.7))
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle=':')
        
        # Add legend to first subplot only
        if i == 0:
            ax.legend(loc='best')
        
        # Annotate standard positions if not already included
        for pos, label in [(0.6, "Min"), (2.0, "Middle"), (3.4, "Max")]:
            if pos not in door_positions:
                ax.axhline(pos, color='black', linestyle=':', alpha=0.2)
            ax.text(ax.get_xlim()[1] * 1.01, pos, label, va='center', fontsize=8)
    
    # Set x-label on bottom subplot only
    axes[-1].set_xlabel('Episode', fontsize=12)
    
    # Add overall title
    fig.suptitle('Door Position Trends Across Training Segments', fontsize=14, y=0.98)
    
    # Ensure proper spacing
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)

def create_room_comparison(runs, output_path):
    """
    Create a comparison of door positions across rooms for each run with improved position markers
    
    Args:
        runs: List of DoorPositionData objects
        output_path: Path to save the output file
    """
    # Determine number of runs
    n_runs = len(runs)
    
    if n_runs == 0:
        return
    
    # Get number of discrete door positions from first run
    num_door_positions = 5  # Default to 5
    if runs and hasattr(runs[0], 'config') and 'discrete_door_positions' in runs[0].config:
        num_door_positions = runs[0].config['discrete_door_positions']
    
    # Calculate the actual door position values
    door_min = 0.6
    door_max = 3.4
    door_positions = []
    
    if num_door_positions > 1:
        # Create evenly spaced door positions
        for i in range(num_door_positions):
            position = door_min + i * ((door_max - door_min) / (num_door_positions - 1))
            door_positions.append(round(position, 2))
    else:
        door_positions = [2.0]  # Default middle position
    
    # Create figure with subplots for each run
    fig, axes = plt.subplots(n_runs, 1, figsize=(12, 5*n_runs), sharex=True)
    
    # Handle single run case
    if n_runs == 1:
        axes = [axes]
    
    # Plot each run
    for i, (run, ax) in enumerate(zip(runs, axes)):
        # Plot each room
        # Inside the room plotting loop:
        for room in ['roomA', 'roomB', 'roomC']:
            if len(run.episodes) > 0 and len(run.positions[room]) > 0:
                color = ROOM_COLORS[room]
                
                # Filter out NaN values
                valid_indices = []
                valid_positions = []
                
                for idx, pos in enumerate(run.positions[room]):
                    if not (isinstance(pos, float) and np.isnan(pos)):
                        valid_indices.append(idx)
                        valid_positions.append(pos)
                        
                if not valid_positions:
                    continue  # Skip if no valid data
                    
                # Get corresponding episodes for valid positions
                valid_episodes = [run.episodes[idx] for idx in valid_indices]
                
                # Calculate moving average for smoother visualization
                window_size = min(20, len(valid_positions) // 5) if len(valid_positions) > 5 else 1
                if window_size > 1:
                    # Use convolution for moving average
                    kernel = np.ones(window_size) / window_size
                    smoothed = np.convolve(valid_positions, kernel, mode='valid')
                    # Pad to match original length
                    pad_size = len(valid_positions) - len(smoothed)
                    smoothed = np.pad(smoothed, (0, pad_size), 'edge')
                else:
                    smoothed = valid_positions
                
                # Plot smoothed line
                ax.plot(valid_episodes, smoothed, '-', color=color, linewidth=2, 
                    label=f"{room}")
                
                # Add horizontal line for mean - use filtered mean
                ax.axhline(np.mean(valid_positions), color=color, linestyle='--', alpha=0.3)
        
        # Set title and labels
        run_title = f"Run: {run.label}"
        
        # Add terminal info if available
        if hasattr(run, 'config') and 'terminal_location' in run.config:
            terminal_loc = run.config['terminal_location']
            if terminal_loc[0] < 4.0:
                run_title += " - Terminal Left"
            elif terminal_loc[0] > 8.0:
                run_title += " - Terminal Right"
            else:
                run_title += " - Terminal Center"
        
        if run.start_episode is not None or run.end_episode is not None:
            run_title += f" (Episodes {run.episodes[0] if run.episodes else 'N/A'}-{run.episodes[-1] if run.episodes else 'N/A'})"
        ax.set_title(run_title, fontsize=14)
        ax.set_ylabel('Door Position', fontsize=12)
        
        # Set y-axis limits to standard door range
        ax.set_ylim(0.4, 3.6)
        
        # Add horizontal lines for discrete door positions
        for pos in door_positions:
            ax.axhline(pos, color='gray', linestyle='--', alpha=0.5)
            ax.text(ax.get_xlim()[0] - 5, pos, f"{pos}", va='center', ha='right', fontsize=8,
                  bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.7))
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle=':')
        
        # Add legend
        ax.legend(loc='upper right')
        
        # Annotate standard positions if not already included
        for pos, label in [(0.6, "Min"), (2.0, "Middle"), (3.4, "Max")]:
            if pos not in door_positions:
                ax.axhline(pos, color='black', linestyle=':', alpha=0.2)
            ax.text(ax.get_xlim()[1] * 1.01, pos, label, va='center', fontsize=8)
    
    # Set x-label on bottom subplot only
    axes[-1].set_xlabel('Episode', fontsize=12)
    
    # Add overall title
    fig.suptitle('Door Position Comparison Across Rooms', fontsize=16, y=0.98)
    
    # Ensure proper spacing
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save figure
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)

def create_distribution_plot(runs, output_path, room='roomA'):
    """
    Create a histogram of door position distributions
    with improved discrete position markers
    
    Args:
        runs: List of DoorPositionData objects
        output_path: Path to save the output file
        room: Room to visualize ('roomA', 'roomB', or 'roomC')
    """
    plt.figure(figsize=(10, 6))
    
    # Get number of discrete door positions from first run
    num_door_positions = 5  # Default to 5
    for run in runs:
        if hasattr(run, 'config') and 'discrete_door_positions' in run.config:
            num_door_positions = run.config['discrete_door_positions']
            break
    
    # Calculate the actual door position values
    door_min = 0.6
    door_max = 3.4
    door_positions = []
    
    if num_door_positions > 1:
        # Create evenly spaced door positions
        for i in range(num_door_positions):
            position = door_min + i * ((door_max - door_min) / (num_door_positions - 1))
            door_positions.append(round(position, 2))
    else:
        door_positions = [2.0]  # Default middle position
    
    # Create bins that align with door positions
    bins = []
    if len(door_positions) > 1:
        # Create bins centered on door positions
        bin_width = (door_max - door_min) / (num_door_positions - 1)
        bins = np.linspace(door_min - bin_width/2, door_max + bin_width/2, num_door_positions + 1)
    else:
        # Default bins if only one position or no positions
        bins = np.linspace(0.6, 3.4, 30)
    
    # Plot histogram for each run
    for i, run in enumerate(runs):
        color = COLORBLIND_PALETTE[i % len(COLORBLIND_PALETTE)]
        
        if len(run.positions[room]) > 0:
            # Filter out NaN values
            valid_positions = [pos for pos in run.positions[room] 
                             if not (isinstance(pos, float) and np.isnan(pos))]
            
            if not valid_positions:
                continue
                
            # Create label with zero handling info
            zero_info = "excl. zeros" if run.exclude_zeros else "incl. zeros"
            label = f"{run.label} ({zero_info})"
            if run.start_episode is not None or run.end_episode is not None:
                label += f" (Eps {run.episodes[0]}-{run.episodes[-1]})"
                
            # Plot histogram with valid_positions instead of run.positions[room]
            plt.hist(valid_positions, bins=bins, alpha=0.6, color=color, 
                     label=label, density=True)
            
            # Plot KDE for smoother visualization - IMPORTANT: Use valid_positions instead of run.positions[room]
            if len(valid_positions) > 10:
                try:
                    from scipy.stats import gaussian_kde
                    # This is the line that was failing - use valid_positions instead
                    density = gaussian_kde(valid_positions)
                    x = np.linspace(0.5, 3.5, 100)
                    plt.plot(x, density(x), color=color, linewidth=2)
                except Exception as e:
                    print(f"Warning: Could not create KDE for {run.label}, {room}: {e}")
            
            # Add vertical lines at the actual positions this run used
            for pos in door_positions:
                plt.axvline(x=pos, color=color, linestyle='--', alpha=0.3, linewidth=1)
    
    # Set labels and title
    plt.xlabel(f'{room} Door Position', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    
    # Add episode range to title if any run has filtered episodes
    filtered_runs = any(run.start_episode is not None or run.end_episode is not None for run in runs)
    room_label = {'roomA': 'Room A', 'roomB': 'Room B', 'roomC': 'Room C'}.get(room, room)
    
    # Add terminal info if available
    terminal_info = ""
    if runs and hasattr(runs[0], 'config') and 'terminal_location' in runs[0].config:
        terminal_loc = runs[0].config['terminal_location']
        if terminal_loc[0] < 4.0:
            terminal_info = "- Terminal Left"
        elif terminal_loc[0] > 8.0:
            terminal_info = "- Terminal Right"
        else:
            terminal_info = "- Terminal Center"
    
    if filtered_runs:
        # Find min and max episodes across all runs
        all_episodes = [ep for run in runs for ep in run.episodes]
        if all_episodes:
            min_ep = min(all_episodes)
            max_ep = max(all_episodes)
            plt.title(f'Distribution of {room_label} Door Positions {terminal_info} (Episodes {min_ep}-{max_ep})', fontsize=14)
        else:
            plt.title(f'Distribution of {room_label} Door Positions {terminal_info} (Filtered)', fontsize=14)
    else:
        plt.title(f'Distribution of {room_label} Door Positions {terminal_info}', fontsize=14)
    
    # Set x-axis limits to standard door range
    plt.xlim(0.5, 3.5)
    
    # Add grid
    plt.grid(True, alpha=0.3, linestyle=':')
    
    # Add vertical lines for discrete door positions with text labels
    for pos in door_positions:
        plt.axvline(pos, color='black', linestyle='-', alpha=0.2, linewidth=1)
        plt.text(pos, plt.ylim()[1] * 0.95, f"{pos}", ha='center', fontsize=9, 
                bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.8))
    
    # Add legend
    plt.legend(loc='best')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()

def create_horizontal_percentage_segment_comparison(runs, output_path, segment_pct=25):
    # Add parameter to docstring
    """
    Create a horizontal version of percentage-based segment comparison 
    with correct y-axis order and spacing
    
    Args:
        runs: List of DoorPositionData objects
        output_path: Path to save the output file
    """
    # Check if we have any valid data
    has_data = False
    for run in runs:
        if run.episodes:
            has_data = True
            break
    
    if not has_data:
        print("Warning: No data available for horizontal percentage segment comparison. Skipping this visualization.")
        plt.figure(figsize=(7, 4))
        plt.text(0.5, 0.5, "No data available for visualization", 
                ha='center', va='center', fontsize=10)
        plt.savefig(output_path, dpi=DPI)
        plt.close()
        return
    
    # Create figure with adjusted size
    fig, ax = plt.subplots(1, 1, figsize=(9, 8))
    
    # Define percentage segments in order 0-25% to 75-100%
    #segments = [(0, 25), (25, 50), (50, 75), (75, 100)]
    #segment_labels = ["0-25%", "25-50%", "50-75%", "75-100%"]
    segments = []
    segment_labels = []

    # Generate segments and labels based on segment_pct
    for start_pct in range(0, 100, segment_pct):
        end_pct = min(start_pct + segment_pct, 100)
        segments.append((start_pct, end_pct))
        segment_labels.append(f"{start_pct}-{end_pct}%")
    # Room data
    rooms = ['roomA', 'roomB', 'roomC']
    room_names = {'roomA': 'Room A', 'roomB': 'Room B', 'roomC': 'Room C'}
    
    # Create room legend
    legend_elements = []
    for room, label in room_names.items():
        rect = plt.Rectangle((0, 0), 1, 1, facecolor=ROOM_COLORS[room], 
                           edgecolor='black', linewidth=1)
        legend_elements.append(rect)
    
    # Add room legend below title with more space
    legend = ax.legend(legend_elements, list(room_names.values()),
                      loc='upper center', 
                      bbox_to_anchor=(0.5, 1.063),
                      ncol=3,
                      frameon=True,
                      fontsize=12)
    
    # Define y positions starting from top (0-25% at y=3)
    y_positions = []
    y_labels = []
    
    for run in runs:
        if not run.episodes:
            continue
        
        # Get total number of episodes
        total_episodes = len(run.episodes)
        
        # Process each percentage segment with reversed y positions
        for seg_idx, (start_pct, end_pct) in enumerate(segments):
            # Calculate episode range for this segment
            start_idx = int(total_episodes * start_pct / 100)
            end_idx = int(total_episodes * end_pct / 100)
            
            # Ensure we have at least one episode in this segment
            if start_idx == end_idx:
                if end_idx < total_episodes:
                    end_idx += 1
                elif start_idx > 0:
                    start_idx -= 1
            
            if start_idx < total_episodes and end_idx <= total_episodes and start_idx < end_idx:
                start_ep = run.episodes[start_idx]
                end_ep = run.episodes[min(end_idx-1, total_episodes-1)]
                
                # Process each room for this segment
                data = []
                positions = []
                colors = []
                
                for room_idx, room in enumerate(rooms):
                    if len(run.positions[room]) > 0:
                        # Extract positions for this segment
                        # Add this inside the room processing loop:
                        segment_positions = run.positions[room][start_idx:end_idx]
                        valid_positions = [pos for pos in segment_positions 
                                        if not (isinstance(pos, float) and np.isnan(pos))]
                                        
                        if valid_positions:
                            data.append(valid_positions)
                            # Position rooms horizontally
                            y_pos = (3 - seg_idx) + room_idx * 0.2 - 0.2  
                            positions.append(y_pos)
                            colors.append(ROOM_COLORS[room])
                
                # Create boxplots for this segment
                if data and positions:
                    bp = ax.boxplot(data, vert=False, positions=positions, 
                                  patch_artist=True, widths=0.15,
                                  showfliers=False, showmeans=True)
                    
                    # Style the boxplots
                    for i, box in enumerate(bp['boxes']):
                        box.set(facecolor=colors[i], alpha=0.7, linewidth=1.5)
                    
                    for i, whisker in enumerate(bp['whiskers']):
                        whisker.set(linewidth=1.5, color=colors[i//2])
                    
                    for i, cap in enumerate(bp['caps']):
                        cap.set(linewidth=1.5, color=colors[i//2])
                    
                    for i, median in enumerate(bp['medians']):
                        median.set(linewidth=2, color='black')
                    
                    for i, mean in enumerate(bp['means']):
                        mean.set(marker='D', markerfacecolor='black', markersize=6)
                
                # Store y position and label for this segment
                y_pos = 3 - seg_idx  # Reverse positioning for labels
                y_positions.append(y_pos)
                y_labels.append(f"{segment_labels[seg_idx]}\n({start_ep}-{end_ep})")
    
    # Set title with adequate padding
    ax.set_title(f"Door Position Distributions Across Training Stages - {run.label}", 
                 fontsize=14, pad=30)
    
    # Set main plot labels
    ax.set_xlabel("Door Positions - Range 0.6 - 3.4", fontsize=12)
    ax.set_ylabel("Training Stages", fontsize=12)
    
    # Set y-axis ticks and labels
    ax.set_yticks(sorted(y_positions, reverse=True))
    ax.set_yticklabels([y_labels[i] for i in range(len(y_labels))], fontsize=11)
    
    # Set x-axis ticks and labels to include 0 and 4
    exact_door_positions = [0.0, 0.6, 1.3, 2.0, 2.7, 3.4, 4.0]
    ax.set_xticks(exact_door_positions)
    ax.set_xticklabels([f"{pos:.1f}" for pos in exact_door_positions])
    
    # Add vertical lines for exact door positions (only the 5 valid positions)
    for pos in [0.6, 1.3, 2.0, 2.7, 3.4]:
        ax.axvline(pos, color='black', linestyle='--', alpha=0.3)
    
    # Add horizontal separators between segments
    for i in range(1, len(segments)):
        ax.axhline(3.5 - i, color='black', linestyle='-', alpha=0.3, linewidth=1)
    
    # Add grid only for x-axis
    ax.grid(True, axis='x', alpha=0.3, linestyle=':')
    
    # Set x-axis limits to include 0 and 4
    ax.set_xlim(0, 4)
    
    # Set y-axis limits with proper padding
    ax.set_ylim(-0.5, 3.5)
    
    # Adjust layout to prevent overlap
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)

def create_door_frequency_plot(runs, output_path, segment_pct=25):
    """
    Create a compact visualization showing the frequency of only the actually used door positions
    with colorblind-friendly colors
    
    Args:
        runs: List of DoorPositionData objects
        output_path: Path to save the output file
        segment_pct: Percentage increment for segments
    """
    # Check if we have any valid data
    has_data = False
    for run in runs:
        if run.episodes:
            has_data = True
            break
    
    if not has_data:
        print("Warning: No data available for door frequency plot. Skipping this visualization.")
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, "No data available for visualization", 
                ha='center', va='center', fontsize=10)
        plt.savefig(output_path, dpi=DPI)
        plt.close()
        return
    
    # Define colorblind-friendly colors for rooms
    # Blue, Orange, and Teal are distinguishable for most types of color blindness
    COLORBLIND_ROOM_COLORS = {
        'roomA': "#696868",  # Blue
        'roomB': "#696868",  # Orange
        'roomC': "#696868"   # Light Blue/Teal
    }

    # gray: #696868
    # #B7AEE0
    # #DC267F
    # #C185A2
    # Calculate the actual door position values
    for run in runs:
        if not run.episodes:
            continue
            
        # Find actually used door positions across all rooms
        used_positions = set()
        for room in ['roomA', 'roomB', 'roomC']:
            if not run.positions[room]:
                continue
                
            # Get all non-NaN positions
            valid_positions = [pos for pos in run.positions[room] 
                             if not (isinstance(pos, float) and np.isnan(pos))]
            
            # Round to nearest 0.1 to account for floating point imprecision
            rounded_positions = [round(pos * 10) / 10 for pos in valid_positions]
            
            # Find the unique positions actually used, with some tolerance
            # Group positions that are very close to standard values
            for pos in rounded_positions:
                # Map positions to standard values
                if 0.5 <= pos <= 0.7:
                    used_positions.add(0.6)
                elif 1.7 <= pos <= 1.9:
                    used_positions.add(1.8)
                elif 2.9 <= pos <= 3.1:
                    used_positions.add(3.0)
                elif 4.1 <= pos <= 4.3:
                    used_positions.add(4.2)
                elif 5.3 <= pos <= 5.5:
                    used_positions.add(5.4)
        
        # Convert to sorted list
        door_positions = sorted(used_positions)
        
        if not door_positions:
            print(f"Warning: No valid door positions found for {run.label}")
            continue
        
        # Define percentage segments
        segments = []
        segment_labels = []
        for start_pct in range(0, 100, segment_pct):
            end_pct = min(start_pct + segment_pct, 100)
            segments.append((start_pct, end_pct))
            segment_labels.append(f"{start_pct}-{end_pct}%")
        
        # Get episode ranges for each segment
        total_episodes = len(run.episodes)
        episode_ranges = []
        for start_pct, end_pct in segments:
            start_idx = int(total_episodes * start_pct / 100)
            end_idx = int(total_episodes * end_pct / 100)
            
            # Ensure valid indices
            if start_idx == end_idx:
                if end_idx < total_episodes:
                    end_idx += 1
                elif start_idx > 0:
                    start_idx -= 1
            
            if start_idx < total_episodes and end_idx <= total_episodes and start_idx < end_idx:
                start_ep = run.episodes[start_idx]
                end_ep = run.episodes[min(end_idx-1, total_episodes-1)]
                episode_ranges.append((start_idx, end_idx, f"{start_ep}-{end_ep}"))
        
        # Create more compact figure with 3 subplots (one per room)
        height_per_position = 0.15
        fig_width = 7
        fig_height = max(5, (len(door_positions) * height_per_position + 1) * 3)
        
        # Create figure with increased top margin
        fig, axes = plt.subplots(3, 1, figsize=(fig_width, fig_height))
        
        rooms = ['roomA', 'roomB', 'roomC']
        room_labels = {'roomA': 'Room A', 'roomB': 'Room B', 'roomC': 'Room C'}
        
        # Calculate number of segments for proper x-axis scaling
        num_segments = len(episode_ranges)
        
        # Process each room
        for room_idx, room in enumerate(rooms):
            ax = axes[room_idx]
            
            if len(run.positions[room]) == 0:
                ax.text(0.5, 0.5, f"No data for {room_labels[room]}", 
                       ha='center', va='center', fontsize=8, transform=ax.transAxes)
                continue
            
            # Remove vertical grid lines
            ax.grid(axis='y', alpha=0.2, linestyle='-')
            ax.grid(axis='x', alpha=0, linestyle='none')
            
            # Process each segment
            for seg_idx, (start_idx, end_idx, ep_range) in enumerate(episode_ranges):
                # Get positions for this segment
                segment_positions = run.positions[room][start_idx:end_idx]
                
                # Count occurrences of each door position
                counts = {}
                total_positions = 0
                
                # Filter out NaN values
                valid_positions = [pos for pos in segment_positions 
                                if not (isinstance(pos, float) and np.isnan(pos))]
                
                if not valid_positions:
                    continue
                
                # Count occurrences with tolerance
                for pos in valid_positions:
                    # Map to standard positions
                    rounded_pos = round(pos * 10) / 10  # Round to nearest 0.1
                    
                    mapped_pos = None
                    if 0.5 <= rounded_pos <= 0.7:
                        mapped_pos = 0.6
                    elif 1.7 <= rounded_pos <= 1.9:
                        mapped_pos = 1.8
                    elif 2.9 <= rounded_pos <= 3.1:
                        mapped_pos = 3.0
                    elif 4.1 <= rounded_pos <= 4.3:
                        mapped_pos = 4.2
                    elif 5.3 <= rounded_pos <= 5.5:
                        mapped_pos = 5.4
                    
                    if mapped_pos in door_positions:
                        counts[mapped_pos] = counts.get(mapped_pos, 0) + 1
                        total_positions += 1
                
                # Skip if no valid positions were found
                if total_positions == 0:
                    continue
                
                # Calculate percentages and display them
                x_offset = seg_idx
                
                # Map door positions to y-scale
                y_positions = {}
                for i, door_pos in enumerate(door_positions):
                    y_positions[door_pos] = i * 0.5
                
                for door_pos in door_positions:
                    # Get compressed y position
                    y_pos = y_positions[door_pos]
                    
                    count = counts.get(door_pos, 0)
                    percentage = (count / total_positions * 100)
                    
                    # Skip drawing if percentage is very small
                    if percentage < 1:
                        continue
                    
                    # Determine color based on percentage (higher = more intense)
                    alpha = min(0.2 + (percentage / 100) * 0.8, 0.9)
                    
                    # Get colorblind-friendly color for this room
                    color = COLORBLIND_ROOM_COLORS[room]
                    
                    # Set box dimensions
                    width = 0.75
                    height = 0.45
                    
                    # Draw rectangle
                    rect = plt.Rectangle((x_offset - width/2, y_pos - height/2), width, height,
                                       facecolor=color, alpha=alpha,
                                       edgecolor='black', linewidth=0.5)
                    ax.add_patch(rect)
                    
                    # Add text with percentage
                    ax.text(x_offset, y_pos, f"{percentage:.0f}%", 
                          ha='center', va='center', fontsize=8, fontweight='bold',
                          color='black' if alpha < 0.6 else 'white')
            
            # Set y-ticks and labels
            ax.set_yticks([y_positions[pos] for pos in door_positions])
            ax.set_yticklabels([f"{pos}" for pos in door_positions], fontsize=7)
            
            # Set y-axis limits
            max_y_pos = max(y_positions.values()) if y_positions else 0
            ax.set_ylim(-0.3, max_y_pos + 0.3)
            
            # Set title
            ax.set_title(f"{room_labels[room]} - Door Position Frequencies", fontsize=8)
            
            # Set y label
            ax.set_ylabel("Door Position", fontsize=8)
            
            # Hide x-ticks for non-bottom plots
            if room_idx < 2:
                ax.set_xticklabels([])
                ax.set_xticks([])
                
            # Set x limits
            ax.set_xlim(-0.5, num_segments - 0.5)
        
        # Set x-ticks with episode ranges
        ax = axes[-1]
        ax.set_xticks(range(len(episode_ranges)))
        ax.set_xticklabels([ep_range for _, _, ep_range in episode_ranges], 
                          rotation=0, va='center', ha='center', fontsize=7)
        ax.set_xlabel("Episode Range", fontsize=8)
        
        # Add overall title with segment info
        terminal_info = ""
        if hasattr(run, 'config') and 'terminal_location' in run.config:
            terminal_loc = run.config['terminal_location']
            if terminal_loc[0] < 4.0:
                terminal_info = " - Terminal Left"
            elif terminal_loc[0] > 8.0:
                terminal_info = " - Terminal Right"
            else:
                terminal_info = " - Terminal Center"
                
        #plt.suptitle(f"Door Position Frequency ({segment_pct}% segments) - {run.label}{terminal_info}", 
        #            fontsize=11, y=0.99)
        
        plt.suptitle(f"Door Position Frequency ({segment_pct}% segments) - Door Position Range [0.6 - 5.4]", 
                    fontsize=9, y=0.99)
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.4, top=0.9)
        
        # Save figure
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
        plt.close(fig)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze and visualize door positions from RL training runs.')
    
    parser.add_argument('files', nargs='+', help='Path to door position data files (txt format)')
    parser.add_argument('--labels', '-l', nargs='+', help='Labels for each run (same order as files)')
    parser.add_argument('--output-dir', '-o', default=OUTPUT_DIR, help='Output directory for figures')
    parser.add_argument('--no-timestamp', action='store_true', help='Disable timestamped subdirectory creation')
    parser.add_argument('--comparison-name', default=None, help='Custom name for the comparison (used in directory name)')
    parser.add_argument('--start-episode', type=int, default=None, help='First episode to include in analysis (inclusive)')
    parser.add_argument('--end-episode', type=int, default=None, help='Last episode to include in analysis (inclusive)')
    parser.add_argument('--word-optimized', action='store_true', help='Optimize figures for embedding in Word documents')
    parser.add_argument('--figure-width', type=float, default=6.5, help='Width of figures in inches (for Word compatibility)')
    parser.add_argument('--figure-dpi', type=int, default=300, help='DPI for output figures')
    parser.add_argument('--exclude-zeros', action='store_true', 
                        help='Exclude 0.0 values (treat them as missing data)')
    parser.add_argument('--segment-pct', type=int, default=25, 
                    help='Percentage increment for segments (e.g., 25 for quarters, 10 for tenths)')
    
    args = parser.parse_args()
    
    # Update global DPI setting if specified
    global DPI
    if args.figure_dpi:
        DPI = args.figure_dpi
    
    # Check if files exist
    for file_path in args.files:
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            return
    
    # Check if labels are provided for all files
    if args.labels and len(args.labels) != len(args.files):
        print(f"Error: Number of labels ({len(args.labels)}) doesn't match number of files ({len(args.files)})")
        return
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create base output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine the final output directory path
    if args.no_timestamp:
        output_dir = args.output_dir
    else:
        # Create a subdirectory with timestamp and optional name
        if args.comparison_name:
            subdir_name = f"{timestamp}_{args.comparison_name}"
        else:
            # Extract meaningful parts from experiment filenames
            file_ids = []
            for file_path in args.files:
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                
                # Handle run_exp__XX patterns
                if 'run_exp__' in base_name:
                    # Extract experiment number (e.g., "11" from "run_exp__11_l_mt_...")
                    parts = base_name.split('_')
                    for part in parts:
                        if part.startswith('exp__'):
                            # Found the experiment identifier
                            exp_id = part.replace('exp__', '')
                            file_ids.append(f"exp{exp_id}")
                            break
                    else:
                        # If we didn't find a clear exp number, use first few characters
                        file_ids.append(base_name[:8])
                else:
                    # For other naming patterns, just use first 5 characters
                    file_ids.append(base_name[:5])
            
            # Create comparison identifier
            comparison_id = "_vs_".join(file_ids)
            
            # Limit overall length
            if len(comparison_id) > 50:
                comparison_id = comparison_id[:47] + "..."
                
            subdir_name = f"{timestamp}_{comparison_id}"
        
        # Add episode range to directory name if specified
        if args.start_episode is not None or args.end_episode is not None:
            episode_range = f"eps{args.start_episode or 'start'}-{args.end_episode or 'end'}"
            subdir_name += f"_{episode_range}"
        
        if args.exclude_zeros:
            subdir_name += "_no_zeros"
            
        output_dir = os.path.join(args.output_dir, subdir_name)
        os.makedirs(output_dir, exist_ok=True)
    
    # Create a README.txt file with information about the comparison
    readme_path = os.path.join(output_dir, "README.txt")
    with open(readme_path, "w") as f:
        f.write(f"Door Position Analysis\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Add episode range if specified
        if args.start_episode is not None or args.end_episode is not None:
            episode_range = f"Episodes: {args.start_episode or 'start'} to {args.end_episode or 'end'}\n\n"
            f.write(episode_range)

        if args.exclude_zeros:
            f.write("Zero values: Excluded (treated as missing data)\n\n")
        else:
            f.write("Zero values: Included\n\n")

        f.write("Files compared:\n")
        for i, file_path in enumerate(args.files):
            label = args.labels[i] if args.labels and i < len(args.labels) else os.path.basename(file_path)
            f.write(f"  {i+1}. {label}: {file_path}\n")
        
        f.write("\nPlots generated:\n")
        f.write("  - timeline_roomX: Door position over episodes for each room\n")
        f.write("  - room_comparison: Side-by-side comparison of all rooms\n")
        f.write("  - distribution_roomX: Distribution of door positions for each room\n")
        f.write("  - percentage_segment_comparison: Door positions across 0-25%, 25-50%, 50-75%, 75-100% of episodes\n")
        f.write("  - segment_boxplots: Door positions across sequential segments\n")
        f.write("  - segment_heatmap: Heatmap of door position frequencies across segments\n")
    
    # Make a backup copy of this script in the output directory
    try:
        script_path = os.path.abspath(__file__)
        script_backup = os.path.join(output_dir, "enhanced_visdoor.py")
        shutil.copy2(script_path, script_backup)
        print(f"Created script backup in output directory")
    except Exception as e:
        print(f"Note: Could not create script backup: {e}")
    
    # Load and process data for each run
    runs = []
    valid_runs = []  # Keep track of runs with actual data
    
    for i, file_path in enumerate(args.files):
        label = args.labels[i] if args.labels and i < len(args.labels) else None
        
        try:
            run = DoorPositionData(file_path, label, args.start_episode, args.end_episode, args.exclude_zeros)
            runs.append(run)
            run.print_statistics()
            
            # Add to valid runs if it has episodes
            if run.episodes:
                valid_runs.append(run)
            else:
                print(f"Warning: No episodes found in {file_path} after filtering.")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            import traceback
            traceback.print_exc()
    
    # If no valid runs with episodes, show warning but continue
    if not valid_runs:
        print("\nWarning: No runs with episodes found. Some visualizations will be skipped or empty.")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Set figure width for Word compatibility if requested
    fig_width = args.figure_width if args.word_optimized else 12
    
    # Generate timeline plots for each room
    for room in ['roomA', 'roomB', 'roomC']:
        output_path = os.path.join(output_dir, f"timeline_{room}.png")
        create_position_timeline(runs, output_path, room)
        print(f"Created timeline plot for {room}")
    
    # Generate room comparison plot
    output_path = os.path.join(output_dir, "room_comparison.png")
    create_room_comparison(runs, output_path)
    print("Created room comparison plot")
    
    # Generate distribution plots for each room
    for room in ['roomA', 'roomB', 'roomC']:
        output_path = os.path.join(output_dir, f"distribution_{room}.png")
        create_distribution_plot(runs, output_path, room)
        print(f"Created distribution plot for {room}")
    
    """# Generate percentage-based segment comparison (0-25%, 25-50%, etc.)
    for run in runs:
        output_path = os.path.join(output_dir, f"percentage_segments_{run.label}.png")
        create_percentage_segment_comparison([run], output_path)
        print(f"Created percentage segment comparison for {run.label}")
    
    # Generate horizontal percentage-based segment comparison
    for run in runs:
        output_path = os.path.join(output_dir, f"horizontal_percentage_segments_{run.label}.png")
        create_horizontal_percentage_segment_comparison([run], output_path)
        print(f"Created horizontal percentage segment comparison for {run.label}")"""
    # Generate percentage-based segment comparison (with configurable increment)
    for run in runs:
        output_path = os.path.join(output_dir, f"percentage_segments_{run.label}.png")
        create_percentage_segment_comparison([run], output_path, args.segment_pct)
        print(f"Created percentage segment comparison for {run.label}")

    # Generate horizontal percentage-based segment comparison
    for run in runs:
        output_path = os.path.join(output_dir, f"horizontal_percentage_segments_{run.label}.png")
        create_horizontal_percentage_segment_comparison([run], output_path, args.segment_pct)
        print(f"Created horizontal percentage segment comparison for {run.label}")
    
    # Generate segment boxplots for each run
    for run in runs:
        output_path = os.path.join(output_dir, f"segment_boxplots_{run.label}.png")
        create_segment_comparison_boxplots([run], output_path, num_segments=4)
        print(f"Created segment boxplot comparison for {run.label}")
    
    # Generate segment heatmaps for each run
    for run in runs:
        output_path = os.path.join(output_dir, f"segment_heatmap_{run.label}.png")
        create_segment_heatmap([run], output_path, num_segments=5, bins=15)
        print(f"Created segment heatmap for {run.label}")

    for run in runs:
        output_path = os.path.join(output_dir, f"door_frequency_{run.label}.png")
        create_door_frequency_plot([run], output_path, args.segment_pct)
        print(f"Created door position frequency plot for {run.label}")
    
    # Generate mode analysis plot
    output_path = os.path.join(output_dir, "mode_analysis.png")
    create_mode_analysis_plot(runs, output_path)
    print("Created mode analysis plot")

    output_path = os.path.join(output_dir, "boxplot_grid.png")
    create_boxplot_grid(runs, output_path)
    print("Created box plot grid")
    
    # Generate segment trend plot
    output_path = os.path.join(output_dir, "segment_trends.png")
    create_segment_trend_plot(runs, output_path)
    print("Created segment trend plot")
    
    # Generate summary table
    output_path = os.path.join(output_dir, "summary_table.png")
    create_summary_table(runs, output_path)
    print("Created summary table")

    print(f"\nAll visualizations generated successfully in: {output_dir}")
    
    if args.word_optimized:
        print("\nFigures were optimized for Word. You should be able to insert them directly into Word documents.")


if __name__ == "__main__":
    main()