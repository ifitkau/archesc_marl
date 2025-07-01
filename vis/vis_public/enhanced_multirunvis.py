#!/usr/bin/env python3
"""
Unified Single-Plot Episode Length Visualization

This script creates a comprehensive single-panel visualization that combines:
- Moving average trends for multiple training runs
- Segment-by-segment analysis with statistical annotations
- Comparative performance metrics across different experimental conditions

The visualization is designed for reinforcement learning episode length analysis,
comparing multiple training runs (e.g., different hyperparameters, environments,
or algorithm variants) on a single plot with detailed segment statistics.

Key Features:
- Automatic data loading from JSON/CSV files
- Moving average trend smoothing
- Training phase segmentation with statistical analysis
- Colorblind-friendly visualization
- Publication-quality output
- Command-line interface for batch processing

Usage:
  python enhanced_multirunvis.py run1.json run2.json run3.json --labels "Run L" "Run C" "Run R"
  python enhanced_multirunvis.py data/*.json --segments 6 --window-size 20
  python enhanced_multirunvis.py experiment.csv --output results/comparison.png

Date: July 2025
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.ndimage import gaussian_filter1d
from matplotlib.patches import Rectangle

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# Moving average window size - controls trend line smoothing
# Larger values = smoother trends but less responsive to changes
WINDOW_SIZE = 10

# Number of training segments for phase analysis
# Divides training into equal phases for comparative analysis
SEGMENT_COUNT = 4

# Gaussian smoothing parameter for raw data visualization
# Higher values create smoother curves
SMOOTHING_SIGMA = 2

# Output image resolution for publication-quality figures
DPI = 300

# Colorblind-friendly color palette
# Based on research-backed accessible color schemes
COLORBLIND_PALETTE = [
    "#0072B2",  # Blue - primary color for first run
    "#D55E00",  # Red/Orange - high contrast secondary color
    "#009E73",  # Green - accessible tertiary color
    "#CC79A7",  # Pink - quaternary color
    "#F0E442",  # Yellow - bright accent color
    "#56B4E9",  # Light Blue - variation for additional runs
    "#E69F00",  # Orange - warm accent
    "#000000"   # Black - final fallback color
]

# ============================================================================
# DATA MANAGEMENT CLASS
# ============================================================================

class RunData:
    """
    Class to store and process training data from a single experimental run
    
    This class handles:
    - Loading data from various file formats (JSON, CSV)
    - Computing statistical metrics (mean, std, min, max)
    - Calculating moving averages for trend analysis
    - Performing segment analysis for training phase comparison
    - Gaussian smoothing for noise reduction
    
    Attributes:
        file_path (str): Path to the source data file
        label (str): Human-readable identifier for this run
        color (str): Matplotlib color for visualization
        steps (numpy.ndarray): Training step/iteration numbers
        values (numpy.ndarray): Episode length values
        mean, median, min, max, std (float): Basic statistics
        segments (list): Segment analysis results
        smoothed (numpy.ndarray): Gaussian-smoothed values
        moving_average_values (numpy.ndarray): Moving average trend
    """
    
    def __init__(self, file_path, label=None, color=None):
        """
        Initialize RunData object with data loading and preprocessing
        
        Args:
            file_path (str): Path to data file (JSON or CSV format)
            label (str, optional): Display name for this run. If None, uses filename
            color (str, optional): Matplotlib color. If None, auto-assigned from palette
        """
        self.file_path = file_path
        # Use filename as default label if none provided
        self.label = label or os.path.splitext(os.path.basename(file_path))[0]
        self.color = color
        
        # ====================================================================
        # DATA LOADING AND PREPROCESSING
        # ====================================================================
        
        # Load raw training data from file
        self.steps, self.values = self.load_data()
        
        # Calculate basic statistical metrics for quick reference
        self.mean = np.mean(self.values)
        self.median = np.median(self.values)
        self.min = np.min(self.values)
        self.max = np.max(self.values)
        self.std = np.std(self.values)
        
        # Perform segment analysis to identify training phases
        self.segments = self.segment_analysis(SEGMENT_COUNT)
        
        # Apply Gaussian smoothing to reduce noise in raw data
        self.smoothed = gaussian_filter1d(self.values, sigma=SMOOTHING_SIGMA)
        
        # Calculate moving average for trend visualization
        self.moving_average_values = self.moving_average(WINDOW_SIZE)
    
    def load_data(self):
        """
        Load training data from file with automatic format detection
        
        Supports multiple data formats:
        - JSON: [[timestamp, step, value], ...] or similar structures
        - CSV: Columns for steps and values (various naming conventions)
        
        Returns:
            tuple: (steps, values) as numpy arrays
            
        Raises:
            ValueError: If file format is unsupported or data structure is invalid
        """
        # Determine file format from extension
        file_ext = os.path.splitext(self.file_path)[1].lower()
        
        # ====================================================================
        # JSON FILE LOADING
        # ====================================================================
        if file_ext == '.json':
            with open(self.file_path, 'r') as f:
                data = json.load(f)
            
            # Handle nested list format: [[timestamp, step, value], ...]
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                # Extract step and value columns (skip timestamp in column 0)
                steps = [entry[1] for entry in data]
                values = [entry[2] for entry in data]
                return np.array(steps), np.array(values)
        
        # ====================================================================
        # CSV FILE LOADING
        # ====================================================================
        elif file_ext == '.csv':
            try:
                import pandas as pd
                df = pd.read_csv(self.file_path)
                
                # Try multiple column naming conventions
                # Standard naming: 'step' and 'value'
                if 'step' in df.columns and 'value' in df.columns:
                    return df['step'].values, df['value'].values
                # Capitalized naming: 'Step' and 'Value'
                elif 'Step' in df.columns and 'Value' in df.columns:
                    return df['Step'].values, df['Value'].values
                # TensorBoard format: [Wall time, Step, Value]
                elif len(df.columns) >= 3:
                    return df.iloc[:, 1].values, df.iloc[:, 2].values
                    
            except ImportError:
                print("pandas not installed, please install with: pip install pandas")
                sys.exit(1)
        
        # If no format matched, raise an error with helpful message
        raise ValueError(f"Could not parse data from {self.file_path}. "
                        f"Supported formats: JSON [[timestamp, step, value]] or "
                        f"CSV with 'step'/'value' columns")
    
    def moving_average(self, window_size):
        """
        Calculate moving average with cumulative start for early training steps
        
        This implementation handles the initial training period where insufficient
        data points exist for a full window by using cumulative averages.
        
        Args:
            window_size (int): Number of points to average over
            
        Returns:
            numpy.ndarray: Moving average values aligned with original data
            
        Note:
            - For steps 0 to window_size-1: uses cumulative average
            - For later steps: uses standard sliding window average
            - This prevents data loss at the beginning of training
        """
        # Handle edge cases
        if len(self.values) < window_size:
            return self.values
        
        # If window is larger than data, return constant mean
        if window_size >= len(self.values):
            return np.full(len(self.values), np.mean(self.values))
        
        # Initialize output array
        ma_values = np.zeros_like(self.values, dtype=float)
        
        # Calculate moving average with cumulative start
        for i in range(len(ma_values)):
            if i < window_size:
                # Cumulative average for initial steps (prevents data loss)
                ma_values[i] = np.mean(self.values[:i+1])
            else:
                # Standard sliding window average
                ma_values[i] = np.mean(self.values[i-window_size+1:i+1])
        
        return ma_values
        
    def segment_analysis(self, num_segments):
        """
        Divide training data into equal segments for phase-based analysis
        
        This method splits the training timeline into equal segments to analyze
        how performance changes across different training phases (e.g., early
        exploration vs. late convergence).
        
        Args:
            num_segments (int): Number of segments to create
            
        Returns:
            list: List of dictionaries containing segment statistics
                  Each dict contains: start_idx, end_idx, start_step, end_step,
                  mean, median, min, max, std, steps, values
                  
        Note:
            Segments are based on training steps, not data indices, ensuring
            equal time coverage even with irregular sampling.
        """
        segments = []
        total_steps = self.steps[-1]  # Final training step
        
        # Create segments based on training step ranges
        for i in range(num_segments):
            # Calculate segment boundaries as percentages of total training
            start_percent = i / num_segments
            end_percent = (i + 1) / num_segments
            
            # Convert percentages to actual step numbers
            start_step = int(total_steps * start_percent)
            end_step = int(total_steps * end_percent)
            
            # Find data indices closest to these step boundaries
            # This handles irregular sampling or missing data points
            start_idx = np.argmin(np.abs(self.steps - start_step))
            end_idx = np.argmin(np.abs(self.steps - end_step))
            
            # Extract segment data
            segment_values = self.values[start_idx:end_idx]
            segment_steps = self.steps[start_idx:end_idx]
            
            # Calculate comprehensive statistics for this segment
            segments.append({
                'start_idx': start_idx,           # Data array start index
                'end_idx': end_idx,               # Data array end index
                'start_step': start_step,         # Training step start
                'end_step': end_step,             # Training step end
                'mean': np.mean(segment_values) if len(segment_values) > 0 else 0,
                'median': np.median(segment_values) if len(segment_values) > 0 else 0,
                'min': np.min(segment_values) if len(segment_values) > 0 else 0,
                'max': np.max(segment_values) if len(segment_values) > 0 else 0,
                'std': np.std(segment_values) if len(segment_values) > 0 else 0,
                'steps': segment_steps,           # Raw step data for this segment
                'values': segment_values          # Raw value data for this segment
            })
        
        return segments

# ============================================================================
# MAIN VISUALIZATION FUNCTION
# ============================================================================

def create_unified_single_plot(runs, output_path, window_size=WINDOW_SIZE, num_segments=SEGMENT_COUNT):
    """
    Create a comprehensive single-plot visualization combining trends and segment analysis
    
    This function creates a unified visualization that shows:
    - Moving average trends for all runs
    - Segment boundaries and labels
    - Statistical annotations for each segment
    - Overall performance comparison metrics
    
    The plot is designed to provide both high-level trend comparison and
    detailed segment-by-segment performance analysis in a single view.
    
    Args:
        runs (list): List of RunData objects to visualize
        output_path (str): File path for saving the output image
        window_size (int): Moving average window size
        num_segments (int): Number of segments for analysis
        
    Returns:
        None (saves plot to file)
        
    Example:
        runs = [RunData("exp1.json", "Baseline"), RunData("exp2.json", "Modified")]
        create_unified_single_plot(runs, "comparison.png", window_size=20, num_segments=6)
    """
    # ========================================================================
    # GLOBAL STATISTICS CALCULATION
    # ========================================================================
    
    # Calculate overall performance metric across all runs for reference
    all_values = np.concatenate([run.values for run in runs])
    overall_mean = np.mean(all_values)
    
    # ========================================================================
    # FIGURE SETUP AND CONFIGURATION
    # ========================================================================
    
    # Create figure with publication-suitable size
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Assign colors to runs if not already specified
    for i, run in enumerate(runs):
        if run.color is None:
            run.color = COLORBLIND_PALETTE[i % len(COLORBLIND_PALETTE)]
    
    # Define descriptive labels for common run naming conventions
    # This improves readability for standard experimental setups
    run_labels = {
        "Run L": "Run L - Terminal Left",
        "Run C": "Run C - Terminal Center", 
        "Run R": "Run R - Terminal Right"
    }
    
    # ========================================================================
    # MAIN TREND LINE PLOTTING
    # ========================================================================
    
    # Plot moving average trends for each run
    for run in runs:
        # Use descriptive labels if available, otherwise use original label
        descriptive_label = run_labels.get(run.label, run.label)
        
        # Plot moving average trend line
        ax.plot(run.steps, run.moving_average_values, color=run.color, linewidth=2.5, 
                label=f"{descriptive_label} (MA={window_size})")
    
    # Add horizontal reference line for overall performance comparison
    ax.axhline(overall_mean, linestyle=':', color='gray', alpha=0.5,
              label=f"Overall Mean: {overall_mean:.1f}")
    
    # ========================================================================
    # AXIS SCALING AND RANGE OPTIMIZATION
    # ========================================================================
    
    # Calculate optimal axis ranges based on actual data
    if runs:
        # X-axis: start from 0, extend to maximum training step
        min_step = 0
        max_step = max([np.max(run.steps) for run in runs])
        
        # Y-axis: focus on actual data range with minimal padding
        y_values = []
        for run in runs:
            # Collect all non-NaN moving average values
            y_values.extend(run.moving_average_values[~np.isnan(run.moving_average_values)])
        
        # Ensure overall mean line is visible
        max_y = max(max(y_values), overall_mean * 1.05)
        
        # Set precise axis limits for tight, professional appearance
        ax.set_xlim(0, max_step)
        ax.set_ylim(0, max_y * 1.05)  # Small top padding for annotations
    
    # ========================================================================
    # SEGMENT BOUNDARY VISUALIZATION
    # ========================================================================
    
    # Add segment boundaries and annotations using first run as reference
    if runs:
        reference_run = runs[0]
        
        # Process each segment for boundary lines and labels
        for i, segment in enumerate(reference_run.segments):
            # Add vertical line for segment boundaries (skip first segment start)
            if i > 0:
                ax.axvline(segment['start_step'], color='gray', linestyle='--', alpha=0.3)
            
            # Add segment boundary labels with step ranges
            # Offset labels slightly for better readability
            label_x = segment['start_step'] if i > 0 else 0
            
            # Add offset for better label positioning
            if i == 0:
                label_x += 750  # Offset first segment label
            if i > 0:
                label_x += 750  # Offset subsequent segment labels
            
            # Create informative segment labels with step ranges
            ax.text(label_x, ax.get_ylim()[1] * 0.98, 
                f"Segment {i+1}\n({segment['start_step']}-{segment['end_step']})", 
                rotation=90, va='top', ha='right', 
                color='gray', alpha=0.7, fontsize=8)
    
    # ========================================================================
    # SEGMENT STATISTICS ANNOTATION
    # ========================================================================
    
    # Create detailed segment information overlays
    if runs and num_segments > 0:
        for i in range(num_segments):
            if i < len(reference_run.segments):
                segment = reference_run.segments[i]
                
                # Calculate segment boundaries and midpoint
                start_step = segment['start_step']
                end_step = segment['end_step']
                mid_step = (start_step + end_step) / 2
                
                # Add main segment label in prominent position
                label_y_pos = ax.get_ylim()[1] * 0.95
                ax.text(mid_step, label_y_pos, f"Segment {i+1}", 
                        ha='center', va='center', fontsize=10, color='gray',
                        bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8, 
                                 boxstyle='round,pad=0.3'),
                        zorder=10)
                
                # Add performance statistics for each run within this segment
                for j, run in enumerate(runs):
                    if i < len(run.segments):
                        run_segment = run.segments[i]
                        
                        # Add horizontal line showing segment mean performance
                        # This provides visual reference for performance level
                        ax.hlines(run_segment['mean'], start_step, end_step, 
                                 colors=run.color, linestyles='--', alpha=0.5, 
                                 linewidth=1.5)
                        
                        # Add marker dot at segment mean for precise identification
                        ax.plot(mid_step, run_segment['mean'], 'o', 
                                color=run.color, markersize=5, alpha=0.7)
                        
                        # Add text annotation with segment statistics
                        # Position annotations vertically to avoid overlap
                        box_y_pos = ax.get_ylim()[1] * (0.85 - j*0.08)
                        
                        ax.text(mid_step, box_y_pos, 
                                f"{run.label}: {run_segment['mean']:.1f}", 
                                ha='center', va='center', fontsize=9, color=run.color,
                                bbox=dict(facecolor='white', edgecolor=run.color, alpha=0.8, 
                                         boxstyle='round,pad=0.3'),
                                zorder=10)
    
    # ========================================================================
    # PLOT FORMATTING AND STYLING
    # ========================================================================
    
    # Set axis labels and title
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Episode Length', fontsize=12)
    ax.set_title(f'Comparison of Episode Length with Segment Analysis (Moving Average={window_size})', 
                 fontsize=14)
    
    # Configure tick formatting for readability
    # X-axis: reasonable number of ticks with clean integer formatting
    ax.xaxis.set_major_locator(ticker.MaxNLocator(10))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    
    # Y-axis: round numbers for episode length scale
    ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

    # Add subtle grid for data reading assistance
    ax.grid(True, alpha=0.3, linestyle=':')
    
    # Position legend in least intrusive location
    ax.legend(loc='lower right')
    
    # ========================================================================
    # LAYOUT OPTIMIZATION AND EXPORT
    # ========================================================================
    
    # Apply tight layout with error handling
    try:
        plt.tight_layout()
    except:
        print("Warning: Could not apply tight_layout. Figure may have suboptimal spacing.")
    
    # Ensure output directory exists
    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    
    # Save high-quality figure suitable for publications
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"Saved unified single plot to {output_path}")
    
    # Free memory by closing figure
    plt.close(fig)

# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

def main():
    """
    Main function providing command-line interface for the visualization tool
    
    This function handles:
    - Command-line argument parsing
    - Input validation
    - Data loading and processing
    - Visualization generation
    - Basic statistics reporting
    
    The CLI supports various options for customizing the visualization,
    including custom labels, colors, analysis parameters, and output settings.
    """
    # ========================================================================
    # ARGUMENT PARSER SETUP
    # ========================================================================
    
    parser = argparse.ArgumentParser(
        description='Unified Single-Plot Episode Length Visualization',
        epilog="""
Examples:
  %(prog)s run1.json run2.json --labels "Baseline" "Modified"
  %(prog)s data/*.json --segments 6 --window-size 20
  %(prog)s experiment.csv --output results/analysis.png
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # ========================================================================
    # REQUIRED ARGUMENTS
    # ========================================================================
    
    # Input data files (JSON or CSV format)
    parser.add_argument('files', nargs='+', 
                        help='Data files containing episode length information. '
                             'Supports JSON format [[timestamp, step, value]] or '
                             'CSV format with step/value columns.')
    
    # ========================================================================
    # OPTIONAL CUSTOMIZATION ARGUMENTS
    # ========================================================================
    
    # Run identification and styling
    parser.add_argument('--labels', nargs='+', default=None, 
                        help='Custom labels for each run (must match number of files). '
                             'If not provided, uses filenames.')
    parser.add_argument('--colors', nargs='+', default=None,
                        help='Custom colors for each run (must match number of files). '
                             'Uses matplotlib color names or hex codes.')
    
    # Analysis parameters
    parser.add_argument('--window-size', type=int, default=WINDOW_SIZE, 
                        help=f'Moving average window size (default: {WINDOW_SIZE}). '
                             'Larger values create smoother trends.')
    parser.add_argument('--segments', type=int, default=SEGMENT_COUNT,
                        help=f'Number of training segments for analysis (default: {SEGMENT_COUNT}). '
                             'More segments provide finer-grained phase analysis.')
    
    # Output configuration
    parser.add_argument('--output', type=str, default='unified_single_plot.png',
                        help='Output file path for the visualization (default: unified_single_plot.png). '
                             'Supports PNG, PDF, SVG formats.')
    
    # Parse command-line arguments
    args = parser.parse_args()
    
    # ========================================================================
    # INPUT VALIDATION
    # ========================================================================
    
    # Validate label count matches file count
    if args.labels and len(args.labels) != len(args.files):
        parser.error(f"Number of labels ({len(args.labels)}) must match "
                    f"number of files ({len(args.files)})")
    
    # Validate color count matches file count
    if args.colors and len(args.colors) != len(args.files):
        parser.error(f"Number of colors ({len(args.colors)}) must match "
                    f"number of files ({len(args.files)})")
    
    # ========================================================================
    # DATA LOADING AND PROCESSING
    # ========================================================================
    
    runs = []
    print("Loading and processing data files...")
    
    for i, file_path in enumerate(args.files):
        # Verify file exists
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            sys.exit(1)
        
        # Extract custom parameters
        label = args.labels[i] if args.labels else None
        color = args.colors[i] if args.colors else None
        
        try:
            # Create RunData object (handles loading and preprocessing)
            run = RunData(file_path, label, color)
            runs.append(run)
            
            # Display basic statistics for user feedback
            print(f"\n{run.label} statistics:")
            print(f"  Data points: {len(run.values)}")
            print(f"  Training steps: {run.steps[0]} to {run.steps[-1]}")
            print(f"  Mean episode length: {run.mean:.2f}")
            print(f"  Standard deviation: {run.std:.2f}")
            print(f"  Range: {run.min:.2f} to {run.max:.2f}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            sys.exit(1)
    
    # ========================================================================
    # VISUALIZATION GENERATION
    # ========================================================================
    
    print(f"\nGenerating visualization with {len(runs)} runs...")
    print(f"Parameters: window_size={args.window_size}, segments={args.segments}")
    
    try:
        # Create the unified visualization
        create_unified_single_plot(
            runs=runs, 
            output_path=args.output,
            window_size=args.window_size,
            num_segments=args.segments
        )
        
        print(f"\nVisualization completed successfully!")
        print(f"Output saved to: {args.output}")
        
    except Exception as e:
        print(f"Error creating visualization: {e}")
        sys.exit(1)

# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    """
    Entry point for command-line execution
    
    Example usage scenarios:
    
    1. Basic comparison of multiple runs:
       python enhanced_multirunvis.py run1.json run2.json run3.json
    
    2. Custom labels and analysis parameters:
       python enhanced_multirunvis.py *.json --labels "Baseline" "Modified" "Advanced" 
                                             --window-size 20 --segments 6
    
    3. Publication-ready output:
       python enhanced_multirunvis.py data/*.json --output figures/comparison.pdf
    
    4. CSV data processing:
       python enhanced_multirunvis.py tensorboard_logs.csv --segments 8
    """
    main()