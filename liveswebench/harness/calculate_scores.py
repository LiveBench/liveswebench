import pandas as pd
import argparse
import re
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.projections import register_projection
from pathlib import Path

from liveswebench.harness.stats import count_hunks_in_patch
from liveswebench.util.tasks import load_tasks

def parse_arguments():
    parser = argparse.ArgumentParser(description='Calculate average scores from task data.')
    parser.add_argument('--csv_files', required=True, nargs='+', help='Paths to the CSV files')
    parser.add_argument('--repo_name', help='Filter by repository name')
    parser.add_argument('--task_num', help='Filter by task number')
    parser.add_argument('--tool_name', help='Filter by tool name')
    parser.add_argument('--exclude-tool', nargs='+', help='Exclude specified tools from results')
    parser.add_argument('--only-all-scores', action='store_true',
                        help='Only include tasks where all tools have a score')
    parser.add_argument('--only-with-tool', metavar='TOOL',
                        help='Only include tasks where specified tool has a score')
    parser.add_argument('--output_graph', help='Path to save the output graph image')
    parser.add_argument('--output-csv', action='store_true',
                        help='Output CSV files with average scores for each task type')
    parser.add_argument('--graph-type', choices=['bar', 'line', 'radar', 'repo-bar', 'patch-size'], default='bar',
                        help='Type of graph to create (bar, line, radar, repo-bar, or patch-size)')
    parser.add_argument('--split-by-repo', action='store_true',
                        help='For single task type, split visualization by repository')
    parser.add_argument('--split-by-task-type', action='store_true',
                        help='For radar graphs with multiple task types, overlay all task types on the same radar chart per tool')
    parser.add_argument('--show-average-only', action='store_true',
                        help='For repo-bar graph, show only average scores per repo instead of by tool')
    parser.add_argument('--task_source', type=str, choices=['local', 'huggingface'], default='huggingface',
                        help='Source of tasks, either local or from huggingface')
    return parser.parse_args()

def extract_task_components(task_str: str) -> tuple[str, str] | None:
    """Extract repo name and task number from task string."""
    pattern = r'(.*?)\s*-\s*(\d+)'
    match = re.match(pattern, task_str)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return None

def normalize_percentage(value: float | str | None) -> float:
    """Convert percentage strings to floats."""
    if pd.isna(value):
        return float('nan')
    if isinstance(value, str):
        value = value.strip()
        if value.endswith('%'):
            return float(value.rstrip('%')) / 100
    return float(value)

def get_task_type_from_filename(filepath: str) -> str:
    """Extract task type from the filename (without extension)."""
    basename = os.path.basename(filepath)
    filename = os.path.splitext(basename)[0]
    return filename

def process_csv_file(filepath: str, args: argparse.Namespace) -> tuple[str, pd.DataFrame] | None:
    """Process a single CSV file and return the results."""
    task_type = get_task_type_from_filename(filepath)
    
    # Read CSV file
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error reading CSV file {filepath}: {e}")
        return None

    # Convert column names to match the expected format
    column_mapping = {}
    for col in df.columns:
        if col.lower().strip() == 'task':
            column_mapping[col] = 'Task'
        elif col.lower().strip() == 'tool':
            column_mapping[col] = 'Tool'
        elif col.lower().strip() == 'score':
            column_mapping[col] = 'Score'
        elif col.lower().strip() == 'complete':
            column_mapping[col] = 'Complete'
        else:
            column_mapping[col] = col

    df.rename(columns=column_mapping, inplace=True)

    # Ensure required columns exist
    required_columns = ['Task', 'Tool', 'Score']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Missing required columns in {filepath}: {', '.join(missing_columns)}")
        return None

    # Convert Score column to numeric values
    df['Score'] = df['Score'].apply(normalize_percentage)

    # Parse task components
    task_components = df['Task'].apply(extract_task_components)
    df['RepoName'] = [comp[0] for comp in task_components]
    df['TaskNum'] = [comp[1] for comp in task_components]

    # Apply filters based on command-line arguments
    filtered_df = df.copy()
    
    # Create a temporary lowercase column for case-insensitive comparisons
    filtered_df['Tool_Lower'] = filtered_df['Tool'].str.lower()
    
    # Apply --exclude-tool first, before any other filtering (make case-insensitive)
    if args.exclude_tool:
        exclude_tools_lower = [tool.lower() for tool in args.exclude_tool]
        filtered_df = filtered_df[~filtered_df['Tool_Lower'].isin(exclude_tools_lower)]

    if args.repo_name:
        filtered_df = filtered_df[filtered_df['RepoName'] == args.repo_name]

    if args.task_num:
        filtered_df = filtered_df[filtered_df['TaskNum'] == args.task_num]

    if args.tool_name:
        # Case-insensitive tool name matching
        filtered_df = filtered_df[filtered_df['Tool_Lower'] == args.tool_name.lower()]

    # Drop the temporary column
    filtered_df = filtered_df.drop('Tool_Lower', axis=1)

    # Get unique tasks
    unique_tasks = filtered_df['Task'].unique()

    # Filter tasks based on the --only-all-scores flag
    if args.only_all_scores:
        tasks_with_all_scores = []
        for task in unique_tasks:
            task_df = filtered_df[filtered_df['Task'] == task]
            if not task_df['Score'].isna().any():
                tasks_with_all_scores.append(task)

        filtered_df = filtered_df[filtered_df['Task'].isin(tasks_with_all_scores)]
        unique_tasks = tasks_with_all_scores

    # Filter tasks based on the --only-with-tool flag
    if args.only_with_tool:
        # Case-insensitive comparison for --only-with-tool as well
        only_with_tool_lower = args.only_with_tool.lower()
        
        # Recreate the temporary column
        filtered_df['Tool_Lower'] = filtered_df['Tool'].str.lower()
        
        tasks_with_tool_score = []
        for task in unique_tasks:
            task_df = filtered_df[(filtered_df['Task'] == task) & 
                                (filtered_df['Tool_Lower'] == only_with_tool_lower)]
            if not task_df.empty and not task_df['Score'].isna().all():
                tasks_with_tool_score.append(task)
        
        # Drop the temporary column again
        filtered_df = filtered_df.drop('Tool_Lower', axis=1)
        
        filtered_df = filtered_df[filtered_df['Task'].isin(tasks_with_tool_score)]

    # Return the filtered DataFrame and task type
    return {
        'task_type': task_type,
        'filtered_df': filtered_df
    }

def create_bar_graph(results_by_task_type: dict[str, pd.DataFrame], output_path: str | None = None, task_type_scores: dict[str, dict[str, float]] | None = None, overall_tool_order: list[str] | None = None, split_by_repo: bool = False):
    """Create a bar graph showing tool performance by task type."""
    # Extract all tools and task types from the results
    all_tools = set()
    all_task_types = list(results_by_task_type.keys())
    tool_scores_by_task = {}
    
    # If we're splitting by repo and have only one task type
    if split_by_repo and len(all_task_types) == 1:
        return create_repo_split_bar_graph(results_by_task_type, output_path, overall_tool_order)
    
    for task_type, df in results_by_task_type.items():
        scores_df = df.dropna(subset=['Score'])
        if not scores_df.empty:
            tool_avg = scores_df.groupby('Tool')['Score'].mean()
            tool_scores_by_task[task_type] = tool_avg
            all_tools.update(tool_avg.index)
    
    if not tool_scores_by_task:
        print("No data available for plotting.")
        return
    
    # Use pre-computed tool order if available
    if len(all_task_types) == 1 and task_type_scores and all_task_types[0] in task_type_scores:
        # Use the individual task type order
        only_task = all_task_types[0]
        scores_dict = task_type_scores[only_task]
        tool_sort_order = sorted(all_tools, key=lambda x: scores_dict.get(x, 0), reverse=True)
    elif overall_tool_order:
        # Use the overall order from main function
        tool_sort_order = [tool for tool in overall_tool_order if tool in all_tools]
    else:
        # Calculate overall average scores for sorting as fallback
        overall_scores = {}
        for tool in all_tools:
            scores = []
            for task_type in all_task_types:
                if task_type in tool_scores_by_task and tool in tool_scores_by_task[task_type]:
                    scores.append(tool_scores_by_task[task_type][tool])
            overall_scores[tool] = sum(scores) / len(scores) if scores else 0
        tool_sort_order = sorted(all_tools, key=lambda x: overall_scores[x], reverse=True)
    
    # Set up the plot
    plt.figure(figsize=(max(10, len(all_tools) * 1.5), 8))
    
    # Create a color map for task types
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_task_types)))
    color_map = {task_type: color for task_type, color in zip(all_task_types, colors)}
    
    # Width of each bar group
    group_width = 0.8
    bar_width = group_width / len(all_task_types)
    
    # Plot bars for each task type
    for i, task_type in enumerate(all_task_types):
        if task_type not in tool_scores_by_task:
            continue
            
        scores = tool_scores_by_task[task_type]
        
        # Position bars within each group
        offset = i * bar_width - group_width/2 + bar_width/2
        plt.bar(
            [tool_sort_order.index(tool) + offset for tool in tool_sort_order if tool in scores], 
            [scores.get(tool, 0) for tool in tool_sort_order if tool in scores],
            width=bar_width,
            color=color_map[task_type],
            label=task_type
        )
    
    # Set up the axes and labels
    plt.xlabel('Tool')
    plt.ylabel('Score (%)')
    plt.title('Tool Performance by Task Type')
    plt.xticks(range(len(tool_sort_order)), tool_sort_order, rotation=45, ha='right')
    plt.yticks(np.arange(0, 1.1, 0.1), [f"{int(x*100)}%" for x in np.arange(0, 1.1, 0.1)])
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Task Type')
    plt.tight_layout()
    
    # Save or show the plot
    if output_path:
        plt.savefig(output_path)
        print(f"Graph saved to {output_path}")
    else:
        plt.show()

def create_repo_split_bar_graph(results_by_task_type: dict[str, pd.DataFrame], output_path: str | None = None, overall_tool_order: list[str] | None = None):
    """Create a bar graph splitting results by repository for a single task type."""
    # Should only be one task type
    task_type = list(results_by_task_type.keys())[0]
    df = results_by_task_type[task_type]
    
    # First, get all unique tools and repos from the raw data
    all_tools = set(df['Tool'].unique())
    all_repos = set(df['RepoName'].unique())
    
    print(f"Found {len(all_tools)} tools and {len(all_repos)} repositories")
    
    # Prepare a dictionary to store tool/repo scores and presence information
    tool_repo_data = {}
    
    # For each tool and repo combination, check if data exists
    # and calculate average score if there are valid scores
    for tool in all_tools:
        tool_repo_data[tool] = {}
        
        for repo in all_repos:
            # Get data for this specific tool and repo
            subset = df[(df['Tool'] == tool) & (df['RepoName'] == repo)]
            
            # If the tool has entries for this repo
            if not subset.empty:
                # Check if there are non-NaN scores
                valid_scores = subset.dropna(subset=['Score'])
                has_data = not valid_scores.empty
                
                # Store information
                avg_score = valid_scores['Score'].mean() if has_data else 0
                tool_repo_data[tool][repo] = {
                    'has_data': has_data,
                    'avg_score': avg_score,
                    'count': len(valid_scores)
                }
                
                # Print debug info for problematic tools
                if tool in ['CodeLLM', 'Aider', 'Amazon Q']:
                    print(f"Tool: {tool}, Repo: {repo}, Has data: {has_data}, Score: {avg_score:.2f}, Entries: {len(subset)}, Valid entries: {len(valid_scores)}")
            else:
                # No data for this combination
                tool_repo_data[tool][repo] = {
                    'has_data': False,
                    'avg_score': 0,
                    'count': 0
                }
    
    # Sort tools by average score across all repos
    tool_avgs = {}
    for tool in all_tools:
        scores = [info['avg_score'] for repo, info in tool_repo_data[tool].items() 
                 if info['has_data']]
        if scores:
            tool_avgs[tool] = sum(scores) / len(scores)
        else:
            tool_avgs[tool] = 0
    
    # Use overall tool order if available, otherwise sort by our calculated average
    if overall_tool_order:
        # Ensure all tools are included
        sorted_tools = [t for t in overall_tool_order if t in all_tools]
        # Add any missing tools at the end
        sorted_tools.extend([t for t in all_tools if t not in sorted_tools])
    else:
        sorted_tools = sorted(all_tools, key=lambda t: tool_avgs.get(t, 0), reverse=True)
    
    # Set up the plot with enough width for the grouped bars
    plt.figure(figsize=(max(12, len(sorted_tools) * 2), 10))
    
    # Create a color map for repositories
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_repos)))
    color_map = {repo: color for repo, color in zip(all_repos, colors)}
    
    # Width of each group of bars
    group_width = 0.8
    bar_width = group_width / len(all_repos)
    
    # Sort repos alphabetically for consistent ordering
    sorted_repos = sorted(all_repos)
    
    # Plot bars for each repository and tool combination
    for repo_idx, repo in enumerate(sorted_repos):
        x_positions = []
        heights = []
        
        # For each tool, check if we have data for this repo
        for tool_idx, tool in enumerate(sorted_tools):
            info = tool_repo_data[tool].get(repo, {'has_data': False})
            
            if info['has_data']:
                # Calculate position for this bar
                x_pos = tool_idx + repo_idx * bar_width - group_width/2 + bar_width/2
                x_positions.append(x_pos)
                heights.append(info['avg_score'])
        
        # Only create bars if we have data for this repo
        if x_positions:
            plt.bar(
                x_positions,
                heights,
                width=bar_width,
                color=color_map[repo],
                label=repo
            )
    
    # Set up the axes and labels
    plt.xlabel('Tool')
    plt.ylabel('Score (%)')
    plt.title(f'Tool Performance for {task_type} Tasks by Repository')
    plt.xticks(range(len(sorted_tools)), sorted_tools, rotation=45, ha='right')
    plt.yticks(np.arange(0, 1.1, 0.1), [f"{int(x*100)}%" for x in np.arange(0, 1.1, 0.1)])
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Repository')
    plt.tight_layout()
    
    # Save or show the plot
    if output_path:
        plt.savefig(output_path)
        print(f"Graph saved to {output_path}")
    else:
        plt.show()
    
    return True

def create_repo_line_chart(results_by_task_type: dict[str, pd.DataFrame], output_path: str | None = None, overall_tool_order: list[str] | None = None):
    """Create a line chart showing tool performance across repositories."""
    # Should only be one task type
    task_type = list(results_by_task_type.keys())[0]
    df = results_by_task_type[task_type]
    
    # Calculate scores by tool and repo
    scores_df = df.dropna(subset=['Score'])
    
    if scores_df.empty:
        print("No data available for plotting.")
        return False
    
    # Create a pivot table for easier data manipulation
    pivot_df = scores_df.pivot_table(values='Score', index='Tool', columns='RepoName', aggfunc='mean')
    
    # Calculate tool averages
    pivot_df['Average'] = pivot_df.mean(axis=1)
    
    # Sort repositories by average difficulty (easier to harder)
    repo_difficulty = pivot_df.drop('Average', axis=1).mean()
    sorted_repos = repo_difficulty.sort_values().index.tolist() + ['Average']
    
    # Reindex the pivot table with the sorted repositories
    pivot_df = pivot_df[sorted_repos]
    
    # Use overall tool order if available, otherwise sort by average
    if overall_tool_order:
        tools_in_data = [t for t in overall_tool_order if t in pivot_df.index]
        # Add any missing tools that exist in the data
        tools_in_data.extend([t for t in pivot_df.index if t not in tools_in_data])
        sorted_tools = tools_in_data
    else:
        sorted_tools = pivot_df.sort_values('Average', ascending=False).index.tolist()
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    
    # Create a color map for tools
    colors = plt.cm.tab10(np.linspace(0, 1, len(sorted_tools)))
    color_map = {tool: color for tool, color in zip(sorted_tools, colors)}
    
    # Create x positions for repositories
    repo_positions = np.arange(len(sorted_repos))
    
    # Plot lines for each tool
    for tool in sorted_tools:
        if tool in pivot_df.index:
            tool_data = pivot_df.loc[tool]
            
            # Filter out NaN values if any
            valid_repos = [repo for repo in sorted_repos if not pd.isna(tool_data[repo])]
            if not valid_repos:
                continue
                
            x_values = [sorted_repos.index(repo) for repo in valid_repos]
            y_values = [tool_data[repo] for repo in valid_repos]
            
            plt.plot(x_values, y_values, 'o-', linewidth=2, label=tool, 
                     color=color_map[tool], markersize=8)
    
    # Set up axes and labels
    plt.xlabel('Repository (ordered by increasing difficulty)')
    plt.ylabel('Score (%)')
    plt.title(f'Tool Performance Across Repositories for {task_type} Tasks')
    plt.xticks(repo_positions, sorted_repos, rotation=45, ha='right')
    plt.yticks(np.arange(0, 1.1, 0.1), [f"{int(x*100)}%" for x in np.arange(0, 1.1, 0.1)])
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add a vertical line before the Average column
    if 'Average' in sorted_repos:
        avg_idx = sorted_repos.index('Average')
        plt.axvline(x=avg_idx-0.5, color='gray', linestyle='--', alpha=0.7)
    
    # Add legend with better placement for many tools
    if len(sorted_tools) > 6:
        plt.legend(title='Tools', bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.legend(title='Tools', loc='best')
    
    plt.tight_layout()
    
    # Save or show the plot
    if output_path:
        plt.savefig(output_path)
        print(f"Line chart saved to {output_path}")
    else:
        plt.show()
    
    return True

def radar_factory(num_vars: int, frame: str = 'circle'):
    """Create a radar chart with `num_vars` axes."""
    # Calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
    
    # Rotate theta such that the first axis is at the top
    theta += np.pi/2
    
    class RadarAxes(plt.PolarAxes):
        name = 'radar'
        RESOLUTION = 1  # The number of points to interpolate between each angle
        frame_type = frame  # Store frame as a class attribute
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location('N')
        
        def get_vertices(self):
            """Generate vertices of polygon for subplot axes on demand."""
            x0, y0, r = 0.5, 0.5, 0.5
            return [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]
            
        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default."""
            return super().fill(closed=closed, *args, **kwargs)
        
        def plot(self, *args, **kwargs):
            """Override plot to add closed kwarg."""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)
            return lines
        
        def _close_line(self, line):
            """Close a line by joining the last and first points."""
            x, y = line.get_data()
            # FIXME: markers at x[0] and x[-1] may be different, but close enough
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)
        
        def set_varlabels(self, labels):
            """Set the radar chart variable labels."""
            self.set_thetagrids(np.degrees(theta), labels)
        
        def _gen_axes_patch(self):
            """Generate the axes patch."""
            if self.frame_type == 'circle':
                return plt.Circle((0.5, 0.5), 0.5)
            elif self.frame_type == 'polygon':
                return plt.Polygon(self.get_vertices(), closed=True)
            else:
                raise ValueError("frame must be either 'circle' or 'polygon'")
    
    # Register the projection
    register_projection(RadarAxes)
    return theta

def create_repo_radar_chart(results_by_task_type: dict[str, pd.DataFrame], output_path: str | None = None, overall_tool_order: list[str] | None = None):
    """Create a radar chart showing tool performance across repositories."""
    # Should only be one task type
    task_type = list(results_by_task_type.keys())[0]
    df = results_by_task_type[task_type]
    
    # Calculate scores by tool and repo
    scores_df = df.dropna(subset=['Score'])
    
    if scores_df.empty:
        print("No data available for plotting.")
        return False
    
    # Create a pivot table for easier data manipulation
    pivot_df = scores_df.pivot_table(values='Score', index='Tool', columns='RepoName', aggfunc='mean')
    
    # Calculate tool averages for sorting but don't include in visualization
    pivot_df_with_avg = pivot_df.copy()
    pivot_df_with_avg['Average'] = pivot_df.mean(axis=1)
    
    # Ensure we have all repositories in the data
    all_repos = sorted(df['RepoName'].unique().tolist())
    for repo in all_repos:
        if repo not in pivot_df.columns:
            pivot_df[repo] = 0
    
    # Fill NaN values with zeros for radar chart
    pivot_df = pivot_df.fillna(0)
    
    # Get repositories (excluding Average)
    repos = sorted(pivot_df.columns.tolist())
    
    print(f"Repositories being used: {repos}")
    
    # Use overall tool order if available, otherwise sort by average
    if overall_tool_order:
        tools_in_data = [t for t in overall_tool_order if t in pivot_df.index]
        # Add any missing tools
        tools_in_data.extend([t for t in pivot_df.index if t not in tools_in_data])
        sorted_tools = tools_in_data
    else:
        sorted_tools = pivot_df_with_avg.sort_values('Average', ascending=False).index.tolist()
    
    # Create a grid of subplots, one for each tool
    num_tools = len(sorted_tools)
    
    # Calculate grid dimensions
    n_cols = min(3, num_tools)  # Maximum 3 columns
    n_rows = (num_tools + n_cols - 1) // n_cols  # Ceiling division for number of rows
    
    # Create figure with appropriate size - reduced height for less whitespace
    fig = plt.figure(figsize=(5.5 * n_cols, 4.7 * n_rows))
    
    # Number of variables (repositories)
    N = len(repos)
    
    # Create angles for each variable
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create a color map for tools
    colors = plt.cm.tab10(np.linspace(0, 1, len(sorted_tools)))
    color_map = {tool: color for tool, color in zip(sorted_tools, colors)}
    
    # Plot each tool in its own subplot
    for i, tool in enumerate(sorted_tools):
        if tool in pivot_df.index:
            # Create subplot with polar projection
            ax = fig.add_subplot(n_rows, n_cols, i + 1, polar=True)
            
            # Extract values for this tool
            values = [pivot_df.loc[tool, repo] for repo in repos]
            
            # Close the loop by adding the first value at the end
            values += values[:1]
            
            # Draw one axis per variable and add labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(repos, size=10)
            
            # Draw ylabels (percentage circles)
            ax.set_rlabel_position(0)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8])
            ax.set_yticklabels([f"{int(x*100)}%" for x in [0.2, 0.4, 0.6, 0.8]], color="grey", size=8)
            ax.set_ylim(0, 1)
            
            # Plot values
            ax.plot(angles, values, linewidth=2, linestyle='solid', color=color_map[tool])
            ax.fill(angles, values, color=color_map[tool], alpha=0.25)
            
            # Add title
            ax.set_title(tool, size=12, y=1.12)
            
            # Improve label position for readability
            for label, angle in zip(ax.get_xticklabels(), angles[:-1]):
                # Set label alignment based on position
                if angle < np.pi/2 or angle > 3*np.pi/2:
                    label.set_horizontalalignment('left')
                else:
                    label.set_horizontalalignment('right')
                
                # Adjust vertical alignment
                if angle <= np.pi:
                    label.set_verticalalignment('bottom')
                else:
                    label.set_verticalalignment('top')
                    
                # Rotate labels
                label.set_rotation(np.rad2deg(angle))
    
    # Add overall title - move up to reduce space
    task_types_str = ', '.join(task_type)
    if len(task_types_str) > 40:  # If the string is too long, truncate it
        task_types_str = task_types_str[:37] + "..."
    fig.suptitle(f'Tool Performance by Repository Across Task Types: {task_types_str}', size=16, y=0.95)
    
    # Add a single legend for all plots - positioned closer to grid
    if len(task_type) > 1:
        fig.legend(legend_handles, legend_labels, 
                  loc='upper center', bbox_to_anchor=(0.5, 0.93), 
                  ncol=min(len(task_type), 4), fontsize=10)
    
    # Adjust layout with reduced top margin - bring grid closer to legend
    plt.tight_layout(rect=[0, 0, 1, 0.92], pad=0.3, h_pad=0.3, w_pad=0.1)
    
    # Save or show the plot
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Radar charts grid saved to {output_path}")
    else:
        plt.show()
    
    return True

def create_line_graph(results_by_task_type: dict[str, pd.DataFrame], output_path: str | None = None, overall_tool_order: list[str] | None = None):
    """Create a line graph showing tool performance across task types."""
    # Extract tools and task types
    all_tools = set()
    all_task_types = list(results_by_task_type.keys())
    tools_data = {}
    
    # Gather data for each tool across task types
    for task_type in all_task_types:
        df = results_by_task_type[task_type]
        scores_df = df.dropna(subset=['Score'])
        if not scores_df.empty:
            tool_avg = scores_df.groupby('Tool')['Score'].mean()
            for tool in tool_avg.index:
                if tool not in tools_data:
                    tools_data[tool] = {}
                tools_data[tool][task_type] = tool_avg[tool]
                all_tools.add(tool)
    
    if not tools_data:
        print("No data available for plotting.")
        return
    
    # Use pre-computed tool order if available
    if overall_tool_order:
        sorted_tools = [tool for tool in overall_tool_order if tool in all_tools]
    else:
        # Sort tools by overall average performance
        overall_avg = {tool: np.mean([score for score in data.values()]) 
                      for tool, data in tools_data.items()}
        sorted_tools = sorted(all_tools, key=lambda x: overall_avg.get(x, 0), reverse=True)
    
    # Find the optimal task type ordering to maximize upward trends
    if len(all_task_types) <= 1:
        optimal_task_order = all_task_types
    else:
        optimal_task_order = find_optimal_task_order(tools_data, all_task_types)
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    
    # Create a color map for tools
    colors = plt.cm.tab10(np.linspace(0, 1, len(sorted_tools)))
    
    # Plot lines for each tool
    for i, tool in enumerate(sorted_tools):
        x_values = []
        y_values = []
        
        # Create x and y arrays, skipping missing values
        for x_idx, task_type in enumerate(optimal_task_order):
            if task_type in tools_data[tool]:
                x_values.append(x_idx)
                y_values.append(tools_data[tool][task_type])
        
        # Only plot if we have data points
        if x_values:
            plt.plot(x_values, y_values, 'o-', linewidth=2, label=tool, color=colors[i], markersize=8)
    
    # Set up axes and labels
    plt.xlabel('Task Type')
    plt.ylabel('Score (%)')
    plt.title('Tool Performance Across Task Types')
    plt.xticks(np.arange(len(optimal_task_order)), optimal_task_order)
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.yticks(np.arange(0, 1.1, 0.1), [f"{int(x*100)}%" for x in np.arange(0, 1.1, 0.1)])
    
    # Add legend with better placement for many tools
    if len(sorted_tools) > 6:
        plt.legend(title='Tools', bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.legend(title='Tools', loc='best')
    
    plt.tight_layout()
    
    # Save or show the plot
    if output_path:
        plt.savefig(output_path)
        print(f"Graph saved to {output_path}")
    else:
        plt.show()
        
def find_optimal_task_order(tools_data: dict[str, dict[str, float]], task_types: list[str]) -> list[str]:
    """Find the optimal ordering of task types to maximize upward trends."""
    import itertools
    
    # Calculate the number of increasing trends for a given task order
    def count_increasing_trends(order):
        increasing_count = 0
        total_transitions = 0
        
        for tool, data in tools_data.items():
            # Get ordered scores, replacing missing values with None
            ordered_scores = [data.get(task_type) for task_type in order]
            
            # Count increasing transitions
            for i in range(len(ordered_scores) - 1):
                # Skip transitions where one value is missing
                if ordered_scores[i] is None or ordered_scores[i+1] is None:
                    continue
                
                total_transitions += 1
                if ordered_scores[i+1] > ordered_scores[i]:
                    increasing_count += 1
        
        # If no valid transitions, return 0
        if total_transitions == 0:
            return 0
            
        return increasing_count / total_transitions
    
    # Try all permutations for up to 5 task types (120 permutations)
    if len(task_types) <= 5:
        all_permutations = list(itertools.permutations(task_types))
        best_order = max(all_permutations, key=count_increasing_trends)
        return list(best_order)
    
    # For more than 5 task types, use a greedy approach
    # Start with the task type that has the lowest average score
    avg_scores = {}
    for task_type in task_types:
        scores = [data[task_type] for tool, data in tools_data.items() if task_type in data]
        avg_scores[task_type] = sum(scores) / len(scores) if scores else 0
    
    current_order = [min(avg_scores, key=avg_scores.get)]
    remaining_tasks = [t for t in task_types if t != current_order[0]]
    
    # Iteratively add the next task that maximizes increasing trends
    while remaining_tasks:
        best_next_task = None
        best_score = -1
        
        for task in remaining_tasks:
            # Try adding this task and see how many tools increase
            test_order = current_order + [task]
            score = count_increasing_trends(test_order)
            if score > best_score:
                best_score = score
                best_next_task = task
        
        current_order.append(best_next_task)
        remaining_tasks.remove(best_next_task)
    
    return current_order

def create_repo_bar_graph(results_by_task_type: dict[str, pd.DataFrame], output_path: str | None = None, show_average_only: bool = False):
    """Create a bar graph showing repository performance with repos on the x-axis.
    
    Args:
        results_by_task_type: Dictionary mapping task types to dataframes of results
        output_path: Path to save the output graph
        show_average_only: If True, show only the average score across all tools for each repo
                          If False, show individual tool scores for each repo
    """
    # Combine all data from different task types
    all_data = pd.concat(results_by_task_type.values())
    
    # Calculate scores by repo (and tool if needed)
    scores_df = all_data.dropna(subset=['Score'])
    
    if scores_df.empty:
        print("No data available for plotting.")
        return False
    
    # Get all unique repositories
    all_repos = sorted(scores_df['RepoName'].unique())
    
    if show_average_only:
        # Calculate average scores per repository
        repo_scores = scores_df.groupby('RepoName')['Score'].mean().reset_index()
        
        # Sort repositories by score for better visualization
        repo_scores = repo_scores.sort_values('Score', ascending=False)
        sorted_repos = repo_scores['RepoName'].tolist()
        
        # Set up the plot
        plt.figure(figsize=(max(10, len(all_repos) * 1.2), 8))
        
        # Create bars
        plt.bar(
            range(len(sorted_repos)),
            repo_scores['Score'],
            width=0.7,
            color=plt.cm.viridis(np.linspace(0, 0.8, len(sorted_repos))),
            label='Average Score'
        )
        
        # Set up axes and labels
        plt.xlabel('Repository')
        plt.ylabel('Average Score (%)')
        plt.title('Average Performance Across Repositories (All Tools)')
        plt.xticks(range(len(sorted_repos)), sorted_repos, rotation=45, ha='right')
        plt.yticks(np.arange(0, 1.1, 0.1), [f"{int(x*100)}%" for x in np.arange(0, 1.1, 0.1)])
        plt.ylim(0, 1.0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add specific values on top of bars
        for i, score in enumerate(repo_scores['Score']):
            plt.text(i, score + 0.02, f"{score:.1%}", ha='center', va='bottom')
            
        # Add the overall average line
        overall_avg = repo_scores['Score'].mean()
        plt.axhline(y=overall_avg, color='r', linestyle='--', label=f'Overall Average: {overall_avg:.1%}')
        plt.legend()
        
    else:
        # Calculate scores by tool and repo
        tool_repo_scores = scores_df.groupby(['Tool', 'RepoName'])['Score'].mean().reset_index()
        
        # Get all unique tools
        all_tools = sorted(scores_df['Tool'].unique())
        
        # Calculate average score per repo for sorting
        repo_avg = scores_df.groupby('RepoName')['Score'].mean()
        sorted_repos = repo_avg.sort_values(ascending=False).index.tolist()
        
        # Set up the plot
        plt.figure(figsize=(max(12, len(all_repos) * 2), 10))
        
        # Width of each group of bars
        group_width = 0.8
        bar_width = group_width / len(all_tools)
        
        # Create a color map for tools
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_tools)))
        color_map = {tool: color for tool, color in zip(all_tools, colors)}
        
        # Plot bars for each tool and repo combination
        for i, tool in enumerate(all_tools):
            # Get scores for this tool
            tool_data = tool_repo_scores[tool_repo_scores['Tool'] == tool]
            
            # Map to the ordered repos
            tool_scores = []
            for repo in sorted_repos:
                repo_data = tool_data[tool_data['RepoName'] == repo]
                if not repo_data.empty:
                    tool_scores.append(repo_data['Score'].values[0])
                else:
                    tool_scores.append(0)  # No score for this repo
            
            # Position bars within each group
            offset = i * bar_width - group_width/2 + bar_width/2
            plt.bar(
                [x + offset for x in range(len(sorted_repos))],
                tool_scores,
                width=bar_width,
                color=color_map[tool],
                label=tool
            )
        
        # Set up axes and labels
        plt.xlabel('Repository')
        plt.ylabel('Score (%)')
        plt.title('Tool Performance by Repository')
        plt.xticks(range(len(sorted_repos)), sorted_repos, rotation=45, ha='right')
        plt.yticks(np.arange(0, 1.1, 0.1), [f"{int(x*100)}%" for x in np.arange(0, 1.1, 0.1)])
        plt.ylim(0, 1.0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add legend with better placement for many tools
        if len(all_tools) > 6:
            plt.legend(title='Tools', bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            plt.legend(title='Tools', loc='best')
    
    plt.tight_layout()
    
    # Save or show the plot
    if output_path:
        plt.savefig(output_path)
        print(f"Repository bar graph saved to {output_path}")
    else:
        plt.show()
    
    return True

def create_overlaid_radar_charts(results_by_task_type: dict[str, pd.DataFrame], output_path: str | None = None, overall_tool_order: list[str] | None = None):
    """Create a grid of radar charts, with one radar per tool containing all task types overlaid."""
    # Get unique tools across all task types
    all_tools = set()
    for task_df in results_by_task_type.values():
        all_tools.update(task_df['Tool'].unique())
    
    # Use overall tool order if available
    if overall_tool_order:
        tools_in_data = [t for t in overall_tool_order if t in all_tools]
        # Add any missing tools
        tools_in_data.extend([t for t in all_tools if t not in tools_in_data])
        sorted_tools = tools_in_data
    else:
        # Calculate overall average per tool for sorting
        overall_avg = {}
        for tool in all_tools:
            scores = []
            for task_df in results_by_task_type.values():
                tool_scores = task_df[task_df['Tool'] == tool]['Score'].dropna()
                if not tool_scores.empty:
                    scores.extend(tool_scores.tolist())
            overall_avg[tool] = np.mean(scores) if scores else 0
        sorted_tools = sorted(all_tools, key=lambda x: overall_avg.get(x, 0), reverse=True)
    
    # Get task types in sorted order
    task_types = sorted(results_by_task_type.keys())
    
    # Calculate grid dimensions for more reasonable layout
    n_tools = len(sorted_tools)
    n_cols = min(3, n_tools)  # Maximum 3 columns
    n_rows = (n_tools + n_cols - 1) // n_cols  # Ceiling division for number of rows
    
    # Create figure with appropriate size - reduced height for less whitespace
    fig = plt.figure(figsize=(4.5 * n_cols, 3.8 * n_rows))
    
    # Create a color map for task types with distinguishable colors
    task_colors = plt.cm.tab10(np.linspace(0, 1, len(task_types)))
    task_color_map = {task: color for task, color in zip(task_types, task_colors)}
    
    # Get all unique repositories across all task types
    all_repos = set()
    for task_df in results_by_task_type.values():
        all_repos.update(task_df['RepoName'].unique())
    all_repos = sorted(all_repos)
    
    # For creating a single legend for all plots
    legend_handles = []
    legend_labels = []
    
    # Process each tool
    for tool_idx, tool in enumerate(sorted_tools):
        # Create subplot for this tool
        ax = fig.add_subplot(n_rows, n_cols, tool_idx + 1, polar=True)
        
        # Keep track of if any data is plotted
        has_data = False
        
        # Process each task type for this tool
        for task_type in task_types:
            df = results_by_task_type[task_type]
            scores_df = df[df['Tool'] == tool].dropna(subset=['Score'])
            
            if scores_df.empty:
                continue
            
            has_data = True
            
            # Create pivot table of scores by repository
            repos = sorted(scores_df['RepoName'].unique())
            repo_scores = {repo: scores_df[scores_df['RepoName'] == repo]['Score'].mean() 
                          for repo in repos}
            
            # Create angles for the radar chart
            # Use all_repos instead of just the repos for this task_type
            # to ensure consistent axes across all plots
            N = len(all_repos)
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Close the loop
            
            # Get values in the right order
            values = [repo_scores.get(repo, 0) for repo in all_repos]
            values += values[:1]  # Close the loop
            
            # Draw radar for this task type
            line, = ax.plot(angles, values, linewidth=2, linestyle='solid', 
                   color=task_color_map[task_type])
            ax.fill(angles, values, color=task_color_map[task_type], alpha=0.2)
            
            # Only add to legend handles once per task type
            if task_type not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(task_type)
        
        # If no data was plotted for this tool, show a message
        if not has_data:
            ax.text(0, 0, "No Data", ha='center', va='center', fontsize=12)
        else:
            # Add gridlines
            ax.grid(True, color='gray', alpha=0.3)
            
            # Enhance grid with concentric circles
            for r in [0.25, 0.5, 0.75]:
                ax.plot(np.linspace(0, 2*np.pi, 100), [r]*100, color='gray', alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Set up axes with more informative labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(all_repos, size=10)
        ax.set_rlabel_position(45)  # Move radius labels for better visibility
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["25%", "50%", "75%", "100%"], color="grey", size=7)
        ax.set_ylim(0, 1)
        
        # Add title for this radar
        ax.set_title(tool, size=14, y=1.12)
        
        # Improve label position for readability
        for label, angle in zip(ax.get_xticklabels(), angles[:-1]):
            # Set label alignment based on position
            if angle < np.pi/2 or angle > 3*np.pi/2:
                label.set_horizontalalignment('left')
            else:
                label.set_horizontalalignment('right')
            
            # Adjust vertical alignment
            if angle <= np.pi:
                label.set_verticalalignment('bottom')
            else:
                label.set_verticalalignment('top')
                
            # Rotate labels
            label.set_rotation(np.rad2deg(angle))
    
    # Add overall title - move up to reduce space
    task_types_str = ', '.join(task_types)
    if len(task_types_str) > 40:  # If the string is too long, truncate it
        task_types_str = task_types_str[:37] + "..."
    fig.suptitle(f'Tool Performance by Repository Across Task Types: {task_types_str}', size=16, y=0.95)
    
    # Add a single legend for all plots - positioned closer to grid
    if len(task_types) > 1:
        fig.legend(legend_handles, legend_labels, 
                  loc='upper center', bbox_to_anchor=(0.5, 0.93), 
                  ncol=min(len(task_types), 4), fontsize=10)
    
    # Adjust layout with reduced top margin - bring grid closer to legend
    plt.tight_layout(rect=[0, 0, 1, 0.92], pad=0.3, h_pad=0.3, w_pad=0.1)
    
    # Save or show the plot
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Overlaid radar charts saved to {output_path}")
    else:
        plt.show()
    
    return True

def create_patch_size_graph(results_by_task_type: dict[str, pd.DataFrame], output_path: str | None = None, overall_tool_order: list[str] | None = None, args = None):
    """Create a line graph showing tool performance by patch size."""
    # Import TaskType here to avoid circular imports
    from liveswebench.util.tasks import TaskType
    
    # Combine data from all task types
    combined_df = pd.concat(results_by_task_type.values())
    
    # Calculate scores by tool and repo
    scores_df = combined_df.dropna(subset=['Score'])
    
    if scores_df.empty:
        print("No data available for plotting.")
        return False
    
    # Get all unique tools
    all_tools = sorted(scores_df['Tool'].unique())
    
    # Use overall tool order if available
    if overall_tool_order:
        tools_in_data = [t for t in overall_tool_order if t in all_tools]
        # Add any missing tools
        tools_in_data.extend([t for t in all_tools if t not in tools_in_data])
        sorted_tools = tools_in_data
    else:
        sorted_tools = all_tools
    
    # Create a dictionary to store patch sizes and scores for each tool
    tool_data = {}
    for tool in sorted_tools:
        tool_data[tool] = {'sizes': [], 'scores': []}
        
        # Get all tasks for this tool
        tool_tasks = scores_df[scores_df['Tool'] == tool]
        
        for _, row in tool_tasks.iterrows():
            # Extract repo name and task number
            repo_name, task_num = extract_task_components(row['Task'])
            if not repo_name or not task_num:
                continue
                
            try:
                # Get task object and ground truth patch
                # Extract task_type from the original task type dictionary
                current_task_type = None
                for orig_task_type, df in results_by_task_type.items():
                    # Check if this task exists in this dataframe
                    if row['Task'] in df['Task'].values and row['Tool'] in df['Tool'].values:
                        current_task_type = orig_task_type
                        break
                
                if current_task_type is None:
                    print(f"Task {row['Task']} with tool {row['Tool']} not found in any task type")
                    continue
                
                # Convert string task_type to TaskType enum
                task_type_enum = TaskType(current_task_type)
                
                task = load_tasks(task_source=args.task_source)[repo_name][int(task_num)]
                
                # Try to get the ground truth patch
                try:
                    ground_truth_patch = task.get_ground_truth_patch(task_type_enum)
                except ValueError as e:
                    # If the autocomplete_patch is None, try to load inline_autocomplete_patch.patch directly
                    if task_type_enum == TaskType.AUTOCOMPLETE and task.autocomplete_patch is None:
                        inline_patch_path = Path(f"liveswebench/data/tasks/{repo_name}/{task_num}/autocomplete_patch.patch")
                        if inline_patch_path.exists():
                            ground_truth_patch = inline_patch_path.read_text()
                        else:
                            raise ValueError(f"Could not find autocomplete_patch.patch for {repo_name}/{task_num}")
                    else:
                        raise e
                
                # Count additions in the patch
                _, _, additions_per_hunk = count_hunks_in_patch(ground_truth_patch)
                total_additions = sum(additions_per_hunk)
                
                # Store the data
                tool_data[tool]['sizes'].append(total_additions)
                tool_data[tool]['scores'].append(row['Score'])
            except Exception as e:
                print(f"Error processing task {row['Task']} with task type {current_task_type}: {e}")
                continue
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    
    # Create a color map for tools
    colors = plt.cm.tab10(np.linspace(0, 1, len(sorted_tools)))
    color_map = {tool: color for tool, color in zip(sorted_tools, colors)}
    
    # Plot lines for each tool
    for tool in sorted_tools:
        if tool_data[tool]['sizes']:
            # Sort by patch size
            sorted_indices = np.argsort(tool_data[tool]['sizes'])
            sizes = np.array(tool_data[tool]['sizes'])[sorted_indices]
            scores = np.array(tool_data[tool]['scores'])[sorted_indices]
            
            # Calculate moving average
            window_size = 5
            if len(sizes) >= window_size:
                scores_ma = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
                sizes_ma = np.convolve(sizes, np.ones(window_size)/window_size, mode='valid')
                plt.plot(sizes_ma, scores_ma, 'o-', linewidth=2, label=tool, color=color_map[tool], markersize=8)
            else:
                plt.plot(sizes, scores, 'o-', linewidth=2, label=tool, color=color_map[tool], markersize=8)
    
    # Set up axes and labels
    plt.xlabel('Number of Additions in Ground Truth Patch')
    plt.ylabel('Score (%)')
    
    # Create title that includes all task types
    if len(results_by_task_type) == 1:
        title = f'Tool Performance by Patch Size for {list(results_by_task_type.keys())[0]} Tasks'
    else:
        # Create a comma-separated list of task types
        task_types_str = ', '.join(results_by_task_type.keys())
        # If the string is too long, truncate it
        if len(task_types_str) > 70:
            task_types_str = task_types_str[:67] + "..."
        title = f'Tool Performance by Patch Size for Tasks: {task_types_str}'
    plt.title(title)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.yticks(np.arange(0, 1.1, 0.1), [f"{int(x*100)}%" for x in np.arange(0, 1.1, 0.1)])
    plt.ylim(0, 1.0)
    
    # Add legend with better placement for many tools
    if len(sorted_tools) > 6:
        plt.legend(title='Tools', bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.legend(title='Tools', loc='best')
    
    plt.tight_layout()
    
    # Save or show the plot
    if output_path:
        plt.savefig(output_path)
        print(f"Patch size graph saved to {output_path}")
    else:
        plt.show()
    
    return True

def main():
    args = parse_arguments()
    results_by_task_type = {}

    # Process each CSV file
    for csv_file in args.csv_files:
        result = process_csv_file(csv_file, args)
        if result:
            results_by_task_type[result['task_type']] = result['filtered_df']
    
    # Calculate and display results for each task type
    task_type_scores = {}
    for task_type, filtered_df in results_by_task_type.items():
        print(f"\n======= RESULTS FOR {task_type.upper()} TASKS =======")
        
        if not filtered_df.empty:
            # Ensure we only include rows where Score is not NaN for each tool
            scores_df = filtered_df.dropna(subset=['Score'])
            
            if not scores_df.empty:
                # Calculate average scores and sort by score (descending)
                tool_scores = scores_df.groupby('Tool')['Score'].mean().reset_index()
                tool_scores = tool_scores.sort_values('Score', ascending=False)
                
                # Store scores for graph
                task_type_scores[task_type] = {row['Tool']: row['Score'] for _, row in tool_scores.iterrows()}
                
                # Format for printing
                tool_scores['Score'] = tool_scores['Score'].apply(lambda x: f"{x:.2%}")

                print("\nAverage Scores by Tool:")
                for _, row in tool_scores.iterrows():
                    print(f"{row['Tool']}: {row['Score']}")

                # Use scores_df (without NaNs) for the overall average
                overall_avg = scores_df['Score'].mean()
                print(f"\nOverall Average: {overall_avg:.2%}")

                # Count tasks used in calculation (only those with scores)
                tasks_used = scores_df['Task'].nunique()
                print(f"\nTasks used in calculation: {tasks_used}")
                print("Tasks included:")
                for task in scores_df['Task'].unique():
                    print(f"- {task}")
                
                # Output CSV file if requested
                if args.output_csv:
                    # Create a new DataFrame with just Tool and raw Score (not formatted as percentage)
                    output_df = scores_df.groupby('Tool')['Score'].mean().reset_index()
                    output_df = output_df.sort_values('Score', ascending=False)
                    output_filename = f"{task_type}_results.csv"
                    output_df.to_csv(output_filename, index=False)
                    print(f"CSV results saved to {output_filename}")
            else:
                print("No valid scores found for this task type after filtering.")
        else:
            print("No data found matching the specified criteria for this task type.")
    
    # Calculate and display overall results across all task types
    overall_tool_order = []
    if results_by_task_type:
        print("\n======= OVERALL RESULTS ACROSS ALL TASK TYPES =======")
        all_data = pd.concat(results_by_task_type.values())
        all_scores = all_data.dropna(subset=['Score'])
        
        if not all_scores.empty:
            # Calculate average scores and sort by score (descending)
            overall_tool_scores = all_scores.groupby('Tool')['Score'].mean().reset_index()
            overall_tool_scores = overall_tool_scores.sort_values('Score', ascending=False)
            
            # Store the overall tool order for the graph
            overall_tool_order = overall_tool_scores['Tool'].tolist()
            
            # Format for printing
            overall_tool_scores['Score'] = overall_tool_scores['Score'].apply(lambda x: f"{x:.2%}")
            
            print("\nAverage Scores by Tool (All Task Types):")
            for _, row in overall_tool_scores.iterrows():
                print(f"{row['Tool']}: {row['Score']}")
            
            overall_avg = all_scores['Score'].mean()
            print(f"\nOverall Average Across All Task Types: {overall_avg:.2%}")
            
            task_types_by_tool = {}
            for tool in all_scores['Tool'].unique():
                task_types_with_tool = []
                for task_type, df in results_by_task_type.items():
                    tool_df = df[df['Tool'] == tool].dropna(subset=['Score'])
                    if not tool_df.empty:
                        task_types_with_tool.append(task_type)
                task_types_by_tool[tool] = task_types_with_tool
            
            print("\nTask Types by Tool:")
            # Sort by the same order as the overall scores
            for tool in overall_tool_scores['Tool']:
                if tool in task_types_by_tool:
                    print(f"{tool}: {', '.join(task_types_by_tool[tool])}")
            
            # Output overall CSV file if requested
            if args.output_csv:
                # Create a new DataFrame with just Tool and raw Score (before formatting)
                output_df = all_scores.groupby('Tool')['Score'].mean().reset_index()
                output_df = output_df.sort_values('Score', ascending=False)
                output_filename = "overall_results.csv"
                output_df.to_csv(output_filename, index=False)
                print(f"Overall CSV results saved to {output_filename}")
    
    # Generate graph if output path is provided
    if args.output_graph:
        # If repo-bar graph type is requested
        if args.graph_type == 'repo-bar':
            create_repo_bar_graph(
                results_by_task_type,
                args.output_graph,
                args.show_average_only
            )
            print(f"Created repository bar graph with {'average scores' if args.show_average_only else 'scores by tool'}")
        # Choose visualization type based on graph_type and number of task types
        elif len(results_by_task_type) <= 1:
            # Single task type - use requested graph type and adjust parameters if needed
            if args.graph_type == 'bar':
                if args.split_by_repo:
                    create_repo_split_bar_graph(
                        results_by_task_type, 
                        args.output_graph, 
                        overall_tool_order
                    )
                    print(f"Created bar chart visualization with repositories for {list(results_by_task_type.keys())[0]} tasks")
                else:
                    create_bar_graph(
                        results_by_task_type, 
                        args.output_graph, 
                        task_type_scores, 
                        overall_tool_order,
                        False
                    )
                    print(f"Created bar chart visualization for {list(results_by_task_type.keys())[0]} tasks")
            elif args.graph_type == 'line':
                if not args.split_by_repo:
                    print("Assuming --split-by-repo for line chart with single task type")
                create_repo_line_chart(
                    results_by_task_type, 
                    args.output_graph, 
                    overall_tool_order
                )
                print(f"Created line chart visualization with repositories for {list(results_by_task_type.keys())[0]} tasks")
            elif args.graph_type == 'radar':
                if not args.split_by_repo:
                    print("Assuming --split-by-repo for radar chart (radar charts require repository comparisons)")
                create_repo_radar_chart(
                    results_by_task_type, 
                    args.output_graph, 
                    overall_tool_order
                )
                print(f"Created radar chart visualization with repositories for {list(results_by_task_type.keys())[0]} tasks")
            elif args.graph_type == 'patch-size':
                create_patch_size_graph(
                    results_by_task_type,
                    args.output_graph,
                    overall_tool_order,
                    args
                )
                if len(results_by_task_type) == 1:
                    print(f"Created patch size visualization for {list(results_by_task_type.keys())[0]} tasks")
                else:
                    task_types_str = ', '.join(results_by_task_type.keys())
                    print(f"Created patch size visualization combining data from: {task_types_str}")
        else:
            # Multiple task types - respect the graph_type argument, adjust if needed
            if args.graph_type == 'bar':
                create_bar_graph(
                    results_by_task_type, 
                    args.output_graph, 
                    task_type_scores, 
                    overall_tool_order,
                    False
                )
                print(f"Created bar chart visualization with {len(results_by_task_type)} task types")
            elif args.graph_type == 'line':
                create_line_graph(
                    results_by_task_type, 
                    args.output_graph, 
                    overall_tool_order
                )
                print(f"Created line chart visualization with {len(results_by_task_type)} task types")
            elif args.graph_type == 'radar':
                if args.split_by_task_type:
                    # Generate overlaid radars with one chart per tool
                    create_overlaid_radar_charts(
                        results_by_task_type,
                        args.output_graph,
                        overall_tool_order
                    )
                    print(f"Created overlaid radar charts with task types combined per tool")
                else:
                    # Default behavior: average scores across task types
                    if len(results_by_task_type) > 1:
                        print("Creating radar charts with averaged scores across task types. Use --split-by-task-type for separate charts per task type.")
                        
                        # Combine data from all task types into a single DataFrame
                        combined_df = pd.concat(results_by_task_type.values())
                        
                        # Average scores for each tool/repo combination
                        combined_scores = combined_df.groupby(['Tool', 'RepoName'])['Score'].mean().reset_index()
                        
                        # Create a new dataframe with the combined data
                        task_types_str = ', '.join(results_by_task_type.keys())
                        combined_task_results = {f'{task_types_str}': pd.DataFrame({
                            'Task': combined_scores['RepoName'] + ' - 0',  # Dummy task number
                            'Tool': combined_scores['Tool'],
                            'Score': combined_scores['Score'],
                            'RepoName': combined_scores['RepoName'],
                            'TaskNum': '0'  # Dummy task number
                        })}
                        
                        # Create radar chart with the combined data
                        create_repo_radar_chart(
                            combined_task_results, 
                            args.output_graph, 
                            overall_tool_order
                        )
                        print(f"Created radar chart visualization using combined data from: {task_types_str}")
                    else:
                        if not args.split_by_repo:
                            print("Assuming --split-by-repo for radar chart (radar charts require repository comparisons)")
                        create_repo_radar_chart(
                            results_by_task_type, 
                            args.output_graph, 
                            overall_tool_order
                        )
                        print(f"Created radar chart visualization with repositories for {list(results_by_task_type.keys())[0]} tasks")
            elif args.graph_type == 'patch-size':
                create_patch_size_graph(
                    results_by_task_type,
                    args.output_graph,
                    overall_tool_order,
                    args
                )
                task_types_str = ', '.join(results_by_task_type.keys())
                print(f"Created patch size visualization combining data from: {task_types_str}")

if __name__ == "__main__":
    main()