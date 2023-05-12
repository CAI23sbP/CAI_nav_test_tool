# Transform data and calculate metrics

To transform the dataset for later plotting and calculate the metrics from the recorded data run `python get_metrics.py --dir <DIR>`, whereas `dir` is the directory which is created in the recording phase. The metrics which are created are shown in the following table:

| Name                 | Datatype                             | Description                                                                                                                               |
| -------------------- | ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------- |
| curvature            | Float[]                              | The curvature of the planner for each <br>timestep calculated with the [menger curvature](https://en.wikipedia.org/wiki/Menger_curvature) |
| normalized curvature | Float[]                              | The curvature multiplied by the length of the<br> path for this specific part.                                                            |
| roughness            | Float[]                              | Describes how sudden and abrupt the planner changes directions.                                                                           |
| path length          | Float                                | The complete length of the part                                                                                                           |
| path length values   | Float[]                              | The length of each path between two continuous timestamps                                                                                 |
| acceleration         | Float[]                              | The acceleration of the robot. Calculated as the gradient between two velocities.                                                         |
| jerk                 | Float[]                              | Describes the change in acceleration.                                                                                                     |
| velocity             | Float[][]                            | The real velocity of the robot.                                                                                                           |
| cmd_vel              | Float[][]                            | The robots desired velocity denoted by the planner                                                                                        |
| collision amount     | Int                                  | Absolute amount of collisions in an episode.                                                                                              |
| collisions           | Int[]                                | Index of the positions in which a collision occured.                                                                                      |
| path                 | Float[][]                            | Array of positions in which the robot was located for specific timestamps.                                                                |
| angle over length    | Float                                | The complete angle over the complete length of the path the robot took.                                                                   |
| time diff            | Int                                  | The complete time of the episode.                                                                                                         |
| result               | TIMEOUT \| GOAL_REACHED \| COLLISION | The reason the episode has ended.                                                                                                         |

# Plot Data

In order to make plotting easy, the plots are created from a declaration file, in which the exaclt data you want to plot is described. The declaration file should have the following schema, which is also shown in `plot_declarations/sample_schema.yaml`:

```yaml
# Wether you want to show or save the plots
show_plots: boolean
# Name of the directory in ./path
save_location: string

# List of all datasets that should be compared
# Name of the directory in ./data
datasets: string[]

# Wether you want to plot the result counts
results:
    # Should plot?
    plot: boolean
    # Title of the plot
    title: string
    # Name of the file the plot should be saved ot
    save_name: string
    # Denotes which data should be shown seperately for a single planner
    differentiate: key in Dataset
    # Additional Plot arguments
    plot_args: {} # Optional


# Plot values that are collected in every time step.
# Thus, being arrays for each episode.
# Possible values are:
# - curvature
# - normalized_curvature
# - roughness
# - path_length_values
# - acceleration
# - jerk
# - velocity

#  It is possible to plot
#  - A line plot to show the course in a single episode
#    You can list multiple value to create multiple plots
single_episode_line:
  # Name of the coloumn you want to plot
  - data_key: string # Required
    # Number of values that should be skipped to reduce datapoints
    step_size: int # Optional -> Defaults to 5
    # Coloumn for differentiation
    # Denotes which data should be shown seperately for a single planner
    differentiate: key in Dataset
    # Index of the episode -> If none all episodes are plotted
    episode: int # Optional -> Defaults to none
    title: string
    save_name: string
    plot_args: {} # Optional
# - A Distributional plot for a single episode
#   You can list multiple value to create multiple plots
single_episode_distribution:
  - data_key: string
    episode: int
    # Denotes which data should be shown seperately for a single planner
    differentiate: key in Dataset
    plot_key: "swarm" | "violin" | "box" | "boxen" | "strip" # Optional -> Defaults to "swarm"
    title: string
    save_name: string
    plot_args: {} # Optional
# - A line plot showing aggregated values for all episodes.
#   Like a line plot for the max value of each episode
aggregated_distribution:
  - data_key: string
    # Denotes which data should be shown seperately for a single planner
    differentiate: key in Dataset
    # Function that should be used for aggregation. We offer: max, min, mean
    aggregate: "max" | "min" | "mean" | "sum"
    # Name of the dist plot you want to use. Can be strip, swarm, box, boxen, violin
    plot_key: "swarm" | "violin" | "box" | "boxen" | "strip" # Optional -> Defaults to "swarm"
    title: string
    save_name: string
    plot_args: {} # Optional
# - A distributional plot for aggregated values for all episodes.
aggregated_line:
  - data_key: string
    # Denotes which data should be shown seperately for a single planner
    differentiate: key in Dataset
    # Function that should be used for aggregation. We offer: max, min, mean
    aggregate: "max" | "min" | "mean" | "sum"
    title: string
    save_name: string
    plot_args: {} # Optional


## Plot values that are collected for each episode.
# Single values for each episode
# Possible values are:
# - time_diff
# - angle_over_length
# - path_length

# It is possible to plot
# - A categorical plot over all episodes to show the values in a line or bar plot
all_episodes_categorical:
  - data_key: string
    # Denotes which data should be shown seperately for a single planner
    differentiate: key in Dataset
    plot_key: "line" | "bar"
    title: string
    save_name: string
    plot_args: {} # Optional
# - Plot a distribution over all episodes
all_episodes_distribution:
  - data_key: string
    # Denotes which data should be shown seperately for a single planner
    differentiate: key in Dataset
    plot_key: "swarm" | "violin" | "box" | "boxen" | "strip" # Optional -> Defaults to "swarm"
    title: string
    save_name: string
    plot_args: {} # Optional


## Plot the path the robots took

# Plot all paths of all episodes for each robot
episode_plots_for_namespaces:
    # list of desired results that should be plotted
    desired_results: ("TIMEOUT" | "GOAL_REACHED" | "COLLISION")[]
    # Wether or not to add the obstacles from the scenario file to the plot
    should_add_obstacles: boolean # Optional -> Defaults to False
    # Wether or not to mark where collisions happened
    should_add_collisions: boolean # Optional -> Defaults to False
    title: string
    save_name: string

# Plot the best path of each robot
# Only select the paths that reached the goal and take the path that took the least amount of time
create_best_plots:
    # Wether or not to add the obstacles from the scenario file to the plot
    should_add_obstacles: boolean # Optional -> Defaults to False
    # Wether or not to mark where collisions happened
    should_add_collisions: boolean # Optional -> Defaults to False
    title: string
    save_name: string

```
