"""Main entry point for the maze pathfinding GA."""

import argparse
from typing import Any

from ga_maze_pathfinding.ga_solver import GASolver
from ga_maze_pathfinding.maze import Maze
from ga_maze_pathfinding.maze_configs import mazes


def run_ga_solver(maze: Maze, **kwargs: Any) -> None:
    """Run the genetic algorithm maze solver on the given maze."""
    print("=" * 60)
    print("SIMPLE MAZE")
    print("=" * 60)
    print("\nMaze:")
    print(maze)
    print()

    solver = GASolver(
        maze=maze,
        **kwargs,
    )

    best_path, stats = solver.solve(verbose=True)

    # Visualize the result
    print("\n" + "=" * 60)
    print("SOLUTION PATH")
    print("=" * 60)
    print(solver.visualize_path(best_path))
    print(f"\nPath directions: {[d.value for d in best_path[:30]]}")
    if len(best_path) > 30:
        print(f"... (showing first 30 of {len(best_path)} moves)")


metaparameters = {
    "simple5x5": {
        "population_size": 300,
        "max_generations": 100,
        "max_path_length": 50,
        "crossover_prob": 0.7,
        "mutation_prob": 0.2,
        "tournament_size": 3,
    },
    "simple5x6": {
        "population_size": 300,
        "max_generations": 300,
        "max_path_length": 50,
        "crossover_prob": 0.7,
        "mutation_prob": 0.2,
        "tournament_size": 3,
    },
    "complex10x10": {
        "population_size": 500,
        "max_generations": 500,
        "max_path_length": 100,
        "crossover_prob": 0.7,
        "mutation_prob": 0.2,
        "tournament_size": 5,
    },
}


def main(maze_name: str = "simple5x5") -> None:
    """Run the genetic algorithm maze solver."""

    maze = mazes.get(maze_name)
    if maze is None:
        raise ValueError(f"Maze '{maze_name}' not found. Available mazes: {list(mazes.keys())}")

    config = metaparameters.get(maze_name, {})
    run_ga_solver(
        maze=maze,
        **config,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the genetic algorithm maze solver.")
    available_mazes = list(mazes.keys())
    parser.add_argument(
        "--maze",
        type=str,
        default=available_mazes[0],
        choices=available_mazes,
        help="Maze configuration to use",
    )
    args = parser.parse_args()
    maze_name = args.maze
    main(maze_name=maze_name)
