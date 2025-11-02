"""Main entry point for the maze pathfinding GA."""

from typing import Any

from ga_maze_pathfinding.ga_solver import GASolver
from ga_maze_pathfinding.maze import Maze


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


def main(mane_name: str = "simple") -> None:
    """Run the genetic algorithm maze solver."""
    if mane_name == "simple":
        maze = Maze.create_simple_maze()
        run_ga_solver(
            maze=maze,
            population_size=300,
            max_generations=100,
            max_path_length=50,
            crossover_prob=0.7,
            mutation_prob=0.2,
            tournament_size=3,
        )
    elif mane_name == "complex":
        complex_maze = Maze.create_complex_maze()
        run_ga_solver(
            maze=complex_maze,
            population_size=500,
            max_generations=200,
            max_path_length=100,
            crossover_prob=0.7,
            mutation_prob=0.2,
            tournament_size=5,
        )
    else:
        raise ValueError(f"Unknown maze name: {mane_name}")


if __name__ == "__main__":
    maze_name = "complex"
    main(mane_name=maze_name)
