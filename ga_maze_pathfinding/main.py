"""Main entry point for the maze pathfinding GA."""

from ga_maze_pathfinding.ga_solver import GASolver
from ga_maze_pathfinding.maze import Maze


def main() -> None:
    """Run the genetic algorithm maze solver."""
    # Create a maze
    print("=" * 60)
    print("SIMPLE MAZE")
    print("=" * 60)
    maze = Maze.create_simple_maze()
    print("\nMaze:")
    print(maze)
    print()

    # Create and run the solver
    solver = GASolver(
        maze=maze,
        population_size=300,
        max_generations=100,
        max_path_length=50,
        crossover_prob=0.7,
        mutation_prob=0.2,
        tournament_size=3,
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

    # Try complex maze
    print("\n\n" + "=" * 60)
    print("COMPLEX MAZE")
    print("=" * 60)
    complex_maze = Maze.create_complex_maze()
    print("\nMaze:")
    print(complex_maze)
    print()

    # Create and run the solver for complex maze
    complex_solver = GASolver(
        maze=complex_maze,
        population_size=500,
        max_generations=200,
        max_path_length=100,
        crossover_prob=0.7,
        mutation_prob=0.2,
        tournament_size=5,
    )

    best_path_complex, stats_complex = complex_solver.solve(verbose=True)

    # Visualize the result
    print("\n" + "=" * 60)
    print("SOLUTION PATH")
    print("=" * 60)
    print(complex_solver.visualize_path(best_path_complex))
    print(f"\nPath directions: {[d.value for d in best_path_complex[:30]]}")
    if len(best_path_complex) > 30:
        print(f"... (showing first 30 of {len(best_path_complex)} moves)")


if __name__ == "__main__":
    main()
