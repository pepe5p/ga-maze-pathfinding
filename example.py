#!/usr/bin/env python3
"""Quick example of using the GA maze pathfinding solver."""

from ga_maze_pathfinding.ga_solver import GASolver
from ga_maze_pathfinding.maze import Maze, Position

# Example 1: Use a pre-built maze
print("Example 1: Simple maze")
print("-" * 60)
maze = Maze.create_simple_maze()
print(maze)
print()

solver = GASolver(
    maze=maze,
    population_size=200,
    max_generations=50,
    max_path_length=40,
)

best_path, stats = solver.solve(verbose=False)
print("\nSolution found!")
print(solver.visualize_path(best_path))
print(f"\nBest fitness: {stats['best_fitness']}")
print(f"Path: {[d.value for d in best_path[:20]]}...")

# Example 2: Create a custom maze
print("\n\nExample 2: Custom maze")
print("-" * 60)
custom_grid = [
    [0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 1, 0],
    [0, 1, 0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0, 1, 0],
    [0, 1, 0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 0],
]

custom_maze = Maze(
    grid=custom_grid,
    start=Position(0, 0),
    end=Position(6, 6),
)

print(custom_maze)
print()

custom_solver = GASolver(
    maze=custom_maze,
    population_size=300,
    max_generations=100,
    max_path_length=50,
)

best_path, stats = custom_solver.solve(verbose=False)
print("\nSolution found!")
print(custom_solver.visualize_path(best_path))
print(f"\nBest fitness: {stats['best_fitness']}")

