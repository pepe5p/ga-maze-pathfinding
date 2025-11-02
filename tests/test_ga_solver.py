"""Tests for GA solver."""

from ga_maze_pathfinding.ga_solver import GASolver
from ga_maze_pathfinding.maze import Direction, Maze


def test_ga_solver_initialization() -> None:
    """Test GA solver initialization."""
    maze = Maze.create_simple_maze()
    solver = GASolver(maze=maze, population_size=50, max_generations=10, max_path_length=30)

    assert solver.maze == maze
    assert solver.population_size == 50
    assert solver.max_generations == 10
    assert solver.max_path_length == 30


def test_ga_solver_runs() -> None:
    """Test that GA solver runs without errors."""
    maze = Maze.create_simple_maze()
    solver = GASolver(maze=maze, population_size=50, max_generations=10, max_path_length=30)

    best_path, stats = solver.solve(verbose=False)

    assert len(best_path) == 30  # Should return a path of max_path_length
    assert "best_fitness" in stats
    assert "logbook" in stats


def test_evaluate_individual() -> None:
    """Test individual evaluation."""
    maze = Maze.create_simple_maze()
    solver = GASolver(maze=maze, population_size=50, max_generations=10, max_path_length=20)

    # Create a simple path that moves right and down
    path = [Direction.RIGHT] * 4 + [Direction.DOWN] * 4
    fitness = solver._evaluate_individual(path)

    # Should return a tuple of two values
    assert isinstance(fitness, tuple)
    assert len(fitness) == 2


def test_visualize_path() -> None:
    """Test path visualization."""
    maze = Maze.create_simple_maze()
    solver = GASolver(maze=maze)

    # Create a simple path
    path = [Direction.RIGHT] * 4 + [Direction.DOWN] * 4

    visualization = solver.visualize_path(path)

    # Should return a string
    assert isinstance(visualization, str)
    assert "S" in visualization  # Start marker
    assert "E" in visualization  # End marker


def test_evaluate_individual_reaches_goal() -> None:
    """Test evaluation when individual reaches the goal."""
    maze = Maze.create_simple_maze()
    solver = GASolver(maze=maze, max_path_length=50)

    # Create a path that reaches the goal
    # Move right 4 times, down 4 times to reach (4, 4)
    path = [Direction.RIGHT, Direction.DOWN] * 8
    fitness = solver._evaluate_individual(path)

    # Should have reached the goal with minimal fitness
    assert fitness[0] >= 0  # Collision penalty
    assert fitness[1] < 50  # Path length less than max


def test_evaluate_individual_with_collisions() -> None:
    """Test evaluation with wall collisions."""
    maze = Maze.create_simple_maze()
    solver = GASolver(maze=maze, max_path_length=20)

    # Create a path that hits walls (go up from start immediately)
    path = [Direction.UP] * 10
    fitness = solver._evaluate_individual(path)

    # Should have high collision penalty
    assert fitness[0] > 0  # Should have collisions


def test_mutate_individual() -> None:
    """Test individual mutation."""
    maze = Maze.create_simple_maze()
    solver = GASolver(maze=maze)

    # Create an individual
    individual = [Direction.RIGHT] * 10

    # Mutate with high probability
    mutated = solver._mutate_individual(individual, indpb=1.0)

    # Should return a tuple
    assert isinstance(mutated, tuple)
    assert len(mutated) == 1


def test_visualize_path_with_goal_reached() -> None:
    """Test visualization when goal is reached."""
    maze = Maze.create_simple_maze()
    solver = GASolver(maze=maze)

    # Create a path that should reach the goal
    path = [Direction.RIGHT] * 4 + [Direction.DOWN] * 4

    visualization = solver.visualize_path(path)

    # Check that it shows the path
    assert "*" in visualization
    assert "Reached goal" in visualization
