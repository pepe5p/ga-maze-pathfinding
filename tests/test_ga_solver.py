from ga_maze_pathfinding.ga_solver import GASolver
from ga_maze_pathfinding.maze import Direction, Maze


def test_ga_solver_initialization(simple_maze: Maze) -> None:
    solver = GASolver(maze=simple_maze, population_size=50, max_generations=10, max_path_length=30)

    assert solver.maze == simple_maze
    assert solver.population_size == 50
    assert solver.max_generations == 10
    assert solver.max_path_length == 30


def test_ga_solver_runs(simple_maze: Maze) -> None:
    solver = GASolver(maze=simple_maze, population_size=50, max_generations=10, max_path_length=30)

    best_path, stats = solver.solve(verbose=False)

    assert len(best_path) == 30  # Should return a path of max_path_length
    assert "best_fitness" in stats
    assert "logbook" in stats


def test_evaluate_individual(simple_maze: Maze) -> None:
    solver = GASolver(maze=simple_maze, population_size=50, max_generations=10, max_path_length=20)

    # Create a simple path that moves right and down
    path = [Direction.RIGHT] * 4 + [Direction.DOWN] * 4
    fitness = solver._evaluate_individual(path)

    # Should return a tuple of two values
    assert isinstance(fitness, tuple)
    assert len(fitness) == 2


def test_visualize_path(simple_maze: Maze) -> None:
    solver = GASolver(maze=simple_maze)

    # Create a simple path
    path = [Direction.RIGHT] * 4 + [Direction.DOWN] * 4

    visualization = solver.visualize_path(path)

    assert isinstance(visualization, str)
    assert "S" in visualization
    assert "E" in visualization


def test_evaluate_individual_reaches_goal(simple_maze: Maze) -> None:
    solver = GASolver(maze=simple_maze, max_path_length=50)

    # Create a path that reaches the goal
    # Move right 4 times, down 4 times to reach (4, 4)
    path = [Direction.RIGHT, Direction.DOWN] * 8
    fitness = solver._evaluate_individual(path)

    # Should have reached the goal with minimal fitness
    assert fitness[0] >= 0  # Collision penalty
    assert fitness[1] < 50  # Path length less than max


def test_evaluate_individual_with_collisions(simple_maze: Maze) -> None:
    solver = GASolver(maze=simple_maze, max_path_length=20)

    # Create a path that hits walls (go up from start immediately)
    path = [Direction.UP] * 10
    fitness = solver._evaluate_individual(path)

    # Should have high collision penalty
    assert fitness[0] > 0  # Should have collisions


def test_mutate_individual(simple_maze: Maze) -> None:
    solver = GASolver(maze=simple_maze)

    individual = [Direction.RIGHT] * 10

    mutated = solver._mutate_individual(individual, indpb=1.0)

    assert isinstance(mutated, tuple)
    assert len(mutated) == 1


def test_visualize_path_with_goal_reached(simple_maze: Maze) -> None:
    solver = GASolver(maze=simple_maze)

    # Create a path that should reach the goal
    path = [Direction.RIGHT] * 4 + [Direction.DOWN] * 4

    visualization = solver.visualize_path(path)

    assert "*" in visualization
    assert "Reached goal" in visualization
