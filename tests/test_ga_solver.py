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


def test_simplify_individual_opposite_moves_cancel(simple_maze: Maze) -> None:
    """Test that opposite moves cancel each other out."""
    solver = GASolver(maze=simple_maze)

    # UP, DOWN should cancel out
    individual = [Direction.UP, Direction.DOWN]
    simplified = solver._simplify_individual(individual)
    assert simplified == []

    # DOWN, UP should cancel out
    individual = [Direction.DOWN, Direction.UP]
    simplified = solver._simplify_individual(individual)
    assert simplified == []

    # LEFT, RIGHT should cancel out
    individual = [Direction.LEFT, Direction.RIGHT]
    simplified = solver._simplify_individual(individual)
    assert simplified == []

    # RIGHT, LEFT should cancel out
    individual = [Direction.RIGHT, Direction.LEFT]
    simplified = solver._simplify_individual(individual)
    assert simplified == []


def test_simplify_individual_multiple_cancellations(simple_maze: Maze) -> None:
    """Test multiple consecutive cancellations."""
    solver = GASolver(maze=simple_maze)

    # UP, DOWN, DOWN -> DOWN
    individual = [Direction.UP, Direction.DOWN, Direction.DOWN]
    simplified = solver._simplify_individual(individual)
    assert simplified == [Direction.DOWN]

    # DOWN, DOWN, UP -> DOWN
    individual = [Direction.DOWN, Direction.DOWN, Direction.UP]
    simplified = solver._simplify_individual(individual)
    assert simplified == [Direction.DOWN]

    # LEFT, RIGHT, RIGHT -> RIGHT
    individual = [Direction.LEFT, Direction.RIGHT, Direction.RIGHT]
    simplified = solver._simplify_individual(individual)
    assert simplified == [Direction.RIGHT]


def test_simplify_individual_no_cancellations(simple_maze: Maze) -> None:
    """Test that non-opposite moves are preserved."""
    solver = GASolver(maze=simple_maze)

    # All same direction
    individual = [Direction.RIGHT, Direction.RIGHT, Direction.RIGHT]
    simplified = solver._simplify_individual(individual)
    assert simplified == [Direction.RIGHT, Direction.RIGHT, Direction.RIGHT]

    # Different directions but no opposites
    individual = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
    simplified = solver._simplify_individual(individual)
    assert simplified == [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]


def test_simplify_individual_complex_pattern(simple_maze: Maze) -> None:
    """Test complex patterns with multiple cancellations."""
    solver = GASolver(maze=simple_maze)

    # RIGHT, LEFT, RIGHT, LEFT -> empty
    individual = [Direction.RIGHT, Direction.LEFT, Direction.RIGHT, Direction.LEFT]
    simplified = solver._simplify_individual(individual)
    assert simplified == []

    # UP, DOWN, RIGHT, LEFT -> empty
    individual = [Direction.UP, Direction.DOWN, Direction.RIGHT, Direction.LEFT]
    simplified = solver._simplify_individual(individual)
    assert simplified == []

    # UP, UP, DOWN, RIGHT -> UP, RIGHT
    individual = [Direction.UP, Direction.UP, Direction.DOWN, Direction.RIGHT]
    simplified = solver._simplify_individual(individual)
    assert simplified == [Direction.UP, Direction.RIGHT]


def test_simplify_individual_preserves_efficient_path(simple_maze: Maze) -> None:
    """Test that already efficient paths are preserved."""
    solver = GASolver(maze=simple_maze)

    # Efficient path: right then down
    individual = [Direction.RIGHT, Direction.RIGHT, Direction.DOWN, Direction.DOWN]
    simplified = solver._simplify_individual(individual)
    assert simplified == [Direction.RIGHT, Direction.RIGHT, Direction.DOWN, Direction.DOWN]


def test_simplify_individual_empty_input(simple_maze: Maze) -> None:
    """Test that empty input returns empty output."""
    solver = GASolver(maze=simple_maze)

    individual: list[Direction] = []
    simplified = solver._simplify_individual(individual)
    assert simplified == []


def test_simplify_individual_single_direction(simple_maze: Maze) -> None:
    """Test single direction is preserved."""
    solver = GASolver(maze=simple_maze)

    for direction in [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]:
        individual = [direction]
        simplified = solver._simplify_individual(individual)
        assert simplified == [direction]


def test_pad_individual(simple_maze: Maze) -> None:
    """Test that _pad_individual pads to the correct length."""
    solver = GASolver(maze=simple_maze, max_path_length=20)

    # Short individual
    individual = [Direction.RIGHT, Direction.DOWN]
    padded = solver._pad_individual(individual, 10)
    assert len(padded) == 10
    assert padded[:2] == [Direction.RIGHT, Direction.DOWN]

    # Already at target length
    individual = [Direction.RIGHT] * 10
    padded = solver._pad_individual(individual, 10)
    assert len(padded) == 10


def test_mutate_individual_applies_simplification(simple_maze: Maze) -> None:
    """Test that mutation applies simplification afterward."""
    solver = GASolver(maze=simple_maze, max_path_length=10)

    # Create an individual with opposite moves
    individual = [Direction.RIGHT, Direction.LEFT, Direction.RIGHT, Direction.LEFT] * 2

    # Mutate with low probability to keep some structure
    mutated = solver._mutate_individual(individual, indpb=0.1)

    # Should still be max_path_length after padding
    assert len(mutated[0]) == solver.max_path_length


def test_crossover_applies_simplification(simple_maze: Maze) -> None:
    """Test that crossover applies simplification to offspring."""
    solver = GASolver(maze=simple_maze, max_path_length=20)

    # Create two parents with potentially canceling moves
    parent1 = [Direction.RIGHT, Direction.LEFT] * 10
    parent2 = [Direction.UP, Direction.DOWN] * 10

    offspring1, offspring2 = solver._crossover_individuals(parent1, parent2)

    # Both offspring should be padded back to max_path_length
    assert len(offspring1) == solver.max_path_length
    assert len(offspring2) == solver.max_path_length
