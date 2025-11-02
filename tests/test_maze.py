"""Tests for maze module."""

from ga_maze_pathfinding.maze import Direction, Maze, Position


def test_maze_creation() -> None:
    """Test creating a simple maze."""
    maze = Maze.create_simple_maze()
    assert maze.rows == 5
    assert maze.cols == 5
    assert maze.start == Position(0, 0)
    assert maze.end == Position(4, 4)


def test_position_validation() -> None:
    """Test position validation."""
    maze = Maze.create_simple_maze()

    assert maze.is_valid_position(Position(0, 0)) is True
    assert maze.is_valid_position(Position(4, 4)) is True
    assert maze.is_valid_position(Position(-1, 0)) is False
    assert maze.is_valid_position(Position(0, -1)) is False
    assert maze.is_valid_position(Position(5, 0)) is False
    assert maze.is_valid_position(Position(0, 5)) is False


def test_wall_detection() -> None:
    """Test wall detection."""
    maze = Maze.create_simple_maze()

    # Position (1, 1) should be a wall
    assert maze.is_wall(Position(1, 1)) is True
    # Position (0, 0) should not be a wall
    assert maze.is_wall(Position(0, 0)) is False


def test_movement() -> None:
    """Test movement in different directions."""
    maze = Maze.create_simple_maze()
    start = Position(2, 2)

    # Test all directions
    up = maze.move(start, Direction.UP)
    assert up == Position(1, 2)

    down = maze.move(start, Direction.DOWN)
    assert down == Position(3, 2)

    left = maze.move(start, Direction.LEFT)
    assert left == Position(2, 1)

    right = maze.move(start, Direction.RIGHT)
    assert right == Position(2, 3)


def test_manhattan_distance() -> None:
    """Test Manhattan distance calculation."""
    maze = Maze.create_simple_maze()

    # Distance from start to end
    dist = maze.manhattan_distance(maze.start)
    assert dist == 8  # |0-4| + |0-4| = 8

    # Distance from end to end
    dist_end = maze.manhattan_distance(maze.end)
    assert dist_end == 0


def test_complex_maze_creation() -> None:
    """Test creating a complex maze."""
    maze = Maze.create_complex_maze()
    assert maze.rows == 10
    assert maze.cols == 10
    assert maze.start == Position(0, 0)
    assert maze.end == Position(9, 9)


def test_maze_string_representation() -> None:
    """Test string representation of maze."""
    maze = Maze.create_simple_maze()
    maze_str = str(maze)

    # Should contain start, end, walls, and empty spaces
    assert "S" in maze_str
    assert "E" in maze_str
    assert "█" in maze_str
    assert "." in maze_str


def test_out_of_bounds_is_wall() -> None:
    """Test that out of bounds positions are treated as walls."""
    maze = Maze.create_simple_maze()

    # Out of bounds positions should be treated as walls
    assert maze.is_wall(Position(-1, 0)) is True
    assert maze.is_wall(Position(0, -1)) is True
    assert maze.is_wall(Position(100, 100)) is True
