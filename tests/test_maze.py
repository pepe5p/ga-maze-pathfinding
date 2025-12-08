from ga_maze_pathfinding.maze import Direction, Maze, Position


def test_position_validation(simple_maze: Maze) -> None:
    assert simple_maze.is_valid_position(Position(0, 0)) is True
    assert simple_maze.is_valid_position(Position(4, 4)) is True
    assert simple_maze.is_valid_position(Position(-1, 0)) is False
    assert simple_maze.is_valid_position(Position(0, -1)) is False
    assert simple_maze.is_valid_position(Position(5, 0)) is False
    assert simple_maze.is_valid_position(Position(0, 5)) is False


def test_wall_detection(simple_maze: Maze) -> None:
    assert simple_maze.is_wall(Position(1, 1)) is True
    assert simple_maze.is_wall(Position(0, 0)) is False


def test_movement(simple_maze: Maze) -> None:
    start = Position(2, 2)

    up = simple_maze.move(start, Direction.UP)
    assert up == Position(1, 2)

    down = simple_maze.move(start, Direction.DOWN)
    assert down == Position(3, 2)

    left = simple_maze.move(start, Direction.LEFT)
    assert left == Position(2, 1)

    right = simple_maze.move(start, Direction.RIGHT)
    assert right == Position(2, 3)


def test_manhattan_distance(simple_maze: Maze) -> None:
    dist = simple_maze.manhattan_distance(simple_maze.start)
    assert dist == 8

    dist_end = simple_maze.manhattan_distance(simple_maze.end)
    assert dist_end == 0


def test_complex_maze_creation(complex_maze: Maze) -> None:
    assert complex_maze.rows == 10
    assert complex_maze.cols == 10
    assert complex_maze.start == Position(0, 0)
    assert complex_maze.end == Position(9, 9)


def test_maze_string_representation(simple_maze: Maze) -> None:
    maze_str = str(simple_maze)

    assert "S" in maze_str
    assert "E" in maze_str
    assert "█" in maze_str
    assert "." in maze_str


def test_out_of_bounds_is_wall(simple_maze: Maze) -> None:
    assert simple_maze.is_wall(Position(-1, 0)) is True
    assert simple_maze.is_wall(Position(0, -1)) is True
    assert simple_maze.is_wall(Position(100, 100)) is True
