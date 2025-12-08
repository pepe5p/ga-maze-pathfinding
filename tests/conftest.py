import pytest

from ga_maze_pathfinding.maze import Maze, Position


@pytest.fixture()
def simple_maze() -> Maze:
    return Maze(
        grid=[
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0],
        ],
        start=Position(0, 0),
        end=Position(4, 4),
    )


@pytest.fixture()
def complex_maze() -> Maze:
    return Maze(
        grid=[
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        ],
        start=Position(0, 0),
        end=Position(9, 9),
    )
