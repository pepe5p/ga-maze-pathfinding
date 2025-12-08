"""Maze representation and utilities."""

from enum import Enum
from typing import NamedTuple


class Cell(Enum):
    """Cell types in the maze."""

    EMPTY = 0
    WALL = 1
    START = 2
    END = 3


class Position(NamedTuple):
    """Position in the maze grid."""

    row: int
    col: int


class Direction(Enum):
    """Movement directions."""

    UP = "U"
    DOWN = "D"
    LEFT = "L"
    RIGHT = "R"


class Maze:
    """Represents a grid maze with walls, start, and end positions."""

    def __init__(self, grid: list[list[int]], start: Position, end: Position):
        """Initialize a maze.

        Args:
            grid: 2D list where 0=empty, 1=wall
            start: Starting position
            end: Goal/target position
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if grid else 0
        self.start = start
        self.end = end

    def is_valid_position(self, pos: Position) -> bool:
        """Check if position is within maze bounds."""
        return 0 <= pos.row < self.rows and 0 <= pos.col < self.cols

    def is_wall(self, pos: Position) -> bool:
        """Check if position is a wall."""
        if not self.is_valid_position(pos):
            return True  # Out of bounds treated as wall
        return self.grid[pos.row][pos.col] == Cell.WALL.value

    def move(self, pos: Position, direction: Direction) -> Position:
        """Calculate new position after moving in given direction."""
        if direction == Direction.UP:
            return Position(pos.row - 1, pos.col)
        elif direction == Direction.DOWN:
            return Position(pos.row + 1, pos.col)
        elif direction == Direction.LEFT:
            return Position(pos.row, pos.col - 1)
        elif direction == Direction.RIGHT:
            return Position(pos.row, pos.col + 1)
        return pos

    def manhattan_distance(self, pos: Position) -> int:
        """Calculate Manhattan distance from position to goal."""
        return abs(pos.row - self.end.row) + abs(pos.col - self.end.col)

    def __str__(self) -> str:
        """String representation of the maze."""
        result = []
        for i, row in enumerate(self.grid):
            row_str = []
            for j, cell in enumerate(row):
                pos = Position(i, j)
                if pos == self.start:
                    row_str.append("S")
                elif pos == self.end:
                    row_str.append("E")
                elif cell == Cell.WALL.value:
                    row_str.append("█")
                else:
                    row_str.append(".")
            result.append(" ".join(row_str))
        return "\n".join(result)
