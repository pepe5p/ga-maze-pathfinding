# Genetic Algorithm for Maze Pathfinding - Summary

## Problem Representation
Each individual represents a potential solution as a fixed-length sequence of movement directions (UP, DOWN, LEFT, RIGHT). The agent starts at the maze entrance and follows these directions sequentially to navigate toward the exit.

## Fitness Function (Multi-Objective)
The algorithm minimizes two objectives simultaneously:

**Objective 1: Navigation Quality**
- **If goal reached:**
  - `collisions × 10 + revisit_penalty × 2`
  - Penalizes wall collisions and revisiting cells (inefficient paths)
  
- **If goal not reached:**
  - `collisions × 10 + manhattan_distance × 5 + revisit_penalty × 2 - unique_cells_explored`
  - Heavily penalizes distance to goal
  - Rewards exploration of new cells to guide evolution toward the exit

**Objective 2: Path Length**
- Actual steps taken if goal reached (encourages shorter solutions)
- Maximum allowed length if goal not reached (forces improvement)

## Path Simplification
A key optimization removes redundant opposite moves during evolution:
- UP followed by DOWN cancels out → removed
- "UP, DOWN, DOWN" simplifies to just "DOWN"
- After simplification, paths are padded back to fixed length with random moves
- Applied after both crossover and mutation operations

## Genetic Operators
- **Selection:** Tournament selection (best of k random individuals)
- **Crossover:** Two-point crossover followed by simplification
- **Mutation:** Random direction changes followed by simplification

## Early Stopping
Evolution terminates if the best fitness doesn't improve by at least a threshold amount (default: 0.001) for a specified number of consecutive generations (default: 20), preventing wasted computation on converged populations.

## Key Innovation
The combination of revisit penalties, exploration bonuses, and automatic path simplification encourages the algorithm to discover efficient, non-redundant paths while maintaining diversity through random padding.
