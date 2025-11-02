# Ga Maze Pathfinding

Attempt to solve Grid Maze Pathfinding Problem using Genetic Algorithm.

## Overview

This project implements a genetic algorithm (GA) using the DEAP library to solve grid maze pathfinding problems. Each individual in the population represents a potential solution as a sequence of moves: `{U, D, L, R}` (Up, Down, Left, Right).

The GA optimizes two objectives simultaneously:
1. **Minimize collisions** with walls
2. **Minimize path length** to reach the goal

## Features

- **DEAP-based implementation**: Uses the powerful DEAP (Distributed Evolutionary Algorithms in Python) library
- **Multi-objective optimization**: Balances collision avoidance and path length
- **Customizable mazes**: Easy to create and test different maze configurations
- **Path visualization**: Visual representation of the solution path
- **Configurable GA parameters**: Adjust population size, mutation rate, crossover rate, etc.

## Installation

Ensure you have Python 3.13+ installed. Install dependencies:

```bash
uv sync
```

## Usage

### Running the Solver

Run the main script to solve both simple and complex mazes:

```bash
python -m ga_maze_pathfinding.main
```

### Using in Your Own Code

```python
from ga_maze_pathfinding.maze import Maze, Position
from ga_maze_pathfinding.ga_solver import GASolver

# Create a custom maze
grid = [
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0],
]
maze = Maze(grid, start=Position(0, 0), end=Position(4, 4))

# Create and configure the solver
solver = GASolver(
    maze=maze,
    population_size=300,
    max_generations=100,
    max_path_length=50,
    crossover_prob=0.7,
    mutation_prob=0.2,
)

# Solve the maze
best_path, stats = solver.solve(verbose=True)

# Visualize the solution
print(solver.visualize_path(best_path))
```

## Algorithm Details

### Representation

- **Individual**: A list of `Direction` enums (UP, DOWN, LEFT, RIGHT) of fixed length
- **Genome**: Each gene is a direction that the agent should take

### Fitness Function

The fitness function evaluates individuals based on two criteria:

1. **Collision Penalty**: 
   - Heavy penalty for hitting walls or going out of bounds
   - `collisions * 10`

2. **Distance/Path Length**:
   - If goal reached: minimize actual path length
   - If goal not reached: heavily penalize Manhattan distance to goal (`distance * 5`)

The fitness is returned as a tuple `(collision_penalty, path_length)` to enable multi-objective optimization.

### Genetic Operators

- **Selection**: Tournament selection (default size: 3)
- **Crossover**: Two-point crossover
- **Mutation**: Random gene replacement with probability `indpb`

### Parameters

Key parameters you can adjust:

- `population_size`: Number of individuals (default: 300)
- `max_generations`: Number of generations to evolve (default: 200)
- `max_path_length`: Length of chromosome/path (default: 100)
- `crossover_prob`: Probability of crossover (default: 0.7)
- `mutation_prob`: Probability of mutation (default: 0.2)
- `tournament_size`: Tournament size for selection (default: 3)

## Project Structure

```
ga_maze_pathfinding/
├── __init__.py
├── maze.py          # Maze representation and utilities
├── ga_solver.py     # DEAP-based genetic algorithm solver
└── main.py          # Example usage and demo
```

## Running Tests

```bash
pytest tests/ -v
```

## Example Output

```
Starting GA with population=300, generations=100
Maze size: 5x5
Start: Position(row=0, col=0), End: Position(row=4, col=4)

gen	nevals	avg    	min    	max    
0  	300   	125.5  	45.0   	250.0  
1  	210   	98.3   	38.0   	220.0  
...

Best fitness: (0.0, 8)
Collisions/Distance penalty: 0.0
Path length: 8

SOLUTION PATH:
S . . . .
. █ █ █ *
. * * █ *
. █ * * *
. . . █ E

Legend: S=Start, E=End, *=Path, █=Wall, .=Empty
Path length: 8 steps
Reached goal: True
```

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

