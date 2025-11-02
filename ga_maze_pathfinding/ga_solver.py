"""Genetic Algorithm solver for maze pathfinding using DEAP."""

import random
from collections.abc import Sequence
from typing import Any

from deap import algorithms, base, creator, tools

from ga_maze_pathfinding.maze import Direction, Maze, Position


class GASolver:
    """Genetic Algorithm solver for maze pathfinding."""

    def __init__(
        self,
        maze: Maze,
        population_size: int = 300,
        max_generations: int = 200,
        max_path_length: int = 100,
        crossover_prob: float = 0.7,
        mutation_prob: float = 0.2,
        tournament_size: int = 3,
    ):
        """Initialize the GA solver.

        Args:
            maze: The maze to solve
            population_size: Number of individuals in population
            max_generations: Maximum number of generations to evolve
            max_path_length: Maximum length of path (chromosome length)
            crossover_prob: Probability of crossover
            mutation_prob: Probability of mutation
            tournament_size: Tournament size for selection
        """
        self.maze = maze
        self.population_size = population_size
        self.max_generations = max_generations
        self.max_path_length = max_path_length
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.tournament_size = tournament_size

        # Setup DEAP
        self._setup_deap()

    def _setup_deap(self) -> None:
        """Setup DEAP creator and toolbox."""
        # Clear any existing fitness and individual classes
        if hasattr(creator, "FitnessMin"):
            del creator.FitnessMin
        if hasattr(creator, "Individual"):
            del creator.Individual

        # Create fitness (minimize both collisions and distance)
        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()

        # Attribute generator - random direction
        directions = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
        self.toolbox.register("attr_direction", random.choice, directions)

        # Structure initializers
        self.toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            self.toolbox.attr_direction,
            n=self.max_path_length,
        )
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Genetic operators
        self.toolbox.register("evaluate", self._evaluate_individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self._mutate_individual, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)

    def _evaluate_individual(self, individual: Sequence[Direction]) -> tuple[float, float]:
        """Evaluate an individual (path) in the maze.

        Returns:
            Tuple of (collisions + out_of_bounds, effective_distance)
            We want to minimize both values.
        """
        position = self.maze.start
        collisions = 0
        visited_positions = [position]

        # Simulate the path
        for direction in individual:
            new_position = self.maze.move(position, direction)

            # Check if hit wall or out of bounds
            if self.maze.is_wall(new_position) or not self.maze.is_valid_position(new_position):
                collisions += 1
                # Don't move if hit wall
            else:
                position = new_position
                visited_positions.append(position)

            # Early stop if reached goal
            if position == self.maze.end:
                break

        # Calculate distance to goal from final position
        distance_to_goal = self.maze.manhattan_distance(position)

        # Penalty for collisions and distance
        # Also consider path length - shorter paths are better if they reach goal
        if position == self.maze.end:
            # Reached goal: minimize collisions and actual path length
            path_length = len(visited_positions)
            fitness = (collisions * 10, path_length)
        else:
            # Did not reach goal: heavily penalize distance to goal and collisions
            fitness = (collisions * 10 + distance_to_goal * 5, self.max_path_length)

        return fitness

    def _mutate_individual(self, individual: Any, indpb: float) -> tuple[Any]:
        """Mutate an individual by randomly changing some directions.

        Args:
            individual: The individual to mutate
            indpb: Independent probability for each attribute to be mutated

        Returns:
            Tuple containing the mutated individual
        """
        directions = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
        for i in range(len(individual)):
            if random.random() < indpb:  # noqa: S311
                individual[i] = random.choice(directions)  # noqa: S311
        return (individual,)

    def solve(self, verbose: bool = True) -> tuple[list[Direction], dict]:
        """Solve the maze using genetic algorithm.

        Args:
            verbose: Whether to print progress

        Returns:
            Tuple of (best_path, statistics_dict)
        """
        # Create initial population
        population = self.toolbox.population(n=self.population_size)

        # Statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", lambda x: sum(f[0] + f[1] for f in x) / len(x))
        stats.register("min", lambda x: min(f[0] + f[1] for f in x))
        stats.register("max", lambda x: max(f[0] + f[1] for f in x))

        # Hall of fame to keep track of best individuals
        hof = tools.HallOfFame(1)

        if verbose:
            print(f"Starting GA with population={self.population_size}, generations={self.max_generations}")  # noqa: T201
            print(f"Maze size: {self.maze.rows}x{self.maze.cols}")  # noqa: T201
            print(f"Start: {self.maze.start}, End: {self.maze.end}\n")  # noqa: T201

        # Run the algorithm
        population, logbook = algorithms.eaSimple(
            population,
            self.toolbox,
            cxpb=self.crossover_prob,
            mutpb=self.mutation_prob,
            ngen=self.max_generations,
            stats=stats,
            halloffame=hof,
            verbose=verbose,
        )

        # Get the best individual
        best_individual = hof[0]
        best_fitness = best_individual.fitness.values

        if verbose:
            print(f"\nBest fitness: {best_fitness}")  # noqa: T201
            print(f"Collisions/Distance penalty: {best_fitness[0]}")  # noqa: T201
            print(f"Path length: {best_fitness[1]}")  # noqa: T201

        # Extract statistics
        stats_dict = {
            "best_fitness": best_fitness,
            "best_individual": best_individual,
            "logbook": logbook,
            "final_population": population,
        }

        return list(best_individual), stats_dict

    def visualize_path(self, path: list[Direction]) -> str:
        """Visualize the path on the maze.

        Args:
            path: List of directions

        Returns:
            String representation of maze with path
        """
        # Create a copy of the grid for visualization
        vis_grid = [row[:] for row in self.maze.grid]

        position = self.maze.start
        path_positions = [position]

        # Trace the path
        for direction in path:
            new_position = self.maze.move(position, direction)

            # Only move if valid
            if not self.maze.is_wall(new_position) and self.maze.is_valid_position(new_position):
                position = new_position
                path_positions.append(position)

            # Stop if reached goal
            if position == self.maze.end:
                break

        # Mark path positions
        result = []
        for i, row in enumerate(vis_grid):
            row_str = []
            for j, cell in enumerate(row):
                pos = Position(i, j)
                if pos == self.maze.start:
                    row_str.append("S")
                elif pos == self.maze.end:
                    row_str.append("E")
                elif pos in path_positions:
                    row_str.append("*")
                elif cell == 1:  # Wall
                    row_str.append("█")
                else:
                    row_str.append(".")
            result.append(" ".join(row_str))

        # Add legend
        result.append("\nLegend: S=Start, E=End, *=Path, █=Wall, .=Empty")
        result.append(f"Path length: {len(path_positions)} steps")
        result.append(f"Reached goal: {position == self.maze.end}")

        return "\n".join(result)
