"""Genetic Algorithm solver for maze pathfinding using DEAP."""

import random
from collections.abc import Sequence
from typing import Any

from deap import base, creator, tools

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
        early_stopping_generations: int = 20,
        early_stopping_min_change: float = 0.001,
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
            early_stopping_generations: Number of generations without improvement to trigger early stopping
            early_stopping_min_change: Minimum fitness change to consider as improvement
        """
        self.maze = maze
        self.population_size = population_size
        self.max_generations = max_generations
        self.max_path_length = max_path_length
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.tournament_size = tournament_size
        self.early_stopping_generations = early_stopping_generations
        self.early_stopping_min_change = early_stopping_min_change

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
        self.toolbox.register("mate", self._crossover_individuals)
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
        unique_positions = {position}  # Track unique positions explored

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
                unique_positions.add(position)

            # Early stop if reached goal
            if position == self.maze.end:
                break

        # Calculate distance to goal from final position
        distance_to_goal = self.maze.manhattan_distance(position)

        # Calculate exploration score (number of unique cells visited)
        unique_cells_visited = len(unique_positions)

        # Calculate revisit penalty (encourages exploring new cells)
        revisit_penalty = len(visited_positions) - unique_cells_visited

        # Penalty for collisions and distance
        # Also consider path length - shorter paths are better if they reach goal
        if position == self.maze.end:
            # Reached goal: minimize collisions and actual path length
            path_length = len(visited_positions)
            # Reward: fewer collisions, shorter path, less revisiting
            fitness = (collisions * 10 + revisit_penalty * 2, path_length)
        else:
            # Did not reach goal: heavily penalize distance to goal and collisions
            # But reward exploring more unique cells
            exploration_bonus = -unique_cells_visited  # Negative because we're minimizing
            fitness = (
                collisions * 10 + distance_to_goal * 5 + revisit_penalty * 2 + exploration_bonus,
                self.max_path_length,
            )

        return fitness

    def _crossover_individuals(self, ind1: Any, ind2: Any) -> tuple[Any, Any]:
        """Crossover two individuals and simplify the offspring.

        Args:
            ind1: First parent
            ind2: Second parent

        Returns:
            Tuple of two offspring
        """
        # Perform two-point crossover
        tools.cxTwoPoint(ind1, ind2)

        # Simplify both offspring
        simplified1 = self._simplify_individual(ind1)
        simplified2 = self._simplify_individual(ind2)

        # Pad back to max_path_length
        ind1[:] = self._pad_individual(simplified1, self.max_path_length)
        ind2[:] = self._pad_individual(simplified2, self.max_path_length)

        return ind1, ind2

    def _simplify_individual(self, individual: Any) -> list[Direction]:
        """Simplify an individual by removing opposite move cancellations.

        For example: UP, DOWN, DOWN -> DOWN
                     LEFT, RIGHT -> (empty)
                     UP, UP, DOWN -> UP

        Args:
            individual: The individual to simplify

        Returns:
            Simplified list of directions
        """
        opposites = {
            Direction.UP: Direction.DOWN,
            Direction.DOWN: Direction.UP,
            Direction.LEFT: Direction.RIGHT,
            Direction.RIGHT: Direction.LEFT,
        }

        simplified: list[Direction] = []
        for direction in individual:
            # If the last move is opposite to current, they cancel out
            if simplified and simplified[-1] == opposites[direction]:
                simplified.pop()
            else:
                simplified.append(direction)

        return simplified

    def _pad_individual(self, individual: list[Direction], target_length: int) -> list[Direction]:
        """Pad an individual to target length with random directions.

        Args:
            individual: The individual to pad
            target_length: Desired length

        Returns:
            Padded individual
        """
        directions = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
        while len(individual) < target_length:
            individual.append(random.choice(directions))  # noqa: S311
        return individual

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

        # Simplify after mutation to remove redundant moves
        simplified = self._simplify_individual(individual)
        individual[:] = self._pad_individual(simplified, self.max_path_length)

        return (individual,)

    def _ea_simple_with_early_stopping(
        self,
        population: list[Any],
        toolbox: base.Toolbox,
        cxpb: float,
        mutpb: float,
        ngen: int,
        stats: tools.Statistics,
        halloffame: tools.HallOfFame,
        verbose: bool = True,
    ) -> tuple[list[Any], tools.Logbook]:
        """Custom evolutionary algorithm with early stopping.

        Args:
            population: Initial population
            toolbox: DEAP toolbox with genetic operators
            cxpb: Crossover probability
            mutpb: Mutation probability
            ngen: Maximum number of generations
            stats: Statistics object
            halloffame: Hall of fame to track best individuals
            verbose: Whether to print progress

        Returns:
            Tuple of (final_population, logbook)
        """
        logbook = tools.Logbook()
        logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

        # Evaluate the initial population
        fitnesses = map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses, strict=False):
            ind.fitness.values = fit

        if halloffame is not None:
            halloffame.update(population)

        record = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=len(population), **record)
        if verbose:
            print(logbook.stream)  # noqa: T201

        # Early stopping tracking
        best_fitness_history: list[float] = []
        generations_without_improvement = 0

        # Evolution loop
        for gen in range(1, ngen + 1):
            # Select the next generation individuals
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))

            # Apply crossover and mutation
            for i in range(1, len(offspring), 2):
                if random.random() < cxpb:  # noqa: S311
                    offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1], offspring[i])
                    del offspring[i - 1].fitness.values
                    del offspring[i].fitness.values

            for i in range(len(offspring)):
                if random.random() < mutpb:  # noqa: S311
                    (offspring[i],) = toolbox.mutate(offspring[i])
                    del offspring[i].fitness.values

            # Evaluate individuals with invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses, strict=False):
                ind.fitness.values = fit

            # Update hall of fame
            if halloffame is not None:
                halloffame.update(offspring)

            # Replace population
            population[:] = offspring

            # Record statistics
            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            if verbose:
                print(logbook.stream)  # noqa: T201

            # Early stopping check
            current_best_fitness = record.get("min", float("inf"))
            best_fitness_history.append(current_best_fitness)

            if len(best_fitness_history) > 1:
                improvement = best_fitness_history[-2] - current_best_fitness
                if improvement < self.early_stopping_min_change:
                    generations_without_improvement += 1
                else:
                    generations_without_improvement = 0

                if generations_without_improvement >= self.early_stopping_generations:
                    if verbose:
                        print(f"\nEarly stopping triggered at generation {gen}")  # noqa: T201
                        print(f"No improvement for {self.early_stopping_generations} generations")  # noqa: T201
                    break

        return population, logbook

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
            print(  # noqa: T201
                f"Early stopping: {self.early_stopping_generations} gens, "
                f"min change: {self.early_stopping_min_change}\n"
            )

        # Run the algorithm with early stopping
        population, logbook = self._ea_simple_with_early_stopping(
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
