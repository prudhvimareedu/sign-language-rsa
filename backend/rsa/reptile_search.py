import numpy as np


class ReptileSearchFeatureSelector:

    def __init__(
        self,
        population_size=10,
        iterations=15,
        random_state=42
    ):
        self.population_size = population_size
        self.iterations = iterations
        self.random_state = random_state

        self.rng = np.random.default_rng(
            random_state
        )

        self.best_position = None
        self.best_fitness = float("inf")

        self.fitness_history = []

    def _sigmoid(self, values):
        values = np.clip(
            values,
            -10,
            10
        )

        return 1.0 / (
            1.0 + np.exp(-values)
        )

    def _to_binary(self, position):

        probabilities = self._sigmoid(
            position
        )

        return (
            probabilities >= 0.5
        ).astype(int)

    def optimize(
        self,
        fitness_function,
        number_of_features
    ):

        population = self.rng.uniform(
            -1,
            1,
            size=(
                self.population_size,
                number_of_features
            )
        )

        fitness_values = np.zeros(
            self.population_size
        )

        # Evaluate initial population
        for i in range(
            self.population_size
        ):

            mask = self._to_binary(
                population[i]
            )

            fitness_values[i] = (
                fitness_function(mask)
            )

        # Find initial best solution
        best_index = np.argmin(
            fitness_values
        )

        self.best_fitness = (
            fitness_values[best_index]
        )

        self.best_position = (
            population[best_index].copy()
        )

        # RSA optimization
        for iteration in range(
            self.iterations
        ):

            progress = (
                iteration
                / max(
                    1,
                    self.iterations - 1
                )
            )

            for i in range(
                self.population_size
            ):

                current = population[i]

                random_index = self.rng.integers(
                    0,
                    self.population_size
                )

                reference = population[
                    random_index
                ]

                random_vector = self.rng.uniform(
                    -1,
                    1,
                    number_of_features
                )

                # High walking
                if progress < 0.25:

                    new_position = (
                        self.best_position
                        - random_vector
                        * (
                            reference
                            - current
                        )
                    )

                # Belly walking
                elif progress < 0.50:

                    new_position = (
                        self.best_position
                        + random_vector
                        * (
                            current
                            - reference
                        )
                    )

                # Hunting coordination
                elif progress < 0.75:

                    new_position = (
                        self.best_position
                        + random_vector
                        * (
                            self.best_position
                            - current
                        )
                    )

                # Hunting cooperation
                else:

                    new_position = (
                        self.best_position
                        + random_vector
                        * (
                            reference
                            - current
                        )
                    )

                # Move gradually toward best solution
                new_position = (
                    (1 - progress)
                    * new_position
                    + progress
                    * self.best_position
                )

                new_position = np.clip(
                    new_position,
                    -5,
                    5
                )

                # Convert to binary feature mask
                new_mask = self._to_binary(
                    new_position
                )

                new_fitness = (
                    fitness_function(
                        new_mask
                    )
                )

                # Accept better solution
                if (
                    new_fitness
                    < fitness_values[i]
                ):

                    population[i] = (
                        new_position
                    )

                    fitness_values[i] = (
                        new_fitness
                    )

                # Update global best
                if (
                    fitness_values[i]
                    < self.best_fitness
                ):

                    self.best_fitness = (
                        fitness_values[i]
                    )

                    self.best_position = (
                        population[i].copy()
                    )

            self.fitness_history.append(
                self.best_fitness
            )

            current_mask = self._to_binary(
                self.best_position
            )

            print(
                f"RSA iteration "
                f"{iteration + 1}/"
                f"{self.iterations} | "
                f"Best fitness: "
                f"{self.best_fitness:.5f} | "
                f"Selected features: "
                f"{np.sum(current_mask)}"
            )

        # Final feature mask
        final_mask = self._to_binary(
            self.best_position
        )

        # Make sure at least 3 features are selected
        if np.sum(final_mask) < 3:

            feature_indices = np.argsort(
                np.abs(
                    self.best_position
                )
            )[-3:]

            final_mask = np.zeros(
                number_of_features,
                dtype=int
            )

            final_mask[
                feature_indices
            ] = 1

        return final_mask