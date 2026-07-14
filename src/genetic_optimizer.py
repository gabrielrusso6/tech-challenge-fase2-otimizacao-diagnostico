import random
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate


@dataclass
class GAConfig:
    population_size: int = 8
    generations: int = 4
    crossover_rate: float = 0.8
    mutation_rate: float = 0.2
    tournament_size: int = 3
    cv_folds: int = 5
    random_state: int = 42


@dataclass
class GAResult:
    config: Dict[str, Any]
    best_individual: Dict[str, Any]
    best_fitness: float
    best_metrics: Dict[str, float]
    history: pd.DataFrame


class GeneticHyperparameterOptimizer:
    """Algoritmo genético para otimizar hiperparâmetros de Logistic Regression.

    A Logistic Regression foi escolhida porque, na Fase 1, apresentou o melhor
    equilíbrio entre accuracy, recall, F1-score e AUC-ROC. O algoritmo genético
    substitui a busca manual/grid search por uma busca evolutiva.
    """

    def __init__(self, config: GAConfig):
        self.config = config
        self.random = random.Random(config.random_state)
        self.history_rows: List[Dict[str, Any]] = []
        self.search_space = {
            "C": [0.001, 0.003, 0.01, 0.03, 0.05, 0.1, 0.3, 1.0, 3.0, 10.0],
            "penalty": ["l1", "l2"],
            "class_weight": [None, "balanced"],
            "fit_intercept": [True, False],
            "intercept_scaling": [0.5, 1.0, 2.0],
        }

    def create_individual(self) -> Dict[str, Any]:
        return {gene: self.random.choice(values) for gene, values in self.search_space.items()}

    def create_population(self) -> List[Dict[str, Any]]:
        # População inicial com uma configuração baseline e candidatos clínicos úteis,
        # seguida por indivíduos aleatórios para preservar diversidade.
        seed_individuals = [
            {"C": 1.0, "penalty": "l2", "class_weight": None, "fit_intercept": True, "intercept_scaling": 1.0},
            {"C": 0.03, "penalty": "l2", "class_weight": "balanced", "fit_intercept": True, "intercept_scaling": 1.0},
            {"C": 0.03, "penalty": "l1", "class_weight": None, "fit_intercept": True, "intercept_scaling": 1.0},
        ]
        population = seed_individuals[: self.config.population_size]
        while len(population) < self.config.population_size:
            population.append(self.create_individual())
        return population

    def build_model(self, individual: Dict[str, Any]) -> LogisticRegression:
        return LogisticRegression(
            **individual,
            solver="liblinear",
            max_iter=10000,
            random_state=self.config.random_state,
            tol=1e-4,
        )

    def fitness(self, individual: Dict[str, Any], X_train, y_train) -> Tuple[float, Dict[str, float]]:
        model = self.build_model(individual)
        cv = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )
        scoring = {
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc",
        }
        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )
        metrics = {
            "accuracy": float(np.mean(scores["test_accuracy"])),
            "precision": float(np.mean(scores["test_precision"])),
            "recall": float(np.mean(scores["test_recall"])),
            "f1_score": float(np.mean(scores["test_f1"])),
            "auc_roc": float(np.mean(scores["test_roc_auc"])),
        }

        # Contexto médico: prioriza F1 e recall para reduzir risco de falsos negativos.
        fitness_value = (
            0.45 * metrics["f1_score"]
            + 0.35 * metrics["recall"]
            + 0.15 * metrics["auc_roc"]
            + 0.05 * metrics["accuracy"]
        )
        return float(fitness_value), metrics

    def tournament_selection(self, evaluated_population: List[Tuple[Dict[str, Any], float, Dict[str, float]]]) -> Dict[str, Any]:
        contenders = self.random.sample(
            evaluated_population,
            k=min(self.config.tournament_size, len(evaluated_population)),
        )
        winner = max(contenders, key=lambda item: item[1])
        return winner[0].copy()

    def crossover(self, parent_a: Dict[str, Any], parent_b: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if self.random.random() > self.config.crossover_rate:
            return parent_a.copy(), parent_b.copy()

        child_a, child_b = {}, {}
        for gene in self.search_space:
            if self.random.random() < 0.5:
                child_a[gene] = parent_a[gene]
                child_b[gene] = parent_b[gene]
            else:
                child_a[gene] = parent_b[gene]
                child_b[gene] = parent_a[gene]
        return child_a, child_b

    def mutate(self, individual: Dict[str, Any]) -> Dict[str, Any]:
        mutated = individual.copy()
        for gene, values in self.search_space.items():
            if self.random.random() < self.config.mutation_rate:
                mutated[gene] = self.random.choice(values)
        return mutated

    def optimize(self, X_train, y_train) -> GAResult:
        population = self.create_population()
        best_individual = None
        best_fitness = -1.0
        best_metrics = {}

        for generation in range(1, self.config.generations + 1):
            evaluated = []
            for individual in population:
                fit, metrics = self.fitness(individual, X_train, y_train)
                evaluated.append((individual, fit, metrics))

                if fit > best_fitness:
                    best_individual = individual.copy()
                    best_fitness = fit
                    best_metrics = metrics.copy()

            generation_best = max(evaluated, key=lambda item: item[1])
            generation_mean = float(np.mean([item[1] for item in evaluated]))
            self.history_rows.append(
                {
                    "generation": generation,
                    "best_fitness_generation": generation_best[1],
                    "mean_fitness_generation": generation_mean,
                    "best_fitness_global": best_fitness,
                    **{f"best_{k}": v for k, v in best_metrics.items()},
                    **{f"param_{k}": v for k, v in best_individual.items()},
                }
            )

            next_population = [best_individual.copy()]  # elitismo
            while len(next_population) < self.config.population_size:
                parent_a = self.tournament_selection(evaluated)
                parent_b = self.tournament_selection(evaluated)
                child_a, child_b = self.crossover(parent_a, parent_b)
                next_population.append(self.mutate(child_a))
                if len(next_population) < self.config.population_size:
                    next_population.append(self.mutate(child_b))
            population = next_population

        return GAResult(
            config=asdict(self.config),
            best_individual=best_individual,
            best_fitness=best_fitness,
            best_metrics=best_metrics,
            history=pd.DataFrame(self.history_rows),
        )


def run_three_experiments(X_train, y_train) -> Tuple[pd.DataFrame, GAResult]:
    configs = [
        GAConfig(population_size=6, generations=4, mutation_rate=0.10, crossover_rate=0.80, random_state=42),
        GAConfig(population_size=8, generations=5, mutation_rate=0.20, crossover_rate=0.85, random_state=43),
        GAConfig(population_size=10, generations=5, mutation_rate=0.30, crossover_rate=0.90, random_state=44),
    ]

    summary_rows = []
    best_result = None
    for idx, config in enumerate(configs, start=1):
        optimizer = GeneticHyperparameterOptimizer(config)
        result = optimizer.optimize(X_train, y_train)
        summary_rows.append(
            {
                "experiment": idx,
                **result.config,
                "best_fitness": result.best_fitness,
                **result.best_metrics,
                **{f"param_{k}": v for k, v in result.best_individual.items()},
            }
        )
        result.history.to_csv(f"reports/ga_history_experiment_{idx}.csv", index=False)
        if best_result is None or result.best_fitness > best_result.best_fitness:
            best_result = result

    return pd.DataFrame(summary_rows), best_result
