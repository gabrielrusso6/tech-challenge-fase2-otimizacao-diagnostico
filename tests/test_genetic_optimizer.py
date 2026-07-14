from src.genetic_optimizer import GAConfig, GeneticHyperparameterOptimizer


def test_genetic_individual_has_expected_genes():
    optimizer = GeneticHyperparameterOptimizer(GAConfig(population_size=2, generations=1))
    individual = optimizer.create_individual()
    expected = {
        "C",
        "penalty",
        "class_weight",
        "fit_intercept",
        "intercept_scaling",
    }
    assert set(individual.keys()) == expected


def test_crossover_preserves_genes():
    optimizer = GeneticHyperparameterOptimizer(GAConfig(population_size=2, generations=1, crossover_rate=1.0))
    parent_a = optimizer.create_individual()
    parent_b = optimizer.create_individual()
    child_a, child_b = optimizer.crossover(parent_a, parent_b)
    assert set(child_a.keys()) == set(parent_a.keys())
    assert set(child_b.keys()) == set(parent_b.keys())
