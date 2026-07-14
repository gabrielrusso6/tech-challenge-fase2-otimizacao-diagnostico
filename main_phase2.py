#!/usr/bin/env python3
"""Pipeline principal - Tech Challenge Fase 2.

Projeto 1: Otimização de Modelos de Diagnóstico usando Algoritmos Genéticos
+ interpretação de resultados com LLM.
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.baseline_model import get_baseline_models
from src.evaluate import ModelEvaluator
from src.genetic_optimizer import run_three_experiments
from src.llm_interpreter import DiagnosisLLMInterpreter
from src.logger_config import setup_logger
from src.preprocessing import BreastCancerPreprocessor

warnings.filterwarnings("ignore")

DATA_PATH = "data/breast-cancer-wisconsin.csv"
REPORTS_DIR = Path("reports")
MODELS_DIR = Path("models")


def get_top_features_from_model(model, feature_columns, top_n=10):
    if hasattr(model, "coef_"):
        values = np.abs(model.coef_[0])
    elif hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    else:
        return []
    feature_importance = pd.DataFrame(
        {"feature": feature_columns, "importance": values}
    ).sort_values("importance", ascending=False)
    feature_importance.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)
    return feature_importance.head(top_n)["feature"].tolist()


def main():
    logger = setup_logger()
    REPORTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    logger.info("Iniciando Tech Challenge Fase 2")
    logger.info("Projeto escolhido: Otimização de Modelos de Diagnóstico")

    print("=" * 72)
    print("Tech Challenge - Fase 2")
    print("Algoritmos Genéticos + LLM para Diagnóstico de Câncer de Mama")
    print("=" * 72)

    # 1. Pré-processamento
    print("\n1. Pré-processamento")
    preprocessor = BreastCancerPreprocessor()
    split = preprocessor.preprocess_pipeline(DATA_PATH, test_size=0.2, random_state=42)
    logger.info("Dados preparados: treino=%s teste=%s", split.X_train.shape, split.X_test.shape)

    # 2. Baselines da Fase 1
    print("\n2. Treinamento dos modelos baseline")
    evaluator = ModelEvaluator()
    baseline_models = get_baseline_models(random_state=42)

    for name, model in baseline_models.items():
        model.fit(split.X_train, split.y_train)
        metrics = evaluator.evaluate_model(model, split.X_test, split.y_test, name)
        print(
            f"{name}: F1={metrics['f1_score']:.4f} | "
            f"Recall={metrics['recall']:.4f} | AUC={metrics['auc_roc']:.4f}"
        )

    # 3. Algoritmo Genético: 3 experimentos obrigatórios
    print("\n3. Otimização com Algoritmo Genético")
    print("Executando 3 experimentos com populações e taxas de mutação diferentes...")
    ga_summary, best_ga_result = run_three_experiments(split.X_train, split.y_train)
    ga_summary.to_csv(REPORTS_DIR / "ga_experiments_summary.csv", index=False)
    print(ga_summary[["experiment", "population_size", "generations", "mutation_rate", "best_fitness", "f1_score", "recall", "auc_roc"]].to_string(index=False))

    # 4. Treinamento final com melhores hiperparâmetros do GA
    print("\n4. Treinamento do modelo otimizado")
    optimized_model = LogisticRegression(
        **best_ga_result.best_individual,
        solver="liblinear",
        max_iter=10000,
        random_state=42,
        tol=1e-4,
    )
    optimized_model.fit(split.X_train, split.y_train)
    optimized_metrics = evaluator.evaluate_model(
        optimized_model, split.X_test, split.y_test, "LogisticRegression_GA_optimized"
    )
    joblib.dump(optimized_model, MODELS_DIR / "logistic_regression_ga_optimized.pkl")

    print("Melhores hiperparâmetros encontrados pelo GA:")
    print(json.dumps(best_ga_result.best_individual, ensure_ascii=False, indent=2))

    # 5. Comparativo final
    print("\n5. Comparativo final")
    comparison_df = evaluator.save_comparison(REPORTS_DIR / "model_comparison.csv")
    print(comparison_df.to_string(index=False))

    # 6. Interpretação com LLM/fallback
    print("\n6. Interpretação de resultado com LLM")
    sample_idx = 0
    sample = split.X_test.iloc[[sample_idx]]
    prediction = optimized_model.predict(sample)[0]
    probability = optimized_model.predict_proba(sample)[0][1]
    prediction_label = "Maligno" if prediction == 1 else "Benigno"
    top_features = get_top_features_from_model(
        optimized_model, split.feature_columns, top_n=10
    )

    interpreter = DiagnosisLLMInterpreter()
    explanation = interpreter.explain_result(
        prediction_label=prediction_label,
        malignant_probability=float(probability),
        model_metrics=optimized_metrics,
        top_features=top_features,
        use_llm=True,
    )
    (REPORTS_DIR / "sample_llm_explanation.txt").write_text(explanation, encoding="utf-8")
    print(explanation)

    # 7. Registro consolidado em JSON para rastreabilidade
    consolidated = {
        "project": "Tech Challenge Fase 2 - Projeto 1",
        "best_ga_config": best_ga_result.config,
        "best_ga_individual": best_ga_result.best_individual,
        "best_ga_cv_metrics": best_ga_result.best_metrics,
        "optimized_test_metrics": {
            k: v for k, v in optimized_metrics.items() if k != "confusion_matrix"
        },
        "optimized_confusion_matrix": optimized_metrics["confusion_matrix"],
        "top_features": top_features,
    }
    (REPORTS_DIR / "consolidated_results.json").write_text(
        json.dumps(consolidated, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    logger.info("Pipeline concluído com sucesso")
    print("\n✅ Pipeline da Fase 2 concluído com sucesso.")
    print("Arquivos gerados na pasta reports/ e modelo salvo em models/.")


if __name__ == "__main__":
    main()
