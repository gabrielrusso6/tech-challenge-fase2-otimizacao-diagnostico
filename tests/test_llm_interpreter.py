from src.llm_interpreter import DiagnosisLLMInterpreter


def test_prompt_contains_medical_safety_instruction():
    interpreter = DiagnosisLLMInterpreter()
    prompt = interpreter.build_prompt(
        prediction_label="Benigno",
        malignant_probability=0.15,
        model_metrics={"accuracy": 0.9, "precision": 0.9, "recall": 0.9, "f1_score": 0.9, "auc_roc": 0.95},
        top_features=["radius_worst", "area_worst"],
    )
    assert "Não afirme diagnóstico definitivo" in prompt
    assert "Benigno" in prompt
    assert "radius_worst" in prompt
