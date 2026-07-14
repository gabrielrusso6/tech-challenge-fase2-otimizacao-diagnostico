import json
import os
from typing import Dict, List, Optional

import requests


class DiagnosisLLMInterpreter:
    """Integração com LLM para interpretação de diagnósticos.

    A integração suporta Ollama local via HTTP. Caso o Ollama não esteja ativo
    ou gere uma resposta incoerente com as regras de segurança, o sistema usa
    uma explicação controlada para manter a demonstração responsável.
    """

    def __init__(self, model_name: Optional[str] = None, base_url: Optional[str] = None):
        self.model_name = model_name or os.getenv("OLLAMA_MODEL", "llama3.2")
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    def build_prompt(
        self,
        prediction_label: str,
        malignant_probability: float,
        model_metrics: Dict[str, float],
        top_features: List[str],
    ) -> str:
        malignant_probability_percent = malignant_probability * 100

        return f"""
Você é um assistente de apoio à decisão clínica em um projeto acadêmico.
Explique o resultado de um modelo de Machine Learning para câncer de mama em português claro.

REGRAS OBRIGATÓRIAS:
- Use somente os dados fornecidos neste prompt.
- Não invente quantidade de amostras, fonte de dados, PVP, sensibilidade, especificidade ou qualquer métrica não informada.
- Não afirme diagnóstico definitivo.
- Não recomende tratamento.
- A classe predita pelo modelo foi: {prediction_label}.
- A probabilidade estimada de malignidade foi: {malignant_probability_percent:.2f}%.
- Essa probabilidade deve ser interpretada como BAIXA quando for menor que 1%.
- Se a classe predita for Benigno e a probabilidade de malignidade for baixa, explique que o modelo indicou baixa probabilidade de malignidade.
- Se a classe predita for Benigno e a probabilidade de malignidade for baixa, NÃO diga que o caso é suspeito de malignidade.
- Não use termos como "alta probabilidade", "probabilidade alta", "risco significativo", "risco considerável", "considerável", "suspeita de malignidade" ou "tratamento agressivo" quando a probabilidade for baixa.
- Explique que o resultado é apoio à triagem e deve ser validado por profissional de saúde.
- Destaque risco de falso negativo e necessidade de avaliação clínica.
- Gere somente as 3 seções solicitadas. Não crie conclusão adicional.

Dados do caso:
- Classe predita pelo modelo: {prediction_label}
- Probabilidade estimada para malignidade: {malignant_probability_percent:.2f}%

Métricas do modelo em teste:
- Accuracy: {model_metrics.get('accuracy', 0):.4f}
- Precision: {model_metrics.get('precision', 0):.4f}
- Recall: {model_metrics.get('recall', 0):.4f}
- F1-score: {model_metrics.get('f1_score', 0):.4f}
- AUC-ROC: {model_metrics.get('auc_roc', 0):.4f}

Principais variáveis consideradas relevantes no projeto:
{', '.join(top_features)}

Gere exatamente 3 seções:
1. Explicação do resultado
2. Pontos de atenção
3. Recomendação de uso responsável
""".strip()

    def generate_with_ollama(self, prompt: str, timeout: int = 45) -> str:
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.7,
                },
            },
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return payload.get("response", "").strip()

    def _response_violates_low_risk_rule(
        self,
        response_text: str,
        prediction_label: str,
        malignant_probability: float,
    ) -> bool:
        is_benign = prediction_label.strip().lower().startswith("benigno")
        is_low_probability = malignant_probability < 0.01

        if not (is_benign and is_low_probability):
            return False

        unsafe_terms = [
            "alta probabilidade",
            "probabilidade alta",
            "risco significativo",
            "risco considerável",
            "considerável",
            "suspeita de malignidade",
            "suspeito de malignidade",
            "tratamento agressivo",
            "alta chance",
            "chance alta",
        ]

        normalized = response_text.lower()
        return any(term in normalized for term in unsafe_terms)

    def generate_controlled_explanation(
        self,
        prediction_label: str,
        malignant_probability: float,
        model_metrics: Dict[str, float],
        top_features: List[str],
    ) -> str:
        malignant_probability_percent = malignant_probability * 100

        return f"""
1. Explicação do resultado

O modelo classificou o caso analisado como {prediction_label}, com probabilidade estimada de malignidade de {malignant_probability_percent:.2f}%. Para este exemplo específico, essa probabilidade é baixa. O resultado deve ser interpretado como apoio à triagem e não como diagnóstico definitivo.

2. Pontos de atenção

Mesmo com métricas elevadas, o modelo pode cometer falsos positivos e falsos negativos. Por isso, o resultado precisa ser analisado junto com histórico clínico, exames complementares e avaliação de profissionais de saúde. O recall de {model_metrics.get('recall', 0):.4f} indica boa capacidade de identificação de casos malignos no conjunto de teste, mas não elimina a necessidade de validação clínica.

As principais variáveis consideradas no projeto incluem: {', '.join(top_features)}.

3. Recomendação de uso responsável

O modelo deve ser usado apenas como ferramenta acadêmica de apoio à decisão, auxiliando na priorização e interpretação inicial dos casos. A decisão clínica final deve sempre ser realizada por profissionais de saúde, considerando o contexto completo do paciente.
""".strip()

    def generate_fallback_explanation(
        self,
        prediction_label: str,
        malignant_probability: float,
        model_metrics: Dict[str, float],
        top_features: List[str],
    ) -> str:
        return self.generate_controlled_explanation(
            prediction_label,
            malignant_probability,
            model_metrics,
            top_features,
        )

    def explain_result(
        self,
        prediction_label: str,
        malignant_probability: float,
        model_metrics: Dict[str, float],
        top_features: List[str],
        use_llm: bool = True,
    ) -> str:
        prompt = self.build_prompt(
            prediction_label,
            malignant_probability,
            model_metrics,
            top_features,
        )

        if use_llm:
            try:
                llm_response = self.generate_with_ollama(prompt)

                if self._response_violates_low_risk_rule(
                    llm_response,
                    prediction_label,
                    malignant_probability,
                ):
                    return (
                        self.generate_controlled_explanation(
                            prediction_label,
                            malignant_probability,
                            model_metrics,
                            top_features,
                        )
                        + "\n\nObservação de segurança: a resposta original da LLM foi descartada por contrariar as regras de interpretação para baixa probabilidade de malignidade."
                    )

                return llm_response

            except Exception as exc:
                return (
                    self.generate_controlled_explanation(
                        prediction_label,
                        malignant_probability,
                        model_metrics,
                        top_features,
                    )
                    + f"\n\nObservação técnica: a chamada ao Ollama não foi concluída ({exc})."
                )

        return self.generate_controlled_explanation(
            prediction_label,
            malignant_probability,
            model_metrics,
            top_features,
        )

    def answer_route_or_diagnosis_question(self, question: str, context: Dict) -> str:
        prompt = f"""
Responda à pergunta abaixo com base no contexto do projeto acadêmico de diagnóstico por Machine Learning.
Não dê diagnóstico médico definitivo.

Pergunta: {question}
Contexto: {json.dumps(context, ensure_ascii=False, indent=2)}
""".strip()

        try:
            return self.generate_with_ollama(prompt)
        except Exception:
            return "Não foi possível consultar a LLM local. Verifique se o Ollama está ativo ou use a explicação estruturada gerada pelo sistema."
