import json
import os
from typing import Dict, List, Optional

import requests


class DiagnosisLLMInterpreter:
    """Integração com LLM para interpretação de diagnósticos.

    A integração suporta Ollama local via HTTP. Caso o Ollama não esteja ativo,
    o sistema gera uma explicação estruturada de fallback para manter a demonstração
    funcionando em ambiente acadêmico.
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
- Não diga que o modelo foi treinado com mais de 10.000 casos.
- A classe predita pelo modelo foi: {prediction_label}.
- A probabilidade estimada de malignidade foi: {malignant_probability_percent:.2f}%.
- Se a classe predita for Benigno, explique que o modelo indicou baixa probabilidade de malignidade.
- Se a classe predita for Maligno, explique que o modelo indicou alta probabilidade de malignidade.
- Explique que o resultado é apoio à triagem e deve ser validado por profissional de saúde.
- Destaque risco de falso negativo e necessidade de avaliação clínica.

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

    def generate_fallback_explanation(
        self,
        prediction_label: str,
        malignant_probability: float,
        model_metrics: Dict[str, float],
        top_features: List[str],
    ) -> str:
        return f"""
Explicação gerada em modo demonstração, sem chamada externa para LLM.

O modelo classificou o caso como {prediction_label}, com probabilidade estimada de malignidade de {malignant_probability:.2%}.
Essa saída deve ser entendida como apoio à triagem, não como diagnóstico definitivo. Em contexto médico, o recall de {model_metrics.get('recall', 0):.4f} é uma métrica especialmente importante, pois indica a capacidade do modelo de identificar corretamente casos malignos.

As variáveis mais relevantes observadas no projeto incluem: {', '.join(top_features)}. Esses atributos estão relacionados a características morfológicas do tumor e ajudam o modelo a separar padrões associados a casos benignos e malignos.

Pontos de atenção: resultados preditivos precisam ser analisados junto com histórico clínico, exames complementares e avaliação de profissionais de saúde. Mesmo com boa performance estatística, podem existir falsos positivos e falsos negativos.

Recomendação: usar o modelo apenas como ferramenta educacional de apoio à decisão e priorização de análise, sempre com validação médica especializada.
""".strip()

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
                return self.generate_with_ollama(prompt)
            except Exception as exc:
                return self.generate_fallback_explanation(
                    prediction_label,
                    malignant_probability,
                    model_metrics,
                    top_features,
                ) + f"\n\nObservação técnica: a chamada ao Ollama não foi concluída ({exc})."

        return self.generate_fallback_explanation(
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
