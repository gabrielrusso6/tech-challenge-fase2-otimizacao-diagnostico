# Tech Challenge - Fase 2: Otimização de Diagnóstico Médico com Algoritmos Genéticos e LLM

## Visão geral

Este projeto evolui o Tech Challenge da Fase 1, que classificava tumores de mama como benignos ou malignos usando Machine Learning. Na Fase 2, o objetivo é otimizar o modelo de diagnóstico com **Algoritmos Genéticos** e adicionar uma camada de **Processamento de Linguagem Natural com LLM** para explicar os resultados em linguagem natural.

## Projeto escolhido

**Projeto 1: Otimização de Modelos de Diagnóstico**

Motivo da escolha: o projeto reaproveita o pipeline da Fase 1, mantendo continuidade técnica e permitindo focar nas novas exigências da Fase 2:

- otimização de hiperparâmetros com algoritmo genético;
- comparação entre modelos baseline e modelo otimizado;
- execução de pelo menos 3 experimentos com configurações diferentes do algoritmo genético;
- geração de explicações em linguagem natural com LLM;
- logging, rastreabilidade e documentação.

## Dataset

**Breast Cancer Wisconsin Dataset**

- 569 amostras;
- 30 variáveis numéricas;
- target: diagnóstico benigno ou maligno;
- uso acadêmico e educacional.

## Arquitetura

```text
tech-challenge-fase2-otimizacao-diagnostico/
├── data/
│   └── breast-cancer-wisconsin.csv
├── src/
│   ├── baseline_model.py
│   ├── evaluate.py
│   ├── genetic_optimizer.py
│   ├── llm_interpreter.py
│   ├── logger_config.py
│   └── preprocessing.py
├── reports/
│   ├── relatorio_tecnico.md
│   ├── ga_experiments_summary.csv
│   ├── model_comparison.csv
│   ├── sample_llm_explanation.txt
│   └── consolidated_results.json
├── tests/
├── main_phase2.py
├── requirements.txt
├── Dockerfile
└── README.md
```

## Como executar

### 1. Criar ambiente virtual

```bash
python3 -m venv .venv
source .venv/bin/activate
```

No Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

### 3. Executar pipeline completo

```bash
python main_phase2.py
```

### 4. Rodar testes

```bash
pytest
```

### 5. Execução com Docker e Ollama

Para executar o projeto completo em Docker usando o Ollama rodando no host Mac:

```bash
docker build -t tech-challenge-fase2 .

docker run --rm \
  -e OLLAMA_MODEL=llama3.2:1b \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -v "$PWD/reports:/app/reports" \
  -v "$PWD/models:/app/models" \
  -v "$PWD/logs:/app/logs" \
  tech-challenge-fase2
```

## Algoritmo genético

O algoritmo genético otimiza hiperparâmetros do modelo Logistic Regression, que foi o modelo com melhor equilíbrio geral na Fase 1.

### Representação genética

Cada indivíduo representa uma configuração de hiperparâmetros:

```text
{
  "C": força de regularização,
  "penalty": tipo de regularização,
  "class_weight": balanceamento de classes,
  "fit_intercept": uso de intercepto,
  "intercept_scaling": escala do intercepto
}
```

### Operadores implementados

- **Seleção:** torneio;
- **Cruzamento:** crossover uniforme;
- **Mutação:** troca aleatória de genes com base na taxa de mutação;
- **Elitismo:** preserva o melhor indivíduo global.

### Função fitness

A função fitness prioriza métricas relevantes para contexto médico:

```text
fitness = 0.45 * F1-score + 0.35 * Recall + 0.15 * AUC-ROC + 0.05 * Accuracy
```

O recall recebe peso alto porque, em diagnóstico médico, falsos negativos podem representar risco maior.

## Experimentos obrigatórios

O pipeline executa 3 experimentos com configurações diferentes:

| Experimento | População | Gerações | Taxa de mutação | Taxa de crossover |
|---|---:|---:|---:|---:|
| 1 | 6 | 4 | 0.10 | 0.80 |
| 2 | 8 | 5 | 0.20 | 0.85 |
| 3 | 10 | 5 | 0.30 | 0.90 |

Os resultados são salvos em:

```text
reports/ga_experiments_summary.csv
reports/ga_history_experiment_1.csv
reports/ga_history_experiment_2.csv
reports/ga_history_experiment_3.csv
```

## Integração com LLM

A integração com LLM está implementada em `src/llm_interpreter.py`.

Por padrão, o projeto tenta usar o **Ollama local**:

```bash
ollama run llama3.2
```

Variáveis opcionais:

```bash
export OLLAMA_MODEL=llama3.2
export OLLAMA_BASE_URL=http://localhost:11434
```

Caso o Ollama não esteja ativo, o sistema gera uma explicação estruturada de fallback. Isso mantém a demonstração funcionando, mas no vídeo é recomendado demonstrar a execução com LLM local ativa.

## Saídas geradas

Após a execução, a pasta `reports/` terá:

- `model_comparison.csv`: comparação de baseline vs modelo otimizado;
- `ga_experiments_summary.csv`: resumo dos 3 experimentos de algoritmo genético;
- `ga_history_experiment_*.csv`: evolução por geração;
- `feature_importance.csv`: principais variáveis do modelo otimizado;
- `sample_llm_explanation.txt`: explicação gerada pela LLM ou fallback;
- `consolidated_results.json`: resultado consolidado para auditoria e relatório.

## Observação ética

Este projeto tem finalidade acadêmica. O modelo não substitui avaliação médica. A saída deve ser interpretada como apoio à triagem e precisa ser validada por profissionais de saúde qualificados.
