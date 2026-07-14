# Relatório Técnico - Tech Challenge Fase 2

## 1. Introdução

Este relatório apresenta a evolução do projeto de diagnóstico de câncer de mama desenvolvido na Fase 1. A nova etapa adiciona otimização de hiperparâmetros com Algoritmos Genéticos e uma camada de interpretação em linguagem natural utilizando LLM.

O projeto escolhido foi o **Projeto 1: Otimização de Modelos de Diagnóstico**, por ser uma continuidade natural do pipeline de Machine Learning já desenvolvido anteriormente.

## 2. Problema abordado

O problema consiste em classificar tumores como benignos ou malignos a partir de características clínicas numéricas do dataset Breast Cancer Wisconsin. Na Fase 2, além de manter o pipeline de classificação, o objetivo foi melhorar a busca por hiperparâmetros e facilitar a interpretação dos resultados por profissionais de saúde.

## 3. Arquitetura da solução

A solução foi estruturada em módulos Python:

- `preprocessing.py`: carregamento, limpeza, encoding, divisão treino/teste, imputação e normalização;
- `baseline_model.py`: definição dos modelos originais usados como comparação;
- `genetic_optimizer.py`: implementação do algoritmo genético;
- `evaluate.py`: cálculo de métricas e geração de comparativos;
- `llm_interpreter.py`: geração de explicações em linguagem natural;
- `logger_config.py`: logging e rastreabilidade;
- `main_phase2.py`: orquestração do pipeline completo.

A divisão treino/teste é feita antes da normalização para evitar vazamento de dados.

## 4. Algoritmo genético

O algoritmo genético foi utilizado para otimizar hiperparâmetros do modelo Logistic Regression. Esse modelo foi escolhido por ter apresentado ótimo equilíbrio na Fase 1. Cada indivíduo da população representa uma configuração possível de hiperparâmetros.

### 4.1 Representação dos genes

Os genes utilizados foram:

- `C`;
- `penalty`;
- `class_weight`;
- `fit_intercept`;
- `intercept_scaling`.

### 4.2 Operadores genéticos

Foram implementados:

- seleção por torneio;
- crossover uniforme;
- mutação por troca aleatória de hiperparâmetros;
- elitismo, preservando o melhor indivíduo global.

### 4.3 Função fitness

A função fitness foi definida considerando o contexto médico:

```text
fitness = 0.45 * F1-score + 0.35 * Recall + 0.15 * AUC-ROC + 0.05 * Accuracy
```

O F1-score foi priorizado por equilibrar precision e recall. O recall também recebeu peso alto porque falsos negativos são especialmente críticos em problemas de triagem diagnóstica.

## 5. Experimentos realizados

Foram definidos três experimentos com configurações diferentes de população, gerações, taxa de mutação e taxa de crossover:

| Experimento | População | Gerações | Mutação | Crossover |
|---|---:|---:|---:|---:|
| 1 | 6 | 4 | 0.10 | 0.80 |
| 2 | 8 | 5 | 0.20 | 0.85 |
| 3 | 10 | 5 | 0.30 | 0.90 |

Os resultados são gerados automaticamente no arquivo `reports/ga_experiments_summary.csv`.

## 6. Comparação entre modelos originais e otimizados

O projeto treina modelos baseline da Fase 1 e compara seus resultados com a Logistic Regression otimizada por algoritmo genético.

As métricas avaliadas são:

- accuracy;
- precision;
- recall;
- F1-score;
- AUC-ROC;
- matriz de confusão.

O arquivo `reports/model_comparison.csv` contém o comparativo final.

## 7. Integração com LLM

A integração com LLM foi implementada com suporte ao Ollama local. O sistema constrói um prompt contendo:

- classe predita;
- probabilidade estimada de malignidade;
- métricas do modelo;
- principais variáveis consideradas relevantes;
- instruções de segurança para evitar diagnóstico definitivo.

A LLM gera uma explicação em linguagem natural para apoiar profissionais de saúde na interpretação do resultado. Caso o Ollama não esteja disponível, o sistema usa uma explicação estruturada de fallback para manter a demonstração funcionando.

## 8. Prompt engineering

O prompt orienta a LLM a:

- explicar o resultado em português claro;
- não afirmar diagnóstico definitivo;
- destacar que o modelo é apoio à triagem;
- citar pontos de atenção;
- recomendar validação por profissionais qualificados.

## 9. Avaliação da qualidade das interpretações

A qualidade da explicação gerada é avaliada com base em critérios qualitativos:

- clareza da linguagem;
- aderência ao contexto médico;
- presença de aviso de limitação clínica;
- coerência com as métricas do modelo;
- utilidade para interpretação por equipe médica.

## 10. Desafios e soluções

Um dos principais desafios foi evitar que a otimização priorizasse apenas accuracy. Em problemas médicos, accuracy isolada pode mascarar falsos negativos. Por isso, a função fitness combinou F1-score, recall, AUC-ROC e accuracy.

Outro desafio foi manter a integração com LLM executável em ambiente acadêmico. Para isso, foi usado Ollama local com fallback estruturado, permitindo que o projeto rode mesmo sem chave de API externa.

## 11. Limitações

O projeto utiliza um dataset público e limitado, com finalidade educacional. A solução não substitui diagnóstico médico e não deve ser utilizada em produção clínica sem validações adicionais, revisão ética, análise de vieses e validação prospectiva.

## 12. Conclusão

A Fase 2 ampliou o projeto inicial ao adicionar otimização evolutiva e interpretação textual dos resultados. O uso de algoritmos genéticos permitiu explorar hiperparâmetros de forma automatizada, enquanto a integração com LLM melhorou a apresentação dos resultados para humanos. A solução mantém rastreabilidade, organização de código e documentação adequada para demonstração acadêmica.


## 13. Resultados obtidos nesta execução

### 13.1 Experimentos do algoritmo genético

Nesta execução, os três experimentos obrigatórios foram executados com sucesso. O melhor fitness de cada experimento foi:

| Experimento | População | Gerações | Taxa de mutação | Fitness | F1-score CV | Recall CV | AUC-ROC CV |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6 | 4 | 0.10 | 0.9695 | 0.9646 | 0.9647 | 0.9938 |
| 2 | 8 | 5 | 0.20 | 0.9732 | 0.9676 | 0.9706 | 0.9956 |
| 3 | 10 | 5 | 0.30 | 0.9754 | 0.9760 | 0.9647 | 0.9962 |


### 13.2 Comparativo final em teste

O comparativo final mostrou que o modelo otimizado por algoritmo genético superou os modelos baseline na base de teste reservada.

| Modelo | Accuracy | Precision | Recall | F1-score | AUC-ROC |
|---|---:|---:|---:|---:|---:|
| LogisticRegression_GA_optimized | 0.9912 | 1.0000 | 0.9762 | 0.9880 | 0.9980 |
| LogisticRegression_baseline | 0.9737 | 0.9756 | 0.9524 | 0.9639 | 0.9960 |
| SVM_baseline | 0.9737 | 1.0000 | 0.9286 | 0.9630 | 0.9947 |
| RandomForest_baseline | 0.9737 | 1.0000 | 0.9286 | 0.9630 | 0.9929 |
| KNN_baseline | 0.9561 | 0.9744 | 0.9048 | 0.9383 | 0.9823 |
| NaiveBayes_baseline | 0.9211 | 0.9231 | 0.8571 | 0.8889 | 0.9891 |


O melhor resultado foi obtido pela Logistic Regression otimizada por algoritmo genético, com melhoria em F1-score, recall e AUC-ROC quando comparada à Logistic Regression baseline.
