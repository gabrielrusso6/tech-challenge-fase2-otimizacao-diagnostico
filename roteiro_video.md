# Roteiro sugerido para o vídeo - até 15 minutos

## 1. Abertura - 1 minuto

Apresentar o projeto: evolução do Tech Challenge Fase 1 para a Fase 2, com foco em otimização de diagnóstico médico usando algoritmos genéticos e LLM.

## 2. Contexto e escolha do projeto - 2 minutos

Explicar que foi escolhido o Projeto 1 porque aproveita o pipeline de câncer de mama da Fase 1. Mostrar rapidamente o dataset e o objetivo de classificar tumores como benignos ou malignos.

## 3. Arquitetura - 2 minutos

Mostrar a estrutura de pastas e explicar cada módulo: preprocessing, baseline, genetic optimizer, evaluate, llm interpreter e main.

## 4. Algoritmo genético - 4 minutos

Explicar genes, função fitness, seleção por torneio, crossover, mutação e elitismo. Mostrar que existem 3 experimentos com configurações diferentes.

## 5. Execução do sistema - 3 minutos

Rodar:

```bash
python main_phase2.py
```

Mostrar os arquivos gerados em reports/.

## 6. Integração com LLM - 2 minutos

Mostrar o arquivo sample_llm_explanation.txt. Explicar que o sistema tenta usar Ollama local e, se não estiver disponível, usa fallback estruturado para demonstração.

## 7. Fechamento - 1 minuto

Reforçar resultados, limitações éticas e que o modelo é apoio educacional à triagem, não diagnóstico definitivo.
