<img src="/static/images/fiap.png" alt="Logo" style="height: 24px; width: auto;" />

# API — Predição de Empréstimo

Bem-vindo! Esta API usa um modelo de Machine Learning para prever a aprovação de empréstimo (classe 0 ou 1).

## Como a API trabalha

1. Você envia um JSON com os dados do solicitante
2. O sistema prepara os dados (usa o mesmo pré-processamento do treino)
3. O modelo calcula a previsão e retornamos `{ "pred": 0 | 1 }`

Os arquivos do modelo ficam em `model_pipeline/model/`:
- `preprocessor.pkl` (transforma os dados)
- `model.pkl` (faz a previsão)

## Endpoint principal

- POST `/api/pred/pred_loan_classification`
  - Entrada: veja o schema `PredRequest` (exemplo abaixo)
  - Saída: `{ "pred": 0 | 1 }`
    - 0 = Aprovado
    - 1 = Rejeitado