# FIAP Tech Challenge — Fase 3

Sistema simples para prever risco de crédito (aprovação de empréstimo) usando um modelo de Machine Learning. A API é feita em FastAPI e o modelo fica salvo em arquivos `.pkl` para uso em produção.

## O que este projeto faz

- Recebe dados de uma pessoa que pediu empréstimo
- Prepara esses dados com um pré-processador salvo
- Usa um modelo treinado para prever a classe (0 ou 1)
- Retorna a previsão em um endpoint HTTP

## Como funciona

- **FastAPI**: cuida da API e das páginas.
- **Modelo e pré-processador**: salvos em `model_pipeline/model/preprocessor.pkl` e `model_pipeline/model/model.pkl`.
- **Inferência**: ao receber um JSON, o sistema transforma os dados (pandas DataFrame), aplica o pré-processador e faz `model.predict`.
- **Documentação**: a UI do Swagger fica em `/docs` e usa o texto do arquivo `SWAGGER.MD`.

## Dados usados e tratamento

- Base usada: `model_pipeline/data/credit.csv` (formato German Credit, com coluna-alvo `default`).
- Em produção, aplicamos o mesmo pré-processamento do treino (codificação de texto, escala de números, seleção de colunas). Isso já está dentro do `preprocessor.pkl`.
- Observação: no payload da API existe o campo `faixa_idade`. Ele é convertido para uma "faixa" (por exemplo, 30 → 3) antes de ir para o modelo.

## Como rodar o projeto

1. Instale as dependências (Python 3.11 + Poetry):
   ```bash
   poetry install
   ```
2. Crie um arquivo `.env` na raiz (opcional, mas recomendado) com, por exemplo:
   ```env
   TITLE=FIAP Tech Challenge
   SECRET_KEY=uma_chave_segura_aqui
   HOST=0.0.0.0
   PORT=5000
   ```
3. Coloque os arquivos do modelo nos caminhos:
   - `model_pipeline/model/preprocessor.pkl`
   - `model_pipeline/model/model.pkl`
4. Rode a aplicação:
   ```bash
   poetry run python main.py
   ```
5. Acesse:
   - Home: http://localhost:5000/
   - Documentação (Swagger UI): http://localhost:5000/docs

> Dica: sem os arquivos `*.pkl`, a API não conseguirá gerar previsões.

## Endpoint principal

- **POST `/api/pred/pred_loan_classification`**
  - Entrada (JSON):
    - **valor_emprestimo**: valor solicitado (R$)
    - **prazo_emprestimo_anos**: prazo do empréstimo em anos
    - **faixa_idade**: idade (ex.: 30)
    - **outros_planos_financiamento**: 0 ou 1
    - **historico_credito**: categoria numérica (0/1)
    - **propriedade**: categoria numérica (0/1/2)
    - **tempo_emprego_atual**: categoria numérica (0/1)
    - **reserva_cc**: categoria numérica (0/1)
    - **tipo_residencia**: 0 (alugada) ou 1 (própria)
    - **conta_corrente**: 0 (negativada/sem) ou 1 (positiva)

  - Exemplo de requisição:
    ```json
    {
      "valor_emprestimo": 10000,
      "prazo_emprestimo_anos": 5,
      "faixa_idade": 30,
      "outros_planos_financiamento": 0,
      "historico_credito": 1,
      "propriedade": 2,
      "tempo_emprego_atual": 1,
      "reserva_cc": 1,
      "tipo_residencia": 1,
      "conta_corrente": 1
    }
    ```

  - Exemplo de resposta:
    ```json
    { "pred": 0 }
    ```

> A saída `pred` é a classe prevista pelo modelo (0 = Aprovado, ou 1 = Rejeitado).

## Estrutura do projeto (resumo)

```
tech_challenge_fase3/
├── app/
│   ├── core/            # Config, paths, templates e alerts
│   ├── routers/
│   │   ├── api/         # Endpoints de API (ex.: pred.py)
│   │   └── pages/       # Páginas (/, /docs)
│   ├── schemas/ml/      # Carga do modelo e validação do payload
│   └── static/          # Imagens, CSS, JS (inclui assets do Swagger)
├── model_pipeline/
│   ├── data/credit.csv
│   └── model/{preprocessor.pkl, model.pkl}
├── main.py              # Cria e inicia a aplicação FastAPI
├── SWAGGER.MD           # Texto mostrado no Swagger UI
└── pyproject.toml       # Dependências (Poetry)
```


