import pickle
from pathlib import Path
import pandas as pd

def load_pkl(file_path):
    try:
        print(f"Carregando pickle: {file_path}")
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Erro ao carregar {file_path}: {e}")
        return None


class Predictions:
    def __init__(self):
        base_path = Path("model_pipeline/model")
        self.preprocessor = load_pkl(base_path / "preprocessor.pkl")
        self.predict_model = load_pkl(base_path / "model.pkl")

    def predict(self, X: dict):
        if self.preprocessor is None or self.predict_model is None:
            print("Modelo ou pré-processador não carregados.")
            return None

        # Converte dict em DataFrame de uma linha
        df = pd.DataFrame([X])

        # Ordena dinamicamente as colunas em ordem alfabética
        df = df[sorted(df.columns)]

        # Aplica pré-processamento
        X_processed = self.preprocessor.transform(df)

        # Retorna a previsão
        return [int(x) for x in self.predict_model.predict(X_processed)]


# Instância global
predictions = Predictions()
