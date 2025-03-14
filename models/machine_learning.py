"""
Modelos de Machine Learning para bioprocessos

Este módulo implementa modelos de aprendizado de máquina para prever
a eficiência de remoção de poluentes e o crescimento da biomassa
em processos de tratamento biológico.
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


class MLModel:
    """Classe base para modelos de machine learning"""

    def __init__(self, name):
        self.name = name
        self.model = None
        self.features = []
        self.target = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.metrics = {
            "train_r2": None,
            "test_r2": None,
            "train_mse": None,
            "test_mse": None
        }

    def __getstate__(self):
        """
        Método especial para serialização do objeto

        Returns:
            dict: Estado do objeto para serialização
        """
        # Obtém o dicionário padrão de atributos
        state = self.__dict__.copy()

        # Adiciona atributos especiais se existirem
        special_attrs = ['feature_names_in_']
        for attr in special_attrs:
            if hasattr(self.model, attr):
                state[f'model_{attr}'] = getattr(self.model, attr)

        return state

    def __setstate__(self, state):
        """
        Método especial para deserialização do objeto

        Args:
            state: Estado do objeto para restauração
        """
        # Restaura o dicionário de atributos
        self.__dict__.update(state)

    def preprocess_data(self, data, features, target, test_size=0.2, random_state=42):
        """
        Pré-processa os dados para treinamento

        Args:
            data: DataFrame com os dados
            features: Lista com os nomes das características
            target: Nome da característica alvo
            test_size: Proporção do conjunto de teste
            random_state: Semente aleatória

        Returns:
            self: O próprio objeto
        """
        self.features = features
        self.target = target

        # Seleciona as características e o alvo
        X = data[features].values
        y = data[target].values.reshape(-1, 1)

        # Divide os dados em treino e teste
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        return self

    def train(self):
        """
        Treina o modelo

        Returns:
            self: O próprio objeto
        """
        raise NotImplementedError("Este método deve ser implementado nas subclasses")

    def predict(self, X):
        """
        Faz previsões com o modelo

        Args:
            X: Array com as características

        Returns:
            Array com as previsões
        """
        if self.model is None:
            raise ValueError("O modelo ainda não foi treinado")

        return self.model.predict(X)

    def evaluate(self):
        """
        Avalia o modelo nos conjuntos de treino e teste

        Returns:
            Dictionary com as métricas de avaliação
        """
        if self.model is None:
            raise ValueError("O modelo ainda não foi treinado")

        # Previsões
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)

        # Métricas
        self.metrics["train_r2"] = r2_score(self.y_train, y_train_pred)
        self.metrics["test_r2"] = r2_score(self.y_test, y_test_pred)
        self.metrics["train_mse"] = mean_squared_error(self.y_train, y_train_pred)
        self.metrics["test_mse"] = mean_squared_error(self.y_test, y_test_pred)

        return self.metrics

    def get_model(self):
        """Retorna o modelo treinado"""
        return self.model

    def get_metrics(self):
        """Retorna as métricas de avaliação"""
        return self.metrics

    def get_feature_importance(self):
        """
        Retorna a importância das características

        Returns:
            DataFrame com a importância das características
        """
        raise NotImplementedError("Este método deve ser implementado nas subclasses")


class RandomForestModel(MLModel):
    """
    Modelo Random Forest para regressão
    """

    def __init__(self, **kwargs):
        """
        Inicializa o modelo Random Forest

        Args:
            **kwargs: Parâmetros do modelo
                n_estimators: Número de árvores
                max_depth: Profundidade máxima das árvores
                min_samples_split: Número mínimo de amostras para dividir um nó
                min_samples_leaf: Número mínimo de amostras em uma folha
                random_state: Semente aleatória
        """
        super().__init__("Random Forest")

        # Parâmetros padrão
        self.params = {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": 42
        }

        # Atualiza os parâmetros
        self.params.update(kwargs)

    def train(self):
        """
        Treina o modelo Random Forest

        Returns:
            self: O próprio objeto
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Os dados não foram pré-processados")

        # Inicializa o modelo
        self.model = RandomForestRegressor(**self.params)

        # Treina o modelo
        self.model.fit(self.X_train, self.y_train.ravel())

        # Avalia o modelo
        self.evaluate()

        return self

    def get_feature_importance(self):
        """
        Retorna a importância das características

        Returns:
            DataFrame com a importância das características
        """
        if self.model is None:
            raise ValueError("O modelo ainda não foi treinado")

        # Importância das características
        importance = self.model.feature_importances_

        # Cria um DataFrame
        importance_df = pd.DataFrame({
            'Feature': self.features,
            'Importance': importance
        })

        # Ordena pelo valor de importância
        importance_df = importance_df.sort_values('Importance', ascending=False)

        return importance_df