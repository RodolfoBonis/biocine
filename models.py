import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


class CineticModels:
    """
    Classe para modelagem cinética de processos de tratamento biológico
    """

    @staticmethod
    def monod_model(S, umax, Ks):
        """
        Modelo de Monod para a taxa específica de crescimento

        Args:
            S: Concentração de substrato
            umax: Taxa máxima de crescimento
            Ks: Constante de meia saturação

        Returns:
            Taxa específica de crescimento (μ)
        """
        return umax * S / (Ks + S)

    @staticmethod
    def logistic_model(t, X0, umax, Xmax):
        """
        Modelo Logístico para o crescimento de biomassa

        Args:
            t: Tempo
            X0: Concentração inicial de biomassa
            umax: Taxa máxima de crescimento
            Xmax: Capacidade máxima do sistema

        Returns:
            Concentração de biomassa (X)
        """
        return Xmax / (1 + ((Xmax - X0) / X0) * np.exp(-umax * t))

    @staticmethod
    def substrate_consumption(t, S0, umax, Ks, Y_XS, X0, Xmax):
        """
        Modelo para o consumo de substrato baseado em Monod

        Args:
            t: Tempo
            S0: Concentração inicial de substrato
            umax: Taxa máxima de crescimento
            Ks: Constante de meia saturação
            Y_XS: Coeficiente de rendimento (biomassa/substrato)
            X0: Concentração inicial de biomassa
            Xmax: Capacidade máxima do sistema

        Returns:
            Concentração de substrato (S)
        """
        # Calculando a biomassa em cada ponto do tempo
        X = CineticModels.logistic_model(t, X0, umax, Xmax)

        # Calculando a variação de biomassa
        dX = X - X0

        # Calculando o consumo de substrato
        S = S0 - dX / Y_XS

        # Garantindo que S não fique negativo
        return np.maximum(S, 0)

    @staticmethod
    def calculate_growth_rates(time_data, biomass_data):
        """
        Calcula as taxas de crescimento a partir dos dados de biomassa

        Args:
            time_data: Array com os tempos
            biomass_data: Array com as concentrações de biomassa

        Returns:
            growth_rates: Taxas de crescimento
        """
        growth_rates = []

        for i in range(1, len(time_data)):
            dt = time_data[i] - time_data[i - 1]
            dX = biomass_data[i] - biomass_data[i - 1]
            X_avg = (biomass_data[i] + biomass_data[i - 1]) / 2

            # Taxa específica de crescimento (dia^-1)
            mu = (1 / X_avg) * (dX / dt)
            growth_rates.append(mu)

        return np.array(growth_rates)

    @staticmethod
    def fit_logistic_model(time_data, biomass_data, is_biomass=True):
        """
        Ajusta o modelo logístico aos dados experimentais

        Args:
            time_data: Array com os tempos
            biomass_data: Array com as concentrações de biomassa ou poluentes
            is_biomass: Boolean indicando se os dados são de biomassa (True) ou poluentes (False)

        Returns:
            popt: Parâmetros otimizados (X0, umax, Xmax)
            pcov: Matriz de covariância
        """
        if is_biomass:
            # Para biomassa (crescimento)
            # Estimativas iniciais
            X0_guess = biomass_data[0]
            Xmax_guess = max(biomass_data) * 1.2
            umax_guess = 0.1

            # Limites para os parâmetros (limites inferiores e superiores)
            bounds = ([X0_guess * 0.5, 0.001, Xmax_guess * 0.5],
                      [X0_guess * 1.5, 10.0, Xmax_guess * 2.0])
        else:
            # Para poluentes (decaimento)
            # Estimativas iniciais - para modelo de decaimento
            X0_guess = biomass_data[0]  # Concentração inicial
            Xmax_guess = X0_guess * 0.1  # Concentração final estimada (10% da inicial)
            umax_guess = 0.1

            # Limites para os parâmetros (limites inferiores e superiores)
            # Para modelos de decaimento, invertemos a lógica dos bounds para X0 e Xmax
            bounds = ([X0_guess * 0.5, 0.001, 0.0],
                      [X0_guess * 1.5, 10.0, X0_guess * 0.5])

        # Ajuste do modelo
        popt, pcov = curve_fit(
            CineticModels.logistic_model,
            time_data,
            biomass_data,
            p0=[X0_guess, umax_guess, Xmax_guess],
            bounds=bounds,
            maxfev=10000
        )

        return popt, pcov

    @staticmethod
    def fit_monod_parameters(substrate_data, growth_rate_data):
        """
        Ajusta os parâmetros do modelo de Monod aos dados experimentais

        Args:
            substrate_data: Array com as concentrações de substrato
            growth_rate_data: Array com as taxas de crescimento

        Returns:
            popt: Parâmetros otimizados (umax, Ks)
            pcov: Matriz de covariância
        """
        # Estimativas iniciais
        umax_guess = max(growth_rate_data) * 1.2
        Ks_guess = np.median(substrate_data)

        # Limites para os parâmetros
        bounds = ([0.001, 0.001], [10.0, 1000.0])

        # Ajuste do modelo
        popt, pcov = curve_fit(
            CineticModels.monod_model,
            substrate_data,
            growth_rate_data,
            p0=[umax_guess, Ks_guess],
            bounds=bounds,
            maxfev=10000
        )

        return popt, pcov


class MachineLearningModels:
    """
    Classe para modelos de machine learning aplicados ao tratamento biológico
    """

    @staticmethod
    def prepare_data(df, target_column, feature_columns):
        """
        Prepara os dados para o treinamento do modelo de machine learning

        Args:
            df: DataFrame com os dados
            target_column: Nome da coluna alvo
            feature_columns: Lista de nomes das colunas de características

        Returns:
            X_train, X_test, y_train, y_test, scaler_X, scaler_y
        """
        # Separar características e alvo
        X = df[feature_columns].values
        y = df[target_column].values.reshape(-1, 1)

        # Escalar os dados
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y)

        # Dividir em conjuntos de treinamento e teste
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=0.2, random_state=42
        )

        return X_train, X_test, y_train, y_test, scaler_X, scaler_y

    @staticmethod
    def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None):
        """
        Treina um modelo Random Forest

        Args:
            X_train: Características de treinamento
            y_train: Alvo de treinamento
            n_estimators: Número de árvores na floresta
            max_depth: Profundidade máxima das árvores

        Returns:
            Modelo treinado
        """
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )

        model.fit(X_train, y_train.ravel())

        return model

    @staticmethod
    def evaluate_model(model, X_test, y_test, scaler_y=None):
        """
        Avalia o modelo treinado

        Args:
            model: Modelo treinado
            X_test: Características de teste
            y_test: Alvo de teste
            scaler_y: Scaler para reverter a normalização do alvo

        Returns:
            mse: Erro quadrático médio
            r2: Coeficiente de determinação (R²)
        """
        # Fazendo previsões
        y_pred = model.predict(X_test)

        # Revertendo a normalização, se necessário
        if scaler_y is not None:
            y_pred_orig = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
            y_test_orig = scaler_y.inverse_transform(y_test)
        else:
            y_pred_orig = y_pred.reshape(-1, 1)
            y_test_orig = y_test

        # Calculando métricas
        mse = mean_squared_error(y_test_orig, y_pred_orig)
        r2 = r2_score(y_test_orig, y_pred_orig)

        return mse, r2

    @staticmethod
    def predict(model, input_data, scaler_X=None, scaler_y=None):
        """
        Faz previsões com o modelo treinado

        Args:
            model: Modelo treinado
            input_data: Dados de entrada
            scaler_X: Scaler para normalizar os dados de entrada
            scaler_y: Scaler para reverter a normalização da previsão

        Returns:
            Previsão do modelo
        """
        # Normalizando dados de entrada, se necessário
        if scaler_X is not None:
            input_data_scaled = scaler_X.transform(input_data)
        else:
            input_data_scaled = input_data

        # Fazendo previsão
        pred_scaled = model.predict(input_data_scaled)

        # Revertendo a normalização, se necessário
        if scaler_y is not None:
            prediction = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))
        else:
            prediction = pred_scaled.reshape(-1, 1)

        return prediction


class DataAnalysis:
    """
    Classe para análise de dados de tratamento biológico
    """

    @staticmethod
    def calculate_removal_efficiency(df, pollutant_column):
        """
        Calcula a eficiência de remoção de um poluente

        Args:
            df: DataFrame com os dados
            pollutant_column: Nome da coluna do poluente

        Returns:
            efficiences: Array com as eficiências de remoção em cada ponto
            final_efficiency: Eficiência de remoção final
        """
        initial_value = df[pollutant_column].iloc[0]
        values = df[pollutant_column].values

        # Calculando eficiência em cada ponto
        efficiencies = [100 * (1 - value / initial_value) for value in values]

        # Eficiência final
        final_efficiency = efficiencies[-1]

        return efficiencies, final_efficiency

    @staticmethod
    def calculate_biomass_productivity(df, biomass_column, time_column="Tempo (dias)"):
        """
        Calcula a produtividade de biomassa

        Args:
            df: DataFrame com os dados
            biomass_column: Nome da coluna de biomassa
            time_column: Nome da coluna de tempo

        Returns:
            productivity: Produtividade de biomassa (g/L/dia)
        """
        # Biomassa inicial e final
        initial_biomass = df[biomass_column].iloc[0]
        final_biomass = df[biomass_column].iloc[-1]

        # Tempo total
        initial_time = df[time_column].iloc[0]
        final_time = df[time_column].iloc[-1]
        time_elapsed = final_time - initial_time

        # Produtividade
        productivity = (final_biomass - initial_biomass) / time_elapsed

        return productivity

    @staticmethod
    def calculate_correlation_matrix(df, columns=None):
        """
        Calcula a matriz de correlação entre os parâmetros

        Args:
            df: DataFrame com os dados
            columns: Lista de colunas para calcular a correlação

        Returns:
            correlation_matrix: Matriz de correlação
        """
        if columns is None:
            # Excluindo a coluna de tempo
            columns = [col for col in df.columns if col != "Tempo (dias)"]

        return df[columns].corr()