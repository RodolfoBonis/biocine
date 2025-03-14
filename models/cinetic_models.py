"""
Modelos Cinéticos para Bioprocessos

Este módulo implementa os principais modelos cinéticos utilizados
para descrever o crescimento microbiano e o consumo de substrato
em processos de tratamento biológico.
"""

import numpy as np
from scipy.integrate import odeint


class CineticModel:
    """Classe base para modelos cinéticos"""

    def __init__(self, name):
        self.name = name
        self.parameters = {}
        self.fitted_params = None
        self.r_squared = None
        self.mse = None

    def set_parameters(self, **kwargs):
        """Define os parâmetros do modelo"""
        for key, value in kwargs.items():
            if key in self.parameters:
                self.parameters[key] = value

    def fit(self, time, data):
        """
        Ajusta o modelo aos dados experimentais

        Args:
            time: Array com os tempos
            data: Array com os dados experimentais

        Returns:
            self: O próprio objeto com os parâmetros ajustados
        """
        raise NotImplementedError("Este método deve ser implementado nas subclasses")

    def predict(self, time):
        """
        Faz previsões com o modelo

        Args:
            time: Array com os tempos para previsão

        Returns:
            Array com as previsões
        """
        raise NotImplementedError("Este método deve ser implementado nas subclasses")

    def get_params(self):
        """Retorna os parâmetros do modelo"""
        return self.parameters

    def get_fitted_params(self):
        """Retorna os parâmetros ajustados"""
        return self.fitted_params

    def get_metrics(self):
        """Retorna as métricas de qualidade do ajuste"""
        return {
            "r_squared": self.r_squared,
            "mse": self.mse
        }


class MonodModel(CineticModel):
    """
    Modelo de Monod para crescimento microbiano limitado por substrato

    Equações:
    μ = μmax * S / (Ks + S)
    dX/dt = μ * X
    dS/dt = -1/Y * μ * X

    Onde:
    μ: Taxa específica de crescimento (dia^-1)
    μmax: Taxa específica máxima de crescimento (dia^-1)
    S: Concentração de substrato (mg/L)
    Ks: Constante de meia saturação (mg/L)
    X: Concentração de biomassa (g/L)
    Y: Coeficiente de rendimento (g biomassa/g substrato)
    """

    def __init__(self):
        super().__init__("Monod")
        self.parameters = {
            "umax": 0.2,  # Taxa específica máxima de crescimento (dia^-1)
            "ks": 10.0,  # Constante de meia saturação (mg/L)
            "y": 0.5,  # Coeficiente de rendimento (g biomassa/g substrato)
            "x0": 0.1,  # Concentração inicial de biomassa (g/L)
            "s0": 100.0  # Concentração inicial de substrato (mg/L)
        }

    def monod_system(self, y, t, umax, ks, Y):
        """
        Sistema de EDOs do modelo de Monod

        Args:
            y: Lista com [X, S]
            t: Tempo
            umax: Taxa específica máxima de crescimento
            ks: Constante de meia saturação
            Y: Coeficiente de rendimento

        Returns:
            Lista com [dX/dt, dS/dt]
        """
        X, S = y

        # Previne divisão por zero ou valores negativos
        if S <= 0:
            S = 1e-10

        # Taxa específica de crescimento
        u = umax * S / (ks + S)

        # Equações diferenciais
        dXdt = u * X
        dSdt = -1 / Y * u * X

        return [dXdt, dSdt]

    def fit(self, time, data):
        """
        Ajusta o modelo de Monod aos dados experimentais

        Args:
            time: Array com os tempos
            data: Dictionary com {'biomassa': array, 'substrato': array}

        Returns:
            self: O próprio objeto com os parâmetros ajustados
        """
        from scipy.optimize import minimize

        # Função objetivo a ser minimizada
        def objective(params):
            umax, ks, y = params

            # Condições iniciais
            X0 = data['biomassa'][0] if data['biomassa'][0] > 0 else self.parameters['x0']
            S0 = data['substrato'][0] if data['substrato'][0] > 0 else self.parameters['s0']

            # Integração do sistema
            y0 = [X0, S0]
            solution = odeint(self.monod_system, y0, time, args=(umax, ks, y))

            # Dados simulados
            X_sim = solution[:, 0]
            S_sim = solution[:, 1]

            # Erro quadrático
            error_X = np.sum((X_sim - data['biomassa']) ** 2)
            error_S = np.sum((S_sim - data['substrato']) ** 2)

            # Retorna a soma dos erros quadráticos
            return error_X + error_S

        # Limites para os parâmetros (umax, ks, y)
        bounds = [(0.01, 2.0), (1.0, 500.0), (0.01, 1.0)]

        # Valores iniciais
        initial_guess = [
            self.parameters['umax'],
            self.parameters['ks'],
            self.parameters['y']
        ]

        # Otimização
        result = minimize(
            objective,
            initial_guess,
            bounds=bounds,
            method='L-BFGS-B'
        )

        # Atualiza os parâmetros
        umax_fit, ks_fit, y_fit = result.x
        self.fitted_params = {
            "umax": umax_fit,
            "ks": ks_fit,
            "y": y_fit
        }

        # Atualiza os parâmetros do modelo
        self.parameters['umax'] = umax_fit
        self.parameters['ks'] = ks_fit
        self.parameters['y'] = y_fit

        # Calcula as métricas
        X0 = data['biomassa'][0] if data['biomassa'][0] > 0 else self.parameters['x0']
        S0 = data['substrato'][0] if data['substrato'][0] > 0 else self.parameters['s0']

        y0 = [X0, S0]
        solution = odeint(
            self.monod_system,
            y0,
            time,
            args=(umax_fit, ks_fit, y_fit)
        )

        X_sim = solution[:, 0]
        S_sim = solution[:, 1]

        # R² para biomassa
        ss_tot_X = np.sum((data['biomassa'] - np.mean(data['biomassa'])) ** 2)
        ss_res_X = np.sum((data['biomassa'] - X_sim) ** 2)
        r2_X = 1 - (ss_res_X / ss_tot_X) if ss_tot_X > 0 else 0

        # R² para substrato
        ss_tot_S = np.sum((data['substrato'] - np.mean(data['substrato'])) ** 2)
        ss_res_S = np.sum((data['substrato'] - S_sim) ** 2)
        r2_S = 1 - (ss_res_S / ss_tot_S) if ss_tot_S > 0 else 0

        # MSE para biomassa e substrato
        mse_X = np.mean((data['biomassa'] - X_sim) ** 2)
        mse_S = np.mean((data['substrato'] - S_sim) ** 2)

        # Média das métricas
        self.r_squared = (r2_X + r2_S) / 2
        self.mse = (mse_X + mse_S) / 2

        return self

    def predict(self, time):
        """
        Faz previsões com o modelo de Monod

        Args:
            time: Array com os tempos para previsão

        Returns:
            Dictionary com {'biomassa': array, 'substrato': array}
        """
        # Condições iniciais
        X0 = self.parameters['x0']
        S0 = self.parameters['s0']

        # Parâmetros
        umax = self.parameters['umax']
        ks = self.parameters['ks']
        y = self.parameters['y']

        # Integração do sistema
        y0 = [X0, S0]
        solution = odeint(
            self.monod_system,
            y0,
            time,
            args=(umax, ks, y)
        )

        # Retorna os resultados
        return {
            'biomassa': solution[:, 0],
            'substrato': solution[:, 1]
        }


class LogisticModel(CineticModel):
    """
    Modelo Logístico para crescimento microbiano com limitação de recursos

    Equação:
    dX/dt = μmax * X * (1 - X/Xmax)

    Onde:
    X: Concentração de biomassa (g/L)
    μmax: Taxa específica máxima de crescimento (dia^-1)
    Xmax: Concentração máxima de biomassa (g/L)
    """

    def __init__(self):
        super().__init__("Logístico")
        self.parameters = {
            "umax": 0.2,  # Taxa específica máxima de crescimento (dia^-1)
            "x0": 0.1,  # Concentração inicial de biomassa (g/L)
            "xmax": 5.0  # Concentração máxima de biomassa (g/L)
        }

    def logistic_eq(self, X, t, umax, xmax):
        """
        Equação diferencial do modelo Logístico

        Args:
            X: Concentração de biomassa
            t: Tempo
            umax: Taxa específica máxima de crescimento
            xmax: Concentração máxima de biomassa

        Returns:
            dX/dt
        """
        # Previne valores negativos
        if X <= 0:
            X = 1e-10

        return umax * X * (1 - X / xmax)

    def fit(self, time, data):
        """
        Ajusta o modelo Logístico aos dados experimentais

        Args:
            time: Array com os tempos
            data: Array com os dados de biomassa

        Returns:
            self: O próprio objeto com os parâmetros ajustados
        """
        from scipy.optimize import minimize

        # Função objetivo a ser minimizada
        def objective(params):
            umax, xmax = params

            # Condição inicial
            X0 = data[0] if data[0] > 0 else self.parameters['x0']

            # Integração da equação
            solution = odeint(self.logistic_eq, X0, time, args=(umax, xmax))
            X_sim = solution.flatten()

            # Erro quadrático
            error = np.sum((X_sim - data) ** 2)

            return error

        # Limites para os parâmetros (umax, xmax)
        bounds = [(0.01, 2.0), (max(data) * 1.1, max(data) * 5.0)]

        # Valores iniciais
        initial_guess = [
            self.parameters['umax'],
            self.parameters['xmax']
        ]

        # Otimização
        result = minimize(
            objective,
            initial_guess,
            bounds=bounds,
            method='L-BFGS-B'
        )

        # Atualiza os parâmetros
        umax_fit, xmax_fit = result.x
        self.fitted_params = {
            "umax": umax_fit,
            "xmax": xmax_fit
        }

        # Atualiza os parâmetros do modelo
        self.parameters['umax'] = umax_fit
        self.parameters['xmax'] = xmax_fit

        # Calcula as métricas
        X0 = data[0] if data[0] > 0 else self.parameters['x0']
        solution = odeint(
            self.logistic_eq,
            X0,
            time,
            args=(umax_fit, xmax_fit)
        )
        X_sim = solution.flatten()

        # R²
        ss_tot = np.sum((data - np.mean(data)) ** 2)
        ss_res = np.sum((data - X_sim) ** 2)
        self.r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # MSE
        self.mse = np.mean((data - X_sim) ** 2)

        return self

    def predict(self, time):
        """
        Faz previsões com o modelo Logístico

        Args:
            time: Array com os tempos para previsão

        Returns:
            Array com as previsões de biomassa
        """
        # Condição inicial
        X0 = self.parameters['x0']

        # Parâmetros
        umax = self.parameters['umax']
        xmax = self.parameters['xmax']

        # Integração da equação
        solution = odeint(
            self.logistic_eq,
            X0,
            time,
            args=(umax, xmax)
        )

        return solution.flatten()

    def analytical_solution(self, time):
        """
        Solução analítica do modelo Logístico

        Args:
            time: Array com os tempos

        Returns:
            Array com as previsões de biomassa
        """
        X0 = self.parameters['x0']
        umax = self.parameters['umax']
        xmax = self.parameters['xmax']

        # Solução analítica
        X = xmax / (1 + (xmax / X0 - 1) * np.exp(-umax * time))

        return X