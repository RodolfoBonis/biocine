"""
Otimização avançada de bioprocessos

Este módulo implementa algoritmos avançados de otimização para processos
de tratamento biológico, incluindo:
- Algoritmos genéticos
- Otimização multiobjetivo
- Planejamento experimental (DoE)
- Superfícies de resposta
- Otimização com restrições
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize, differential_evolution, dual_annealing
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor


class ProcessOptimizer:
    """
    Otimizador avançado para processos de tratamento biológico
    """

    def __init__(self, models=None, constraints=None):
        """
        Inicializa o otimizador de processo

        Args:
            models: Dicionário de modelos preditivos {alvo: modelo}
            constraints: Lista de funções que definem restrições
        """
        self.models = models if models is not None else {}
        self.constraints = constraints if constraints is not None else []
        self.optimization_results = {}
        self.parameter_bounds = {}
        self.parameter_names = []
        self.objective_function = None
        self.objective_mode = 'maximize'  # Ou 'minimize'
        self.surrogate_model = None
        self.experimental_design = None

    def add_model(self, target, model):
        """
        Adiciona um modelo preditivo ao otimizador

        Args:
            target: Nome do alvo (ex: 'biomassa', 'eficiencia_remocao_n')
            model: Modelo treinado com método predict()
        """
        self.models[target] = model

    def set_parameter_bounds(self, bounds_dict):
        """
        Define os limites dos parâmetros

        Args:
            bounds_dict: Dicionário {param_name: (lower_bound, upper_bound)}
        """
        self.parameter_bounds = bounds_dict
        self.parameter_names = list(bounds_dict.keys())

    def set_objective(self, objective_func=None, mode='maximize', target=None):
        """
        Define a função objetivo para otimização

        Args:
            objective_func: Função personalizada de X -> valor
            mode: 'maximize' ou 'minimize'
            target: Alvo para otimizar se não for fornecida função específica
        """
        self.objective_mode = mode

        if objective_func is not None:
            # Função personalizada
            self.objective_function = objective_func
        elif target is not None and target in self.models:
            # Usando modelo predefinido
            if mode == 'maximize':
                def obj_func(X):
                    X_dict = {name: val for name, val in zip(self.parameter_names, X)}
                    return -float(self.models[target].predict(pd.DataFrame([X_dict]))[0])
            else:
                def obj_func(X):
                    X_dict = {name: val for name, val in zip(self.parameter_names, X)}
                    return float(self.models[target].predict(pd.DataFrame([X_dict]))[0])

            self.objective_function = obj_func
        else:
            raise ValueError("Forneça uma função objetivo válida ou um alvo válido")

    def add_constraint(self, constraint_func, constraint_type='ineq'):
        """
        Adiciona uma restrição à otimização

        Args:
            constraint_func: Função que define a restrição
            constraint_type: 'eq' para igualdade (=0), 'ineq' para desigualdade (>=0)
        """
        self.constraints.append({
            'type': constraint_type,
            'fun': constraint_func
        })

    def optimize_process(self, method='SLSQP', max_iter=100, population=50):
        """
        Realiza a otimização do processo usando diferentes algoritmos

        Args:
            method: Método de otimização ('SLSQP', 'genetic', 'annealing')
            max_iter: Número máximo de iterações
            population: Tamanho da população para algoritmos evolutivos

        Returns:
            Dictionary com resultados da otimização
        """
        if self.objective_function is None:
            raise ValueError("Defina uma função objetivo antes de otimizar")

        if not self.parameter_bounds:
            raise ValueError("Defina os limites dos parâmetros antes de otimizar")

        # Prepara os limites no formato correto para otimização
        bounds = [self.parameter_bounds[param] for param in self.parameter_names]

        # Ponto inicial (médio entre limites inferior e superior)
        x0 = [(b[0] + b[1]) / 2 for b in bounds]

        # Escolhe o método de otimização
        if method == 'SLSQP':
            # Otimização com restrições
            result = minimize(
                self.objective_function,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=self.constraints,
                options={'maxiter': max_iter, 'disp': True}
            )

            optimal_params = result.x
            optimal_value = result.fun
            success = result.success
            message = result.message

        elif method == 'genetic':
            # Algoritmo genético
            result = differential_evolution(
                self.objective_function,
                bounds,
                maxiter=max_iter,
                popsize=population,
                strategy='best1bin',
                disp=True
            )

            optimal_params = result.x
            optimal_value = result.fun
            success = result.success
            message = result.message

        elif method == 'annealing':
            # Simulated annealing
            result = dual_annealing(
                self.objective_function,
                bounds,
                maxiter=max_iter
            )

            optimal_params = result.x
            optimal_value = result.fun
            success = result.success
            message = result.message

        else:
            raise ValueError(f"Método de otimização desconhecido: {method}")

        # Ajusta o valor se o objetivo era maximizar
        if self.objective_mode == 'maximize':
            optimal_value = -optimal_value

        # Formata os resultados
        opt_params_dict = {
            name: value for name, value in zip(self.parameter_names, optimal_params)
        }

        # Armazena os resultados
        optimization_result = {
            'optimal_parameters': opt_params_dict,
            'optimal_value': optimal_value,
            'success': success,
            'message': message,
            'method': method,
            'mode': self.objective_mode,
            'raw_result': result
        }

        # Armazena o resultado no histórico
        self.optimization_results[method] = optimization_result

        return optimization_result

    def multi_objective_optimization(self, target_weights, n_points=100):
        """
        Realiza otimização multiobjetivo com pesos

        Args:
            target_weights: Dicionário {target: weight}
            n_points: Número de pontos Pareto

        Returns:
            DataFrame com pontos Pareto
        """
        # Verifica se há pelo menos dois alvos
        targets = list(target_weights.keys())
        if len(targets) < 2:
            raise ValueError("Otimização multiobjetivo requer pelo menos dois alvos")

        # Verifica se todos os alvos têm modelos
        for target in targets:
            if target not in self.models:
                raise ValueError(f"Não existe modelo para o alvo: {target}")

        # Função objetivo ponderada
        def weighted_objective(X):
            X_dict = {name: val for name, val in zip(self.parameter_names, X)}
            X_df = pd.DataFrame([X_dict])

            weighted_sum = 0
            for target, weight in target_weights.items():
                prediction = float(self.models[target].predict(X_df)[0])
                if target.startswith('eficiencia') or target == 'biomassa':
                    # Maximize these targets
                    weighted_sum -= weight * prediction
                else:
                    # Minimize other targets (like concentration)
                    weighted_sum += weight * prediction

            return weighted_sum

        # Prepara diferentes combinações de pesos
        weight_combinations = []
        pareto_results = []

        # Para 2 objetivos, gera pesos ao longo de uma linha
        if len(targets) == 2:
            alphas = np.linspace(0, 1, n_points)
            for alpha in alphas:
                weights = {
                    targets[0]: alpha,
                    targets[1]: 1 - alpha
                }
                weight_combinations.append(weights)
        else:
            # Para mais objetivos, usa amostragem aleatória do simplex
            from scipy.stats import dirichlet
            weight_samples = dirichlet.rvs([1] * len(targets), size=n_points)
            for i in range(n_points):
                weights = {target: weight for target, weight in zip(targets, weight_samples[i])}
                weight_combinations.append(weights)

        # Executa a otimização para cada combinação de pesos
        for weights in weight_combinations:
            # Define a função objetivo para esta combinação
            self.set_objective(objective_func=lambda X: weighted_objective(X), mode='minimize')

            # Otimiza
            result = self.optimize_process(method='SLSQP')

            # Avalia os valores para cada objetivo
            opt_params_dict = result['optimal_parameters']
            opt_params_df = pd.DataFrame([opt_params_dict])

            result_row = {'weights': weights}
            result_row.update(opt_params_dict)

            for target in targets:
                pred_value = float(self.models[target].predict(opt_params_df)[0])
                result_row[f'{target}_value'] = pred_value

            pareto_results.append(result_row)

        # Cria DataFrame com resultados Pareto
        pareto_df = pd.DataFrame(pareto_results)

        return pareto_df

    def create_surrogate_model(self, target, model_type='gp', n_samples=100, random_state=42):
        """
        Cria um modelo substituto (surrogate) para exploração do espaço de parâmetros

        Args:
            target: Alvo a ser modelado
            model_type: Tipo de modelo ('gp', 'rf', 'nn')
            n_samples: Número de amostras para treinar o modelo
            random_state: Semente aleatória

        Returns:
            Modelo treinado
        """
        # Verifica se o modelo para o alvo existe
        if target not in self.models:
            raise ValueError(f"Não existe modelo para o alvo: {target}")

        # Gera amostras aleatórias do espaço de parâmetros
        np.random.seed(random_state)
        param_samples = []

        for _ in range(n_samples):
            sample = {}
            for param, bounds in self.parameter_bounds.items():
                lower, upper = bounds
                sample[param] = np.random.uniform(lower, upper)
            param_samples.append(sample)

        # Cria DataFrame com as amostras
        X_samples = pd.DataFrame(param_samples)

        # Prediz valores usando o modelo original
        y_samples = self.models[target].predict(X_samples)

        # Cria e treina o modelo substituto
        if model_type == 'gp':
            # Gaussian Process
            kernel = ConstantKernel() * Matern(nu=2.5)
            model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=random_state)
        elif model_type == 'rf':
            # Random Forest
            model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=random_state)
        elif model_type == 'nn':
            # Neural Network
            model = MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=1000, random_state=random_state)
        else:
            raise ValueError(f"Tipo de modelo desconhecido: {model_type}")

        # Treina o modelo
        X_array = X_samples.values
        model.fit(X_array, y_samples)

        # Avalia a qualidade do modelo
        y_pred = model.predict(X_array)
        r2 = r2_score(y_samples, y_pred)
        rmse = np.sqrt(mean_squared_error(y_samples, y_pred))

        print(f"Modelo substituto para {target} - R²: {r2:.4f}, RMSE: {rmse:.4f}")

        # Armazena o modelo
        self.surrogate_model = {
            'model': model,
            'target': target,
            'type': model_type,
            'r2': r2,
            'rmse': rmse,
            'parameter_names': self.parameter_names
        }

        return self.surrogate_model

    def create_response_surface(self, param1, param2, resolution=20, target=None, fixed_params=None):
        """
        Cria superfície de resposta para dois parâmetros

        Args:
            param1: Nome do primeiro parâmetro
            param2: Nome do segundo parâmetro
            resolution: Resolução da grade
            target: Alvo para modelar (usa surrogate_model se None)
            fixed_params: Valores fixos para outros parâmetros

        Returns:
            Figura do plotly com superfície de resposta
        """
        if param1 not in self.parameter_bounds or param2 not in self.parameter_bounds:
            raise ValueError("Os parâmetros devem estar nos limites definidos")

        # Prepara modelo e alvo
        if target is not None and target in self.models:
            model = self.models[target]
            target_name = target
            parameter_names = self.parameter_names
        elif self.surrogate_model is not None:
            model = self.surrogate_model['model']
            target_name = self.surrogate_model['target']
            parameter_names = self.surrogate_model['parameter_names']
        else:
            raise ValueError("Forneça um alvo válido ou crie um modelo substituto primeiro")

        # Cria grade de valores para os dois parâmetros
        p1_bounds = self.parameter_bounds[param1]
        p2_bounds = self.parameter_bounds[param2]

        p1_values = np.linspace(p1_bounds[0], p1_bounds[1], resolution)
        p2_values = np.linspace(p2_bounds[0], p2_bounds[1], resolution)

        P1, P2 = np.meshgrid(p1_values, p2_values)

        # Prepara valores para outros parâmetros
        if fixed_params is None:
            fixed_params = {}

        for param in parameter_names:
            if param not in [param1, param2] and param not in fixed_params:
                # Define valor médio para parâmetros não especificados
                bounds = self.parameter_bounds[param]
                fixed_params[param] = (bounds[0] + bounds[1]) / 2

        # Cria matriz de valores de resposta
        Z = np.zeros_like(P1)

        for i in range(resolution):
            for j in range(resolution):
                # Cria vetor de entrada para o modelo
                X = {}
                X[param1] = P1[i, j]
                X[param2] = P2[i, j]

                # Adiciona parâmetros fixos
                for param, value in fixed_params.items():
                    X[param] = value

                # Prepara o formato correto para o modelo
                if type(model).__name__ == 'GaussianProcessRegressor' or \
                        type(model).__name__ == 'MLPRegressor' or \
                        type(model).__name__ == 'RandomForestRegressor':
                    # Converte dicionário para array na ordem correta
                    X_array = np.array([X.get(param, 0) for param in parameter_names]).reshape(1, -1)
                    Z[i, j] = model.predict(X_array)[0]
                else:
                    # Assume que o modelo espera um DataFrame
                    X_df = pd.DataFrame([X])
                    Z[i, j] = model.predict(X_df)[0]

        # Cria figura da superfície de resposta
        fig = go.Figure(data=[
            go.Surface(
                x=P1, y=P2, z=Z,
                colorscale='Viridis',
                colorbar=dict(title=target_name)
            )
        ])

        # Atualiza layout
        fig.update_layout(
            title=f'Superfície de Resposta: {target_name}',
            scene=dict(
                xaxis_title=param1,
                yaxis_title=param2,
                zaxis_title=target_name
            ),
            width=800,
            height=800
        )

        return fig

    def create_contour_plot(self, param1, param2, resolution=50, target=None, fixed_params=None):
        """
        Cria gráfico de contorno para dois parâmetros

        Args:
            param1: Nome do primeiro parâmetro
            param2: Nome do segundo parâmetro
            resolution: Resolução da grade
            target: Alvo para modelar (usa surrogate_model se None)
            fixed_params: Valores fixos para outros parâmetros

        Returns:
            Figura do plotly com gráfico de contorno
        """
        if param1 not in self.parameter_bounds or param2 not in self.parameter_bounds:
            raise ValueError("Os parâmetros devem estar nos limites definidos")

        # Prepara modelo e alvo
        if target is not None and target in self.models:
            model = self.models[target]
            target_name = target
            parameter_names = self.parameter_names
        elif self.surrogate_model is not None:
            model = self.surrogate_model['model']
            target_name = self.surrogate_model['target']
            parameter_names = self.surrogate_model['parameter_names']
        else:
            raise ValueError("Forneça um alvo válido ou crie um modelo substituto primeiro")

        # Cria grade de valores para os dois parâmetros
        p1_bounds = self.parameter_bounds[param1]
        p2_bounds = self.parameter_bounds[param2]

        p1_values = np.linspace(p1_bounds[0], p1_bounds[1], resolution)
        p2_values = np.linspace(p2_bounds[0], p2_bounds[1], resolution)

        P1, P2 = np.meshgrid(p1_values, p2_values)

        # Prepara valores para outros parâmetros
        if fixed_params is None:
            fixed_params = {}

        for param in parameter_names:
            if param not in [param1, param2] and param not in fixed_params:
                # Define valor médio para parâmetros não especificados
                bounds = self.parameter_bounds[param]
                fixed_params[param] = (bounds[0] + bounds[1]) / 2

        # Cria matriz de valores de resposta
        Z = np.zeros_like(P1)

        for i in range(resolution):
            for j in range(resolution):
                # Cria vetor de entrada para o modelo
                X = {}
                X[param1] = P1[i, j]
                X[param2] = P2[i, j]

                # Adiciona parâmetros fixos
                for param, value in fixed_params.items():
                    X[param] = value

                # Prepara o formato correto para o modelo
                if type(model).__name__ == 'GaussianProcessRegressor' or \
                        type(model).__name__ == 'MLPRegressor' or \
                        type(model).__name__ == 'RandomForestRegressor':
                    # Converte dicionário para array na ordem correta
                    X_array = np.array([X.get(param, 0) for param in parameter_names]).reshape(1, -1)
                    Z[i, j] = model.predict(X_array)[0]
                else:
                    # Assume que o modelo espera um DataFrame
                    X_df = pd.DataFrame([X])
                    Z[i, j] = model.predict(X_df)[0]

        # Adiciona resultados otimizados se disponíveis
        opt_points = []

        for method, result in self.optimization_results.items():
            if result['success']:
                opt_params = result['optimal_parameters']
                if param1 in opt_params and param2 in opt_params:
                    opt_points.append({
                        'x': opt_params[param1],
                        'y': opt_params[param2],
                        'method': method,
                        'value': result['optimal_value']
                    })

        # Cria figura do contorno
        fig = go.Figure()

        # Adiciona contorno
        fig.add_trace(
            go.Contour(
                x=p1_values,
                y=p2_values,
                z=Z,
                colorscale='Viridis',
                colorbar=dict(title=target_name),
                contours=dict(
                    showlabels=True,
                    labelfont=dict(size=12, color='white')
                )
            )
        )

        # Adiciona pontos ótimos
        for point in opt_points:
            fig.add_trace(
                go.Scatter(
                    x=[point['x']],
                    y=[point['y']],
                    mode='markers+text',
                    marker=dict(
                        symbol='star',
                        size=15,
                        color='red',
                        line=dict(width=2, color='DarkSlateGrey')
                    ),
                    name=f"{point['method']} (valor: {point['value']:.4f})",
                    text=[f"{point['method']}"],
                    textposition="top right"
                )
            )

        # Atualiza layout
        fig.update_layout(
            title=f'Gráfico de Contorno: {target_name}',
            xaxis_title=param1,
            yaxis_title=param2,
            width=800,
            height=700
        )

        return fig

    def design_of_experiments(self, design_type='central_composite', center_points=1):
        """
        Gera um plano experimental para exploração sistemática do espaço de parâmetros

        Args:
            design_type: Tipo de planejamento ('full_factorial', 'fractional_factorial',
                         'central_composite', 'box_behnken', 'latin_hypercube')
            center_points: Número de pontos centrais

        Returns:
            DataFrame com pontos experimentais
        """
        try:
            import pyDOE2 as pyDOE
        except ImportError:
            import pyDOE

        # Número de fatores (parâmetros)
        n_factors = len(self.parameter_bounds)

        if n_factors == 0:
            raise ValueError("Defina os limites dos parâmetros antes de gerar o planejamento")

        # Gera o planejamento experimental no espaço normalizado [-1, 1]
        if design_type == 'full_factorial':
            # Planejamento fatorial completo (2^n)
            doe_matrix = pyDOE.ff2n(n_factors)
        elif design_type == 'fractional_factorial':
            # Planejamento fatorial fracionado (escolhe automaticamente resolução máxima)
            if n_factors <= 4:
                resolution = 'full'
            else:
                resolution = 'max'
            doe_matrix = pyDOE.fracfact(pyDOE.fracfact_by_res(n_factors, resolution))
        elif design_type == 'central_composite':
            # Planejamento composto central
            doe_matrix = pyDOE.ccdesign(n_factors, center=center_points)
        elif design_type == 'box_behnken':
            # Planejamento Box-Behnken
            if n_factors < 3:
                raise ValueError("Planejamento Box-Behnken requer pelo menos 3 fatores")
            doe_matrix = pyDOE.bbdesign(n_factors, center=center_points)
        elif design_type == 'latin_hypercube':
            # Amostragem por hipercubo latino
            samples = max(10, 2 ** n_factors)
            doe_matrix = pyDOE.lhs(n_factors, samples=samples)
            # Converte para [-1, 1]
            doe_matrix = doe_matrix * 2 - 1
        else:
            raise ValueError(f"Tipo de planejamento desconhecido: {design_type}")

        # Converte de [-1, 1] para os limites reais
        real_matrix = np.zeros_like(doe_matrix)

        for i, param in enumerate(self.parameter_names):
            lower, upper = self.parameter_bounds[param]
            # Mapeia de [-1, 1] para [lower, upper]
            real_matrix[:, i] = lower + (doe_matrix[:, i] + 1) * (upper - lower) / 2

        # Cria DataFrame com o planejamento
        doe_df = pd.DataFrame(real_matrix, columns=self.parameter_names)

        # Adiciona informações sobre o planejamento
        doe_df['design_type'] = design_type

        # Armazena o planejamento
        self.experimental_design = {
            'design_type': design_type,
            'dataframe': doe_df,
            'normalized_matrix': doe_matrix
        }

        return doe_df

    def evaluate_doe_predictions(self, target):
        """
        Avalia as previsões do modelo para o planejamento experimental

        Args:
            target: Alvo a ser avaliado

        Returns:
            DataFrame com previsões
        """
        if self.experimental_design is None:
            raise ValueError("Gere um planejamento experimental primeiro")

        if target not in self.models:
            raise ValueError(f"Não existe modelo para o alvo: {target}")

        # Obtém o planejamento
        doe_df = self.experimental_design['dataframe']

        # Faz previsões com o modelo
        predictions = self.models[target].predict(doe_df)

        # Adiciona as previsões ao DataFrame
        result_df = doe_df.copy()
        result_df[target] = predictions

        # Ordena por valor de resposta
        result_df = result_df.sort_values(by=target, ascending=False)

        return result_df

    def plot_doe_results(self, target_results, main_effects=True, interaction_effects=True):
        """
        Plota os resultados do planejamento experimental

        Args:
            target_results: DataFrame com resultados de evaluate_doe_predictions
            main_effects: Plotar efeitos principais
            interaction_effects: Plotar efeitos de interação

        Returns:
            Figura do plotly ou dicionário de figuras
        """
        if self.experimental_design is None:
            raise ValueError("Gere um planejamento experimental primeiro")

        # Target name (assume último campo da coluna que não é um parâmetro ou design_type)
        target = None
        for col in target_results.columns:
            if col not in self.parameter_names and col != 'design_type':
                target = col
                break

        if target is None:
            raise ValueError("Não foi possível identificar o alvo nos resultados")

        figures = {}

        # Efeitos principais
        if main_effects:
            # Cria figura para efeitos principais
            main_effects_fig = make_subplots(
                rows=len(self.parameter_names),
                cols=1,
                subplot_titles=[f"Efeito de {param}" for param in self.parameter_names],
                shared_xaxes=False,
                vertical_spacing=0.08
            )

            for i, param in enumerate(self.parameter_names):
                # Agrupa resultados por valor do parâmetro
                # Usando bins para parâmetros contínuos
                param_values = target_results[param].values

                if len(np.unique(param_values)) > 10:
                    # Parâmetro contínuo, usa bins
                    bins = np.linspace(min(param_values), max(param_values), 10)
                    target_results['bin'] = pd.cut(target_results[param], bins)

                    # Calcula média e erro padrão por bin
                    grouped = target_results.groupby('bin')
                    means = grouped[target].mean()
                    sem = grouped[target].sem()

                    # Converte bins para valores médios para plotagem
                    bin_centers = [(interval.left + interval.right) / 2 for interval in means.index]

                    # Adiciona trace
                    main_effects_fig.add_trace(
                        go.Scatter(
                            x=bin_centers,
                            y=means,
                            mode='lines+markers',
                            name=param,
                            line=dict(width=2),
                            error_y=dict(
                                type='data',
                                array=sem.values,
                                visible=True
                            )
                        ),
                        row=i + 1, col=1
                    )

                    # Remove a coluna temporária
                    target_results = target_results.drop('bin', axis=1)

                else:
                    # Parâmetro discreto, usa valores diretos
                    grouped = target_results.groupby(param)
                    means = grouped[target].mean()
                    sem = grouped[target].sem()

                    # Adiciona trace
                    main_effects_fig.add_trace(
                        go.Scatter(
                            x=means.index,
                            y=means.values,
                            mode='lines+markers',
                            name=param,
                            line=dict(width=2),
                            error_y=dict(
                                type='data',
                                array=sem.values,
                                visible=True
                            )
                        ),
                        row=i + 1, col=1
                    )

                # Atualiza eixos
                main_effects_fig.update_xaxes(title_text=param, row=i + 1, col=1)
                if i == 0:
                    main_effects_fig.update_yaxes(title_text=target, row=i + 1, col=1)

            # Atualiza layout
            main_effects_fig.update_layout(
                title_text="Análise de Efeitos Principais",
                height=300 * len(self.parameter_names),
                width=800,
                showlegend=False
            )

            figures['main_effects'] = main_effects_fig

        # Efeitos de interação (para 2 parâmetros)
        if interaction_effects and len(self.parameter_names) >= 2:
            # Cria figura de matriz de interação
            n_params = len(self.parameter_names)
            interaction_fig = make_subplots(
                rows=n_params - 1,
                cols=n_params - 1,
                subplot_titles=[f"{p1} x {p2}" for p1 in self.parameter_names[:-1] for p2 in self.parameter_names[1:] if
                                p1 != p2],
                shared_xaxes=False,
                shared_yaxes=False,
                horizontal_spacing=0.05,
                vertical_spacing=0.05
            )

            # Contador para subplots
            plot_idx = 0

            for i, param1 in enumerate(self.parameter_names[:-1]):
                for j, param2 in enumerate(self.parameter_names[1:], 1):
                    if param1 != param2:
                        # Cria contorno de interação
                        # Discretiza em bins
                        p1_values = target_results[param1].values
                        p2_values = target_results[param2].values

                        n_bins = min(10, len(np.unique(p1_values)), len(np.unique(p2_values)))

                        p1_bins = pd.cut(p1_values, n_bins)
                        p2_bins = pd.cut(p2_values, n_bins)

                        # Adiciona colunas de bins
                        target_results['p1_bin'] = p1_bins
                        target_results['p2_bin'] = p2_bins

                        # Agrupa por bins
                        grouped = target_results.groupby(['p1_bin', 'p2_bin'])[target].mean().reset_index()

                        # Converte bins para arrays para contorno
                        p1_centers = [(interval.left + interval.right) / 2 for interval in grouped['p1_bin'].unique()]
                        p2_centers = [(interval.left + interval.right) / 2 for interval in grouped['p2_bin'].unique()]

                        # Cria matriz Z para contorno
                        Z = np.zeros((len(p1_centers), len(p2_centers)))

                        for idx, row in grouped.iterrows():
                            p1_idx = list(grouped['p1_bin'].unique()).index(row['p1_bin'])
                            p2_idx = list(grouped['p2_bin'].unique()).index(row['p2_bin'])
                            Z[p1_idx, p2_idx] = row[target]

                        # Adiciona contorno
                        interaction_fig.add_trace(
                            go.Contour(
                                z=Z,
                                x=p2_centers,  # Note a inversão de x e y
                                y=p1_centers,
                                colorscale='Viridis',
                                showscale=False
                            ),
                            row=i + 1, col=j
                        )

                        # Remove colunas temporárias
                        target_results = target_results.drop(['p1_bin', 'p2_bin'], axis=1)

                        # Atualiza eixos
                        interaction_fig.update_xaxes(title_text=param2, row=i + 1, col=j)
                        interaction_fig.update_yaxes(title_text=param1, row=i + 1, col=j)

                        plot_idx += 1

            # Atualiza layout
            interaction_fig.update_layout(
                title_text="Análise de Interações",
                height=300 * (n_params - 1),
                width=800 * (n_params - 1),
                showlegend=False
            )

            figures['interaction_effects'] = interaction_fig

        # Pareto de efeitos
        # Simplificação: usa magnitude da média dos valores extremos como estimativa de efeito
        effects = []

        for param in self.parameter_names:
            # Encontra valores extremos do parâmetro
            param_min = target_results[param].min()
            param_max = target_results[param].max()

            # Filtra resultados para valores próximos aos extremos
            min_data = target_results[target_results[param] <= param_min * 1.1]
            max_data = target_results[target_results[param] >= param_max * 0.9]

            # Calcula médias
            mean_at_min = min_data[target].mean()
            mean_at_max = max_data[target].mean()

            # Calcula efeito
            effect = abs(mean_at_max - mean_at_min)
            effects.append({
                'parameter': param,
                'effect': effect
            })

        # Cria DataFrame e ordena
        effects_df = pd.DataFrame(effects)
        effects_df = effects_df.sort_values(by='effect', ascending=False)

        # Plota Pareto
        pareto_fig = px.bar(
            effects_df,
            x='parameter',
            y='effect',
            title='Pareto de Efeitos',
            labels={'parameter': 'Parâmetro', 'effect': 'Magnitude do Efeito'},
            text='effect'
        )

        pareto_fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')

        figures['pareto'] = pareto_fig

        # Se apenas uma figura solicitada, retorna diretamente
        if len(figures) == 1:
            return list(figures.values())[0]

        return figures