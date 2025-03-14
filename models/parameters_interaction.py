"""
Análise de interação entre parâmetros para bioprocessos

Este módulo implementa métodos avançados para análise e visualização
de interações entre múltiplos parâmetros em processos de tratamento biológico.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler


class ParameterInteractionAnalyzer:
    """
    Classe para análise de interações entre parâmetros de processo
    """

    def __init__(self, data=None):
        """
        Inicializa o analisador com dados

        Args:
            data: DataFrame com os dados do processo
        """
        self.data = data
        self.feature_names = []
        self.target_name = None
        self.model = None
        self.pca_results = None
        self.interaction_scores = None

    def set_data(self, data):
        """
        Define os dados para análise

        Args:
            data: DataFrame com os dados do processo
        """
        self.data = data

    def set_features_target(self, features, target):
        """
        Define as características e alvo para análise

        Args:
            features: Lista de nomes das características
            target: Nome do alvo
        """
        self.feature_names = features
        self.target_name = target

    def fit_model(self, model_type='rf', **kwargs):
        """
        Ajusta um modelo para análise de importância e dependência parcial

        Args:
            model_type: Tipo de modelo ('rf' para Random Forest)
            **kwargs: Parâmetros específicos do modelo

        Returns:
            self: O próprio objeto
        """
        if self.data is None:
            raise ValueError("Dados não definidos. Use set_data() primeiro.")

        if not self.feature_names or not self.target_name:
            raise ValueError("Características e alvo não definidos. Use set_features_target() primeiro.")

        # Prepara dados
        X = self.data[self.feature_names].values
        y = self.data[self.target_name].values

        # Cria e ajusta o modelo
        if model_type == 'rf':
            self.model = RandomForestRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', None),
                random_state=kwargs.get('random_state', 42)
            )
        else:
            raise ValueError(f"Tipo de modelo desconhecido: {model_type}")

        self.model.fit(X, y)

        return self

    def calculate_pca(self, n_components=2, scale=True):
        """
        Realiza análise de componentes principais (PCA)

        Args:
            n_components: Número de componentes principais
            scale: Se True, padroniza os dados antes do PCA

        Returns:
            Dictionary com resultados do PCA
        """
        if self.data is None:
            raise ValueError("Dados não definidos. Use set_data() primeiro.")

        if not self.feature_names:
            raise ValueError("Características não definidas. Use set_features_target() primeiro.")

        # Prepara dados
        X = self.data[self.feature_names].values

        # Padroniza dados se solicitado
        if scale:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        # Executa PCA
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(X)

        # Cria DataFrame com componentes principais
        pca_df = pd.DataFrame(
            data=principal_components,
            columns=[f'PC{i + 1}' for i in range(n_components)]
        )

        # Adiciona alvo se definido
        if self.target_name is not None:
            pca_df[self.target_name] = self.data[self.target_name].values

        # Armazena loadings (contribuições das características)
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        loadings_df = pd.DataFrame(
            data=loadings,
            columns=[f'PC{i + 1}' for i in range(n_components)],
            index=self.feature_names
        )

        # Armazena resultados
        self.pca_results = {
            'pca_object': pca,
            'explained_variance': pca.explained_variance_ratio_,
            'pca_df': pca_df,
            'loadings': loadings_df,
            'scaled_data': X if scale else None
        }

        return self.pca_results

    def plot_pca_biplot(self):
        """
        Plota o biplot do PCA (pontos e vetores de contribuição)

        Returns:
            Figura do plotly
        """
        if self.pca_results is None:
            raise ValueError("Execute calculate_pca() primeiro.")

        # Dados para plotagem
        pca_df = self.pca_results['pca_df']
        loadings_df = self.pca_results['loadings']
        explained_variance = self.pca_results['explained_variance']

        # Cria figura base
        fig = go.Figure()

        # Adiciona pontos
        if self.target_name is not None:
            # Colorido por alvo
            fig = px.scatter(
                pca_df,
                x='PC1',
                y='PC2',
                color=self.target_name,
                color_continuous_scale='Viridis'
            )
        else:
            # Sem coloração
            fig.add_trace(
                go.Scatter(
                    x=pca_df['PC1'],
                    y=pca_df['PC2'],
                    mode='markers',
                    marker=dict(
                        color='blue',
                        size=8,
                        opacity=0.7
                    ),
                    name='Amostras'
                )
            )

        # Adiciona vetores de loadings
        for i, feature in enumerate(self.feature_names):
            fig.add_trace(
                go.Scatter(
                    x=[0, loadings_df.iloc[i, 0] * 3],  # Escala para melhor visualização
                    y=[0, loadings_df.iloc[i, 1] * 3],
                    mode='lines+text',
                    line=dict(color='red', width=1),
                    text='',
                    textposition='top center',
                    name=feature
                )
            )

            # Adiciona texto no final do vetor
            fig.add_trace(
                go.Scatter(
                    x=[loadings_df.iloc[i, 0] * 3],
                    y=[loadings_df.iloc[i, 1] * 3],
                    mode='text',
                    text=[feature],
                    textposition='top center',
                    textfont=dict(color='red'),
                    showlegend=False
                )
            )

        # Títulos e layout
        fig.update_layout(
            title='Biplot de PCA',
            xaxis_title=f'Componente Principal 1 ({explained_variance[0]:.2%} da variância)',
            yaxis_title=f'Componente Principal 2 ({explained_variance[1]:.2%} da variância)',
            legend=dict(title=self.target_name if self.target_name else 'Amostras'),
            height=700,
            width=900
        )

        # Adiciona origem e linhas de grade
        fig.add_shape(
            type='line',
            x0=-max(abs(pca_df['PC1'])) * 1.1,
            y0=0,
            x1=max(abs(pca_df['PC1'])) * 1.1,
            y1=0,
            line=dict(color='gray', width=1, dash='dash')
        )

        fig.add_shape(
            type='line',
            x0=0,
            y0=-max(abs(pca_df['PC2'])) * 1.1,
            x1=0,
            y1=max(abs(pca_df['PC2'])) * 1.1,
            line=dict(color='gray', width=1, dash='dash')
        )

        return fig

    def detect_feature_interactions(self, method='mutual_info'):
        """
        Detecta interações entre características

        Args:
            method: Método de detecção ('mutual_info', 'correlation', 'model_based')

        Returns:
            DataFrame com pontuações de interação
        """
        if self.data is None:
            raise ValueError("Dados não definidos. Use set_data() primeiro.")

        if not self.feature_names:
            raise ValueError("Características não definidas. Use set_features_target() primeiro.")

        # Prepara dados
        X = self.data[self.feature_names]

        # Detecta interações com base no método
        interaction_matrix = pd.DataFrame(
            index=self.feature_names,
            columns=self.feature_names
        )

        if method == 'mutual_info':
            # Usa informação mútua para detectar relações não lineares
            for i, feature1 in enumerate(self.feature_names):
                for j, feature2 in enumerate(self.feature_names):
                    if i == j:
                        # Valores na diagonal principal são 1
                        interaction_matrix.loc[feature1, feature2] = 1.0
                    else:
                        # Calcula informação mútua normalizada
                        try:
                            mi = mutual_info_regression(
                                X[feature1].values.reshape(-1, 1),
                                X[feature2].values
                            )[0]

                            # Normaliza para [0, 1]
                            h1 = mutual_info_regression(
                                X[feature1].values.reshape(-1, 1),
                                X[feature1].values
                            )[0]

                            h2 = mutual_info_regression(
                                X[feature2].values.reshape(-1, 1),
                                X[feature2].values
                            )[0]

                            if h1 * h2 > 0:
                                mi_normalized = mi / np.sqrt(h1 * h2)
                            else:
                                mi_normalized = mi

                            interaction_matrix.loc[feature1, feature2] = mi_normalized
                        except:
                            interaction_matrix.loc[feature1, feature2] = np.nan

        elif method == 'correlation':
            # Usa correlação (linear ou rank)
            for i, feature1 in enumerate(self.feature_names):
                for j, feature2 in enumerate(self.feature_names):
                    if i == j:
                        # Valores na diagonal principal são 1
                        interaction_matrix.loc[feature1, feature2] = 1.0
                    else:
                        # Calcula correlação de Spearman (rank)
                        try:
                            corr, _ = spearmanr(X[feature1], X[feature2])
                            interaction_matrix.loc[feature1, feature2] = abs(corr)
                        except:
                            interaction_matrix.loc[feature1, feature2] = np.nan

        elif method == 'model_based':
            # Usa importância de permutação em subconjuntos se tivermos um modelo
            if self.model is None:
                raise ValueError("Modelo não ajustado. Use fit_model() primeiro.")

            if not hasattr(self.model, 'feature_importances_'):
                raise ValueError("O modelo não suporta feature_importances_")

            # Calcula importância de características individuais
            importances = self.model.feature_importances_
            base_importances = {feature: imp for feature, imp in zip(self.feature_names, importances)}

            # Calcula importância para pares de características
            for i, feature1 in enumerate(self.feature_names):
                interaction_matrix.loc[feature1, feature1] = 1.0

                for j, feature2 in enumerate(self.feature_names):
                    if i < j:  # Evita cálculos duplicados
                        # Cria novo modelo sem estas características
                        other_features = [f for f in self.feature_names if f not in [feature1, feature2]]

                        if not other_features:
                            # Se não houver outras características, não é possível estimar a interação
                            interaction_matrix.loc[feature1, feature2] = np.nan
                            interaction_matrix.loc[feature2, feature1] = np.nan
                            continue

                        # Prepara dados sem estas características
                        X_subset = self.data[other_features].values
                        y = self.data[self.target_name].values

                        # Cria e ajusta o modelo
                        subset_model = RandomForestRegressor(
                            n_estimators=100,
                            random_state=42
                        )

                        subset_model.fit(X_subset, y)

                        # Calcula perda de importância
                        subset_r2 = subset_model.score(X_subset, y)
                        full_r2 = self.model.score(self.data[self.feature_names].values, y)

                        # Diferença no R² é uma medida da importância conjunta
                        importance_diff = full_r2 - subset_r2

                        # Calcula excesso de importância além das importâncias individuais
                        individual_importance = base_importances[feature1] + base_importances[feature2]

                        # Interação é quanto a importância conjunta excede a soma das importâncias
                        # Normalizada para [0, 1]
                        if individual_importance > 0:
                            interaction_score = max(0, importance_diff - individual_importance) / individual_importance
                        else:
                            interaction_score = 0

                        interaction_matrix.loc[feature1, feature2] = interaction_score
                        interaction_matrix.loc[feature2, feature1] = interaction_score

        else:
            raise ValueError(f"Método desconhecido: {method}")

        # Armazena as interações
        self.interaction_scores = interaction_matrix

        return interaction_matrix

    def plot_interaction_heatmap(self):
        """
        Plota mapa de calor das interações entre características

        Returns:
            Figura do plotly
        """
        if self.interaction_scores is None:
            raise ValueError("Execute detect_feature_interactions() primeiro.")

        # Cria figura de heatmap
        fig = px.imshow(
            self.interaction_scores,
            text_auto=True,
            color_continuous_scale='Viridis',
            title='Interações entre Parâmetros',
            zmin=0,
            zmax=1
        )

        fig.update_layout(
            xaxis_title='Parâmetro',
            yaxis_title='Parâmetro',
            height=600,
            width=700
        )

        return fig

    def calculate_partial_dependence(self, features=None, kind='average'):
        """
        Calcula dependência parcial para características

        Args:
            features: Lista de características para análise (default: todas)
            kind: Tipo de dependência ('average', 'individual', 'both')

        Returns:
            Dictionary com curvas de dependência parcial
        """
        if self.model is None:
            raise ValueError("Modelo não ajustado. Use fit_model() primeiro.")

        if not self.feature_names:
            raise ValueError("Características não definidas. Use set_features_target() primeiro.")

        # Define características para análise
        if features is None:
            features = self.feature_names

        # Limita o número de características para não sobrecarregar
        if len(features) > 8:
            features = features[:8]
            print(f"Aviso: Limitando a análise para as primeiras 8 características")

        # Verifica se há características válidas
        if not features:
            import warnings
            warnings.warn("Nenhuma característica selecionada para análise")
            return {}  # Retorna dicionário vazio

        # Prepara dados
        X = self.data[self.feature_names].values

        # Mapeia nomes para índices
        feature_indices = []
        selected_features = []
        for feature in features:
            if feature in self.feature_names:
                feature_indices.append(self.feature_names.index(feature))
                selected_features.append(feature)
            else:
                print(f"Aviso: Característica '{feature}' não encontrada e será ignorada")

        # Verifica se há índices válidos
        if not feature_indices:
            import warnings
            warnings.warn("Nenhuma característica válida encontrada para análise")
            return {}  # Retorna dicionário vazio

        # Calcula dependência parcial
        pdp_results = {}

        for idx, feature in zip(feature_indices, selected_features):
            try:
                # Implementação manual de dependência parcial
                # Esta abordagem funciona para qualquer modelo com método predict

                # Cria uma grade de valores para a característica
                feature_values = self.data[feature].values
                min_val = feature_values.min()
                max_val = feature_values.max()

                # Cria uma sequência de valores dentro do range
                grid_points = 20
                grid = np.linspace(min_val, max_val, grid_points)

                # Prepara array para armazenar resultados
                pdp = np.zeros(grid_points)

                # Para cada valor na grade
                for i, val in enumerate(grid):
                    # Cria cópias dos dados
                    X_copies = np.repeat(X, 1, axis=0)

                    # Substitui o valor da característica em análise
                    X_copies[:, idx] = val

                    # Faz predição
                    y_pred = self.model.predict(X_copies)

                    # Média das predições
                    pdp[i] = np.mean(y_pred)

                # Armazena resultados
                pdp_results[feature] = {
                    'values': grid,
                    'pdp': pdp
                }

            except Exception as e:
                print(f"Erro ao calcular dependência parcial para {feature}: {str(e)}")

        return pdp_results

    def plot_partial_dependence(self, pdp_results=None, n_cols=2):
        """
        Plota curvas de dependência parcial

        Args:
            pdp_results: Resultados de calculate_partial_dependence
            n_cols: Número de colunas no grid

        Returns:
            Figura do plotly
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        if pdp_results is None:
            raise ValueError("Forneça resultados de calculate_partial_dependence()")

        # Check if pdp_results is empty
        if not pdp_results:
            # Return a simple figure with a message
            fig = go.Figure()
            fig.add_annotation(
                text="Nenhum resultado disponível para plotar",
                showarrow=False,
                font=dict(size=15)
            )
            fig.update_layout(
                title="Dependência Parcial",
                height=400,
                width=700
            )
            return fig

        # Número de características
        n_features = len(pdp_results)

        # Ensure n_rows is at least 1
        n_rows = max(1, (n_features + n_cols - 1) // n_cols)

        # Ensure n_cols doesn't exceed number of features
        n_cols = min(n_cols, n_features)

        # Cria subplots
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=list(pdp_results.keys())
        )

        # Contador para subplots
        feature_idx = 0

        for feature, result in pdp_results.items():
            # Calcula linha e coluna
            row = feature_idx // n_cols + 1
            col = feature_idx % n_cols + 1

            # Adiciona curva de dependência parcial
            fig.add_trace(
                go.Scatter(
                    x=result['values'],
                    y=result['pdp'],
                    mode='lines+markers',
                    name=feature,
                    line=dict(width=2),
                    marker=dict(size=6)
                ),
                row=row,
                col=col
            )

            # Atualiza eixos
            fig.update_xaxes(title_text=feature, row=row, col=col)

            if col == 1:
                fig.update_yaxes(title_text='Efeito Parcial', row=row, col=col)

            feature_idx += 1

        # Atualiza layout
        fig.update_layout(
            title_text="Análise de Dependência Parcial",
            height=300 * n_rows,
            width=500 * n_cols,
            showlegend=False
        )

        return fig

    def calculate_2d_partial_dependence(self, feature_pairs=None, grid_resolution=20):
        """
        Calcula dependência parcial 2D para pares de características

        Args:
            feature_pairs: Lista de pares de características
            grid_resolution: Resolução da grade

        Returns:
            Dictionary com superfícies de dependência parcial
        """
        if self.model is None:
            raise ValueError("Modelo não ajustado. Use fit_model() primeiro.")

        if not self.feature_names:
            raise ValueError("Características não definidas. Use set_features_target() primeiro.")

        # Define pares de características
        if feature_pairs is None:
            # Se houver interações, usa os pares com maior interação
            if self.interaction_scores is not None:
                # Ordenar pares por pontuação de interação
                pairs = []
                for i, feature1 in enumerate(self.feature_names):
                    for j, feature2 in enumerate(self.feature_names):
                        if i < j:  # Evita duplicações
                            score = self.interaction_scores.loc[feature1, feature2]
                            pairs.append((feature1, feature2, score))

                # Ordena por pontuação decrescente
                pairs.sort(key=lambda x: x[2], reverse=True)

                # Pega os 3 primeiros pares ou menos
                feature_pairs = [(pair[0], pair[1]) for pair in pairs[:min(3, len(pairs))]]
            else:
                # Sem informações de interação, usa primeiros 3 pares
                feature_pairs = []
                for i, feature1 in enumerate(self.feature_names):
                    for j, feature2 in enumerate(self.feature_names):
                        if i < j:  # Evita duplicações
                            feature_pairs.append((feature1, feature2))
                            if len(feature_pairs) >= 3:
                                break
                    if len(feature_pairs) >= 3:
                        break

        # Prepara dados
        X = self.data[self.feature_names].values

        # Calcula dependência parcial 2D
        pdp_2d_results = {}

        for feature1, feature2 in feature_pairs:
            try:
                # Mapeia nomes para índices
                idx1 = self.feature_names.index(feature1)
                idx2 = self.feature_names.index(feature2)

                # Calcula grade e valores de dependência parcial 2D
                pdp_avg = partial_dependence(
                    self.model,
                    X,
                    [(idx1, idx2)],
                    kind='average',
                    grid_resolution=grid_resolution
                )

                # Armazena resultados
                pdp_2d_results[(feature1, feature2)] = {
                    'values1': pdp_avg['values'][0][0],
                    'values2': pdp_avg['values'][0][1],
                    'pdp': pdp_avg['average'][0]
                }
            except Exception as e:
                print(f"Erro ao calcular dependência parcial 2D para {feature1}, {feature2}: {str(e)}")

        return pdp_2d_results

    def plot_2d_partial_dependence(self, pdp_2d_results=None):
        """
        Plota superfícies de dependência parcial 2D

        Args:
            pdp_2d_results: Resultados de calculate_2d_partial_dependence

        Returns:
            Dictionary com figuras do plotly
        """
        if pdp_2d_results is None:
            raise ValueError("Forneça resultados de calculate_2d_partial_dependence()")

        # Cria uma figura para cada par
        figures = {}

        for (feature1, feature2), result in pdp_2d_results.items():
            # Cria um grid 2D para plotagem
            X, Y = np.meshgrid(result['values1'], result['values2'])
            Z = result['pdp'].reshape(len(result['values2']), len(result['values1'])).T

            # Cria figura com contorno e superfície
            contour_fig = go.Figure(data=[
                go.Contour(
                    z=Z,
                    x=result['values1'],
                    y=result['values2'],
                    colorscale='Viridis',
                    colorbar=dict(title='Efeito Parcial')
                )
            ])

            # Atualiza layout
            contour_fig.update_layout(
                title=f'Dependência Parcial 2D: {feature1} x {feature2}',
                xaxis_title=feature1,
                yaxis_title=feature2,
                height=600,
                width=700
            )

            # Superfície 3D
            surface_fig = go.Figure(data=[
                go.Surface(
                    z=Z,
                    x=result['values1'],
                    y=result['values2'],
                    colorscale='Viridis',
                    colorbar=dict(title='Efeito Parcial')
                )
            ])

            # Atualiza layout
            surface_fig.update_layout(
                title=f'Dependência Parcial 3D: {feature1} x {feature2}',
                scene=dict(
                    xaxis_title=feature1,
                    yaxis_title=feature2,
                    zaxis_title='Efeito Parcial'
                ),
                height=700,
                width=700
            )

            # Armazena figuras
            figures[(feature1, feature2)] = {
                'contour': contour_fig,
                'surface': surface_fig
            }

        return figures

    def create_parallel_coordinates(self, highlight_target=True):
        """
        Cria visualização de coordenadas paralelas

        Args:
            highlight_target: Colorir por valor do alvo

        Returns:
            Figura do plotly
        """
        if self.data is None:
            raise ValueError("Dados não definidos. Use set_data() primeiro.")

        if not self.feature_names:
            raise ValueError("Características não definidas. Use set_features_target() primeiro.")

        # Prepara dados
        plot_data = self.data[self.feature_names].copy()

        # Adiciona alvo se definido e solicitado
        if self.target_name is not None and highlight_target:
            plot_data[self.target_name] = self.data[self.target_name]
            color_col = self.target_name
        else:
            color_col = None

        # Cria visualização de coordenadas paralelas
        if color_col:
            fig = px.parallel_coordinates(
                plot_data,
                color=color_col,
                color_continuous_scale='Viridis',
                title='Visualização de Coordenadas Paralelas'
            )
        else:
            fig = px.parallel_coordinates(
                plot_data,
                title='Visualização de Coordenadas Paralelas'
            )

        # Atualiza layout
        fig.update_layout(
            height=600,
            width=900
        )

        return fig

    def create_high_dimensional_scatter(self, dimensions=None, n_dims=4):
        """
        Cria gráfico de dispersão de alta dimensão (splom ou coords. paralelas)

        Args:
            dimensions: Lista de características a incluir
            n_dims: Número máximo de dimensões

        Returns:
            Figura do plotly
        """
        if self.data is None:
            raise ValueError("Dados não definidos. Use set_data() primeiro.")

        if not self.feature_names:
            raise ValueError("Características não definidas. Use set_features_target() primeiro.")

        # Define dimensões para plotagem
        if dimensions is None:
            # Se tiver interações, usa características com maiores interações
            if self.interaction_scores is not None:
                # Soma das interações para cada característica
                interaction_sums = self.interaction_scores.sum().sort_values(ascending=False)
                dimensions = list(interaction_sums.index)[:n_dims]
            else:
                # Sem interações, usa primeiras características
                dimensions = self.feature_names[:n_dims]

        # Limita o número de dimensões
        if len(dimensions) > n_dims:
            dimensions = dimensions[:n_dims]
            print(f"Aviso: Limitando visualização para {n_dims} dimensões")

        # Prepara dados
        plot_data = self.data[dimensions].copy()

        # Adiciona alvo se definido
        if self.target_name is not None:
            plot_data[self.target_name] = self.data[self.target_name]
            color_col = self.target_name
        else:
            color_col = None

        # Cria matriz de gráficos de dispersão
        if color_col:
            fig = px.scatter_matrix(
                plot_data,
                dimensions=dimensions,
                color=color_col,
                color_continuous_scale='Viridis',
                title='Matriz de Gráficos de Dispersão',
                opacity=0.7
            )
        else:
            fig = px.scatter_matrix(
                plot_data,
                dimensions=dimensions,
                title='Matriz de Gráficos de Dispersão',
                opacity=0.7
            )

        # Atualiza layout
        fig.update_layout(
            height=800,
            width=800
        )

        return fig

    def create_andrews_curves(self):
        """
        Cria curvas de Andrews para visualização de alta dimensão

        Returns:
            Figura do plotly
        """
        if self.data is None:
            raise ValueError("Dados não definidos. Use set_data() primeiro.")

        if not self.feature_names:
            raise ValueError("Características não definidas. Use set_features_target() primeiro.")

        # Prepara dados
        X = self.data[self.feature_names].values

        # Padroniza dados
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Calcula curvas de Andrews
        def andrews_curve(coeffs, t):
            """Calcula curva de Andrews para um conjunto de coeficientes"""
            a = coeffs[0] / np.sqrt(2)

            andrews = a * np.ones_like(t)
            for i in range(1, len(coeffs)):
                if i % 2 == 1:
                    andrews += coeffs[i] * np.sin(t * (i + 1) / 2)
                else:
                    andrews += coeffs[i] * np.cos(t * i / 2)

            return andrews

        # Gera pontos para plotagem
        t = np.linspace(-np.pi, np.pi, 1000)

        # Cria figura
        fig = go.Figure()

        # Adiciona curva para cada amostra
        if self.target_name is not None:
            # Colorir por alvo
            # Discretiza o alvo se contínuo
            target_values = self.data[self.target_name].values

            if len(np.unique(target_values)) > 10:
                # Target contínuo, usa quantis para discretizar
                n_groups = 5
                target_bins = pd.qcut(target_values, n_groups, labels=False)

                # Mapa de cores
                colormap = px.colors.qualitative.Set1[:n_groups]

                # Agrupa amostras
                for i in range(n_groups):
                    mask = target_bins == i
                    group_X = X_scaled[mask]
                    group_target = target_values[mask]

                    # Nome do grupo
                    label = f"{self.target_name}: {np.mean(group_target):.2f}"

                    # Adiciona curvas para este grupo
                    for sample in group_X:
                        curve = andrews_curve(sample, t)

                        fig.add_trace(
                            go.Scatter(
                                x=t,
                                y=curve,
                                mode='lines',
                                opacity=0.3,
                                line=dict(color=colormap[i], width=1),
                                showlegend=False,
                                hoverinfo='skip'
                            )
                        )

                    # Adiciona média do grupo como referência
                    mean_curve = andrews_curve(np.mean(group_X, axis=0), t)

                    fig.add_trace(
                        go.Scatter(
                            x=t,
                            y=mean_curve,
                            mode='lines',
                            line=dict(color=colormap[i], width=3),
                            name=label
                        )
                    )
            else:
                # Alvo discreto
                unique_targets = np.unique(target_values)

                # Usa mapa de cores
                colormap = px.colors.qualitative.Set1[:len(unique_targets)]

                # Agrupa amostras por valor do alvo
                for i, target_val in enumerate(unique_targets):
                    mask = target_values == target_val
                    group_X = X_scaled[mask]

                    # Nome do grupo
                    label = f"{self.target_name}: {target_val}"

                    # Adiciona curvas para este grupo
                    for sample in group_X:
                        curve = andrews_curve(sample, t)

                        fig.add_trace(
                            go.Scatter(
                                x=t,
                                y=curve,
                                mode='lines',
                                opacity=0.3,
                                line=dict(color=colormap[i], width=1),
                                showlegend=False,
                                hoverinfo='skip'
                            )
                        )

                    # Adiciona média do grupo como referência
                    mean_curve = andrews_curve(np.mean(group_X, axis=0), t)

                    fig.add_trace(
                        go.Scatter(
                            x=t,
                            y=mean_curve,
                            mode='lines',
                            line=dict(color=colormap[i], width=3),
                            name=label
                        )
                    )
        else:
            # Sem coloração por alvo
            for i, sample in enumerate(X_scaled):
                curve = andrews_curve(sample, t)

                fig.add_trace(
                    go.Scatter(
                        x=t,
                        y=curve,
                        mode='lines',
                        opacity=0.3,
                        line=dict(width=1),
                        showlegend=False,
                        hoverinfo='skip'
                    )
                )

            # Adiciona média como referência
            mean_curve = andrews_curve(np.mean(X_scaled, axis=0), t)

            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=mean_curve,
                    mode='lines',
                    line=dict(color='red', width=3),
                    name='Média'
                )
            )

        # Atualiza layout
        fig.update_layout(
            title='Curvas de Andrews',
            xaxis_title='t',
            yaxis_title='f(t)',
            height=600,
            width=900
        )

        return fig

    def streamlit_interface(self):
        """
        Renderiza interface do Streamlit para análise de interação

        Esta função deve ser chamada dentro de uma aplicação Streamlit
        """
        import streamlit as st

        st.title("Análise de Interação entre Parâmetros")

        # Verifica se há dados disponíveis
        if self.data is None:
            st.warning("Nenhum dado disponível. Por favor, carregue dados primeiro.")
            return

        # Interface para seleção de características e alvo
        st.header("Configuração de Dados")

        # Seleção de características
        all_columns = list(self.data.columns)

        # Seleção de alvo
        target_col = st.selectbox(
            "Selecione o Alvo",
            ["Nenhum"] + all_columns,
            index=min(6, len(all_columns)) if len(all_columns) > 5 else 0,
            help="Variável a ser prevista ou analisada"
        )

        if target_col == "Nenhum":
            target_col = None

        # Define características (excluindo o alvo)
        feature_options = [col for col in all_columns if col != target_col]

        # Usa todas as características numéricas por padrão
        numeric_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        default_features = [col for col in numeric_cols if col != target_col]

        # Limita o número padrão de características
        if len(default_features) > 8:
            default_features = default_features[:8]

        # Seleção de características
        selected_features = st.multiselect(
            "Selecione os Parâmetros para Análise",
            feature_options,
            default=default_features,
            help="Parâmetros para analisar interações"
        )

        # Atualiza características e alvo
        if selected_features and (target_col is None or target_col != "Nenhum"):
            self.set_features_target(selected_features, target_col)

            # Tabs para diferentes análises
            tabs = st.tabs([
                "Correlação e Interação",
                "Análise de Componentes",
                "Dependência Parcial",
                "Visualização de Alta Dimensão"
            ])

            with tabs[0]:
                st.subheader("Análise de Correlação e Interação")

                # Método de detecção de interação
                method = st.radio(
                    "Método de Detecção de Interação",
                    ["mutual_info", "correlation"],
                    horizontal=True,
                    help="mutual_info: detecta relações não lineares, correlation: apenas relações lineares"
                )

                # Botão para calcular interações
                if st.button("Calcular Interações", key="calc_interactions"):
                    with st.spinner("Calculando interações..."):
                        try:
                            # Detecta interações
                            interaction_matrix = self.detect_feature_interactions(method=method)

                            # Exibe mapa de calor
                            fig = self.plot_interaction_heatmap()
                            st.plotly_chart(fig, use_container_width=True)

                            # Exibe matriz de interação
                            st.subheader("Matriz de Interação")
                            st.dataframe(interaction_matrix.style.background_gradient(cmap='viridis', axis=None))

                            # Pares com maior interação
                            st.subheader("Pares com Maior Interação")

                            pairs = []
                            for i, feature1 in enumerate(selected_features):
                                for j, feature2 in enumerate(selected_features):
                                    if i < j:  # Evita duplicações
                                        score = interaction_matrix.loc[feature1, feature2]
                                        pairs.append((feature1, feature2, score))

                            # Ordena por pontuação decrescente
                            pairs.sort(key=lambda x: x[2], reverse=True)

                            # Mostra os 5 primeiros pares ou menos
                            pairs_df = pd.DataFrame(
                                [(p[0], p[1], p[2]) for p in pairs[:min(5, len(pairs))]],
                                columns=['Parâmetro 1', 'Parâmetro 2', 'Pontuação de Interação']
                            )

                            st.dataframe(pairs_df)

                        except Exception as e:
                            st.error(f"Erro ao calcular interações: {str(e)}")

            with tabs[1]:
                st.subheader("Análise de Componentes Principais")

                # Número de componentes
                n_components = st.slider(
                    "Número de Componentes",
                    2, min(5, len(selected_features)), 2,
                    help="Número de componentes principais a calcular"
                )

                # Padronização
                standardize = st.checkbox(
                    "Padronizar Dados",
                    value=True,
                    help="Recomendado quando as variáveis têm escalas diferentes"
                )

                # Botão para calcular PCA
                if st.button("Calcular PCA", key="calc_pca"):
                    with st.spinner("Calculando PCA..."):
                        try:
                            # Calcula PCA
                            pca_results = self.calculate_pca(
                                n_components=n_components,
                                scale=standardize
                            )

                            # Exibe biplot
                            fig = self.plot_pca_biplot()
                            st.plotly_chart(fig, use_container_width=True)

                            # Variância explicada
                            st.subheader("Variância Explicada")

                            # Cria DataFrame para variância explicada
                            var_exp_df = pd.DataFrame({
                                'Componente': [f'PC{i + 1}' for i in range(n_components)],
                                'Variância Explicada (%)': pca_results['explained_variance'] * 100,
                                'Variância Acumulada (%)': np.cumsum(pca_results['explained_variance']) * 100
                            })

                            st.dataframe(var_exp_df)

                            # Loadings das componentes
                            st.subheader("Loadings das Componentes")

                            # Formatação do DataFrame de loadings
                            loadings_df = pca_results['loadings']
                            st.dataframe(loadings_df.style.background_gradient(cmap='coolwarm', axis=None))

                        except Exception as e:
                            st.error(f"Erro ao calcular PCA: {str(e)}")

            with tabs[2]:
                st.subheader("Análise de Dependência Parcial")

                # Tipo de modelo
                model_type = st.radio(
                    "Tipo de Modelo",
                    ["rf"],
                    index=0,
                    horizontal=True,
                    help="rf: Random Forest"
                )

                # Parâmetros do modelo
                if model_type == "rf":
                    n_estimators = st.slider(
                        "Número de Árvores",
                        10, 200, 100, 10,
                        help="Mais árvores = modelo mais estável, mas mais lento"
                    )

                    max_depth = st.slider(
                        "Profundidade Máxima",
                        3, 20, 10, 1,
                        help="Profundidade máxima das árvores"
                    )

                    model_params = {
                        'n_estimators': n_estimators,
                        'max_depth': max_depth,
                        'random_state': 42
                    }
                else:
                    model_params = {}

                # Botão para ajustar modelo e calcular dependência parcial
                if st.button("Calcular Dependência Parcial", key="calc_pdp"):
                    if target_col is None or target_col == "Nenhum":
                        st.error("Selecione um alvo para calcular a dependência parcial.")
                    else:
                        with st.spinner("Ajustando modelo e calculando dependência parcial..."):
                            try:
                                # Ajusta o modelo
                                self.fit_model(model_type=model_type, **model_params)

                                # Calcula dependência parcial 1D
                                pdp_results = self.calculate_partial_dependence()

                                # Plota dependência parcial 1D
                                fig = self.plot_partial_dependence(pdp_results)
                                st.plotly_chart(fig, use_container_width=True)

                                # Calcula dependência parcial 2D para pares com maior interação
                                if self.interaction_scores is not None:
                                    # Pares com maior interação
                                    pairs = []
                                    for i, feature1 in enumerate(selected_features):
                                        for j, feature2 in enumerate(selected_features):
                                            if i < j:  # Evita duplicações
                                                score = self.interaction_scores.loc[feature1, feature2]
                                                pairs.append((feature1, feature2, score))

                                    # Ordena por pontuação decrescente
                                    pairs.sort(key=lambda x: x[2], reverse=True)

                                    # Pega os 2 primeiros pares
                                    feature_pairs = [(p[0], p[1]) for p in pairs[:min(2, len(pairs))]]
                                else:
                                    # Sem informações de interação, usa primeiros 2 pares
                                    feature_pairs = []
                                    for i, feature1 in enumerate(selected_features):
                                        for j, feature2 in enumerate(selected_features):
                                            if i < j:  # Evita duplicações
                                                feature_pairs.append((feature1, feature2))
                                                if len(feature_pairs) >= 2:
                                                    break
                                        if len(feature_pairs) >= 2:
                                            break

                                # Calcula dependência parcial 2D
                                pdp_2d_results = self.calculate_2d_partial_dependence(feature_pairs)

                                # Plota dependência parcial 2D
                                pdp_2d_figs = self.plot_2d_partial_dependence(pdp_2d_results)

                                # Exibe figuras 2D
                                st.subheader("Dependência Parcial 2D")

                                for (feature1, feature2), figs in pdp_2d_figs.items():
                                    st.markdown(f"**{feature1} x {feature2}**")

                                    # Cria guias para contorno e superfície
                                    cont_tab, surf_tab = st.tabs(["Contorno", "Superfície 3D"])

                                    with cont_tab:
                                        st.plotly_chart(figs['contour'], use_container_width=True)

                                    with surf_tab:
                                        st.plotly_chart(figs['surface'], use_container_width=True)

                            except Exception as e:
                                st.error(f"Erro ao calcular dependência parcial: {str(e)}")
                                import traceback
                                st.error(traceback.format_exc())

            with tabs[3]:
                st.subheader("Visualização de Alta Dimensão")

                # Tipo de visualização
                viz_type = st.radio(
                    "Tipo de Visualização",
                    ["Coordenadas Paralelas", "Matriz de Dispersão", "Curvas de Andrews"],
                    horizontal=True,
                    help="Diferentes técnicas para visualizar dados multidimensionais"
                )

                # Botão para gerar visualização
                if st.button("Gerar Visualização", key="gen_viz"):
                    with st.spinner("Gerando visualização..."):
                        try:
                            if viz_type == "Coordenadas Paralelas":
                                # Gera visualização de coordenadas paralelas
                                fig = self.create_parallel_coordinates()
                                st.plotly_chart(fig, use_container_width=True)

                            elif viz_type == "Matriz de Dispersão":
                                # Limita dimensões para matriz não ficar muito grande
                                max_dims = min(6, len(selected_features))

                                # Seleciona características com maior variância
                                if len(selected_features) > max_dims:
                                    # Calcula variância
                                    var = self.data[selected_features].var()

                                    # Ordena por variância
                                    sorted_features = var.sort_values(ascending=False).index[:max_dims]
                                else:
                                    sorted_features = selected_features

                                # Gera matriz de dispersão
                                fig = self.create_high_dimensional_scatter(dimensions=sorted_features)
                                st.plotly_chart(fig, use_container_width=True)

                            elif viz_type == "Curvas de Andrews":
                                # Gera curvas de Andrews
                                fig = self.create_andrews_curves()
                                st.plotly_chart(fig, use_container_width=True)

                        except Exception as e:
                            st.error(f"Erro ao gerar visualização: {str(e)}")
        else:
            st.warning("Selecione pelo menos um parâmetro para análise.")