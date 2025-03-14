"""
Utilitários para cálculo e visualização de dependência parcial

Este módulo contém funções robustas para calcular e visualizar
a dependência parcial em modelos de aprendizado de máquina.
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def validate_data_for_pdp(data, features, target):
    """
    Verifica se os dados são adequados para cálculo de dependência parcial

    Args:
        data: DataFrame com os dados
        features: Lista de características
        target: Nome da característica alvo

    Returns:
        bool: True se os dados são válidos, False caso contrário
        str: Mensagem de erro ou None
    """
    # Verifica se há dados
    if data is None or data.empty:
        return False, "Dados ausentes ou vazios"

    # Verifica se as características existem
    missing_features = [f for f in features if f not in data.columns]
    if missing_features:
        return False, f"Características ausentes nos dados: {', '.join(missing_features)}"

    # Verifica se o alvo existe
    if target not in data.columns:
        return False, f"Característica alvo '{target}' ausente nos dados"

    # Verifica se há valores válidos
    for feature in features:
        if data[feature].isnull().all():
            return False, f"Característica '{feature}' contém apenas valores nulos"

    if data[target].isnull().all():
        return False, f"Alvo '{target}' contém apenas valores nulos"

    # Verifica se há variação nos dados
    for feature in features:
        if len(data[feature].unique()) < 2:
            return False, f"Característica '{feature}' não possui variação (valores únicos: {data[feature].unique()[0]})"

    return True, None


def calculate_partial_dependence_safely(model, data, features, target):
    """
    Calcula dependência parcial com verificações de segurança

    Args:
        model: Modelo treinado
        data: DataFrame com os dados
        features: Lista de características
        target: Nome da característica alvo

    Returns:
        dict: Resultados da dependência parcial ou dict vazio se houver erro
        str: Mensagem de erro ou None
    """
    if model is None:
        return {}, "Modelo não disponível"

    # Valida os dados
    is_valid, error_msg = validate_data_for_pdp(data, features, target)
    if not is_valid:
        return {}, error_msg

    # Cálculo de dependência parcial
    pdp_results = {}

    for feature in features:
        try:
            # Cria uma grade de valores para a característica
            feature_values = data[feature].values
            min_val = np.nanmin(feature_values)
            max_val = np.nanmax(feature_values)

            # Se min_val e max_val são iguais, ajusta levemente
            if min_val == max_val:
                min_val = min_val * 0.95 if min_val != 0 else -0.1
                max_val = max_val * 1.05 if max_val != 0 else 0.1

            # Cria uma sequência de valores dentro do range
            grid_points = min(20, len(np.unique(feature_values)))
            if grid_points < 2:
                grid_points = 2  # Garante pelo menos 2 pontos

            grid = np.linspace(min_val, max_val, grid_points)

            # Arrays para armazenar resultados
            pdp_values = []

            # Filtra linhas sem valores nulos para evitar problemas
            valid_data = data.dropna(subset=[feature, target])

            if valid_data.empty:
                continue

            X_features = [f for f in features if f != feature]
            X = valid_data[X_features].values

            # Para cada valor na grade
            for val in grid:
                # Cria cópias para cada linha do dataset
                X_copies = np.repeat(X, 1, axis=0)

                # Prepara vetor de entrada para o modelo
                X_with_feature = []

                for row in X_copies:
                    # Cria dicionário com todas as características
                    row_dict = {f: v for f, v in zip(X_features, row)}
                    row_dict[feature] = val

                    # Mantém apenas as colunas que o modelo conhece
                    X_with_feature.append([row_dict[f] for f in features])

                X_with_feature = np.array(X_with_feature)

                # Verifica se temos dados válidos
                if len(X_with_feature) == 0:
                    continue

                # Faz predição (com tratamento de erro)
                try:
                    y_pred = model.predict(X_with_feature)
                    avg_pred = np.mean(y_pred)
                    pdp_values.append(avg_pred)
                except Exception as e:
                    st.warning(f"Erro ao fazer predição para {feature}={val}: {str(e)}")
                    continue

            # Verifica se temos resultados
            if len(pdp_values) > 0:
                pdp_results[feature] = {
                    'values': grid[:len(pdp_values)],
                    'pdp': np.array(pdp_values)
                }

        except Exception as e:
            st.warning(f"Erro ao calcular dependência parcial para {feature}: {str(e)}")
            import traceback
            st.info(f"Detalhes técnicos: {traceback.format_exc()}")

    return pdp_results, None


def plot_partial_dependence_safely(pdp_results):
    """
    Plota curvas de dependência parcial com verificações de segurança

    Args:
        pdp_results: Resultados de calculate_partial_dependence

    Returns:
        fig: Figura do plotly ou None se não for possível plotar
    """
    # Verifica se há resultados
    if pdp_results is None or len(pdp_results) == 0:
        # Retorna uma figura simples com uma mensagem
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

    # Configuração de subplots
    n_cols = min(2, n_features)
    n_rows = max(1, (n_features + n_cols - 1) // n_cols)

    # Cria subplots
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=list(pdp_results.keys())
    )

    # Contador para subplots
    feature_idx = 0

    for feature, result in pdp_results.items():
        # Verifica se temos dados válidos
        if 'values' not in result or 'pdp' not in result:
            continue

        if len(result['values']) == 0 or len(result['pdp']) == 0:
            continue

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


def calculate_2d_partial_dependence_safely(model, data, feature_pairs, features, grid_resolution=20):
    """
    Calcula dependência parcial 2D para pares de características com tratamento seguro

    Args:
        model: Modelo treinado
        data: DataFrame com os dados
        feature_pairs: Lista de pares de características
        features: Lista completa de características usadas pelo modelo
        grid_resolution: Resolução da grade

    Returns:
        Dictionary com superfícies de dependência parcial ou dict vazio se houver erro
        str: Mensagem de erro ou None
    """
    if model is None:
        return {}, "Modelo não disponível"

    # Verifica se há pares de características válidos
    if not feature_pairs:
        return {}, "Nenhum par de características fornecido"

    # Prepara dados para o modelo
    X = data[features].values

    # Verifica se temos dados suficientes
    if len(X) < 10:  # Define um mínimo arbitrário
        return {}, "Dados insuficientes para cálculo de dependência parcial 2D"

    # Calcula dependência parcial 2D
    pdp_2d_results = {}

    for feature1, feature2 in feature_pairs:
        try:
            # Verifica se as características existem nos dados
            if feature1 not in features or feature2 not in features:
                st.warning(f"Características {feature1} ou {feature2} não encontradas no modelo")
                continue

            # Mapeia nomes para índices
            idx1 = features.index(feature1)
            idx2 = features.index(feature2)

            # Determina valores mínimos e máximos para cada característica
            min1, max1 = np.min(data[feature1]), np.max(data[feature1])
            min2, max2 = np.min(data[feature2]), np.max(data[feature2])

            # Cria grades de valores
            grid1 = np.linspace(min1, max1, grid_resolution)
            grid2 = np.linspace(min2, max2, grid_resolution)

            # Inicializa matriz de resultados
            pdp_matrix = np.zeros((grid_resolution, grid_resolution))

            for i, val1 in enumerate(grid1):
                for j, val2 in enumerate(grid2):
                    # Cria cópias do conjunto de dados
                    X_modified = X.copy()

                    # Substitui os valores das características
                    X_modified[:, idx1] = val1
                    X_modified[:, idx2] = val2

                    # Faz predições e calcula média
                    try:
                        y_pred = model.predict(X_modified)
                        pdp_matrix[i, j] = np.mean(y_pred)
                    except Exception as e:
                        st.warning(f"Erro na predição para {feature1}={val1}, {feature2}={val2}: {str(e)}")
                        # Usa média global como fallback
                        pdp_matrix[i, j] = np.mean(model.predict(X))

            # Armazena resultados
            pdp_2d_results[(feature1, feature2)] = {
                'values1': grid1,
                'values2': grid2,
                'pdp': pdp_matrix
            }

        except Exception as e:
            import traceback
            st.warning(f"Erro ao calcular dependência parcial 2D para {feature1}, {feature2}: {str(e)}")
            st.info(traceback.format_exc())  # Mostra stack trace para debug

    return pdp_2d_results, None


def plot_2d_partial_dependence_safely(pdp_2d_results):
    """
    Plota superfícies de dependência parcial 2D com verificações de segurança

    Args:
        pdp_2d_results: Resultados de calculate_2d_partial_dependence

    Returns:
        Dictionary com figuras do plotly ou dict vazio se houver erro
    """
    if not pdp_2d_results:
        st.warning("Nenhum resultado de dependência parcial 2D disponível para plotar")

        # Retorna uma figura vazia para evitar erros na interface
        fig = go.Figure()
        fig.add_annotation(text="Nenhum dado para plotar", showarrow=False)
        fig.update_layout(
            title="Dependência Parcial 2D",
            height=400,
            width=600
        )

        return {"empty": {"contour": fig, "surface": fig}}

    # Cria uma figura para cada par
    figures = {}

    for (feature1, feature2), result in pdp_2d_results.items():
        try:
            # Verifica se tem os campos necessários
            if not all(key in result for key in ['values1', 'values2', 'pdp']):
                st.warning(f"Dados incompletos para o par {feature1}, {feature2}")
                continue

            # Verifica se os arrays têm o formato correto
            values1 = result['values1']
            values2 = result['values2']
            pdp_matrix = result['pdp']

            if len(values1) == 0 or len(values2) == 0:
                st.warning(f"Arrays vazios para o par {feature1}, {feature2}")
                continue

            # Verifica as dimensões da matriz pdp
            if pdp_matrix.shape != (len(values1), len(values2)):
                # Tenta redimensionar se possível
                try:
                    pdp_matrix = pdp_matrix.reshape(len(values1), len(values2))
                except:
                    st.warning(f"Dimensões incompatíveis para a matriz PDL do par {feature1}, {feature2}")
                    continue

            # Cria figura com contorno
            contour_fig = go.Figure(data=[
                go.Contour(
                    z=pdp_matrix,
                    x=values1,
                    y=values2,
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
                    z=pdp_matrix,
                    x=values1,
                    y=values2,
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

        except Exception as e:
            import traceback
            st.warning(f"Erro ao gerar visualização para {feature1}, {feature2}: {str(e)}")
            st.info(traceback.format_exc())

    # Se não conseguiu gerar nenhuma figura, retorna uma figura vazia
    if not figures:
        fig = go.Figure()
        fig.add_annotation(text="Nenhum gráfico pôde ser gerado", showarrow=False)
        fig.update_layout(
            title="Dependência Parcial 2D",
            height=400,
            width=600
        )

        figures["empty"] = {"contour": fig, "surface": fig}

    return figures