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