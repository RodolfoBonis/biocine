"""
Visualização de dados e resultados de modelagem

Este módulo implementa funções para visualização de dados experimentais,
resultados de modelagem cinética e de machine learning.
"""

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots


def plot_growth_curve(time, biomass, model_prediction=None, title="Curva de Crescimento",
                      xlabel="Tempo (dias)", ylabel="Biomassa (g/L)"):
    """
    Plota a curva de crescimento experimental e prevista

    Args:
        time: Array com os tempos
        biomass: Array com os dados experimentais de biomassa
        model_prediction: Array com as previsões do modelo (opcional)
        title: Título do gráfico
        xlabel: Rótulo do eixo x
        ylabel: Rótulo do eixo y

    Returns:
        Figura do matplotlib
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Dados experimentais
    ax.scatter(time, biomass, color='blue', label='Dados Experimentais')

    # Previsão do modelo
    if model_prediction is not None:
        ax.plot(time, model_prediction, color='red', label='Modelo')

    # Configurações do gráfico
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()

    plt.tight_layout()
    return fig


def plot_substrate_consumption(time, substrate, model_prediction=None, title="Consumo de Substrato",
                               xlabel="Tempo (dias)", ylabel="Substrato (mg/L)"):
    """
    Plota a curva de consumo de substrato experimental e prevista

    Args:
        time: Array com os tempos
        substrate: Array com os dados experimentais de substrato
        model_prediction: Array com as previsões do modelo (opcional)
        title: Título do gráfico
        xlabel: Rótulo do eixo x
        ylabel: Rótulo do eixo y

    Returns:
        Figura do matplotlib
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Dados experimentais
    ax.scatter(time, substrate, color='green', label='Dados Experimentais')

    # Previsão do modelo
    if model_prediction is not None:
        ax.plot(time, model_prediction, color='red', label='Modelo')

    # Configurações do gráfico
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()

    plt.tight_layout()
    return fig


def plot_removal_efficiency(time, efficiency, pollutant_type, title=None,
                            xlabel="Tempo (dias)", ylabel="Eficiência de Remoção (%)"):
    """
    Plota a eficiência de remoção de poluentes

    Args:
        time: Array com os tempos
        efficiency: Array com os dados de eficiência de remoção
        pollutant_type: Tipo de poluente ('N', 'P', 'DQO')
        title: Título do gráfico (opcional)
        xlabel: Rótulo do eixo x
        ylabel: Rótulo do eixo y

    Returns:
        Figura do matplotlib
    """
    if title is None:
        title = f"Eficiência de Remoção de {pollutant_type}"

    fig, ax = plt.subplots(figsize=(10, 6))

    # Cores para diferentes poluentes
    colors = {'N': 'blue', 'P': 'green', 'DQO': 'purple'}
    color = colors.get(pollutant_type, 'gray')

    # Dados de eficiência
    ax.plot(time, efficiency, marker='o', color=color, label=f'{pollutant_type}')

    # Configurações do gráfico
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 105)  # Eficiência de 0% a 100%, com margem
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()

    plt.tight_layout()
    return fig


def plot_correlation_matrix(data, title="Matriz de Correlação"):
    """
    Plota a matriz de correlação entre as variáveis

    Args:
        data: DataFrame com os dados
        title: Título do gráfico

    Returns:
        Figura do matplotlib
    """
    # Calcula a matriz de correlação
    corr_matrix = data.corr()

    # Cria a figura
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plota o heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)

    # Configurações do gráfico
    ax.set_title(title)

    plt.tight_layout()
    return fig


def plot_feature_importance(importance_df, title="Importância das Características"):
    """
    Plota a importância das características de um modelo de machine learning

    Args:
        importance_df: DataFrame com as colunas 'Feature' e 'Importance'
        title: Título do gráfico

    Returns:
        Figura do matplotlib
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plota a importância das características
    sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)

    # Configurações do gráfico
    ax.set_title(title)
    ax.set_xlabel('Importância')
    ax.set_ylabel('Característica')

    plt.tight_layout()
    return fig


def plot_model_comparison(models_metrics, title="Comparação de Modelos"):
    """
    Plota a comparação de métricas entre diferentes modelos

    Args:
        models_metrics: Dictionary com {nome_do_modelo: {métrica: valor}}
        title: Título do gráfico

    Returns:
        Figura do matplotlib
    """
    # Prepara os dados
    model_names = list(models_metrics.keys())
    r2_scores = [metrics.get('r_squared', 0) for metrics in models_metrics.values()]
    mse_scores = [metrics.get('mse', 0) for metrics in models_metrics.values()]

    # Cria a figura
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plota o R²
    ax1.bar(model_names, r2_scores, color='blue')
    ax1.set_title('Coeficiente de Determinação (R²)')
    ax1.set_ylim(0, 1)
    ax1.grid(True, linestyle='--', alpha=0.7, axis='y')

    # Plota o MSE
    ax2.bar(model_names, mse_scores, color='red')
    ax2.set_title('Erro Quadrático Médio (MSE)')
    ax2.grid(True, linestyle='--', alpha=0.7, axis='y')

    # Configurações do gráfico
    fig.suptitle(title)
    plt.tight_layout()

    return fig


def plot_interactive_growth_curve(time, biomass, model_prediction=None, title="Curva de Crescimento"):
    """
    Plota a curva de crescimento interativa com Plotly

    Args:
        time: Array com os tempos
        biomass: Array com os dados experimentais de biomassa
        model_prediction: Array com as previsões do modelo (opcional)
        title: Título do gráfico

    Returns:
        Figura do Plotly
    """
    fig = go.Figure()

    # Dados experimentais
    fig.add_trace(go.Scatter(
        x=time,
        y=biomass,
        mode='markers',
        name='Dados Experimentais',
        marker=dict(color='blue', size=8)
    ))

    # Previsão do modelo
    if model_prediction is not None:
        fig.add_trace(go.Scatter(
            x=time,
            y=model_prediction,
            mode='lines',
            name='Modelo',
            line=dict(color='red', width=2)
        ))

    # Configurações do layout
    fig.update_layout(
        title=title,
        xaxis_title='Tempo (dias)',
        yaxis_title='Biomassa (g/L)',
        legend=dict(x=0.01, y=0.99),
        hovermode='closest',
        template='plotly_white'
    )

    return fig


def plot_interactive_combined(time, data, model_predictions=None, title="Processo de Tratamento"):
    """
    Plota gráficos interativos combinados do processo de tratamento com Plotly

    Args:
        time: Array com os tempos
        data: Dictionary com {nome_variável: valores}
        model_predictions: Dictionary com {nome_variável: valores_preditos} (opcional)
        title: Título do gráfico

    Returns:
        Figura do Plotly
    """
    # Determina o número de subplots necessários
    n_plots = len(data)

    # Cria os subplots
    fig = make_subplots(rows=n_plots, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.1,
                        subplot_titles=list(data.keys()))

    # Cores para diferentes variáveis
    colors = ['blue', 'green', 'purple', 'orange', 'brown', 'pink']

    # Adiciona os traços para cada variável
    for i, (var_name, values) in enumerate(data.items()):
        # Índice baseado em 1 para o subplot
        row = i + 1
        color = colors[i % len(colors)]

        # Dados experimentais
        fig.add_trace(
            go.Scatter(
                x=time,
                y=values,
                mode='markers',
                name=f'{var_name} (Exp)',
                marker=dict(color=color, size=8),
                showlegend=True
            ),
            row=row, col=1
        )

        # Previsões do modelo, se disponíveis
        if model_predictions is not None and var_name in model_predictions:
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=model_predictions[var_name],
                    mode='lines',
                    name=f'{var_name} (Modelo)',
                    line=dict(color=color, width=2, dash='dash'),
                    showlegend=True
                ),
                row=row, col=1
            )

    # Atualiza o layout
    fig.update_layout(
        title=title,
        height=300 * n_plots,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template='plotly_white'
    )

    # Atualiza os eixos x
    fig.update_xaxes(title_text="Tempo (dias)", row=n_plots, col=1)

    return fig