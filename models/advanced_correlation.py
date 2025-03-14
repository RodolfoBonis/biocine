"""
Análise de correlação avançada para bioprocessos

Este módulo estende as funcionalidades básicas de correlação, adicionando:
- Testes de significância estatística
- Correlações parciais
- Análise de multicolinearidade
- Correlações não lineares
- Visualizações avançadas de correlação
"""

import numpy as np
import pandas as pd
import pingouin as pg
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor


def correlation_analysis(data, method='pearson', alpha=0.05, include_tests=True,
                         plot_type='heatmap', vif_threshold=5):
    """
    Realiza análise de correlação avançada com testes estatísticos.

    Args:
        data: DataFrame com os dados
        method: Método de correlação ('pearson', 'spearman', 'kendall')
        alpha: Nível de significância para testes estatísticos
        include_tests: Se True, inclui testes de significância
        plot_type: Tipo de visualização ('heatmap', 'pairplot', 'network')
        vif_threshold: Limiar para detectar multicolinearidade

    Returns:
        Dictionary com resultados da análise e figura
    """
    # Certifica-se de usar apenas colunas numéricas
    numeric_data = data.select_dtypes(include=[np.number])

    if numeric_data.empty:
        raise ValueError("Nenhuma coluna numérica encontrada nos dados")

    # 1. Matriz de correlação básica
    if method == 'pearson':
        corr_matrix = numeric_data.corr(method=method)
    elif method == 'spearman':
        corr_matrix = numeric_data.corr(method=method)
    elif method == 'kendall':
        corr_matrix = numeric_data.corr(method=method)
    else:
        raise ValueError(f"Método de correlação inválido: {method}")

    # 2. Testes de significância estatística
    if include_tests:
        # Matriz de p-valores (utilizando pingouin)
        pval_matrix = pd.DataFrame(np.zeros_like(corr_matrix),
                                   index=corr_matrix.index,
                                   columns=corr_matrix.columns)

        # Preenche a matriz de p-valores
        for i, col_i in enumerate(numeric_data.columns):
            for j, col_j in enumerate(numeric_data.columns):
                if i != j:  # Skip diagonal elements
                    if method == 'pearson':
                        corr_test = pg.corr(numeric_data[col_i], numeric_data[col_j], method=method)
                        pval_matrix.loc[col_i, col_j] = corr_test['p-val'].values[0]
                    else:
                        # Teste de correlação para Spearman/Kendall
                        corr, pval = stats.spearmanr(numeric_data[col_i], numeric_data[col_j]) if method == 'spearman' \
                            else stats.kendalltau(numeric_data[col_i], numeric_data[col_j])
                        pval_matrix.loc[col_i, col_j] = pval

        # Matriz de significância (True para correlações significativas)
        sig_matrix = pval_matrix < alpha
    else:
        pval_matrix = None
        sig_matrix = None

    # 3. Análise de multicolinearidade (VIF)
    vif_data = pd.DataFrame()

    try:
        # Adiciona constante para cálculo VIF
        from statsmodels.tools.tools import add_constant
        X = add_constant(numeric_data)

        # Calcula VIF para cada feature
        vif_data['Feature'] = X.columns[1:]  # Exclui a constante
        vif_data['VIF'] = [variance_inflation_factor(X.values, i + 1) for i in range(len(X.columns[1:]))]

        # Identifica features com multicolinearidade
        vif_data['High_Multicollinearity'] = vif_data['VIF'] > vif_threshold
    except:
        # Em caso de erro (ex: multicolinearidade perfeita), apenas informa que não foi possível calcular
        vif_data = pd.DataFrame({'Message': ['Não foi possível calcular VIF. Possível colinearidade perfeita.']})

    # 4. Correlações parciais
    # Calcula matriz de correlações parciais, controlando para todas as outras variáveis
    try:
        pcorr_matrix = pg.pcorr(numeric_data)
    except:
        pcorr_matrix = pd.DataFrame({'Message': ['Não foi possível calcular correlações parciais.']})

    # 5. Detecção de correlações não-lineares - usando Mutual Information
    from sklearn.feature_selection import mutual_info_regression

    # Dictionary para armazenar scores de informação mútua
    mi_scores = {}

    for col in numeric_data.columns:
        # Usa cada coluna como target para calcular MI com outras features
        mi_scores[col] = mutual_info_regression(numeric_data.drop(columns=[col]), numeric_data[col])

    # Cria DataFrame de MI Scores
    mi_df = pd.DataFrame(index=numeric_data.columns)

    for col, scores in mi_scores.items():
        # Para cada coluna, adiciona os scores MI com todas as outras
        col_scores = pd.Series(scores, index=numeric_data.drop(columns=[col]).columns)
        mi_df[col] = pd.Series(0, index=numeric_data.columns)  # Inicializa com zeros
        mi_df.loc[col_scores.index, col] = col_scores

    # 6. Visualizações
    if plot_type == 'heatmap':
        # Heatmap avançado com significância
        fig = create_advanced_heatmap(corr_matrix, pval_matrix, alpha)
    elif plot_type == 'pairplot':
        # Pairplot para visualizar relações bivariadas
        fig = px.scatter_matrix(
            numeric_data,
            dimensions=numeric_data.columns,
            title="Matriz de Dispersão",
            opacity=0.7
        )
    elif plot_type == 'network':
        # Grafo de rede de correlações
        fig = create_correlation_network(corr_matrix, threshold=0.5)
    else:
        fig = None

    # Prepara resultado final
    result = {
        'correlation_matrix': corr_matrix,
        'p_values': pval_matrix,
        'significant_correlations': sig_matrix,
        'vif_analysis': vif_data,
        'partial_correlations': pcorr_matrix,
        'mutual_information': mi_df,
        'figure': fig
    }

    return result


def create_advanced_heatmap(corr_matrix, pval_matrix=None, alpha=0.05):
    """
    Cria um heatmap avançado com marcadores para correlações significativas.

    Args:
        corr_matrix: Matriz de correlação
        pval_matrix: Matriz de p-valores
        alpha: Nível de significância

    Returns:
        Figura do plotly
    """
    # Cria uma máscara para correlações não significativas
    mask = np.ones_like(corr_matrix)

    if pval_matrix is not None:
        mask = pval_matrix < alpha

    # Converte para um formato que o plotly pode usar
    corr_z = corr_matrix.values

    # Cria figura base
    fig = go.Figure()

    # Heatmap das correlações
    heatmap = go.Heatmap(
        z=corr_z,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu_r',
        zmin=-1,
        zmax=1,
        showscale=True
    )

    fig.add_trace(heatmap)

    # Adiciona asteriscos para indicar significância
    if pval_matrix is not None:
        annotations = []
        for i, row in enumerate(pval_matrix.index):
            for j, col in enumerate(pval_matrix.columns):
                if i != j:  # Skip diagonal
                    if pval_matrix.iloc[i, j] < 0.001:
                        symbol = '***'
                    elif pval_matrix.iloc[i, j] < 0.01:
                        symbol = '**'
                    elif pval_matrix.iloc[i, j] < 0.05:
                        symbol = '*'
                    else:
                        symbol = ''

                    if symbol:
                        annotations.append(dict(
                            x=col,
                            y=row,
                            text=symbol,
                            showarrow=False,
                            font=dict(color='white' if abs(corr_matrix.iloc[i, j]) > 0.6 else 'black')
                        ))

        fig.update_layout(annotations=annotations)

    # Configurações de layout
    fig.update_layout(
        title="Matriz de Correlação com Significância Estatística",
        xaxis_title="",
        yaxis_title="",
        xaxis=dict(tickangle=-45),
        height=700,
        width=700,
        coloraxis_colorbar=dict(
            title="Coeficiente de Correlação",
            thicknessmode="pixels", thickness=15,
            lenmode="pixels", len=300,
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=['-1', '-0.5', '0', '0.5', '1']
        )
    )

    return fig


def create_correlation_network(corr_matrix, threshold=0.5):
    """
    Cria um grafo de rede de correlações.

    Args:
        corr_matrix: Matriz de correlação
        threshold: Limiar mínimo de correlação absoluta para mostrar conexões

    Returns:
        Figura do plotly
    """
    # Prepara dados para o grafo
    edges = []
    edge_weights = []

    for i, source in enumerate(corr_matrix.columns):
        for j, target in enumerate(corr_matrix.columns):
            if i < j:  # Evita duplicações
                corr = corr_matrix.loc[source, target]
                if abs(corr) >= threshold:
                    edges.append((source, target))
                    edge_weights.append(corr)

    # Cria layout do grafo
    import networkx as nx
    G = nx.Graph()

    # Adiciona nós
    for col in corr_matrix.columns:
        G.add_node(col)

    # Adiciona arestas
    for (source, target), weight in zip(edges, edge_weights):
        G.add_edge(source, target, weight=weight)

    # Posicionamento dos nós
    pos = nx.spring_layout(G, seed=42)

    # Converte para coordenadas plotly
    node_x = []
    node_y = []
    for node, position in pos.items():
        node_x.append(position[0])
        node_y.append(position[1])

    # Define os nós
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=list(G.nodes()),
        textposition="top center",
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=15,
            colorbar=dict(
                thickness=15,
                title='Centralidade de Grau',
                xanchor='left',
                titleside='right'
            )
        ),
        hoverinfo='text'
    )

    # Calcula centralidade de grau para cor dos nós
    node_degrees = dict(G.degree())
    node_trace.marker.color = list(node_degrees.values())

    # Define as arestas
    edge_x = []
    edge_y = []
    edge_colors = []

    for (source, target), weight in zip(edges, edge_weights):
        x0, y0 = pos[source]
        x1, y1 = pos[target]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_colors.append(weight)

    # Cria o trace das arestas
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='rgba(150,150,150,0.6)'),
        hoverinfo='none',
        mode='lines'
    )

    # Cria a figura
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Rede de Correlações',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))

    return fig


def nonlinear_relationship_exploration(data, x_col, y_col):
    """
    Explora a relação não linear entre duas variáveis.

    Args:
        data: DataFrame com os dados
        x_col: Nome da coluna para o eixo x
        y_col: Nome da coluna para o eixo y

    Returns:
        Figura do plotly com múltiplas visualizações
    """
    # Extrai os dados
    x = data[x_col].values
    y = data[y_col].values

    # Cria subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Dispersão',
            'Transformação Log-Log',
            'LOWESS (Suavização Local)',
            'Modelo Polinomial'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )

    # 1. Gráfico de dispersão simples
    fig.add_trace(
        go.Scatter(
            x=x, y=y,
            mode='markers',
            name='Dados Originais',
            marker=dict(color='blue', size=8)
        ),
        row=1, col=1
    )

    # Calcula correlação linear
    pearson_corr, p_value = stats.pearsonr(x, y)
    linear_text = f'Correlação de Pearson: {pearson_corr:.3f}<br>p-valor: {p_value:.4f}'

    fig.add_annotation(
        x=0.98, y=0.15,
        xref='x domain', yref='y domain',
        text=linear_text,
        showarrow=False,
        row=1, col=1
    )

    # 2. Transformação Log-Log
    # Lida com valores zero ou negativos
    x_log = np.log(x + 1e-10) if np.any(x <= 0) else np.log(x)
    y_log = np.log(y + 1e-10) if np.any(y <= 0) else np.log(y)

    fig.add_trace(
        go.Scatter(
            x=x_log, y=y_log,
            mode='markers',
            name='Log-Log',
            marker=dict(color='green', size=8)
        ),
        row=1, col=2
    )

    # Calcula correlação log-log
    log_corr, log_p = stats.pearsonr(x_log, y_log)
    log_text = f'Correlação Log-Log: {log_corr:.3f}<br>p-valor: {log_p:.4f}'

    fig.add_annotation(
        x=0.98, y=0.15,
        xref='x2 domain', yref='y2 domain',
        text=log_text,
        showarrow=False,
        row=1, col=2
    )

    # 3. LOWESS (Suavização local)
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        # Calcula a suavização LOWESS
        z = lowess(y, x, frac=0.3)

        fig.add_trace(
            go.Scatter(
                x=x, y=y,
                mode='markers',
                name='Dados',
                marker=dict(color='blue', size=5, opacity=0.5),
                showlegend=False
            ),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=z[:, 0], y=z[:, 1],
                mode='lines',
                name='LOWESS',
                line=dict(color='red', width=2)
            ),
            row=2, col=1
        )
    except:
        fig.add_trace(
            go.Scatter(
                x=[0], y=[0],
                mode='markers',
                marker=dict(opacity=0)
            ),
            row=2, col=1
        )

        fig.add_annotation(
            x=0.5, y=0.5,
            xref='x3 domain', yref='y3 domain',
            text='Erro ao calcular LOWESS',
            showarrow=False,
            row=2, col=1
        )

    # 4. Modelo Polinomial
    try:
        # Ajusta um polinômio de grau 3
        z = np.polyfit(x, y, 3)
        p = np.poly1d(z)

        # Gera pontos para plotar a linha
        x_line = np.linspace(min(x), max(x), 100)
        y_line = p(x_line)

        # Calcula R²
        y_pred = p(x)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        r_squared = 1 - (ss_residual / ss_total)

        fig.add_trace(
            go.Scatter(
                x=x, y=y,
                mode='markers',
                name='Dados',
                marker=dict(color='blue', size=5, opacity=0.5),
                showlegend=False
            ),
            row=2, col=2
        )

        fig.add_trace(
            go.Scatter(
                x=x_line, y=y_line,
                mode='lines',
                name='Polinômio (Grau 3)',
                line=dict(color='purple', width=2)
            ),
            row=2, col=2
        )

        poly_text = f'Modelo Polinomial<br>R²: {r_squared:.3f}'

        fig.add_annotation(
            x=0.98, y=0.15,
            xref='x4 domain', yref='y4 domain',
            text=poly_text,
            showarrow=False,
            row=2, col=2
        )
    except:
        fig.add_trace(
            go.Scatter(
                x=[0], y=[0],
                mode='markers',
                marker=dict(opacity=0)
            ),
            row=2, col=2
        )

        fig.add_annotation(
            x=0.5, y=0.5,
            xref='x4 domain', yref='y4 domain',
            text='Erro ao ajustar polinômio',
            showarrow=False,
            row=2, col=2
        )

    # Atualiza layout
    fig.update_layout(
        title=f'Análise de Relação entre {x_col} e {y_col}',
        height=800,
        width=1000
    )

    return fig


# Adiciona a função para ser utilizada com o pacote plotly
from plotly.subplots import make_subplots