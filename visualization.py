import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from models import CineticModels


class Visualization:
    """
    Classe para visualização avançada dos dados e resultados de modelagem
    """

    @staticmethod
    def create_growth_graph(df, biomass_column="Biomassa Consórcio (g/L)", time_column="Tempo (dias)"):
        """
        Cria um gráfico de crescimento de biomassa

        Args:
            df: DataFrame com os dados
            biomass_column: Nome da coluna de biomassa
            time_column: Nome da coluna de tempo

        Returns:
            fig: Objeto de figura do Plotly
        """
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df[time_column],
            y=df[biomass_column],
            mode='lines+markers',
            name=biomass_column
        ))

        fig.update_layout(
            title='Crescimento de Biomassa ao Longo do Tempo',
            xaxis_title='Tempo (dias)',
            yaxis_title='Biomassa (g/L)',
            legend=dict(x=0.02, y=0.98),
            template='plotly_white'
        )

        return fig

    @staticmethod
    def create_pollutant_removal_graph(df, pollutant_column, time_column="Tempo (dias)"):
        """
        Cria um gráfico de remoção de poluente com eficiência

        Args:
            df: DataFrame com os dados
            pollutant_column: Nome da coluna do poluente
            time_column: Nome da coluna de tempo

        Returns:
            fig: Objeto de figura do Plotly
        """
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Concentração do poluente
        fig.add_trace(
            go.Scatter(
                x=df[time_column],
                y=df[pollutant_column],
                mode='lines+markers',
                name=f'Concentração de {pollutant_column}'
            ),
            secondary_y=False
        )

        # Calculando a eficiência de remoção
        valor_inicial = df[pollutant_column].iloc[0]
        eficiencia = [100 * (1 - valor / valor_inicial) for valor in df[pollutant_column]]

        # Eficiência de remoção
        fig.add_trace(
            go.Scatter(
                x=df[time_column],
                y=eficiencia,
                mode='lines+markers',
                name='Eficiência de Remoção (%)',
                line=dict(color='red')
            ),
            secondary_y=True
        )

        fig.update_layout(
            title=f'Remoção de {pollutant_column} ao Longo do Tempo',
            template='plotly_white'
        )

        fig.update_xaxes(title_text='Tempo (dias)')
        fig.update_yaxes(title_text=pollutant_column, secondary_y=False)
        fig.update_yaxes(title_text='Eficiência de Remoção (%)', secondary_y=True)

        return fig

    @staticmethod
    def create_model_fit_graph(time_data, param_data, param_name, model_type, model_params):
        """
        Cria um gráfico de ajuste de modelo

        Args:
            time_data: Array com os tempos
            param_data: Array com os dados do parâmetro
            param_name: Nome do parâmetro
            model_type: Tipo de modelo ('logistic' ou 'monod')
            model_params: Dicionário com os parâmetros do modelo

        Returns:
            fig: Objeto de figura do Plotly
        """
        fig = go.Figure()

        # Dados experimentais
        fig.add_trace(go.Scatter(
            x=time_data,
            y=param_data,
            mode='markers',
            name='Dados Experimentais'
        ))

        # Curva ajustada
        t_fine = np.linspace(min(time_data), max(time_data), 100)

        if model_type == 'logistic':
            X0, umax, Xmax = model_params['X0'], model_params['umax'], model_params['Xmax']
            y_pred = CineticModels.logistic_model(t_fine, X0, umax, Xmax)

            model_name = 'Modelo Logístico'
            param_info = f'X₀={X0:.2f}, μₘₐₓ={umax:.2f}, Xₘₐₓ={Xmax:.2f}'

        elif model_type == 'logistic_inverted':
            S0, k, Smax = model_params['S0'], model_params['k'], model_params['Smax']
            y_pred_inv = CineticModels.logistic_model(t_fine, 0, k, Smax)
            y_pred = S0 - y_pred_inv

            model_name = 'Modelo Invertido'
            param_info = f'S₀={S0:.2f}, k={k:.2f}, Sₘₐₓ={Smax:.2f}'

        else:
            return fig

        fig.add_trace(go.Scatter(
            x=t_fine,
            y=y_pred,
            mode='lines',
            name=model_name,
            line=dict(width=3)
        ))

        fig.update_layout(
            title=f'Ajuste de {model_name} - {param_name}<br><sup>{param_info}</sup>',
            xaxis_title='Tempo (dias)',
            yaxis_title=param_name,
            legend=dict(x=0.02, y=0.98),
            template='plotly_white'
        )

        return fig

    @staticmethod
    def create_monod_fit_graph(substrate_data, growth_rates, model_params):
        """
        Cria um gráfico de ajuste do modelo de Monod

        Args:
            substrate_data: Array com as concentrações de substrato
            growth_rates: Array com as taxas de crescimento
            model_params: Dicionário com os parâmetros do modelo (umax, Ks)

        Returns:
            fig: Objeto de figura do Plotly
        """
        fig = go.Figure()

        # Dados experimentais
        fig.add_trace(go.Scatter(
            x=substrate_data,
            y=growth_rates,
            mode='markers',
            name='Taxas de Crescimento Experimentais'
        ))

        # Curva ajustada
        umax, Ks = model_params['umax'], model_params['Ks']
        s_fine = np.linspace(min(substrate_data), max(substrate_data), 100)
        mu_pred = CineticModels.monod_model(s_fine, umax, Ks)

        fig.add_trace(go.Scatter(
            x=s_fine,
            y=mu_pred,
            mode='lines',
            name='Modelo de Monod Ajustado',
            line=dict(width=3)
        ))

        fig.update_layout(
            title=f'Ajuste do Modelo de Monod<br><sup>μₘₐₓ={umax:.2f}, Kₛ={Ks:.2f}</sup>',
            xaxis_title='Concentração de Substrato (mg/L)',
            yaxis_title='Taxa de Crescimento Específica (dia⁻¹)',
            legend=dict(x=0.02, y=0.98),
            template='plotly_white'
        )

        return fig

    @staticmethod
    def create_multi_parameter_graph(df, params, time_column="Tempo (dias)"):
        """
        Cria um gráfico com múltiplos parâmetros normalizados

        Args:
            df: DataFrame com os dados
            params: Lista de parâmetros para incluir
            time_column: Nome da coluna de tempo

        Returns:
            fig: Objeto de figura do Plotly
        """
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        biomass_params = [p for p in params if "Biomassa" in p]
        pollutant_params = [p for p in params if "Biomassa" not in p]

        # Adicionar parâmetros de biomassa
        for param in biomass_params:
            # Normalizar pelo valor máximo
            norm_data = df[param] / df[param].max()

            fig.add_trace(
                go.Scatter(
                    x=df[time_column],
                    y=norm_data,
                    mode='lines+markers',
                    name=param
                ),
                secondary_y=False
            )

        # Adicionar parâmetros de poluentes
        for param in pollutant_params:
            # Normalizar e inverter (para mostrar remoção)
            norm_data = 1 - (df[param] / df[param].iloc[0])

            fig.add_trace(
                go.Scatter(
                    x=df[time_column],
                    y=norm_data,
                    mode='lines+markers',
                    name=f'Remoção de {param}'
                ),
                secondary_y=True
            )

        fig.update_layout(
            title='Comparação de Parâmetros ao Longo do Tempo',
            template='plotly_white'
        )

        fig.update_xaxes(title_text='Tempo (dias)')
        fig.update_yaxes(title_text='Biomassa Normalizada', secondary_y=False)
        fig.update_yaxes(title_text='Eficiência de Remoção', secondary_y=True)

        return fig

    @staticmethod
    def create_correlation_heatmap(df, columns=None):
        """
        Cria um mapa de calor de correlação entre os parâmetros

        Args:
            df: DataFrame com os dados
            columns: Lista de colunas para incluir na correlação

        Returns:
            fig: Objeto de figura do Plotly
        """
        if columns is None:
            # Excluindo a coluna de tempo
            columns = [col for col in df.columns if col != "Tempo (dias)"]

        # Calculando a matriz de correlação
        corr_matrix = df[columns].corr()

        # Criando o mapa de calor
        fig = px.imshow(
            corr_matrix,
            x=columns,
            y=columns,
            color_continuous_scale='RdBu_r',
            zmin=-1,
            zmax=1,
            title='Matriz de Correlação entre Parâmetros'
        )

        fig.update_layout(
            template='plotly_white'
        )

        return fig

    @staticmethod
    def create_removal_efficiency_bars(df, pollutant_columns):
        """
        Cria um gráfico de barras com as eficiências de remoção

        Args:
            df: DataFrame com os dados
            pollutant_columns: Lista de colunas de poluentes

        Returns:
            fig: Objeto de figura do Plotly
        """
        # Calculando eficiências de remoção
        eficiencias = {}

        for poluente in pollutant_columns:
            if poluente in df.columns:
                inicial = df[poluente].iloc[0]
                final = df[poluente].iloc[-1]
                eficiencia = 100 * (1 - final / inicial)
                eficiencias[poluente] = eficiencia

        # Criando gráfico de barras
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=list(eficiencias.keys()),
            y=list(eficiencias.values()),
            text=[f"{val:.2f}%" for val in eficiencias.values()],
            textposition='auto'
        ))

        fig.update_layout(
            title='Eficiência de Remoção Final',
            xaxis_title='Parâmetro',
            yaxis_title='Eficiência (%)',
            yaxis=dict(range=[0, 100]),
            template='plotly_white'
        )

        return fig

    @staticmethod
    def create_ml_evaluation_graph(y_true, y_pred, target_name):
        """
        Cria um gráfico de avaliação do modelo de machine learning

        Args:
            y_true: Valores reais
            y_pred: Valores previstos
            target_name: Nome do parâmetro alvo

        Returns:
            fig: Objeto de figura do Plotly
        """
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=y_true.flatten(),
            y=y_pred.flatten(),
            mode='markers',
            name='Teste'
        ))

        # Linha de referência (y=x)
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))

        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Ideal (y=x)',
            line=dict(dash='dash')
        ))

        fig.update_layout(
            title=f'Valores Reais vs. Previstos - {target_name}',
            xaxis_title='Valores Reais',
            yaxis_title='Valores Previstos',
            template='plotly_white'
        )

        return fig

    @staticmethod
    def create_feature_importance_graph(model, feature_names):
        """
        Cria um gráfico de importância das características para modelos de machine learning

        Args:
            model: Modelo treinado (com atributo feature_importances_)
            feature_names: Lista com os nomes das características

        Returns:
            fig: Objeto de figura do Plotly
        """
        # Obtendo importâncias
        importances = model.feature_importances_

        # Criando DataFrame
        feature_imp = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)

        # Criando gráfico
        fig = px.bar(
            feature_imp,
            y='Feature',
            x='Importance',
            orientation='h',
            title='Importância das Características'
        )

        fig.update_layout(
            xaxis_title='Importância',
            yaxis_title='Característica',
            template='plotly_white'
        )

        return fig

    @staticmethod
    def create_optimization_graph(param_range, response_values, param_name, response_name, optimal_point=None):
        """
        Cria um gráfico para visualização de otimização

        Args:
            param_range: Array com o intervalo do parâmetro
            response_values: Array com os valores de resposta
            param_name: Nome do parâmetro
            response_name: Nome da resposta
            optimal_point: Tupla (param_val, response_val) com o ponto ótimo

        Returns:
            fig: Objeto de figura do Plotly
        """
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=param_range,
            y=response_values,
            mode='lines',
            name='Resposta'
        ))

        if optimal_point is not None:
            fig.add_trace(go.Scatter(
                x=[optimal_point[0]],
                y=[optimal_point[1]],
                mode='markers',
                marker=dict(size=10, color='red'),
                name='Ponto ótimo'
            ))

        fig.update_layout(
            title=f'Otimização de {response_name} em função de {param_name}',
            xaxis_title=param_name,
            yaxis_title=response_name,
            template='plotly_white'
        )

        return fig