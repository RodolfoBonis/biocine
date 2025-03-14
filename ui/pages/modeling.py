"""
Página de modelagem cinética do BioCine

Esta página permite ajustar modelos cinéticos aos dados experimentais e
visualizar os resultados do ajuste.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from models import ModelFactory
from utils import (
    plot_interactive_growth_curve,
    plot_interactive_combined
)

def show_modeling():
    """
    Renderiza a página de modelagem cinética
    """
    st.markdown("<h2 class='sub-header'>Modelagem Cinética</h2>", unsafe_allow_html=True)

    # Verifica se há dados disponíveis
    if 'data' not in st.session_state or st.session_state.data is None:
        st.warning("Nenhum dado disponível. Por favor, vá para a seção 'Entrada de Dados' primeiro.")
        return

    # Obtém os dados da sessão
    data = st.session_state.data

    # Verifica se as colunas necessárias estão presentes
    required_cols = ['tempo', 'biomassa']
    missing_cols = [col for col in required_cols if col not in data.columns]

    if missing_cols:
        st.error(f"Colunas necessárias ausentes: {', '.join(missing_cols)}. Por favor, verifique seus dados.")
        return

    # Interface para modelagem cinética
    st.markdown("<div class='info-box'>Ajuste modelos cinéticos aos seus dados experimentais e visualize os resultados.</div>", unsafe_allow_html=True)

    # Seleção de modelo
    model_type = st.selectbox(
        "Selecione o modelo cinético",
        ["Logístico", "Monod", "Comparar ambos"],
        index=0
    )

    # Parâmetros do modelo
    st.subheader("Parâmetros do Modelo")

    col1, col2 = st.columns(2)

    with col1:
        if model_type in ["Logístico", "Comparar ambos"]:
            st.markdown("#### Modelo Logístico")
            umax_logistic = st.slider("μmax (Taxa máxima de crescimento)", 0.01, 2.0, 0.5, 0.01)
            x0_logistic = st.slider("X0 (Concentração inicial de biomassa)", 0.01, 2.0, 0.1, 0.01)
            xmax_logistic = st.slider("Xmax (Concentração máxima de biomassa)", 1.0, 10.0, 5.0, 0.1)

    with col2:
        if model_type in ["Monod", "Comparar ambos"]:
            st.markdown("#### Modelo de Monod")
            umax_monod = st.slider("μmax (Taxa máxima de crescimento)", 0.01, 2.0, 0.5, 0.01, key="monod_umax")
            ks = st.slider("Ks (Constante de meia saturação)", 1.0, 200.0, 10.0, 1.0)
            y = st.slider("Y (Coeficiente de rendimento)", 0.01, 1.0, 0.5, 0.01)

            # Verificação para modelo de Monod
            if "substrato" not in data.columns:
                st.warning("Coluna 'substrato' não encontrada. O modelo de Monod requer dados de substrato.")
                if model_type == "Monod":  # Se apenas Monod for selecionado
                    return

    # Botão para ajustar modelo
    if st.button("Ajustar Modelo"):
        # Dados para ajuste
        time_data = data['tempo'].values
        biomass_data = data['biomassa'].values

        # Dicionário para armazenar os modelos ajustados
        models_fitted = {}

        try:
            with st.spinner("Ajustando modelo..."):
                # Container para os resultados
                results_container = st.container()

                if model_type in ["Logístico", "Comparar ambos"]:
                    # Cria o modelo Logístico
                    logistic_model = ModelFactory.create_cinetic_model(
                        'logistic',
                        umax=umax_logistic,
                        x0=x0_logistic,
                        xmax=xmax_logistic
                    )

                    # Ajusta o modelo
                    logistic_model.fit(time_data, biomass_data)

                    # Previsões
                    logistic_predictions = logistic_model.predict(time_data)

                    # Armazena o modelo
                    models_fitted['Logístico'] = {
                        'model': logistic_model,
                        'predictions': logistic_predictions
                    }

                if model_type in ["Monod", "Comparar ambos"] and "substrato" in data.columns:
                    # Dados para o modelo de Monod
                    substrate_data = data['substrato'].values

                    # Cria o modelo de Monod
                    monod_model = ModelFactory.create_cinetic_model(
                        'monod',
                        umax=umax_monod,
                        ks=ks,
                        y=y,
                        x0=biomass_data[0],
                        s0=substrate_data[0]
                    )

                    # Ajusta o modelo
                    monod_model.fit(
                        time_data,
                        {'biomassa': biomass_data, 'substrato': substrate_data}
                    )

                    # Previsões
                    monod_predictions = monod_model.predict(time_data)

                    # Armazena o modelo
                    models_fitted['Monod'] = {
                        'model': monod_model,
                        'predictions': monod_predictions
                    }

                # Salva os modelos na sessão
                st.session_state.models = models_fitted

                try:
                    # Dados experimentais
                    time_data = data['tempo'].values

                    # Cria um DataFrame com tempo e valores experimentais
                    pred_data = pd.DataFrame()
                    pred_data['tempo'] = time_data

                    if 'biomassa' in data.columns:
                        pred_data['biomassa_experimental'] = data['biomassa']

                    if 'substrato' in data.columns:
                        pred_data['substrato_experimental'] = data['substrato']

                    # Adiciona previsões de cada modelo
                    for model_name, model_info in models_fitted.items():
                        predictions = model_info['predictions']

                        if isinstance(predictions, dict):
                            for var_name, values in predictions.items():
                                pred_data[f'{model_name}_{var_name}'] = values
                        else:
                            pred_data[f'{model_name}_biomassa'] = predictions

                    # Salva os dados em um arquivo
                    import os
                    import yaml
                    import datetime

                    # Carrega configurações
                    config_path = os.path.join(
                        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "config", "config.yaml")

                    with open(config_path, "r") as f:
                        config = yaml.safe_load(f)

                    # Define o caminho
                    processed_dir = os.path.join(
                        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        config['paths']['processed_data'])

                    os.makedirs(processed_dir, exist_ok=True)

                    # Cria nome de arquivo
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    models_str = "_".join([m.lower() for m in models_fitted.keys()])

                    filename = f"model_predictions_{models_str}_{timestamp}.csv"
                    file_path = os.path.join(processed_dir, filename)

                    # Salva os dados
                    from utils.data_processor import save_processed_data
                    success = save_processed_data(pred_data, file_path)

                    if success:
                        st.info(f"Previsões do modelo salvas em: {file_path}")

                except Exception as e:
                    print(f"Erro ao salvar previsões: {str(e)}")  # Apenas log, não mostra para o usuário

                # Exibe os resultados
                with results_container:
                    st.subheader("Resultados do Ajuste")

                    # Tabs para diferentes visualizações
                    tab1, tab2, tab3 = st.tabs(["Curva de Crescimento", "Parâmetros", "Métricas"])

                    with tab1:
                        if model_type in ["Logístico", "Comparar ambos"]:
                            # Gráfico da curva de crescimento para o modelo Logístico
                            st.markdown("#### Modelo Logístico")

                            fig_logistic = plot_interactive_growth_curve(
                                time_data,
                                biomass_data,
                                models_fitted['Logístico']['predictions'],
                                title="Curva de Crescimento - Modelo Logístico"
                            )

                            st.plotly_chart(fig_logistic, use_container_width=True)

                        if model_type in ["Monod", "Comparar ambos"] and "substrato" in data.columns:
                            # Gráfico da curva de crescimento para o modelo de Monod
                            st.markdown("#### Modelo de Monod")

                            # Cria uma figura com múltiplos gráficos
                            monod_data = {
                                'Biomassa (g/L)': biomass_data,
                                'Substrato (mg/L)': substrate_data
                            }

                            monod_pred = {
                                'Biomassa (g/L)': monod_predictions['biomassa'],
                                'Substrato (mg/L)': monod_predictions['substrato']
                            }

                            fig_monod = plot_interactive_combined(
                                time_data,
                                monod_data,
                                monod_pred,
                                title="Modelo de Monod - Biomassa e Substrato"
                            )

                            st.plotly_chart(fig_monod, use_container_width=True)

                    with tab2:
                        # Parâmetros dos modelos
                        if models_fitted:
                            st.markdown("#### Parâmetros dos Modelos")

                            # Cria uma tabela com os parâmetros
                            params_data = []

                            for model_name, model_info in models_fitted.items():
                                model = model_info['model']
                                params = model.get_fitted_params() if model.get_fitted_params() else model.get_params()

                                for param_name, param_value in params.items():
                                    params_data.append({
                                        'Modelo': model_name,
                                        'Parâmetro': param_name,
                                        'Valor': param_value
                                    })

                            # Exibe a tabela
                            params_df = pd.DataFrame(params_data)
                            st.dataframe(params_df)

                    with tab3:
                        # Métricas de qualidade do ajuste
                        if models_fitted:
                            st.markdown("#### Métricas de Qualidade do Ajuste")

                            # Cria uma tabela com as métricas
                            metrics_data = []

                            for model_name, model_info in models_fitted.items():
                                model = model_info['model']
                                metrics = model.get_metrics()

                                for metric_name, metric_value in metrics.items():
                                    metrics_data.append({
                                        'Modelo': model_name,
                                        'Métrica': metric_name,
                                        'Valor': metric_value
                                    })

                            # Exibe a tabela
                            metrics_df = pd.DataFrame(metrics_data)
                            st.dataframe(metrics_df)

                            # Gráfico comparativo de métricas (usando Plotly)
                            if len(models_fitted) > 1:
                                st.markdown("#### Comparação de Métricas")

                                # Agrupa as métricas por modelo
                                models_metrics = {}

                                for model_name, model_info in models_fitted.items():
                                    models_metrics[model_name] = model_info['model'].get_metrics()

                                # Compara R²
                                r2_values = [metrics.get('r_squared', 0) for metrics in models_metrics.values()]

                                # Cria um DataFrame para o gráfico de barras
                                comparison_df = pd.DataFrame({
                                    'Modelo': list(models_metrics.keys()),
                                    'R²': r2_values
                                })

                                # Plotly bar chart
                                fig = px.bar(
                                    comparison_df,
                                    x='Modelo',
                                    y='R²',
                                    title='Comparação de R²',
                                    color='Modelo',
                                    text='R²',
                                    range_y=[0, 1]
                                )

                                fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')

                                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Erro ao ajustar o modelo: {str(e)}")

    # Simulação com modelos ajustados
    if 'models' in st.session_state and st.session_state.models:
        st.subheader("Simulação")

        st.markdown("<div class='info-box'>Simule o comportamento do sistema com diferentes condições iniciais.</div>", unsafe_allow_html=True)

        # Seleção de modelo
        model_names = list(st.session_state.models.keys())
        sim_model_name = st.selectbox("Selecione o modelo para simulação", model_names)

        # Modelo selecionado
        sim_model_info = st.session_state.models[sim_model_name]
        sim_model = sim_model_info['model']

        # Parâmetros para simulação
        st.markdown("#### Condições Iniciais")

        col1, col2 = st.columns(2)

        with col1:
            sim_time_max = st.slider("Tempo máximo (dias)", 1.0, 50.0, 20.0, 1.0)
            sim_points = st.slider("Número de pontos", 10, 200, 100, 10)

        with col2:
            # Parâmetros específicos para cada modelo
            if sim_model_name == "Logístico":
                sim_x0 = st.slider("X0 (Concentração inicial de biomassa)", 0.01, 2.0,
                                  float(sim_model.parameters['x0']), 0.01, key="sim_x0")
            elif sim_model_name == "Monod":
                sim_x0 = st.slider("X0 (Concentração inicial de biomassa)", 0.01, 2.0,
                                  float(sim_model.parameters['x0']), 0.01, key="sim_x0_monod")
                sim_s0 = st.slider("S0 (Concentração inicial de substrato)", 1.0, 200.0,
                                  float(sim_model.parameters['s0']), 1.0)

        # Botão para executar simulação
        if st.button("Executar Simulação"):
            # Tempo para simulação
            sim_time = np.linspace(0, sim_time_max, sim_points)

            try:
                # Atualiza parâmetros
                if sim_model_name == "Logístico":
                    sim_model.parameters['x0'] = sim_x0

                    # Previsões
                    sim_predictions = sim_model.predict(sim_time)

                    # Gráfico da simulação
                    fig = plot_interactive_growth_curve(
                        sim_time,
                        None,
                        sim_predictions,
                        title=f"Simulação - Modelo {sim_model_name}"
                    )

                    st.plotly_chart(fig, use_container_width=True)

                elif sim_model_name == "Monod":
                    sim_model.parameters['x0'] = sim_x0
                    sim_model.parameters['s0'] = sim_s0

                    # Previsões
                    sim_predictions = sim_model.predict(sim_time)

                    # Gráfico da simulação
                    sim_data = {
                        'Biomassa (g/L)': sim_predictions['biomassa'],
                        'Substrato (mg/L)': sim_predictions['substrato']
                    }

                    fig = plot_interactive_combined(
                        sim_time,
                        sim_data,
                        None,
                        title=f"Simulação - Modelo {sim_model_name}"
                    )

                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Erro na simulação: {str(e)}")