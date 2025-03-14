"""
Página de resultados do BioCine

Esta página permite visualizar e exportar os resultados da modelagem
cinética e de machine learning.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import tempfile
from utils import (
    generate_model_summary,
    generate_ml_summary,
    generate_data_summary,
    export_report_to_html,
    export_results_to_excel
)

def show_results():
    """
    Renderiza a página de resultados
    """
    st.markdown("<h2 class='sub-header'>Resultados</h2>", unsafe_allow_html=True)

    # Verifica se há dados disponíveis
    if 'data' not in st.session_state or st.session_state.data is None:
        st.warning("Nenhum dado disponível. Por favor, vá para a seção 'Entrada de Dados' primeiro.")
        return

    # Verifica se há modelos ajustados
    has_cinetic_models = 'models' in st.session_state and st.session_state.models
    has_ml_models = 'ml_models' in st.session_state and st.session_state.ml_models

    if not has_cinetic_models and not has_ml_models:
        st.warning("Nenhum modelo ajustado. Por favor, ajuste modelos nas seções 'Modelagem Cinética' ou 'Machine Learning' primeiro.")
        return

    # Interface para visualização e exportação de resultados
    st.markdown("<div class='info-box'>Visualize e exporte os resultados da modelagem cinética e de machine learning.</div>", unsafe_allow_html=True)

    # Tabs para diferentes funcionalidades
    tab1, tab2, tab3 = st.tabs(["Resumo dos Resultados", "Visualização Avançada", "Exportação"])

    with tab1:
        st.markdown("### Resumo dos Resultados")

        # Resumo dos dados
        st.markdown("#### Dados Experimentais")

        data = st.session_state.data

        # Informações básicas
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**Número de Amostras:** {len(data)}")
            st.markdown(f"**Colunas:** {', '.join(data.columns)}")

        # Visualização dos dados
        st.dataframe(data.head(10))

        # Resumo dos modelos cinéticos
        if has_cinetic_models:
            st.markdown("#### Modelos Cinéticos")

            # Tabela com parâmetros dos modelos
            params_data = []

            for model_name, model_info in st.session_state.models.items():
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

            # Métricas de qualidade do ajuste
            metrics_data = []

            for model_name, model_info in st.session_state.models.items():
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

        # Resumo dos modelos de machine learning
        if has_ml_models:
            st.markdown("#### Modelos de Machine Learning")

            # Tabela com métricas dos modelos
            ml_metrics_data = []

            for model_name, model in st.session_state.ml_models.items():
                metrics = model.get_metrics()

                ml_metrics_data.append({
                    'Modelo': model_name,
                    'R² (Treino)': metrics['train_r2'],
                    'R² (Teste)': metrics['test_r2'],
                    'MSE (Treino)': metrics['train_mse'],
                    'MSE (Teste)': metrics['test_mse']
                })

            # Exibe a tabela
            ml_metrics_df = pd.DataFrame(ml_metrics_data)
            st.dataframe(ml_metrics_df)

    with tab2:
        st.markdown("### Visualização Avançada")

        # Seleciona o tipo de visualização
        viz_type = st.selectbox(
            "Selecione o tipo de visualização",
            ["Comparação de Modelos Cinéticos", "Análise de Resíduos", "Importância de Características"]
        )

        if viz_type == "Comparação de Modelos Cinéticos" and has_cinetic_models:
            st.markdown("#### Comparação de Modelos Cinéticos")

            # Seleciona os modelos para comparação
            model_names = list(st.session_state.models.keys())
            selected_models = st.multiselect(
                "Selecione os modelos para comparação",
                model_names,
                default=model_names
            )

            if not selected_models:
                st.warning("Selecione pelo menos um modelo para visualização.")
                return

            # Gráfico de comparação
            if 'tempo' in data.columns and 'biomassa' in data.columns:
                # Dados experimentais
                time_data = data['tempo'].values
                biomass_data = data['biomassa'].values

                # Criar figura interativa com plotly
                fig = go.Figure()

                # Dados experimentais
                fig.add_trace(
                    go.Scatter(
                        x=time_data,
                        y=biomass_data,
                        mode='markers',
                        name='Dados Experimentais',
                        marker=dict(color='black', size=8)
                    )
                )

                # Previsões dos modelos
                for model_name in selected_models:
                    model_info = st.session_state.models[model_name]

                    if 'predictions' in model_info:
                        predictions = model_info['predictions']

                        if isinstance(predictions, dict) and 'biomassa' in predictions:
                            # Para modelos com múltiplas saídas (como Monod)
                            fig.add_trace(
                                go.Scatter(
                                    x=time_data,
                                    y=predictions['biomassa'],
                                    mode='lines',
                                    name=f'{model_name} (Biomassa)'
                                )
                            )
                        else:
                            # Para modelos com uma saída (como Logístico)
                            fig.add_trace(
                                go.Scatter(
                                    x=time_data,
                                    y=predictions,
                                    mode='lines',
                                    name=model_name
                                )
                            )

                # Configurações do layout
                fig.update_layout(
                    title='Comparação de Modelos Cinéticos',
                    xaxis_title='Tempo (dias)',
                    yaxis_title='Biomassa (g/L)',
                    legend=dict(x=0.01, y=0.99),
                    hovermode='closest',
                    template='plotly_white'
                )

                st.plotly_chart(fig, use_container_width=True)

                # Gráfico de substratos, se disponíveis
                if 'substrato' in data.columns:
                    # Dados experimentais
                    substrate_data = data['substrato'].values

                    # Criar figura interativa com plotly
                    fig_substrate = go.Figure()

                    # Dados experimentais
                    fig_substrate.add_trace(
                        go.Scatter(
                            x=time_data,
                            y=substrate_data,
                            mode='markers',
                            name='Dados Experimentais',
                            marker=dict(color='black', size=8)
                        )
                    )

                    # Previsões dos modelos
                    for model_name in selected_models:
                        model_info = st.session_state.models[model_name]

                        if 'predictions' in model_info:
                            predictions = model_info['predictions']

                            if isinstance(predictions, dict) and 'substrato' in predictions:
                                # Para modelos com múltiplas saídas (como Monod)
                                fig_substrate.add_trace(
                                    go.Scatter(
                                        x=time_data,
                                        y=predictions['substrato'],
                                        mode='lines',
                                        name=f'{model_name} (Substrato)'
                                    )
                                )

                    # Configurações do layout
                    fig_substrate.update_layout(
                        title='Comparação de Modelos Cinéticos - Substrato',
                        xaxis_title='Tempo (dias)',
                        yaxis_title='Substrato (mg/L)',
                        legend=dict(x=0.01, y=0.99),
                        hovermode='closest',
                        template='plotly_white'
                    )

                    st.plotly_chart(fig_substrate, use_container_width=True)
            else:
                st.warning("Dados de tempo e biomassa não encontrados. Verifique os nomes das colunas ('tempo' e 'biomassa').")

        elif viz_type == "Análise de Resíduos" and has_cinetic_models:
            st.markdown("#### Análise de Resíduos")

            # Seleciona o modelo para análise
            model_name = st.selectbox(
                "Selecione o modelo para análise de resíduos",
                list(st.session_state.models.keys())
            )

            # Dados experimentais
            if 'tempo' in data.columns and 'biomassa' in data.columns:
                time_data = data['tempo'].values
                biomass_data = data['biomassa'].values

                # Previsões do modelo
                model_info = st.session_state.models[model_name]

                if 'predictions' in model_info:
                    predictions = model_info['predictions']

                    if isinstance(predictions, dict) and 'biomassa' in predictions:
                        # Para modelos com múltiplas saídas (como Monod)
                        biomass_pred = predictions['biomassa']
                    else:
                        # Para modelos com uma saída (como Logístico)
                        biomass_pred = predictions

                    # Calcula os resíduos
                    residuals = biomass_data - biomass_pred

                    # Cria dataframe para plotly
                    df_residuals = pd.DataFrame({
                        'Valores Preditos': biomass_pred,
                        'Resíduos': residuals
                    })

                    # Gráfico de resíduos interativo com plotly
                    fig = px.scatter(
                        df_residuals,
                        x='Valores Preditos',
                        y='Resíduos',
                        title='Resíduos vs. Valores Preditos'
                    )

                    # Adiciona linha de referência y=0
                    fig.add_hline(
                        y=0,
                        line_dash="dash",
                        line_color="red",
                        annotation_text="Resíduo Zero",
                        annotation_position="bottom right"
                    )

                    fig.update_layout(
                        xaxis_title='Valores Preditos',
                        yaxis_title='Resíduos'
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # QQ Plot interativo
                    from scipy import stats

                    # Cria dados para qq-plot
                    qq_data = []
                    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")

                    for i in range(len(osm)):
                        qq_data.append({
                            'Quantis Teóricos': osm[i],
                            'Quantis Ordenados': osr[i]
                        })

                    qq_df = pd.DataFrame(qq_data)

                    fig_qq = px.scatter(
                        qq_df,
                        x='Quantis Teóricos',
                        y='Quantis Ordenados',
                        title='Q-Q Plot dos Resíduos'
                    )

                    # Adiciona linha de ajuste
                    fig_qq.add_trace(
                        go.Scatter(
                            x=osm,
                            y=slope * osm + intercept,
                            mode='lines',
                            name='Linha de Referência',
                            line=dict(color='red', dash='dash')
                        )
                    )

                    fig_qq.update_layout(
                        xaxis_title='Quantis Teóricos',
                        yaxis_title='Quantis Ordenados'
                    )

                    st.plotly_chart(fig_qq, use_container_width=True)

                    # Estatísticas dos resíduos
                    st.markdown("#### Estatísticas dos Resíduos")

                    residuals_mean = np.mean(residuals)
                    residuals_std = np.std(residuals)
                    residuals_max = np.max(np.abs(residuals))

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Média", f"{residuals_mean:.4f}")

                    with col2:
                        st.metric("Desvio Padrão", f"{residuals_std:.4f}")

                    with col3:
                        st.metric("Máximo Absoluto", f"{residuals_max:.4f}")

                    # Teste de normalidade
                    shapiro_test = stats.shapiro(residuals)
                    st.markdown(f"**Teste de Shapiro-Wilk para Normalidade:** p-valor = {shapiro_test.pvalue:.4f}")

                    if shapiro_test.pvalue < 0.05:
                        st.warning("Os resíduos não parecem seguir uma distribuição normal (p < 0.05).")
                    else:
                        st.success("Os resíduos parecem seguir uma distribuição normal (p >= 0.05).")
            else:
                st.warning("Dados de tempo e biomassa não encontrados. Verifique os nomes das colunas ('tempo' e 'biomassa').")

        elif viz_type == "Importância de Características" and has_ml_models:
            st.markdown("#### Importância de Características")

            # Seleciona o modelo para visualização
            model_name = st.selectbox(
                "Selecione o modelo de ML",
                list(st.session_state.ml_models.keys())
            )

            # Obtém o modelo
            ml_model = st.session_state.ml_models[model_name]

            try:
                # Obtém a importância das características
                importance_df = ml_model.get_feature_importance()

                # Visualização interativa com plotly
                fig = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title=f'Importância das Características - {model_name}',
                    text='Importance',
                    labels={'Importance': 'Importância', 'Feature': 'Característica'}
                )

                fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
                fig.update_layout(xaxis_range=[0, importance_df['Importance'].max() * 1.1])

                st.plotly_chart(fig, use_container_width=True)

                # Exibe a tabela
                st.dataframe(importance_df)

            except Exception as e:
                st.error(f"Erro ao obter a importância das características: {str(e)}")

    with tab3:
        st.markdown("### Exportação de Resultados")

        # Seleciona o formato de exportação
        export_format = st.selectbox(
            "Selecione o formato de exportação",
            ["HTML", "Excel", "CSV"]
        )

        if export_format == "HTML":
            st.markdown("#### Exportação para HTML")

            # Opções de conteúdo
            st.markdown("Selecione o conteúdo para incluir no relatório:")

            include_data = st.checkbox("Dados Experimentais", value=True)
            include_cinetic = st.checkbox("Modelos Cinéticos", value=has_cinetic_models)
            include_ml = st.checkbox("Modelos de Machine Learning", value=has_ml_models)
            include_figures = st.checkbox("Figuras e Gráficos", value=True)

            # Botão para exportar
            if st.button("Gerar Relatório HTML"):
                try:
                    with st.spinner("Gerando relatório..."):
                        # Prepara os resumos
                        data_summary = generate_data_summary(data) if include_data else None

                        # Resumo dos modelos cinéticos
                        models_summary = {}
                        if include_cinetic and has_cinetic_models:
                            for model_name, model_info in st.session_state.models.items():
                                models_summary[model_name] = generate_model_summary(model_info['model'])

                        # Resumo dos modelos de ML
                        ml_summary = {}
                        if include_ml and has_ml_models:
                            for model_name, model in st.session_state.ml_models.items():
                                ml_summary[model_name] = generate_ml_summary(model)

                        # Figuras
                        figures = {}
                        if include_figures:
                            # Para manter compatibilidade com o relatório HTML,
                            # ainda precisamos de figuras Matplotlib
                            import matplotlib.pyplot as plt

                            # Adiciona figuras se necessário
                            if include_cinetic and has_cinetic_models and 'tempo' in data.columns and 'biomassa' in data.columns:
                                # Dados experimentais
                                time_data = data['tempo'].values
                                biomass_data = data['biomassa'].values

                                # Figura de comparação
                                fig, ax = plt.subplots(figsize=(10, 6))

                                # Dados experimentais
                                ax.scatter(time_data, biomass_data, color='black', label='Dados Experimentais')

                                # Previsões dos modelos
                                for model_name, model_info in st.session_state.models.items():
                                    if 'predictions' in model_info:
                                        predictions = model_info['predictions']

                                        if isinstance(predictions, dict) and 'biomassa' in predictions:
                                            # Para modelos com múltiplas saídas (como Monod)
                                            ax.plot(time_data, predictions['biomassa'], label=f'{model_name} (Biomassa)')
                                        else:
                                            # Para modelos com uma saída (como Logístico)
                                            ax.plot(time_data, predictions, label=model_name)

                                # Configurações do gráfico
                                ax.set_xlabel('Tempo (dias)')
                                ax.set_ylabel('Biomassa (g/L)')
                                ax.set_title('Comparação de Modelos Cinéticos')
                                ax.legend()
                                ax.grid(True, linestyle='--', alpha=0.7)

                                figures['comparacao_modelos'] = fig

                            if include_ml and has_ml_models:
                                # Adiciona figura de importância de características para o primeiro modelo
                                model_name = list(st.session_state.ml_models.keys())[0]
                                ml_model = st.session_state.ml_models[model_name]

                                try:
                                    importance_df = ml_model.get_feature_importance()

                                    fig, ax = plt.subplots(figsize=(10, 8))
                                    ax.barh(importance_df['Feature'], importance_df['Importance'])
                                    ax.set_xlabel('Importância')
                                    ax.set_title(f'Importância das Características - {model_name}')

                                    figures['importancia_caracteristicas'] = fig
                                except:
                                    pass

                        # Cria diretório temporário para o relatório
                        with tempfile.TemporaryDirectory() as temp_dir:
                            # Caminho para o relatório
                            report_path = os.path.join(temp_dir, "relatorio_biocine.html")

                            # Exporta o relatório
                            success = export_report_to_html(
                                models_summary,
                                ml_summary,
                                data_summary,
                                figures,
                                report_path
                            )

                            if success:
                                # Lê o arquivo HTML
                                with open(report_path, "r", encoding="utf-8") as f:
                                    html_content = f.read()

                                # Cria botão de download
                                st.download_button(
                                    label="Baixar Relatório HTML",
                                    data=html_content,
                                    file_name="relatorio_biocine.html",
                                    mime="text/html"
                                )
                            else:
                                st.error("Erro ao gerar o relatório HTML.")

                except Exception as e:
                    st.error(f"Erro ao exportar para HTML: {str(e)}")

        elif export_format == "Excel":
            st.markdown("#### Exportação para Excel")

            # Botão para exportar
            if st.button("Gerar Excel"):
                try:
                    with st.spinner("Gerando arquivo Excel..."):
                        # Cria diretório temporário
                        with tempfile.TemporaryDirectory() as temp_dir:
                            # Caminho para o arquivo Excel
                            excel_path = os.path.join(temp_dir, "resultados_biocine.xlsx")

                            # Prepara as previsões
                            predictions = {}
                            if has_cinetic_models:
                                for model_name, model_info in st.session_state.models.items():
                                    if 'predictions' in model_info:
                                        predictions[model_name] = model_info['predictions']

                            # Prepara os resumos dos modelos
                            models_summary = {}
                            if has_cinetic_models:
                                for model_name, model_info in st.session_state.models.items():
                                    models_summary[model_name] = generate_model_summary(model_info['model'])

                            # Prepara os resumos dos modelos de ML
                            ml_summary = {}
                            if has_ml_models:
                                for model_name, model in st.session_state.ml_models.items():
                                    ml_summary[model_name] = generate_ml_summary(model)

                            # Exporta para Excel
                            success = export_results_to_excel(
                                models_summary,
                                ml_summary,
                                data,
                                predictions,
                                excel_path
                            )

                            if success:
                                # Lê o arquivo Excel
                                with open(excel_path, "rb") as f:
                                    excel_data = f.read()

                                # Cria botão de download
                                st.download_button(
                                    label="Baixar Excel",
                                    data=excel_data,
                                    file_name="resultados_biocine.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                            else:
                                st.error("Erro ao gerar o arquivo Excel.")

                except Exception as e:
                    st.error(f"Erro ao exportar para Excel: {str(e)}")

        elif export_format == "CSV":
            st.markdown("#### Exportação para CSV")

            # Seleciona o conteúdo
            export_content = st.selectbox(
                "Selecione o conteúdo para exportar",
                ["Dados Experimentais", "Previsões dos Modelos Cinéticos", "Métricas dos Modelos"]
            )

            # Botão para exportar
            if st.button("Gerar CSV"):
                try:
                    if export_content == "Dados Experimentais":
                        # Exporta os dados experimentais
                        csv_data = data.to_csv(index=False)

                        st.download_button(
                            label="Baixar CSV",
                            data=csv_data,
                            file_name="dados_experimentais.csv",
                            mime="text/csv"
                        )

                    elif export_content == "Previsões dos Modelos Cinéticos" and has_cinetic_models:
                        # Cria um DataFrame com as previsões
                        predictions_df = pd.DataFrame()

                        # Adiciona o tempo
                        if 'tempo' in data.columns:
                            predictions_df['tempo'] = data['tempo']

                        # Adiciona os dados experimentais
                        if 'biomassa' in data.columns:
                            predictions_df['biomassa_experimental'] = data['biomassa']

                        if 'substrato' in data.columns:
                            predictions_df['substrato_experimental'] = data['substrato']

                        # Adiciona as previsões dos modelos
                        for model_name, model_info in st.session_state.models.items():
                            if 'predictions' in model_info:
                                predictions = model_info['predictions']

                                if isinstance(predictions, dict):
                                    # Para modelos com múltiplas saídas (como Monod)
                                    for key, values in predictions.items():
                                        predictions_df[f'{model_name}_{key}'] = values
                                else:
                                    # Para modelos com uma saída (como Logístico)
                                    predictions_df[f'{model_name}'] = predictions

                        # Exporta as previsões
                        csv_data = predictions_df.to_csv(index=False)

                        st.download_button(
                            label="Baixar CSV",
                            data=csv_data,
                            file_name="previsoes_modelos.csv",
                            mime="text/csv"
                        )

                    elif export_content == "Métricas dos Modelos":
                        # Cria um DataFrame para métricas dos modelos cinéticos
                        cinetic_metrics_data = []

                        if has_cinetic_models:
                            for model_name, model_info in st.session_state.models.items():
                                model = model_info['model']
                                metrics = model.get_metrics()

                                metrics_row = {'Modelo': model_name, 'Tipo': 'Cinético'}
                                metrics_row.update(metrics)

                                cinetic_metrics_data.append(metrics_row)

                        # Cria um DataFrame para métricas dos modelos de ML
                        ml_metrics_data = []

                        if has_ml_models:
                            for model_name, model in st.session_state.ml_models.items():
                                metrics = model.get_metrics()

                                metrics_row = {'Modelo': model_name, 'Tipo': 'Machine Learning'}
                                metrics_row.update(metrics)

                                ml_metrics_data.append(metrics_row)

                        # Combina as métricas
                        all_metrics = cinetic_metrics_data + ml_metrics_data

                        if all_metrics:
                            # Cria o DataFrame
                            metrics_df = pd.DataFrame(all_metrics)

                            # Exporta as métricas
                            csv_data = metrics_df.to_csv(index=False)

                            st.download_button(
                                label="Baixar CSV",
                                data=csv_data,
                                file_name="metricas_modelos.csv",
                                mime="text/csv"
                            )
                        else:
                            st.warning("Nenhuma métrica disponível para exportação.")

                except Exception as e:
                    st.error(f"Erro ao exportar para CSV: {str(e)}")