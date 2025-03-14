"""
Página de Machine Learning do BioCine

Esta página permite treinar modelos de machine learning para prever
parâmetros do processo de tratamento biológico.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from models import ModelFactory
from utils import plot_interactive_growth_curve

def show_ml():
    """
    Renderiza a página de machine learning
    """
    st.markdown("<h2 class='sub-header'>Machine Learning</h2>", unsafe_allow_html=True)

    # Verifica se há dados disponíveis
    if 'data' not in st.session_state or st.session_state.data is None:
        st.warning("Nenhum dado disponível. Por favor, vá para a seção 'Entrada de Dados' primeiro.")
        return

    # Obtém os dados da sessão
    data = st.session_state.data

    # Interface para machine learning
    st.markdown("<div class='info-box'>Utilize algoritmos de machine learning para prever parâmetros do processo de tratamento.</div>", unsafe_allow_html=True)

    # Tabs para diferentes funcionalidades
    tab1, tab2 = st.tabs(["Previsão de Parâmetros", "Otimização"])

    with tab1:
        st.markdown("### Previsão de Parâmetros")

        # Seleção do alvo
        target_options = [col for col in data.columns if col != 'tempo']
        target_column = st.selectbox("Selecione o parâmetro a ser previsto", target_options)

        # Seleção das características
        feature_options = [col for col in data.columns if col != target_column]
        selected_features = st.multiselect(
            "Selecione as características para o modelo",
            feature_options,
            default=['tempo'] if 'tempo' in feature_options else []
        )

        if not selected_features:
            st.warning("Selecione pelo menos uma característica para o modelo.")
            return

        # Configuração do modelo
        st.markdown("#### Configuração do Modelo")

        # Seleção do tipo de modelo
        model_type = st.selectbox(
            "Selecione o tipo de modelo",
            ["Random Forest"],
            index=0
        )

        # Parâmetros do modelo Random Forest
        n_estimators = st.slider("Número de Árvores", 10, 200, 100, 10)
        max_depth = st.slider("Profundidade Máxima", 1, 20, 10, 1)

        # Proporção de teste
        test_size = st.slider("Proporção do Conjunto de Teste", 0.1, 0.5, 0.2, 0.05)

        # Botão para treinar modelo
        if st.button("Treinar Modelo"):
            try:
                with st.spinner("Treinando modelo..."):
                    # Cria o modelo
                    ml_model = ModelFactory.create_ml_model(
                        model_type.lower().replace(" ", "_"),
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_split=2,
                        min_samples_leaf=1,
                        random_state=42
                    )

                    # Pré-processa os dados
                    ml_model.preprocess_data(
                        data,
                        selected_features,
                        target_column,
                        test_size=test_size,
                        random_state=42
                    )

                    # Treina o modelo
                    ml_model.train()

                    # Avalia o modelo
                    metrics = ml_model.evaluate()

                    # Salva o modelo na sessão
                    if 'ml_models' not in st.session_state:
                        st.session_state.ml_models = {}

                    st.session_state.ml_models[target_column] = ml_model

                    # Exibe os resultados
                    st.subheader("Resultados do Treinamento")

                    # Métricas
                    st.markdown("#### Métricas de Avaliação")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("R² (Treino)", f"{metrics['train_r2']:.4f}")
                        st.metric("MSE (Treino)", f"{metrics['train_mse']:.4f}")

                    with col2:
                        st.metric("R² (Teste)", f"{metrics['test_r2']:.4f}")
                        st.metric("MSE (Teste)", f"{metrics['test_mse']:.4f}")

                    # Gráfico de importância de características
                    st.markdown("#### Importância das Características")

                    try:
                        importance_df = ml_model.get_feature_importance()

                        # Plotly horizontal bar chart para importância de características
                        fig = px.bar(
                            importance_df,
                            x='Importance',
                            y='Feature',
                            orientation='h',
                            title='Importância das Características',
                            text='Importance',
                            labels={'Importance': 'Importância', 'Feature': 'Característica'}
                        )

                        fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
                        fig.update_layout(xaxis_range=[0, importance_df['Importance'].max() * 1.1])

                        st.plotly_chart(fig, use_container_width=True)

                    except Exception as e:
                        st.warning(f"Não foi possível gerar o gráfico de importância: {str(e)}")

                    # Previsões vs. valores reais
                    st.markdown("#### Previsões vs. Valores Reais")

                    # Faz previsões no conjunto de teste
                    y_test_pred = ml_model.predict(ml_model.X_test)

                    # Cria o gráfico de dispersão com Plotly
                    scatter_data = pd.DataFrame({
                        'Valores Reais': ml_model.y_test.flatten(),
                        'Previsões': y_test_pred.flatten()
                    })

                    fig = px.scatter(
                        scatter_data,
                        x='Valores Reais',
                        y='Previsões',
                        title='Previsões vs. Valores Reais'
                    )

                    # Adiciona linha de referência (y=x)
                    min_val = min(scatter_data['Valores Reais'].min(), scatter_data['Previsões'].min())
                    max_val = max(scatter_data['Valores Reais'].max(), scatter_data['Previsões'].max())

                    fig.add_trace(
                        go.Scatter(
                            x=[min_val, max_val],
                            y=[min_val, max_val],
                            mode='lines',
                            name='y=x',
                            line=dict(color='red', dash='dash')
                        )
                    )

                    fig.update_layout(
                        xaxis_title='Valores Reais',
                        yaxis_title='Previsões'
                    )

                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Erro ao treinar o modelo: {str(e)}")

        # Interface de previsão com modelos existentes
        if 'ml_models' in st.session_state and st.session_state.ml_models:
            st.markdown("### Fazer Nova Previsão")

            # Seleciona o modelo para previsão
            available_models = list(st.session_state.ml_models.keys())
            model_for_prediction = st.selectbox(
                "Selecione o modelo para previsão",
                available_models
            )

            # Obtém o modelo selecionado
            selected_model = st.session_state.ml_models[model_for_prediction]

            # Interface para entrada de valores
            st.markdown("#### Valores para Previsão")

            # Cria inputs para cada característica
            feature_values = {}

            for feature in selected_model.features:
                # Determina os limites com base nos dados
                if feature in data.columns:
                    min_val = float(data[feature].min())
                    max_val = float(data[feature].max())
                    default_val = float(data[feature].mean())
                else:
                    min_val, max_val, default_val = 0, 100, 50

                # Cria o slider
                feature_values[feature] = st.slider(
                    feature,
                    min_val,
                    max_val,
                    default_val,
                    (max_val - min_val) / 100,
                    key=f"pred_{feature}"
                )

            # Botão para fazer previsão
            if st.button("Fazer Previsão"):
                # Prepara os dados de entrada
                input_data = np.array([feature_values[f] for f in selected_model.features]).reshape(1, -1)

                try:
                    # Faz a previsão
                    prediction = selected_model.predict(input_data)

                    # Trata diferentes formatos de retorno possíveis
                    if isinstance(prediction, np.ndarray):
                        if prediction.ndim == 2:
                            pred_value = prediction[0, 0]
                        elif prediction.ndim == 1:
                            pred_value = prediction[0]
                        else:
                            pred_value = prediction
                    else:
                        pred_value = prediction

                    # Exibe o resultado
                    st.success(f"Previsão: {float(pred_value):.4f}")

                    # Cria uma visualização da previsão
                    if 'tempo' in feature_values:
                        # Se 'tempo' é uma das características, podemos criar um gráfico de evolução
                        time_points = np.linspace(
                            data['tempo'].min(),
                            data['tempo'].max(),
                            100
                        )

                        predictions = []

                        # Faz previsões para diferentes pontos de tempo
                        for t in time_points:
                            # Copia os valores de entrada
                            temp_values = feature_values.copy()
                            # Atualiza o tempo
                            temp_values['tempo'] = t
                            # Prepara os dados
                            temp_input = np.array([temp_values[f] for f in selected_model.features]).reshape(1, -1)
                            # Faz a previsão
                            temp_pred = selected_model.predict(temp_input)

                            # Trata diferentes formatos de retorno possíveis
                            if isinstance(temp_pred, np.ndarray):
                                if temp_pred.ndim == 2:
                                    temp_value = temp_pred[0, 0]
                                elif temp_pred.ndim == 1:
                                    temp_value = temp_pred[0]
                                else:
                                    temp_value = temp_pred
                            else:
                                temp_value = temp_pred

                            # Armazena a previsão
                            predictions.append(float(temp_value))

                        # Cria o gráfico de evolução
                        fig = plot_interactive_growth_curve(
                            time_points,
                            None,
                            predictions,
                            title=f"Previsão de {model_for_prediction} ao longo do tempo"
                        )

                        st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Erro ao fazer previsão: {str(e)}")

    with tab2:
        st.markdown("### Otimização de Processo")

        # Verifica se há modelos treinados
        if 'ml_models' not in st.session_state or not st.session_state.ml_models:
            st.warning("Nenhum modelo de machine learning treinado. Treine modelos na seção 'Previsão de Parâmetros' primeiro.")
            return

        st.markdown("<div class='info-box'>Otimize os parâmetros do processo para maximizar a eficiência de remoção ou produção de biomassa.</div>", unsafe_allow_html=True)

        # Seleciona o objetivo
        objective = st.selectbox(
            "Selecione o objetivo da otimização",
            ["Maximizar produção de biomassa", "Maximizar remoção de poluentes"]
        )

        if objective == "Maximizar produção de biomassa":
            # Verifica se há um modelo para biomassa
            biomass_models = [k for k in st.session_state.ml_models.keys() if "biomassa" in k.lower()]

            if not biomass_models:
                st.warning("Nenhum modelo para biomassa encontrado. Treine um modelo para biomassa primeiro.")
                return

            # Seleciona o modelo
            biomass_model_name = st.selectbox(
                "Selecione o modelo de biomassa",
                biomass_models
            )

            # Obtém o modelo
            biomass_model = st.session_state.ml_models[biomass_model_name]

            # Parâmetros para otimização
            st.markdown("#### Parâmetros para Otimização")

            # Define os intervalos para cada parâmetro
            param_ranges = {}

            for feature in biomass_model.features:
                if feature == 'tempo':
                    continue  # O tempo será variado automaticamente

                # Determina os limites com base nos dados
                if feature in data.columns:
                    min_val = float(data[feature].min())
                    max_val = float(data[feature].max())
                    default_val = float(data[feature].mean())
                else:
                    min_val, max_val, default_val = 0, 100, 50

                # Cria sliders
                param_ranges[feature] = st.slider(
                    feature,
                    min_val,
                    max_val,
                    default_val,
                    (max_val - min_val) / 100,
                    key=f"opt_{feature}"
                )

            # Tempo máximo para simulação
            max_time = st.slider("Tempo Máximo (dias)", 1.0, 50.0, 20.0, 1.0)

            # Botão para executar otimização
            if st.button("Executar Otimização"):
                # Verifica se 'tempo' é uma característica do modelo
                if 'tempo' not in biomass_model.features:
                    st.error("O modelo não utiliza 'tempo' como característica, o que é necessário para otimização.")
                    return

                # Gera pontos de tempo
                time_points = np.linspace(0, max_time, 100)

                # Faz previsões para cada ponto de tempo
                predictions = []

                for t in time_points:
                    # Cria o vetor de características
                    features = []
                    for feature in biomass_model.features:
                        if feature == 'tempo':
                            features.append(t)
                        else:
                            features.append(param_ranges[feature])

                    # Faz a previsão
                    pred = biomass_model.predict(np.array(features).reshape(1, -1))
                    predictions.append(pred[0][0])

                # Encontra o ponto ótimo
                best_idx = np.argmax(predictions)
                best_time = time_points[best_idx]
                best_biomass = predictions[best_idx]

                # Exibe os resultados
                st.subheader("Resultados da Otimização")

                st.success(f"Tempo ótimo: {best_time:.2f} dias, Biomassa: {best_biomass:.4f} g/L")

                # Gráfico da evolução da biomassa
                fig = plot_interactive_growth_curve(
                    time_points,
                    None,
                    predictions,
                    title="Evolução da Biomassa ao Longo do Tempo"
                )

                st.plotly_chart(fig, use_container_width=True)

                # Parâmetros ótimos
                st.markdown("#### Parâmetros Ótimos")

                param_data = []

                for feature, value in param_ranges.items():
                    param_data.append({
                        'Parâmetro': feature,
                        'Valor': value
                    })

                param_data.append({
                    'Parâmetro': 'tempo',
                    'Valor': best_time
                })

                param_df = pd.DataFrame(param_data)
                st.dataframe(param_df)

        elif objective == "Maximizar remoção de poluentes":
            # Verifica se há modelos para poluentes
            pollutant_models = [k for k in st.session_state.ml_models.keys()
                              if any(p in k.lower() for p in ['nitrogenio', 'fosforo', 'dqo', 'concentracao_n', 'concentracao_p', 'concentracao_dqo'])]

            if not pollutant_models:
                st.warning("Nenhum modelo para poluentes encontrado. Treine um modelo para concentração de poluentes primeiro.")
                return

            # Seleciona o modelo
            pollutant_model_name = st.selectbox(
                "Selecione o modelo de poluente",
                pollutant_models
            )

            # Obtém o modelo
            pollutant_model = st.session_state.ml_models[pollutant_model_name]

            # Determina o tipo de poluente
            pollutant_type = "desconhecido"
            for p_type in ['nitrogenio', 'fosforo', 'dqo', 'concentracao_n', 'concentracao_p', 'concentracao_dqo']:
                if p_type in pollutant_model_name.lower():
                    pollutant_type = p_type
                    break

            # Parâmetros para otimização
            st.markdown("#### Parâmetros para Otimização")

            # Define os intervalos para cada parâmetro
            param_ranges = {}

            for feature in pollutant_model.features:
                if feature == 'tempo':
                    continue  # O tempo será variado automaticamente

                # Determina os limites com base nos dados
                if feature in data.columns:
                    min_val = float(data[feature].min())
                    max_val = float(data[feature].max())
                    default_val = float(data[feature].mean())
                else:
                    min_val, max_val, default_val = 0, 100, 50

                # Cria sliders
                param_ranges[feature] = st.slider(
                    feature,
                    min_val,
                    max_val,
                    default_val,
                    (max_val - min_val) / 100,
                    key=f"opt_{feature}"
                )

            # Tempo máximo para simulação
            max_time = st.slider("Tempo Máximo (dias)", 1.0, 50.0, 20.0, 1.0)

            # Obtem a concentração inicial
            if pollutant_model_name in data.columns:
                initial_concentration = data[pollutant_model_name].iloc[0]
            else:
                initial_concentration = 100  # Valor padrão

            # Botão para executar otimização
            if st.button("Executar Otimização"):
                # Verifica se 'tempo' é uma característica do modelo
                if 'tempo' not in pollutant_model.features:
                    st.error("O modelo não utiliza 'tempo' como característica, o que é necessário para otimização.")
                    return

                # Gera pontos de tempo
                time_points = np.linspace(0, max_time, 100)

                # Faz previsões para cada ponto de tempo
                predictions = []

                for t in time_points:
                    # Cria o vetor de características
                    features = []
                    for feature in pollutant_model.features:
                        if feature == 'tempo':
                            features.append(t)
                        else:
                            features.append(param_ranges[feature])

                    # Faz a previsão
                    pred = pollutant_model.predict(np.array(features).reshape(1, -1))
                    predictions.append(pred[0][0])

                # Calcula a eficiência de remoção
                removal_efficiency = [(initial_concentration - p) / initial_concentration * 100 for p in predictions]

                # Encontra o ponto ótimo
                best_idx = np.argmax(removal_efficiency)
                best_time = time_points[best_idx]
                best_concentration = predictions[best_idx]
                best_efficiency = removal_efficiency[best_idx]

                # Exibe os resultados
                st.subheader("Resultados da Otimização")

                st.success(f"Tempo ótimo: {best_time:.2f} dias, Concentração: {best_concentration:.4f}, Eficiência: {best_efficiency:.2f}%")

                # Criando dataframes para os gráficos
                df_conc = pd.DataFrame({
                    'Tempo (dias)': time_points,
                    'Concentração': predictions
                })

                df_eff = pd.DataFrame({
                    'Tempo (dias)': time_points,
                    'Eficiência de Remoção (%)': removal_efficiency
                })

                # Gráficos interativos com Plotly
                fig_conc = px.line(
                    df_conc,
                    x='Tempo (dias)',
                    y='Concentração',
                    title='Evolução da Concentração ao Longo do Tempo'
                )

                fig_eff = px.line(
                    df_eff,
                    x='Tempo (dias)',
                    y='Eficiência de Remoção (%)',
                    title='Eficiência de Remoção ao Longo do Tempo'
                )

                st.plotly_chart(fig_conc, use_container_width=True)
                st.plotly_chart(fig_eff, use_container_width=True)

                # Parâmetros ótimos
                st.markdown("#### Parâmetros Ótimos")

                param_data = []

                for feature, value in param_ranges.items():
                    param_data.append({
                        'Parâmetro': feature,
                        'Valor': value
                    })

                param_data.append({
                    'Parâmetro': 'tempo',
                    'Valor': best_time
                })

                param_df = pd.DataFrame(param_data)
                st.dataframe(param_df)