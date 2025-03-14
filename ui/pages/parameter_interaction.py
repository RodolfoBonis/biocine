"""
Interface completa para Análise de Interação entre Parâmetros

Este módulo implementa a interface Streamlit completa para a classe
ParameterInteractionAnalyzer, incluindo todas as visualizações e análises.
"""

import streamlit as st
import pandas as pd
import numpy as np
from models import ParameterInteractionAnalyzer


def show_parameter_interaction():
    """Página de Análise de Interação entre Parâmetros"""
    st.markdown("<h2 class='sub-header'>Análise de Interação entre Parâmetros</h2>", unsafe_allow_html=True)

    # Verifica se há dados disponíveis
    if 'data' not in st.session_state or st.session_state.data is None:
        st.warning("Nenhum dado disponível. Por favor, vá para a seção 'Entrada de Dados' primeiro.")
        return

    # Inicializa o analisador se não existir
    if 'parameter_analyzer' not in st.session_state:
        st.session_state.parameter_analyzer = ParameterInteractionAnalyzer()

    # Define dados
    st.session_state.parameter_analyzer.set_data(st.session_state.data)

    # Interface para seleção de características e alvo
    st.header("Configuração de Dados")

    # Seleção de características
    all_columns = list(st.session_state.data.columns)

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
    numeric_cols = st.session_state.data.select_dtypes(include=['float64', 'int64']).columns
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
        st.session_state.parameter_analyzer.set_features_target(selected_features, target_col)

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
                        interaction_matrix = st.session_state.parameter_analyzer.detect_feature_interactions(
                            method=method)

                        # Exibe mapa de calor
                        fig = st.session_state.parameter_analyzer.plot_interaction_heatmap()
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
                        import traceback
                        st.error(traceback.format_exc())

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
                        pca_results = st.session_state.parameter_analyzer.calculate_pca(
                            n_components=n_components,
                            scale=standardize
                        )

                        # Exibe biplot
                        fig = st.session_state.parameter_analyzer.plot_pca_biplot()
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
                        import traceback
                        st.error(traceback.format_exc())

        with tabs[2]:
            st.subheader("Análise de Dependência Parcial")

            if target_col is None or target_col == "Nenhum":
                st.warning("Selecione um alvo para calcular a dependência parcial.")
            else:
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
                    with st.spinner("Ajustando modelo e calculando dependência parcial..."):
                        try:
                            # Ajusta o modelo
                            st.session_state.parameter_analyzer.fit_model(model_type=model_type, **model_params)

                            # Calcula dependência parcial 1D
                            pdp_results = st.session_state.parameter_analyzer.calculate_partial_dependence()

                            # Plota dependência parcial 1D
                            fig = st.session_state.parameter_analyzer.plot_partial_dependence(pdp_results)
                            st.plotly_chart(fig, use_container_width=True)

                            # Calcula dependência parcial 2D para pares com maior interação
                            if hasattr(st.session_state.parameter_analyzer,
                                       'interaction_scores') and st.session_state.parameter_analyzer.interaction_scores is not None:
                                # Pares com maior interação
                                pairs = []
                                for i, feature1 in enumerate(selected_features):
                                    for j, feature2 in enumerate(selected_features):
                                        if i < j:  # Evita duplicações
                                            score = st.session_state.parameter_analyzer.interaction_scores.loc[
                                                feature1, feature2]
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
                            pdp_2d_results = st.session_state.parameter_analyzer.calculate_2d_partial_dependence(
                                feature_pairs)

                            # Plota dependência parcial 2D
                            pdp_2d_figs = st.session_state.parameter_analyzer.plot_2d_partial_dependence(pdp_2d_results)

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
                            fig = st.session_state.parameter_analyzer.create_parallel_coordinates()
                            st.plotly_chart(fig, use_container_width=True)

                            # Adiciona explicação
                            st.info("""
                            **Coordenadas Paralelas**: Cada linha representa uma amostra, e cada eixo vertical representa um parâmetro. 
                            Esta visualização é útil para identificar padrões entre múltiplas dimensões simultaneamente.
                            """)

                        elif viz_type == "Matriz de Dispersão":
                            # Limita dimensões para matriz não ficar muito grande
                            max_dims = min(6, len(selected_features))

                            # Seleciona características com maior variância
                            if len(selected_features) > max_dims:
                                # Calcula variância
                                var = st.session_state.data[selected_features].var()

                                # Ordena por variância
                                sorted_features = var.sort_values(ascending=False).index[:max_dims]
                            else:
                                sorted_features = selected_features

                            # Gera matriz de dispersão
                            fig = st.session_state.parameter_analyzer.create_high_dimensional_scatter(
                                dimensions=sorted_features)
                            st.plotly_chart(fig, use_container_width=True)

                            # Adiciona explicação
                            st.info("""
                            **Matriz de Dispersão**: Mostra gráficos de dispersão para cada par de parâmetros. 
                            A diagonal mostra a distribuição de cada parâmetro individualmente.
                            Útil para identificar relações entre pares de variáveis.
                            """)

                        elif viz_type == "Curvas de Andrews":
                            # Gera curvas de Andrews
                            fig = st.session_state.parameter_analyzer.create_andrews_curves()
                            st.plotly_chart(fig, use_container_width=True)

                            # Adiciona explicação
                            st.info("""
                            **Curvas de Andrews**: Representa cada amostra multidimensional como uma curva.
                            Amostras similares terão curvas similares, facilitando a identificação de grupos.
                            Uma ferramenta poderosa para detectar padrões em dados multidimensionais.
                            """)

                    except Exception as e:
                        st.error(f"Erro ao gerar visualização: {str(e)}")
                        import traceback
                        st.error(traceback.format_exc())
    else:
        st.warning("Selecione pelo menos um parâmetro para análise.")