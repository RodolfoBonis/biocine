import numpy as np
import streamlit as st

from models import nonlinear_relationship_exploration, correlation_analysis


def show_correlation_analysis():
    st.title("Análise de Correlação Avançada")

    # Verifica se há dados disponíveis
    if 'data' not in st.session_state or st.session_state.data is None:
        st.warning("Nenhum dado disponível. Por favor, carregue dados primeiro.")
        return

    data = st.session_state.data

    # Configurações de análise
    st.subheader("Configurações de Análise")

    col1, col2 = st.columns(2)

    with col1:
        method = st.selectbox(
            "Método de Correlação",
            ["pearson", "spearman", "kendall"],
            index=0,
            help="Pearson: linear, Spearman: monotônica, Kendall: concordância de rankings"
        )

        alpha = st.slider(
            "Nível de Significância (α)",
            0.001, 0.1, 0.05, 0.001,
            help="Nível abaixo do qual a correlação é considerada estatisticamente significativa"
        )

    with col2:
        plot_type = st.selectbox(
            "Tipo de Visualização",
            ["heatmap", "pairplot", "network"],
            index=0,
            help="Formato de visualização das correlações"
        )

        include_tests = st.checkbox(
            "Incluir Testes Estatísticos",
            value=True,
            help="Calcula p-valores e realiza testes de significância"
        )

    # Seleção de variáveis para análise não-linear
    st.subheader("Análise de Relação Não-Linear")

    col1, col2 = st.columns(2)

    with col1:
        x_col = st.selectbox(
            "Variável X",
            data.select_dtypes(include=[np.number]).columns,
            index=0
        )

    with col2:
        y_col = st.selectbox(
            "Variável Y",
            data.select_dtypes(include=[np.number]).columns,
            index=min(1, len(data.select_dtypes(include=[np.number]).columns) - 1)
        )

    # Botão para executar análise
    if st.button("Executar Análise de Correlação"):
        with st.spinner("Realizando análise de correlação..."):
            try:
                # Análise de correlação principal
                results = correlation_analysis(
                    data,
                    method=method,
                    alpha=alpha,
                    include_tests=include_tests,
                    plot_type=plot_type
                )

                # Exibe heatmap ou gráfico selecionado
                if results['figure'] is not None:
                    st.plotly_chart(results['figure'], use_container_width=True)

                # Matriz de correlação
                st.subheader("Matriz de Correlação")
                st.dataframe(
                    results['correlation_matrix'].style.background_gradient(cmap='coolwarm', axis=None, vmin=-1,
                                                                            vmax=1))

                # Testes estatísticos
                if include_tests and results['p_values'] is not None:
                    st.subheader("Matriz de p-valores")
                    # Formatando com 4 casas decimais e destacando valores significativos
                    styled_pvals = results['p_values'].style.format("{:.4f}")
                    styled_pvals = styled_pvals.applymap(
                        lambda v: 'background-color: rgba(0,255,0,0.2)' if v < alpha else '')
                    st.dataframe(styled_pvals)

                # VIF e multicolinearidade
                st.subheader("Análise de Multicolinearidade (VIF)")
                st.dataframe(results['vif_analysis'])

                # Correlações parciais
                st.subheader("Correlações Parciais")
                st.dataframe(
                    results['partial_correlations'].style.background_gradient(cmap='coolwarm', axis=None, vmin=-1,
                                                                              vmax=1))

                # Mutual Information (para relações não-lineares)
                st.subheader("Informação Mútua (Relações Não-Lineares)")
                st.dataframe(results['mutual_information'].style.background_gradient(cmap='viridis', axis=None))

                # Análise de relação não-linear entre duas variáveis
                if x_col != y_col:
                    st.subheader(f"Análise de Relação entre {x_col} e {y_col}")
                    nonlinear_fig = nonlinear_relationship_exploration(data, x_col, y_col)
                    st.plotly_chart(nonlinear_fig, use_container_width=True)

            except Exception as e:
                st.error(f"Erro na análise de correlação: {str(e)}")
                import traceback
                st.error(traceback.format_exc())

