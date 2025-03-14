"""
Página inicial do aplicativo BioCine

Esta página apresenta uma visão geral do aplicativo e suas funcionalidades.
"""

import streamlit as st


def show_home():
    """
    Renderiza a página inicial
    """
    st.markdown("<h1 class='main-header'>BioCine: Modelagem Cinética de Bioprocessos</h1>", unsafe_allow_html=True)

    # Introdução
    st.markdown("""
        <div class='info-box'>
        <h3>Modelagem Cinética do Processo de Tratamento Terciário em Batelada/Semicontínuo do Soro do Leite por Microalgas e Fungos Filamentosos</h3>
        <p>BioCine é uma ferramenta para modelagem e análise de processos de tratamento biológico, com foco no tratamento do soro de leite utilizando microalgas e fungos filamentosos.</p>
        </div>
    """, unsafe_allow_html=True)

    # Seções principais
    st.subheader("Funcionalidades Principais")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Modelagem Cinética")
        st.markdown("""
            - **Modelo de Monod**: Crescimento microbiano limitado por substrato
            - **Modelo Logístico**: Crescimento com limitação de recursos
            - **Ajuste de Parâmetros**: Otimização de parâmetros a partir de dados experimentais
            - **Simulação**: Previsão do comportamento do sistema em diferentes condições
        """)

        st.markdown("#### Análise de Dados")
        st.markdown("""
            - **Importação de Dados**: Carregamento de dados experimentais em formato CSV
            - **Visualização**: Gráficos interativos dos dados e resultados dos modelos
            - **Eficiência de Remoção**: Cálculo da eficiência de remoção de poluentes
            - **Correlação**: Análise de correlação entre variáveis
        """)

    with col2:
        st.markdown("#### Machine Learning")
        st.markdown("""
            - **Random Forest**: Modelo para predição de variáveis-alvo
            - **Avaliação de Modelos**: Métricas de desempenho (R², MSE)
            - **Importância de Características**: Identificação das variáveis mais relevantes
            - **Otimização de Parâmetros**: Ajuste dos hiperparâmetros do modelo
        """)

        st.markdown("#### Relatórios")
        st.markdown("""
            - **Exportação de Resultados**: Geração de relatórios em HTML e Excel
            - **Resumo dos Modelos**: Parâmetros, métricas e previsões
            - **Visualizações**: Inclusão de gráficos e figuras
            - **Comparação de Modelos**: Análise comparativa de diferentes abordagens
        """)

    # Fluxo de trabalho
    st.subheader("Fluxo de Trabalho")

    st.markdown("""
        1. **Entrada de Dados**: Carregue seus dados experimentais ou utilize dados de exemplo
        2. **Modelagem Cinética**: Ajuste modelos cinéticos aos dados e analise os resultados
        3. **Machine Learning**: Treine modelos de ML para prever comportamentos futuros
        4. **Resultados**: Visualize e exporte os resultados da análise
    """)

    # Referências
    st.subheader("Referências")

    st.markdown("""
        - He, Y., Chen, L., Zhou, Y. et al. (2016). Analysis and model delineation of marine microalgae growth and lipid accumulation in flat-plate photobioreactor. Biochemical Engineering Journal, v. 111, p. 108-116.
        - Shuler, M. L., & Kargi, F. (2002). Bioprocess Engineering: Basic Concepts. 2nd ed. Prentice Hall.
        - Soares, A.P.M.R., Carvalho, F.O., De Farias Silva, C.E. (2020). Random Forest as a promising application to predict basic-dye biosorption process using orange waste. Journal of Environmental Chemical Engineering, v. 8, n. 4, 103952.
    """)

    # Nota de rodapé
    st.markdown("---")
    st.markdown("""
        <p style='text-align: center; color: #666;'>Desenvolvido como parte do projeto de pesquisa PIBIC, Ciclo 2024-2025</p>
    """, unsafe_allow_html=True)