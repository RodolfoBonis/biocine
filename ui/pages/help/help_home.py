"""
Página inicial de ajuda e documentação do BioCine
"""

import streamlit as st


def show_help_home():
    """
    Renderiza a página inicial de ajuda e documentação
    """
    st.markdown("<h1 class='main-header'>Ajuda e Documentação do BioCine</h1>", unsafe_allow_html=True)

    # Introdução
    st.markdown("""
        <div class='info-box'>
        <h3>Bem-vindo à Documentação do BioCine</h3>
        <p>BioCine é uma ferramenta especializada para modelagem cinética do processo de tratamento terciário em batelada/semicontínuo do soro do leite por microalgas e fungos filamentosos.</p>
        </div>
    """, unsafe_allow_html=True)

    # Visão geral
    st.markdown("### Visão Geral")

    st.markdown("""
        O BioCine permite:

        - Importar e analisar dados experimentais
        - Ajustar modelos cinéticos (Monod e Logístico)
        - Treinar modelos de Machine Learning para prever parâmetros do processo
        - Otimizar condições de processo para maximizar eficiência
        - Gerar relatórios e visualizações interativas
    """)

    # Tópicos de Ajuda
    st.markdown("### Tópicos de Ajuda")

    help_topics = {
        "Modelo de Monod": "Descrição do modelo, parâmetros, aplicações e interpretação",
        "Modelo Logístico": "Descrição do modelo, parâmetros, aplicações e interpretação",
        "Machine Learning": "Algoritmo Random Forest, parâmetros, treinamento e avaliação",
        "Parâmetros de Configuração": "Configurações disponíveis e valores recomendados",
        "Fluxo de Trabalho": "Processo completo de análise de dados no BioCine",
        "Exportação de Resultados": "Formatos disponíveis e usos recomendados"
    }

    col1, col2 = st.columns(2)

    for i, (topic, desc) in enumerate(help_topics.items()):
        if i % 2 == 0:
            with col1:
                st.markdown(f"**{topic}**")
                st.markdown(f"{desc}")
                st.markdown("---")
        else:
            with col2:
                st.markdown(f"**{topic}**")
                st.markdown(f"{desc}")
                st.markdown("---")

    # Como usar esta documentação
    st.markdown("### Como Usar Esta Documentação")

    st.markdown("""
        Esta documentação está disponível de três formas:

        1. **Menu Lateral**: Acesse esta seção de ajuda completa a qualquer momento.

        2. **Ajuda Contextual**: Clique nos botões de ajuda (ícones ⓘ) ao lado de elementos da interface para obter informações específicas.

        3. **Tooltips**: Passe o mouse sobre parâmetros e elementos da interface para ver dicas rápidas.
    """)

    # Referências
    st.markdown("### Referências")

    st.markdown("""
        1. **Modelo de Monod**:
           - Shuler, M. L., & Kargi, F. (2002). Bioprocess Engineering: Basic Concepts. 2nd ed. Prentice Hall.

        2. **Modelo Logístico**:
           - He, Y., Chen, L., Zhou, Y. et al. (2016). Analysis and model delineation of marine microalgae growth and lipid accumulation in flat-plate photobioreactor. Biochemical Engineering Journal, v. 111, p. 108-116.

        3. **Random Forest**:
           - Soares, A.P.M.R., Carvalho, F.O., De Farias Silva, C.E. (2020). Random Forest as a promising application to predict basic-dye biosorption process using orange waste. Journal of Environmental Chemical Engineering, v. 8, n. 4, 103952.

        4. **Tratamento de Soro de Leite**:
           - Li, Y., Miros, S., Kiani, H., et al. (2023). Mechanism of lactose assimilation in microalgae for the bioremediation of dairy processing side-streams and co-production of valuable food products. Journal of Applied Phycology, p. 1-13.
           - Veiter, L., Rajamanickam, V., & Herwig, C. (2018). The filamentous fungal pellet-relationship between morphology and productivity. Applied Microbiology and Biotechnology, v. 102, n. 7, p. 2997-3006.
    """)