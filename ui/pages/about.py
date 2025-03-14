"""
Página Sobre do BioCine

Esta página apresenta informações sobre o software, sua fundamentação
teórica e referências bibliográficas.
"""

import streamlit as st
from ui.pages.help import show_context_help


def show_about():
    """
    Renderiza a página Sobre
    """
    st.markdown("<h2 class='sub-header'>Sobre o BioCine</h2>", unsafe_allow_html=True)

    # Botão de ajuda
    col1, col2 = st.columns([10, 1])
    with col2:
        help_button = st.button("ⓘ", help="Mostrar ajuda sobre o BioCine")
        if help_button:
            st.session_state.show_help = True
            st.session_state.help_context = "about"
            return st.rerun()


    # Introdução
    st.markdown("""
        <div class='info-box'>
        <h3>BioCine: Software de Modelagem Cinética de Bioprocessos</h3>
        <p>BioCine é uma ferramenta desenvolvida para a modelagem cinética do processo de tratamento terciário em batelada/semicontínuo do soro do leite por microalgas e fungos filamentosos.</p>
        </div>
    """, unsafe_allow_html=True)

    # Informações do software
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### Fundamentação Teórica")

        st.markdown("""
            O BioCine se baseia em modelos cinéticos clássicos e técnicas de aprendizado de máquina para descrever e prever o comportamento de sistemas biológicos utilizados no tratamento de soro de leite.

            #### Modelagem Cinética

            A modelagem cinética é fundamental para descrever, prever e otimizar processos biológicos. Os principais modelos implementados são:

            **Modelo de Monod**

            Descreve a relação entre a taxa específica de crescimento dos micro-organismos e a concentração de substrato limitante:

            μ = μmax * S / (Ks + S)

            Onde:
            - μ: Taxa específica de crescimento (dia⁻¹)
            - μmax: Taxa específica máxima de crescimento (dia⁻¹)
            - S: Concentração de substrato (mg/L)
            - Ks: Constante de meia saturação (mg/L)

            **Modelo Logístico**

            Descreve o crescimento microbiano com limitação de recursos:

            dX/dt = μmax * X * (1 - X/Xmax)

            Onde:
            - X: Concentração de biomassa (g/L)
            - μmax: Taxa específica máxima de crescimento (dia⁻¹)
            - Xmax: Concentração máxima de biomassa (g/L)

            #### Machine Learning

            O aprendizado de máquina é utilizado para prever a eficiência de remoção de poluentes e o crescimento da biomassa com base em dados experimentais. O principal algoritmo implementado é o Random Forest, que permite identificar as características mais relevantes para o processo.
        """)

        st.markdown("### Equipe de Desenvolvimento")

        st.markdown("""
            **Pesquisadora:** Micaela Almeida Alves do Nascimento

            **Plano de Trabalho:** Modelagem Cinética do Processo de Tratamento Terciário em Batelada/Semicontínuo do Soro do Leite por Microalgas e Fungos Filamentosos

            **Orientação:** Laboratório de Bioprocessos

            **Ciclo:** 2024-2025
        """)

    with col2:
        st.markdown("### Versão do Software")

        st.markdown("""
            **BioCine:** 1.0.0

            **Data de Lançamento:** Março 2025

            **Licença:** MIT
        """)

        st.markdown("### Tecnologias Utilizadas")

        st.markdown("""
            - Python
            - Streamlit
            - NumPy
            - Pandas
            - SciPy
            - Scikit-learn
            - Matplotlib
            - Plotly
        """)

    # Referências
    st.markdown("### Referências Bibliográficas")

    st.markdown("""
        1. He, Y., Chen, L., Zhou, Y. et al. (2016). Analysis and model delineation of marine microalgae growth and lipid accumulation in flat-plate photobioreactor. Biochemical Engineering Journal, v. 111, p. 108-116.

        2. Shuler, M. L., & Kargi, F. (2002). Bioprocess Engineering: Basic Concepts. 2nd ed. Prentice Hall.

        3. Maltsev, Y., & Maltseva, K. (2021). Fatty acids of microalgae: Diversity and applications. Reviews in Environmental Science and Bio/Technology, v. 20, p. 515-547.

        4. Li, Y., Miros, S., Kiani, H., et al. (2023). Mechanism of lactose assimilation in microalgae for the bioremediation of dairy processing side-streams and co-production of valuable food products. Journal of Applied Phycology, p. 1-13.

        5. Veiter, L., Rajamanickam, V., & Herwig, C. (2018). The filamentous fungal pellet-relationship between morphology and productivity. Applied Microbiology and Biotechnology, v. 102, n. 7, p. 2997-3006.

        6. Zhao, K., Wu, Y. W., Young, S. & Chen, X. J. (2020). Biological Treatment of Dairy Wastewater: A Mini Review. Journal of Environmental Informatics Letters, v. 4, p. 22-31.

        7. Soares, A.P.M.R., Carvalho, F.O., & De Farias Silva, C.E. (2020). Random Forest as a promising application to predict basic-dye biosorption process using orange waste. Journal of Environmental Chemical Engineering, v. 8, n. 4, 103952.
    """)

    # Contato
    st.markdown("### Contato")

    st.markdown("""
        Para mais informações, sugestões ou relatos de problemas, entre em contato:

        **Email:** contato@exemplo.com

        **Website:** [www.exemplo.com](https://www.exemplo.com)

        **GitHub:** [github.com/biocine](https://github.com/biocine)
    """)

    # Nota de rodapé
    st.markdown("---")
    st.markdown("""
        <p style='text-align: center; color: #666;'>BioCine © 2025. Todos os direitos reservados.</p>
    """, unsafe_allow_html=True)