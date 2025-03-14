"""
Página de ajuda sobre o modelo de Monod
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def show_monod_help():
    """
    Renderiza a página de ajuda sobre o modelo de Monod
    """
    st.markdown("<h1 class='main-header'>Modelo de Monod</h1>", unsafe_allow_html=True)

    # Introdução
    st.markdown("""
        <div class='info-box'>
        <h3>O que é o Modelo de Monod?</h3>
        <p>O Modelo de Monod descreve a relação entre a taxa específica de crescimento de micro-organismos e a concentração do substrato limitante. É particularmente útil para modelar sistemas onde a taxa de crescimento é limitada pela disponibilidade de um nutriente.</p>
        </div>
    """, unsafe_allow_html=True)

    # Equações principais
    st.markdown("### Equações Principais")

    st.markdown(r"""
        As equações que compõem o modelo de Monod são:

        $$\mu = \mu_{max} \cdot \frac{S}{K_s + S}$$

        $$\frac{dX}{dt} = \mu \cdot X$$

        $$\frac{dS}{dt} = -\frac{1}{Y} \cdot \mu \cdot X$$

        Onde:
        - $\mu$: Taxa específica de crescimento (dia$^{-1}$)
        - $\mu_{max}$: Taxa específica máxima de crescimento (dia$^{-1}$)
        - $S$: Concentração de substrato (mg/L)
        - $K_s$: Constante de meia saturação (mg/L)
        - $X$: Concentração de biomassa (g/L)
        - $Y$: Coeficiente de rendimento (g biomassa/g substrato)
    """)

    # Visualização
    st.markdown("### Visualização da Relação de Monod")

    # Criar dados para o gráfico
    s_values = np.linspace(0, 100, 100)

    # Função para calcular a taxa específica de crescimento pelo modelo de Monod
    def monod_growth_rate(s, umax, ks):
        return umax * s / (ks + s)

    # Configurações interativas
    st.markdown("Ajuste os parâmetros para ver como a curva de Monod muda:")

    col1, col2 = st.columns(2)

    with col1:
        umax = st.slider("μmax (Taxa máxima de crescimento)", 0.1, 1.0, 0.5, 0.1)

    with col2:
        ks = st.slider("Ks (Constante de meia saturação)", 1.0, 50.0, 10.0, 1.0)

    # Calcular taxas de crescimento para diferentes valores de Ks
    u_values = monod_growth_rate(s_values, umax, ks)

    # Criar DataFrame para Plotly
    df = pd.DataFrame({
        'Concentração de Substrato (mg/L)': s_values,
        'Taxa Específica de Crescimento (dia⁻¹)': u_values
    })

    # Plotar com Plotly
    fig = px.line(df, x='Concentração de Substrato (mg/L)', y='Taxa Específica de Crescimento (dia⁻¹)',
                  title=f'Relação de Monod (μmax = {umax}, Ks = {ks})')

    # Adicionar linha horizontal para μmax
    fig.add_shape(
        type="line",
        x0=0, y0=umax,
        x1=max(s_values), y1=umax,
        line=dict(color="red", width=1, dash="dash"),
    )

    # Adicionar linha vertical para Ks
    fig.add_shape(
        type="line",
        x0=ks, y0=0,
        x1=ks, y1=umax / 2,
        line=dict(color="green", width=1, dash="dash"),
    )

    # Adicionar ponto para (Ks, μmax/2)
    fig.add_trace(go.Scatter(
        x=[ks],
        y=[umax / 2],
        mode="markers",
        marker=dict(color="green", size=10),
        name="Ponto Ks"
    ))

    # Adicionar anotações
    fig.add_annotation(
        x=max(s_values) * 0.9,
        y=umax,
        text="μmax",
        showarrow=False,
        yshift=10
    )

    fig.add_annotation(
        x=ks,
        y=0,
        text="Ks",
        showarrow=False,
        yshift=-15
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
        **Observações:**
        - Quando a concentração de substrato é muito maior que Ks (S >> Ks), a taxa de crescimento se aproxima de μmax
        - Quando S = Ks, a taxa de crescimento é exatamente metade do máximo (μ = μmax/2)
        - Menores valores de Ks indicam maior afinidade pelo substrato
    """)

    # Parâmetros
    st.markdown("### Parâmetros do Modelo")

    st.markdown("""
        | Parâmetro | Descrição | Significado Biológico | Faixa Típica |
        |-----------|-----------|------------------------|--------------|
        | μmax | Taxa específica máxima de crescimento (dia⁻¹) | Representa a máxima taxa de crescimento que o microrganismo pode atingir em condições ótimas | Microalgas: 0.1 - 1.0 dia⁻¹<br>Fungos: 0.05 - 0.5 dia⁻¹ |
        | Ks | Constante de meia saturação (mg/L) | Concentração de substrato na qual a taxa de crescimento é metade da máxima | 5 - 50 mg/L |
        | Y | Coeficiente de rendimento (g biomassa/g substrato) | Quantidade de biomassa produzida por unidade de substrato consumido | 0.1 - 0.7 |
    """)

    # Quando usar
    st.markdown("### Quando Usar o Modelo de Monod")

    st.markdown("""
        O modelo de Monod é mais adequado nas seguintes situações:

        - Quando o crescimento microbiano é claramente limitado por um substrato específico identificável
        - Quando você tem dados experimentais tanto de biomassa quanto do substrato limitante
        - Quando a taxa de crescimento específica não é constante, mas varia com a concentração de substrato
        - Para sistemas onde a diminuição da concentração de substrato correlaciona diretamente com o aumento da biomassa

        **Vantagens:**
        - Descreve o fenômeno físico de como o substrato limita o crescimento
        - Permite prever tanto o crescimento da biomassa quanto o consumo de substrato
        - Modelo bem estabelecido na literatura com muitos casos de uso

        **Limitações:**
        - Requer dados experimentais do substrato limitante
        - Não considera inibições ou múltiplos substratos limitantes
        - Mais complexo de ajustar do que o modelo Logístico
    """)

    # Interpretação
    st.markdown("### Interpretação dos Parâmetros")

    st.markdown("""
        **Taxa específica máxima de crescimento (μmax):**
        - Um valor maior indica um crescimento mais rápido em condições ótimas
        - Fatores que afetam: temperatura, pH, espécie do microrganismo, condições de cultivo
        - Para microalgas em tratamento de soro de leite, valores entre 0.2 e 0.6 dia⁻¹ são comuns

        **Constante de meia saturação (Ks):**
        - Um valor menor indica maior afinidade pelo substrato
        - Organismos com Ks menor são mais eficientes em baixas concentrações de substrato
        - Para nitrogênio em tratamento de soro de leite, valores entre 10 e 30 mg/L são típicos

        **Coeficiente de rendimento (Y):**
        - Um valor maior indica conversão mais eficiente de substrato em biomassa
        - Depende da via metabólica e eficiência do organismo
        - Organismos com maior Y são geralmente mais desejáveis para tratamento biológico
    """)

    # Exemplos
    st.markdown("### Aplicações no Tratamento de Soro de Leite")

    st.markdown("""
        No tratamento de soro de leite por microalgas e fungos filamentosos, o modelo de Monod pode ser aplicado para:

        1. **Predição de remoção de nutrientes** - Monitorar a taxa de remoção de nitrogênio e fósforo
        2. **Otimização de condições** - Determinar as melhores condições operacionais para maximizar a eficiência
        3. **Dimensionamento de reatores** - Calcular tempos de residência necessários para atingir determinada eficiência
        4. **Comparação entre espécies** - Avaliar qual espécie é mais eficiente para o tratamento

        Os parâmetros obtidos do modelo de Monod fornecem insights valiosos sobre a eficiência do processo de tratamento.
    """)

    # Referências específicas
    st.markdown("### Referências")

    st.markdown("""
        1. Monod, J. (1949). The growth of bacterial cultures. Annual Review of Microbiology, 3(1), 371-394.

        2. Shuler, M. L., & Kargi, F. (2002). Bioprocess Engineering: Basic Concepts. 2nd ed. Prentice Hall.

        3. Zhao, K., Wu, Y. W., Young, S. & Chen, X. J. (2020). Biological Treatment of Dairy Wastewater: A Mini Review. Journal of Environmental Informatics Letters, v. 4, p. 22-31.
    """)