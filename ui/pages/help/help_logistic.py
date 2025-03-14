"""
Página de ajuda sobre o modelo Logístico
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def show_logistic_help():
    """
    Renderiza a página de ajuda sobre o modelo Logístico
    """
    st.markdown("<h1 class='main-header'>Modelo Logístico</h1>", unsafe_allow_html=True)

    # Introdução
    st.markdown("""
        <div class='info-box'>
        <h3>O que é o Modelo Logístico?</h3>
        <p>O Modelo Logístico descreve o crescimento de uma população microbiana em um ambiente com recursos limitados. O crescimento é inicialmente exponencial, mas diminui à medida que a população se aproxima da capacidade de suporte do ambiente.</p>
        </div>
    """, unsafe_allow_html=True)

    # Equação principal
    st.markdown("### Equação Principal")

    st.markdown(r"""
        A equação diferencial que descreve o modelo Logístico é:

        $$\frac{dX}{dt} = \mu_{max} \cdot X \cdot \left(1 - \frac{X}{X_{max}}\right)$$

        Cuja solução analítica é:

        $$X(t) = \frac{X_{max}}{1 + \left(\frac{X_{max}}{X_0} - 1\right) \cdot e^{-\mu_{max} \cdot t}}$$

        Onde:
        - $X$: Concentração de biomassa (g/L)
        - $X_0$: Concentração inicial de biomassa (g/L)
        - $X_{max}$: Concentração máxima de biomassa (g/L)
        - $\mu_{max}$: Taxa específica máxima de crescimento (dia$^{-1}$)
        - $t$: Tempo (dias)
    """)

    # Visualização
    st.markdown("### Visualização do Crescimento Logístico")

    # Configurações interativas
    st.markdown("Ajuste os parâmetros para ver como a curva de crescimento muda:")

    col1, col2, col3 = st.columns(3)

    with col1:
        umax = st.slider("μmax (Taxa máxima de crescimento)", 0.1, 1.0, 0.5, 0.1, key="logistic_umax")

    with col2:
        x0 = st.slider("X₀ (Biomassa inicial)", 0.01, 0.5, 0.1, 0.01)

    with col3:
        xmax = st.slider("Xmax (Capacidade de suporte)", 1.0, 10.0, 5.0, 0.5)

    # Criar dados para o gráfico
    t_values = np.linspace(0, 20, 100)

    # Função para calcular o crescimento logístico
    def logistic_growth(t, x0, umax, xmax):
        return xmax / (1 + (xmax / x0 - 1) * np.exp(-umax * t))

    # Calcular crescimento
    x_values = logistic_growth(t_values, x0, umax, xmax)

    # Criar DataFrame para Plotly
    df = pd.DataFrame({
        'Tempo (dias)': t_values,
        'Biomassa (g/L)': x_values
    })

    # Plotar com Plotly
    fig = px.line(df, x='Tempo (dias)', y='Biomassa (g/L)',
                  title=f'Crescimento Logístico (μmax = {umax}, X₀ = {x0}, Xmax = {xmax})')

    # Adicionar linha horizontal para Xmax
    fig.add_shape(
        type="line",
        x0=0, y0=xmax,
        x1=max(t_values), y1=xmax,
        line=dict(color="red", width=1, dash="dash"),
    )

    # Adicionar ponto para biomassa inicial
    fig.add_trace(go.Scatter(
        x=[0],
        y=[x0],
        mode="markers",
        marker=dict(color="green", size=10),
        name="Biomassa Inicial"
    ))

    # Adicionar anotações
    fig.add_annotation(
        x=max(t_values) * 0.9,
        y=xmax,
        text="Xmax",
        showarrow=False,
        yshift=10
    )

    # Destacar as fases do crescimento
    # Calculamos pontos aproximados para as diferentes fases
    t_lag_end = 1.0  # Aproximadamente
    t_exp_mid = t_lag_end + 2.0
    t_deceleration = min(t_values[np.where(x_values > 0.7 * xmax)[0][0]], max(t_values))
    t_stationary = min(t_values[np.where(x_values > 0.95 * xmax)[0][0]], max(t_values))

    # Adicionar anotações para as fases
    annotations = [
        dict(x=t_lag_end / 2, y=x0 * 1.2, text="Fase Lag", showarrow=False),
        dict(x=(t_lag_end + t_exp_mid) / 2, y=xmax * 0.3, text="Fase Exponencial", showarrow=False),
        dict(x=(t_exp_mid + t_deceleration) / 2, y=xmax * 0.75, text="Fase de Desaceleração", showarrow=False),
        dict(x=(t_deceleration + max(t_values)) / 2, y=xmax * 0.98, text="Fase Estacionária", showarrow=False)
    ]

    for annotation in annotations:
        fig.add_annotation(annotation)

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
        **Fases do crescimento logístico:**
        1. **Fase Lag**: Período inicial de adaptação com crescimento lento
        2. **Fase Exponencial**: Crescimento rápido, quase exponencial
        3. **Fase de Desaceleração**: A taxa de crescimento diminui à medida que os recursos se tornam limitados
        4. **Fase Estacionária**: A população se estabiliza próximo à capacidade de suporte (Xmax)
    """)

    # Parâmetros
    st.markdown("### Parâmetros do Modelo")

    st.markdown("""
        | Parâmetro | Descrição | Significado Biológico | Faixa Típica |
        |-----------|-----------|------------------------|--------------|
        | μmax | Taxa específica máxima de crescimento (dia⁻¹) | Taxa máxima de crescimento durante a fase exponencial | Microalgas: 0.1 - 1.0 dia⁻¹<br>Fungos: 0.05 - 0.5 dia⁻¹ |
        | X₀ | Concentração inicial de biomassa (g/L) | Inóculo inicial ou biomassa no tempo zero | 0.01 - 0.5 g/L |
        | Xmax | Concentração máxima de biomassa (g/L) | Capacidade de suporte do ambiente | Microalgas: 1.0 - 8.0 g/L<br>Fungos: 3.0 - 15.0 g/L |
    """)

    # Quando usar
    st.markdown("### Quando Usar o Modelo Logístico")

    st.markdown("""
        O modelo Logístico é mais adequado nas seguintes situações:

        - Quando você observa um crescimento microbiano que começa exponencialmente e depois reduz a velocidade até estabilizar
        - Quando você não tem dados detalhados do substrato limitante, apenas da biomassa
        - Para sistemas onde o fator limitante não é claramente definido ou é uma combinação de diversos fatores
        - Quando você precisa de um modelo mais simples com menos parâmetros

        **Vantagens:**
        - Mais simples de ajustar do que o modelo de Monod (requer menos dados)
        - Descreve bem o comportamento de crescimento com limitação de recursos
        - Possui solução analítica, facilitando cálculos e simulações

        **Limitações:**
        - Não relaciona diretamente o crescimento ao consumo de substrato
        - Não considera mecanismos específicos de limitação de crescimento
        - Menos detalhado que o modelo de Monod
    """)

    # Interpretação
    st.markdown("### Interpretação dos Parâmetros")

    st.markdown("""
        **Taxa específica máxima de crescimento (μmax):**
        - Um valor maior indica crescimento mais rápido na fase exponencial
        - Afeta diretamente a inclinação da curva na fase exponencial
        - Determina quanto tempo leva para a população atingir a capacidade de suporte

        **Concentração inicial de biomassa (X₀):**
        - Representa o inóculo inicial
        - Afeta principalmente o tempo necessário para entrar na fase exponencial
        - Valores muito baixos podem resultar em fase lag prolongada

        **Concentração máxima de biomassa (Xmax):**
        - Representa a capacidade de suporte do ambiente
        - Determinada pelos recursos disponíveis (nutrientes, luz, espaço)
        - Em tratamento de efluentes, pode ser limitada por nutrientes essenciais
    """)

    # Comparação com Monod
    st.markdown("### Comparação com o Modelo de Monod")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Modelo Logístico**")
        st.markdown("""
            - Depende apenas do tempo e da concentração de biomassa
            - Parâmetros: μmax, X₀, Xmax
            - Mais simples de ajustar
            - Não considera explicitamente o substrato
            - Ideal quando se tem apenas dados de biomassa
        """)

    with col2:
        st.markdown("**Modelo de Monod**")
        st.markdown("""
            - Relaciona crescimento à concentração de substrato
            - Parâmetros: μmax, Ks, Y
            - Mais complexo de ajustar
            - Considera explicitamente o substrato limitante
            - Requer dados tanto de biomassa quanto de substrato
        """)

    # Exemplos
    st.markdown("### Aplicações no Tratamento de Soro de Leite")

    st.markdown("""
        No tratamento de soro de leite por microalgas e fungos filamentosos, o modelo Logístico pode ser aplicado para:

        1. **Predição do crescimento da biomassa** - Estimar a produção de biomassa ao longo do tempo
        2. **Determinação da duração ótima do processo** - Identificar quando o sistema atinge a fase estacionária
        3. **Comparação entre condições de cultivo** - Avaliar como diferentes condições afetam os parâmetros de crescimento
        4. **Estimativa da capacidade de tratamento** - Relacionar a capacidade de suporte com a eficiência de remoção de poluentes

        O modelo Logístico é particularmente útil nas fases iniciais da pesquisa, quando se deseja uma caracterização simples do crescimento microbiano.
    """)

    # Referências
    st.markdown("### Referências")

    st.markdown("""
        1. Verhulst, P. F. (1838). Notice sur la loi que la population suit dans son accroissement. Corresp. Math. Phys. 10, 113–121.

        2. He, Y., Chen, L., Zhou, Y. et al. (2016). Analysis and model delineation of marine microalgae growth and lipid accumulation in flat-plate photobioreactor. Biochemical Engineering Journal, v. 111, p. 108-116.

        3. Veiter, L., Rajamanickam, V., & Herwig, C. (2018). The filamentous fungal pellet-relationship between morphology and productivity. Applied Microbiology and Biotechnology, v. 102, n. 7, p. 2997-3006.
    """)