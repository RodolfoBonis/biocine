"""
Página de ajuda sobre o fluxo de trabalho do BioCine
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def show_workflow_help():
    """
    Renderiza a página de ajuda sobre o fluxo de trabalho
    """
    st.markdown("<h1 class='main-header'>Fluxo de Trabalho no BioCine</h1>", unsafe_allow_html=True)

    # Introdução
    st.markdown("""
        <div class='info-box'>
        <h3>Como utilizar o BioCine de forma eficiente</h3>
        <p>O BioCine segue um fluxo de trabalho estruturado para modelar e analisar o processo de tratamento terciário do soro do leite por microalgas e fungos filamentosos. Esta página explica como navegar pelo sistema e obter os melhores resultados.</p>
        </div>
    """, unsafe_allow_html=True)

    # Visão geral
    st.markdown("### Visão Geral do Processo")

    # Criando um diagrama do fluxo de trabalho com Plotly
    fig = go.Figure()

    # Definindo posições para os nós do diagrama
    nodes = {
        "Entrada de Dados": {"x": 0, "y": 0},
        "Modelagem Cinética": {"x": 0, "y": -1},
        "Machine Learning": {"x": 0, "y": -2},
        "Resultados": {"x": 0, "y": -3},
        "Importar CSV": {"x": -1, "y": 0},
        "Gerar Dados": {"x": 1, "y": 0},
        "Modelo Monod": {"x": -1, "y": -1},
        "Modelo Logístico": {"x": 1, "y": -1},
        "Previsão": {"x": -1, "y": -2},
        "Otimização": {"x": 1, "y": -2},
        "Visualização": {"x": -1, "y": -3},
        "Exportação": {"x": 1, "y": -3}
    }

    # Adicionando nós
    for node, pos in nodes.items():
        fig.add_trace(go.Scatter(
            x=[pos["x"]],
            y=[pos["y"]],
            mode="markers+text",
            marker=dict(size=25,
                        color="royalblue" if node in ["Entrada de Dados", "Modelagem Cinética", "Machine Learning",
                                                      "Resultados"] else "lightblue"),
            text=node,
            textposition="middle center",
            hoverinfo="text",
            name=node
        ))

    # Definindo as conexões
    connections = [
        ("Entrada de Dados", "Modelagem Cinética"),
        ("Modelagem Cinética", "Machine Learning"),
        ("Machine Learning", "Resultados"),
        ("Entrada de Dados", "Importar CSV"),
        ("Entrada de Dados", "Gerar Dados"),
        ("Modelagem Cinética", "Modelo Monod"),
        ("Modelagem Cinética", "Modelo Logístico"),
        ("Machine Learning", "Previsão"),
        ("Machine Learning", "Otimização"),
        ("Resultados", "Visualização"),
        ("Resultados", "Exportação")
    ]

    # Adicionando linhas de conexão
    for source, target in connections:
        x0, y0 = nodes[source]["x"], nodes[source]["y"]
        x1, y1 = nodes[target]["x"], nodes[target]["y"]

        fig.add_trace(go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            mode="lines",
            line=dict(width=2, color="gray"),
            hoverinfo="none",
            showlegend=False
        ))

    # Configurando o layout
    fig.update_layout(
        title="Fluxo de Trabalho no BioCine",
        showlegend=False,
        height=500,
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        plot_bgcolor="white"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Explicação passo a passo
    st.markdown("### Passo a Passo")

    # Etapa 1: Entrada de Dados
    st.markdown("#### 1. Entrada de Dados")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
            **1.1. Importar dados experimentais (CSV)**
            - Acesse a página "Entrada de Dados"
            - Selecione a guia "Importar CSV"
            - Faça upload do arquivo CSV com seus dados
            - Configure os delimitadores e separadores decimais
            - Mapeie as colunas necessárias (tempo, biomassa, substrato)
            - Selecione as opções de pré-processamento
            - Clique em "Processar Dados"
        """)

    with col2:
        st.markdown("""
            **1.2. Gerar dados de exemplo (opcional)**
            - Acesse a página "Entrada de Dados"
            - Selecione a guia "Gerar Dados de Exemplo"
            - Ajuste o número de amostras e a semente aleatória
            - Clique em "Gerar Dados de Exemplo"

            **1.3. Visualizar os dados**
            - Selecione a guia "Visualizar Dados"
            - Explore os dados importados ou gerados
            - Visualize estatísticas descritivas e gráficos básicos
        """)

    st.markdown("""
        **Requisitos dos dados:**
        - Coluna de tempo (dias)
        - Coluna de biomassa (g/L)
        - Coluna de substrato (mg/L) - opcional, mas necessária para o modelo de Monod
        - Outras colunas desejáveis: temperatura, pH, concentrações de N, P, DQO

        **Formato recomendado do CSV:**
        ```
        tempo,biomassa,substrato,temperatura,ph,concentracao_n,concentracao_p,concentracao_dqo
        0,0.1,100,25,7.0,50,10,1000
        1,0.15,95,25,7.0,48,9.8,980
        2,0.22,89,25,7.0,45,9.5,950
        ...
        ```
    """)

    # Etapa 2: Modelagem Cinética
    st.markdown("#### 2. Modelagem Cinética")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
            **2.1. Selecionar modelo**
            - Acesse a página "Modelagem Cinética"
            - Escolha o modelo desejado (Logístico, Monod ou ambos)
            - Ajuste os parâmetros iniciais conforme necessário
            - Clique em "Ajustar Modelo"

            **2.2. Analisar resultados**
            - Visualize as curvas de crescimento/consumo de substrato
            - Avalie os parâmetros ajustados
            - Verifique as métricas de qualidade (R², MSE)
        """)

    with col2:
        st.markdown("""
            **2.3. Simulação (opcional)**
            - Selecione o modelo para simulação
            - Defina o tempo máximo e o número de pontos
            - Ajuste as condições iniciais (X0, S0)
            - Clique em "Executar Simulação"
            - Visualize os resultados da simulação

            **Quando usar cada modelo:**
            - **Monod**: Quando você tem dados de substrato e biomassa
            - **Logístico**: Quando você tem apenas dados de biomassa
        """)

    # Exemplo visual da modelagem cinética
    # Dados simulados para visualização
    time = np.linspace(0, 10, 20)
    biomass_exp = 5 / (1 + (5 / 0.1 - 1) * np.exp(-0.5 * time)) + np.random.normal(0, 0.2, 20)
    biomass_model = 5 / (1 + (5 / 0.1 - 1) * np.exp(-0.5 * time))

    # Criar DataFrame para Plotly
    df_model = pd.DataFrame({
        'Tempo (dias)': time,
        'Biomassa Experimental (g/L)': biomass_exp,
        'Biomassa Modelo (g/L)': biomass_model
    })

    # Plotar com Plotly
    fig_model = px.scatter(
        df_model,
        x='Tempo (dias)',
        y='Biomassa Experimental (g/L)',
        title='Exemplo de Ajuste de Modelo Cinético'
    )

    fig_model.add_trace(
        go.Scatter(
            x=df_model['Tempo (dias)'],
            y=df_model['Biomassa Modelo (g/L)'],
            mode='lines',
            name='Modelo Logístico',
            line=dict(color='red')
        )
    )

    st.plotly_chart(fig_model, use_container_width=True)

    # Etapa 3: Machine Learning
    st.markdown("#### 3. Machine Learning")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
            **3.1. Previsão de parâmetros**
            - Acesse a página "Machine Learning"
            - Selecione a guia "Previsão de Parâmetros"
            - Escolha o parâmetro alvo (ex: biomassa, eficiência de remoção)
            - Selecione as características para o modelo
            - Configure os parâmetros do Random Forest
            - Clique em "Treinar Modelo"
            - Analise os resultados do treinamento
            - Faça novas previsões com diferentes entradas
        """)

    with col2:
        st.markdown("""
            **3.2. Otimização de processo**
            - Selecione a guia "Otimização"
            - Escolha o objetivo (maximizar biomassa ou remoção)
            - Selecione o modelo treinado
            - Ajuste os parâmetros do processo
            - Defina o tempo máximo para simulação
            - Clique em "Executar Otimização"
            - Analise os resultados da otimização
            - Identifique as condições ótimas
        """)

    # Exemplo visual da importância das características
    # Dados simulados
    features = ['Tempo', 'Temperatura', 'pH', 'Conc. Inicial N', 'Conc. Inicial P']
    importance = [0.45, 0.25, 0.15, 0.10, 0.05]

    # Criar DataFrame para Plotly
    df_imp = pd.DataFrame({
        'Característica': features,
        'Importância': importance
    })

    # Plotar com Plotly
    fig_imp = px.bar(
        df_imp,
        x='Importância',
        y='Característica',
        orientation='h',
        title='Exemplo de Importância das Características',
        text='Importância'
    )

    fig_imp.update_traces(texttemplate='%{text:.2f}', textposition='outside')

    st.plotly_chart(fig_imp, use_container_width=True)

    # Etapa 4: Resultados
    st.markdown("#### 4. Resultados")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
            **4.1. Resumo dos Resultados**
            - Acesse a página "Resultados"
            - Visualize o resumo geral dos dados e modelos
            - Analise os parâmetros e métricas dos modelos cinéticos
            - Verifique o desempenho dos modelos de machine learning

            **4.2. Visualização Avançada**
            - Selecione a guia "Visualização Avançada"
            - Escolha o tipo de visualização:
              - Comparação de Modelos Cinéticos
              - Análise de Resíduos
              - Importância de Características
            - Explore os gráficos interativos
        """)

    with col2:
        st.markdown("""
            **4.3. Exportação**
            - Selecione a guia "Exportação"
            - Escolha o formato de exportação:
              - HTML: Relatório completo com figuras
              - Excel: Planilha estruturada com múltiplas abas
              - CSV: Formato simples para análise posterior
            - Selecione o conteúdo a incluir
            - Clique no botão de exportação
            - Baixe o arquivo gerado

            **Recomendação:** O formato HTML é ideal para relatórios completos, o Excel para análise posterior em planilhas, e o CSV para importação em outros softwares estatísticos.
        """)

    # Exemplo visual de um relatório
    st.markdown("#### Exemplo de Relatório HTML Gerado")

    st.image("https://raw.githubusercontent.com/streamlit/demo-self-driving/master/streamlit_app_final.png",
             caption="Exemplo ilustrativo de um relatório (imagem genérica)")

    # Dicas e boas práticas
    st.markdown("### Dicas e Boas Práticas")

    st.markdown("""
        **1. Preparação de dados**
        - Verifique a qualidade dos dados antes de importá-los
        - Remova outliers que possam afetar o ajuste dos modelos
        - Tenha pelo menos 10-15 pontos para um bom ajuste de modelos cinéticos
        - Para machine learning, quanto mais dados, melhor (idealmente > 30 pontos)

        **2. Modelagem Cinética**
        - Comece com o modelo Logístico se não tiver dados de substrato
        - Use o modelo de Monod quando quiser relacionar crescimento e consumo de substrato
        - Compare os resultados de diferentes modelos
        - Verifique se os parâmetros ajustados fazem sentido biologicamente

        **3. Machine Learning**
        - Selecione características relevantes para o processo
        - Evite usar muitas características com poucos dados (risco de overfitting)
        - Verifique o desempenho no conjunto de teste (não apenas no treino)
        - Use a análise de importância para identificar os fatores mais relevantes

        **4. Interpretação**
        - Avalie os resultados à luz do conhecimento biológico do processo
        - Considere a reprodutibilidade experimental ao interpretar pequenas diferenças
        - Valide previsões do modelo com novos experimentos quando possível
    """)

    # Fluxo de trabalho recomendado
    st.markdown("### Fluxo de Trabalho Recomendado")

    # Criando um diagrama de fluxo de trabalho recomendado
    steps = ['Importar/Preparar Dados', 'Ajustar Modelo Logístico', 'Ajustar Modelo de Monod (se possível)',
             'Comparar Modelos', 'Treinar Modelo ML', 'Otimizar Parâmetros', 'Exportar Resultados']

    step_desc = [
        "Importe seus dados experimentais e verifique a qualidade",
        "Comece com o modelo mais simples que requer apenas dados de biomassa",
        "Se tiver dados de substrato, ajuste o modelo de Monod",
        "Compare os modelos e escolha o mais adequado",
        "Treine um modelo Random Forest para prever parâmetros de interesse",
        "Use o modelo para identificar condições ótimas",
        "Exporte um relatório completo com suas descobertas"
    ]

    # Criando uma figura com múltiplos subplots
    fig = make_subplots(rows=len(steps), cols=1, subplot_titles=steps, vertical_spacing=0.05)

    # Adicionando etapas como barras horizontais
    for i, (step, desc) in enumerate(zip(steps, step_desc)):
        fig.add_trace(
            go.Bar(
                x=[1],
                y=[step],
                orientation='h',
                text=[desc],
                textposition='inside',
                marker=dict(color='royalblue'),
                hoverinfo='text',
                showlegend=False
            ),
            row=i + 1, col=1
        )

    # Configurando o layout
    fig.update_layout(
        title='Fluxo de Trabalho Recomendado',
        height=700,
        margin=dict(l=200, r=20, t=50, b=20),
        yaxis=dict(autorange="reversed")
    )

    # Atualizando todos os eixos
    for i in range(1, len(steps) + 1):
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=i, col=1)
        fig.update_yaxes(showticklabels=True, showgrid=False, zeroline=False, row=i, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # Exemplos práticos
    st.markdown("### Exemplos Práticos")

    example_tabs = st.tabs([
        "Microalgas - Remoção de N e P",
        "Fungos - Produção de Biomassa",
        "Sistema Misto - Otimização"
    ])

    with example_tabs[0]:
        st.markdown("#### Exemplo 1: Microalgas para Remoção de Nitrogênio e Fósforo")

        st.markdown("""
            **Objetivo:** Modelar e otimizar o processo de remoção de nitrogênio e fósforo do soro de leite usando microalgas _Chlorella vulgaris_.

            **Passo 1:** Importe os dados experimentais contendo tempo, biomassa, concentrações de N e P, temperatura e pH.

            **Passo 2:** Ajuste o modelo Logístico para descrever o crescimento da biomassa.

            **Passo 3:** Treine modelos de ML separados para prever a remoção de N e P com base no tempo, biomassa, temperatura e pH.

            **Passo 4:** Use a otimização para determinar as condições ideais (temperatura, pH, tempo de retenção) que maximizam a remoção de ambos os nutrientes.

            **Passo 5:** Exporte um relatório HTML completo com os resultados e recomendações.

            **Resultado esperado:** Identificação de condições em que a eficiência de remoção de N e P seja > 80%, com parâmetros operacionais viáveis.
        """)

    with example_tabs[1]:
        st.markdown("#### Exemplo 2: Fungos Filamentosos para Produção de Biomassa")

        st.markdown("""
            **Objetivo:** Maximizar a produção de biomassa fúngica a partir do soro de leite para aplicações biotecnológicas.

            **Passo 1:** Importe os dados experimentais contendo tempo, biomassa, DQO, temperatura e pH.

            **Passo 2:** Ajuste o modelo de Monod considerando a DQO como substrato limitante.

            **Passo 3:** Treine um modelo de ML para prever a produção de biomassa em função do tempo, DQO inicial, temperatura e pH.

            **Passo 4:** Use a otimização para determinar as condições que maximizam a produção de biomassa.

            **Passo 5:** Exporte os resultados em formato Excel para análise detalhada.

            **Resultado esperado:** Determinação de condições ótimas para produção de biomassa fúngica > 5 g/L com consumo eficiente de DQO.
        """)

    with example_tabs[2]:
        st.markdown("#### Exemplo 3: Sistema Misto para Tratamento Otimizado")

        st.markdown("""
            **Objetivo:** Otimizar um sistema misto de microalgas e fungos filamentosos para tratamento completo do soro de leite.

            **Passo 1:** Importe dados de experimentos com culturas mistas, incluindo tempo, biomassa, N, P, DQO, temperatura e pH.

            **Passo 2:** Ajuste modelos Logísticos separados para crescimento de microalgas e fungos.

            **Passo 3:** Treine modelos de ML para prever remoção de N, P e DQO no sistema misto.

            **Passo 4:** Execute a otimização para encontrar a proporção ideal de microalgas:fungos e condições operacionais.

            **Passo 5:** Exporte um relatório completo com as análises e recomendações.

            **Resultado esperado:** Determinação da proporção ótima de inóculo e condições que maximizam a eficiência global de tratamento, com remoção balanceada de N, P e DQO.
        """)

    # Resolução de problemas
    st.markdown("### Resolução de Problemas Comuns")

    problems = pd.DataFrame({
        'Problema': [
            'O modelo não converge',
            'R² muito baixo',
            'Parâmetros não realistas',
            'Erro ao importar CSV',
            'Overfitting no modelo de ML',
            'Resultados da otimização inconsistentes'
        ],
        'Possível Causa': [
            'Dados insuficientes ou com ruído excessivo',
            'Modelo inadequado para o fenômeno observado',
            'Restrições inadequadas nos parâmetros',
            'Formato incorreto ou delimitador errado',
            'Muitas características para poucos dados',
            'Espaço de parâmetros muito amplo'
        ],
        'Solução': [
            'Aumente o número de pontos experimentais ou filtre outliers',
            'Tente um modelo diferente ou modifique as restrições',
            'Ajuste os limites dos parâmetros com base no conhecimento biológico',
            'Verifique o formato e use o delimitador correto (vírgula ou ponto-e-vírgula)',
            'Reduza o número de características ou aumente a restrição de profundidade',
            'Restrinja o espaço de busca com valores mais próximos do esperado'
        ]
    })

    st.dataframe(problems)

    # Referências
    st.markdown("### Referências")

    st.markdown("""
        1. Shuler, M. L., & Kargi, F. (2002). Bioprocess Engineering: Basic Concepts. 2nd ed. Prentice Hall.

        2. He, Y., Chen, L., Zhou, Y. et al. (2016). Analysis and model delineation of marine microalgae growth and lipid accumulation in flat-plate photobioreactor. Biochemical Engineering Journal, v. 111, p. 108-116.

        3. Li, Y., Miros, S., Kiani, H., et al. (2023). Mechanism of lactose assimilation in microalgae for the bioremediation of dairy processing side-streams and co-production of valuable food products. Journal of Applied Phycology, p. 1-13.

        4. Soares, A.P.M.R., Carvalho, F.O., De Farias Silva, C.E. (2020). Random Forest as a promising application to predict basic-dye biosorption process using orange waste. Journal of Environmental Chemical Engineering, v. 8, n. 4, 103952.
    """)