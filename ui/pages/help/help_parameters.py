"""
Página de ajuda sobre os parâmetros de configuração do BioCine
"""

import streamlit as st
import pandas as pd
import yaml
import os
import plotly.express as px
import plotly.graph_objects as go
import numpy as np


def show_parameters_help():
    """
    Renderiza a página de ajuda sobre os parâmetros de configuração
    """
    st.markdown("<h1 class='main-header'>Parâmetros de Configuração</h1>", unsafe_allow_html=True)

    # Introdução
    st.markdown("""
        <div class='info-box'>
        <h3>Configurando o BioCine</h3>
        <p>O BioCine permite ajustar diversos parâmetros para adequar a modelagem às características específicas do seu processo de tratamento de soro de leite com microalgas e fungos filamentosos.</p>
        </div>
    """, unsafe_allow_html=True)

    # Carrega o arquivo de configuração para mostrar ao usuário
    try:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
            "config", "config.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except Exception:
        # Se não conseguir carregar, usa um exemplo
        config = {
            "modeling": {
                "monod": {
                    "umax_default": 0.2,
                    "ks_default": 10.0,
                    "y_default": 0.5
                },
                "logistic": {
                    "umax_default": 0.2,
                    "x0_default": 0.1,
                    "xmax_default": 5.0
                }
            },
            "ml": {
                "random_forest": {
                    "n_estimators": 100,
                    "max_depth": 10,
                    "min_samples_split": 2,
                    "min_samples_leaf": 1,
                    "random_state": 42
                },
                "train_test_split": 0.8,
                "default_features": [
                    "tempo", "temperatura", "ph",
                    "concentracao_inicial_n",
                    "concentracao_inicial_p",
                    "concentracao_inicial_dqo"
                ]
            }
        }

    # Organização da página
    st.markdown("### Visão Geral dos Parâmetros")

    st.markdown("""
        Os parâmetros do BioCine estão organizados nas seguintes categorias:

        1. **Modelagem Cinética**: Parâmetros dos modelos Monod e Logístico
        2. **Machine Learning**: Configurações do algoritmo Random Forest e processamento de dados
        3. **Visualização**: Configurações para gráficos e figuras
        4. **Caminhos de Diretórios**: Onde os dados e relatórios são armazenados
    """)

    # Tabs para diferentes categorias de parâmetros
    tab1, tab2, tab3 = st.tabs(["Modelagem Cinética", "Machine Learning", "Recomendações"])

    with tab1:
        st.markdown("## Parâmetros de Modelagem Cinética")

        # Modelo de Monod
        st.markdown("### Modelo de Monod")

        monod_params = pd.DataFrame({
            'Parâmetro': ['umax_default', 'ks_default', 'y_default'],
            'Valor Padrão': [
                config['modeling']['monod']['umax_default'],
                config['modeling']['monod']['ks_default'],
                config['modeling']['monod']['y_default']
            ],
            'Descrição': [
                'Taxa específica máxima de crescimento (dia⁻¹)',
                'Constante de meia saturação (mg/L)',
                'Coeficiente de rendimento (g biomassa/g substrato)'
            ],
            'Faixa Recomendada': [
                '0.1 - 1.0 (microalgas), 0.05 - 0.5 (fungos)',
                '5 - 50',
                '0.1 - 0.7'
            ]
        })

        st.dataframe(monod_params)

        st.markdown("""
            #### Efeito dos parâmetros de Monod:

            - **Taxa específica máxima de crescimento (μmax)**: Determina quão rápido os microrganismos podem crescer em condições ideais. Valores mais altos resultam em crescimento mais rápido.

            - **Constante de meia saturação (Ks)**: Indica a concentração de substrato na qual a taxa de crescimento é metade da máxima. Valores menores indicam maior afinidade pelo substrato (o microrganismo consegue crescer bem mesmo com pouco substrato).

            - **Coeficiente de rendimento (Y)**: Mostra quanto de biomassa é produzida por unidade de substrato consumido. Valores mais altos indicam conversão mais eficiente.
        """)

        # Efeito visual da alteração de parâmetros de Monod
        st.markdown("#### Visualização do efeito dos parâmetros no modelo de Monod")

        col1, col2 = st.columns(2)

        with col1:
            umax_monod = st.slider("μmax (Monod)", 0.1, 1.0, 0.5, 0.1, key="param_monod_umax")

        with col2:
            ks_monod = st.slider("Ks", 1.0, 50.0, 10.0, 1.0, key="param_monod_ks")

        # Criar dados para o gráfico
        s_values = np.linspace(0, 100, 100)

        # Função para calcular a taxa específica de crescimento pelo modelo de Monod
        def monod_growth_rate(s, umax, ks):
            return umax * s / (ks + s)

        # Calcular taxas de crescimento
        u_values = monod_growth_rate(s_values, umax_monod, ks_monod)

        # Criar DataFrame para Plotly
        df_monod = pd.DataFrame({
            'Concentração de Substrato (mg/L)': s_values,
            'Taxa de Crescimento (dia⁻¹)': u_values
        })

        # Plotar com Plotly
        fig_monod = px.line(
            df_monod,
            x='Concentração de Substrato (mg/L)',
            y='Taxa de Crescimento (dia⁻¹)',
            title=f'Relação de Monod (μmax = {umax_monod}, Ks = {ks_monod})'
        )

        st.plotly_chart(fig_monod, use_container_width=True)

        # Modelo Logístico
        st.markdown("### Modelo Logístico")

        logistic_params = pd.DataFrame({
            'Parâmetro': ['umax_default', 'x0_default', 'xmax_default'],
            'Valor Padrão': [
                config['modeling']['logistic']['umax_default'],
                config['modeling']['logistic']['x0_default'],
                config['modeling']['logistic']['xmax_default']
            ],
            'Descrição': [
                'Taxa específica máxima de crescimento (dia⁻¹)',
                'Concentração inicial de biomassa (g/L)',
                'Concentração máxima de biomassa (g/L)'
            ],
            'Faixa Recomendada': [
                '0.1 - 1.0 (microalgas), 0.05 - 0.5 (fungos)',
                '0.01 - 0.5',
                '1.0 - 8.0 (microalgas), 3.0 - 15.0 (fungos)'
            ]
        })

        st.dataframe(logistic_params)

        st.markdown("""
            #### Efeito dos parâmetros Logísticos:

            - **Taxa específica máxima de crescimento (μmax)**: Determina a velocidade de crescimento na fase exponencial.

            - **Concentração inicial de biomassa (X0)**: Representa o inóculo inicial ou a biomassa no início do experimento.

            - **Concentração máxima de biomassa (Xmax)**: É o valor máximo que a biomassa pode atingir com os recursos disponíveis (capacidade de suporte).
        """)

        # Efeito visual da alteração de parâmetros Logísticos
        st.markdown("#### Visualização do efeito dos parâmetros no modelo Logístico")

        col1, col2 = st.columns(2)

        with col1:
            umax_logistic = st.slider("μmax (Logístico)", 0.1, 1.0, 0.5, 0.1, key="param_logistic_umax")
            x0_logistic = st.slider("X0", 0.01, 0.5, 0.1, 0.01)

        with col2:
            xmax_logistic = st.slider("Xmax", 1.0, 10.0, 5.0, 0.5)

        # Criar dados para o gráfico
        t_values = np.linspace(0, 20, 100)

        # Função para calcular o crescimento logístico
        def logistic_growth(t, x0, umax, xmax):
            return xmax / (1 + (xmax / x0 - 1) * np.exp(-umax * t))

        # Calcular crescimento
        x_values = logistic_growth(t_values, x0_logistic, umax_logistic, xmax_logistic)

        # Criar DataFrame para Plotly
        df_logistic = pd.DataFrame({
            'Tempo (dias)': t_values,
            'Biomassa (g/L)': x_values
        })

        # Plotar com Plotly
        fig_logistic = px.line(
            df_logistic,
            x='Tempo (dias)',
            y='Biomassa (g/L)',
            title=f'Crescimento Logístico (μmax = {umax_logistic}, X0 = {x0_logistic}, Xmax = {xmax_logistic})'
        )

        # Adicionar linha para Xmax
        fig_logistic.add_shape(
            type="line",
            x0=0, y0=xmax_logistic,
            x1=max(t_values), y1=xmax_logistic,
            line=dict(color="red", width=1, dash="dash"),
        )

        st.plotly_chart(fig_logistic, use_container_width=True)

    with tab2:
        st.markdown("## Parâmetros de Machine Learning")

        # Parâmetros do Random Forest
        st.markdown("### Random Forest")

        rf_params = pd.DataFrame({
            'Parâmetro': ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'random_state',
                          'train_test_split'],
            'Valor Padrão': [
                config['ml']['random_forest']['n_estimators'],
                config['ml']['random_forest']['max_depth'],
                config['ml']['random_forest']['min_samples_split'],
                config['ml']['random_forest']['min_samples_leaf'],
                config['ml']['random_forest']['random_state'],
                config['ml']['train_test_split']
            ],
            'Descrição': [
                'Número de árvores na floresta',
                'Profundidade máxima de cada árvore',
                'Número mínimo de amostras para dividir um nó',
                'Número mínimo de amostras em um nó folha',
                'Semente para geração de números aleatórios',
                'Proporção de dados usados para treinamento'
            ],
            'Quando Ajustar': [
                'Aumentar para melhorar precisão; diminuir para reduzir tempo de processamento',
                'Diminuir para reduzir overfitting; aumentar para capturar relações mais complexas',
                'Aumentar para reduzir overfitting em conjuntos pequenos',
                'Aumentar para tornar o modelo mais conservador',
                'Manter um valor fixo para resultados reproduzíveis',
                'Reduzir para conjuntos pequenos (0.7); aumentar para conjuntos grandes (0.9)'
            ]
        })

        st.dataframe(rf_params)

        # Características padrão
        st.markdown("### Características Padrão")

        features_df = pd.DataFrame({
            'Característica': config['ml']['default_features'],
            'Descrição': [
                'Duração do processo (dias)',
                'Temperatura do meio de cultivo (°C)',
                'Acidez/basicidade do meio',
                'Concentração inicial de nitrogênio (mg/L)',
                'Concentração inicial de fósforo (mg/L)',
                'Concentração inicial de DQO (mg/L)'
            ],
            'Importância': [
                'Crítica - O crescimento e a remoção são processos temporais',
                'Alta - Afeta diretamente a atividade metabólica',
                'Alta - Influencia a atividade enzimática e disponibilidade de nutrientes',
                'Alta - Nutriente essencial que pode limitar o crescimento',
                'Média - Nutriente essencial para síntese de ácidos nucleicos',
                'Alta - Indica a quantidade de matéria orgânica disponível'
            ],
            'Faixa Típica': [
                '0 - 30 dias',
                '15 - 35°C (microalgas), 25 - 30°C (fungos)',
                '6.5 - 8.5 (microalgas), 4.5 - 6.0 (fungos)',
                '20 - 150 mg/L no soro de leite',
                '5 - 50 mg/L no soro de leite',
                '500 - 5000 mg/L no soro de leite'
            ]
        })

        st.dataframe(features_df)

        # Efeito do número de árvores e profundidade na precisão
        st.markdown("### Efeito do número de árvores e profundidade na precisão do modelo")

        # Simulação do efeito de n_estimators e max_depth na performance
        # Dados simulados
        n_trees = [10, 50, 100, 200, 300]
        depths = [2, 5, 10, 15, 20]

        # Matriz de accuracy simulada - valores crescem com mais árvores/profundidade mas estabilizam/pioram com overfitting
        r2_train = np.array([
            [0.75, 0.80, 0.85, 0.95, 0.99],  # depth=2
            [0.80, 0.85, 0.90, 0.97, 0.99],  # depth=5
            [0.82, 0.88, 0.93, 0.98, 0.99],  # depth=10
            [0.84, 0.90, 0.95, 0.98, 0.99],  # depth=15
            [0.85, 0.92, 0.96, 0.99, 0.99]  # depth=20
        ])

        r2_test = np.array([
            [0.70, 0.75, 0.77, 0.78, 0.79],  # depth=2
            [0.75, 0.80, 0.82, 0.82, 0.81],  # depth=5
            [0.76, 0.81, 0.84, 0.83, 0.82],  # depth=10
            [0.74, 0.80, 0.82, 0.80, 0.78],  # depth=15
            [0.72, 0.78, 0.80, 0.77, 0.74]  # depth=20
        ])

        # Criar dados para o heatmap
        heatmap_data_train = []
        heatmap_data_test = []

        for i, depth in enumerate(depths):
            for j, n_tree in enumerate(n_trees):
                heatmap_data_train.append({
                    'Número de Árvores': str(n_tree),
                    'Profundidade Máxima': str(depth),
                    'R²': r2_train[i, j],
                    'Conjunto': 'Treino'
                })
                heatmap_data_test.append({
                    'Número de Árvores': str(n_tree),
                    'Profundidade Máxima': str(depth),
                    'R²': r2_test[i, j],
                    'Conjunto': 'Teste'
                })

        heatmap_data = pd.concat([pd.DataFrame(heatmap_data_train), pd.DataFrame(heatmap_data_test)])

        # Criar duas abas para os heatmaps
        subtab1, subtab2 = st.tabs(["R² Treino", "R² Teste"])

        with subtab1:
            # Filtrar apenas dados de treino
            train_data = heatmap_data[heatmap_data['Conjunto'] == 'Treino']

            fig_train = px.density_heatmap(
                train_data,
                x='Número de Árvores',
                y='Profundidade Máxima',
                z='R²',
                color_continuous_scale='Blues',
                title='R² no Conjunto de Treino'
            )

            fig_train.update_layout(
                xaxis_title='Número de Árvores',
                yaxis_title='Profundidade Máxima'
            )

            st.plotly_chart(fig_train, use_container_width=True)

            st.markdown("""
                **Observação:** O R² no conjunto de treino geralmente aumenta com mais árvores e maior profundidade, 
                potencialmente levando a overfitting (modelo memoriza os dados de treino, mas não generaliza bem).
            """)

        with subtab2:
            # Filtrar apenas dados de teste
            test_data = heatmap_data[heatmap_data['Conjunto'] == 'Teste']

            fig_test = px.density_heatmap(
                test_data,
                x='Número de Árvores',
                y='Profundidade Máxima',
                z='R²',
                color_continuous_scale='Greens',
                title='R² no Conjunto de Teste'
            )

            fig_test.update_layout(
                xaxis_title='Número de Árvores',
                yaxis_title='Profundidade Máxima'
            )

            st.plotly_chart(fig_test, use_container_width=True)

            st.markdown("""
                **Observação:** O R² no conjunto de teste inicialmente melhora com mais árvores e maior profundidade, 
                mas depois piora com o overfitting. O melhor modelo é aquele que maximiza o desempenho no conjunto de teste.
            """)

    with tab3:
        st.markdown("## Recomendações para Configuração de Parâmetros")

        st.markdown("### Para Microalgas")

        st.markdown("""
            #### Modelo de Monod
            - **μmax**: 0.2 - 0.6 dia⁻¹
            - **Ks**: 10 - 30 mg/L para nitrogênio; 1 - 5 mg/L para fósforo
            - **Y**: 0.3 - 0.5 g biomassa/g substrato

            #### Modelo Logístico
            - **μmax**: 0.2 - 0.6 dia⁻¹
            - **X0**: Valores reais de seu inóculo (tipicamente 0.05 - 0.2 g/L)
            - **Xmax**: 2.0 - 5.0 g/L (dependendo das condições de cultivo)

            #### Condições de Cultivo Típicas
            - **Temperatura**: 20 - 30°C (ótimo ≈ 25°C)
            - **pH**: 7.0 - 8.5
            - **Fotoperíodo**: 12:12 a 16:8 (luz:escuro)
        """)

        st.markdown("### Para Fungos Filamentosos")

        st.markdown("""
            #### Modelo de Monod
            - **μmax**: 0.1 - 0.3 dia⁻¹
            - **Ks**: 15 - 40 mg/L para nitrogênio; 2 - 8 mg/L para fósforo
            - **Y**: 0.3 - 0.6 g biomassa/g substrato

            #### Modelo Logístico
            - **μmax**: 0.1 - 0.3 dia⁻¹
            - **X0**: Valores reais de seu inóculo (tipicamente 0.05 - 0.2 g/L)
            - **Xmax**: 3.0 - 8.0 g/L (dependendo das condições de cultivo)

            #### Condições de Cultivo Típicas
            - **Temperatura**: 25 - 30°C
            - **pH**: 4.5 - 6.0
            - **Agitação**: 150 - 200 rpm (para cultivos agitados)
        """)

        st.markdown("### Para Tratamento de Soro de Leite")

        st.markdown("""
            #### Diluição Recomendada
            - **Microalgas**: 10 - 30% v/v de soro em meio de cultivo
            - **Fungos**: 20 - 50% v/v de soro em meio de cultivo

            #### Suplementação
            - **Microalgas**: Considere suplementar com fósforo se a concentração inicial for < 5 mg/L
            - **Fungos**: Geralmente não requer suplementação, mas pode se beneficiar de adição de micronutrientes

            #### Tempo de Retenção Hidráulica
            - **Microalgas**: 7 - 14 dias
            - **Fungos**: 3 - 7 dias

            #### Random Forest para Otimização
            - **Número de Árvores**: 100 - 200
            - **Profundidade Máxima**: 8 - 12
            - **Principais Características**: Tempo, temperatura, pH, concentração inicial de poluentes
        """)

        st.markdown("### Para pequenos conjuntos de dados (< 30 pontos experimentais)")

        st.markdown("""
            #### Modelagem Cinética
            - Prefira o modelo Logístico por ter menos parâmetros
            - Mantenha os parâmetros dentro das faixas típicas para o organismo
            - Confie mais na inspeção visual do ajuste do que apenas nas métricas numéricas

            #### Machine Learning
            - Reduza a profundidade máxima (max_depth ≤ 5)
            - Aumente min_samples_split (≥ 3) e min_samples_leaf (≥ 2)
            - Use validação cruzada (se disponível) em vez de divisão simples treino/teste
            - Reduza o número de características (features) para as 3-4 mais importantes
            - Considere train_test_split = 0.7 (70% treino, 30% teste)
        """)

    # Referências
    st.markdown("### Referências")

    st.markdown("""
        1. Shuler, M. L., & Kargi, F. (2002). Bioprocess Engineering: Basic Concepts. 2nd ed. Prentice Hall.

        2. He, Y., Chen, L., Zhou, Y. et al. (2016). Analysis and model delineation of marine microalgae growth and lipid accumulation in flat-plate photobioreactor. Biochemical Engineering Journal, v. 111, p. 108-116.

        3. Breiman, L. (2001). Random forests. Machine learning, 45(1), 5-32.

        4. Li, Y., Miros, S., Kiani, H., et al. (2023). Mechanism of lactose assimilation in microalgae for the bioremediation of dairy processing side-streams and co-production of valuable food products. Journal of Applied Phycology, p. 1-13.

        5. Veiter, L., Rajamanickam, V., & Herwig, C. (2018). The filamentous fungal pellet-relationship between morphology and productivity. Applied Microbiology and Biotechnology, v. 102, n. 7, p. 2997-3006.
    """)