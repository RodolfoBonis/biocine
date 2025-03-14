import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, r2_score
import base64

# Importando módulos próprios
from models import CineticModels, MachineLearningModels
from visualization import Visualization
from data_utils import DataProcessor

# Configuração da página
st.set_page_config(
    page_title="BioCine - Modelagem de Tratamento de Soro de Leite",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #34495e;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #3498db;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .info-text {
        font-size: 1rem;
        color: #7f8c8d;
    }
    .highlight {
        background-color: #2b2b2b;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.5rem solid #3498db;
    }
</style>
""", unsafe_allow_html=True)


# Função para criar um link de download para o dataframe
def create_download_link(df, filename):
    """
    Cria um link HTML para download do dataframe como CSV
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'
    return href


# Função para criar gráfico de download
def create_download_link(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'
    return href


# Função principal da interface
def main():
    st.markdown('<h1 class="main-header">BioCine - Software de Modelagem Cinética</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="info-text">Modelagem Cinética do Processo de Tratamento Terciário em Batelada/Semicontínuo do Soro do Leite por Microalgas e Fungos Filamentosos</p>',
        unsafe_allow_html=True)

    # Menu lateral
    st.sidebar.title("Navegação")
    opcao = st.sidebar.radio("Selecione uma seção:", [
        "Início",
        "Entrada de Dados",
        "Modelagem Cinética",
        "Machine Learning",
        "Visualização e Resultados",
        "Sobre o Software"
    ])

    # Estado da aplicação para armazenar dados
    if 'data' not in st.session_state:
        st.session_state['data'] = None

    if 'model_params' not in st.session_state:
        st.session_state['model_params'] = {
            'microalga': None,
            'fungo': None,
            'nitrogenio': None,
            'fosforo': None,
            'dqo': None
        }

    if 'ml_models' not in st.session_state:
        st.session_state['ml_models'] = {
            'microalga': None,
            'fungo': None,
            'nitrogenio': None,
            'fosforo': None,
            'dqo': None
        }

    # Seção de Início
    if opcao == "Início":
        st.markdown('<h2 class="sub-header">Bem-vindo ao BioCine</h2>', unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("""
            <div class="highlight">
            <p>O BioCine é um software especializado para modelagem cinética e previsão de eficiência no tratamento 
            terciário do soro de leite utilizando microalgas e fungos filamentosos.</p>

            <p>Com este software, você pode:</p>
            <ul>
                <li>Importar e gerenciar dados experimentais</li>
                <li>Aplicar modelos cinéticos de Monod e Logístico</li>
                <li>Utilizar técnicas de machine learning para previsão</li>
                <li>Visualizar resultados através de gráficos interativos</li>
                <li>Exportar relatórios e análises</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<h3 class="section-header">Como começar:</h3>', unsafe_allow_html=True)
            st.markdown("""
            1. Vá para a seção "Entrada de Dados"
            2. Carregue seus dados experimentais ou use os dados de exemplo
            3. Explore as funcionalidades de modelagem e previsão
            """)

        with col2:
            # Imagem representativa (usando Plotly para criar um gráfico simples)
            fig = go.Figure()

            # Dados de exemplo para o gráfico
            t = np.linspace(0, 10, 100)
            biomassa = CineticModels.logistic_model(t, 0.2, 0.5, 1.6)
            substrato = 120 - 70 * (biomassa - 0.2) / (1.6 - 0.2)

            # Adicionar traços
            fig.add_trace(go.Scatter(x=t, y=biomassa, mode='lines', name='Crescimento de Biomassa',
                                     line=dict(color='green', width=2)))
            fig.add_trace(go.Scatter(x=t, y=substrato / 120, mode='lines', name='Consumo de Substrato',
                                     line=dict(color='blue', width=2)))

            # Layout
            fig.update_layout(
                title='Simulação: Crescimento e Consumo',
                xaxis_title='Tempo (dias)',
                yaxis_title='Concentração Normalizada',
                legend=dict(x=0.02, y=0.98),
                margin=dict(l=0, r=0, t=40, b=0)
            )

            st.plotly_chart(fig, use_container_width=True)

    # Seção de Entrada de Dados
    elif opcao == "Entrada de Dados":
        st.markdown('<h2 class="sub-header">Entrada de Dados Experimentais</h2>', unsafe_allow_html=True)

        entrada_opcao = st.radio(
            "Escolha a forma de entrada de dados:",
            ["Usar dados de exemplo", "Carregar arquivo CSV", "Inserir dados manualmente"]
        )

        if entrada_opcao == "Usar dados de exemplo":
            st.markdown('<p class="info-text">Usando dados de exemplo para demonstração.</p>', unsafe_allow_html=True)
            st.session_state['data'] = DataProcessor.create_example_data()

            st.markdown('<h3 class="section-header">Visualização dos dados:</h3>', unsafe_allow_html=True)
            st.dataframe(st.session_state['data'])

            st.markdown(create_download_link(st.session_state['data'], "dados_exemplo.csv"), unsafe_allow_html=True)

        elif entrada_opcao == "Carregar arquivo CSV":
            st.markdown('<p class="info-text">Carregue um arquivo CSV com seus dados experimentais.</p>',
                        unsafe_allow_html=True)

            uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")

            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state['data'] = df

                    st.success("Arquivo carregado com sucesso!")
                    st.markdown('<h3 class="section-header">Visualização dos dados:</h3>', unsafe_allow_html=True)
                    st.dataframe(df)

                    # Permitir ao usuário mapear as colunas
                    st.markdown('<h3 class="section-header">Mapeamento de colunas:</h3>', unsafe_allow_html=True)
                    st.markdown(
                        '<p class="info-text">Selecione as colunas correspondentes aos parâmetros necessários.</p>',
                        unsafe_allow_html=True)

                    col_tempo = st.selectbox("Coluna de tempo:", df.columns)
                    col_biomassa = st.selectbox("Coluna de biomassa do consórcio microalga+fungo:", df.columns)
                    col_nitrogenio = st.selectbox("Coluna de concentração de nitrogênio:", df.columns)
                    col_fosforo = st.selectbox("Coluna de concentração de fósforo:", df.columns)
                    col_dqo = st.selectbox("Coluna de DQO:", df.columns)

                    if st.button("Confirmar mapeamento"):
                        # Renomear colunas para o formato padrão
                        df_mapped = df.rename(columns={
                            col_tempo: "Tempo (dias)",
                            col_biomassa: "Biomassa Consórcio (g/L)",
                            col_nitrogenio: "Nitrogênio (mg/L)",
                            col_fosforo: "Fósforo (mg/L)",
                            col_dqo: "DQO (mg/L)"
                        })

                        st.session_state['data'] = df_mapped[["Tempo (dias)", "Biomassa Consórcio (g/L)",
                                                              "Nitrogênio (mg/L)", "Fósforo (mg/L)",
                                                              "DQO (mg/L)"]]

                        st.success("Mapeamento concluído!")
                        st.dataframe(st.session_state['data'])

                except Exception as e:
                    st.error(f"Erro ao carregar o arquivo: {e}")

        elif entrada_opcao == "Inserir dados manualmente":
            st.markdown('<p class="info-text">Insira seus dados experimentais manualmente.</p>', unsafe_allow_html=True)

            # Determinando o número de pontos de tempo
            num_pontos = st.number_input("Número de pontos de dados:", min_value=3, max_value=20, value=5)

            # Criando formulário para entrada de dados
            with st.form("entrada_manual"):
                # Criando listas para armazenar os dados
                tempos = []
                biomassa = []
                nitrogenio = []
                fosforo = []
                dqo = []

                # Criando várias colunas para os pontos de dados
                for i in range(num_pontos):
                    st.markdown(f"### Ponto {i + 1}")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        tempos.append(st.number_input(f"Tempo (dias) - Ponto {i + 1}", value=float(i)))
                        biomassa.append(
                            st.number_input(f"Biomassa Consórcio (g/L) - Ponto {i + 1}", value=0.35 + float(i) * 0.3))

                    with col2:
                        nitrogenio.append(
                            st.number_input(f"Nitrogênio (mg/L) - Ponto {i + 1}", value=120.0 - float(i) * 10.0))
                        fosforo.append(st.number_input(f"Fósforo (mg/L) - Ponto {i + 1}", value=30.0 - float(i) * 2.0))

                    with col3:
                        dqo.append(st.number_input(f"DQO (mg/L) - Ponto {i + 1}", value=5000.0 - float(i) * 400.0))

                submit_button = st.form_submit_button("Salvar dados")

                if submit_button:
                    # Criando DataFrame com os dados inseridos
                    df_manual = pd.DataFrame({
                        "Tempo (dias)": tempos,
                        "Biomassa Consórcio (g/L)": biomassa,
                        "Nitrogênio (mg/L)": nitrogenio,
                        "Fósforo (mg/L)": fosforo,
                        "DQO (mg/L)": dqo
                    })

                    st.session_state['data'] = df_manual
                    st.success("Dados salvos com sucesso!")
                    st.dataframe(df_manual)

        # Verifica se há dados disponíveis para exibir gráficos preliminares
        if st.session_state['data'] is not None:
            st.markdown('<h3 class="section-header">Gráficos preliminares:</h3>', unsafe_allow_html=True)

            # Seleção de tipo de gráfico
            tipo_grafico = st.selectbox(
                "Selecione o tipo de gráfico:",
                ["Biomassa ao longo do tempo", "Remoção de poluentes", "Correlação entre parâmetros"]
            )

            df = st.session_state['data']

            if tipo_grafico == "Biomassa ao longo do tempo":
                # Gráfico de biomassa ao longo do tempo
                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=df["Tempo (dias)"],
                    y=df["Biomassa Consórcio (g/L)"],
                    mode='lines+markers',
                    name='Consórcio Microalga+Fungo'
                ))

                fig.update_layout(
                    title='Crescimento de Biomassa ao Longo do Tempo',
                    xaxis_title='Tempo (dias)',
                    yaxis_title='Biomassa (g/L)',
                    legend=dict(x=0.02, y=0.98),
                    margin=dict(l=0, r=0, t=40, b=0)
                )

                st.plotly_chart(fig, use_container_width=True)

            elif tipo_grafico == "Remoção de poluentes":
                # Gráfico de remoção de poluentes
                parametro = st.selectbox(
                    "Selecione o parâmetro:",
                    ["Nitrogênio (mg/L)", "Fósforo (mg/L)", "DQO (mg/L)"]
                )

                fig = go.Figure()

                # Calculando a eficiência de remoção
                valor_inicial = df[parametro].iloc[0]
                eficiencia = [100 * (1 - valor / valor_inicial) for valor in df[parametro]]

                fig.add_trace(go.Scatter(
                    x=df["Tempo (dias)"],
                    y=df[parametro],
                    mode='lines+markers',
                    name=f'Concentração de {parametro}'
                ))

                fig.add_trace(go.Scatter(
                    x=df["Tempo (dias)"],
                    y=eficiencia,
                    mode='lines+markers',
                    name='Eficiência de Remoção (%)',
                    yaxis='y2'
                ))

                fig.update_layout(
                    title=f'Remoção de {parametro} ao Longo do Tempo',
                    xaxis_title='Tempo (dias)',
                    yaxis_title=parametro,
                    yaxis2=dict(
                        title='Eficiência de Remoção (%)',
                        overlaying='y',
                        side='right'
                    ),
                    legend=dict(x=0.02, y=0.98),
                    margin=dict(l=0, r=0, t=40, b=0)
                )

                st.plotly_chart(fig, use_container_width=True)

            elif tipo_grafico == "Correlação entre parâmetros":
                # Gráfico de correlação entre parâmetros
                parametro_x = st.selectbox(
                    "Selecione o parâmetro para o eixo X:",
                    df.columns.tolist()
                )

                parametro_y = st.selectbox(
                    "Selecione o parâmetro para o eixo Y:",
                    [col for col in df.columns if col != parametro_x]
                )

                fig = px.scatter(
                    df,
                    x=parametro_x,
                    y=parametro_y,
                    trendline="ols",
                    title=f'Correlação entre {parametro_x} e {parametro_y}'
                )

                st.plotly_chart(fig, use_container_width=True)

    # Seção de Modelagem Cinética
    elif opcao == "Modelagem Cinética":
        st.markdown('<h2 class="sub-header">Modelagem Cinética</h2>', unsafe_allow_html=True)

        if st.session_state['data'] is None:
            st.warning("Nenhum dado disponível. Por favor, vá para a seção 'Entrada de Dados' primeiro.")
        else:
            df = st.session_state['data']

            st.markdown('<p class="info-text">Ajuste modelos cinéticos aos seus dados experimentais.</p>',
                        unsafe_allow_html=True)

            modelo_opcao = st.radio(
                "Selecione o modelo cinético:",
                ["Modelo Logístico", "Modelo de Monod", "Ambos os modelos"]
            )

            parametro = st.selectbox(
                "Selecione o parâmetro para modelar:",
                ["Biomassa Consórcio (g/L)", "Nitrogênio (mg/L)", "Fósforo (mg/L)", "DQO (mg/L)"]
            )

            # Correspondência entre parâmetros e chaves no dicionário de modelos
            param_key_map = {
                "Biomassa Consórcio (g/L)": "biomassa",
                "Nitrogênio (mg/L)": "nitrogenio",
                "Fósforo (mg/L)": "fosforo",
                "DQO (mg/L)": "dqo"
            }

            param_key = param_key_map[parametro]

            if st.button("Ajustar Modelo"):
                # Dados para o ajuste
                time_data = df["Tempo (dias)"].values
                param_data = df[parametro].values

                try:
                    with st.spinner("Ajustando modelo..."):
                        if modelo_opcao in ["Modelo Logístico", "Ambos os modelos"]:
                            # Para biomassa, usamos diretamente o modelo logístico
                            if parametro in ["Biomassa Consórcio (g/L)"]:
                                popt, pcov = CineticModels.fit_logistic_model(time_data, param_data)
                                X0, umax, Xmax = popt

                                # Salvando os parâmetros do modelo
                                st.session_state['model_params'][param_key] = {
                                    'logistic': {
                                        'X0': X0,
                                        'umax': umax,
                                        'Xmax': Xmax
                                    }
                                }

                                # Gráfico do ajuste
                                fig = go.Figure()

                                # Dados experimentais
                                fig.add_trace(go.Scatter(
                                    x=time_data,
                                    y=param_data,
                                    mode='markers',
                                    name='Dados Experimentais'
                                ))

                                # Curva ajustada
                                t_fine = np.linspace(min(time_data), max(time_data), 100)
                                y_pred = CineticModels.logistic_model(t_fine, X0, umax, Xmax)

                                fig.add_trace(go.Scatter(
                                    x=t_fine,
                                    y=y_pred,
                                    mode='lines',
                                    name='Modelo Logístico Ajustado'
                                ))

                                fig.update_layout(
                                    title=f'Ajuste do Modelo Logístico - {parametro}',
                                    xaxis_title='Tempo (dias)',
                                    yaxis_title=parametro,
                                    legend=dict(x=0.02, y=0.98)
                                )

                                st.plotly_chart(fig, use_container_width=True)

                                # Exibindo os parâmetros ajustados
                                st.markdown('<h3 class="section-header">Parâmetros do Modelo Logístico:</h3>',
                                            unsafe_allow_html=True)
                                st.markdown(f"""
                                - **X0 (Biomassa inicial):** {X0:.4f} g/L
                                - **μmax (Taxa máxima de crescimento):** {umax:.4f} dia⁻¹
                                - **Xmax (Capacidade máxima):** {Xmax:.4f} g/L
                                """)

                                # Calculando taxas de crescimento para ajuste do modelo de Monod
                                if modelo_opcao == "Ambos os modelos":
                                    growth_rates = CineticModels.calculate_growth_rates(time_data, param_data)

                                    # Para cálculo da taxa de crescimento específica, precisamos de um substrato
                                    # Vamos usar o nitrogênio como exemplo
                                    substrate_data = df["Nitrogênio (mg/L)"].values[:-1]  # Um ponto a menos que o tempo

                                    try:
                                        # Ajuste do modelo de Monod
                                        monod_popt, monod_pcov = CineticModels.fit_monod_parameters(substrate_data,
                                                                                                    growth_rates)
                                        umax_monod, Ks = monod_popt

                                        # Salvando os parâmetros do modelo
                                        if 'monod' not in st.session_state['model_params'][param_key]:
                                            st.session_state['model_params'][param_key]['monod'] = {}

                                        st.session_state['model_params'][param_key]['monod'] = {
                                            'umax': umax_monod,
                                            'Ks': Ks
                                        }

                                        # Gráfico do ajuste de Monod
                                        fig_monod = go.Figure()

                                        # Dados experimentais
                                        fig_monod.add_trace(go.Scatter(
                                            x=substrate_data,
                                            y=growth_rates,
                                            mode='markers',
                                            name='Taxas de Crescimento Experimentais'
                                        ))

                                        # Curva ajustada
                                        s_fine = np.linspace(min(substrate_data), max(substrate_data), 100)
                                        mu_pred = CineticModels.monod_model(s_fine, umax_monod, Ks)

                                        fig_monod.add_trace(go.Scatter(
                                            x=s_fine,
                                            y=mu_pred,
                                            mode='lines',
                                            name='Modelo de Monod Ajustado'
                                        ))

                                        fig_monod.update_layout(
                                            title=f'Ajuste do Modelo de Monod - {parametro}',
                                            xaxis_title='Concentração de Substrato (mg/L)',
                                            yaxis_title='Taxa de Crescimento Específica (dia⁻¹)',
                                            legend=dict(x=0.02, y=0.98)
                                        )

                                        st.plotly_chart(fig_monod, use_container_width=True)

                                        # Exibindo os parâmetros ajustados
                                        st.markdown('<h3 class="section-header">Parâmetros do Modelo de Monod:</h3>',
                                                    unsafe_allow_html=True)
                                        st.markdown(f"""
                                        - **μmax (Taxa máxima de crescimento):** {umax_monod:.4f} dia⁻¹
                                        - **Ks (Constante de meia saturação):** {Ks:.4f} mg/L
                                        """)

                                    except Exception as e:
                                        st.warning(f"Não foi possível ajustar o modelo de Monod: {e}")

                            # Para substrato (nitrogênio, fósforo, DQO), ajustamos diretamente os dados de decaimento
                            else:
                                # Ajuste direto usando o flag is_biomass=False para indicar que são poluentes
                                popt, pcov = CineticModels.fit_logistic_model(time_data, param_data, is_biomass=False)
                                S0, k, Smax = popt

                                # Salvando os parâmetros do modelo
                                st.session_state['model_params'][param_key] = {
                                    'logistic_inverted': {
                                        'S0': S0,
                                        'k': k,
                                        'Smax': Smax
                                    }
                                }

                                # Gráfico do ajuste
                                fig = go.Figure()

                                # Dados experimentais
                                fig.add_trace(go.Scatter(
                                    x=time_data,
                                    y=param_data,
                                    mode='markers',
                                    name='Dados Experimentais'
                                ))

                                # Curva ajustada
                                t_fine = np.linspace(min(time_data), max(time_data), 100)
                                y_pred = CineticModels.logistic_model(t_fine, S0, k, Smax)

                                fig.add_trace(go.Scatter(
                                    x=t_fine,
                                    y=y_pred,
                                    mode='lines',
                                    name='Modelo Ajustado'
                                ))

                                fig.update_layout(
                                    title=f'Ajuste do Modelo - {parametro}',
                                    xaxis_title='Tempo (dias)',
                                    yaxis_title=parametro,
                                    legend=dict(x=0.02, y=0.98)
                                )

                                st.plotly_chart(fig, use_container_width=True)

                                # Exibindo os parâmetros ajustados
                                st.markdown('<h3 class="section-header">Parâmetros do Modelo:</h3>',
                                            unsafe_allow_html=True)
                                st.markdown(f"""
                                - **S0 (Concentração inicial):** {S0:.4f} mg/L
                                - **k (Taxa de decaimento):** {k:.4f} dia⁻¹
                                - **Smax (Redução máxima):** {Smax:.4f} mg/L
                                """)

                        elif modelo_opcao == "Modelo de Monod":
                            # Para o modelo de Monod, precisamos das taxas de crescimento
                            if parametro in ["Biomassa Consórcio (g/L)"]:
                                growth_rates = CineticModels.calculate_growth_rates(time_data, param_data)

                                # Usando nitrogênio como substrato
                                substrate_data = df["Nitrogênio (mg/L)"].values[:-1]  # Um ponto a menos que o tempo

                                try:
                                    # Ajuste do modelo de Monod
                                    monod_popt, monod_pcov = CineticModels.fit_monod_parameters(substrate_data, growth_rates)
                                    umax_monod, Ks = monod_popt

                                    # Salvando os parâmetros do modelo
                                    st.session_state['model_params'][param_key] = {
                                        'monod': {
                                            'umax': umax_monod,
                                            'Ks': Ks
                                        }
                                    }

                                    # Gráfico do ajuste de Monod
                                    fig_monod = go.Figure()

                                    # Dados experimentais
                                    fig_monod.add_trace(go.Scatter(
                                        x=substrate_data,
                                        y=growth_rates,
                                        mode='markers',
                                        name='Taxas de Crescimento Experimentais'
                                    ))

                                    # Curva ajustada
                                    s_fine = np.linspace(min(substrate_data), max(substrate_data), 100)
                                    mu_pred = CineticModels.monod_model(s_fine, umax_monod, Ks)

                                    fig_monod.add_trace(go.Scatter(
                                        x=s_fine,
                                        y=mu_pred,
                                        mode='lines',
                                        name='Modelo de Monod Ajustado'
                                    ))

                                    fig_monod.update_layout(
                                        title=f'Ajuste do Modelo de Monod - {parametro}',
                                        xaxis_title='Concentração de Substrato (mg/L)',
                                        yaxis_title='Taxa de Crescimento Específica (dia⁻¹)',
                                        legend=dict(x=0.02, y=0.98)
                                    )

                                    st.plotly_chart(fig_monod, use_container_width=True)

                                    # Exibindo os parâmetros ajustados
                                    st.markdown('<h3 class="section-header">Parâmetros do Modelo de Monod:</h3>',
                                                unsafe_allow_html=True)
                                    st.markdown(f"""
                                    - **μmax (Taxa máxima de crescimento):** {umax_monod:.4f} dia⁻¹
                                    - **Ks (Constante de meia saturação):** {Ks:.4f} mg/L
                                    """)

                                except Exception as e:
                                    st.warning(f"Não foi possível ajustar o modelo de Monod: {e}")
                            else:
                                st.warning("O modelo de Monod é aplicável apenas para biomassa.")

                    st.success("Modelo ajustado com sucesso!")

                except Exception as e:
                    st.error(f"Erro ao ajustar o modelo: {e}")

    # Seção de Machine Learning
    elif opcao == "Machine Learning":
        st.markdown('<h2 class="sub-header">Machine Learning</h2>', unsafe_allow_html=True)

        if st.session_state['data'] is None:
            st.warning("Nenhum dado disponível. Por favor, vá para a seção 'Entrada de Dados' primeiro.")
        else:
            df = st.session_state['data']

            st.markdown(
                '<p class="info-text">Use técnicas de machine learning para prever parâmetros e otimizar o processo.</p>',
                unsafe_allow_html=True)

            ml_opcao = st.radio(
                "Selecione a operação de machine learning:",
                ["Previsão de parâmetros", "Otimização de processo"]
            )

            if ml_opcao == "Previsão de parâmetros":
                # Seleção do parâmetro alvo
                target_param = st.selectbox(
                    "Selecione o parâmetro a ser previsto:",
                    ["Biomassa Consórcio (g/L)", "Nitrogênio (mg/L)", "Fósforo (mg/L)", "DQO (mg/L)"]
                )

                # Seleção dos parâmetros de entrada
                available_features = [col for col in df.columns if col != target_param]
                feature_params = st.multiselect(
                    "Selecione os parâmetros de entrada:",
                    available_features,
                    default=["Tempo (dias)"]
                )

                if len(feature_params) == 0:
                    st.warning("Selecione pelo menos um parâmetro de entrada.")
                else:
                    # Botão para treinar modelo
                    train_button = st.button("Treinar modelo")

                    # Armazenando informações na sessão
                    param_key = target_param.split(" ")[1].lower()

                    # Verificar se o modelo já foi treinado com segurança
                    model_trained = 'ml_trained' in st.session_state and st.session_state['ml_trained']

                    # Verifica se o botão foi clicado ou se já temos um modelo treinado
                    if train_button or model_trained:
                        # Se o botão foi clicado ou já treinamos o modelo antes
                        if train_button:  # Se foi clicado agora, treina o modelo
                            try:
                                with st.spinner("Treinando modelo..."):
                                    # Preparando dados
                                    X_train, X_test, y_train, y_test, scaler_X, scaler_y = MachineLearningModels.prepare_data(
                                        df, target_param, feature_params
                                    )

                                    # Treinando modelo
                                    model = MachineLearningModels.train_random_forest(X_train, y_train)

                                    # Avaliando modelo
                                    y_pred = model.predict(X_test)
                                    y_pred_orig = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
                                    y_test_orig = scaler_y.inverse_transform(y_test)

                                    mse = mean_squared_error(y_test_orig, y_pred_orig)
                                    r2 = r2_score(y_test_orig, y_pred_orig)

                                    # Salvar modelo na sessão
                                    st.session_state['ml_models'][param_key] = {
                                        'model': model,
                                        'scaler_X': scaler_X,
                                        'scaler_y': scaler_y,
                                        'features': feature_params,
                                        'target': target_param,
                                        'mse': mse,
                                        'r2': r2
                                    }

                                    # Marcar que treinamos o modelo
                                    st.session_state['ml_trained'] = True

                                    # Exibindo métricas
                                    st.markdown('<h3 class="section-header">Métricas do Modelo:</h3>',
                                                unsafe_allow_html=True)
                                    st.markdown(f"""
                                    - **Erro Quadrático Médio (MSE):** {mse:.4f}
                                    - **Coeficiente de Determinação (R²):** {r2:.4f}
                                    """)

                                    # Gráfico de valores reais vs. previstos
                                    fig = go.Figure()

                                    fig.add_trace(go.Scatter(
                                        x=y_test_orig.flatten(),
                                        y=y_pred_orig.flatten(),
                                        mode='markers',
                                        name='Teste'
                                    ))

                                    # Linha de referência (y=x)
                                    min_val = min(np.min(y_test_orig), np.min(y_pred_orig))
                                    max_val = max(np.max(y_test_orig), np.max(y_pred_orig))

                                    fig.add_trace(go.Scatter(
                                        x=[min_val, max_val],
                                        y=[min_val, max_val],
                                        mode='lines',
                                        name='Ideal (y=x)',
                                        line=dict(dash='dash')
                                    ))

                                    fig.update_layout(
                                        title=f'Valores Reais vs. Previstos - {target_param}',
                                        xaxis_title='Valores Reais',
                                        yaxis_title='Valores Previstos'
                                    )

                                    st.plotly_chart(fig, use_container_width=True)

                                    # Feature importance
                                    st.markdown('<h3 class="section-header">Importância das Características:</h3>',
                                                unsafe_allow_html=True)

                                    importances = model.feature_importances_
                                    feature_imp = pd.DataFrame({
                                        'Feature': feature_params,
                                        'Importance': importances
                                    }).sort_values('Importance', ascending=False)

                                    fig_imp = px.bar(
                                        feature_imp,
                                        y='Feature',
                                        x='Importance',
                                        orientation='h',
                                        title='Importância das Características'
                                    )

                                    st.plotly_chart(fig_imp, use_container_width=True)

                                st.success("Modelo treinado com sucesso!")

                            except Exception as e:
                                st.error(f"Erro ao treinar o modelo: {e}")

                        # Mostrar interface de previsão (tanto se acabou de treinar quanto se já estava treinado)
                        model_available = ('ml_models' in st.session_state and
                                           param_key in st.session_state['ml_models'] and
                                           st.session_state['ml_models'][param_key] is not None)

                        if 'ml_trained' in st.session_state and st.session_state['ml_trained'] and model_available:
                            # Interface para fazer novas previsões
                            st.markdown('<h3 class="section-header">Fazer novas previsões:</h3>',
                                        unsafe_allow_html=True)

                            # Recuperando os parâmetros do modelo
                            ml_info = st.session_state['ml_models'][param_key]
                            features = ml_info['features']
                            scaler_X = ml_info['scaler_X']
                            scaler_y = ml_info['scaler_y']
                            model = ml_info['model']

                            # Criando entradas para cada característica
                            new_data = {}

                            with st.form("prediction_form"):
                                for feature in features:
                                    min_val = df[feature].min()
                                    max_val = df[feature].max()
                                    step = (max_val - min_val) / 100

                                    # Verificando se já temos um valor armazenado
                                    default_val = st.session_state.get('ml_inputs', {}).get(feature, float(
                                        min_val + (max_val - min_val) / 2))

                                    new_val = st.slider(
                                        f"{feature}:",
                                        min_value=float(min_val),
                                        max_value=float(max_val),
                                        value=default_val,
                                        step=float(step)
                                    )

                                    # Armazenando o valor no estado da sessão
                                    if 'ml_inputs' not in st.session_state:
                                        st.session_state['ml_inputs'] = {}
                                    st.session_state['ml_inputs'][feature] = new_val
                                    new_data[feature] = new_val

                                predict_button = st.form_submit_button("Fazer previsão")

                                if predict_button:
                                    # Preparando dados de entrada
                                    input_data = np.array([new_data[f] for f in features]).reshape(1, -1)

                                    # Escalando dados
                                    input_scaled = scaler_X.transform(input_data)

                                    # Fazendo previsão
                                    pred_scaled = model.predict(input_scaled)
                                    prediction = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]

                                    st.markdown(f"""
                                    <div class="highlight">
                                    <h4>Resultado da Previsão:</h4>
                                    <p>{target_param}: <strong>{prediction:.4f}</strong></p>
                                    </div>
                                    """, unsafe_allow_html=True)

            elif ml_opcao == "Otimização de processo":
                st.markdown('<h3 class="section-header">Otimização de Parâmetros do Processo</h3>',
                            unsafe_allow_html=True)
                st.markdown(
                    '<p class="info-text">Encontre os parâmetros ótimos para maximizar a eficiência do processo.</p>',
                    unsafe_allow_html=True)

                # Verificando se há modelos treinados
                if all(model is None for model in st.session_state['ml_models'].values()):
                    st.warning(
                        "Nenhum modelo de machine learning treinado. Por favor, treine modelos na seção 'Previsão de parâmetros' primeiro.")
                else:
                    # Selecionando o objetivo da otimização
                    objetivo = st.selectbox(
                        "Selecione o objetivo da otimização:",
                        ["Maximizar remoção de nitrogênio", "Maximizar remoção de fósforo",
                         "Maximizar remoção de DQO", "Maximizar produção de biomassa"]
                    )

                    # Definindo parâmetros ajustáveis
                    st.markdown('<h4>Parâmetros Ajustáveis:</h4>', unsafe_allow_html=True)

                    tempo = st.slider(
                        "Tempo de tratamento (dias):",
                        min_value=1.0,
                        max_value=15.0,
                        value=10.0,
                        step=0.5
                    )

                    conc_inicial = st.slider(
                        "Concentração inicial de nitrogênio (mg/L):",
                        min_value=50.0,
                        max_value=200.0,
                        value=120.0,
                        step=10.0
                    )

                    if st.button("Otimizar"):
                        st.markdown('<h4>Resultados da Otimização:</h4>', unsafe_allow_html=True)

                        # Simulação simples com base no objetivo
                        if objetivo == "Maximizar remoção de nitrogênio":
                            eficiencia = 95 - 80 * np.exp(-0.2 * tempo)
                            valor_final = conc_inicial * (1 - eficiencia / 100)

                            st.markdown(f"""
                            <div class="highlight">
                            <p>Com os parâmetros selecionados, estima-se:</p>
                            <ul>
                                <li>Eficiência de remoção de nitrogênio: <strong>{eficiencia:.2f}%</strong></li>
                                <li>Concentração final de nitrogênio: <strong>{valor_final:.2f} mg/L</strong></li>
                                <li>Tempo de tratamento: <strong>{tempo} dias</strong></li>
                            </ul>
                            </div>
                            """, unsafe_allow_html=True)

                        elif objetivo == "Maximizar remoção de fósforo":
                            eficiencia = 90 - 75 * np.exp(-0.25 * tempo)
                            valor_final = 30 * (1 - eficiencia / 100)

                            st.markdown(f"""
                            <div class="highlight">
                            <p>Com os parâmetros selecionados, estima-se:</p>
                            <ul>
                                <li>Eficiência de remoção de fósforo: <strong>{eficiencia:.2f}%</strong></li>
                                <li>Concentração final de fósforo: <strong>{valor_final:.2f} mg/L</strong></li>
                                <li>Tempo de tratamento: <strong>{tempo} dias</strong></li>
                            </ul>
                            </div>
                            """, unsafe_allow_html=True)

                        elif objetivo == "Maximizar remoção de DQO":
                            eficiencia = 85 - 70 * np.exp(-0.15 * tempo)
                            valor_final = 5000 * (1 - eficiencia / 100)

                            st.markdown(f"""
                            <div class="highlight">
                            <p>Com os parâmetros selecionados, estima-se:</p>
                            <ul>
                                <li>Eficiência de remoção de DQO: <strong>{eficiencia:.2f}%</strong></li>
                                <li>Concentração final de DQO: <strong>{valor_final:.2f} mg/L</strong></li>
                                <li>Tempo de tratamento: <strong>{tempo} dias</strong></li>
                            </ul>
                            </div>
                            """, unsafe_allow_html=True)

                        elif objetivo == "Maximizar produção de biomassa":
                            biomassa_final = 0.2 + 1.4 * (1 - np.exp(-0.3 * tempo))

                            st.markdown(f"""
                            <div class="highlight">
                            <p>Com os parâmetros selecionados, estima-se:</p>
                            <ul>
                                <li>Produção final de biomassa: <strong>{biomassa_final:.2f} g/L</strong></li>
                                <li>Tempo de cultivo: <strong>{tempo} dias</strong></li>
                            </ul>
                            </div>
                            """, unsafe_allow_html=True)

                        # Gráfico da otimização
                        fig = go.Figure()

                        # Simulando dados para o gráfico
                        t_range = np.linspace(0, 15, 100)

                        if objetivo in ["Maximizar remoção de nitrogênio", "Maximizar remoção de fósforo",
                                        "Maximizar remoção de DQO"]:
                            if objetivo == "Maximizar remoção de nitrogênio":
                                eficiencia_range = 95 - 80 * np.exp(-0.2 * t_range)
                                y_label = "Eficiência de remoção de N (%)"
                            elif objetivo == "Maximizar remoção de fósforo":
                                eficiencia_range = 90 - 75 * np.exp(-0.25 * t_range)
                                y_label = "Eficiência de remoção de P (%)"
                            else:
                                eficiencia_range = 85 - 70 * np.exp(-0.15 * t_range)
                                y_label = "Eficiência de remoção de DQO (%)"

                            fig.add_trace(go.Scatter(
                                x=t_range,
                                y=eficiencia_range,
                                mode='lines',
                                name='Eficiência'
                            ))

                            # Marcando o ponto selecionado
                            if objetivo == "Maximizar remoção de nitrogênio":
                                eficiencia_sel = 95 - 80 * np.exp(-0.2 * tempo)
                            elif objetivo == "Maximizar remoção de fósforo":
                                eficiencia_sel = 90 - 75 * np.exp(-0.25 * tempo)
                            else:
                                eficiencia_sel = 85 - 70 * np.exp(-0.15 * tempo)

                            fig.add_trace(go.Scatter(
                                x=[tempo],
                                y=[eficiencia_sel],
                                mode='markers',
                                marker=dict(size=10, color='red'),
                                name='Ponto selecionado'
                            ))

                            fig.update_layout(
                                title='Eficiência de Remoção vs. Tempo',
                                xaxis_title='Tempo (dias)',
                                yaxis_title=y_label
                            )

                        else:  # Maximizar produção de biomassa
                            biomassa_range = CineticModels.logistic_model(t_range, 0.35, 0.3, 3.0)

                            fig.add_trace(go.Scatter(
                                x=t_range,
                                y=biomassa_range,
                                mode='lines',
                                name='Biomassa'
                            ))

                            # Marcando o ponto selecionado
                            biomassa_sel = CineticModels.logistic_model(tempo, 0.35, 0.3, 3.0)

                            fig.add_trace(go.Scatter(
                                x=[tempo],
                                y=[biomassa_sel],
                                mode='markers',
                                marker=dict(size=10, color='red'),
                                name='Ponto selecionado'
                            ))

                            fig.update_layout(
                                title='Produção de Biomassa vs. Tempo',
                                xaxis_title='Tempo (dias)',
                                yaxis_title='Biomassa (g/L)'
                            )

                        st.plotly_chart(fig, use_container_width=True)

    # Seção de Visualização e Resultados
    elif opcao == "Visualização e Resultados":
        st.markdown('<h2 class="sub-header">Visualização e Resultados</h2>', unsafe_allow_html=True)

        if st.session_state['data'] is None:
            st.warning("Nenhum dado disponível. Por favor, vá para a seção 'Entrada de Dados' primeiro.")
        else:
            df = st.session_state['data']

            st.markdown('<p class="info-text">Visualize os resultados e gere relatórios.</p>', unsafe_allow_html=True)

            viz_opcao = st.radio(
                "Selecione a visualização:",
                ["Dados consolidados", "Comparação entre modelos", "Exportar relatório"]
            )

            if viz_opcao == "Dados consolidados":
                # Gráfico multiplos parâmetros
                st.markdown('<h3 class="section-header">Visualização de Múltiplos Parâmetros:</h3>',
                            unsafe_allow_html=True)

                # Seleção de parâmetros
                params = st.multiselect(
                    "Selecione os parâmetros para visualizar:",
                    df.columns[1:],  # Excluindo a coluna de tempo
                    default=df.columns[1:3]  # Selecionando os dois primeiros por padrão
                )

                if len(params) > 0:
                    # Criando gráfico
                    fig = go.Figure()

                    # Adicionando cada parâmetro
                    for param in params:
                        # Normalizando dados para melhor visualização
                        if param in ["Biomassa Microalga (g/L)", "Biomassa Fungo (g/L)"]:
                            # Para biomassa, normalizar pelo valor máximo
                            norm_data = df[param] / df[param].max()
                            y_axis = "y"
                        else:
                            # Para poluentes, normalizar e inverter (para mostrar remoção)
                            norm_data = 1 - (df[param] / df[param].iloc[0])
                            y_axis = "y2"

                        fig.add_trace(go.Scatter(
                            x=df["Tempo (dias)"],
                            y=norm_data,
                            mode='lines+markers',
                            name=param,
                            yaxis=y_axis
                        ))

                    fig.update_layout(
                        title='Comparação de Parâmetros ao Longo do Tempo',
                        xaxis_title='Tempo (dias)',
                        yaxis=dict(
                            title='Biomassa Normalizada',
                            range=[0, 1.1]
                        ),
                        yaxis2=dict(
                            title='Eficiência de Remoção',
                            overlaying='y',
                            side='right',
                            range=[0, 1.1]
                        ),
                        legend=dict(x=0.02, y=0.98)
                    )

                    st.plotly_chart(fig, use_container_width=True)

                # Calculando eficiências de remoção
                st.markdown('<h3 class="section-header">Eficiências de Remoção:</h3>', unsafe_allow_html=True)

                poluentes = ["Nitrogênio (mg/L)", "Fósforo (mg/L)", "DQO (mg/L)"]
                eficiencias = {}

                for poluente in poluentes:
                    if poluente in df.columns:
                        inicial = df[poluente].iloc[0]
                        final = df[poluente].iloc[-1]
                        eficiencia = 100 * (1 - final / inicial)
                        eficiencias[poluente] = eficiencia

                # Criando gráfico de barras
                if len(eficiencias) > 0:
                    fig = go.Figure()

                    fig.add_trace(go.Bar(
                        x=list(eficiencias.keys()),
                        y=list(eficiencias.values()),
                        text=[f"{val:.2f}%" for val in eficiencias.values()],
                        textposition='auto'
                    ))

                    fig.update_layout(
                        title='Eficiência de Remoção Final',
                        xaxis_title='Parâmetro',
                        yaxis_title='Eficiência (%)',
                        yaxis=dict(range=[0, 100])
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Tabela com valores
                    ef_df = pd.DataFrame({
                        'Parâmetro': list(eficiencias.keys()),
                        'Valor Inicial': [df[p].iloc[0] for p in eficiencias.keys()],
                        'Valor Final': [df[p].iloc[-1] for p in eficiencias.keys()],
                        'Eficiência (%)': [eficiencias[p] for p in eficiencias.keys()]
                    })

                    st.dataframe(ef_df)

            elif viz_opcao == "Comparação entre modelos":
                # Verificação de modelos ajustados
                if all(model is None for model in st.session_state['model_params'].values()):
                    st.warning(
                        "Nenhum modelo ajustado. Por favor, ajuste modelos na seção 'Modelagem Cinética' primeiro.")
                else:
                    st.markdown('<h3 class="section-header">Comparação entre Modelos:</h3>', unsafe_allow_html=True)

                    # Seleção de parâmetro
                    parametro = st.selectbox(
                        "Selecione o parâmetro para comparar modelos:",
                        ["Biomassa Microalga (g/L)", "Biomassa Fungo (g/L)", "Nitrogênio (mg/L)", "Fósforo (mg/L)",
                         "DQO (mg/L)"]
                    )

                    param_key_map = {
                        "Biomassa Microalga (g/L)": "microalga",
                        "Biomassa Fungo (g/L)": "fungo",
                        "Nitrogênio (mg/L)": "nitrogenio",
                        "Fósforo (mg/L)": "fosforo",
                        "DQO (mg/L)": "dqo"
                    }

                    param_key = param_key_map[parametro]

                    if st.session_state['model_params'][param_key] is not None:
                        # Dados experimentais
                        time_data = df["Tempo (dias)"].values
                        param_data = df[parametro].values

                        # Criando gráfico
                        fig = go.Figure()

                        # Dados experimentais
                        fig.add_trace(go.Scatter(
                            x=time_data,
                            y=param_data,
                            mode='markers',
                            name='Dados Experimentais'
                        ))

                        # Tempos para previsão
                        t_fine = np.linspace(min(time_data), max(time_data), 100)

                        # Adicionando modelos ajustados
                        for model_key, params in st.session_state['model_params'][param_key].items():
                            if model_key == 'logistic':
                                X0, umax, Xmax = params['X0'], params['umax'], params['Xmax']
                                y_pred = CineticModels.logistic_model(t_fine, X0, umax, Xmax)

                                fig.add_trace(go.Scatter(
                                    x=t_fine,
                                    y=y_pred,
                                    mode='lines',
                                    name='Modelo Logístico'
                                ))

                            elif model_key == 'logistic_inverted':
                                S0, k, Smax = params['S0'], params['k'], params['Smax']
                                y_pred = CineticModels.logistic_model(t_fine, S0, k, Smax)

                                fig.add_trace(go.Scatter(
                                    x=t_fine,
                                    y=y_pred,
                                    mode='lines',
                                    name='Modelo de Decaimento'
                                ))

                        fig.update_layout(
                            title=f'Comparação de Modelos - {parametro}',
                            xaxis_title='Tempo (dias)',
                            yaxis_title=parametro,
                            legend=dict(x=0.02, y=0.98)
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        # Métricas de ajuste
                        st.markdown('<h3 class="section-header">Métricas de Ajuste:</h3>', unsafe_allow_html=True)

                        # Calculando métricas para cada modelo
                        metrics = []

                        for model_key, params in st.session_state['model_params'][param_key].items():
                            if model_key == 'logistic':
                                X0, umax, Xmax = params['X0'], params['umax'], params['Xmax']
                                y_pred = CineticModels.logistic_model(time_data, X0, umax, Xmax)

                            elif model_key == 'logistic_inverted':
                                S0, k, Smax = params['S0'], params['k'], params['Smax']
                                y_pred = CineticModels.logistic_model(time_data, S0, k, Smax)

                            else:
                                continue

                            mse = mean_squared_error(param_data, y_pred)
                            r2 = r2_score(param_data, y_pred)

                            metrics.append({
                                'Modelo': model_key,
                                'MSE': mse,
                                'R²': r2
                            })

                        if len(metrics) > 0:
                            metrics_df = pd.DataFrame(metrics)
                            st.dataframe(metrics_df)
                    else:
                        st.warning(f"Nenhum modelo ajustado para {parametro}.")

            elif viz_opcao == "Exportar relatório":
                st.markdown('<h3 class="section-header">Exportar Relatório:</h3>', unsafe_allow_html=True)

                # Opções de exportação
                incluir_dados = st.checkbox("Incluir dados brutos", value=True)
                incluir_modelos = st.checkbox("Incluir parâmetros dos modelos", value=True)
                incluir_graficos = st.checkbox("Incluir gráficos", value=True)
                incluir_eficiencias = st.checkbox("Incluir eficiências de remoção", value=True)

                if st.button("Gerar relatório"):
                    with st.spinner("Gerando relatório..."):
                        # Criando relatório em markdown
                        report = []

                        # Cabeçalho
                        report.append("# Relatório de Tratamento Terciário do Soro de Leite")
                        report.append("## Modelagem Cinética e Análise de Eficiência")
                        report.append(f"Data: {pd.Timestamp.now().strftime('%d/%m/%Y')}")
                        report.append("\n")

                        # Dados brutos
                        if incluir_dados:
                            report.append("## Dados Experimentais")
                            report.append("\n```")
                            report.append(df.to_string())
                            report.append("```\n")

                        # Parâmetros dos modelos
                        if incluir_modelos:
                            report.append("## Parâmetros dos Modelos")

                            for param_key, models in st.session_state['model_params'].items():
                                if models is not None:
                                    report.append(f"\n### {param_key.capitalize()}")

                                    for model_key, params in models.items():
                                        report.append(f"\n#### Modelo {model_key.capitalize()}")

                                        for param_name, param_value in params.items():
                                            report.append(f"- {param_name}: {param_value:.4f}")

                            report.append("\n")

                        # Eficiências de remoção
                        if incluir_eficiencias:
                            report.append("## Eficiências de Remoção")

                            poluentes = ["Nitrogênio (mg/L)", "Fósforo (mg/L)", "DQO (mg/L)"]
                            for poluente in poluentes:
                                if poluente in df.columns:
                                    inicial = df[poluente].iloc[0]
                                    final = df[poluente].iloc[-1]
                                    eficiencia = 100 * (1 - final / inicial)

                                    report.append(f"\n### {poluente}")
                                    report.append(f"- Valor inicial: {inicial:.2f} mg/L")
                                    report.append(f"- Valor final: {final:.2f} mg/L")
                                    report.append(f"- Eficiência de remoção: {eficiencia:.2f}%")

                            report.append("\n")

                        # Gráficos
                        if incluir_graficos:
                            report.append("## Gráficos\n")
                            report.append("Gráficos disponíveis na interface do software.")

                        # Conclusão
                        report.append("\n## Conclusão")
                        report.append("Este relatório foi gerado automaticamente pelo software BioCine.")

                        # Unindo o relatório
                        report_text = "\n".join(report)

                        # Criando download link
                        b64 = base64.b64encode(report_text.encode()).decode()
                        href = f'<a href="data:text/plain;base64,{b64}" download="relatorio_tratamento.md">Download do Relatório (Markdown)</a>'

                        st.markdown(href, unsafe_allow_html=True)

                        # Exibindo prévia
                        st.markdown("### Prévia do relatório:")
                        st.text_area("", report_text, height=300)

    # Seção Sobre o Software
    elif opcao == "Sobre o Software":
        st.markdown('<h2 class="sub-header">Sobre o Software</h2>', unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("""
            <div class="highlight">
            <h3>BioCine - Software de Modelagem Cinética</h3>
            <p>O BioCine é um software especializado para modelagem cinética e previsão do processo de tratamento 
            terciário em batelada/semicontínuo do soro do leite utilizando microalgas e fungos filamentosos.</p>

            <h4>Principais funcionalidades:</h4>
            <ul>
                <li>Importação e gerenciamento de dados experimentais</li>
                <li>Modelagem cinética com modelos de Monod e Logístico</li>
                <li>Análise de eficiência de remoção de poluentes</li>
                <li>Previsão e otimização do processo usando machine learning</li>
                <li>Visualização interativa de resultados</li>
                <li>Exportação de relatórios</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="highlight">
            <h3>Embasamento Teórico</h3>
            <p>Este software baseia-se em estudos sobre modelagem cinética aplicada ao tratamento biológico 
            de efluentes, com foco no soro de leite. Os modelos implementados (Monod e Logístico) são 
            amplamente utilizados para descrever o crescimento microbiano e a remoção de poluentes.</p>

            <p>Referências principais:</p>
            <ul>
                <li>He, Y. et al. (2016). Analysis and model delineation of marine microalgae growth and 
                lipid accumulation in flat-plate photobioreactor.</li>
                <li>Soares, A.P.M.R. et al. (2020). Random Forest as a promising application to predict 
                basic-dye biosorption process using orange waste.</li>
                <li>Maltsev, Y. & Maltseva, K. (2021). Fatty acids of microalgae: Diversity and applications.</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<h3 class="section-header">Modelagem Cinética:</h3>', unsafe_allow_html=True)

        # Explicação dos modelos
        st.markdown("""
        <div class="highlight">
        <h4>Modelo de Monod</h4>
        <p>O modelo de Monod descreve a relação entre a taxa específica de crescimento dos micro-organismos e a 
        concentração de substrato limitante. A equação básica é:</p>
        <p style="text-align: center;"><strong>μ = μmax * S / (Ks + S)</strong></p>
        <p>Onde:</p>
        <ul>
            <li>μ: Taxa específica de crescimento</li>
            <li>μmax: Taxa máxima específica de crescimento</li>
            <li>S: Concentração de substrato</li>
            <li>Ks: Constante de meia saturação (concentração de substrato onde μ = μmax/2)</li>
        </ul>
        </div>

        <div class="highlight" style="margin-top: 20px;">
        <h4>Modelo Logístico</h4>
        <p>O modelo Logístico descreve o crescimento de populações biológicas em ambientes onde os recursos são 
        limitados. A equação básica é:</p>
        <p style="text-align: center;"><strong>dX/dt = μmax * X * (1 - X/Xmax)</strong></p>
        <p>Ou em sua forma integrada:</p>
        <p style="text-align: center;"><strong>X(t) = Xmax / (1 + ((Xmax - X0) / X0) * exp(-μmax * t))</strong></p>
        <p>Onde:</p>
        <ul>
            <li>X: Concentração de biomassa</li>
            <li>X0: Concentração inicial de biomassa</li>
            <li>Xmax: Capacidade máxima do sistema</li>
            <li>μmax: Taxa máxima de crescimento</li>
            <li>t: Tempo</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)


# Executando a aplicação
if __name__ == "__main__":
    main()