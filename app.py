import streamlit as st
import yaml
import os
import sys

# Adiciona os caminhos ao PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importa os módulos da UI
from ui import show_home, show_data_input, show_modeling, show_ml, show_results, show_about


# Carrega configurações
def load_config():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


# Configuração da página
def setup_page():
    st.set_page_config(
        page_title="BioCine - Modelagem Cinética de Bioprocessos",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Estilo CSS personalizado
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #4CAF50;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #2E7D32;
            margin-bottom: 1rem;
        }
        .info-box {
            background-color: #3498db;
            border-radius: 5px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)


def main():
    # Carrega configurações
    config = load_config()
    setup_page()

    # Barra lateral
    st.sidebar.image("assets/logo.png", width=250)
    st.sidebar.title("BioCine")
    st.sidebar.subheader("Modelagem Cinética de Bioprocessos")

    # Menu de navegação
    menu_options = [
        "Home",
        "Entrada de Dados",
        "Modelagem Cinética",
        "Machine Learning",
        "Resultados",
        "Sobre"
    ]

    selection = st.sidebar.radio("Navegação", menu_options)

    # Inicializa o estado da sessão se necessário
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'models' not in st.session_state:
        st.session_state.models = {}
    if 'ml_models' not in st.session_state:
        st.session_state.ml_models = {}
    if 'results' not in st.session_state:
        st.session_state.results = {}

    # Renderiza a página selecionada
    if selection == "Home":
        show_home()
    elif selection == "Entrada de Dados":
        show_data_input()
    elif selection == "Modelagem Cinética":
        show_modeling()
    elif selection == "Machine Learning":
        show_ml()
    elif selection == "Resultados":
        show_results()
    elif selection == "Sobre":
        show_about()


if __name__ == "__main__":
    main()