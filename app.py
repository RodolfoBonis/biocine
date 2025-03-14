import streamlit as st
import yaml
import os
import sys

# Adiciona os caminhos ao PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importa os módulos da UI
from ui import show_home, show_data_input, show_modeling, show_ml, show_results, show_about
from ui.pages.help import (
    show_help_home,
    show_monod_help,
    show_logistic_help,
    show_ml_help,
    show_parameters_help,
    show_workflow_help,
    show_export_help,
    show_context_help
)


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
        .help-button {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background-color: #f1f1f1;
            color: #0066cc;
            border-radius: 50%;
            width: 24px;
            height: 24px;
            text-align: center;
            font-weight: bold;
            cursor: pointer;
            margin-left: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
            transition: all 0.3s cubic-bezier(.25,.8,.25,1);
        }
        .help-button:hover {
            box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);
            background-color: #e0e0e0;
        }
        </style>
    """, unsafe_allow_html=True)


def main():
    # Carrega configurações
    config = load_config()
    setup_page()

    # Inicializa o estado da sessão se necessário
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'models' not in st.session_state:
        st.session_state.models = {}
    if 'ml_models' not in st.session_state:
        st.session_state.ml_models = {}
    if 'results' not in st.session_state:
        st.session_state.results = {}
    if 'show_help' not in st.session_state:
        st.session_state.show_help = False
    if 'help_context' not in st.session_state:
        st.session_state.help_context = None

    # Barra lateral
    st.sidebar.image("assets/logo.png", width=250)
    st.sidebar.title("BioCine")
    st.sidebar.subheader("Modelagem Cinética de Bioprocessos")

    # Menu de navegação principal
    menu_options = [
        "Home",
        "Entrada de Dados",
        "Modelagem Cinética",
        "Machine Learning",
        "Resultados",
        "Sobre"
    ]

    selection = st.sidebar.radio("Navegação", menu_options)

    # Menu de ajuda na barra lateral
    st.sidebar.markdown("---")
    st.sidebar.subheader("Ajuda e Documentação")

    help_options = [
        "Nenhum (Mostrar Aplicativo)",
        "Visão Geral",
        "Modelo de Monod",
        "Modelo Logístico",
        "Machine Learning",
        "Parâmetros de Configuração",
        "Fluxo de Trabalho",
        "Exportação de Resultados",
    ]

    # Define a seleção do menu de ajuda com base no help_context da sessão
    if st.session_state.show_help and st.session_state.help_context:
        help_selection = {
            "monod": "Modelo de Monod",
            "logistic": "Modelo Logístico",
            "ml": "Machine Learning",
            "export": "Exportação de Resultados",
            "modeling": "Ajuda Contextual"
        }.get(st.session_state.help_context, "Ajuda Contextual")
    else:
        help_selection = st.sidebar.selectbox("Escolha um tópico de ajuda", help_options)

    # Botão para sair da ajuda contextual quando estiver ativa
    if st.session_state.show_help:
        if st.sidebar.button("Voltar para o Aplicativo"):
            st.session_state.show_help = False
            st.session_state.help_context = None
            return st.rerun()

    # Se uma opção de ajuda foi selecionada na barra lateral, mostrar essa ajuda
    if help_selection != "Nenhum (Mostrar Aplicativo)" and help_selection != "Ajuda Contextual":
        if help_selection == "Visão Geral":
            show_help_home()
        elif help_selection == "Modelo de Monod":
            show_monod_help()
        elif help_selection == "Modelo Logístico":
            show_logistic_help()
        elif help_selection == "Machine Learning":
            show_ml_help()
        elif help_selection == "Parâmetros de Configuração":
            show_parameters_help()
        elif help_selection == "Fluxo de Trabalho":
            show_workflow_help()
        elif help_selection == "Exportação de Resultados":
            show_export_help()
        return  # Retorna após mostrar a ajuda

    # Se a ajuda contextual estiver ativa, mostrar a página de ajuda correspondente
    if st.session_state.show_help and st.session_state.help_context:
        context = st.session_state.help_context
        if context == "monod":
            show_monod_help()
        elif context == "logistic":
            show_logistic_help()
        elif context == "ml":
            show_ml_help()
        elif context == "export":
            show_export_help()
        else:
            show_context_help(context)
        return  # Retorna após mostrar a ajuda

    # Se nenhuma ajuda foi selecionada, renderiza a página selecionada no menu principal
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