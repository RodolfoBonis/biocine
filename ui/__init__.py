"""
Pacote de interface de usuário do BioCine

Este pacote contém os módulos de interface do usuário para a aplicação Streamlit.
"""

from ui.pages.home import show_home
from ui.pages.data_input import show_data_input
from ui.pages.modeling import show_modeling
from ui.pages.ml import show_ml
from ui.pages.results import show_results
from ui.pages.about import show_about
from ui.pages.correlation_analysis import show_correlation_analysis
from ui.pages.optimization import show_advanced_optimization
from ui.pages.parameter_interaction import show_parameter_interaction

__all__ = [
    'show_home',
    'show_data_input',
    'show_modeling',
    'show_ml',
    'show_results',
    'show_about',
    'show_correlation_analysis',
    'show_advanced_optimization',
    'show_parameter_interaction'
]