"""
Pacote de utilitários do BioCine

Este pacote contém módulos de utilidades para processamento de dados,
visualização e geração de relatórios.
"""

from utils.data_processor import (
    import_csv,
    generate_example_data,
    validate_data,
    preprocess_data,
    calculate_removal_efficiency,
    save_processed_data,
    load_processed_data
)

from utils.visualization import (
    plot_growth_curve,
    plot_substrate_consumption,
    plot_removal_efficiency,
    plot_correlation_matrix,
    plot_feature_importance,
    plot_model_comparison,
    plot_interactive_growth_curve,
    plot_interactive_combined
)

from utils.report_generator import (
    generate_model_summary,
    generate_ml_summary,
    generate_data_summary,
    export_report_to_html,
    export_results_to_excel
)

from utils.pdp_utils import validate_data_for_pdp, calculate_partial_dependence_safely, plot_partial_dependence_safely

__all__ = [
    # Data Processor
    'import_csv',
    'generate_example_data',
    'validate_data',
    'preprocess_data',
    'calculate_removal_efficiency',
    'save_processed_data',
    'load_processed_data',

    # Visualization
    'plot_growth_curve',
    'plot_substrate_consumption',
    'plot_removal_efficiency',
    'plot_correlation_matrix',
    'plot_feature_importance',
    'plot_model_comparison',
    'plot_interactive_growth_curve',
    'plot_interactive_combined',

    # Report Generator
    'generate_model_summary',
    'generate_ml_summary',
    'generate_data_summary',
    'export_report_to_html',
    'export_results_to_excel',

    # PDP Utils
    'validate_data_for_pdp',
    'calculate_partial_dependence_safely',
    'plot_partial_dependence_safely'
]