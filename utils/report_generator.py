"""
Geração de relatórios com os resultados da modelagem

Este módulo implementa funções para geração de relatórios com os
resultados da modelagem cinética e de machine learning.
"""

import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from jinja2 import Template


def generate_model_summary(model, data=None, predictions=None):
    """
    Gera um resumo do modelo cinético

    Args:
        model: Modelo cinético
        data: Dictionary com os dados experimentais (opcional)
        predictions: Dictionary com as previsões do modelo (opcional)

    Returns:
        Dictionary com o resumo do modelo
    """
    # Informações básicas
    summary = {
        "model_name": model.name,
        "parameters": model.get_params(),
        "fitted_params": model.get_fitted_params(),
        "metrics": model.get_metrics()
    }

    return summary


def generate_ml_summary(ml_model):
    """
    Gera um resumo do modelo de machine learning

    Args:
        ml_model: Modelo de machine learning

    Returns:
        Dictionary com o resumo do modelo
    """
    # Informações básicas
    summary = {
        "model_name": ml_model.name,
        "features": ml_model.features,
        "target": ml_model.target,
        "metrics": ml_model.get_metrics()
    }

    # Adiciona a importância das características, se disponível
    try:
        summary["feature_importance"] = ml_model.get_feature_importance()
    except (NotImplementedError, AttributeError):
        summary["feature_importance"] = None

    return summary


def export_report_to_html(models_summary, ml_summary=None, data_summary=None,
                          figures=None, output_path="report.html"):
    """
    Exporta um relatório em formato HTML

    Args:
        models_summary: Dictionary com os resumos dos modelos cinéticos
        ml_summary: Dictionary com os resumos dos modelos de ML (opcional)
        data_summary: Dictionary com o resumo dos dados (opcional)
        figures: Dictionary com as figuras (opcional)
        output_path: Caminho para o arquivo HTML

    Returns:
        True se a exportação foi bem-sucedida, False caso contrário
    """
    # Template HTML
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Relatório de Modelagem Cinética</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                color: #333;
            }
            h1, h2, h3 {
                color: #2C3E50;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
            }
            .section {
                margin-bottom: 30px;
                padding: 20px;
                background-color: #f9f9f9;
                border-radius: 5px;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }
            th, td {
                padding: 10px;
                border: 1px solid #ddd;
                text-align: left;
            }
            th {
                background-color: #f2f2f2;
            }
            .figure {
                margin: 20px 0;
                text-align: center;
            }
            .figure img {
                max-width: 100%;
                height: auto;
            }
            .header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
            }
            .header-info {
                text-align: right;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Relatório de Modelagem Cinética</h1>
                <div class="header-info">
                    <p><strong>Data:</strong> {{ date }}</p>
                </div>
            </div>

            {% if data_summary %}
            <div class="section">
                <h2>Resumo dos Dados</h2>
                <p><strong>Número de amostras:</strong> {{ data_summary.n_samples }}</p>
                {% if data_summary.columns %}
                <p><strong>Colunas:</strong> {{ data_summary.columns|join(', ') }}</p>
                {% endif %}

                {% if data_summary.statistics %}
                <h3>Estatísticas Descritivas</h3>
                <table>
                    <tr>
                        <th>Variável</th>
                        <th>Média</th>
                        <th>Desvio Padrão</th>
                        <th>Mínimo</th>
                        <th>Máximo</th>
                    </tr>
                    {% for var, stats in data_summary.statistics.items() %}
                    <tr>
                        <td>{{ var }}</td>
                        <td>{{ "%.4f"|format(stats.mean) }}</td>
                        <td>{{ "%.4f"|format(stats.std) }}</td>
                        <td>{{ "%.4f"|format(stats.min) }}</td>
                        <td>{{ "%.4f"|format(stats.max) }}</td>
                    </tr>
                    {% endfor %}
                </table>
                {% endif %}
            </div>
            {% endif %}

            <div class="section">
                <h2>Modelos Cinéticos</h2>
                {% for model_name, summary in models_summary.items() %}
                <h3>{{ summary.model_name }}</h3>

                <h4>Parâmetros</h4>
                <table>
                    <tr>
                        <th>Parâmetro</th>
                        <th>Valor</th>
                    </tr>
                    {% for param, value in summary.parameters.items() %}
                    <tr>
                        <td>{{ param }}</td>
                        <td>{{ "%.4f"|format(value) if value is number else value }}</td>
                    </tr>
                    {% endfor %}
                </table>

                {% if summary.fitted_params %}
                <h4>Parâmetros Ajustados</h4>
                <table>
                    <tr>
                        <th>Parâmetro</th>
                        <th>Valor</th>
                    </tr>
                    {% for param, value in summary.fitted_params.items() %}
                    <tr>
                        <td>{{ param }}</td>
                        <td>{{ "%.4f"|format(value) if value is number else value }}</td>
                    </tr>
                    {% endfor %}
                </table>
                {% endif %}

                {% if summary.metrics %}
                <h4>Métricas</h4>
                <table>
                    <tr>
                        <th>Métrica</th>
                        <th>Valor</th>
                    </tr>
                    {% for metric, value in summary.metrics.items() %}
                    <tr>
                        <td>{{ metric }}</td>
                        <td>{{ "%.4f"|format(value) if value is number else value }}</td>
                    </tr>
                    {% endfor %}
                </table>
                {% endif %}

                {% endfor %}
            </div>

            {% if ml_summary %}
            <div class="section">
                <h2>Modelos de Machine Learning</h2>
                {% for model_name, summary in ml_summary.items() %}
                <h3>{{ summary.model_name }}</h3>

                <p><strong>Características:</strong> {{ summary.features|join(', ') }}</p>
                <p><strong>Alvo:</strong> {{ summary.target }}</p>

                {% if summary.metrics %}
                <h4>Métricas</h4>
                <table>
                    <tr>
                        <th>Métrica</th>
                        <th>Valor</th>
                    </tr>
                    {% for metric, value in summary.metrics.items() %}
                    <tr>
                        <td>{{ metric }}</td>
                        <td>{{ "%.4f"|format(value) if value is number else value }}</td>
                    </tr>
                    {% endfor %}
                </table>
                {% endif %}

                {% if summary.feature_importance is not none %}
                <h4>Importância das Características</h4>
                <table>
                    <tr>
                        <th>Característica</th>
                        <th>Importância</th>
                    </tr>
                    {% for _, row in summary.feature_importance.iterrows() %}
                    <tr>
                        <td>{{ row.Feature }}</td>
                        <td>{{ "%.4f"|format(row.Importance) }}</td>
                    </tr>
                    {% endfor %}
                </table>
                {% endif %}

                {% endfor %}
            </div>
            {% endif %}

            {% if figures %}
            <div class="section">
                <h2>Figuras</h2>
                {% for fig_name, fig_path in figures.items() %}
                <div class="figure">
                    <h3>{{ fig_name }}</h3>
                    <img src="{{ fig_path }}" alt="{{ fig_name }}">
                </div>
                {% endfor %}
            </div>
            {% endif %}

            <div class="section">
                <h2>Conclusões</h2>
                <p>Este relatório apresenta os resultados da modelagem cinética do processo de tratamento terciário em batelada/semicontínuo do soro do leite por microalgas e fungos filamentosos.</p>

                {% if models_summary %}
                <p>Os modelos cinéticos utilizados foram:
                    <ul>
                    {% for model_name in models_summary.keys() %}
                        <li>{{ model_name }}</li>
                    {% endfor %}
                    </ul>
                </p>
                {% endif %}

                {% if ml_summary %}
                <p>Os modelos de machine learning utilizados foram:
                    <ul>
                    {% for model_name in ml_summary.keys() %}
                        <li>{{ model_name }}</li>
                    {% endfor %}
                    </ul>
                </p>
                {% endif %}

                <p>Para mais detalhes sobre a metodologia e resultados, consulte o relatório completo.</p>
            </div>
        </div>
    </body>
    </html>
    """

    try:
        # Cria o diretório se não existir
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Salva as figuras, se disponíveis
        figures_paths = {}
        if figures:
            figures_dir = os.path.join(os.path.dirname(output_path), "figures")
            os.makedirs(figures_dir, exist_ok=True)

            for fig_name, fig in figures.items():
                fig_path = os.path.join(figures_dir, f"{fig_name}.png")
                fig.savefig(fig_path, bbox_inches='tight')
                figures_paths[fig_name] = os.path.relpath(fig_path, os.path.dirname(output_path))

        # Renderiza o template
        template = Template(html_template)
        html_content = template.render(
            date=datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            models_summary=models_summary,
            ml_summary=ml_summary,
            data_summary=data_summary,
            figures=figures_paths
        )

        # Salva o arquivo HTML
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return True
    except Exception as e:
        print(f"Erro ao exportar o relatório: {str(e)}")
        return False


def export_results_to_excel(models_summary, ml_summary=None, data=None,
                            predictions=None, output_path="results.xlsx"):
    """
    Exporta os resultados em formato Excel

    Args:
        models_summary: Dictionary com os resumos dos modelos cinéticos
        ml_summary: Dictionary com os resumos dos modelos de ML (opcional)
        data: DataFrame com os dados experimentais (opcional)
        predictions: Dictionary com as previsões dos modelos (opcional)
        output_path: Caminho para o arquivo Excel

    Returns:
        True se a exportação foi bem-sucedida, False caso contrário
    """
    try:
        # Cria o diretório se não existir
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Cria um escritor Excel
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Dados experimentais
            if data is not None:
                data.to_excel(writer, sheet_name='Dados Experimentais', index=False)

            # Previsões dos modelos
            if predictions is not None:
                for model_name, pred in predictions.items():
                    if isinstance(pred, dict):
                        # Se pred é um dicionário, cria um DataFrame
                        pred_df = pd.DataFrame(pred)
                    elif isinstance(pred, (list, np.ndarray)):
                        # Se pred é uma lista ou array, converte para DataFrame
                        pred_df = pd.DataFrame({f"{model_name}_prediction": pred})
                    else:
                        continue

                    pred_df.to_excel(writer, sheet_name=f'Previsão_{model_name}', index=False)

            # Parâmetros dos modelos cinéticos
            if models_summary:
                # Cria um DataFrame com os parâmetros
                params_data = []
                for model_name, summary in models_summary.items():
                    for param_name, param_value in summary.get('parameters', {}).items():
                        params_data.append({
                            'Modelo': model_name,
                            'Parâmetro': param_name,
                            'Valor': param_value
                        })

                if params_data:
                    params_df = pd.DataFrame(params_data)
                    params_df.to_excel(writer, sheet_name='Parâmetros', index=False)

                # Cria um DataFrame com as métricas
                metrics_data = []
                for model_name, summary in models_summary.items():
                    for metric_name, metric_value in summary.get('metrics', {}).items():
                        metrics_data.append({
                            'Modelo': model_name,
                            'Métrica': metric_name,
                            'Valor': metric_value
                        })

                if metrics_data:
                    metrics_df = pd.DataFrame(metrics_data)
                    metrics_df.to_excel(writer, sheet_name='Métricas', index=False)

            # Resumo dos modelos de machine learning
            if ml_summary:
                for model_name, summary in ml_summary.items():
                    # Métricas
                    if summary.get('metrics'):
                        ml_metrics = pd.DataFrame([summary['metrics']])
                        ml_metrics.to_excel(writer, sheet_name=f'ML_{model_name}_Métricas', index=False)

                    # Importância das características
                    if summary.get('feature_importance') is not None:
                        summary['feature_importance'].to_excel(
                            writer, sheet_name=f'ML_{model_name}_Importância', index=False
                        )

        return True
    except Exception as e:
        print(f"Erro ao exportar os resultados para Excel: {str(e)}")
        return False


def generate_data_summary(data):
    """
    Gera um resumo dos dados experimentais

    Args:
        data: DataFrame com os dados

    Returns:
        Dictionary com o resumo dos dados
    """
    if data is None or data.empty:
        return None

    # Resumo básico
    summary = {
        "n_samples": len(data),
        "columns": list(data.columns)
    }

    # Estatísticas descritivas
    statistics = {}
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns

    for col in numeric_cols:
        statistics[col] = {
            "mean": data[col].mean(),
            "std": data[col].std(),
            "min": data[col].min(),
            "max": data[col].max()
        }

    summary["statistics"] = statistics

    return summary