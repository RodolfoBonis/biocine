import pandas as pd
import numpy as np
import io
import base64


class DataProcessor:
    """
    Classe para processamento e manipulação de dados experimentais
    """

    @staticmethod
    def load_csv(file_path):
        """
        Carrega dados de um arquivo CSV

        Args:
            file_path: Caminho do arquivo

        Returns:
            DataFrame com os dados
        """
        try:
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            raise Exception(f"Erro ao carregar o arquivo CSV: {e}")

    @staticmethod
    def create_example_data():
        """
        Cria um conjunto de dados de exemplo para demonstração

        Returns:
            DataFrame com dados de exemplo
        """
        # Tempos (dias)
        time = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        # Biomassa do consórcio microalga+fungo (g/L)
        # Combinando as biomassas da microalga e do fungo em um único consórcio
        consorcio_biomass = np.array([0.35, 0.43, 0.60, 0.90, 1.50, 2.10, 2.55, 2.80, 2.90, 2.96, 3.00])

        # Nitrogênio (mg/L)
        nitrogenio = np.array([120, 115, 105, 90, 70, 50, 35, 25, 20, 18, 15])

        # Fósforo (mg/L)
        fosforo = np.array([30, 28, 25, 22, 18, 14, 10, 8, 7, 6, 5])

        # DQO (mg/L)
        dqo = np.array([5000, 4800, 4500, 4000, 3300, 2500, 1800, 1400, 1200, 1100, 1000])

        # Criar DataFrame
        df = pd.DataFrame({
            'Tempo (dias)': time,
            'Biomassa Consórcio (g/L)': consorcio_biomass,
            'Nitrogênio (mg/L)': nitrogenio,
            'Fósforo (mg/L)': fosforo,
            'DQO (mg/L)': dqo
        })

        return df

    @staticmethod
    def map_columns(df, column_mapping):
        """
        Renomeia as colunas de acordo com o mapeamento especificado

        Args:
            df: DataFrame com os dados
            column_mapping: Dicionário com o mapeamento {nome_original: novo_nome}

        Returns:
            DataFrame com colunas renomeadas
        """
        # Verificando se todas as colunas existem
        for col in column_mapping.keys():
            if col not in df.columns:
                raise Exception(f"Coluna '{col}' não encontrada no DataFrame")

        # Renomeando colunas
        df_renamed = df.rename(columns=column_mapping)

        return df_renamed

    @staticmethod
    def validate_data(df, required_columns=None):
        """
        Valida o DataFrame quanto à presença das colunas necessárias

        Args:
            df: DataFrame com os dados
            required_columns: Lista de colunas necessárias

        Returns:
            bool: True se válido, False caso contrário
        """
        if required_columns is None:
            required_columns = [
                "Tempo (dias)",
                "Biomassa Consórcio (g/L)",
                "Nitrogênio (mg/L)",
                "Fósforo (mg/L)",
                "DQO (mg/L)"
            ]

        # Verificando colunas
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            return False, f"Colunas ausentes: {', '.join(missing_columns)}"

        # Verificando tipos de dados
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                return False, f"A coluna '{col}' contém valores não numéricos"

        # Verificando valores ausentes
        if df.isnull().any().any():
            return False, "O DataFrame contém valores ausentes"

        return True, "Dados válidos"

    @staticmethod
    def preprocess_data(df):
        """
        Realiza o pré-processamento dos dados

        Args:
            df: DataFrame com os dados

        Returns:
            DataFrame processado
        """
        # Copiando para não modificar o original
        df_processed = df.copy()

        # Ordenando por tempo
        if "Tempo (dias)" in df_processed.columns:
            df_processed = df_processed.sort_values(by="Tempo (dias)")

        # Removendo valores duplicados
        df_processed = df_processed.drop_duplicates()

        # Convertendo para tipos numéricos (forçando conversão)
        for col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')

        # Tratando valores ausentes (interpolação)
        df_processed = df_processed.interpolate(method='linear')

        return df_processed

    @staticmethod
    def calculate_derived_parameters(df):
        """
        Calcula parâmetros derivados dos dados originais

        Args:
            df: DataFrame com os dados

        Returns:
            DataFrame com parâmetros adicionais
        """
        # Copiando para não modificar o original
        df_derived = df.copy()

        # Calculando eficiências de remoção
        for col in ["Nitrogênio (mg/L)", "Fósforo (mg/L)", "DQO (mg/L)"]:
            if col in df_derived.columns:
                valor_inicial = df_derived[col].iloc[0]
                df_derived[f"Remoção {col} (%)"] = 100 * (1 - df_derived[col] / valor_inicial)

        # Calculando produtividade de biomassa
        for col in ["Biomassa Microalga (g/L)", "Biomassa Fungo (g/L)"]:
            if col in df_derived.columns:
                df_derived[f"Produtividade {col} (g/L/dia)"] = df_derived[col].diff() / df_derived[
                    "Tempo (dias)"].diff()
                # Substituindo NaN por 0 no primeiro ponto
                df_derived[f"Produtividade {col} (g/L/dia)"].iloc[0] = 0

        return df_derived

    @staticmethod
    def create_download_link(df, filename):
        """
        Cria um link para download do DataFrame como CSV

        Args:
            df: DataFrame
            filename: Nome do arquivo

        Returns:
            HTML do link de download
        """
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
        return href

    @staticmethod
    def generate_report_markdown(df, model_params=None, removal_efficiencies=None, ml_metrics=None):
        """
        Gera um relatório em formato Markdown com os resultados

        Args:
            df: DataFrame com os dados
            model_params: Dicionário com parâmetros dos modelos
            removal_efficiencies: Dicionário com eficiências de remoção
            ml_metrics: Dicionário com métricas de ML

        Returns:
            Texto do relatório em formato Markdown
        """
        report = []

        # Cabeçalho
        report.append("# Relatório de Tratamento Terciário do Soro de Leite")
        report.append("## Modelagem Cinética e Análise de Eficiência")
        report.append(f"Data: {pd.Timestamp.now().strftime('%d/%m/%Y')}")
        report.append("\n")

        # Resumo dos dados
        report.append("## Resumo dos Dados")
        report.append("\n")

        # Número de pontos
        report.append(f"* **Número de pontos experimentais:** {len(df)}")

        # Tempo total
        if "Tempo (dias)" in df.columns:
            tempo_total = df["Tempo (dias)"].max() - df["Tempo (dias)"].min()
            report.append(f"* **Tempo total de tratamento:** {tempo_total:.1f} dias")

        # Biomassas
        for col in ["Biomassa Microalga (g/L)", "Biomassa Fungo (g/L)"]:
            if col in df.columns:
                biomassa_inicial = df[col].iloc[0]
                biomassa_final = df[col].iloc[-1]
                aumento = biomassa_final - biomassa_inicial
                aumento_perc = 100 * aumento / biomassa_inicial

                report.append(f"\n### {col}")
                report.append(f"* **Valor inicial:** {biomassa_inicial:.2f} g/L")
                report.append(f"* **Valor final:** {biomassa_final:.2f} g/L")
                report.append(f"* **Aumento:** {aumento:.2f} g/L ({aumento_perc:.1f}%)")

        # Eficiências de remoção
        if removal_efficiencies:
            report.append("\n## Eficiências de Remoção")

            for poluente, eficiencia in removal_efficiencies.items():
                valor_inicial = df[poluente].iloc[0]
                valor_final = df[poluente].iloc[-1]

                report.append(f"\n### {poluente}")
                report.append(f"* **Valor inicial:** {valor_inicial:.2f} mg/L")
                report.append(f"* **Valor final:** {valor_final:.2f} mg/L")
                report.append(f"* **Eficiência de remoção:** {eficiencia:.2f}%")

        # Parâmetros dos modelos
        if model_params:
            report.append("\n## Parâmetros dos Modelos")

            for param_key, models in model_params.items():
                if models is not None:
                    report.append(f"\n### {param_key.capitalize()}")

                    for model_key, params in models.items():
                        report.append(f"\n#### Modelo {model_key.capitalize()}")

                        for param_name, param_value in params.items():
                            report.append(f"* **{param_name}:** {param_value:.4f}")

        # Métricas de ML
        if ml_metrics:
            report.append("\n## Métricas de Machine Learning")

            for target, metrics in ml_metrics.items():
                report.append(f"\n### Previsão de {target}")

                for metric_name, metric_value in metrics.items():
                    report.append(f"* **{metric_name}:** {metric_value:.4f}")

        # Conclusão
        report.append("\n## Conclusão")
        report.append("""
O tratamento terciário do soro de leite usando microalgas e fungos filamentosos apresentou
resultados satisfatórios, com eficiências de remoção significativas para os poluentes analisados.
Os modelos cinéticos ajustados descrevem adequadamente o comportamento do sistema, permitindo
prever o desempenho do processo em diferentes condições operacionais.

A aplicação de técnicas de machine learning permitiu identificar os principais fatores que
influenciam o processo, possibilitando a otimização das condições de tratamento para maximizar
a eficiência.

Este relatório foi gerado automaticamente pelo software BioCine.
        """)

        # Unindo o relatório
        report_text = "\n".join(report)

        return report_text