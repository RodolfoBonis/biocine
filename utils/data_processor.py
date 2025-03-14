"""
Processamento de dados para modelagem cinética e machine learning

Este módulo implementa funções para importação, processamento e
validação de dados experimentais para modelagem cinética e
aprendizado de máquina.
"""

import os
import numpy as np
import pandas as pd


def import_csv(file_path, **kwargs):
    """
    Importa dados de um arquivo CSV

    Args:
        file_path: Caminho para o arquivo CSV
        **kwargs: Parâmetros adicionais para pd.read_csv

    Returns:
        DataFrame com os dados importados
    """
    try:
        data = pd.read_csv(file_path, **kwargs)
        return data
    except Exception as e:
        raise ValueError(f"Erro ao importar o arquivo CSV: {str(e)}")


def generate_example_data(n_samples=20, seed=42):
    """
    Gera dados de exemplo para modelagem cinética

    Args:
        n_samples: Número de amostras
        seed: Semente aleatória

    Returns:
        DataFrame com os dados de exemplo
    """
    np.random.seed(seed)

    # Tempo em dias
    time = np.linspace(0, 10, n_samples)

    # Parâmetros do modelo Logístico
    umax = 0.5  # Taxa específica máxima de crescimento (dia^-1)
    x0 = 0.1  # Concentração inicial de biomassa (g/L)
    xmax = 5.0  # Concentração máxima de biomassa (g/L)

    # Solução analítica do modelo Logístico
    biomass = xmax / (1 + (xmax / x0 - 1) * np.exp(-umax * time))

    # Adiciona ruído
    biomass_noise = biomass + np.random.normal(0, 0.2, n_samples)
    biomass_noise = np.maximum(biomass_noise, 0.01)  # Garante valores positivos

    # Parâmetros do modelo de Monod
    ks = 10.0  # Constante de meia saturação (mg/L)
    y = 0.5  # Coeficiente de rendimento (g biomassa/g substrato)
    s0 = 100.0  # Concentração inicial de substrato (mg/L)

    # Substrato consumido pela biomassa
    substrate = s0 - (biomass - x0) / y
    substrate = np.maximum(substrate, 0.0)  # Garante valores não negativos

    # Adiciona ruído
    substrate_noise = substrate + np.random.normal(0, 2.0, n_samples)
    substrate_noise = np.maximum(substrate_noise, 0.0)  # Garante valores não negativos

    # Parâmetros ambientais
    temperature = 25.0 + np.random.normal(0, 2.0, n_samples)  # Temperatura (°C)
    ph = 7.0 + np.random.normal(0, 0.5, n_samples)  # pH

    # Concentrações iniciais de nutrientes
    n_initial = 50.0  # mg/L
    p_initial = 10.0  # mg/L
    dqo_initial = 1000.0  # mg/L

    # Remoção de nutrientes com base no crescimento da biomassa
    n_removal_efficiency = 0.8 * (1 - np.exp(-0.3 * biomass))
    p_removal_efficiency = 0.7 * (1 - np.exp(-0.25 * biomass))
    dqo_removal_efficiency = 0.9 * (1 - np.exp(-0.2 * biomass))

    n_concentration = n_initial * (1 - n_removal_efficiency)
    p_concentration = p_initial * (1 - p_removal_efficiency)
    dqo_concentration = dqo_initial * (1 - dqo_removal_efficiency)

    # Cria o DataFrame
    data = pd.DataFrame({
        'tempo': time,
        'biomassa': biomass_noise,
        'substrato': substrate_noise,
        'temperatura': temperature,
        'ph': ph,
        'concentracao_n': n_concentration,
        'concentracao_p': p_concentration,
        'concentracao_dqo': dqo_concentration,
        'eficiencia_remocao_n': n_removal_efficiency * 100,  # Percentual
        'eficiencia_remocao_p': p_removal_efficiency * 100,  # Percentual
        'eficiencia_remocao_dqo': dqo_removal_efficiency * 100  # Percentual
    })

    return data


def calculate_removal_efficiency(initial, final):
    """
    Calcula a eficiência de remoção de poluentes

    Args:
        initial: Concentração inicial
        final: Concentração final

    Returns:
        Eficiência de remoção em percentual
    """
    if initial <= 0:
        return 0.0

    efficiency = (initial - final) / initial * 100
    return max(0.0, min(100.0, efficiency))  # Limita entre 0% e 100%


def validate_data(data, required_columns=None):
    """
    Valida os dados de entrada

    Args:
        data: DataFrame com os dados
        required_columns: Lista com as colunas obrigatórias

    Returns:
        True se os dados são válidos, False caso contrário
    """
    if data is None or data.empty:
        return False

    if required_columns is not None:
        if not all(col in data.columns for col in required_columns):
            return False

    return True


def preprocess_data(data, fill_missing=True, normalize=False):
    """
    Pré-processa os dados

    Args:
        data: DataFrame com os dados
        fill_missing: Se True, preenche valores ausentes
        normalize: Se True, normaliza os dados

    Returns:
        DataFrame com os dados pré-processados
    """
    # Cria uma cópia para evitar modificar o original
    df = data.copy()

    # Preenche valores ausentes
    if fill_missing:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
        df[non_numeric_cols] = df[non_numeric_cols].fillna(df[non_numeric_cols].mode().iloc[0])

    # Normaliza os dados
    if normalize:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    return df


def save_processed_data(data, file_path):
    """
    Salva dados processados em um arquivo CSV

    Args:
        data: DataFrame com os dados
        file_path: Caminho para o arquivo CSV

    Returns:
        True se o salvamento foi bem-sucedido, False caso contrário
    """
    try:
        # Cria o diretório se não existir
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Salva o arquivo
        data.to_csv(file_path, index=False)
        return True
    except Exception:
        return False


def load_processed_data(file_path):
    """
    Carrega dados processados de um arquivo CSV

    Args:
        file_path: Caminho para o arquivo CSV

    Returns:
        DataFrame com os dados carregados
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

    return pd.read_csv(file_path)