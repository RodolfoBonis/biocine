"""
Utilitários para manipulação de modelos de machine learning

Este módulo contém funções auxiliares para trabalhar com modelos
de machine learning, especialmente para extração de metadados e
manipulação de características.
"""

import pandas as pd
import numpy as np


def extract_model_features(model, parameter_names=None):
    """
    Extrai as características que um modelo espera de forma robusta

    Args:
        model: O modelo de machine learning
        parameter_names: Lista de nomes de parâmetros disponíveis (fallback)

    Returns:
        List[str]: Lista de nomes de características que o modelo espera
        str: Fonte da informação (feature_names_in_, feature_importances_, etc.)
    """
    import streamlit as st

    # Inicializa o resultado
    expected_features = None
    source = "unknown"

    # Método 1: Verifica feature_names_in_
    if hasattr(model, 'feature_names_in_'):
        expected_features = list(model.feature_names_in_)
        source = "feature_names_in_"

    # Método 2: Verifica feature_names
    elif hasattr(model, 'feature_names'):
        expected_features = list(model.feature_names)
        source = "feature_names"

    # Método 3: Verifica features (nossos modelos customizados)
    elif hasattr(model, 'features'):
        expected_features = model.features
        source = "custom_features"

    # Método 4: Verifica feature_importances_
    elif hasattr(model, 'feature_importances_'):
        # Se temos importâncias mas não nomes, usamos parâmetros como fallback
        n_features = len(model.feature_importances_)
        if parameter_names and len(parameter_names) >= n_features:
            expected_features = parameter_names[:n_features]
            source = "feature_importances_"

    # Método 5: Verifica o próprio modelo para pistas
    elif hasattr(model, 'n_features_in_'):
        n_features = model.n_features_in_
        if parameter_names and len(parameter_names) >= n_features:
            expected_features = parameter_names[:n_features]
            source = "n_features_in_"

    # Método 6: Verificar na estrutura interna do modelo
    elif hasattr(model, 'estimators_'):
        # Para modelos como RandomForest que têm estimadores internos
        if model.estimators_ and hasattr(model.estimators_[0], 'feature_names_in_'):
            expected_features = list(model.estimators_[0].feature_names_in_)
            source = "estimators_feature_names"
        elif model.estimators_ and hasattr(model.estimators_[0], 'n_features_in_'):
            n_features = model.estimators_[0].n_features_in_
            if parameter_names and len(parameter_names) >= n_features:
                expected_features = parameter_names[:n_features]
                source = "estimators_n_features"

    # Se tudo falhar, tenta método de tentativa e erro
    if expected_features is None and parameter_names:
        # Começa tentando todas as características e vai reduzindo
        for n in range(len(parameter_names), 0, -1):
            test_features = parameter_names[:n]
            try:
                # Tenta fazer uma previsão simples para ver se funciona
                test_data = pd.DataFrame({f: [0.0] for f in test_features})
                model.predict(test_data)
                expected_features = test_features
                source = "trial_and_error"
                break
            except Exception:
                # Ignora erros e continua tentando
                pass

    return expected_features, source


def create_prediction_dataframe(X_dict, feature_names):
    """
    Cria um DataFrame compatível com as características esperadas por um modelo

    Args:
        X_dict: Dicionário com valores de características
        feature_names: Lista de características esperadas pelo modelo

    Returns:
        pd.DataFrame: DataFrame pronto para fazer predições
    """
    # Cria DataFrame com apenas as características esperadas, na ordem correta
    X_df = pd.DataFrame({feature: [X_dict.get(feature, 0.0)] for feature in feature_names})

    # Verificação de sanidade
    if len(X_df.columns) != len(feature_names):
        raise ValueError(f"DataFrame criado tem {len(X_df.columns)} colunas, mas o modelo espera {len(feature_names)}")

    return X_df


def test_model_compatibility(model, param_names, selected_params=None):
    """
    Testa se um modelo é compatível com um conjunto de parâmetros

    Args:
        model: O modelo de machine learning
        param_names: Lista completa de nomes de parâmetros disponíveis
        selected_params: Lista de parâmetros selecionados a verificar (opcional)

    Returns:
        bool: True se o modelo é compatível, False caso contrário
        str: Mensagem com detalhes do resultado
    """
    # Extrai características esperadas pelo modelo
    expected_features, source = extract_model_features(model, param_names)

    if not expected_features:
        return False, "Não foi possível determinar as características esperadas pelo modelo"

    if selected_params:
        # Verifica se os parâmetros selecionados estão entre as características esperadas
        missing = [p for p in selected_params if p not in expected_features]
        if missing:
            return (
                False,
                f"Os parâmetros {', '.join(missing)} não estão entre as características esperadas pelo modelo: {', '.join(expected_features)}"
            )

    # Tenta fazer uma previsão simples para confirmar compatibilidade
    try:
        test_data = pd.DataFrame({feature: [0.0] for feature in expected_features})
        model.predict(test_data)
        return True, f"Modelo compatível. Espera {len(expected_features)} características ({source})"
    except Exception as e:
        return False, f"Erro ao testar modelo: {str(e)}"