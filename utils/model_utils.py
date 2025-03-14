"""
Utilitários para manipulação de modelos de machine learning

Este módulo contém funções auxiliares para trabalhar com modelos
de machine learning, especialmente para extração de metadados e
manipulação de características.
"""

import os
import pickle
import json
import datetime
import pandas as pd
import numpy as np
import joblib

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


def save_model(model, model_name, feature_names=None, target_name=None, description=None):
    """
    Salva um modelo treinado em disco usando uma pasta dedicada para cada modelo

    Args:
        model: O modelo treinado
        model_name: Nome para o modelo
        feature_names: Lista com nomes das features usadas no treinamento
        target_name: Nome da variável alvo
        description: Descrição do modelo (opcional)

    Returns:
        dict: Informações sobre o modelo salvo
    """
    # Garante que a pasta models existe
    models_dir = os.path.join('data', 'models')
    os.makedirs(models_dir, exist_ok=True)

    # Sanitiza o nome do arquivo/pasta
    safe_name = model_name.replace(' ', '_').replace('/', '-').lower()

    # Adiciona timestamp para garantir unicidade
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dirname = f"{safe_name}_{timestamp}"

    # Cria pasta específica para este modelo
    model_dir = os.path.join(models_dir, model_dirname)
    os.makedirs(model_dir, exist_ok=True)

    # Informações sobre o modelo
    model_info = {
        'name': model_name,
        'model_dir': model_dirname,
        'timestamp': timestamp,
        'datetime': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'features': feature_names if feature_names is not None else [],
        'target': target_name,
        'description': description if description is not None else "",
        'model_type': type(model).__name__,
        'serialization': 'unknown'  # Será atualizado com o método usado
    }

    # Tenta extrair métricas de desempenho se disponíveis
    if hasattr(model, 'get_metrics'):
        model_info['metrics'] = model.get_metrics()

    # Tenta diferentes métodos de serialização
    serialization_success = False

    # Estratégia 1: Verifique se o modelo tem método __getstate__ personalizado
    if hasattr(model, '__getstate__'):
        model_info['serialization'] = 'pickle-getstate'
        try:
            model_path = os.path.join(model_dir, "model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            model_info['model_file'] = "model.pkl"
            serialization_success = True
        except Exception as e:
            print(f"Erro ao usar pickle com __getstate__: {str(e)}")

    # Estratégia 2: Tente com joblib
    if not serialization_success:
        try:
            model_path = os.path.join(model_dir, "model.joblib")
            joblib.dump(model, model_path)
            model_info['model_file'] = "model.joblib"
            model_info['serialization'] = 'joblib'
            serialization_success = True
        except Exception as e:
            print(f"Erro ao usar joblib: {str(e)}")

    # Estratégia 3: Tente componentes individuais para modelos customizados
    if not serialization_success and hasattr(model, 'model'):
        try:
            # Cria diretório de componentes dentro da pasta do modelo
            components_dir = os.path.join(model_dir, "components")
            os.makedirs(components_dir, exist_ok=True)

            # Salva o modelo interno
            joblib.dump(model.model, os.path.join(components_dir, "internal_model.joblib"))

            # Salva outros atributos como um dicionário
            model_attrs = {}
            for attr in ['features', 'target', 'X_train', 'X_test', 'y_train', 'y_test', 'metrics', 'name']:
                if hasattr(model, attr):
                    model_attrs[attr] = getattr(model, attr)

            with open(os.path.join(components_dir, "attributes.pkl"), 'wb') as f:
                pickle.dump(model_attrs, f)

            # Salva o tipo do modelo customizado
            with open(os.path.join(components_dir, "model_class.txt"), 'w') as f:
                f.write(type(model).__name__)

            model_info['model_file'] = "components/"
            model_info['serialization'] = 'components'
            serialization_success = True

        except Exception as e:
            print(f"Erro ao salvar componentes: {str(e)}")

    # Se todas as estratégias falharem
    if not serialization_success:
        raise ValueError(
            f"Não foi possível serializar o modelo. Considere implementar método __getstate__ na classe {type(model).__name__}.")

    # Salva metadados do modelo em um arquivo JSON
    metadata_path = os.path.join(model_dir, "metadata.json")

    # Converte valores numpy para JSON serializável
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    # Salva o JSON com os metadados dentro da pasta do modelo
    with open(metadata_path, 'w') as f:
        json.dump(model_info, f, default=convert_numpy, indent=4)

    # Salva um arquivo JSON index.json na pasta raiz de modelos para facilitar listagem
    index_path = os.path.join(models_dir, "index.json")

    # Carrega o índice existente ou cria um novo
    if os.path.exists(index_path):
        try:
            with open(index_path, 'r') as f:
                index = json.load(f)
        except:
            index = {"models": []}
    else:
        index = {"models": []}

    # Adiciona este modelo ao índice
    index["models"].append({
        "name": model_name,
        "directory": model_dirname,
        "timestamp": timestamp,
        "target": target_name,
        "model_type": type(model).__name__
    })

    # Atualiza o arquivo de índice
    with open(index_path, 'w') as f:
        json.dump(index, f, default=convert_numpy, indent=4)

    return model_info


def load_model(model_path):
    """
    Carrega um modelo a partir de uma pasta de modelo

    Args:
        model_path: Caminho para a pasta do modelo ou arquivo específico

    Returns:
        tuple: (modelo, informações do modelo)
    """
    # Verifica se é uma pasta ou arquivo
    if os.path.isdir(model_path):
        # É uma pasta de modelo, procura por metadata.json
        metadata_path = os.path.join(model_path, "metadata.json")

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Arquivo de metadados não encontrado em: {metadata_path}")

        # Carrega os metadados
        with open(metadata_path, 'r') as f:
            model_info = json.load(f)

        # Determina o método de serialização e arquivo correto
        serialization = model_info.get('serialization', 'unknown')
        model_file = model_info.get('model_file', None)

        if not model_file:
            # Tenta inferir o arquivo do modelo
            possible_files = ["model.joblib", "model.pkl", "components"]
            for file in possible_files:
                if os.path.exists(os.path.join(model_path, file)):
                    model_file = file
                    break

            if not model_file:
                raise FileNotFoundError(f"Arquivo do modelo não encontrado em: {model_path}")

        # Carrega o modelo de acordo com o método de serialização
        full_model_path = os.path.join(model_path, model_file)

        if serialization == 'components' or full_model_path.endswith('components'):
            # Carrega a partir de componentes
            components_dir = full_model_path

            # Carrega o modelo interno
            internal_model_path = os.path.join(components_dir, "internal_model.joblib")
            internal_model = joblib.load(internal_model_path)

            # Carrega atributos
            attributes_path = os.path.join(components_dir, "attributes.pkl")
            with open(attributes_path, 'rb') as f:
                attributes = pickle.load(f)

            # Carrega a classe do modelo
            class_path = os.path.join(components_dir, "model_class.txt")
            with open(class_path, 'r') as f:
                model_class_name = f.read().strip()

            # Recria o modelo com base na classe
            if model_class_name == 'RandomForestModel':
                from models.machine_learning import RandomForestModel
                model = RandomForestModel()
                model.model = internal_model

                # Restaura atributos
                for attr, value in attributes.items():
                    setattr(model, attr, value)

                return model, model_info
            else:
                raise ValueError(f"Tipo de modelo não suportado para carregamento de componentes: {model_class_name}")

        elif full_model_path.endswith('.joblib'):
            # Carrega com joblib
            model = joblib.load(full_model_path)
        elif full_model_path.endswith('.pkl'):
            # Carrega com pickle
            with open(full_model_path, 'rb') as f:
                model = pickle.load(f)
        else:
            raise ValueError(f"Formato de arquivo não suportado: {full_model_path}")

        return model, model_info

    else:
        # É um arquivo específico, verifica se existe
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Arquivo do modelo não encontrado: {model_path}")

        # Tenta carregar o modelo diretamente
        if model_path.endswith('.joblib'):
            # Carrega com joblib
            model = joblib.load(model_path)
        elif model_path.endswith('.pkl'):
            # Carrega com pickle
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        else:
            raise ValueError(f"Formato de arquivo não suportado: {model_path}")

        # Tenta encontrar metadados na pasta pai
        parent_dir = os.path.dirname(model_path)
        metadata_path = os.path.join(parent_dir, "metadata.json")

        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                model_info = json.load(f)
        else:
            # Cria informações básicas a partir do nome do arquivo
            model_info = {
                'name': os.path.basename(parent_dir),
                'model_file': os.path.basename(model_path),
                'model_type': type(model).__name__,
                'features': [],
                'target': None
            }

        return model, model_info


def list_saved_models():
    """
    Lista todos os modelos salvos usando o índice ou pesquisa de diretórios

    Returns:
        DataFrame: Informações sobre os modelos salvos
    """
    models_dir = os.path.join('data', 'models')

    # Verifica se a pasta existe
    if not os.path.exists(models_dir):
        return pd.DataFrame()

    # Tenta usar o arquivo de índice se existir
    index_path = os.path.join(models_dir, "index.json")

    if os.path.exists(index_path):
        try:
            with open(index_path, 'r') as f:
                index = json.load(f)

            if "models" in index and index["models"]:
                models_info = []

                for model_entry in index["models"]:
                    model_dir = os.path.join(models_dir, model_entry["directory"])

                    # Verifica se a pasta ainda existe
                    if os.path.isdir(model_dir):
                        # Carrega metadados completos
                        metadata_path = os.path.join(model_dir, "metadata.json")

                        if os.path.exists(metadata_path):
                            with open(metadata_path, 'r') as f:
                                model_info = json.load(f)

                            # Adiciona caminho completo
                            model_info['full_path'] = model_dir
                            models_info.append(model_info)
                        else:
                            # Usa informações básicas do índice
                            model_entry['full_path'] = model_dir
                            models_info.append(model_entry)

                if models_info:
                    df = pd.DataFrame(models_info)

                    # Organiza colunas
                    columns = ['name', 'target', 'model_type', 'datetime', 'description', 'serialization', 'full_path']

                    # Adiciona outras colunas que podem existir
                    for col in df.columns:
                        if col not in columns:
                            columns.append(col)

                    # Filtra colunas que existem
                    columns = [col for col in columns if col in df.columns]

                    # Retorna DataFrame organizado
                    return df[columns].sort_values('timestamp', ascending=False)
        except Exception as e:
            print(f"Erro ao carregar índice: {str(e)}")

    # Se não conseguiu usar o índice, faz busca por diretórios
    model_dirs = [d for d in os.listdir(models_dir)
                  if os.path.isdir(os.path.join(models_dir, d))
                  and d != "__pycache__"]

    if not model_dirs:
        return pd.DataFrame()

    # Carrega as informações de cada modelo
    models_info = []

    for model_dir in model_dirs:
        dir_path = os.path.join(models_dir, model_dir)
        metadata_path = os.path.join(dir_path, "metadata.json")

        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    model_info = json.load(f)

                model_info['full_path'] = dir_path
                models_info.append(model_info)
            except Exception as e:
                print(f"Erro ao carregar metadados de {model_dir}: {str(e)}")

    # Cria DataFrame com as informações
    if models_info:
        df = pd.DataFrame(models_info)

        # Organiza colunas
        columns = ['name', 'target', 'model_type', 'datetime', 'description', 'serialization', 'full_path']

        # Adiciona outras colunas que podem existir
        for col in df.columns:
            if col not in columns:
                columns.append(col)

        # Filtra colunas que existem
        columns = [col for col in columns if col in df.columns]

        # Retorna DataFrame organizado
        try:
            return df[columns].sort_values('timestamp', ascending=False)
        except:
            # Se não puder ordenar por timestamp, retorna sem ordenação
            return df[[c for c in columns if c in df.columns]]

    return pd.DataFrame()