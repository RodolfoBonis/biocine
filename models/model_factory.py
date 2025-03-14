"""
Factory para criação de modelos cinéticos e de machine learning

Este módulo implementa o padrão Factory para criar instâncias
de modelos cinéticos e de machine learning de forma padronizada.
"""

from models import MonodModel, LogisticModel, RandomForestModel


class ModelFactory:
    """
    Factory para criação de modelos
    """

    @staticmethod
    def create_cinetic_model(model_type, **kwargs):
        """
        Cria um modelo cinético

        Args:
            model_type: Tipo de modelo ('monod', 'logistic')
            **kwargs: Parâmetros do modelo

        Returns:
            Instância do modelo cinético
        """
        if model_type.lower() == 'monod':
            model = MonodModel()
        elif model_type.lower() == 'logistic':
            model = LogisticModel()
        else:
            raise ValueError(f"Tipo de modelo cinético desconhecido: {model_type}")

        # Define os parâmetros do modelo
        if kwargs:
            model.set_parameters(**kwargs)

        return model

    @staticmethod
    def create_ml_model(model_type, **kwargs):
        """
        Cria um modelo de machine learning

        Args:
            model_type: Tipo de modelo ('random_forest')
            **kwargs: Parâmetros do modelo

        Returns:
            Instância do modelo de machine learning
        """
        if model_type.lower() == 'random_forest':
            model = RandomForestModel(**kwargs)
        else:
            raise ValueError(f"Tipo de modelo de ML desconhecido: {model_type}")

        return model