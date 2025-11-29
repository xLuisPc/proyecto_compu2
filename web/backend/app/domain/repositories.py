"""
Repositorios - Interfaces para acceso a datos
"""
from abc import ABC, abstractmethod
from app.domain.entities import Prediction
import numpy as np


class ModelRepository(ABC):
    """Interfaz para el repositorio del modelo"""
    
    @abstractmethod
    def predict(self, image_array: np.ndarray) -> Prediction:
        """
        Realiza una predicción sobre una imagen preprocesada
        
        Args:
            image_array: Array numpy de la imagen (256x256x3, valores normalizados)
            
        Returns:
            Prediction con la clase predicha y confianza
        """
        pass

