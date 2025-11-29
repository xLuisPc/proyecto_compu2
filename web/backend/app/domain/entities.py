"""
Entidades del dominio - Representan los conceptos del negocio
"""
from dataclasses import dataclass
from typing import List


@dataclass
class Prediction:
    """Resultado de una predicción del modelo"""
    class_name: str
    confidence: float
    all_predictions: List[dict]


@dataclass
class ImageClass:
    """Clase de imagen que puede ser clasificada"""
    name: str
    display_name: str
    emoji: str


# Clases disponibles en el modelo
CLASSES = [
    ImageClass("cloudy", "Nubes", "☁️"),
    ImageClass("desert", "Desierto", "🏜️"),
    ImageClass("green_area", "Área Verde", "🌿"),
    ImageClass("water", "Agua", "💧"),
]

