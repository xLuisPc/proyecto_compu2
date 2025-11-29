"""
Implementación del servicio de modelo - Capa de infraestructura
"""
import os
import numpy as np
from PIL import Image
from tensorflow import keras
from app.domain.entities import Prediction, CLASSES
from app.domain.repositories import ModelRepository


class TensorFlowModelRepository(ModelRepository):
    """Implementación del repositorio usando TensorFlow"""
    
    def __init__(self, model_path: str):
        """
        Inicializa el repositorio cargando el modelo
        
        Args:
            model_path: Ruta al archivo del modelo (.h5)
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo no encontrado en: {model_path}")
        
        self.model = keras.models.load_model(model_path)
        self.class_names = [cls.name for cls in CLASSES]
        self.class_display_names = {cls.name: cls.display_name for cls in CLASSES}
        self.class_emojis = {cls.name: cls.emoji for cls in CLASSES}
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Preprocesa una imagen para el modelo
        
        Args:
            image: Imagen PIL
            
        Returns:
            Array numpy preprocesado (1, 256, 256, 3)
        """
        # Convertir a RGB si es necesario
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Redimensionar a 256x256
        image = image.resize((256, 256))
        
        # Convertir a array y normalizar
        img_array = np.array(image, dtype=np.float32) / 255.0
        
        # Asegurar que tenga 3 canales
        if len(img_array.shape) == 2:  # Escala de grises
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[2] == 4:  # RGBA
            img_array = img_array[:, :, :3]  # Solo RGB
        
        # Agregar dimensión de batch
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict(self, image_array: np.ndarray) -> Prediction:
        """
        Realiza una predicción sobre una imagen preprocesada
        
        Args:
            image_array: Array numpy de la imagen (1, 256, 256, 3)
            
        Returns:
            Prediction con la clase predicha y confianza
        """
        # Realizar predicción
        predictions = self.model.predict(image_array, verbose=0)
        prediction_probs = predictions[0]
        
        # Obtener clase predicha
        predicted_idx = np.argmax(prediction_probs)
        predicted_class = self.class_names[predicted_idx]
        confidence = float(prediction_probs[predicted_idx])
        
        # Crear lista de todas las predicciones
        all_predictions = [
            {
                "class": self.class_names[i],
                "display_name": self.class_display_names[self.class_names[i]],
                "emoji": self.class_emojis[self.class_names[i]],
                "confidence": float(prediction_probs[i])
            }
            for i in range(len(self.class_names))
        ]
        
        # Ordenar por confianza descendente
        all_predictions.sort(key=lambda x: x["confidence"], reverse=True)
        
        return Prediction(
            class_name=predicted_class,
            confidence=confidence,
            all_predictions=all_predictions
        )

