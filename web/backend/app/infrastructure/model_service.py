"""
Implementación del servicio de modelo - Capa de infraestructura
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from tensorflow import keras
from typing import Tuple, Dict, Any
from app.domain.entities import Prediction, CLASSES
from app.domain.repositories import ModelRepository
import io


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
        
        # Verificar la forma de salida del modelo
        output_shape = self.model.output_shape
        num_output_classes = output_shape[1] if len(output_shape) > 1 else output_shape[0]
        print(f"\n[INFO] Modelo cargado. Shape de salida: {output_shape}")
        print(f"[INFO] Número de clases en el modelo: {num_output_classes}")
        
        # IMPORTANTE: Keras flow_from_directory ordena las clases alfabéticamente
        # El orden debe ser: ['cloudy', 'desert', 'green_area', 'water']
        # Índices: 0=cloudy, 1=desert, 2=green_area, 3=water
        # Asegurémonos de que el orden coincida con el del entrenamiento
        expected_order = ['cloudy', 'desert', 'green_area', 'water']
        self.class_names = expected_order.copy()
        
        # Verificar que el número de clases coincida
        if num_output_classes != len(self.class_names):
            raise ValueError(
                f"El modelo tiene {num_output_classes} clases de salida, "
                f"pero se esperan {len(self.class_names)} clases. "
                f"Verifique que el modelo sea compatible."
            )
        
        # Crear diccionarios de mapeo
        all_classes_dict = {cls.name: cls for cls in CLASSES}
        self.class_display_names = {name: all_classes_dict[name].display_name for name in self.class_names}
        self.class_emojis = {name: all_classes_dict[name].emoji for name in self.class_names}
        
        # Verificar que todas las clases esperadas existen
        for name in self.class_names:
            if name not in all_classes_dict:
                raise ValueError(f"Clase '{name}' no encontrada en CLASSES")
        
        print(f"[INFO] Orden de clases configurado: {self.class_names}")
        print(f"[INFO] Mapeo: {dict(enumerate(self.class_names))}\n")
    
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
        
        # IMPORTANTE: Verificar el orden de las clases del modelo
        # Keras flow_from_directory ordena las clases alfabéticamente
        # El orden esperado es: ['cloudy', 'desert', 'green_area', 'water']
        # Índices: 0=cloudy, 1=desert, 2=green_area, 3=water
        
        # DEBUG: Imprimir todas las probabilidades para diagnóstico
        print("\n" + "="*60)
        print("DEBUG: Probabilidades del modelo")
        print("="*60)
        for i, prob in enumerate(prediction_probs):
            if i < len(self.class_names):
                marker = " <-- PREDICCIÓN" if i == np.argmax(prediction_probs) else ""
                print(f"  [{i}] {self.class_names[i]:15s}: {prob:.4f} ({prob*100:.2f}%){marker}")
        print("="*60 + "\n")
        
        # Obtener clase predicha
        predicted_idx = int(np.argmax(prediction_probs))
        
        # Validar que el índice esté en rango
        if predicted_idx >= len(self.class_names):
            raise ValueError(
                f"Índice de clase predicha ({predicted_idx}) fuera de rango. "
                f"Clases disponibles: {len(self.class_names)}"
            )
        
        # Validar que el número de clases coincida
        if len(prediction_probs) != len(self.class_names):
            raise ValueError(
                f"El modelo devuelve {len(prediction_probs)} clases, pero se esperan {len(self.class_names)}. "
                f"Verifique que el modelo sea compatible."
            )
        
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
    
    def segment_image(
        self, 
        image: Image.Image, 
        tile_size: int = 64, 
        stride: int = 32,
        border_width: int = 2
    ) -> Image.Image:
        """
        Segmenta una imagen dividiéndola en tiles pequeños y clasificando cada uno.
        Luego dibuja bordes de colores según la clasificación.
        
        Args:
            image: Imagen PIL a segmentar
            tile_size: Tamaño de cada tile (default: 64)
            stride: Paso entre tiles (default: 32, 50% overlap)
            border_width: Grosor de los bordes (default: 2)
            
        Returns:
            Imagen PIL con bordes de colores superpuestos
        """
        # Convertir a RGB si es necesario
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        original_size = image.size
        img_array = np.array(image, dtype=np.float32)
        height, width = img_array.shape[:2]
        
        # Colores para cada clase (RGB)
        class_colors = {
            'cloudy': (255, 200, 100),      # Naranja claro
            'desert': (255, 220, 177),      # Beige
            'green_area': (100, 255, 100),   # Verde
            'water': (100, 150, 255),        # Azul
        }
        
        # Crear mapa de probabilidades
        num_classes = len(self.class_names)
        prob_map = np.zeros((height, width, num_classes), dtype=np.float32)
        count_map = np.zeros((height, width), dtype=np.float32)
        
        # Procesar cada tile
        tiles_processed = 0
        total_tiles = ((height - tile_size) // stride + 1) * ((width - tile_size) // stride + 1)
        
        for y in range(0, height - tile_size + 1, stride):
            for x in range(0, width - tile_size + 1, stride):
                # Extraer tile
                tile = img_array[y:y+tile_size, x:x+tile_size]
                tile_pil = Image.fromarray(tile.astype(np.uint8))
                
                # Preprocesar y predecir
                tile_array = self.preprocess_image(tile_pil)
                predictions = self.model.predict(tile_array, verbose=0)[0]
                
                # Acumular probabilidades en el mapa
                prob_map[y:y+tile_size, x:x+tile_size] += predictions
                count_map[y:y+tile_size, x:x+tile_size] += 1
                
                tiles_processed += 1
        
        # Promediar probabilidades
        prob_map /= np.maximum(count_map[..., np.newaxis], 1)
        
        # Crear mapa de segmentación (clase dominante)
        segmentation = np.argmax(prob_map, axis=2)
        
        # Crear imagen de salida (copia de la original)
        output_image = image.copy()
        draw = ImageDraw.Draw(output_image)
        
        # Encontrar y dibujar bordes
        # Para cada píxel, verificar si tiene un vecino de diferente clase
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                current_class = segmentation[y, x]
                
                # Verificar vecinos
                neighbors = [
                    segmentation[y-1, x],   # Arriba
                    segmentation[y+1, x],   # Abajo
                    segmentation[y, x-1],    # Izquierda
                    segmentation[y, x+1],    # Derecha
                ]
                
                # Si hay un vecino diferente, dibujar borde
                if any(neighbor != current_class for neighbor in neighbors):
                    class_name = self.class_names[current_class]
                    color = class_colors.get(class_name, (255, 255, 255))
                    
                    # Dibujar borde (píxel actual)
                    for dy in range(-border_width//2, border_width//2 + 1):
                        for dx in range(-border_width//2, border_width//2 + 1):
                            if 0 <= y + dy < height and 0 <= x + dx < width:
                                output_image.putpixel((x + dx, y + dy), color)
        
        return output_image
    
    def segment_image_optimized(
        self,
        image: Image.Image,
        tile_size: int = 64,
        stride: int = 32,
        border_width: int = 2
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Versión optimizada que también devuelve estadísticas.
        Procesa tiles en batch cuando es posible.
        """
        # Convertir a RGB si es necesario
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_array = np.array(image, dtype=np.float32)
        height, width = img_array.shape[:2]
        
        # Colores para cada clase
        class_colors = {
            'cloudy': (255, 200, 100),
            'desert': (255, 220, 177),
            'green_area': (100, 255, 100),
            'water': (100, 150, 255),
        }
        
        # Crear mapa de probabilidades
        num_classes = len(self.class_names)
        prob_map = np.zeros((height, width, num_classes), dtype=np.float32)
        count_map = np.zeros((height, width), dtype=np.float32)
        
        # Recopilar todos los tiles primero
        tiles = []
        tile_positions = []
        
        for y in range(0, height - tile_size + 1, stride):
            for x in range(0, width - tile_size + 1, stride):
                tile = img_array[y:y+tile_size, x:x+tile_size]
                tile_pil = Image.fromarray(tile.astype(np.uint8))
                tiles.append(tile_pil)
                tile_positions.append((y, x))
        
        # Procesar en batches para mejor rendimiento
        batch_size = 32
        total_tiles = len(tiles)
        
        for i in range(0, total_tiles, batch_size):
            batch_tiles = tiles[i:i+batch_size]
            batch_positions = tile_positions[i:i+batch_size]
            
            # Preprocesar batch
            batch_arrays = np.array([
                self.preprocess_image(tile)[0] for tile in batch_tiles
            ])
            
            # Predecir batch
            batch_predictions = self.model.predict(batch_arrays, verbose=0)
            
            # Acumular en mapa
            for idx, (y, x) in enumerate(batch_positions):
                predictions = batch_predictions[idx]
                prob_map[y:y+tile_size, x:x+tile_size] += predictions
                count_map[y:y+tile_size, x:x+tile_size] += 1
        
        # Promediar
        prob_map /= np.maximum(count_map[..., np.newaxis], 1)
        
        # Crear segmentación
        segmentation = np.argmax(prob_map, axis=2)
        
        # Calcular estadísticas
        class_counts = {}
        for class_idx, class_name in enumerate(self.class_names):
            count = np.sum(segmentation == class_idx)
            percentage = (count / (height * width)) * 100
            class_counts[class_name] = {
                'pixels': int(count),
                'percentage': round(percentage, 2)
            }
        
        # Crear imagen con bordes
        output_image = image.copy()
        draw = ImageDraw.Draw(output_image)
        
        # Encontrar bordes de manera eficiente usando numpy
        # Un píxel es borde si tiene un vecino de diferente clase
        segmentation_padded = np.pad(segmentation, 1, mode='edge')
        
        # Comparar con vecinos (arriba, abajo, izquierda, derecha)
        up_diff = segmentation != segmentation_padded[0:height, 1:width+1]
        down_diff = segmentation != segmentation_padded[2:height+2, 1:width+1]
        left_diff = segmentation != segmentation_padded[1:height+1, 0:width]
        right_diff = segmentation != segmentation_padded[1:height+1, 2:width+2]
        
        borders = up_diff | down_diff | left_diff | right_diff
        
        # Convertir a array de píxeles para dibujar
        output_array = np.array(output_image)
        
        # Dibujar bordes
        border_pixels = np.where(borders)
        for y, x in zip(border_pixels[0], border_pixels[1]):
            class_idx = segmentation[y, x]
            class_name = self.class_names[class_idx]
            color = class_colors.get(class_name, (255, 255, 255))
            
            # Dibujar con grosor
            for dy in range(-border_width//2, border_width//2 + 1):
                for dx in range(-border_width//2, border_width//2 + 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        output_array[ny, nx] = color
        
        output_image = Image.fromarray(output_array.astype(np.uint8))
        
        stats = {
            'total_tiles': total_tiles,
            'image_size': {'width': width, 'height': height},
            'tile_size': tile_size,
            'stride': stride,
            'class_distribution': class_counts
        }
        
        return output_image, stats

