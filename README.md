# Clasificación de Imágenes Satelitales

Sistema completo de clasificación de imágenes satelitales utilizando redes neuronales convolucionales (CNN) para clasificar imágenes en 4 categorías: nubes, desierto, áreas verdes y agua.

## 📋 Tabla de Contenidos

- [Descripción del Proyecto](#descripción-del-proyecto)
- [Dataset](#dataset)
- [Arquitecturas de Modelos](#arquitecturas-de-modelos)
- [Metodología de Entrenamiento](#metodología-de-entrenamiento)
- [Resultados](#resultados)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Requisitos e Instalación](#requisitos-e-instalación)
- [Uso](#uso)

---

## 📖 Descripción del Proyecto

Este proyecto implementa un sistema de clasificación de imágenes satelitales utilizando dos arquitecturas diferentes de redes neuronales convolucionales (CNN). El objetivo es comparar el desempeño de ambas arquitecturas mediante validación cruzada y seleccionar el mejor modelo para la clasificación final.

**Clases a clasificar:**
- ☁️ **Cloudy** (Nubes)
- 🏜️ **Desert** (Desierto)
- 🌿 **Green Area** (Área Verde)
- 💧 **Water** (Agua)

---

## 📊 Dataset

### Información del Dataset

El dataset contiene **5,631 imágenes satelitales** distribuidas en 4 clases:

| Clase | Cantidad de Imágenes |
|-------|---------------------|
| `cloudy` | 1,500 |
| `desert` | 1,131 |
| `green_area` | 1,500 |
| `water` | 1,500 |
| **Total** | **5,631** |

**Fuente del Dataset:** [Kaggle - Satellite Image Classification](https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification)

### Preprocesamiento

- **Tamaño de imagen**: Todas las imágenes se redimensionan a **256x256 píxeles**
- **Normalización**: Los valores de píxeles se normalizan al rango [0, 1] dividiendo por 255
- **Conversión a RGB**: Las imágenes se convierten al formato RGB si están en otro espacio de color

---

## 🧠 Arquitecturas de Modelos

Se implementaron y compararon dos arquitecturas CNN diferentes:

### CNN 1: Red Simple

Esta arquitectura utiliza capas convolucionales básicas con pooling y dropout para regularización.

**Estructura:**
```
Input (256x256x3)
  ↓
Conv2D(32 filters, 3x3) + ReLU
  ↓
MaxPooling2D(2x2)
  ↓
Conv2D(64 filters, 3x3) + ReLU
  ↓
MaxPooling2D(2x2)
  ↓
Conv2D(128 filters, 3x3) + ReLU
  ↓
MaxPooling2D(2x2)
  ↓
Conv2D(128 filters, 3x3) + ReLU
  ↓
MaxPooling2D(2x2)
  ↓
Flatten
  ↓
Dropout(0.5)
  ↓
Dense(512) + ReLU
  ↓
Dense(4) + Softmax
```

**Características:**
- 4 bloques convolucionales progresivos (32 → 64 → 128 → 128 filtros)
- Pooling después de cada bloque convolucional para reducir dimensiones
- Dropout del 50% antes de la capa densa final para prevenir overfitting
- Capa de salida con 4 neuronas (una por clase) y activación softmax

**Optimizador:** Adam  
**Función de pérdida:** Categorical Crossentropy  
**Métricas:** Accuracy

---

### CNN 2: Red con Batch Normalization

Esta arquitectura es más compleja y utiliza técnicas avanzadas como Batch Normalization y múltiples capas de dropout estratégicamente colocadas.

**Estructura:**
```
Input (256x256x3)
  ↓
Conv2D(32 filters, 3x3) + ReLU
  ↓
BatchNormalization
  ↓
Conv2D(32 filters, 3x3) + ReLU
  ↓
MaxPooling2D(2x2)
  ↓
Dropout(0.25)
  ↓
Conv2D(64 filters, 3x3) + ReLU
  ↓
BatchNormalization
  ↓
Conv2D(64 filters, 3x3) + ReLU
  ↓
MaxPooling2D(2x2)
  ↓
Dropout(0.25)
  ↓
Conv2D(128 filters, 3x3) + ReLU
  ↓
BatchNormalization
  ↓
Conv2D(128 filters, 3x3) + ReLU
  ↓
MaxPooling2D(2x2)
  ↓
Dropout(0.25)
  ↓
Conv2D(256 filters, 3x3) + ReLU
  ↓
BatchNormalization
  ↓
MaxPooling2D(2x2)
  ↓
Dropout(0.25)
  ↓
Flatten
  ↓
Dense(512) + ReLU
  ↓
BatchNormalization
  ↓
Dropout(0.5)
  ↓
Dense(256) + ReLU
  ↓
Dropout(0.5)
  ↓
Dense(4) + Softmax
```

**Características:**
- **Bloques convolucionales dobles**: Cada nivel tiene dos capas convolucionales antes del pooling
- **Batch Normalization**: Después de cada capa convolucional y en la capa densa intermedia para estabilizar el entrenamiento
- **Dropout progresivo**: 25% en capas convolucionales, 50% en capas densas
- **Más filtros**: Progresión 32 → 64 → 128 → 256 filtros
- **Más capas densas**: Dos capas densas (512 y 256 neuronas) en lugar de una

**Optimizador:** Adam  
**Función de pérdida:** Categorical Crossentropy  
**Métricas:** Accuracy

---

## 🔬 Metodología de Entrenamiento

### Proceso de Entrenamiento

El entrenamiento se realiza en **3 etapas principales**:

#### Etapa 1: Partición de Datos

1. **División Train/Test**: 
   - **80%** de los datos → Conjunto de entrenamiento (`train/`)
   - **20%** de los datos → Conjunto de prueba (`test/`)
   - División estratificada por clase
   - Semilla aleatoria fija (random_state=42) para reproducibilidad

2. **Distribución de imágenes**:
   - **Training**: ~4,505 imágenes
   - **Testing**: ~1,126 imágenes

#### Etapa 2: Validación Cruzada (Cross-Validation)

Se realiza **validación cruzada de 3 folds (3-Fold Cross-Validation)** para cada arquitectura:

1. **Partición de datos de entrenamiento**:
   - Los datos de entrenamiento se dividen en 3 folds
   - Cada fold se usa una vez como conjunto de validación
   - Los otros 2 folds se usan para entrenar

2. **Entrenamiento por fold**:
   - **Épocas por fold**: 10 épocas
   - **Batch size**: 32
   - **Métrica evaluada**: Accuracy en el conjunto de validación

3. **Cálculo de métricas**:
   - Se calcula la accuracy promedio de los 3 folds
   - Se calcula la desviación estándar para medir la consistencia
   - Resultado: `Accuracy promedio ± Desviación estándar`

4. **Selección del mejor modelo**:
   - Se compara la accuracy promedio de CNN1 vs CNN2
   - El modelo con mayor accuracy promedio es seleccionado

#### Etapa 3: Entrenamiento Final y Evaluación

1. **Entrenamiento del modelo seleccionado**:
   - Se entrena con **todo el conjunto de entrenamiento** (sin dividir)
   - **Épocas**: 20 épocas
   - **Batch size**: 32
   - Sin conjunto de validación (ya se validó en la etapa anterior)

2. **Guardado del modelo**:
   - El modelo se guarda en formato `.h5` (HDF5)
   - Ruta: `modelos/best_model.h5` o raíz del proyecto

3. **Evaluación en conjunto de prueba**:
   - Se evalúa el modelo en el conjunto de test (nunca visto durante el entrenamiento)
   - Se calculan las siguientes métricas:
     - **Accuracy**: Precisión general
     - **Precision**: Precisión por clase (weighted average)
     - **Recall**: Sensibilidad por clase (weighted average)
     - **F1-Score**: Media armónica de precisión y recall (weighted average)
   - Se genera la **matriz de confusión**
   - Se generan visualizaciones de ejemplos correctos e incorrectos

### Configuración del Entrenamiento

| Parámetro | Valor |
|-----------|-------|
| Tamaño de imagen | 256x256 píxeles |
| Batch size | 32 |
| Épocas (validación cruzada) | 10 |
| Épocas (entrenamiento final) | 20 |
| Folds (validación cruzada) | 3 |
| Optimizador | Adam |
| Función de pérdida | Categorical Crossentropy |
| Métrica principal | Accuracy |

---

## 📈 Resultados

### Visualizaciones Generadas

El proceso genera varios archivos de visualización y análisis:

#### 1. Matriz de Confusión (`confusion_matrix.png`)

Muestra la distribución de predicciones vs etiquetas reales:
- Diagonal principal: Predicciones correctas
- Fuera de la diagonal: Errores de clasificación
- Permite identificar qué clases se confunden más entre sí

#### 2. Predicciones Correctas (`correct_predictions.png`)

Muestra 5 ejemplos de imágenes correctamente clasificadas:
- Se selecciona al menos un ejemplo de cada clase (si es posible)
- Muestra la etiqueta real, la predicha y el nivel de confianza

#### 3. Predicciones Incorrectas (`incorrect_predictions.png`)

Muestra 5 ejemplos de imágenes mal clasificadas:
- Permite analizar los casos más difíciles
- Muestra la etiqueta real, la predicha (incorrecta) y el nivel de confianza

### Interpretación de Resultados

#### Confianza (Confidence Score)

El valor de confianza indica qué tan seguro está el modelo de su predicción:
- **0.0 - 1.0**: Probabilidad de que la predicción sea correcta
- **> 0.9**: Muy confiado
- **0.7 - 0.9**: Moderadamente confiado
- **0.5 - 0.7**: Poca confianza
- **< 0.5**: Muy poco confiado (posible confusión entre clases)

**Ejemplo:** Si `Conf: 0.66`, significa que el modelo está 66% seguro de su predicción.

#### Métricas de Evaluación

- **Accuracy**: Proporción de predicciones correctas sobre el total
- **Precision**: De todas las predicciones de una clase, cuántas fueron correctas
- **Recall**: De todas las instancias reales de una clase, cuántas fueron detectadas
- **F1-Score**: Balance entre precision y recall (media armónica)

---

## 📁 Estructura del Proyecto

```
ModeloCompu2/
├── README.md                    # Este archivo
├── modelo.ipynb                # Notebook para Google Colab (con descarga automática)
│
├── best_model.h5               # Modelo entrenado final (guardado)
├── confusion_matrix.png        # Matriz de confusión generada
├── correct_predictions.png     # Ejemplos de predicciones correctas
├── incorrect_predictions.png   # Ejemplos de predicciones incorrectas
│
├── dataset/                    # Dataset local
│   └── data/
│       ├── cloudy/            # 1,500 imágenes
│       ├── desert/            # 1,131 imágenes
│       ├── green_area/        # 1,500 imágenes
│       └── water/             # 1,500 imágenes
│
├── train/                      # Se crea automáticamente (80% datos)
│   ├── cloudy/
│   ├── desert/
│   ├── green_area/
│   └── water/
│
├── test/                       # Se crea automáticamente (20% datos)
│   ├── cloudy/
│   ├── desert/
│   ├── green_area/
│   └── water/
```

---

## 📦 Requisitos e Instalación

### Dependencias Python

```bash
pip install tensorflow scikit-learn matplotlib seaborn numpy pandas pillow
```

Para el notebook de Colab, también necesitas:
```bash
pip install kagglehub
```

### Requisitos del Sistema

- **Python**: 3.7 o superior
- **TensorFlow**: 2.x
- **Memoria RAM**: Mínimo 8GB recomendado
- **GPU**: Opcional pero altamente recomendada para acelerar el entrenamiento
  - CUDA compatible para TensorFlow GPU
  - O usar Google Colab con GPU T4 (gratis)

---

## 🚀 Uso

### Opción 1: Entrenamiento Local (Script Python)

1. **Preparar el dataset**:
   - Asegúrate de que el dataset esté en `dataset/data/` con las 4 carpetas de clases

2. **Ejecutar el script**:
   ```bash
   python entrenar_modelo.py
   ```
   
   O con permisos de ejecución:
   ```bash
   ./entrenar_modelo.py
   ```

3. **El script ejecutará automáticamente**:
   - Partición de datos (train/test)
   - Validación cruzada de ambas arquitecturas
   - Selección del mejor modelo
   - Entrenamiento final
   - Evaluación y generación de visualizaciones

### Opción 2: Entrenamiento en Google Colab (Notebook)

1. Abre `modelo.ipynb` en Google Colab
2. Activa GPU: **Runtime → Change runtime type → GPU → T4**
3. Ejecuta todas las celdas en orden
4. El notebook descargará automáticamente el dataset desde Kaggle

### Uso del Modelo Entrenado

Para cargar y usar el modelo entrenado:

```python
import tensorflow as tf
from tensorflow import keras

# Cargar el modelo
model = keras.models.load_model('best_model.h5')

# Predecir una imagen
from PIL import Image
import numpy as np

# Cargar y preprocesar imagen
img = Image.open('ruta/a/imagen.jpg')
img = img.resize((256, 256))
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predecir
predictions = model.predict(img_array)
predicted_class_idx = np.argmax(predictions[0])
confidence = np.max(predictions[0])

# Clases
CLASSES = ['cloudy', 'desert', 'green_area', 'water']
print(f"Predicción: {CLASSES[predicted_class_idx]} (Confianza: {confidence:.2%})")
```

---

## 📝 Notas Técnicas

### Preprocesamiento de Imágenes

- Todas las imágenes se convierten a RGB antes del procesamiento
- Las imágenes en escala de grises se convierten a RGB duplicando los canales
- Las imágenes con canal alpha (RGBA) se convierten a RGB descartando el canal alpha

### Optimizaciones de Memoria

- El script usa `ImageDataGenerator` para cargar imágenes en lotes y evitar cargar todo el dataset en memoria
- Se limpia la sesión de TensorFlow después de cada fold en la validación cruzada
- Se eliminan temporalmente las carpetas de cada fold después de su evaluación

### Reproducibilidad

- Se usa `random_state=42` en todas las divisiones de datos
- Las semillas aleatorias están fijas para garantizar resultados reproducibles

---

## 🔍 Análisis de Errores Comunes

### Casos de Confusión Frecuentes

Basado en los ejemplos de predicciones incorrectas, las clases que pueden confundirse son:

- **Water ↔ Green Area**: Áreas verdes cerca del agua pueden confundir al modelo
- **Desert ↔ Cloudy**: Colores similares en ciertas condiciones de iluminación

### Factores que Afectan la Precisión

1. **Calidad de la imagen**: Resolución, contraste, iluminación
2. **Transiciones**: Imágenes en los bordes entre dos clases
3. **Condiciones atmosféricas**: Nubes que cubren parcialmente otras áreas
4. **Ángulo de visión**: Perspectiva satelital diferente

---

## 📚 Referencias

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Documentation](https://keras.io/)
- [Scikit-learn Cross-Validation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Dataset: Satellite Image Classification](https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification)

---

## 📄 Licencia

Este proyecto es de carácter educativo/académico. El dataset utilizado pertenece a su respectivo propietario en Kaggle.

---

**Autor:** Proyecto de Computación 2  
**Fecha:** 2024  
**Propósito:** Clasificación de imágenes satelitales utilizando CNN

