#!/usr/bin/env python3
"""
Clasificación de Imágenes Satelitales - Script de Entrenamiento Local

Este script implementa un sistema completo de clasificación de imágenes satelitales:
- Lectura de datos locales (dataset ya descargado)
- Dos arquitecturas CNN
- Validación cruzada
- Evaluación completa

Versión totalmente local - Lee el dataset de la carpeta del proyecto y guarda todo localmente.
"""

import os
import sys
import shutil
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report)
import matplotlib
matplotlib.use('Agg')  # Para que funcione sin display
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Configuración del modelo
IMG_SIZE = 256
BATCH_SIZE = 32
EPOCHS_CV = 10
EPOCHS_FINAL = 20
NUM_FOLDS = 3

# Clases del dataset
CLASSES = ['cloudy', 'desert', 'green_area', 'water']
NUM_CLASSES = len(CLASSES)

# Configurar rutas del proyecto
PROJECT_DIR = os.getcwd()
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")
TRAIN_DIR = os.path.join(PROJECT_DIR, "train")
TEST_DIR = os.path.join(PROJECT_DIR, "test")

# Crear carpetas para resultados organizados
RESULTS_DIR = os.path.join(PROJECT_DIR, "resultados")
MODELS_DIR = os.path.join(PROJECT_DIR, "modelos")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def print_section(title):
    """Imprime un título de sección formateado"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def setup_gpu():
    """Configura GPU si está disponible"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✓ GPU detectada: {len(gpus)} dispositivo(s)")
        for gpu in gpus:
            print(f"  - {gpu}")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✓ Configuración de GPU completada")
            if tf.test.is_gpu_available():
                print("✓ TensorFlow puede usar GPU - El entrenamiento será más rápido")
        except Exception as e:
            print(f"⚠ Error configurando GPU: {e}")
    else:
        print("⚠ No se detectó GPU - Se usará CPU (el entrenamiento será más lento)")
        print("  Nota: El entrenamiento puede tardar varias horas en CPU")

def find_dataset():
    """Busca y valida el dataset local"""
    if not os.path.exists(DATASET_DIR):
        raise FileNotFoundError(
            f"❌ ERROR: No se encontró la carpeta 'dataset' en {PROJECT_DIR}\n"
            "Por favor, asegúrate de que el dataset esté en la carpeta 'dataset' del proyecto."
        )
    
    classes_found = []
    dataset_path = None
    
    # Opción 1: Buscar en dataset/data/
    data_path = os.path.join(DATASET_DIR, "data")
    if os.path.exists(data_path):
        for class_name in CLASSES:
            class_path = os.path.join(data_path, class_name)
            if os.path.exists(class_path):
                num_images = len(glob.glob(os.path.join(class_path, "*.jpg")) + 
                                glob.glob(os.path.join(class_path, "*.png")) +
                                glob.glob(os.path.join(class_path, "*.JPG")) +
                                glob.glob(os.path.join(class_path, "*.PNG")))
                if num_images > 0:
                    classes_found.append(class_name)
                    print(f"  ✓ {class_name}: {num_images} imágenes encontradas")
        
        if len(classes_found) == 4:
            dataset_path = data_path
    
    # Opción 2: Buscar directamente en dataset/
    if dataset_path is None:
        classes_found = []
        for class_name in CLASSES:
            class_path = os.path.join(DATASET_DIR, class_name)
            if os.path.exists(class_path):
                num_images = len(glob.glob(os.path.join(class_path, "*.jpg")) + 
                                glob.glob(os.path.join(class_path, "*.png")) +
                                glob.glob(os.path.join(class_path, "*.JPG")) +
                                glob.glob(os.path.join(class_path, "*.PNG")))
                if num_images > 0:
                    classes_found.append(class_name)
                    print(f"  ✓ {class_name}: {num_images} imágenes encontradas")
        
        if len(classes_found) == 4:
            dataset_path = DATASET_DIR
    
    if dataset_path is None or len(classes_found) < 4:
        missing = [c for c in CLASSES if c not in classes_found]
        raise FileNotFoundError(
            f"❌ ERROR: No se encontraron todas las clases del dataset.\n"
            f"Clases encontradas: {classes_found}\n"
            f"Clases faltantes: {missing}"
        )
    
    return dataset_path

def create_train_test_split(source_path, train_path, test_path, test_size=0.2, random_state=42):
    """Crea partición 80/20 de train/test"""
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    
    for class_name in CLASSES:
        train_class_path = os.path.join(train_path, class_name)
        test_class_path = os.path.join(test_path, class_name)
        os.makedirs(train_class_path, exist_ok=True)
        os.makedirs(test_class_path, exist_ok=True)
        
        images = []
        class_source_path = os.path.join(source_path, class_name)
        if os.path.exists(class_source_path):
            images = glob.glob(os.path.join(class_source_path, "*.jpg")) + \
                     glob.glob(os.path.join(class_source_path, "*.png")) + \
                     glob.glob(os.path.join(class_source_path, "*.JPG")) + \
                     glob.glob(os.path.join(class_source_path, "*.PNG"))
        
        if len(images) == 0:
            data_class_path = os.path.join(source_path, "data", class_name)
            if os.path.exists(data_class_path):
                images = glob.glob(os.path.join(data_class_path, "*.jpg")) + \
                         glob.glob(os.path.join(data_class_path, "*.png")) + \
                         glob.glob(os.path.join(data_class_path, "*.JPG")) + \
                         glob.glob(os.path.join(data_class_path, "*.PNG"))
        
        if len(images) > 0:
            train_images, test_images = train_test_split(
                images, test_size=test_size, random_state=random_state
            )
            
            for img_path in train_images:
                shutil.copy2(img_path, train_class_path)
            for img_path in test_images:
                shutil.copy2(img_path, test_class_path)
            
            print(f"  ✓ {class_name}: {len(train_images)} train, {len(test_images)} test")
        else:
            print(f"  ⚠ Advertencia: No se encontraron imágenes para {class_name}")

def create_data_generator(data_dir, batch_size=32, shuffle=True, validation_split=0.0):
    """Crea generador de datos"""
    datagen = ImageDataGenerator(rescale=1./255, validation_split=validation_split)
    generator = datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=shuffle,
        subset='training' if validation_split > 0 else None
    )
    return generator

def create_cnn1():
    """Arquitectura CNN 1: Red Simple"""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_cnn2():
    """Arquitectura CNN 2: Red con Batch Normalization"""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def cross_validate_cnn(model_creator, model_name, train_dir, n_splits=3, epochs=10):
    """Realiza validación cruzada k-fold"""
    image_paths = []
    labels = []
    
    for class_idx, class_name in enumerate(CLASSES):
        class_dir = os.path.join(train_dir, class_name)
        if os.path.exists(class_dir):
            images = glob.glob(os.path.join(class_dir, "*.jpg")) + \
                     glob.glob(os.path.join(class_dir, "*.png")) + \
                     glob.glob(os.path.join(class_dir, "*.JPG")) + \
                     glob.glob(os.path.join(class_dir, "*.PNG"))
            image_paths.extend(images)
            labels.extend([class_idx] * len(images))
    
    image_paths = np.array(image_paths)
    labels = np.array(labels)
    
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies = []
    
    print(f"\n=== Validación Cruzada {n_splits}-Fold para {model_name} ===")
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(image_paths)):
        print(f"\nFold {fold + 1}/{n_splits}")
        
        temp_train_dir = f"temp_train_fold_{fold}"
        temp_val_dir = f"temp_val_fold_{fold}"
        os.makedirs(temp_train_dir, exist_ok=True)
        os.makedirs(temp_val_dir, exist_ok=True)
        
        for class_name in CLASSES:
            os.makedirs(os.path.join(temp_train_dir, class_name), exist_ok=True)
            os.makedirs(os.path.join(temp_val_dir, class_name), exist_ok=True)
        
        train_paths = image_paths[train_idx]
        train_labels = labels[train_idx]
        val_paths = image_paths[val_idx]
        val_labels = labels[val_idx]
        
        for img_path, label in zip(train_paths, train_labels):
            shutil.copy2(img_path, os.path.join(temp_train_dir, CLASSES[label]))
        
        for img_path, label in zip(val_paths, val_labels):
            shutil.copy2(img_path, os.path.join(temp_val_dir, CLASSES[label]))
        
        train_gen = create_data_generator(temp_train_dir, BATCH_SIZE, shuffle=True)
        val_gen = create_data_generator(temp_val_dir, BATCH_SIZE, shuffle=False)
        
        model = model_creator()
        steps_per_epoch = len(train_paths) // BATCH_SIZE
        validation_steps = len(val_paths) // BATCH_SIZE
        
        model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=val_gen,
            validation_steps=validation_steps,
            verbose=1
        )
        
        val_loss, val_acc = model.evaluate(val_gen, steps=validation_steps, verbose=0)
        accuracies.append(val_acc)
        print(f"Fold {fold + 1} - Accuracy: {val_acc:.4f}")
        
        shutil.rmtree(temp_train_dir)
        shutil.rmtree(temp_val_dir)
        del model
        tf.keras.backend.clear_session()
    
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    
    print(f"\n{model_name} - Accuracy promedio: {mean_acc:.4f} +/- {std_acc:.4f}")
    
    return mean_acc, std_acc, accuracies

def load_and_preprocess_image(img_path):
    """Carga y preprocesa una imagen"""
    img = Image.open(img_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img, dtype=np.float32) / 255.0
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    img_array = img_array.astype(np.float32)
    return img_array

# ============================================================================
# PROGRAMA PRINCIPAL
# ============================================================================

def main():
    print_section("CLASIFICACIÓN DE IMÁGENES SATELITALES - ENTRENAMIENTO LOCAL")
    print(f"Directorio del proyecto: {PROJECT_DIR}")
    print(f"TensorFlow version: {tf.__version__}")
    
    # Configurar GPU
    setup_gpu()
    
    # ========================================================================
    # a) Lectura y Partición de Datos
    # ========================================================================
    print_section("a) Lectura y Partición de Datos Locales")
    
    dataset_path = find_dataset()
    print(f"\n✓ Dataset encontrado en: {dataset_path}")
    
    # Crear particiones
    print("\nCreando partición train/test (80/20)...")
    create_train_test_split(dataset_path, TRAIN_DIR, TEST_DIR)
    print(f"\n✓ Partición completada:")
    print(f"  Train: {TRAIN_DIR}")
    print(f"  Test: {TEST_DIR}")
    
    # ========================================================================
    # b) Validación Cruzada
    # ========================================================================
    print_section("b) Validación Cruzada con 2 Arquitecturas CNN")
    
    results = {}
    
    print("\n" + "=" * 70)
    mean1, std1, accs1 = cross_validate_cnn(create_cnn1, "CNN1", TRAIN_DIR, 
                                            n_splits=NUM_FOLDS, epochs=EPOCHS_CV)
    results['CNN1'] = {'mean': mean1, 'std': std1, 'accuracies': accs1}
    
    print("\n" + "=" * 70)
    mean2, std2, accs2 = cross_validate_cnn(create_cnn2, "CNN2", TRAIN_DIR, 
                                            n_splits=NUM_FOLDS, epochs=EPOCHS_CV)
    results['CNN2'] = {'mean': mean2, 'std': std2, 'accuracies': accs2}
    
    # Guardar tabla de resultados
    results_table = pd.DataFrame({
        'CNN': ['CNN1', 'CNN2'],
        'Desempeño': [
            f"{results['CNN1']['mean']:.4f} +/- {results['CNN1']['std']:.4f}",
            f"{results['CNN2']['mean']:.4f} +/- {results['CNN2']['std']:.4f}"
        ]
    })
    
    results_csv_path = os.path.join(RESULTS_DIR, "resultados_validacion_cruzada.csv")
    results_table.to_csv(results_csv_path, index=False)
    
    print("\n" + "=" * 70)
    print("TABLA DE RESULTADOS - VALIDACIÓN CRUZADA (3-Fold)")
    print("=" * 70)
    print(results_table.to_string(index=False))
    print("=" * 70)
    print(f"\n✓ Tabla guardada en: {results_csv_path}")
    
    # Determinar mejor modelo
    if mean1 > mean2:
        best_model_name = "CNN1"
        best_model_creator = create_cnn1
        print(f"\n✓ Mejor modelo: CNN1 (Accuracy: {mean1:.4f} +/- {std1:.4f})")
    else:
        best_model_name = "CNN2"
        best_model_creator = create_cnn2
        print(f"\n✓ Mejor modelo: CNN2 (Accuracy: {mean2:.4f} +/- {std2:.4f})")
    
    # ========================================================================
    # c) Entrenamiento Final y Evaluación
    # ========================================================================
    print_section("c) Entrenamiento Final y Evaluación en Test")
    
    print(f"\nEntrenando {best_model_name} con todo el conjunto de entrenamiento...")
    
    train_gen = create_data_generator(TRAIN_DIR, BATCH_SIZE, shuffle=True)
    test_gen = create_data_generator(TEST_DIR, BATCH_SIZE, shuffle=False)
    
    final_model = best_model_creator()
    
    train_steps = len(glob.glob(os.path.join(TRAIN_DIR, "**", "*.jpg"), recursive=True) + 
                      glob.glob(os.path.join(TRAIN_DIR, "**", "*.png"), recursive=True)) // BATCH_SIZE
    test_steps = len(glob.glob(os.path.join(TEST_DIR, "**", "*.jpg"), recursive=True) + 
                     glob.glob(os.path.join(TEST_DIR, "**", "*.png"), recursive=True)) // BATCH_SIZE
    
    print(f"\nEntrenando por {EPOCHS_FINAL} épocas...")
    history = final_model.fit(
        train_gen,
        steps_per_epoch=train_steps,
        epochs=EPOCHS_FINAL,
        verbose=1
    )
    
    # Guardar modelo
    model_path = os.path.join(MODELS_DIR, "best_model.h5")
    final_model.save(model_path)
    print(f"\n✓ Modelo guardado en: {model_path}")
    
    # Evaluar en test
    print("\nEvaluando en conjunto de prueba...")
    test_gen.reset()
    y_true = []
    y_pred = []
    
    for i in range(test_steps):
        batch_x, batch_y = next(test_gen)
        predictions = final_model.predict(batch_x, verbose=0)
        y_true.extend(np.argmax(batch_y, axis=1))
        y_pred.extend(np.argmax(predictions, axis=1))
    
    # Calcular métricas
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)
    
    # Guardar métricas
    metrics_path = os.path.join(RESULTS_DIR, "metricas_finales.txt")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("RESULTADOS EN CONJUNTO DE PRUEBA\n")
        f.write("=" * 70 + "\n")
        f.write(f"Accuracy:  {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1-Score:  {f1:.4f}\n")
        f.write("=" * 70 + "\n\n")
        f.write("Reporte de Clasificación:\n")
        f.write(classification_report(y_true, y_pred, target_names=CLASSES))
    
    print("\n" + "=" * 70)
    print("RESULTADOS EN CONJUNTO DE PRUEBA")
    print("=" * 70)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("=" * 70)
    print(f"\n✓ Métricas guardadas en: {metrics_path}")
    
    # Matriz de confusión
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Matriz de Confusión')
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta Predicha')
    plt.tight_layout()
    confusion_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(confusion_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Matriz de confusión guardada en: {confusion_path}")
    
    # ========================================================================
    # d) Ejemplos de Predicciones
    # ========================================================================
    print_section("d) Ejemplos de Predicciones Correctas e Incorrectas")
    
    test_images_data = []
    for class_idx, class_name in enumerate(CLASSES):
        class_dir = os.path.join(TEST_DIR, class_name)
        if os.path.exists(class_dir):
            images = glob.glob(os.path.join(class_dir, "*.jpg")) + \
                     glob.glob(os.path.join(class_dir, "*.png")) + \
                     glob.glob(os.path.join(class_dir, "*.JPG")) + \
                     glob.glob(os.path.join(class_dir, "*.PNG"))
            for img_path in images:
                test_images_data.append({
                    'path': img_path,
                    'true_label': class_idx,
                    'true_class': class_name
                })
    
    print(f"Generando predicciones para {len(test_images_data)} imágenes...")
    for idx, item in enumerate(test_images_data):
        try:
            img_array = load_and_preprocess_image(item['path'])
            img_batch = np.expand_dims(img_array, axis=0)
            prediction = final_model.predict(img_batch, verbose=0)
            item['pred_label'] = np.argmax(prediction[0])
            item['pred_class'] = CLASSES[item['pred_label']]
            item['confidence'] = np.max(prediction[0])
            item['correct'] = item['true_label'] == item['pred_label']
            
            if (idx + 1) % 100 == 0:
                print(f"  Procesadas {idx + 1}/{len(test_images_data)} imágenes...")
        except Exception as e:
            print(f"  ⚠ Error procesando {item['path']}: {e}")
            continue
    
    correct_predictions = [item for item in test_images_data if item['correct']]
    incorrect_predictions = [item for item in test_images_data if not item['correct']]
    
    print(f"\nTotal de imágenes: {len(test_images_data)}")
    print(f"Correctas: {len(correct_predictions)}")
    print(f"Incorrectas: {len(incorrect_predictions)}")
    
    # Seleccionar ejemplos
    correct_examples = []
    used_classes = set()
    for item in correct_predictions:
        if item['true_class'] not in used_classes and len(correct_examples) < 5:
            correct_examples.append(item)
            used_classes.add(item['true_class'])
    if len(correct_examples) < 5:
        for item in correct_predictions:
            if len(correct_examples) < 5:
                correct_examples.append(item)
    
    incorrect_examples = []
    used_classes = set()
    for item in incorrect_predictions:
        if item['true_class'] not in used_classes and len(incorrect_examples) < 5:
            incorrect_examples.append(item)
            used_classes.add(item['true_class'])
    if len(incorrect_examples) < 5:
        for item in incorrect_predictions:
            if len(incorrect_examples) < 5:
                incorrect_examples.append(item)
    
    # Visualizar ejemplos correctos
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle('Ejemplos de Predicciones CORRECTAS', fontsize=16, fontweight='bold')
    for idx, example in enumerate(correct_examples[:5]):
        img = Image.open(example['path'])
        axes[idx].imshow(img)
        axes[idx].set_title(
            f"Real: {example['true_class']}\nPred: {example['pred_class']}\nConf: {example['confidence']:.2f}",
            fontsize=10
        )
        axes[idx].axis('off')
    plt.tight_layout()
    correct_path = os.path.join(RESULTS_DIR, "correct_predictions.png")
    plt.savefig(correct_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Ejemplos correctos guardados en: {correct_path}")
    
    # Visualizar ejemplos incorrectos
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle('Ejemplos de Predicciones INCORRECTAS', fontsize=16, fontweight='bold', color='red')
    for idx, example in enumerate(incorrect_examples[:5]):
        img = Image.open(example['path'])
        axes[idx].imshow(img)
        axes[idx].set_title(
            f"Real: {example['true_class']}\nPred: {example['pred_class']}\nConf: {example['confidence']:.2f}",
            fontsize=10,
            color='red'
        )
        axes[idx].axis('off')
    plt.tight_layout()
    incorrect_path = os.path.join(RESULTS_DIR, "incorrect_predictions.png")
    plt.savefig(incorrect_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Ejemplos incorrectos guardados en: {incorrect_path}")
    
    # ========================================================================
    # Resumen Final
    # ========================================================================
    print_section("PROYECTO COMPLETADO")
    print(f"✓ Modelo entrenado y guardado en: {model_path}")
    print(f"✓ Resultados guardados en: {RESULTS_DIR}")
    print(f"  - Tabla de validación cruzada: resultados_validacion_cruzada.csv")
    print(f"  - Métricas finales: metricas_finales.txt")
    print(f"  - Matriz de confusión: confusion_matrix.png")
    print(f"  - Ejemplos correctos: correct_predictions.png")
    print(f"  - Ejemplos incorrectos: incorrect_predictions.png")
    print("\nEl modelo está listo para usar. Puedes cargarlo con:")
    print(f"  model = keras.models.load_model('{model_path}')")
    print("=" * 70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Entrenamiento interrumpido por el usuario.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

