"""
API REST - Capa de presentación
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
import base64
from typing import List, Optional
from app.infrastructure.model_service import TensorFlowModelRepository
from app.domain.entities import CLASSES
import os

# Inicializar FastAPI
app = FastAPI(
    title="Clasificador de Imágenes Satelitales",
    description="API para clasificar imágenes satelitales usando CNN",
    version="1.0.0"
)

# Configurar CORS para permitir requests del frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite y React por defecto
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar el modelo al iniciar
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../../..", "best_model.h5")
model_repository: TensorFlowModelRepository = None


@app.on_event("startup")
async def startup_event():
    """Carga el modelo al iniciar la aplicación"""
    global model_repository
    try:
        model_repository = TensorFlowModelRepository(MODEL_PATH)
        print(f"✅ Modelo cargado exitosamente desde: {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Error al cargar el modelo: {e}")
        raise


@app.get("/")
async def root():
    """Endpoint raíz"""
    return {
        "message": "API de Clasificación de Imágenes Satelitales",
        "version": "1.0.0"
    }


@app.get("/classes")
async def get_classes():
    """Obtiene las clases disponibles para clasificación"""
    return {
        "classes": [
            {
                "name": cls.name,
                "display_name": cls.display_name,
                "emoji": cls.emoji
            }
            for cls in CLASSES
        ]
    }


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Clasifica una imagen satelital
    
    Args:
        file: Archivo de imagen a clasificar
        
    Returns:
        JSON con la predicción y confianza
    """
    if model_repository is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    # Validar tipo de archivo
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="El archivo debe ser una imagen"
        )
    
    try:
        # Leer imagen
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Preprocesar imagen
        image_array = model_repository.preprocess_image(image)
        
        # Realizar predicción
        prediction = model_repository.predict(image_array)
        
        return JSONResponse({
            "success": True,
            "prediction": {
                "class": prediction.class_name,
                "display_name": next(
                    cls.display_name for cls in CLASSES 
                    if cls.name == prediction.class_name
                ),
                "emoji": next(
                    cls.emoji for cls in CLASSES 
                    if cls.name == prediction.class_name
                ),
                "confidence": round(prediction.confidence * 100, 2)
            },
            "all_predictions": [
                {
                    **pred,
                    "confidence": round(pred["confidence"] * 100, 2)
                }
                for pred in prediction.all_predictions
            ]
        })
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al procesar la imagen: {str(e)}"
        )


@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Clasifica múltiples imágenes satelitales en lote
    
    Args:
        files: Lista de archivos de imagen a clasificar
        
    Returns:
        JSON con las predicciones de todas las imágenes
    """
    if model_repository is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    if not files or len(files) == 0:
        raise HTTPException(
            status_code=400,
            detail="Debe proporcionar al menos una imagen"
        )
    
    results = []
    
    for idx, file in enumerate(files):
        # Validar tipo de archivo
        if not file.content_type or not file.content_type.startswith("image/"):
            results.append({
                "filename": file.filename or f"imagen_{idx}",
                "success": False,
                "error": "El archivo debe ser una imagen"
            })
            continue
        
        try:
            # Leer imagen
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data))
            
            # Preprocesar imagen
            image_array = model_repository.preprocess_image(image)
            
            # Realizar predicción
            prediction = model_repository.predict(image_array)
            
            results.append({
                "filename": file.filename or f"imagen_{idx}",
                "success": True,
                "prediction": {
                    "class": prediction.class_name,
                    "display_name": next(
                        cls.display_name for cls in CLASSES 
                        if cls.name == prediction.class_name
                    ),
                    "emoji": next(
                        cls.emoji for cls in CLASSES 
                        if cls.name == prediction.class_name
                    ),
                    "confidence": round(prediction.confidence * 100, 2)
                },
                "all_predictions": [
                    {
                        **pred,
                        "confidence": round(pred["confidence"] * 100, 2)
                    }
                    for pred in prediction.all_predictions
                ]
            })
        
        except Exception as e:
            results.append({
                "filename": file.filename or f"imagen_{idx}",
                "success": False,
                "error": f"Error al procesar la imagen: {str(e)}"
            })
    
    return JSONResponse({
        "success": True,
        "total": len(files),
        "processed": len([r for r in results if r.get("success", False)]),
        "failed": len([r for r in results if not r.get("success", False)]),
        "results": results
    })


@app.post("/predict/segment")
async def predict_segment(
    file: UploadFile = File(...),
    tile_size: int = Query(64, ge=32, le=256, description="Tamaño de cada tile"),
    stride: int = Query(32, ge=16, le=128, description="Paso entre tiles (overlap)"),
    border_width: int = Query(2, ge=1, le=10, description="Grosor de los bordes")
):
    """
    Segmenta una imagen dividiéndola en tiles y clasificando cada uno.
    Devuelve la imagen con bordes de colores según la clasificación.
    
    Args:
        file: Archivo de imagen a segmentar
        tile_size: Tamaño de cada tile (32-256, default: 64)
        stride: Paso entre tiles (16-128, default: 32)
        border_width: Grosor de los bordes (1-10, default: 2)
        
    Returns:
        JSON con la imagen segmentada en base64 y estadísticas
    """
    if model_repository is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    # Validar tipo de archivo
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="El archivo debe ser una imagen"
        )
    
    # Validar stride
    if stride >= tile_size:
        raise HTTPException(
            status_code=400,
            detail="El stride debe ser menor que el tile_size"
        )
    
    try:
        # Leer imagen
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Validar tamaño máximo (para evitar problemas de memoria)
        max_size = 4096
        if image.size[0] > max_size or image.size[1] > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"La imagen es demasiado grande. Tamaño máximo: {max_size}x{max_size}"
            )
        
        # Segmentar imagen
        segmented_image, stats = model_repository.segment_image_optimized(
            image,
            tile_size=tile_size,
            stride=stride,
            border_width=border_width
        )
        
        # Convertir imagen segmentada a base64
        img_buffer = io.BytesIO()
        segmented_image.save(img_buffer, format='PNG')
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        
        # Obtener información de clases con colores
        class_info = []
        class_colors_map = {
            'cloudy': {'rgb': [255, 200, 100], 'hex': '#FFC864'},
            'desert': {'rgb': [255, 220, 177], 'hex': '#FFDCB1'},
            'green_area': {'rgb': [100, 255, 100], 'hex': '#64FF64'},
            'water': {'rgb': [100, 150, 255], 'hex': '#6496FF'},
        }
        
        for cls in CLASSES:
            class_name = cls.name
            color_info = class_colors_map.get(class_name, {'rgb': [255, 255, 255], 'hex': '#FFFFFF'})
            distribution = stats['class_distribution'].get(class_name, {'pixels': 0, 'percentage': 0})
            
            class_info.append({
                'name': class_name,
                'display_name': cls.display_name,
                'emoji': cls.emoji,
                'color': color_info,
                'pixels': distribution['pixels'],
                'percentage': distribution['percentage']
            })
        
        return JSONResponse({
            "success": True,
            "image_base64": img_base64,
            "original_size": {
                "width": image.size[0],
                "height": image.size[1]
            },
            "segmented_size": {
                "width": segmented_image.size[0],
                "height": segmented_image.size[1]
            },
            "parameters": {
                "tile_size": tile_size,
                "stride": stride,
                "border_width": border_width,
                "total_tiles": stats['total_tiles']
            },
            "class_distribution": class_info,
            "stats": stats
        })
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al procesar la segmentación: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Endpoint de salud para verificar que el servicio está funcionando"""
    return {
        "status": "healthy",
        "model_loaded": model_repository is not None
    }

