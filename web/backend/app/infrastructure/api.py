"""
API REST - Capa de presentación
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
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


@app.get("/health")
async def health_check():
    """Endpoint de salud para verificar que el servicio está funcionando"""
    return {
        "status": "healthy",
        "model_loaded": model_repository is not None
    }

