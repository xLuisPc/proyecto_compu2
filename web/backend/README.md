# Backend - API de Clasificación de Imágenes Satelitales

Backend desarrollado con FastAPI siguiendo arquitectura limpia.

## 🏗️ Arquitectura

```
backend/
├── app/
│   ├── domain/              # Capa de dominio (entidades, interfaces)
│   │   ├── entities.py      # Entidades del negocio
│   │   └── repositories.py  # Interfaces de repositorios
│   ├── infrastructure/      # Capa de infraestructura (implementaciones)
│   │   ├── model_service.py # Implementación del servicio de modelo
│   │   └── api.py           # API REST con FastAPI
│   └── main.py              # Punto de entrada
├── requirements.txt
└── README.md
```

## 📦 Instalación

1. Crear entorno virtual:
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## 🚀 Ejecución

```bash
python main.py
```

O directamente con uvicorn:
```bash
uvicorn app.infrastructure.api:app --reload --port 8000
```

La API estará disponible en: `http://localhost:8000`

## 📚 Endpoints

### GET `/`
Información básica de la API

### GET `/classes`
Obtiene las clases disponibles para clasificación

### POST `/predict`
Clasifica una imagen satelital

**Request:**
- Content-Type: `multipart/form-data`
- Body: archivo de imagen

**Response:**
```json
{
  "success": true,
  "prediction": {
    "class": "cloudy",
    "display_name": "Nubes",
    "emoji": "☁️",
    "confidence": 95.23
  },
  "all_predictions": [
    {
      "class": "cloudy",
      "display_name": "Nubes",
      "emoji": "☁️",
      "confidence": 95.23
    },
    ...
  ]
}
```

### GET `/health`
Verifica el estado del servicio

## 🔧 Configuración

El modelo se carga desde `best_model.h5` en la raíz del proyecto. Asegúrate de que el archivo exista antes de iniciar el servidor.

