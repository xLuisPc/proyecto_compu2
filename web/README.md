# Aplicación Web - Clasificador de Imágenes Satelitales

Aplicación web completa para clasificar imágenes satelitales usando el modelo entrenado. Desarrollada con arquitectura limpia.

## 🏗️ Arquitectura

El proyecto está dividido en dos partes principales:

### Backend (Python + FastAPI)
- **Capa de Dominio**: Entidades y interfaces
- **Capa de Infraestructura**: Implementación del modelo y API REST
- **Capa de Presentación**: Endpoints FastAPI

### Frontend (TypeScript + React + Vite)
- **Capa de Dominio**: Entidades y interfaces TypeScript
- **Capa de Infraestructura**: Cliente API
- **Capa de Presentación**: Componentes React

## 📁 Estructura del Proyecto

```
web/
├── backend/                 # API REST con FastAPI
│   ├── app/
│   │   ├── domain/         # Entidades e interfaces
│   │   └── infrastructure/ # Implementaciones
│   ├── main.py
│   └── requirements.txt
│
└── frontend/               # Aplicación React
    ├── src/
    │   ├── domain/         # Entidades e interfaces
    │   ├── infrastructure/ # Cliente API
    │   └── presentation/   # Componentes React
    └── package.json
```

## 🚀 Inicio Rápido

### 1. Backend

```bash
cd backend

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar servidor
python main.py
```

El backend estará disponible en: `http://localhost:8000`

### 2. Frontend

```bash
cd frontend

# Instalar dependencias
npm install

# Ejecutar en modo desarrollo
npm run dev
```

El frontend estará disponible en: `http://localhost:5173`

## 📚 Endpoints del Backend

- `GET /` - Información de la API
- `GET /classes` - Obtener clases disponibles
- `POST /predict` - Clasificar una imagen
- `GET /health` - Estado del servicio

## 🎨 Características del Frontend

- ✅ Carga de imágenes por clic o drag & drop
- ✅ Preview de imagen
- ✅ Visualización de resultados con confianza
- ✅ Mostrar todas las probabilidades
- ✅ Diseño responsive y moderno

## 🔧 Requisitos

- Python 3.7+
- Node.js 18+
- Modelo entrenado (`best_model.h5`) en la raíz del proyecto

## 📝 Notas

- El modelo debe estar en la raíz del proyecto como `best_model.h5`
- El backend carga el modelo al iniciar
- El frontend se conecta automáticamente al backend en `http://localhost:8000`

