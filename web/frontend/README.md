# Frontend - Clasificador de Imágenes Satelitales

Frontend desarrollado con React + TypeScript + Vite siguiendo arquitectura limpia.

## 🏗️ Arquitectura

```
frontend/
├── src/
│   ├── domain/              # Capa de dominio (entidades, interfaces)
│   │   ├── entities.ts      # Entidades del negocio
│   │   └── repositories.ts # Interfaces de repositorios
│   ├── infrastructure/      # Capa de infraestructura (implementaciones)
│   │   └── api-client.ts    # Cliente API
│   ├── presentation/        # Capa de presentación (componentes React)
│   │   ├── components/      # Componentes reutilizables
│   │   ├── App.tsx          # Componente principal
│   │   └── App.css          # Estilos
│   └── main.tsx            # Punto de entrada
├── package.json
└── README.md
```

## 📦 Instalación

```bash
npm install
```

## 🚀 Ejecución

### Modo desarrollo:
```bash
npm run dev
```

La aplicación estará disponible en: `http://localhost:5173`

### Compilar para producción:
```bash
npm run build
```

### Preview de producción:
```bash
npm run preview
```

## 🔧 Configuración

Por defecto, la aplicación se conecta a `http://localhost:8000`. Puedes cambiar esto creando un archivo `.env`:

```env
VITE_API_URL=http://localhost:8000
```

## 📚 Características

- ✅ Carga de imágenes por clic o arrastrar y soltar
- ✅ Preview de imagen antes de clasificar
- ✅ Visualización de resultados con confianza
- ✅ Mostrar todas las probabilidades de clases
- ✅ Diseño responsive y moderno
- ✅ Arquitectura limpia con separación de capas

