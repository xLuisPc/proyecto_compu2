# Configuración del Backend

## ⚠️ Problema con Python 3.14

TensorFlow actualmente **no es compatible con Python 3.14**. TensorFlow soporta Python 3.9 hasta Python 3.12.

## Solución: Usar Python 3.11 o 3.12

### Opción 1: Crear un nuevo entorno virtual con Python 3.11 o 3.12

```bash
# Eliminar el entorno virtual actual
rm -rf venv

# Crear nuevo entorno con Python 3.11 (si está instalado)
python3.11 -m venv venv

# O con Python 3.12
python3.12 -m venv venv

# Activar el entorno
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### Opción 2: Verificar versiones de Python disponibles

```bash
# Ver qué versiones de Python tienes instaladas
ls /usr/local/bin/python*  # macOS con Homebrew
# o
which python3.11
which python3.12
```

### Opción 3: Instalar Python 3.12 con Homebrew (macOS)

```bash
brew install python@3.12
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Verificar la instalación

Después de instalar las dependencias, verifica que TensorFlow esté instalado:

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

Deberías ver algo como: `2.15.0` o similar.

