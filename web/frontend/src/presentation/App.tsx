/**
 * Componente principal de la aplicación
 */
import React, { useState } from 'react';
import { ImageUploader } from './components/ImageUploader';
import { PredictionResult } from './components/PredictionResult';
import { ApiClient } from '@infrastructure/api-client';
import { PredictionResponse } from '@domain/entities';
import './App.css';

const apiClient = new ApiClient();

export const App: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleImageSelect = (file: File) => {
    setSelectedFile(file);
    setResult(null);
    setError(null);

    // Crear preview de la imagen
    const reader = new FileReader();
    reader.onloadend = () => {
      setSelectedImage(reader.result as string);
    };
    reader.readAsDataURL(file);
  };

  const handlePredict = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);

    try {
      const prediction = await apiClient.predictImage(selectedFile);
      setResult(prediction);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Error desconocido');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setSelectedImage(null);
    setResult(null);
    setError(null);
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>🌍 Clasificador de Imágenes Satelitales</h1>
        <p>Sube una imagen y descubre a qué categoría pertenece</p>
      </header>

      <main className="app-main">
        <div className="upload-section">
          <ImageUploader
            onImageSelect={handleImageSelect}
            selectedImage={selectedImage}
          />
          
          <div className="actions">
            {selectedFile && !loading && (
              <>
                <button onClick={handlePredict} className="btn btn-primary">
                  Clasificar Imagen
                </button>
                <button onClick={handleReset} className="btn btn-secondary">
                  Cambiar Imagen
                </button>
              </>
            )}
          </div>
        </div>

        <div className="result-section">
          <PredictionResult
            result={result}
            loading={loading}
            error={error}
          />
        </div>
      </main>

      <footer className="app-footer">
        <p>
          Clasifica imágenes en: <strong>Nubes ☁️</strong>,{' '}
          <strong>Desierto 🏜️</strong>, <strong>Área Verde 🌿</strong>,{' '}
          <strong>Agua 💧</strong>
        </p>
      </footer>
    </div>
  );
};

