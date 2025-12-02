/**
 * Componente principal de la aplicación
 */
import React, { useState } from 'react';
import { ImageUploader } from './components/ImageUploader';
import { PredictionResult } from './components/PredictionResult';
import { BatchResults } from './components/BatchResults';
import { ApiClient } from '@infrastructure/api-client';
import { PredictionResponse, BatchResponse } from '@domain/entities';
import './App.css';

const apiClient = new ApiClient();

type Mode = 'single' | 'batch';

export const App: React.FC = () => {
  const [mode, setMode] = useState<Mode>('single');
  
  // Modo single
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  
  // Modo batch
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [imagePreviews, setImagePreviews] = useState<{ [key: string]: string }>({});
  const [batchResult, setBatchResult] = useState<BatchResponse | null>(null);
  
  // Estados compartidos
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

  const handleImagesSelect = (files: File[]) => {
    setSelectedFiles(files);
    setBatchResult(null);
    setError(null);

    // Crear previews de todas las imágenes
    const previews: { [key: string]: string } = {};
    files.forEach((file) => {
      const reader = new FileReader();
      reader.onloadend = () => {
        previews[file.name] = reader.result as string;
        setImagePreviews({ ...previews });
      };
      reader.readAsDataURL(file);
    });
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

  const handlePredictBatch = async () => {
    if (selectedFiles.length === 0) return;

    setLoading(true);
    setError(null);

    try {
      const batchResponse = await apiClient.predictBatch(selectedFiles);
      setBatchResult(batchResponse);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Error desconocido');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    if (mode === 'single') {
      setSelectedFile(null);
      setSelectedImage(null);
      setResult(null);
    } else {
      setSelectedFiles([]);
      setImagePreviews({});
      setBatchResult(null);
    }
    setError(null);
  };

  const handleModeChange = (newMode: Mode) => {
    setMode(newMode);
    handleReset();
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>🌍 Clasificador de Imágenes Satelitales</h1>
        <p>Sube una o múltiples imágenes y descubre a qué categoría pertenecen</p>
        <div className="mode-selector">
          <button
            className={`mode-btn ${mode === 'single' ? 'active' : ''}`}
            onClick={() => handleModeChange('single')}
          >
            📷 Una Imagen
          </button>
          <button
            className={`mode-btn ${mode === 'batch' ? 'active' : ''}`}
            onClick={() => handleModeChange('batch')}
          >
            📚 Múltiples Imágenes
          </button>
        </div>
      </header>

      <main className={`app-main ${mode === 'batch' ? 'batch-mode' : ''}`}>
        {mode === 'single' ? (
          <>
            <div className="upload-section">
              <ImageUploader
                onImageSelect={handleImageSelect}
                selectedImage={selectedImage}
                multiple={false}
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
          </>
        ) : (
          <div className="batch-section">
            <div className="upload-section">
              <ImageUploader
                onImagesSelect={handleImagesSelect}
                selectedImage={null}
                multiple={true}
                selectedFiles={selectedFiles}
              />
              
              <div className="actions">
                {selectedFiles.length > 0 && !loading && (
                  <>
                    <button onClick={handlePredictBatch} className="btn btn-primary">
                      Clasificar {selectedFiles.length} Imagen{selectedFiles.length > 1 ? 'es' : ''}
                    </button>
                    <button onClick={handleReset} className="btn btn-secondary">
                      Cambiar Imágenes
                    </button>
                  </>
                )}
              </div>
            </div>

            <div className="result-section">
              <BatchResults
                batchResponse={batchResult}
                imagePreviews={imagePreviews}
                loading={loading}
                error={error}
              />
            </div>
          </div>
        )}
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

