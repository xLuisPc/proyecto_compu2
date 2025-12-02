/**
 * Componente para mostrar los resultados de predicción en lote
 */
import React, { useState } from 'react';
import { BatchResponse, BatchResultItem } from '@domain/entities';

interface BatchResultsProps {
  batchResponse: BatchResponse | null;
  imagePreviews: { [key: string]: string };
  loading: boolean;
  error: string | null;
}

export const BatchResults: React.FC<BatchResultsProps> = ({
  batchResponse,
  imagePreviews,
  loading,
  error,
}) => {
  const [expandedImage, setExpandedImage] = useState<string | null>(null);

  if (loading) {
    return (
      <div className="batch-results loading">
        <div className="spinner"></div>
        <p>Procesando imágenes...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="batch-results error">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor">
          <circle cx="12" cy="12" r="10" />
          <line x1="12" y1="8" x2="12" y2="12" />
          <line x1="12" y1="16" x2="12.01" y2="16" />
        </svg>
        <p>{error}</p>
      </div>
    );
  }

  if (!batchResponse || !batchResponse.results || batchResponse.results.length === 0) {
    return null;
  }

  const { results, total, processed, failed } = batchResponse;

  return (
    <div className="batch-results">
      <div className="batch-summary">
        <h2>Resultados del Procesamiento</h2>
        <div className="summary-stats">
          <div className="stat-item">
            <span className="stat-label">Total:</span>
            <span className="stat-value">{total}</span>
          </div>
          <div className="stat-item success">
            <span className="stat-label">Exitosas:</span>
            <span className="stat-value">{processed}</span>
          </div>
          {failed > 0 && (
            <div className="stat-item error">
              <span className="stat-label">Fallidas:</span>
              <span className="stat-value">{failed}</span>
            </div>
          )}
        </div>
      </div>

      <div className="batch-grid">
        {results.map((result, index) => (
          <BatchResultCard
            key={index}
            result={result}
            imagePreview={imagePreviews[result.filename]}
            isExpanded={expandedImage === result.filename}
            onToggleExpand={() => 
              setExpandedImage(expandedImage === result.filename ? null : result.filename)
            }
          />
        ))}
      </div>
    </div>
  );
};

interface BatchResultCardProps {
  result: BatchResultItem;
  imagePreview: string | undefined;
  isExpanded: boolean;
  onToggleExpand: () => void;
}

const BatchResultCard: React.FC<BatchResultCardProps> = ({
  result,
  imagePreview,
  isExpanded,
  onToggleExpand,
}) => {
  if (!result.success) {
    return (
      <div className="batch-card error-card">
        <div className="card-image">
          {imagePreview ? (
            <img src={imagePreview} alt={result.filename} />
          ) : (
            <div className="image-placeholder">
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                <circle cx="8.5" cy="8.5" r="1.5" />
                <polyline points="21 15 16 10 5 21" />
              </svg>
            </div>
          )}
        </div>
        <div className="card-content">
          <h3 className="card-filename">{result.filename}</h3>
          <div className="error-message">
            <span className="error-icon">⚠️</span>
            <span>{result.error || 'Error desconocido'}</span>
          </div>
        </div>
      </div>
    );
  }

  const { prediction, all_predictions } = result;

  return (
    <div className={`batch-card ${isExpanded ? 'expanded' : ''}`}>
      <div className="card-image" onClick={onToggleExpand}>
        {imagePreview ? (
          <img src={imagePreview} alt={result.filename} />
        ) : (
          <div className="image-placeholder">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
              <circle cx="8.5" cy="8.5" r="1.5" />
              <polyline points="21 15 16 10 5 21" />
            </svg>
          </div>
        )}
        <div className="image-overlay">
          <span className="overlay-icon">{prediction?.emoji}</span>
        </div>
      </div>
      <div className="card-content">
        <h3 className="card-filename" title={result.filename}>
          {result.filename}
        </h3>
        <div className="card-prediction">
          <div className="prediction-main">
            <span className="prediction-emoji">{prediction?.emoji}</span>
            <span className="prediction-name">{prediction?.display_name}</span>
            <span className="prediction-confidence">
              {prediction?.confidence.toFixed(1)}%
            </span>
          </div>
          <div className="confidence-bar-small">
            <div
              className="confidence-fill-small"
              style={{ width: `${prediction?.confidence || 0}%` }}
            />
          </div>
        </div>
        {isExpanded && all_predictions && (
          <div className="expanded-details">
            <h4>Todas las predicciones:</h4>
            <div className="expanded-predictions">
              {all_predictions.map((pred, idx) => (
                <div key={idx} className="expanded-prediction-item">
                  <span className="pred-emoji">{pred.emoji}</span>
                  <span className="pred-name">{pred.display_name}</span>
                  <span className="pred-percentage">
                    {pred.confidence.toFixed(2)}%
                  </span>
                  <div className="pred-bar">
                    <div
                      className="pred-bar-fill"
                      style={{ width: `${pred.confidence}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
        <button 
          className="expand-button"
          onClick={onToggleExpand}
        >
          {isExpanded ? 'Ver menos' : 'Ver más'}
        </button>
      </div>
    </div>
  );
};

