/**
 * Componente para mostrar los resultados de la predicción
 */
import React from 'react';
import { PredictionResponse } from '@domain/entities';

interface PredictionResultProps {
  result: PredictionResponse | null;
  loading: boolean;
  error: string | null;
}

export const PredictionResult: React.FC<PredictionResultProps> = ({
  result,
  loading,
  error,
}) => {
  if (loading) {
    return (
      <div className="prediction-result loading">
        <div className="spinner"></div>
        <p>Analizando imagen...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="prediction-result error">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor">
          <circle cx="12" cy="12" r="10" />
          <line x1="12" y1="8" x2="12" y2="12" />
          <line x1="12" y1="16" x2="12.01" y2="16" />
        </svg>
        <p>{error}</p>
      </div>
    );
  }

  if (!result) {
    return null;
  }

  const { prediction, all_predictions } = result;

  return (
    <div className="prediction-result">
      <div className="main-prediction">
        <div className="prediction-icon">{prediction.emoji}</div>
        <h2>{prediction.display_name}</h2>
        <div className="confidence-bar">
          <div
            className="confidence-fill"
            style={{ width: `${prediction.confidence}%` }}
          />
        </div>
        <p className="confidence-text">
          Confianza: <strong>{prediction.confidence}%</strong>
        </p>
      </div>

      <div className="all-predictions">
        <h3>Probabilidades de todas las clases:</h3>
        <div className="predictions-list">
          {all_predictions.map((pred, index) => (
            <div key={index} className="prediction-item">
              <div className="prediction-header">
                <span className="prediction-emoji">{pred.emoji}</span>
                <span className="prediction-name">{pred.display_name}</span>
                <span className="prediction-percentage">
                  {pred.confidence.toFixed(2)}%
                </span>
              </div>
              <div className="prediction-bar">
                <div
                  className="prediction-bar-fill"
                  style={{ width: `${pred.confidence}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

