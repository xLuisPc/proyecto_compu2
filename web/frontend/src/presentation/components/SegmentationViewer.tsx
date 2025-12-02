/**
 * Componente para mostrar la imagen segmentada con bordes de colores
 */
import React from 'react';
import { SegmentationResponse } from '@domain/entities';

interface SegmentationViewerProps {
  segmentationResponse: SegmentationResponse | null;
  loading: boolean;
  error: string | null;
}

export const SegmentationViewer: React.FC<SegmentationViewerProps> = ({
  segmentationResponse,
  loading,
  error,
}) => {
  if (loading) {
    return (
      <div className="segmentation-viewer loading">
        <div className="spinner"></div>
        <p>Procesando segmentación...</p>
        <p className="loading-hint">Esto puede tardar varios segundos dependiendo del tamaño de la imagen</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="segmentation-viewer error">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor">
          <circle cx="12" cy="12" r="10" />
          <line x1="12" y1="8" x2="12" y2="12" />
          <line x1="12" y1="16" x2="12.01" y2="16" />
        </svg>
        <p>{error}</p>
      </div>
    );
  }

  if (!segmentationResponse) {
    return null;
  }

  const { image_base64, parameters, class_distribution, original_size } = segmentationResponse;

  return (
    <div className="segmentation-viewer">
      <div className="segmentation-header">
        <h2>Imagen Segmentada</h2>
        <div className="segmentation-stats">
          <div className="stat-item">
            <span className="stat-label">Tiles procesados:</span>
            <span className="stat-value">{parameters.total_tiles}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Tamaño original:</span>
            <span className="stat-value">
              {original_size.width} × {original_size.height} px
            </span>
          </div>
        </div>
      </div>

      <div className="segmented-image-container">
        <img
          src={`data:image/png;base64,${image_base64}`}
          alt="Imagen segmentada"
          className="segmented-image"
        />
      </div>

      <div className="segmentation-legend">
        <h3>Distribución de Biomas</h3>
        <div className="legend-items">
          {class_distribution.map((classInfo, index) => (
            <div key={index} className="legend-item">
              <div
                className="legend-color"
                style={{ backgroundColor: classInfo.color.hex }}
              />
              <div className="legend-info">
                <span className="legend-emoji">{classInfo.emoji}</span>
                <span className="legend-name">{classInfo.display_name}</span>
                <span className="legend-percentage">
                  {classInfo.percentage.toFixed(2)}%
                </span>
              </div>
              <div className="legend-bar">
                <div
                  className="legend-bar-fill"
                  style={{
                    width: `${classInfo.percentage}%`,
                    backgroundColor: classInfo.color.hex,
                  }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="segmentation-parameters">
        <h3>Parámetros de Segmentación</h3>
        <div className="parameters-grid">
          <div className="parameter-item">
            <span className="parameter-label">Tamaño de tile:</span>
            <span className="parameter-value">{parameters.tile_size} × {parameters.tile_size} px</span>
          </div>
          <div className="parameter-item">
            <span className="parameter-label">Stride (overlap):</span>
            <span className="parameter-value">{parameters.stride} px</span>
          </div>
          <div className="parameter-item">
            <span className="parameter-label">Grosor de bordes:</span>
            <span className="parameter-value">{parameters.border_width} px</span>
          </div>
        </div>
      </div>
    </div>
  );
};

