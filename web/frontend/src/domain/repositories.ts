/**
 * Repositorios - Interfaces para acceso a datos
 */
import { PredictionResponse, ImageClass } from './entities';

import { BatchResponse, SegmentationResponse } from './entities';

export interface ApiRepository {
  /**
   * Obtiene las clases disponibles para clasificación
   */
  getClasses(): Promise<{ classes: ImageClass[] }>;

  /**
   * Clasifica una imagen satelital
   * @param file Archivo de imagen a clasificar
   */
  predictImage(file: File): Promise<PredictionResponse>;

  /**
   * Clasifica múltiples imágenes satelitales en lote
   * @param files Array de archivos de imagen a clasificar
   */
  predictBatch(files: File[]): Promise<BatchResponse>;

  /**
   * Segmenta una imagen dividiéndola en tiles y clasificando cada uno
   * @param file Archivo de imagen a segmentar
   * @param tileSize Tamaño de cada tile (default: 64)
   * @param stride Paso entre tiles (default: 32)
   * @param borderWidth Grosor de los bordes (default: 2)
   */
  predictSegment(
    file: File,
    tileSize?: number,
    stride?: number,
    borderWidth?: number
  ): Promise<SegmentationResponse>;
}

