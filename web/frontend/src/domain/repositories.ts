/**
 * Repositorios - Interfaces para acceso a datos
 */
import { PredictionResponse, ImageClass } from './entities';

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
}

