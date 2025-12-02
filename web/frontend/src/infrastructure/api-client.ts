/**
 * Cliente API - Implementación del repositorio
 */
import { ApiRepository } from '@domain/repositories';
import { PredictionResponse, ImageClass, BatchResponse } from '@domain/entities';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export class ApiClient implements ApiRepository {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  async getClasses(): Promise<{ classes: ImageClass[] }> {
    const response = await fetch(`${this.baseUrl}/classes`);
    
    if (!response.ok) {
      throw new Error(`Error al obtener clases: ${response.statusText}`);
    }

    return await response.json();
  }

  async predictImage(file: File): Promise<PredictionResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${this.baseUrl}/predict`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(error.detail || `Error al predecir: ${response.statusText}`);
    }

    return await response.json();
  }

  async predictBatch(files: File[]): Promise<BatchResponse> {
    const formData = new FormData();
    files.forEach((file) => {
      formData.append('files', file);
    });

    const response = await fetch(`${this.baseUrl}/predict/batch`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(error.detail || `Error al predecir en lote: ${response.statusText}`);
    }

    return await response.json();
  }
}

