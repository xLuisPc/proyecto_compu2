/**
 * Entidades del dominio - Representan los conceptos del negocio
 */

export interface Prediction {
  class: string;
  display_name: string;
  emoji: string;
  confidence: number;
}

export interface AllPrediction {
  class: string;
  display_name: string;
  emoji: string;
  confidence: number;
}

export interface PredictionResponse {
  success: boolean;
  prediction: Prediction;
  all_predictions: AllPrediction[];
}

export interface ImageClass {
  name: string;
  display_name: string;
  emoji: string;
}

