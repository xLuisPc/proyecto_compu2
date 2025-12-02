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

export interface BatchResultItem {
  filename: string;
  success: boolean;
  prediction?: Prediction;
  all_predictions?: AllPrediction[];
  error?: string;
}

export interface BatchResponse {
  success: boolean;
  total: number;
  processed: number;
  failed: number;
  results: BatchResultItem[];
}

export interface ClassColor {
  rgb: number[];
  hex: string;
}

export interface ClassDistribution {
  name: string;
  display_name: string;
  emoji: string;
  color: ClassColor;
  pixels: number;
  percentage: number;
}

export interface SegmentationResponse {
  success: boolean;
  image_base64: string;
  original_size: {
    width: number;
    height: number;
  };
  segmented_size: {
    width: number;
    height: number;
  };
  parameters: {
    tile_size: number;
    stride: number;
    border_width: number;
    total_tiles: number;
  };
  class_distribution: ClassDistribution[];
  stats: {
    total_tiles: number;
    image_size: {
      width: number;
      height: number;
    };
    tile_size: number;
    stride: number;
    class_distribution: {
      [key: string]: {
        pixels: number;
        percentage: number;
      };
    };
  };
}

