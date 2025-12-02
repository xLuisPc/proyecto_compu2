/**
 * Componente para cargar imágenes
 */
import React, { useRef, useState } from 'react';

interface ImageUploaderProps {
  onImageSelect: (file: File) => void;
  onImagesSelect?: (files: File[]) => void;
  selectedImage: string | null;
  multiple?: boolean;
  selectedFiles?: File[];
}

export const ImageUploader: React.FC<ImageUploaderProps> = ({
  onImageSelect,
  onImagesSelect,
  selectedImage,
  multiple = false,
  selectedFiles = [],
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragActive, setDragActive] = useState(false);

  const handleFileSelect = (file: File) => {
    if (file && file.type.startsWith('image/')) {
      onImageSelect(file);
    }
  };

  const handleFilesSelect = (files: FileList | null) => {
    if (!files || files.length === 0) return;
    
    const imageFiles = Array.from(files).filter(file => file.type.startsWith('image/'));
    if (imageFiles.length > 0 && onImagesSelect) {
      onImagesSelect(imageFiles);
    }
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (multiple && onImagesSelect) {
      handleFilesSelect(e.dataTransfer.files);
    } else if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (multiple && onImagesSelect) {
      handleFilesSelect(e.target.files);
    } else if (e.target.files && e.target.files[0]) {
      handleFileSelect(e.target.files[0]);
    }
  };

  const handleClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <div
      className={`upload-area ${dragActive ? 'drag-active' : ''} ${selectedImage ? 'has-image' : ''}`}
      onDragEnter={handleDrag}
      onDragLeave={handleDrag}
      onDragOver={handleDrag}
      onDrop={handleDrop}
      onClick={handleClick}
    >
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        multiple={multiple}
        onChange={handleChange}
        style={{ display: 'none' }}
      />
      
      {multiple ? (
        <div className="upload-placeholder">
          <svg
            width="64"
            height="64"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="17 8 12 3 7 8" />
            <line x1="12" y1="3" x2="12" y2="15" />
          </svg>
          <p>Haz clic o arrastra múltiples imágenes aquí</p>
          <span className="upload-hint">
            {selectedFiles.length > 0 
              ? `${selectedFiles.length} imagen(es) seleccionada(s)` 
              : 'Formatos: JPG, PNG, GIF'}
          </span>
        </div>
      ) : selectedImage ? (
        <div className="image-preview">
          <img src={selectedImage} alt="Preview" />
        </div>
      ) : (
        <div className="upload-placeholder">
          <svg
            width="64"
            height="64"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="17 8 12 3 7 8" />
            <line x1="12" y1="3" x2="12" y2="15" />
          </svg>
          <p>Haz clic o arrastra una imagen aquí</p>
          <span className="upload-hint">Formatos: JPG, PNG, GIF</span>
        </div>
      )}
    </div>
  );
};

