import React, { useState, useRef } from "react";
import {
  UploadCloud,
  CheckCircle2,
  AlertTriangle,
  FileText,
  RotateCw,
  Layers,
  ArrowRight,
} from "lucide-react";
import { validateCustomScene } from "../services/api";

export default function UserUpload({ onSceneValidated }) {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [isDragging, setIsDragging] = useState(false);
  const [isValidating, setIsValidating] = useState(false);
  const [validationInfo, setValidationInfo] = useState(null);
  const [validationError, setValidationError] = useState(null);

  const fileInputRef = useRef(null);

  const handleFiles = async (files) => {
    if (!files || files.length === 0) return;
    const fileList = Array.from(files);
    setSelectedFiles(fileList);
    setValidationInfo(null);
    setValidationError(null);

    // Auto-validate immediately against strict Level-2A requirements
    setIsValidating(true);
    try {
      const valResp = await validateCustomScene(fileList);
      setValidationInfo(valResp);
      if (valResp && valResp.is_valid && onSceneValidated) {
        // Automatically proceed to workstation so user sees tiles and super-resolved imagery immediately
        setTimeout(() => {
          onSceneValidated(valResp);
        }, 500);
      }
    } catch (err) {
      setValidationError(err.message || "Failed to validate Sentinel-2 imagery.");
    } finally {
      setIsValidating(false);
    }
  };

  const onDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const onDragLeave = () => {
    setIsDragging(false);
  };

  const onDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFiles(e.dataTransfer.files);
    }
  };

  const resetUpload = () => {
    setSelectedFiles([]);
    setValidationInfo(null);
    setValidationError(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleProceedToGrid = () => {
    if (!validationInfo) return;
    if (onSceneValidated) {
      onSceneValidated(validationInfo);
    }
  };

  return (
    <div className="user-upload-container">
      {/* Top Banner & Specifications */}
      <div className="upload-header-row">
        <div>
          <h2 className="upload-title">
            <UploadCloud className="icon-cyan" size={24} />
            User-Provided Sentinel-2 Imagery Pipeline
          </h2>
          <p className="upload-subtitle">
            Upload your own Sentinel-2 Level-2A imagery to execute learned direct 10m → 4m super-resolution,
            epistemic uncertainty quantification, empirical trust fusion, and cryptographic receipt generation.
          </p>
        </div>
        {selectedFiles.length > 0 && (
          <button className="btn-secondary btn-reset" onClick={resetUpload} disabled={isValidating}>
            <RotateCw size={14} />
            Reset Upload
          </button>
        )}
      </div>

      {/* Drag & Drop Hero Zone */}
      <div
        className={`upload-dropzone ${isDragging ? "dragging" : ""} ${validationInfo ? "has-valid-file" : ""}`}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        onClick={() => {
          if (!isValidating && fileInputRef.current) {
            fileInputRef.current.click();
          }
        }}
      >
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".tif,.tiff,.jp2,.TIF,.TIFF"
          style={{ display: "none" }}
          onChange={(e) => handleFiles(e.target.files)}
          disabled={isValidating}
        />

        <div className="dropzone-content">
          <div className="dropzone-icon-box">
            <UploadCloud size={38} className="dropzone-icon" />
          </div>
          <div className="dropzone-instructions">
            <p className="dropzone-primary-text">
              <strong>Click to browse</strong> or drag & drop Sentinel-2 raster files here
            </p>
            <p className="dropzone-secondary-text">
              Accepts 4 individual band files (<code>B02</code>, <code>B03</code>, <code>B04</code>, <code>B08</code>)
              or a single 4-band multi-spectral GeoTIFF / JP2 (~10m GSD)
            </p>
          </div>
          <div className="dropzone-supported-tags">
            <span className="spec-tag">GeoTIFF (.tif)</span>
            <span className="spec-tag">JPEG 2000 (.jp2)</span>
            <span className="spec-tag">10m GSD</span>
            <span className="spec-tag">Level-2A BOA Reflectance</span>
          </div>
        </div>
      </div>

      {/* File List Chips */}
      {selectedFiles.length > 0 && (
        <div className="file-chips-card">
          <div className="file-chips-header">
            <span className="file-chips-title">Staged Imagery Files ({selectedFiles.length})</span>
          </div>
          <div className="file-chips-grid">
            {selectedFiles.map((file, idx) => (
              <div key={idx} className="file-chip">
                <FileText size={14} className="icon-cyan" />
                <span className="file-chip-name" title={file.name}>{file.name}</span>
                <span className="file-chip-size">{(file.size / 1024 / 1024).toFixed(2)} MB</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Validation In Progress */}
      {isValidating && (
        <div className="validation-status-card validating">
          <RotateCw size={18} className="spin icon-cyan" />
          <span>Validating raster spatial alignment, band headers, and Sentinel-2 10m GSD compliance...</span>
        </div>
      )}

      {/* Validation Error Message */}
      {validationError && (
        <div className="validation-status-card error">
          <AlertTriangle size={18} className="icon-red" />
          <div>
            <div style={{ fontWeight: "600", marginBottom: "2px" }}>Raster Validation Rejected</div>
            <div style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>{validationError}</div>
          </div>
        </div>
      )}

      {/* Validation Success Strip & Proceed Action */}
      {validationInfo && !isValidating && (
        <div className="validation-success-strip">
          <div className="validation-success-header">
            <div className="validation-badge-title">
              <CheckCircle2 size={18} className="icon-emerald" />
              <span>SCENE READY & VALIDATED</span>
            </div>
            <div className="validation-band-pills">
              <span className="band-pill b02">B02 Blue</span>
              <span className="band-pill b03">B03 Green</span>
              <span className="band-pill b04">B04 Red</span>
              <span className="band-pill b08">B08 NIR</span>
            </div>
          </div>

          <div className="validation-meta-row">
            <div className="meta-item">
              <span className="meta-label">Format</span>
              <span className="meta-val">{validationInfo.format === "separate_bands" ? "4 Separate Bands" : "Single 4-Band GeoTIFF"}</span>
            </div>
            <div className="meta-item">
              <span className="meta-label">Native Dimensions</span>
              <span className="meta-val">{validationInfo.shape[1]} × {validationInfo.shape[0]} px</span>
            </div>
            <div className="meta-item">
              <span className="meta-label">Spatial Resolution</span>
              <span className="meta-val">{validationInfo.resolution[0]}m GSD</span>
            </div>
            <div className="meta-item">
              <span className="meta-label">Coordinate Reference System</span>
              <span className="meta-val crs-mono">{validationInfo.crs}</span>
            </div>
          </div>

          {/* Grid Partition Ready Card */}
          <div style={{ marginTop: "1rem", padding: "1rem", backgroundColor: "var(--bg-secondary)", borderRadius: "6px", border: "1px solid var(--border-subtle)" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.5rem" }}>
              <span style={{ fontSize: "0.85rem", fontWeight: "600", color: "var(--text-primary)" }}>
                Partitioned into {validationInfo.grid?.total_tiles || 25} Interactive Sub-Tiles (128×128 px)
              </span>
              <span className="source-tag user_input">Ready for Inspection</span>
            </div>
            <p style={{ fontSize: "0.78rem", color: "var(--text-secondary)", marginBottom: "1rem" }}>
              The scene has been automatically partitioned into a 5×5 spatial tile grid matching the primary GeoFUSE workflow.
              Select any regional tile to execute 2.5× super-resolution, epistemic uncertainty quantification, and trust evaluation.
            </p>

            <button
              className="btn-primary"
              style={{ width: "100%", padding: "0.75rem 1rem", fontSize: "0.9rem", display: "flex", alignItems: "center", justifyContent: "center", gap: "0.5rem" }}
              onClick={handleProceedToGrid}
            >
              <Layers size={16} />
              Open Interactive 5×5 Tile Grid & Inspect Scene
              <ArrowRight size={16} />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
