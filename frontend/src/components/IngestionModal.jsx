import React, { useState, useRef } from "react";
import {
  UploadCloud,
  FileCheck,
  AlertTriangle,
  X,
  CheckCircle2,
  Layers,
  MapPin,
  Cpu,
  ArrowRight
} from "lucide-react";
import { validateCustomScene } from "../services/api";

export default function IngestionModal({ isOpen, onClose, onSceneValidated }) {
  const [dragOver, setDragOver] = useState(false);
  const [stagedFiles, setStagedFiles] = useState([]);
  const [validating, setValidating] = useState(false);
  const [valResult, setValResult] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  if (!isOpen) return null;

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragOver(true);
    } else if (e.type === "dragleave") {
      setDragOver(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFiles(Array.from(e.dataTransfer.files));
    }
  };

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFiles(Array.from(e.target.files));
    }
  };

  const handleFiles = (files) => {
    const tifFiles = files.filter(
      (f) => f.name.toLowerCase().endsWith(".tif") || f.name.toLowerCase().endsWith(".tiff")
    );
    if (tifFiles.length === 0) {
      setError("Only GeoTIFF files (.tif, .tiff) are accepted.");
      return;
    }
    setError(null);
    setValResult(null);
    setStagedFiles(tifFiles);
  };

  const handleValidateAndStage = async () => {
    if (stagedFiles.length === 0) return;
    setValidating(true);
    setError(null);

    try {
      const res = await validateCustomScene(stagedFiles);
      setValResult(res);
      if (res.is_valid) {
        if (onSceneValidated) {
          onSceneValidated(res);
        }
        setTimeout(() => {
          onClose();
        }, 1200);
      } else {
        setError(res.message || "GeoTIFF validation failed.");
      }
    } catch (err) {
      setError(`Validation failed: ${err.message}`);
    } finally {
      setValidating(false);
    }
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="ingestion-modal-card" onClick={(e) => e.stopPropagation()}>
        {/* Modal Header */}
        <div className="modal-header">
          <div className="modal-title-box">
            <UploadCloud size={20} className="text-cyan-400" />
            <div>
              <h3 className="modal-title">Sentinel-2 L2A GeoTIFF Ingestion Modal</h3>
              <span className="modal-sub font-mono">MISSION RACK // EPSG:32643 • 10M GSD VALIDATOR</span>
            </div>
          </div>
          <button className="modal-close-btn" onClick={onClose} type="button">
            <X size={18} />
          </button>
        </div>

        {/* Modal Body */}
        <div className="modal-body">
          {/* Dropzone */}
          <div
            className={`ingestion-dropzone ${dragOver ? "dragover" : ""}`}
            onDragEnter={handleDrag}
            onDragOver={handleDrag}
            onDragLeave={handleDrag}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept=".tif,.tiff"
              onChange={handleFileInput}
              style={{ display: "none" }}
            />
            <div className="dropzone-icon">
              <UploadCloud size={36} className="text-cyan-400" />
            </div>
            <div className="dropzone-text">
              <span className="dropzone-prompt">Drop Sentinel-2 GeoTIFF Imagery Here</span>
              <span className="dropzone-sub">
                Supports 4-band composite (B02-B03-B04-B08) or individual band files
              </span>
            </div>
            <button className="select-files-btn" type="button">
              BROWSE LOCAL STORAGE
            </button>
          </div>

          {/* Staged Files List */}
          {stagedFiles.length > 0 && (
            <div className="staged-files-box">
              <div className="staged-header">
                <span className="staged-title">STAGED FILES ({stagedFiles.length})</span>
                <span className="staged-size font-mono">
                  {(stagedFiles.reduce((acc, f) => acc + f.size, 0) / (1024 * 1024)).toFixed(2)} MB
                </span>
              </div>
              <div className="staged-list">
                {stagedFiles.map((file, idx) => (
                  <div key={idx} className="staged-item">
                    <FileCheck size={14} className="text-cyan-400" />
                    <span className="staged-filename font-mono">{file.name}</span>
                    <span className="staged-filesize font-mono">
                      {(file.size / (1024 * 1024)).toFixed(2)} MB
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Verification Checklist */}
          <div className="pre-val-checklist">
            <span className="checklist-title">AUTOMATIC VERIFICATION PIPELINE:</span>
            <div className="checklist-grid">
              <div className="checklist-item">
                <CheckCircle2 size={13} className="text-emerald-400" />
                <span>Native 10m Ground Sample Distance Check</span>
              </div>
              <div className="checklist-item">
                <CheckCircle2 size={13} className="text-emerald-400" />
                <span>WGS84 / UTM Coordinate Reference System (EPSG:32643)</span>
              </div>
              <div className="checklist-item">
                <CheckCircle2 size={13} className="text-emerald-400" />
                <span>4-Band Radiometric Channels (B02, B03, B04, B08)</span>
              </div>
              <div className="checklist-item">
                <CheckCircle2 size={13} className="text-emerald-400" />
                <span>Automated 5×5 Tile Grid Partitioning</span>
              </div>
            </div>
          </div>

          {/* Status / Feedback message */}
          {error && (
            <div className="ingestion-error-msg">
              <AlertTriangle size={15} />
              <span>{error}</span>
            </div>
          )}

          {valResult && valResult.is_valid && (
            <div className="ingestion-success-msg">
              <CheckCircle2 size={15} />
              <span>GeoTIFF imagery validated successfully! Loading 5×5 tile grid into workspace...</span>
            </div>
          )}
        </div>

        {/* Modal Footer */}
        <div className="modal-footer">
          <button className="btn-cancel" onClick={onClose} type="button">
            CANCEL
          </button>
          <button
            className="btn-validate"
            onClick={handleValidateAndStage}
            disabled={stagedFiles.length === 0 || validating}
            type="button"
          >
            {validating ? (
              <span>VERIFYING RASTER...</span>
            ) : (
              <>
                <span>VALIDATE & PARTITION 5×5 GRID</span>
                <ArrowRight size={15} />
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
