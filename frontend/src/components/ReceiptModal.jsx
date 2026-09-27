import React, { useState, useEffect } from "react";
import { X, Download, ShieldCheck, Copy, Check } from "lucide-react";
import { getReceipt } from "../services/api";

export default function ReceiptModal({ receiptId, onClose }) {
  const [receiptData, setReceiptData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    let isMounted = true;
    if (receiptId) {
      setLoading(true);
      getReceipt(receiptId)
        .then((data) => {
          if (isMounted) setReceiptData(data);
        })
        .catch((err) => {
          console.error("Failed to load receipt:", err);
        })
        .finally(() => {
          if (isMounted) setLoading(false);
        });
    }
    return () => {
      isMounted = false;
    };
  }, [receiptId]);

  const handleDownload = () => {
    if (!receiptData) return;
    const blob = new Blob([JSON.stringify(receiptData, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${receiptId}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleCopy = () => {
    if (!receiptData) return;
    navigator.clipboard.writeText(JSON.stringify(receiptData, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        {/* Modal Header */}
        <div className="modal-header">
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <ShieldCheck size={18} style={{ color: "var(--accent-cyan)" }} />
            <h2 style={{ fontSize: "1rem", fontWeight: "700", color: "#ffffff" }}>
              Authoritative Machine-Readable Trust Receipt
            </h2>
          </div>
          <button
            onClick={onClose}
            style={{ background: "transparent", border: "none", color: "var(--text-muted)", cursor: "pointer" }}
          >
            <X size={18} />
          </button>
        </div>

        {/* Modal Body */}
        <div className="modal-body">
          {loading ? (
            <div style={{ padding: "3rem", textAlign: "center" }}>
              <div className="spinner" style={{ margin: "0 auto 1rem" }} />
              <p style={{ color: "var(--text-muted)" }}>Retrieving verified provenance certificate...</p>
            </div>
          ) : receiptData ? (
            <div>
              {/* Receipt Summary Header Card */}
              <div
                style={{
                  backgroundColor: "var(--bg-primary)",
                  border: "1px solid var(--border-subtle)",
                  borderRadius: "6px",
                  padding: "1rem",
                  marginBottom: "1rem",
                  display: "grid",
                  gridTemplateColumns: "1fr 1fr",
                  gap: "0.75rem",
                }}
              >
                <div>
                  <span style={{ fontSize: "0.7rem", color: "var(--text-muted)", display: "block" }}>RECEIPT ID</span>
                  <span className="font-mono" style={{ fontSize: "0.85rem", fontWeight: "600", color: "var(--accent-cyan)" }}>
                    {receiptData.receipt_id}
                  </span>
                </div>
                <div>
                  <span style={{ fontSize: "0.7rem", color: "var(--text-muted)", display: "block" }}>ISSUED AT</span>
                  <span className="font-mono" style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>
                    {receiptData.generation_timestamp}
                  </span>
                </div>
                <div>
                  <span style={{ fontSize: "0.7rem", color: "var(--text-muted)", display: "block" }}>MODEL ARCHITECTURE</span>
                  <span style={{ fontSize: "0.8rem", color: "var(--text-primary)" }}>
                    {receiptData.model_provenance?.architecture} ({receiptData.model_provenance?.ensemble_size}-Member Ensemble)
                  </span>
                </div>
                <div>
                  <span style={{ fontSize: "0.7rem", color: "var(--text-muted)", display: "block" }}>RESOLUTION GSD</span>
                  <span style={{ fontSize: "0.8rem", color: "var(--text-primary)" }}>
                    10.0m → 4.0m (2.5× Super-Resolution)
                  </span>
                </div>
              </div>

              {/* Scientific Mandate */}
              <div
                style={{
                  backgroundColor: "rgba(6, 182, 212, 0.05)",
                  border: "1px solid rgba(6, 182, 212, 0.2)",
                  borderRadius: "6px",
                  padding: "0.75rem",
                  marginBottom: "1rem",
                  fontSize: "0.75rem",
                  color: "#a5f3fc",
                }}
              >
                <strong>Scientific Transparency Mandate:</strong> {receiptData.scientific_transparency_mandate?.heuristic_nature}
              </div>

              {/* Raw Machine-Readable Certificate JSON */}
              <div style={{ position: "relative" }}>
                <pre
                  className="font-mono"
                  style={{
                    backgroundColor: "#05080f",
                    border: "1px solid var(--border-subtle)",
                    borderRadius: "6px",
                    padding: "1rem",
                    maxHeight: "260px",
                    overflowY: "auto",
                    fontSize: "0.75rem",
                    color: "#94a3b8",
                  }}
                >
                  {JSON.stringify(receiptData, null, 2)}
                </pre>
                <button
                  onClick={handleCopy}
                  title="Copy Certificate JSON"
                  style={{
                    position: "absolute",
                    top: "8px",
                    right: "8px",
                    background: "var(--bg-secondary)",
                    border: "1px solid var(--border-subtle)",
                    color: "var(--text-secondary)",
                    padding: "4px 8px",
                    borderRadius: "4px",
                    cursor: "pointer",
                    fontSize: "0.7rem",
                    display: "flex",
                    alignItems: "center",
                    gap: "4px",
                  }}
                >
                  {copied ? <Check size={12} color="#10b981" /> : <Copy size={12} />}
                  {copied ? "Copied" : "Copy"}
                </button>
              </div>
            </div>
          ) : (
            <div style={{ color: "var(--accent-red)", padding: "1rem 0" }}>
              Unable to load receipt with ID: {receiptId}
            </div>
          )}
        </div>

        {/* Modal Footer */}
        <div className="modal-footer">
          <button
            onClick={onClose}
            style={{
              padding: "0.5rem 1rem",
              background: "transparent",
              border: "1px solid var(--border-subtle)",
              color: "var(--text-secondary)",
              borderRadius: "6px",
              cursor: "pointer",
              fontSize: "0.8rem",
            }}
          >
            Close
          </button>
          <button
            onClick={handleDownload}
            disabled={!receiptData}
            style={{
              padding: "0.5rem 1rem",
              backgroundColor: "var(--accent-cyan)",
              border: "none",
              color: "#041019",
              fontWeight: "600",
              borderRadius: "6px",
              cursor: "pointer",
              fontSize: "0.8rem",
              display: "flex",
              alignItems: "center",
              gap: "0.4rem",
            }}
          >
            <Download size={14} />
            Download JSON Certificate
          </button>
        </div>
      </div>
    </div>
  );
}
