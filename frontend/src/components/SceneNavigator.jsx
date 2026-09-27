import React from "react";
import { MapPin, Grid, Info, Sparkles } from "lucide-react";

export default function SceneNavigator({
  scenes,
  selectedSceneId,
  onSceneChange,
  gridData,
  selectedTileId,
  onTileSelect,
  loading,
}) {
  const activeTile = gridData?.tiles?.find((t) => t.tile_id === selectedTileId);

  const isCustomScene = selectedSceneId?.startsWith("custom_");

  return (
    <div className="card-panel">
      <div className="panel-header">
        <span className="panel-title">
          <Grid size={15} />
          {isCustomScene ? "User Input Tile Navigator" : "Scene & Tile Navigator"}
        </span>
        {loading && <div className="spinner" style={{ width: "16px", height: "16px", borderWidth: "2px" }} />}
      </div>

      <div className="panel-body">
        {/* Target Scene Header or Dropdown */}
        <label style={{ fontSize: "0.75rem", color: "var(--text-muted)", display: "block", marginBottom: "4px" }}>
          {isCustomScene ? "ACTIVE UPLOADED RASTER" : "TARGET SENTINEL-2 SCENE"}
        </label>

        {isCustomScene ? (
          <div
            style={{
              padding: "0.45rem 0.75rem",
              backgroundColor: "var(--bg-primary)",
              border: "1px solid var(--accent-cyan)",
              borderRadius: "6px",
              marginBottom: "0.75rem",
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
            }}
          >
            <div>
              <div style={{ fontSize: "0.82rem", fontWeight: "600", color: "var(--accent-cyan)" }}>
                {gridData?.scene_name || "Custom Upload Scene"}
              </div>
              <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
                ID: {selectedSceneId}
              </div>
            </div>
            <span className="source-tag user_input" style={{ fontSize: "0.7rem" }}>
              User Input
            </span>
          </div>
        ) : (
          <select
            className="scene-select"
            value={selectedSceneId}
            onChange={(e) => onSceneChange(e.target.value)}
            disabled={loading}
          >
            {scenes.map((s) => (
              <option key={s.scene_id} value={s.scene_id}>
                {s.name}
              </option>
            ))}
          </select>
        )}

        {/* Macro Scene Overview Image with Tile Highlight */}
        {gridData?.macro_preview_url && (
          <div style={{ position: "relative", marginBottom: "0.75rem", borderRadius: "6px", overflow: "hidden", border: "1px solid var(--border-subtle)" }}>
            <img
              src={gridData.macro_preview_url}
              alt="Macro Scene Overview"
              style={{ width: "100%", display: "block" }}
            />
            <div
              style={{
                position: "absolute",
                bottom: "4px",
                right: "6px",
                background: "rgba(0,0,0,0.75)",
                fontSize: "0.65rem",
                padding: "2px 6px",
                borderRadius: "3px",
                color: "var(--text-secondary)",
              }}
            >
              512×512 px (26.2 km²)
            </div>
          </div>
        )}

        {/* 5x5 Partition Tile Grid */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "4px" }}>
          <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
            25 PARTITION TILES (128×128)
          </span>
          <span style={{ fontSize: "0.65rem", color: "var(--accent-amber)", display: "flex", alignItems: "center", gap: "3px" }}>
            <Sparkles size={10} /> Dot = Precomputed
          </span>
        </div>

        <div className="tile-grid">
          {gridData?.tiles?.map((tile) => {
            const isActive = tile.tile_id === selectedTileId;
            return (
              <button
                key={tile.tile_id}
                className={`tile-cell ${isActive ? "active" : ""} ${tile.has_demo_cache ? "demo-available" : ""}`}
                onClick={() => onTileSelect(tile.tile_id)}
                title={tile.label}
                disabled={loading}
              >
                #{tile.tile_id.toString().padStart(2, "0")}
              </button>
            );
          })}
        </div>

        {/* Active Tile Metadata Card */}
        {activeTile && (
          <div style={{ marginTop: "1rem", padding: "0.75rem", backgroundColor: "var(--bg-primary)", borderRadius: "6px", border: "1px solid var(--border-subtle)" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "5px", color: "var(--accent-cyan)", fontSize: "0.8rem", fontWeight: "600", marginBottom: "4px" }}>
              <MapPin size={13} />
              Tile #{activeTile.tile_id.toString().padStart(2, "0")} [R{activeTile.row}:C{activeTile.col}]
            </div>
            <p style={{ fontSize: "0.75rem", color: "var(--text-secondary)", marginBottom: "6px" }}>
              {activeTile.description}
            </p>
            <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
              X: {activeTile.x}..{activeTile.x + activeTile.w} px | Y: {activeTile.y}..{activeTile.y + activeTile.h} px
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
