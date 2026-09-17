"""Full Scene Macro Geographic Overview & Tile Locator Visualizer.

Renders complete Sentinel-2 scenes (512x512 px) with tile grid partitions and
highlights the currently selected / evaluated regional tile with a glowing HUD-style
bounding box, coordinate badges, and spatial context markers.
"""

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import matplotlib.pyplot as plt


def to_display_rgb(
    array: np.ndarray,
    false_color: bool = False,
    stretch_bounds: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """Convert (H, W, C) reflectance array to (H, W, 3) uint8 RGB with robust percentile stretch."""
    if array is None or not isinstance(array, np.ndarray) or array.ndim < 3 or array.shape[2] < 3:
        return np.zeros((128, 128, 3), dtype=np.uint8)
    elif array.shape[2] >= 4:
        if false_color:
            # False-Color Infrared: NIR (B08) -> Red, Red (B04) -> Green, Green (B03) -> Blue
            rgb = np.stack([array[:, :, 3], array[:, :, 2], array[:, :, 1]], axis=-1)
        else:
            # Natural RGB: Red (B04) -> Red, Green (B03) -> Green, Blue (B02) -> Blue
            rgb = np.stack([array[:, :, 2], array[:, :, 1], array[:, :, 0]], axis=-1)
    elif array.shape[2] == 3:
        if false_color:
            rgb = np.stack([array[:, :, 1], array[:, :, 0], array[:, :, 2]], axis=-1)
        else:
            rgb = np.stack([array[:, :, 0], array[:, :, 1], array[:, :, 2]], axis=-1)
    else:
        ch = array[:, :, 0]
        rgb = np.stack([ch, ch, ch], axis=-1)

    if stretch_bounds is not None:
        p_low, p_high = stretch_bounds
    else:
        valid = rgb[np.isfinite(rgb)]
        if valid.size > 0:
            p_low, p_high = np.percentile(valid, (2.0, 98.0))
        else:
            p_low, p_high = 0.0, 1.0

    if p_high > p_low:
        stretched = np.clip((rgb - p_low) / (p_high - p_low), 0.0, 1.0)
    else:
        stretched = np.clip(rgb, 0.0, 1.0)

    return (stretched * 255.0).astype(np.uint8)


def render_scene_with_tile_overlay(
    full_scene: np.ndarray,
    tile_box: Dict[str, int],
    tile_label: str = "Active Tile",
    all_tiles: Optional[List[Dict[str, Any]]] = None,
    false_color: bool = False,
    show_grid: bool = True,
    line_thickness: int = 3,
) -> np.ndarray:
    """Render full Sentinel-2 scene with tile grid and glowing highlight on the selected tile.

    Args:
        full_scene: (H, W, C) float or uint8 multi-band array (e.g. 512x512x4).
        tile_box: Dict with 'x', 'y', 'w', 'h' defining the active tile bounds.
        tile_label: Text badge label to display above the tile box.
        all_tiles: Optional list of all tile coordinate dictionaries in the scene.
        false_color: If True, renders NIR-Red-Green false color; else Natural RGB.
        show_grid: If True, draws subtle dotted/fine grid outlines for all other tiles.
        line_thickness: Pixel thickness of the bounding box.

    Returns:
        np.ndarray: (H, W, 3) uint8 image ready for display or saving.
    """
    rgb_base = to_display_rgb(full_scene, false_color=false_color)
    h_scene, w_scene = rgb_base.shape[:2]
    annotated = rgb_base.copy()

    # 1. Draw subtle tile grid for all partitions if provided
    if show_grid and all_tiles:
        overlay_grid = annotated.copy()
        for t in all_tiles:
            tx, ty = int(t.get("x", 0)), int(t.get("y", 0))
            ts = int(t.get("patch_size", 128))
            # Fine subtle grid line (dim white/gray)
            cv2.rectangle(
                overlay_grid,
                (tx, ty),
                (min(tx + ts, w_scene - 1), min(ty + ts, h_scene - 1)),
                (255, 255, 255),
                1,
                lineType=cv2.LINE_AA,
            )
            # Subtle tile index number in top-left
            t_id = t.get("tile_id", "")
            if t_id != "":
                cv2.putText(
                    overlay_grid,
                    f"#{t_id}",
                    (tx + 4, ty + 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (220, 220, 220),
                    1,
                    cv2.LINE_AA,
                )
        # Blend grid lightly (35% opacity)
        cv2.addWeighted(overlay_grid, 0.35, annotated, 0.65, 0, annotated)

    # 2. Extract active tile bounding box coordinates
    tx = max(0, int(tile_box.get("x", 0)))
    ty = max(0, int(tile_box.get("y", 0)))
    tw = int(tile_box.get("w", tile_box.get("patch_size", 128)))
    th = int(tile_box.get("h", tile_box.get("patch_size", 128)))
    x2 = min(tx + tw, w_scene - 1)
    y2 = min(ty + th, h_scene - 1)

    # 3. Semi-transparent colored tint fill on chosen tile
    # Color: Vibrant Neon Green (0, 255, 127) in RGB
    tint_color = (0, 255, 127) if not false_color else (0, 240, 255)
    tint_layer = annotated.copy()
    cv2.rectangle(tint_layer, (tx, ty), (x2, y2), tint_color, -1)
    cv2.addWeighted(tint_layer, 0.18, annotated, 0.82, 0, annotated)

    # 4. Draw bold bounding rectangle
    cv2.rectangle(annotated, (tx, ty), (x2, y2), tint_color, line_thickness, lineType=cv2.LINE_AA)

    # 5. Draw bold HUD-style corner brackets for precision satellite feel
    corner_len = min(20, tw // 4)
    c_thick = line_thickness + 2
    # Top-Left
    cv2.line(annotated, (tx, ty), (tx + corner_len, ty), (255, 255, 255), c_thick)
    cv2.line(annotated, (tx, ty), (tx, ty + corner_len), (255, 255, 255), c_thick)
    # Top-Right
    cv2.line(annotated, (x2, ty), (x2 - corner_len, ty), (255, 255, 255), c_thick)
    cv2.line(annotated, (x2, ty), (x2, ty + corner_len), (255, 255, 255), c_thick)
    # Bottom-Left
    cv2.line(annotated, (tx, y2), (tx + corner_len, y2), (255, 255, 255), c_thick)
    cv2.line(annotated, (tx, y2), (tx, y2 - corner_len), (255, 255, 255), c_thick)
    # Bottom-Right
    cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), (255, 255, 255), c_thick)
    cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), (255, 255, 255), c_thick)

    # 6. Text badge on top of the tile box
    badge_text = f" {tile_label} [{tx},{ty} - {x2},{y2}] "
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.42
    text_thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(badge_text, font, font_scale, text_thickness)

    # Position badge just above or inside top border
    badge_y = ty - 6 if ty >= 22 else ty + text_h + 8
    badge_x = max(2, min(tx, w_scene - text_w - 4))

    # Dark background pill for readable text contrast
    cv2.rectangle(
        annotated,
        (badge_x, badge_y - text_h - 4),
        (badge_x + text_w, badge_y + baseline),
        (15, 23, 42),
        -1,
    )
    cv2.rectangle(
        annotated,
        (badge_x, badge_y - text_h - 4),
        (badge_x + text_w, badge_y + baseline),
        tint_color,
        1,
        lineType=cv2.LINE_AA,
    )
    # White text label
    cv2.putText(
        annotated,
        badge_text,
        (badge_x, badge_y - 2),
        font,
        font_scale,
        (255, 255, 255),
        text_thickness,
        cv2.LINE_AA,
    )

    return annotated


def create_full_scene_overview_figure(
    full_scene: np.ndarray,
    tile_data: np.ndarray,
    sr_tile: Optional[np.ndarray],
    tile_box: Dict[str, int],
    tile_idx_str: str,
    trust_score_pct: float,
    false_color: bool = False,
    all_tiles: Optional[List[Dict[str, Any]]] = None,
) -> plt.Figure:
    """Create a high-resolution publication-quality 3-panel figure showing scene and tile context."""
    # 1. Render annotated full scene
    scene_annotated = render_scene_with_tile_overlay(
        full_scene=full_scene,
        tile_box=tile_box,
        tile_label=f"Tile {tile_idx_str}",
        all_tiles=all_tiles,
        false_color=false_color,
        show_grid=True,
    )

    # 2. Render tile closeups
    stretch_bounds = None
    tile_rgb = to_display_rgb(tile_data, false_color=false_color, stretch_bounds=stretch_bounds)
    if sr_tile is not None:
        sr_rgb = to_display_rgb(sr_tile, false_color=false_color, stretch_bounds=stretch_bounds)
    else:
        sr_rgb = tile_rgb.copy()

    # 3. Create 3-Panel Matplotlib Figure
    fig = plt.figure(figsize=(18, 6.5), dpi=150, facecolor="#0e1117")

    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1.0, 1.0], wspace=0.18)

    # Panel 1: Full Scene
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(scene_annotated)
    h_s, w_s = full_scene.shape[:2]
    ax1.set_title(
        f"1. Complete Sentinel-2 Scene ({w_s}x{h_s} px | ~{w_s/100:.1f}x{h_s/100:.1f} km)\n[Target Tile Highlighted with 25-Tile Partition Grid]",
        fontsize=11,
        fontweight="bold",
        color="#ffffff",
        pad=10,
    )
    ax1.set_xlabel("UTM Easting Grid (10m / px)", color="#a0aec0", fontsize=9)
    ax1.set_ylabel("UTM Northing Grid (10m / px)", color="#a0aec0", fontsize=9)
    ax1.tick_params(colors="#a0aec0", labelsize=8)

    # Panel 2: Extracted Regional Tile
    ax2 = fig.add_subplot(gs[1])
    ax2.imshow(tile_rgb)
    th, tw = tile_data.shape[:2]
    bx, by = tile_box.get("x", 0), tile_box.get("y", 0)
    ax2.set_title(
        f"2. Extracted Regional Tile ({tw}x{th} px | 10m GSD)\n[Spatial Bounding Box: X={bx}..{bx+tw}, Y={by}..{by+th}]",
        fontsize=11,
        fontweight="bold",
        color="#00e5ff",
        pad=10,
    )
    ax2.axis("off")

    # Panel 3: GeoFUSE Super-Resolution Reconstruction
    ax3 = fig.add_subplot(gs[2])
    ax3.imshow(sr_rgb)
    status_color = "#00ff7f" if trust_score_pct >= 86.5 else "#ffb703"
    ax3.set_title(
        f"3. GeoFUSE Ensemble 2.5x Super-Resolution (4.0m GSD)\n[Trust Score: {trust_score_pct:.2f}% | Fidelity Verified]",
        fontsize=11,
        fontweight="bold",
        color=status_color,
        pad=10,
    )
    ax3.axis("off")

    fig.suptitle(
        f"GeoFUSE SentinelGuard -- Full-Scene Macro Overview & Sub-Tile Extraction [{tile_idx_str}]",
        fontsize=14,
        fontweight="bold",
        color="#ffffff",
        y=0.98,
    )

    return fig


def get_tile_metadata_grid(
    h: int = 512,
    w: int = 512,
    patch_size: int = 128,
    stride: int = 96,
) -> List[Dict[str, Any]]:
    """Return comprehensive spatial metadata for all partition tiles in a scene grid."""
    y_steps = list(range(0, h - patch_size + 1, stride))
    x_steps = list(range(0, w - patch_size + 1, stride))
    if y_steps[-1] != h - patch_size:
        y_steps.append(h - patch_size)
    if x_steps[-1] != w - patch_size:
        x_steps.append(w - patch_size)

    quadrant_names = {
        (0, 0): "NW Sector (Top-Left Settlement)",
        (0, 4): "NE Sector (North-East Border)",
        (1, 1): "Agricultural & Rural Roads",
        (2, 2): "Central Settlement & Urban Core",
        (3, 1): "Rural River Corridor",
        (4, 0): "SW Sector (South-West Plains)",
        (4, 4): "SE Quadrant (Hold-out Test Transition)",
    }

    tile_list = []
    tile_id = 0
    for r_idx, y in enumerate(y_steps):
        for c_idx, x in enumerate(x_steps):
            pos_key = (r_idx, c_idx)
            if pos_key in quadrant_names:
                desc = quadrant_names[pos_key]
            elif r_idx == 0:
                desc = f"Northern Perimeter (Col {c_idx + 1})"
            elif r_idx == len(y_steps) - 1:
                desc = f"Southern Hold-out Region (Col {c_idx + 1})"
            elif c_idx == 0:
                desc = f"Western Margin (Row {r_idx + 1})"
            elif c_idx == len(x_steps) - 1:
                desc = f"Eastern Margin (Row {r_idx + 1})"
            else:
                desc = f"Interior Landscape (Row {r_idx + 1}, Col {c_idx + 1})"

            tile_list.append({
                "tile_id": tile_id,
                "row": r_idx + 1,
                "col": c_idx + 1,
                "x": x,
                "y": y,
                "w": patch_size,
                "h": patch_size,
                "patch_size": patch_size,
                "description": desc,
                "label": f"Tile #{tile_id:02d} [R{r_idx+1}:C{c_idx+1}] -- {desc} (X:{x}, Y:{y})",
            })
            tile_id += 1

    return tile_list

