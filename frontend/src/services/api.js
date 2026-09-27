/**
 * GeoFUSE SentinelGuard — Frontend API Client.
 *
 * Dedicated service layer managing all communication with the FastAPI backend.
 * Components must consume this client rather than executing raw fetch() calls.
 */

const API_BASE = "";

async function handleResponse(response) {
  if (!response.ok) {
    let errorDetail = `HTTP ${response.status} ${response.statusText}`;
    try {
      const errJson = await response.json();
      if (errJson && errJson.detail) {
        errorDetail = errJson.detail;
      }
    } catch {
      // Fallback to status text
    }
    throw new Error(errorDetail);
  }
  return response.json();
}

/**
 * Fetch hardware, GPU, and model status telemetry.
 */
export async function getSystemStatus() {
  const resp = await fetch(`${API_BASE}/api/system/status`);
  return handleResponse(resp);
}

/**
 * Fetch all available local Sentinel-2 scenes.
 */
export async function getScenes() {
  const resp = await fetch(`${API_BASE}/api/scenes/list`);
  return handleResponse(resp);
}

/**
 * Fetch 5x5 tile grid partition layout and macro overview image URL.
 */
export async function getSceneGrid(sceneId) {
  const resp = await fetch(`${API_BASE}/api/scenes/${encodeURIComponent(sceneId)}/grid`);
  return handleResponse(resp);
}

/**
 * Execute or fetch cached super-resolution inference on a scene tile.
 */
export async function runTileInference(sceneId, tileId, forceLive = false, signal = null) {
  const resp = await fetch(`${API_BASE}/api/inference/tile`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      scene_id: sceneId,
      tile_id: Number(tileId),
      force_live: Boolean(forceLive),
    }),
    signal: signal || undefined,
  });
  return handleResponse(resp);
}

/**
 * Fetch authoritative JSON Trust Receipt by receipt_id or run_id.
 */
export async function getReceipt(receiptId) {
  const resp = await fetch(`${API_BASE}/api/receipt/${encodeURIComponent(receiptId)}`);
  return handleResponse(resp);
}

/**
 * Fetch historical hold-out evaluation benchmark summary.
 */
export async function getBenchmarkSummary() {
  const resp = await fetch(`${API_BASE}/api/benchmark/summary`);
  return handleResponse(resp);
}

/**
 * Validate user-uploaded Sentinel-2 GeoTIFF files before execution.
 * @param {File[]} files - List of File objects.
 */
export async function validateCustomScene(files) {
  const formData = new FormData();
  for (const file of files) {
    formData.append("files", file);
  }
  const resp = await fetch(`${API_BASE}/api/inference/validate`, {
    method: "POST",
    body: formData,
  });
  return handleResponse(resp);
}

/**
 * Execute end-to-end direct 2.5x learned super-resolution on user-uploaded imagery.
 * @param {Object} options
 * @param {File[]} [options.files] - Uploaded File objects (optional if uploadId provided)
 * @param {string} [options.uploadId] - Pre-validated upload session hash
 * @param {number} [options.cropX] - Optional crop origin X
 * @param {number} [options.cropY] - Optional crop origin Y
 * @param {number} [options.tileSize=128] - Target tile size in pixels
 * @param {boolean} [options.forceLive=false] - Bypass cache and force GPU execution
 */
export async function runCustomInference({ files, uploadId, cropX, cropY, tileSize = 128, forceLive = false } = {}) {
  const formData = new FormData();
  if (files && files.length > 0) {
    for (const file of files) {
      formData.append("files", file);
    }
  }
  if (uploadId) {
    formData.append("upload_id", uploadId);
  }
  if (cropX !== undefined && cropX !== null) {
    formData.append("crop_x", String(cropX));
  }
  if (cropY !== undefined && cropY !== null) {
    formData.append("crop_y", String(cropY));
  }
  formData.append("tile_size", String(tileSize));
  formData.append("force_live", String(forceLive));

  const resp = await fetch(`${API_BASE}/api/inference/custom`, {
    method: "POST",
    body: formData,
  });
  return handleResponse(resp);
}

