import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image

# Reuse your MVP pieces
from mvp_reco import (
    ClipEncoder, load_catalog, detect_instances, cos_sim, draw_and_save,
    YOLO, DEVICE
)

# ---------------------------
# Config (env-overridable)
# ---------------------------
CATALOG_CSV = Path(os.getenv("CATALOG_CSV", "catalog/catalog.csv"))
EMB_DIR = Path(os.getenv("EMBEDDINGS_DIR", "embeddings"))
RUNS_DIR = Path(os.getenv("RUNS_DIR", "runs"))
STATIC_DIR = Path(os.getenv("STATIC_DIR", "catalog"))
TOPK_DEFAULT = int(os.getenv("TOPK", "3"))
ALPHA_IMG_DEFAULT = float(os.getenv("ALPHA_IMG", "0.7"))  # image vs text weight

EMB_DIR.mkdir(parents=True, exist_ok=True)
(RUNS_DIR / "uploads").mkdir(parents=True, exist_ok=True)

# ---------------------------
# App + CORS
# ---------------------------
app = FastAPI(title="Pic2Product API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (product images) at /static
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
else:
    # fallback: auto-detect a common alt folder
    for alt in ["catalog", "static"]:
        p = Path(alt)
        if p.exists():
            app.mount("/static", StaticFiles(directory=str(p)), name="static")
            break

# Serve uploads/visualizations at /runs
app.mount("/runs", StaticFiles(directory=str(RUNS_DIR)), name="runs")


# ---------------------------
# In-memory state
# ---------------------------
class State:
    det = None
    clip = None
    catalog_rows: List[Dict[str, Any]] = []
    img_embs: Optional[np.ndarray] = None
    txt_embs: Optional[np.ndarray] = None
    embedding_dim: Optional[int] = None
    catalog_path: Path = CATALOG_CSV

state = State()


# ---------------------------
# Request/Response models
# ---------------------------
class RebuildRequest(BaseModel):
    force: bool = False
    catalog_csv: Optional[str] = None

class QueryRequest(BaseModel):
    sku_ids: List[str] = Field(default_factory=list)

class RecommendResponse(BaseModel):
    image_url: Optional[str]
    vis_url: Optional[str]
    instances: List[Dict[str, Any]]


# ---------------------------
# Cache helpers
# ---------------------------
def _save_cache(rows, img_embs, txt_embs):
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(EMB_DIR / "catalog_embeddings.npz", img_embs=img_embs, txt_embs=txt_embs)
    idx = {
        "embedding_dim": int(img_embs.shape[1]),
        "ids": [r["sku_id"] for r in rows],
        "rows": rows,
        "updated_at": int(time.time())
    }
    with open(EMB_DIR / "catalog_index.json", "w", encoding="utf-8") as f:
        json.dump(idx, f, ensure_ascii=False, indent=2)

def _try_load_cache():
    npz_path = EMB_DIR / "catalog_embeddings.npz"
    idx_path = EMB_DIR / "catalog_index.json"
    if not (npz_path.exists() and idx_path.exists()):
        return False
    try:
        npz = np.load(npz_path)
        with open(idx_path, "r", encoding="utf-8") as f:
            idx = json.load(f)
        state.img_embs = np.array(npz["img_embs"])
        state.txt_embs = np.array(npz["txt_embs"])
        state.catalog_rows = idx["rows"]
        state.embedding_dim = int(idx.get("embedding_dim", state.img_embs.shape[1]))
        return True
    except Exception as e:
        print(f"[WARN] Failed to load cache: {e}")
        return False

def _build_catalog(clip: ClipEncoder, csv_path: Path, force: bool = False):
    if not force and _try_load_cache():
        return {
            "status": "loaded",
            "catalog_size": len(state.catalog_rows),
            "embedding_dim": state.embedding_dim,
            "message": "Loaded embeddings from cache."
        }

    catalog = load_catalog(clip, str(csv_path))
    state.catalog_rows = catalog["rows"]
    state.img_embs = catalog["img_embs"]
    state.txt_embs = catalog["txt_embs"]
    state.embedding_dim = int(state.img_embs.shape[1])

    _save_cache(state.catalog_rows, state.img_embs, state.txt_embs)
    return {
        "status": "success",
        "catalog_size": len(state.catalog_rows),
        "embedding_dim": state.embedding_dim,
        "message": "Catalog embeddings rebuilt and cached."
    }


# ---------------------------
# Lifecycle
# ---------------------------
@app.on_event("startup")
def _startup():
    print(f"[INFO] Device: {DEVICE}")
    state.det = YOLO("yolov8n.pt")
    state.clip = ClipEncoder(model_name="ViT-B-32", pretrained="openai")

    # Load cache or best-effort build
    if not _try_load_cache() and state.catalog_path.exists():
        print(f"[INFO] Building catalog from {state.catalog_path} ...")
        _build_catalog(state.clip, state.catalog_path, force=False)


# ---------------------------
# Routes
# ---------------------------
@app.get("/health")
def health():
    return {
        "ok": True,
        "models_ready": state.det is not None and state.clip is not None,
        "catalog_ready": state.img_embs is not None and state.txt_embs is not None
    }

@app.post("/catalog/rebuild")
def catalog_rebuild(req: RebuildRequest):
    csv_path = Path(req.catalog_csv) if req.catalog_csv else state.catalog_path
    if not csv_path.exists():
        return JSONResponse(status_code=400, content={"error": f"CSV not found: {csv_path}"})
    state.catalog_path = csv_path
    return _build_catalog(state.clip, csv_path, force=req.force)

@app.post("/catalog/query")
def catalog_query(req: QueryRequest):
    if not state.catalog_rows and state.catalog_path.exists():
        _build_catalog(state.clip, state.catalog_path, force=False)

    rows_by_id = {r["sku_id"]: r for r in state.catalog_rows}
    items, missing = [], []
    for sid in req.sku_ids:
        r = rows_by_id.get(str(sid))
        if r:
            items.append({
                "sku_id": r["sku_id"],
                "title": r["title"],
                "brand": r["brand"],
                "image_url": r.get("image_path"),
            })
        else:
            missing.append(str(sid))
    return {"items": items, "missing": missing}

@app.post("/recommend", response_model=RecommendResponse)
async def recommend(
    image: UploadFile = File(...),
    topk: int = Form(TOPK_DEFAULT),
    alpha_img: float = Form(ALPHA_IMG_DEFAULT),
    return_vis: bool = Form(True),
):
    if state.img_embs is None or state.txt_embs is None:
        return JSONResponse(status_code=400, content={"error": "Catalog not ready. Call /catalog/rebuild first."})

    # Save upload
    stamp = int(time.time() * 1000)
    img_path = RUNS_DIR / "uploads" / f"{stamp}_{image.filename}"
    with open(img_path, "wb") as f:
        f.write(await image.read())

    # Detect + crop instances
    pil, instances = detect_instances(state.det, str(img_path))

    # Rank for each instance
    results = []
    for inst in instances:
        emb = state.clip.encode_image(inst["crop"])
        s_img = cos_sim(emb, state.img_embs)
        s_txt = cos_sim(emb, state.txt_embs)
        score = alpha_img * s_img + (1 - alpha_img) * s_txt

        idx = np.argsort(-score)[: int(topk)]
        recos = []
        for j in idx:
            row = state.catalog_rows[j]
            recos.append({
                "sku_id": row["sku_id"],
                "title": row["title"],
                "brand": row["brand"],
                "image_url": row["image_path"],
                "score": float(score[j]),
            })

        results.append({
            "bbox": inst["bbox"],
            "class": inst["class"],
            "det_conf": float(inst["conf"]),
            "top_k": recos,
            "top1": recos[0] if recos else None,
        })

    vis_url = None
    if return_vis and results:
        vis_path = RUNS_DIR / f"{img_path.stem}_vis.jpg"
        draw_and_save(pil, results, vis_path)
        vis_url = f"/runs/{vis_path.name}"

    return {
        "image_url": f"/runs/uploads/{img_path.name}",
        "vis_url": vis_url,
        "instances": results,
    }


# Entrypoint
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
