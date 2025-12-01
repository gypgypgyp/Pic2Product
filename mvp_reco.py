# mvp_reco.py
import sqlite3
import os, json, math, csv
from pathlib import Path
from typing import List, Dict, Any, Tuple

# import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

import torch
import torchvision.transforms as T
import open_clip
from ultralytics import YOLO

# ====== 可调参数 ======
MODEL_DET = "yolov8n.pt"   # 轻量，够用
CONF_THRES = 0.25
IOU_THRES = 0.45
TOPK = 1                   # 每个实例返回的 SKU 数
ALPHA_IMG = 0.7            # 图像相似度权重（图像 vs 文本 = 0.7 : 0.3）

CATALOG_CSV = "catalog/catalog.csv"
RUNS_DIR = Path("runs")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "catalog" / "catalog.db"

# 仅为了可视化（不同系统字体可能不一样）
def _get_font(size=14):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except:
        return ImageFont.load_default()

# ====== OpenCLIP 封装 ======
class ClipEncoder:
    def __init__(self, model_name="ViT-B-32", pretrained="openai", device=DEVICE):
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

    @torch.no_grad()
    def encode_image(self, pil_img: Image.Image) -> np.ndarray:
        img = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        feat = self.model.encode_image(img)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.squeeze(0).detach().cpu().numpy()

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        toks = self.tokenizer([text]).to(self.device)
        feat = self.model.encode_text(toks)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.squeeze(0).detach().cpu().numpy()

# ====== 商品库构建 ======
def load_catalog(clip: ClipEncoder, csv_path: str) -> Dict[str, Any]:
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise RuntimeError("CSV 格式错误，缺少表头。")
        required = {"sku_id", "title", "brand", "image_path"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise RuntimeError(f"CSV 缺少必须列: {', '.join(sorted(missing))}")
        rows_csv = list(reader)

    imgs, img_embs, txts, txt_embs, rows = [], [], [], [], []

    for row in tqdm(rows_csv, total=len(rows_csv), desc="Building catalog embeddings"):
        sku = str(row["sku_id"])
        title = str(row["title"])
        brand = str(row["brand"])
        img_path = str(row["image_path"])

        p = Path(csv_path).parent / img_path
        if not p.exists():
            print(f"[WARN] image not found: {p}")
            continue

        pil = Image.open(p).convert("RGB")
        img_emb = clip.encode_image(pil)
        text = f"{brand} {title}".strip()
        txt_emb = clip.encode_text(text)

        imgs.append(str(p))
        img_embs.append(img_emb)
        txts.append(text)
        txt_embs.append(txt_emb)
        rows.append({
            "sku_id": sku,
            "title": title,
            "brand": brand,
            "image_path": str(p),
            "text": text
        })

    if not rows:
        raise RuntimeError("Catalog is empty. 请放入至少一条商品及主图。")

    return {
        "rows": rows,
        "img_embs": np.vstack(img_embs),   # [N, D]
        "txt_embs": np.vstack(txt_embs),   # [N, D]
    }

# ====== 数据库交互 ======
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# ====== 从数据库加载商品库 ======
def load_catalog_from_db(conn):
    rows = conn.execute("""
        SELECT sku_id, title, brand, image_path
        FROM products
        ORDER BY sku_id
    """).fetchall()

    catalog_rows = [
        {
            "sku_id": r["sku_id"],
            "title": r["title"],
            "brand": r["brand"],
            "image_path": r["image_path"],
        }
        for r in rows
    ]
    return catalog_rows

# ====== 保存商品库向量到数据库 ======
def save_embeddings_to_db(conn, catalog_rows, img_embs: np.ndarray):
    dim = img_embs.shape[1]
    data = []
    for row, vec in zip(catalog_rows, img_embs):
        data.append(
            (row["sku_id"], vec.astype("float32").tobytes(), dim)
        )

    conn.executemany("""
        INSERT OR REPLACE INTO embeddings (sku_id, embedding, dim)
        VALUES (?,?,?)
    """, data)
    conn.commit()

# 余弦相似度
def cos_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a: [D], b: [N,D] -> [N]
    return (b @ a) / (np.linalg.norm(b, axis=1) * np.linalg.norm(a) + 1e-8)

# ====== 检测 + 实例裁剪 ======
def detect_instances(det_model: YOLO, image_path: str) -> Tuple[Image.Image, List[Dict[str, Any]]]:
    res = det_model.predict(
        source=image_path, conf=CONF_THRES, iou=IOU_THRES, device=0 if DEVICE=="cuda" else "cpu", verbose=False
    )[0]

    names = det_model.names
    pil = Image.open(image_path).convert("RGB")
    W, H = pil.size

    instances = []
    if res.boxes is not None:
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            cls_id = int(b.cls[0].item())
            conf = float(b.conf[0].item())
            # 安全裁剪
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(W-1, int(x2)), min(H-1, int(y2))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = pil.crop((x1, y1, x2, y2))
            instances.append({
                "bbox": [x1, y1, x2, y2],
                "class": names[cls_id],
                "conf": conf,
                "crop": crop
            })
    return pil, instances

# ====== 可视化与结果保存 ======
def draw_and_save(pil: Image.Image, recos_per_inst: List[Dict[str, Any]], out_path: Path):
    img = pil.copy()
    draw = ImageDraw.Draw(img)
    font = _get_font(14)

    def measure(text: str):
        # Pillow ≥10
        if hasattr(draw, "textbbox"):
            left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
            return right - left, bottom - top
        # Pillow <10 兜底
        if hasattr(font, "getsize"):
            return font.getsize(text)
        # 最后兜底
        return (len(text) * 7, 14)

    for inst in recos_per_inst:
        x1, y1, x2, y2 = inst["bbox"]
        label = f'{inst["class"]} {inst["det_conf"]:.2f} → {inst["top1"]["sku_id"]}'

        draw.rectangle([x1, y1, x2, y2], width=2)

        tw, th = measure(label)
        # 文本背景条
        bg_top = max(0, y1 - th - 4)
        draw.rectangle([x1, bg_top, x1 + tw + 4, bg_top + th + 4], fill=(0, 0, 0))
        draw.text((x1 + 2, bg_top + 2), label, fill=(255, 255, 255), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


# ====== 主流程 ======
def recommend_for_image(image_path: str, out_dir: Path):
    print(f"[INFO] Device: {DEVICE}")
    # 1) 模型加载
    det = YOLO(MODEL_DET)
    clip = ClipEncoder(model_name="ViT-B-32", pretrained="openai", device=DEVICE)

    # 2) 商品库向量
    catalog = load_catalog(clip, CATALOG_CSV)
    img_bank = catalog["img_embs"]
    txt_bank = catalog["txt_embs"]

    # 3) 检测 + 裁剪
    pil, instances = detect_instances(det, image_path)
    if not instances:
        print("[INFO] 没检测到物体，结束。")
        return

    # 4) 每个实例做检索（图→图 + 图→文）
    results = []
    for inst in instances:
        emb = clip.encode_image(inst["crop"])
        s_img = cos_sim(emb, img_bank)           # [N]
        s_txt = cos_sim(emb, txt_bank)           # [N]
        score = ALPHA_IMG * s_img + (1-ALPHA_IMG) * s_txt

        idx = np.argsort(-score)[:TOPK]
        recos = []
        for j in idx:
            row = catalog["rows"][j]
            recos.append({
                "sku_id": row["sku_id"],
                "title": row["title"],
                "brand": row["brand"],
                "image_path": row["image_path"],
                "score": float(score[j])
            })

        results.append({
            "bbox": inst["bbox"],
            "class": inst["class"],
            "det_conf": inst["conf"],
            "topk": recos,
            "top1": recos[0]
        })

    # 5) 保存可视化与 JSON
    img_out = out_dir / f"{Path(image_path).stem}_vis.jpg"
    json_out = out_dir / f"{Path(image_path).stem}_rec.json"
    draw_and_save(pil, results, img_out)
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump({
            "image": image_path,
            "results": results
        }, f, ensure_ascii=False, indent=2)

    # 控制台友好输出
    print(f"\n[OK] visualization: {img_out}")
    print(f"[OK] JSON: {json_out}\n")
    for i, inst in enumerate(results, 1):
        print(f"Instance #{i} [{inst['class']} @ {inst['bbox']}] det_conf={inst['det_conf']:.2f}")
        for k, rec in enumerate(inst["topk"], 1):
            print(f"  {k}. {rec['sku_id']} | {rec['brand']} - {rec['title']} | score={rec['score']:.3f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Social-to-SKU MVP")
    parser.add_argument("image", help="Path to social image")
    parser.add_argument("--out", default="runs", help="Output dir")
    args = parser.parse_args()

    recommend_for_image(args.image, Path(args.out))
