# Pic2Product

Pic2Product 是一个以 FastAPI + YOLOv8 + CLIP 为核心的以图搜商品 Demo：用户上传图片 → 后端检测商品 → 根据图像/文本向量在本地商品库中检索最相似的 SKU → 前端展示检测框与推荐卡片。

## 架构概览

```
Shopper ─→ React 前端 (Vite) ─→ FastAPI (/recommend, /catalog/*)
                                          │
                                          │ YOLOv8 检测 + CLIP 编码
                                          ↓
                              本地 catalog.csv + embeddings/*.npz
```

- 图片与可视化结果存放在 `runs/`。
- 商品主图位于 `catalog/images/`，商品元数据在 `catalog/catalog.csv`。
- 已缓存的图像/文本向量位于 `embeddings/catalog_embeddings.npz` 与 `embeddings/catalog_index.json`。

## 仓库结构

| 目录 / 文件 | 说明 |
| --- | --- |
| `api.py` | FastAPI 入口，封装 /recommend、/catalog/rebuild、/catalog/query。 |
| `mvp_reco.py` | YOLO + CLIP 推理与可视化逻辑，被 API 重用。 |
| `catalog/` | 商品 CSV + 主图。CSV 至少包含 `sku_id,title,brand,image_path`。 |
| `embeddings/` | 构建完成后的向量缓存（可删，重启时会重新写入）。 |
| `runs/` | 上传的用户图片、推理可视化图。 |
| `frontend/` | React + Vite 前端（`npm run dev` → http://localhost:5173）。 |
| `requirements.txt` | 模型与数据处理依赖。FastAPI 依赖见下文。 |

## 环境要求

- Python 3.10+（建议 3.11）
- Node.js 18+（建议 20 LTS）
- 至少 8GB RAM；GPU 可选（自动检测 CUDA）

## 后端（FastAPI）

### 1. 创建虚拟环境并安装依赖

```bash
cd Pic2Product
python -m venv .venv
source .venv/bin/activate  # Windows 使用 .venv\Scripts\activate

python -m pip install -U pip setuptools wheel
pip install -r requirements.txt
pip install "fastapi>=0.110" "uvicorn[standard]>=0.30" python-multipart pydantic>=2
```

> 如需 GPU，可根据平台单独安装匹配版本的 `torch`/`torchvision`。

### 2. 准备商品库与权重

1. `catalog/catalog.csv`：提供 SKU 元数据与图像路径（相对于 catalog 目录）。
2. `catalog/images/<sku_id>.jpg`：与 CSV 中的 `image_path` 对应。
3. `yolov8n.pt`：仓库已自带，也可以替换为更大模型。

### 3. 启动 API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
# 或
python api.py
```

常用环境变量（可在 `.env` 或 shell 中设置）：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `CATALOG_CSV` | `catalog/catalog.csv` | CSV 路径 |
| `EMBEDDINGS_DIR` | `embeddings` | 缓存 npz/json 的目录 |
| `RUNS_DIR` | `runs` | 上传图像与可视化 |
| `STATIC_DIR` | `catalog` | FastAPI 挂载为 `/static` 的根目录 |
| `TOPK` | `3` | /recommend 默认返回的 SKU 数 |

首次运行或 CSV 发生变化时，需要构建向量缓存：

```bash
curl -X POST http://localhost:8000/catalog/rebuild \
     -H "Content-Type: application/json" \
     -d '{"force": true}'
```

API 启动后可通过 `http://localhost:8000/docs` 查看 Swagger。

### 4. API 速览

| Endpoint | 方法 | 说明 |
| --- | --- | --- |
| `/health` | GET | 检查模型与 catalog 是否就绪。 |
| `/catalog/rebuild` | POST | 重建（或加载缓存）向量库，body: `{ "force": bool, "catalog_csv": "path" }`。 |
| `/catalog/query` | POST | 根据 SKU 列表返回元数据。 |
| `/recommend` | POST | 上传图片表单字段 `image`，可附 `topk`、`alpha_img`。返回检测框、Top-K SKU、可视化。 |

示例：上传图片并拿到推荐

```bash
curl -X POST http://localhost:8000/recommend \
  -F "image=@tests/demo.jpg" \
  -F "topk=3" \
  -F "alpha_img=0.7" \
  -F "return_vis=true"
```

返回字段示例：

```json
{
  "image_url": "/runs/uploads/1716524123456_demo.jpg",
  "vis_url": "/runs/1716524123456_demo_vis.jpg",
  "instances": [
    {
      "bbox": [77, 90, 240, 380],
      "class": "bag",
      "top_k": [
        {"sku_id": "SKU001", "title": "...", "brand": "...", "score": 0.87}
      ]
    }
  ]
}
```

通过 FastAPI 自动挂载：

- `/static/...` 指向 `catalog`（商品主图）。
- `/runs/...` 指向上传图片与可视化结果。

## 前端（React + Vite）

前端默认跑在 `http://localhost:5173`，通过 `VITE_API_BASE_URL` 指定后端地址。

### 1. 安装依赖

```bash
cd frontend
npm install
```

### 2. 配置 API 地址

创建 `frontend/.env.local`（或 `.env`）：

```
VITE_API_BASE_URL=http://localhost:8000
```

### 3. 启动/构建

```bash
# 开发热更新
npm run dev

# 构建生产版本
npm run build

# 预览构建结果
npm run preview
```

前端会自动请求 `/recommend`、`/catalog/query` 并将 `/static`、`/runs` 等相对路径转换为带有 `VITE_API_BASE_URL` 前缀的可访问 URL。

## 数据处理流程

1. YOLOv8 检测出图片中的每个实例并裁剪出 ROI。
2. 对裁剪图像编码得到图像向量；并与 catalog 内的 CLIP 文本向量进行融合匹配，得出 Top-K SKU。
3. 将预测结果写入 `runs` 目录（上传原图 + 绘制 bounding box 的可视化图）。
4. 前端通过 `/catalog/query` 拉取补充字段（价格、描述、图片链接），最终渲染卡片。

## 常见问题

- **运行时提示 “Catalog not ready”**：先调用 `/catalog/rebuild`。
- **向量缓存过旧**：删除 `embeddings/*.npz` 与 `embeddings/catalog_index.json`，然后重新构建。
- **图片路径 404**：确保 CSV 中的 `image_path` 指向 `catalog/` 内的文件，或修改 `STATIC_DIR`。
- **前端访问不到静态资源**：确认 `.env.local` 的 `VITE_API_BASE_URL` 与后端端口一致。
