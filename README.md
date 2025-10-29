# Pic2Product

## architecture

Shopper (Web/Mobile)
        │ uploads image
        ▼
Frontend UI
        │ POST /recommend
        ▼
Backend API (FastAPI)
        │ 1) YOLOv8 detect objects
        │ 2) Crop product regions
        │ 3) CLIP embeddings
        │ 4) Vector search (catalog_embeddings.npy)
        ▼
Retrieve Top-K sku_id
        │
        ▼
POST /catalog/query
        │ returns title, brand, price, image_url
        ▼
Frontend Renders:
- bounding boxes on image
- product cards with links


## Data flow
YOLOv8 → crop → CLIP embedding → cosine top-K → 得到 sku_id 列表 → POST /catalog/query → 返回详情 → 前端展示


## Data Schema
local storage: pictures, embeddings. 图片访问快 & 训练/embedding 重建更方便;Retrieval 阶段直接 load .npy → 无数据库延迟
MongoDB: catelog (metadata)

商品图像
本地: /static/catalog/images/<sku_id>.jpg

向量 embeddings
本地: embeddings/catalog_embeddings.npy 和 catalog_index.json

catalog_embeddings.npy
存放所有商品的向量（和catalog_index.json中一一对应）：
```bash
索引号（index） → 向量值
0 → [0.12, -0.88, 0.56, ...]
1 → [0.02, -0.44, 0.91, ...]
2 → [-0.31,  0.67, 0.12, ...]
...
```

catalog_index.json
```bash
{
  "embedding_dim": 512,
  "ids": [
    "SKU001",
    "SKU002",
    "SKU003"
  ]
}
```


商品元数据（title/brand/price等）
MongoDB products 集合
```bash
MongoDB Schema（products collection）
{
  "_id": "SKU001",                // 直接用 sku_id 当主键，省索引
  "sku_id": "SKU001",             // 冗余一份，方便查询
  "title": "White low-top leather sneakers",
  "brand": "Nike",
  "category": "Shoes",
  "price": 129.00,
  "image_path": "static/catalog/images/SKU001.jpg",  // 本地路径
  "attributes": {                 // 可选字段，方便今后增强匹配能力
    "color": "white",
    "material": "leather",
    "gender": "unisex"
  },
  "updated_at": "2025-03-02T10:21:00Z"
}
```


## APIs

POST /recommend
Purpose：给一张图片 → 返回检测到的商品的boundingbox + 每个部位对应推荐的 SKU 列表，包括score(cosine similarity)等数据。
前端可以结合bbox框出原图中出现了哪些商品，并罗列相关商品的sku等信息
Request Body: multipart/form-data { image }
Response
```bash
{
  "instances": [
    {
      "bbox": [x1,y1,x2,y2],
      "class": "shoe",
      "top_k": [
        {"sku_id": "SKU001", "score": 0.87, "title": "...", "brand": "...", "link": "..."}
      ]
    }
  ]
}
```


POST /catalog/rebuild
Purpose：根据 catalog.csv 加载或重新计算所有 SKU 的图像/文本 embedding（保存在本地）
前端可以设置一个按键用来重新生成embeddings
Request(不传 force → 加载已有缓存（快）; 传 force=true → 重新跑一次 embedding（慢）)
```bash
{
  "force": true
}
```
Response
```bash
{
  "status": "success",
  "catalog_size": 320,
  "embedding_dim": 512,
  "message": "Catalog embeddings rebuilt and stored in memory."
}
```


POST /catalog/query
前端或推荐系统 发送一批 sku_id，后台从数据库 / CSV / 缓存中 返回对应商品信息和图片路径。（考虑到buget，商品图片可能还是保存于本地）
Request：Content-Type: application/json
Body:
```bash
{
  "sku_ids": ["SKU001", "SKU003", "SKU017"]
}
```
Response
```bash
{
  "items": [
    {
      "sku_id": "SKU001",
      "title": "White low-top leather sneakers",
      "brand": "Nike",
      "price": 129.00,
      "image_url": "/static/catalog/images/nike_white_lowtop.jpg"
    },
    {
      "sku_id": "SKU003",
      "title": "Beige tote bag with logo",
      "brand": "Coach",
      "price": 249.50,
      "image_url": "/static/catalog/images/coach_tote_beige.jpg"
    }
  ],
  "missing": ["SKU017"]
}
```


## How to use Backend

### 1) 新建并进入环境（Python 3.10）
```bash
conda create -n pic2product python=3.11
conda activate pic2product
```
### 2) 用 conda-forge 把“底座”装好（版本彼此兼容）
```bash
conda install -y -c conda-forge \
    numpy=1.26.4 pandas=2.2.2 pillow tqdm \
    opencv=4.10.0 \
    protobuf=4.25.* abseil-cpp=20240116.*
```
### 3) 升级 pip 工具
```bash
python -m pip install -U pip setuptools wheel
```
### 4) 装 PyTorch（macOS 上直接 pip）
```bash
python -m pip install "torch==2.3.*" "torchvision==0.18.*"
```
### 5) 业务库（带上 ultralytics 依赖）
```bash
python -m pip install \
  "ultralytics==8.3.30" "open_clip_torch==2.24.0" \
  ftfy regex sentencepiece huggingface_hub \
  matplotlib psutil py-cpuinfo "scipy<1.13" seaborn ultralytics-thop
```
