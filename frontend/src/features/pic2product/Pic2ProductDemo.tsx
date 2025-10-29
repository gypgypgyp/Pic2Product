import React, { useEffect, useMemo, useRef, useState } from "react";

export default function Pic2ProductDemo() {
  // UI state
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // API data
  const [instances, setInstances] = useState<DetectInstance[]>([]);
  const [skuMap, setSkuMap] = useState<Record<string, ProductItem>>({});

  // 参数（仅前端使用，不改变后端 API）
  const [topK, setTopK] = useState<number>(5); // 只用于前端展示截断
  const [scoreThreshold, setScoreThreshold] = useState<number>(0.0); // 只用于前端过滤

  // image measure
  const imgRef = useRef<HTMLImageElement | null>(null);
  const [imgNatural, setImgNatural] = useState<{ w: number; h: number }>({ w: 1, h: 1 });
  const [imgDisplay, setImgDisplay] = useState<{ w: number; h: number }>({ w: 1, h: 1 });

  // ========= Types =========
  type TopKItem = { sku_id: string; score: number; title?: string; brand?: string; link?: string };
  type DetectInstance = {
    bbox: [number, number, number, number]; // [x1,y1,x2,y2] 基于原图像素
    class: string;
    top_k: TopKItem[];
  };
  type RecommendResponse = {
    instances: DetectInstance[];
  };
  type ProductItem = {
    sku_id: string;
    title: string;
    brand?: string;
    price?: number;
    image_url?: string;
  };
  type CatalogQueryResponse = {
    items: ProductItem[];
    missing?: string[];
  };

  // ========= Effects =========
  useEffect(() => {
    if (!file) return;
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  // 当图片加载后记录原始与显示尺寸
  const onImgLoad = () => {
    const img = imgRef.current;
    if (!img) return;
    setImgNatural({ w: img.naturalWidth, h: img.naturalHeight });
    setImgDisplay({ w: img.clientWidth, h: img.clientHeight });
  };

  // 计算缩放比（原图 → 显示图）
  const scale = useMemo(
    () => ({ x: imgDisplay.w / imgNatural.w, y: imgDisplay.h / imgNatural.h }),
    [imgDisplay.w, imgDisplay.h, imgNatural.w, imgNatural.h]
  );

  // ========= Handlers =========
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setError(null);
    const f = e.target.files?.[0] || null;
    if (!f) return;
    // 简单限制：图片 ≤ 5MB
    if (f.size > 5 * 1024 * 1024) {
      setError("图片太大了，建议≤5MB");
      return;
    }
    setFile(f);
  };

  const handleRecommend = async () => {
    if (!file) {
      setError("请先选择一张图片");
      return;
    }

    setLoading(true);
    setError(null);
    setInstances([]);
    setSkuMap({});

    try {
      const form = new FormData();
      form.append("image", file); // 按照API契约：multipart/form-data { image }

      const res = await fetch("/recommend", { method: "POST", body: form });
      if (!res.ok) throw new Error(`Request failed: ${res.status}`);

      const data: RecommendResponse = await res.json();

      // 安全兜底
      if (!data.instances || data.instances.length === 0) {
        setInstances([]);
        setError("没有检测到商品");
        setLoading(false);
        return;
      }

      // 前端按 scoreThreshold 过滤 & topK 截断（不改变后端）
      const filtered = data.instances.map((ins) => ({
        ...ins,
        top_k: ins.top_k.filter((t) => t.score >= scoreThreshold).slice(0, topK),
      }));

      setInstances(filtered);

      // 收集所有 sku_id 去查详情（用于补齐 price/image_url）
      const skuSet = new Set<string>();
      filtered.forEach((ins) => ins.top_k?.forEach((t) => skuSet.add(t.sku_id)));
      if (skuSet.size > 0) {
        const queryRes = await fetch("/catalog/query", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ sku_ids: Array.from(skuSet) }), // 严格按 API契约：Content-Type: application/json
        });
        if (!queryRes.ok) throw new Error(`查询商品信息失败：${queryRes.status}`);
        const q: CatalogQueryResponse = await queryRes.json();
        const map: Record<string, ProductItem> = {};
        q.items?.forEach((it) => (map[it.sku_id] = it));
        setSkuMap(map);
        // q.missing 如有需要可用于 UI 提示
      }
    } catch (err: any) {
      setError(err?.message || "发生未知错误");
    } finally {
      setLoading(false);
    }
  };

  // ========= UI =========
  return (
    <div className="min-h-screen bg-gray-50 text-gray-900">
      <header className="sticky top-0 z-10 bg-white/80 backdrop-blur border-b border-gray-200">
        <div className="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between">
          <h1 className="text-xl font-semibold">Pic2Product Demo</h1>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-6 grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* 左侧：上传 + 图片预览 */}
        <section className="lg:col-span-3">
          <div className="bg-white rounded-2xl shadow p-4">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-lg font-semibold">1) Upload an image</h2>
                <p className="text-sm text-gray-500">Upload an image, click "Start Recommendation"</p>
              </div>

              <div className="flex items-center gap-3 text-sm">
                <label className="inline-flex items-center gap-2">
                  <span className="text-gray-600">top_k</span>
                  <input type="number" value={topK} min={1} max={20} onChange={(e)=>setTopK(parseInt(e.target.value||"5"))} className="w-16 border rounded px-2 py-1" />
                </label>
                <label className="inline-flex items-center gap-2">
                  <span className="text-gray-600">threshold</span>
                  <input type="number" step="0.01" value={scoreThreshold} onChange={(e)=>setScoreThreshold(parseFloat(e.target.value||"0"))} className="w-20 border rounded px-2 py-1" />
                </label>
              </div>
            </div>

            <div className="mt-3 flex items-center gap-3">
              <input type="file" accept="image/*" onChange={handleFileChange} className="block" />
              <button onClick={handleRecommend} disabled={loading || !file} className="px-4 py-2 rounded-xl bg-black text-white disabled:opacity-50">
                {loading ? "Analyzing..." : "Start Recommendation"}
              </button>
            </div>

            {error && <div className="mt-3 p-3 rounded-xl bg-red-50 text-red-700 text-sm">{error}</div>}

            <div className="mt-4">
              {previewUrl ? (
                <div className="relative w-full">
                  <img
                    ref={imgRef}
                    src={previewUrl}
                    onLoad={onImgLoad}
                    alt="preview"
                    className="max-h-[60vh] w-auto rounded-xl border border-gray-200"
                  />

                  {/* 叠加 bbox */}
                  {instances.map((ins, idx) => {
                    const [x1, y1, x2, y2] = ins.bbox;
                    const left = x1 * scale.x;
                    const top = y1 * scale.y;
                    const width = (x2 - x1) * scale.x;
                    const height = (y2 - y1) * scale.y;
                    return (
                      <div key={idx} className="absolute" style={{ left, top, width, height }}>
                        <div className="absolute inset-0 rounded-xl border-2 border-emerald-500/80"></div>
                        <div className="absolute -top-6 left-0 px-2 py-0.5 rounded-md bg-emerald-500 text-white text-xs shadow">
                          {ins.class}
                        </div>
                      </div>
                    );
                  })}
                </div>
              ) : (
                <div className="h-48 grid place-items-center text-gray-400 border-2 border-dashed rounded-2xl">
                  选择一张图片预览
                </div>
              )}
            </div>
          </div>
        </section>

        {/* 右侧：推荐结果列表 */}
        <section className="lg:col-span-2">
          <div className="bg-white rounded-2xl shadow p-4 h-full">
            <h2 className="text-lg font-semibold">2) Recommendation results</h2>
            <p className="text-sm text-gray-500">Show similar products for each detected region</p>

            {instances.length === 0 && (
              <div className="mt-6 text-sm text-gray-500">No results</div>
            )}

            <div className="mt-4 space-y-6">
              {instances.map((ins, idx) => (
                <div key={idx} className="">
                  <div className="flex items-center gap-2">
                    <span className="px-2 py-0.5 text-xs rounded-md bg-gray-100 text-gray-700">#{idx + 1}</span>
                    <span className="text-sm text-gray-600">{ins.class}</span>
                  </div>

                  <div className="mt-2 grid grid-cols-1 sm:grid-cols-2 gap-3">
                    {ins.top_k?.map((t, j) => {
                      const prod = skuMap[t.sku_id];
                      const title = prod?.title || t.title || t.sku_id;
                      const brand = prod?.brand || t.brand || "";
                      const price = prod?.price;
                      const imageUrl = prod?.image_url; // /recommend 没有图片与价格，按规范从 /catalog/query 获取

                      return (
                        <article key={t.sku_id + j} className="border rounded-xl overflow-hidden hover:shadow">
                          {imageUrl ? (
                            <img src={imageUrl} alt={title} className="w-full h-36 object-cover" />
                          ) : (
                            <div className="w-full h-36 grid place-items-center text-gray-400 bg-gray-50">无图片</div>
                          )}
                          <div className="p-3 space-y-1">
                            <div className="text-sm font-medium line-clamp-2">{title}</div>
                            <div className="text-xs text-gray-500">{brand}</div>
                            <div className="text-sm text-gray-700">
                              {typeof price === "number" ? (
                                <>${price}</>
                              ) : (
                                <span className="text-xs text-gray-400">价格暂无</span>
                              )}
                            </div>
                            <div className="text-xs text-gray-500">相似度：{t.score.toFixed(2)}</div>
                            {t.link && (
                              <a href={t.link} target="_blank" className="text-xs text-blue-600 hover:underline">查看详情</a>
                            )}
                          </div>
                        </article>
                      );
                    }).slice(0, topK)}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>
      </main>

      <footer className="max-w-6xl mx-auto px-4 pb-12 text-xs text-gray-500">
        <div className="mt-6">Tip: If you always get no results, please confirm that the backend has called <code>/catalog/rebuild</code> to generate embedding, and ensure that the directory has product images and metadata.</div>
      </footer>
    </div>
  );
}
