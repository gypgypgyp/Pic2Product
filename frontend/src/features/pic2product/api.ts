/**
 * Handle inconsistent backend image paths (e.g. catalog/images → /static/images)
 * and always prefix URLs with backend base (http://localhost:8000)
 */
const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");

/**
 * Unify inconsistent paths coming from the backend — especially those that start with catalog/ 
 * so that everything becomes /static/... (matching FastAPI static mount). 
 */
const normalizePath = (p?: string | null) => {
  if (!p) return p ?? undefined;

  // If it's already a full URL (e.g. https://...), just return it
  if (/^https?:\/\//i.test(p)) return p;

  let path = p.trim();

  // Convert old paths like "catalog/images/..." → "/static/images/..."
  if (path.startsWith("catalog/")) {
    path = "/static/" + path.slice("catalog/".length);
  }

  // Ensure every relative path starts with a leading slash
  if (!path.startsWith("/")) path = "/" + path;

  return path;
};

/**
 * Make sure any image or resource path (/runs/..., /static/..., etc.) always gets the backend prefix
 * so the browser requests it from port 8000, not port 5173.
 */
const toAbsoluteUrl = (path?: string | null) => {
  const norm = normalizePath(path);
  if (!norm) return norm ?? undefined;

  // Skip if it’s already a full URL
  if (/^https?:\/\//i.test(norm)) return norm;

  // Add backend base (e.g. http://localhost:8000)
  return API_BASE_URL ? `${API_BASE_URL}${norm}` : norm;
};

/**
 * Ensure even API calls like /recommend or /catalog/query
 * are safely combined with backend base URL, avoiding inconsistent slashes.
 */
const makeUrl = (path: string) => {
  const norm = normalizePath(path) as string;
  return API_BASE_URL ? `${API_BASE_URL}${norm}` : norm;
};

export type TopKItem = {
  sku_id: string;
  score: number;
  title?: string;
  brand?: string;
  link?: string;
  image_url?: string;
};

export type DetectInstance = {
  bbox: [number, number, number, number];
  class: string;
  det_conf?: number;
  top_k: TopKItem[];
  top1?: TopKItem | null;
};

export type RecommendResponse = {
  image_url?: string | null;
  vis_url?: string | null;
  instances: DetectInstance[];
};

export type ProductItem = {
  sku_id: string;
  title: string;
  brand?: string;
  price?: number;
  image_url?: string;
  link?: string;
};

export type CatalogQueryResponse = {
  items: ProductItem[];
  missing?: string[];
};

type RecommendParams = {
  image: File;
  topk: number;
  alphaImg: number;
  returnVis?: boolean;
};

export async function callRecommend(params: RecommendParams): Promise<RecommendResponse> {
  const form = new FormData();
  form.append("image", params.image);
  form.append("topk", String(params.topk));
  form.append("alpha_img", String(params.alphaImg));
  form.append("return_vis", String(params.returnVis ?? true));

  const res = await fetch(makeUrl("/recommend"), {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    throw new Error(`Request failed: ${res.status}`);
  }
  const data: RecommendResponse = await res.json();

  return {
    ...data,
    image_url: toAbsoluteUrl(data.image_url) ?? null,
    vis_url: toAbsoluteUrl(data.vis_url) ?? null,
    instances: (data.instances || []).map((inst) => ({
      ...inst,
      top_k: (inst.top_k || []).map((item) => ({
        ...item,
        image_url: toAbsoluteUrl(item.image_url),
      })),
      top1: inst.top1
        ? {
            ...inst.top1,
            image_url: toAbsoluteUrl(inst.top1.image_url),
          }
        : inst.top1,
    })),
  };
}

export async function catalogQuery(skuIds: string[]): Promise<CatalogQueryResponse> {
  if (skuIds.length === 0) {
    return { items: [], missing: [] };
  }
  const res = await fetch(makeUrl("/catalog/query"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sku_ids: skuIds }),
  });
  if (!res.ok) {
    throw new Error(`Failed to query product information:${res.status}`);
  }
  const data: CatalogQueryResponse = await res.json();
  return {
    items: (data.items || []).map((item) => ({
      ...item,
      image_url: toAbsoluteUrl(item.image_url),
    })),
    missing: data.missing ?? [],
  };
}
