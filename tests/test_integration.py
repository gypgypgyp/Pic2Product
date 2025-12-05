import importlib
import io
import shutil
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from fastapi.testclient import TestClient
except ModuleNotFoundError:
    TestClient = None


# Integration coverage:
#   1. YOLO → crop → CLIP → FAISS-style retrieval pipeline.
#   2. Backend FastAPI recommend endpoint writing uploads/visualizations.
#   3. Frontend-style REST workflow (recommend + catalog/query payloads).


def _load_std_unittest():
    project_dir = str(Path(__file__).resolve().parent)
    removed = False
    if project_dir in sys.path:
        sys.path.remove(project_dir)
        removed = True
    try:
        module = importlib.import_module("unittest")
    finally:
        if removed:
            sys.path.insert(0, project_dir)
    return module


unittest = _load_std_unittest()


def _ensure_lightweight_deps():
    if "torch" not in sys.modules:
        def _no_grad():
            def decorator(fn):
                return fn
            return decorator

        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: False),
            no_grad=_no_grad,
        )
        sys.modules["torch"] = fake_torch

    if "torchvision" not in sys.modules:
        fake_tv = types.ModuleType("torchvision")
        fake_tv.transforms = types.SimpleNamespace()
        sys.modules["torchvision"] = fake_tv
        sys.modules["torchvision.transforms"] = fake_tv.transforms

    if "open_clip" not in sys.modules:
        def _not_available(*args, **kwargs):
            raise RuntimeError("open_clip not available in tests.")

        fake_clip = types.SimpleNamespace(
            create_model_and_transforms=_not_available,
            get_tokenizer=_not_available,
        )
        sys.modules["open_clip"] = fake_clip

    if "ultralytics" not in sys.modules:
        class _DummyYOLO:
            def __init__(self, *args, **kwargs):
                self.names = {}

            def predict(self, *args, **kwargs):
                raise RuntimeError("YOLO predict unavailable in tests.")

        fake_ultra = types.SimpleNamespace(YOLO=_DummyYOLO)
        sys.modules["ultralytics"] = fake_ultra


_ensure_lightweight_deps()

import mvp_reco  # noqa: E402

try:
    import api  # noqa: E402
except ModuleNotFoundError:
    api = None


class FakeFaissIndex:
    """Minimal FAISS-like index performing dot-product search."""

    def __init__(self):
        self._vecs = None

    def add(self, vecs: np.ndarray):
        arr = np.array(vecs, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        self._vecs = arr

    def search(self, queries: np.ndarray, top_k: int):
        if self._vecs is None:
            raise RuntimeError("Index empty")
        q = np.array(queries, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        diff = self._vecs[None, :, :] - q[:, None, :]
        sq_dist = np.sum(diff ** 2, axis=2)  # [Q, N]
        sims = -sq_dist  # higher is better (closer)
        idx = np.argsort(-sims, axis=1)[:, :top_k]
        top_scores = np.take_along_axis(sims, idx, axis=1)
        return top_scores, idx


class PipelineClip:
    def encode_image(self, pil_img: Image.Image) -> np.ndarray:
        return np.array([float(pil_img.width), float(pil_img.height)], dtype=np.float32)


class PipelineIntegrationTests(unittest.TestCase):
    def test_yolo_crop_clip_faiss_pipeline(self):
        class _Scalar:
            def __init__(self, value):
                self._value = value

            def item(self):
                return self._value

        class _DummyBox:
            def __init__(self, coords, cls_id, conf):
                self.xyxy = [np.array(coords, dtype=float)]
                self.cls = [_Scalar(cls_id)]
                self.conf = [_Scalar(conf)]

        class _DummyYOLO:
            def __init__(self):
                self.names = {0: "chair"}

            def predict(self, source, **kwargs):
                return [types.SimpleNamespace(boxes=[_DummyBox([0, 0, 3, 3], 0, 0.97)])]

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "scene.jpg"
            Image.new("RGB", (8, 8), color=(123, 222, 111)).save(img_path)
            pil, instances = mvp_reco.detect_instances(_DummyYOLO(), str(img_path))

            clip = PipelineClip()
            query_emb = clip.encode_image(instances[0]["crop"])

            ref_imgs = [
                Image.new("RGB", (4, 4), color=(0, 0, 0)),
                Image.new("RGB", (3, 3), color=(0, 0, 0)),
            ]
            ref_embs = np.vstack([clip.encode_image(img) for img in ref_imgs])
            index = FakeFaissIndex()
            index.add(ref_embs)

            scores, indices = index.search(query_emb, 1)

        self.assertEqual(instances[0]["bbox"], [0, 0, 3, 3])
        self.assertEqual(tuple(query_emb.tolist()), (3.0, 3.0))
        self.assertEqual(indices[0][0], 1)  # matches 3x3 reference
        self.assertLessEqual(scores[0][0], 0.0)


class APITestBase(unittest.TestCase):
    def setUp(self):
        if api is None or TestClient is None:
            self.skipTest("FastAPI or TestClient not available")

        self.orig_detect = api.detect_instances
        self.orig_draw = api.draw_and_save
        self.orig_try = api._try_load_cache
        self.orig_yolo = api.YOLO
        self.orig_clip = api.ClipEncoder
        self.orig_runs = api.RUNS_DIR
        self.tmp_runs = Path(tempfile.mkdtemp())
        api.RUNS_DIR = self.tmp_runs
        (api.RUNS_DIR / "uploads").mkdir(parents=True, exist_ok=True)

        class _APIDummyYOLO:
            def __init__(self, *args, **kwargs):
                self.names = {0: "det-item"}

            def predict(self, *args, **kwargs):
                return []

        class _APIDummyClip:
            def __init__(self, *args, **kwargs):
                pass

            def encode_image(self, pil_img):
                return np.array([float(pil_img.width), float(pil_img.height)], dtype=np.float32)

        def fake_detect(det_model, image_path):
            pil = Image.open(image_path).convert("RGB")
            crop = pil.crop((0, 0, 4, 4))
            return pil, [{
                "bbox": [0, 0, 4, 4],
                "class": "det-item",
                "conf": 0.92,
                "crop": crop,
            }]

        def fake_draw(pil, results, out_path):
            pil.save(out_path)

        def fake_try_load_cache():
            api.state.catalog_rows = [
                {"sku_id": "sku-a", "title": "Item A", "brand": "Brand A", "image_path": "/static/a.jpg"},
                {"sku_id": "sku-b", "title": "Item B", "brand": "Brand B", "image_path": "/static/b.jpg"},
            ]
            api.state.img_embs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
            api.state.txt_embs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
            api.state.embedding_dim = 2
            return True

        api.YOLO = _APIDummyYOLO
        api.ClipEncoder = _APIDummyClip
        api.detect_instances = fake_detect
        api.draw_and_save = fake_draw
        api._try_load_cache = fake_try_load_cache

        # Force init paths
        api.state.det = api.YOLO()
        api.state.clip = api.ClipEncoder()
        api._try_load_cache()

    def tearDown(self):
        if api is None:
            return
        api.detect_instances = self.orig_detect
        api.draw_and_save = self.orig_draw
        api._try_load_cache = self.orig_try
        api.YOLO = self.orig_yolo
        api.ClipEncoder = self.orig_clip
        api.RUNS_DIR = self.orig_runs
        api.state.det = None
        api.state.clip = None
        api.state.catalog_rows = []
        api.state.img_embs = None
        api.state.txt_embs = None
        shutil.rmtree(self.tmp_runs, ignore_errors=True)

    def make_client(self):
        return TestClient(api.app)


class BackendStorageIntegrationTests(APITestBase):
    def test_recommend_saves_upload_and_visualization(self):
        with self.make_client() as client:
            img_buf = io.BytesIO()
            Image.new("RGB", (6, 6), color=(255, 0, 0)).save(img_buf, format="JPEG")
            img_buf.seek(0)
            files = {"image": ("scene.jpg", img_buf.getvalue(), "image/jpeg")}
            resp = client.post("/recommend", data={"topk": "1", "alpha_img": "0.7"}, files=files)

        self.assertEqual(resp.status_code, 200, resp.text)
        data = resp.json()
        self.assertTrue(data["image_url"].startswith("/runs/"))
        self.assertTrue(data["vis_url"].startswith("/runs/"))
        upload_subpath = Path(data["image_url"].lstrip("/")).relative_to("runs")
        vis_subpath = Path(data["vis_url"].lstrip("/")).relative_to("runs")
        self.assertTrue((api.RUNS_DIR / upload_subpath).is_file())
        self.assertTrue((api.RUNS_DIR / vis_subpath).is_file())


class FrontendBackendIntegrationTests(APITestBase):
    def test_frontend_request_shapes_are_supported(self):
        with self.make_client() as client:
            img_buf = io.BytesIO()
            Image.new("RGB", (5, 5), color=(0, 255, 0)).save(img_buf, format="JPEG")
            img_buf.seek(0)
            files = {"image": ("green.jpg", img_buf.getvalue(), "image/jpeg")}
            resp = client.post(
                "/recommend",
                data={"topk": "2", "alpha_img": "0.5", "return_vis": "true"},
                files=files,
            )
            self.assertEqual(resp.status_code, 200, resp.text)
            payload = resp.json()
            sku_ids = [rec["sku_id"] for inst in payload["instances"] for rec in inst["top_k"]]
            q = client.post("/catalog/query", json={"sku_ids": sku_ids})
            self.assertEqual(q.status_code, 200, q.text)
            catalog = q.json()

        self.assertGreaterEqual(len(payload["instances"]), 1)
        self.assertIn("items", catalog)
        returned_ids = {item["sku_id"] for item in catalog["items"]}
        self.assertTrue(set(sku_ids).issubset(returned_ids.union(set(catalog.get("missing", [])))))


if __name__ == "__main__":
    unittest.main()
