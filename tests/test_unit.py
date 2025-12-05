import csv
import importlib
import importlib.util
import io
import shutil
import sys
import tempfile
import types
from pathlib import Path

# This unified unittest runner covers:
#   1. mvp_reco.load_catalog: verifies CSV parsing + embedding generation with a DummyClip.
#   2. mvp_reco.cos_sim: checks cosine similarity math on simple vectors.
#   3. Image detection + cropping utilities (detect_instances + draw/save).
#   4. Catalog database helpers (load_catalog_from_db + save_embeddings_to_db).
#   5. data_merge.choose_text_field for language handling.
#   6. FastAPI request/response handling (health + recommend endpoints).
# Heavy dependencies (torch/torchvision/open_clip/YOLO) are stubbed so the tests stay lightweight.

import numpy as np
from PIL import Image

try:
    from fastapi.testclient import TestClient
except ModuleNotFoundError:
    TestClient = None


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
    """Provide minimal stubs so importing mvp_reco works without heavy deps."""
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
            raise RuntimeError("open_clip is not available in unit tests.")

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
                raise RuntimeError("YOLO predict() is not available in unit tests.")

        fake_ultra = types.SimpleNamespace(YOLO=_DummyYOLO)
        sys.modules["ultralytics"] = fake_ultra


_ensure_lightweight_deps()

import mvp_reco
from data_merge import choose_text_field

try:
    import api
except ModuleNotFoundError:
    api = None

init_db = None
_init_db_path = Path(__file__).resolve().parent / "scripts" / "init_db_from_csv.py"
if _init_db_path.exists():
    spec = importlib.util.spec_from_file_location("init_db_from_csv_module", _init_db_path)
    if spec and spec.loader:
        init_db = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(init_db)


class DummyClip:
    """Minimal stub to avoid loading the real CLIP weights."""

    def encode_image(self, pil_img: Image.Image) -> np.ndarray:
        # Encode image into a simple 2D vector based on dimensions.
        return np.array([float(pil_img.width), float(pil_img.height)], dtype=np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        # Encode text into a stable 2D vector based on length.
        return np.array([float(len(text)), 1.0], dtype=np.float32)


class DetectionAndCroppingTests(unittest.TestCase):
    def test_detect_instances_converts_yolo_boxes(self):
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
                return [types.SimpleNamespace(boxes=[_DummyBox([1.2, 2.7, 8.9, 9.1], 0, 0.95)])]

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "input.jpg"
            Image.new("RGB", (10, 10), color=(0, 0, 0)).save(img_path)

            pil, instances = mvp_reco.detect_instances(_DummyYOLO(), str(img_path))

        self.assertEqual(pil.size, (10, 10))
        self.assertEqual(len(instances), 1)
        inst = instances[0]
        self.assertEqual(inst["bbox"], [1, 2, 8, 9])
        self.assertEqual(inst["class"], "chair")
        self.assertAlmostEqual(inst["conf"], 0.95)
        self.assertEqual(inst["crop"].size, (7, 7))

    def test_draw_and_save_outputs_visualization(self):
        pil = Image.new("RGB", (8, 8), color=(255, 255, 255))
        results = [{
            "bbox": [1, 1, 6, 6],
            "class": "chair",
            "det_conf": 0.88,
            "top1": {"sku_id": "sku-42"},
        }]

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "vis.jpg"
            mvp_reco.draw_and_save(pil, results, out_path)
            self.assertTrue(out_path.is_file())
            saved = Image.open(out_path)
            try:
                self.assertEqual(saved.size, pil.size)
            finally:
                saved.close()


class LoadCatalogTests(unittest.TestCase):
    def test_load_catalog_builds_embeddings_from_csv(self):
        clip = DummyClip()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            img_dir = tmp_path / "images"
            img_dir.mkdir()
            img_path = img_dir / "sample.jpg"
            Image.new("RGB", (10, 20), color=(255, 0, 0)).save(img_path)

            csv_path = tmp_path / "catalog.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["sku_id", "title", "brand", "image_path"])
                writer.writeheader()
                writer.writerow(
                    {
                        "sku_id": "sku-1",
                        "title": "Demo Product",
                        "brand": "Demo Brand",
                        "image_path": f"images/{img_path.name}",
                    }
                )

            catalog = mvp_reco.load_catalog(clip, str(csv_path))

        self.assertEqual(len(catalog["rows"]), 1)
        self.assertEqual(catalog["rows"][0]["sku_id"], "sku-1")
        self.assertEqual(catalog["img_embs"].shape, (1, 2))
        self.assertEqual(catalog["txt_embs"].shape, (1, 2))


class CosSimTests(unittest.TestCase):
    def test_cos_sim_returns_expected_values(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        bank = np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32)

        sims = mvp_reco.cos_sim(a, bank)

        self.assertTrue(np.allclose(sims[0], 1.0))
        self.assertAlmostEqual(sims[1], 1 / np.sqrt(2), places=6)


class CatalogDatabaseTests(unittest.TestCase):
    def test_init_db_script_loads_csv(self):
        if init_db is None:
            self.skipTest("init_db_from_csv module not available")
        original_db_path = init_db.DB_PATH
        original_csv_path = init_db.CSV_PATH
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            db_path = tmpdir / "catalog.db"
            csv_path = tmpdir / "catalog.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["sku_id", "title", "brand", "image_path"])
                writer.writeheader()
                writer.writerow({
                    "sku_id": "sku-db",
                    "title": "DB Item",
                    "brand": "DB Brand",
                    "image_path": "images/db.jpg",
                })

            init_db.DB_PATH = db_path
            init_db.CSV_PATH = csv_path
            conn = init_db.get_conn()
            try:
                init_db.init_schema(conn)
                init_db.load_csv_into_products(conn)
                rows = conn.execute("SELECT sku_id, title, brand, image_path FROM products").fetchall()
                self.assertEqual(len(rows), 1)
                r = rows[0]
                self.assertEqual(r["sku_id"], "sku-db")
                self.assertEqual(r["title"], "DB Item")
                self.assertEqual(r["brand"], "DB Brand")
                self.assertEqual(r["image_path"], "images/db.jpg")
            finally:
                conn.close()
        init_db.DB_PATH = original_db_path
        init_db.CSV_PATH = original_csv_path


class ChooseTextFieldTests(unittest.TestCase):
    def test_prefers_english_value(self):
        field_list = [
            {"language_tag": "fr_FR", "value": "Bonjour"},
            {"language_tag": "en_US", "value": "Hello"},
        ]
        self.assertEqual(choose_text_field(field_list, default=""), "Hello")

    def test_fallback_to_first_value(self):
        field_list = [
            {"language_tag": "es_ES", "value": "Hola"},
            {"language_tag": "fr_FR", "value": "Salut"},
        ]
        self.assertEqual(choose_text_field(field_list, default="Default"), "Hola")

    def test_returns_default_for_empty_list(self):
        self.assertEqual(choose_text_field([], default="Missing"), "Missing")


class FrontendAPITests(unittest.TestCase):
    def setUp(self):
        if api is None or TestClient is None:
            self.skipTest("FastAPI components not available")
        self.orig_detect = api.detect_instances
        self.orig_draw = api.draw_and_save
        self.orig_try = api._try_load_cache
        self.orig_yolo = api.YOLO
        self.orig_clip = api.ClipEncoder
        self.orig_runs_dir = api.RUNS_DIR
        self.tmp_runs_dir = Path(tempfile.mkdtemp())
        api.RUNS_DIR = self.tmp_runs_dir
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
                return np.array([1.0, 0.0], dtype=np.float32)

        def fake_detect_instances(det_model, image_path):
            pil = Image.new("RGB", (6, 6), color=(0, 0, 0))
            crop = pil.crop((0, 0, 4, 4))
            return pil, [{
                "bbox": [0, 0, 4, 4],
                "class": "det-item",
                "conf": 0.91,
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
        api.detect_instances = fake_detect_instances
        api.draw_and_save = fake_draw
        api._try_load_cache = fake_try_load_cache

    def tearDown(self):
        api.detect_instances = self.orig_detect
        api.draw_and_save = self.orig_draw
        api._try_load_cache = self.orig_try
        api.YOLO = self.orig_yolo
        api.ClipEncoder = self.orig_clip
        api.state.det = None
        api.state.clip = None
        api.state.catalog_rows = []
        api.state.img_embs = None
        api.state.txt_embs = None
        api.RUNS_DIR = self.orig_runs_dir
        shutil.rmtree(self.tmp_runs_dir, ignore_errors=True)

    def _make_client(self):
        return TestClient(api.app)

    def test_health_endpoint_reports_ready(self):
        with self._make_client() as client:
            resp = client.get("/health")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertTrue(data["ok"])
            self.assertTrue(data["models_ready"])
            self.assertTrue(data["catalog_ready"])

    def test_recommend_endpoint_returns_instances(self):
        with self._make_client() as client:
            img_bytes = io.BytesIO()
            Image.new("RGB", (4, 4), color=(255, 0, 0)).save(img_bytes, format="JPEG")
            img_bytes.seek(0)
            files = {
                "image": ("test.jpg", img_bytes.getvalue(), "image/jpeg")
            }
            resp = client.post(
                "/recommend",
                data={"topk": "1", "alpha_img": "0.5", "return_vis": "false"},
                files=files,
            )
            self.assertEqual(resp.status_code, 200, resp.text)
            data = resp.json()
            self.assertIn("instances", data)
            self.assertEqual(len(data["instances"]), 1)
            inst = data["instances"][0]
            self.assertEqual(inst["bbox"], [0, 0, 4, 4])
            self.assertEqual(len(inst["top_k"]), 1)
            self.assertIsNotNone(data["image_url"])


if __name__ == "__main__":
    unittest.main()
