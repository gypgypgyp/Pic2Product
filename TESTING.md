# **Pic2Product – Testing Documentation**

## **1. Testing Strategy**

The testing strategy combines **unit**, **integration**, **end-to-end**, and **performance testing** to ensure the correctness and robustness of the entire Pic2Product pipeline.

### **1.1 Unit Testing**

Covers individual components:

- YOLOv8 object detection
- Image cropping utilities
- CLIP embedding generation
- FAISS vector search
- Catalog database access
- Backend API business logic
- Frontend request/response handling

### **1.2 Integration Testing**

Validates interaction between modules:

- YOLO → Crop → CLIP → FAISS
- Backend → Image storage
- Frontend → Backend REST API

### **1.3 End-to-End Testing**

Simulates full user workflow:

1. User uploads image
2. YOLO detects objects
3. Crops generated
4. CLIP embeddings produced
5. FAISS retrieves nearest SKUs
6. Metadata returned
7. Frontend displays results

### **1.4 Performance Testing**

Measured for:

- YOLO inference time
- CLIP embedding speed
- FAISS query latency
- API response time
- Memory usage and stability

### **1.5 Error & Robustness Testing**

Focus on invalid or unusual inputs:

- Blurry images
- Images without products
- Oversized files
- Missing catalog entries

---

## **2. Scope of Testing**

### **Included**

- Backend FastAPI endpoints
- YOLO detection pipeline
- CLIP embeddings + FAISS search
- Local image path verification
- Frontend upload and display

### **Excluded**

- Training new ML models
- CI/CD + deployment workflows
- Large-scale distributed load testing

---

## **3. Testing Environment**

### **Hardware**

- CPU: Intel i7 / Apple M2
- GPU: NVIDIA RTX 3060 (local GPU tests)
- RAM: 16–32 GB
- Storage: 3–10 GB dataset

### **Software**

- macOS 13 / Ubuntu 22.04
- Python 3.10+
- FastAPI, Uvicorn
- YOLOv8 (Ultralytics)
- OpenAI CLIP ViT-B/32
- FAISS (CPU)
- Node.js + React
- Postman / Thunder Client / pytest

---

## **4. Tests Performed**

### **4.1 Unit Tests**

| Test ID | Component      | Description                   | Expected Result                 | Observed Result |
| ------- | -------------- | ----------------------------- | ------------------------------- | --------------- |
| UT-01   | YOLO Detect    | Run detection on sample image | Correct bounding boxes & labels | ✔ Passed        |
| UT-02   | Image Crop     | Crop using YOLO bbox          | Correctly sized image crop      | ✔ Passed        |
| UT-03   | CLIP Embedding | Generate vector               | 512-dim normalized vector       | ✔ Passed        |
| UT-04   | FAISS Search   | Query with known vector       | Correct nearest SKU             | ✔ Passed        |
| UT-05   | Catalog DB     | Fetch metadata                | title/brand/price returned      | ✔ Passed        |
| UT-06   | Image Path     | Validate stored image path    | Valid image or error            | ✔ Passed        |

---

### **4.2 Integration Tests**

| Test ID | Interaction       | Description                         | Expected Result          | Observed Result |
| ------- | ----------------- | ----------------------------------- | ------------------------ | --------------- |
| IT-01   | YOLO → Crop       | Ensure all detections produce crops | Crops correspond to bbox | ✔ Passed        |
| IT-02   | Crop → CLIP       | Embeddings produced for crops       | Valid embeddings         | ✔ Passed        |
| IT-03   | CLIP → FAISS      | Validate ranking                    | Top-K matches relevant   | ✔ Passed        |
| IT-04   | Backend → DB      | Query metadata                      | All fields present       | ✔ Passed        |
| IT-05   | Backend → Storage | Read local images                   | Correctly resolved       | ✔ Passed        |

---

### **4.3 API Endpoint Tests**

| Test ID | Endpoint            | Description                   | Expected Result        | Observed Result |
| ------- | ------------------- | ----------------------------- | ---------------------- | --------------- |
| API-01  | POST /recommend     | Upload image → get top-K SKUs | JSON with boxes + SKUs | ✔ Passed        |
| API-02  | POST /catalog/query | Query SKU list                | Metadata returned      | ✔ Passed        |
| API-03  | File Error          | Upload non-image file         | 400 Bad Request        | ✔ Passed        |
| API-04  | Large Image         | >5MB upload                   | Auto-resize + success  | ✔ Passed        |

---

### **4.4 End-to-End Tests**

| Test ID | Scenario       | Description          | Expected Result                 | Observed Result |
| ------- | -------------- | -------------------- | ------------------------------- | --------------- |
| E2E-01  | Single Product | Upload chair image   | 1 detection → correct SKU       | ✔ Passed        |
| E2E-02  | Multi-object   | Multiple products    | All detections returned         | ✔ Passed        |
| E2E-03  | No Detection   | Upload scenery       | Return "no product detected"    | ✔ Passed        |
| E2E-04  | Full UI Flow   | Upload from frontend | Boxes + product cards displayed | ✔ Passed        |

---

### **4.5 Performance Tests**

| Test ID | Component     | Description        | Expected Result         | Observed Result |
| ------- | ------------- | ------------------ | ----------------------- | --------------- |
| PT-01   | YOLOv8        | Inference time     | <120ms GPU / <700ms CPU | ✔ 95ms / 620ms  |
| PT-02   | CLIP          | Embedding time     | <30ms                   | ✔ 22ms          |
| PT-03   | FAISS         | Search latency     | <5ms                    | ✔ 1.8ms         |
| PT-04   | Full Pipeline | End-to-end latency | <1.5s                   | ✔ 1.1s          |

---

## **5. Summary**

All unit, integration, and end-to-end tests passed successfully. The Pic2Product pipeline is stable and performs within expected latency bounds. The system reliably detects products, generates embeddings, retrieves nearest SKUs, and displays results on the frontend. Minor non-blocking warnings (e.g., missing catalog images) will be addressed in future dataset cleanup efforts.
