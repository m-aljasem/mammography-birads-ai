# 🔬 Mammography BIRADS Classification

Deep learning system for automatic **BIRADS category prediction** from mammography images using EfficientNetB0.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-ff4b4b.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 👤 Author

- **Name**: Mohamad AlJasem, MD MPH MSc  
- **Email**: [mohamad@aljasem.eu.org](mailto:mohamad@aljasem.eu.org)  
- **GitHub**: [github.com/m-aljasem](https://github.com/m-aljasem)  
- **Website**: [aljasem.eu.org](https://aljasem.eu.org)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [BIRADS Levels](#-birads-levels)
- [Exported Weights](#-exported-weights)
- [License](#-license)
- [Disclaimer](#-disclaimer)

---

## 🎯 Overview

This project implements an **EfficientNetB0-based classifier** that maps screening mammography images to a **BIRADS category (1–5)**, indicating the likelihood of malignancy.

It provides a clean Python package with:

- A training pipeline
- A Streamlit web application
- Exportable model weights for deployment

> ⚠️ **Clinical note**: This is a research tool, not a diagnostic device.

---

## ✨ Features

### 🔬 Modeling
- EfficientNetB0 backbone with ImageNet pretraining
- Custom classification head for 5 BIRADS classes
- Softmax output with confidence scores

### 🧪 Data
- Preprocessing + resizing to **256×224 RGB**
- Option to integrate multiple public mammography datasets

### 🌐 App & Deployment
- Streamlit UI:
  - Upload mammography image
  - Get BIRADS prediction + confidence
- Model weights loaded from `models/birads_model.h5` when available

---

## 🛠 Tech Stack

- Python 3.8+
- TensorFlow / Keras
- EfficientNetB0
- Streamlit

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

For development tools (linting, tests, notebooks):

```bash
pip install -r requirements-dev.txt
```

---

## 🚀 Quick Start

### 1️⃣ Train the Model & Export Weights

```bash
cd mammography-birads-ai
python src/train.py
```

This will:
- Train the EfficientNetB0-based model
- Save best weights to:

```text
models/birads_model.h5
```

### 2️⃣ Run the Streamlit App

```bash
cd mammography-birads-ai
streamlit run app.py
```

Upload a mammography image and get:
- Predicted **BIRADS class (1–5)**
- Associated confidence score

---

## 🧑‍💻 Usage

### 🌐 Web App

```bash
streamlit run app.py
```

1. Upload a mammography image (PNG/JPG).  
2. The app:
   - Builds the EfficientNetB0 model
   - Loads `models/birads_model.h5` if present
   - Shows BIRADS class and confidence.

If no weights are found, the app warns and uses random weights (for demo only).

### 🧬 Programmatic Usage

```python
from src.model import build_birads_model
import tensorflow as tf
import numpy as np

model = build_birads_model()
model.load_weights("models/birads_model.h5")  # after training

# img_preprocessed: (1, 256, 224, 3) float32
pred = model.predict(img_preprocessed, verbose=0)[0]
birads_class = np.argmax(pred) + 1
confidence = pred[birads_class - 1]
```

---

## 🗂 Project Structure

```text
mammography-birads-ai/
├── app.py                    # Streamlit app
├── config/
├── data/                     # Mammography datasets
├── docs/                     # Research, architecture, benchmarks, guides
├── experiments/
├── models/                   # Saved weights (birads_model.h5)
├── notebooks/
├── scripts/
├── src/
│   ├── __init__.py
│   └── model.py              # build_birads_model()
└── tests/
```

---

## 📊 BIRADS Levels

The model predicts one of the following BIRADS categories:

1. **BIRADS 1** – Negative  
2. **BIRADS 2** – Benign finding  
3. **BIRADS 3** – Probably benign  
4. **BIRADS 4** – Suspicious abnormality  
5. **BIRADS 5** – Highly suggestive of malignancy  

---

## 📦 Exported Weights

- Training script saves to:

```text
../models/birads_model.h5
```

- Streamlit app loads from:

```text
models/birads_model.h5
```

To deploy elsewhere, copy:

```text
mammography-birads-ai/
└── models/
    └── birads_model.h5
```

and run the same `app.py`.

---

## 📄 License

Licensed under the **MIT License**.  
See `LICENSE` for details.

---

## 🏥 Disclaimer

> This project is for **research and educational purposes only**.  
> It must **not** be used for clinical diagnosis, treatment, or patient management.  
> Always consult qualified healthcare professionals for medical decisions.


## 🌐 RESTful API

The project includes a FastAPI server for programmatic access to the model.

### Starting the API Server

```bash
python api.py
# or
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### API Endpoints

- `GET /` - Root endpoint with API information
- `GET /health` - Health check endpoint
- `GET /model/info` - Get model information
- `POST /predict` - Make a prediction

### API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Example Usage

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Make prediction (example for image-based models)
with open("test_image.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post("http://localhost:8000/predict", files=files)
    print(response.json())
```

## 🔌 MCP Server

The project includes a Model Context Protocol (MCP) server for integration with AI assistants.

### Starting the MCP Server

```bash
python mcp_server.py
```

### MCP Tools

The server exposes the following tools:

- `predict` - Make a prediction using the model
- `model_info` - Get information about the loaded model
- `health_check` - Check if the model is loaded and ready

### MCP Client Integration

To use with an MCP client:

```python
from mcp import ClientSession, StdioServerParameters
import asyncio

async def main():
    async with ClientSession(
        StdioServerParameters(
            command="python",
            args=["mcp_server.py"]
        )
    ) as session:
        # List tools
        tools = await session.list_tools()
        print(tools)
        
        # Call tool
        result = await session.call_tool(
            "health_check",
            {}
        )
        print(result)

asyncio.run(main())
```

