# People Detector API

This project is a simple object detection microservice built with **YOLOv8** and **FastAPI**.  
It allows you to upload an image and get back object detection results — either in **JSON** or as a **processed image with bounding boxes**.

> The goal is to prepare for real-time ML inference on a Raspberry Pi using a lightweight object detection pipeline.

---

## Features

- Object detection using a YOLOv8 model (trained on people)
- REST API with FastAPI
- Two endpoints:
  - `/predict`: returns detection results as JSON
  - `/predict-image`: returns image with drawn bounding boxes
- Private paths and tokens are managed in `private_paths.py` (excluded from Git)

---

## Tech Stack

- Python 3.12
- Ultralytics YOLOv8
- FastAPI + Uvicorn
- OpenCV (for drawing boxes)
- torch + torchvision (CUDA enabled)

---

## Project Structure
<pre><code>people_detector/ 
├── api/ 
│   └── main.py # FastAPI app 
├── data/ # Datasets 
├── runs/ # YOLO training results 
├── yolo_train.py # Training script 
├── requirements.txt
└── README.md</code></pre>

