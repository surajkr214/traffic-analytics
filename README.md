# 🚦 Traffic Analytics Pro

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://traffic-analytics.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/AI-YOLOv8-green?logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-purple)

> **An Industrial-Grade Computer Vision System for Real-Time Traffic Intelligence.**

## 🔗 Live Demo
**[PROTOTYPE](https://traffic-analytics.streamlit.app)**
---

## 📖 Executive Summary
**Traffic Analytics Pro** is a state-of-the-art AI dashboard engineered to transform raw video footage into actionable traffic insights. By leveraging a custom-trained **YOLOv8 model (VisDrone dataset)**, the system goes beyond simple car counting—distinguishing between pedestrians, heavy trucks, buses, and two-wheelers with high precision.

Designed for scalability, this solution supports multiple input streams (Uploads, YouTube, RTSP) and provides real-time analytics for smart city planning and congestion management.

---

## 🚀 Key Features

### 🧠 **Advanced AI Detection**
* **Custom VisDrone Model:** Specialized detection for 10+ classes including `Pedestrian`, `Truck`, `Bus`, `Motor`, and `Tricycle`.
* **High Precision:** Fine-tuned confidence thresholds to minimize false positives in dense traffic.

### 📊 **Real-Time Intelligence**
* **Live Dashboard:** Instant visualization of traffic volume, vehicle composition, and lane usage.
* **Congestion Alerts:** Automated "Heavy Traffic" warnings triggered when vehicle density exceeds safety thresholds (e.g., >20 vehicles).

### 💾 **Enterprise-Grade Logging**
* **SQL Database:** Every detection is timestamped and logged into an SQLite database for historical auditing.
* **Exportable Reports:** Download detailed CSV reports filtered by vehicle type, congestion level, or time of day.

---

## 🛠️ Technical Architecture

| Component | Technology | Description |
| :--- | :--- | :--- |
| **AI Engine** | **YOLOv8 (Ultralytics)** | Deep learning object detection trained on VisDrone data. |
| **Frontend** | **Streamlit** | Interactive web-based dashboard for real-time visualization. |
| **Vision** | **OpenCV (Headless)** | High-performance video frame processing and annotation. |
| **Database** | **SQLite** | Lightweight, serverless transactional SQL database engine. |
| **Charts** | **Plotly** | Interactive, publication-quality graphing library. |

---

## 💻 Installation & Setup

### **Prerequisites**
* Python 3.9 or higher
* Git

### **1. Clone the Repository**
```bash
git clone [https://github.com/YOUR_USERNAME/traffic_analytics.git](https://github.com/YOUR_USERNAME/traffic_analytics.git)
cd traffic_analytics

## 📂 Project Structure
```text
traffic_analytics/
├── app.py                  # Main Dashboard Application
├── traffic_script.py       # Standalone processing script
├── model.pt                # Custom Trained YOLOv8 Weights
├── requirements.txt        # Python Dependencies
├── packages.txt            # Linux System Dependencies
└── README.md               # Documentation