---
title: Plant Disease AI
emoji: 🌿
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
---

# 🌿 AI Plant Disease Detection System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange)
![Gradio](https://img.shields.io/badge/Gradio-App-green)
![DenseNet121](https://img.shields.io/badge/Model-DenseNet121-red)

## 📖 Introduction
This project is an AI-powered system designed to detect and classify plant diseases from leaf images. Utilizing the **DenseNet121** architecture and Transfer Learning, the model can identify **38 different classes** of plant diseases with high accuracy.

The application is deployed with a user-friendly web interface using **Gradio**, allowing users to upload images or use their camera for real-time diagnosis.

## 🚀 Live Demo
Try the web app here: **[https://huggingface.co/spaces/smolsosweet/plant-disease-ai](https://huggingface.co/spaces/smolsosweet/plant-disease-ai)**
*(Click the link to test the model on Hugging Face Spaces)*

## ✨ Key Features
* **Real-time Camera Support:** Capture images directly from your mobile or laptop webcam.
* **Smart Confidence Threshold:** Rejects unclear or non-leaf images to prevent false predictions.
* **Treatment Recommendations:** Provides actionable agricultural advice and treatment steps for the diagnosed plant disease.
* **Example Datasets:** Built-in sample images for quick testing.

## 🛠️ Tech Stack
* **Core:** Python, TensorFlow, Keras
* **Model:** DenseNet121 (Transfer Learning)
* **Interface:** Gradio
* **Deployment:** Hugging Face Spaces

## 📂 Dataset
* **Source:** New Plant Diseases Dataset (Augmented)
* **Classes:** 38 classes (Apple, Tomato, Corn, Potato, etc.)
* **Preprocessing:** Image Rescaling, Horizontal Flip Augmentation.

## 📊 Performance
* **Architecture:** DenseNet121 (Pre-trained on ImageNet)
* **Training Epochs:** 20
* **Optimizer:** Adam
* **Best Accuracy:** ~95% (Em điền số thực tế vào đây)

## 🔧 Installation & Usage
To run this project locally:

1. **Clone the repo:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/plant-disease-detection-densenet.git](https://github.com/YOUR_USERNAME/plant-disease-detection-densenet.git)
   cd plant-disease-detection-densenet
