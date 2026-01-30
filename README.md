# 🌿 AI Plant Disease Detection System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange)
![Gradio](https://img.shields.io/badge/Gradio-App-green)
![DenseNet121](https://img.shields.io/badge/Model-DenseNet121-red)

## 📖 Introduction
This project is an AI-powered system designed to detect and classify plant diseases from leaf images. Utilizing the **DenseNet121** architecture and Transfer Learning, the model can identify **38 different classes** of plant diseases with high accuracy.

The application is deployed with a user-friendly web interface using **Gradio**, allowing users to upload images or use their camera for real-time diagnosis.

## 🚀 Live Demo
Try the web app here: **[LINK_HUGGING_FACE_CUA_EM_O_DAY]**
*(Click the link to test the model on Hugging Face Spaces)*

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
