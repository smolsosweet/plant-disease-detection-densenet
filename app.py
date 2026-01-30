import gradio as gr
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# =========================
# 1. LOAD MODEL & LABELS
# =========================
MODEL_PATH = "best_densenet_model.h5"
JSON_PATH = "class_indices.json"

print("Đang load model AI...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Đang load danh sách nhãn bệnh...")
with open(JSON_PATH, 'r', encoding="utf-8") as f:
    class_indices = json.load(f)

# index -> class name
class_names = {v: k for k, v in class_indices.items()}

# =========================
# 2. NHÃN TIẾNG VIỆT
# =========================
VIETNAMESE_LABELS = {
    "Apple___Apple_scab": "Táo – Bệnh ghẻ táo",
    "Apple___Black_rot": "Táo – Bệnh thối đen",
    "Apple___Cedar_apple_rust": "Táo – Bệnh gỉ sắt",
    "Apple___healthy": "Táo – Khỏe mạnh",

    "Blueberry___healthy": "Việt quất – Khỏe mạnh",

    "Cherry_(including_sour)___Powdery_mildew": "Anh đào – Bệnh phấn trắng",
    "Cherry_(including_sour)___healthy": "Anh đào – Khỏe mạnh",

    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Ngô – Đốm lá xám",
    "Corn_(maize)___Common_rust_": "Ngô – Bệnh gỉ sắt",
    "Corn_(maize)___Northern_Leaf_Blight": "Ngô – Bệnh cháy lá",
    "Corn_(maize)___healthy": "Ngô – Khỏe mạnh",

    "Grape___Black_rot": "Nho – Bệnh thối đen",
    "Grape___Esca_(Black_Measles)": "Nho – Bệnh Esca",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Nho – Bệnh cháy lá",
    "Grape___healthy": "Nho – Khỏe mạnh",

    "Orange___Haunglongbing_(Citrus_greening)": "Cam – Bệnh Greening (HLB)",

    "Peach___Bacterial_spot": "Đào – Bệnh đốm vi khuẩn",
    "Peach___healthy": "Đào – Khỏe mạnh",

    "Pepper,_bell___Bacterial_spot": "Ớt chuông – Bệnh đốm vi khuẩn",
    "Pepper,_bell___healthy": "Ớt chuông – Khỏe mạnh",

    "Potato___Early_blight": "Khoai tây – Bệnh cháy sớm",
    "Potato___Late_blight": "Khoai tây – Bệnh cháy muộn",
    "Potato___healthy": "Khoai tây – Khỏe mạnh",

    "Raspberry___healthy": "Mâm xôi – Khỏe mạnh",
    "Soybean___healthy": "Đậu nành – Khỏe mạnh",

    "Squash___Powdery_mildew": "Bí – Bệnh phấn trắng",

    "Strawberry___Leaf_scorch": "Dâu tây – Bệnh cháy lá",
    "Strawberry___healthy": "Dâu tây – Khỏe mạnh",

    "Tomato___Bacterial_spot": "Cà chua – Bệnh đốm vi khuẩn",
    "Tomato___Early_blight": "Cà chua – Bệnh cháy sớm",
    "Tomato___Late_blight": "Cà chua – Bệnh cháy muộn",
    "Tomato___Leaf_Mold": "Cà chua – Bệnh mốc lá",
    "Tomato___Septoria_leaf_spot": "Cà chua – Bệnh đốm lá Septoria",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Cà chua – Nhện đỏ",
    "Tomato___Target_Spot": "Cà chua – Bệnh đốm mục tiêu",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Cà chua – Virus xoăn lá",
    "Tomato___Tomato_mosaic_virus": "Cà chua – Virus khảm",
    "Tomato___healthy": "Cà chua – Khỏe mạnh"
}

# =========================
# 3. HÀM DỰ ĐOÁN
# =========================
def predict_disease(image):
    image = image.convert("RGB").resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)[0]

    results = {}
    for i, prob in enumerate(predictions):
        eng = class_names[i]
        viet = VIETNAMESE_LABELS.get(eng, eng)
        results[viet] = float(prob)

    return results

# =========================
# 4. GIAO DIỆN GRADIO (ĐẸP & DỄ DÙNG)
# =========================
theme = gr.themes.Default(
    primary_hue="blue",
    secondary_hue="indigo",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Roboto"),
    font_mono=gr.themes.GoogleFont("Roboto Mono"),
    radius_size="lg",
    spacing_size="md",
)

with gr.Blocks(theme=theme, title="Hệ Thống Chẩn Đoán Bệnh Cây Trồng AI", css="""
    footer {
        display: none !important;
    }
    """) as app:
    
    gr.Markdown("""
    <div style="text-align: center; padding: 20px; background-color: #f8fafc; border-bottom: 1px solid #e2e8f0;">
        <h1 style="margin: 0; color: #1e40af;">🌿 Hệ Thống Chẩn Đoán Bệnh Cây Trồng Bằng AI</h1>
        <p style="margin: 5px 0 0; color: #64748b;">Ứng dụng Deep Learning (DenseNet121) trong Nông nghiệp Thông minh</p>
    </div>
    """)
    
    gr.Markdown("""
    ### Hướng dẫn sử dụng:
    - Tải lên ảnh lá cây rõ nét để AI phân tích.
    - Nhấn nút "Phân tích" để nhận kết quả chẩn đoán.
    - Kết quả hiển thị top 3 khả năng cao nhất với xác suất.
    """, elem_classes="guide")
    
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="Tải lên ảnh lá cây",
                height=400,
                interactive=True,
                show_label=True,
                elem_id="image-upload"
            )
        with gr.Column(scale=1):
            output = gr.Label(
                num_top_classes=3,
                label="Kết quả chẩn đoán",
                show_label=True,
                elem_id="result-output"
            )
    
    with gr.Row():
        predict_btn = gr.Button(
            "🔍 Phân tích bệnh",
            variant="primary",
            size="lg",
            elem_id="analyze-button"
        )
    
    predict_btn.click(predict_disease, image_input, output)

    app.launch(footer_links=[])