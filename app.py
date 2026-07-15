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
VIETNAMESE_LABELS_PATH = "vietnamese_labels.json"
print("Đang load nhãn tiếng Việt...")
with open(VIETNAMESE_LABELS_PATH, 'r', encoding="utf-8") as f:
    VIETNAMESE_LABELS = json.load(f)

# =========================
# 3. HÀM DỰ ĐOÁN
# =========================
def predict_disease(image):
    if image is None:
        return {"Lỗi: Vui lòng tải ảnh lên trước!": 1.0}
    
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
                elem_id="image-upload",
                sources=["upload", "webcam"]
            )
            gr.Examples(
                examples=[
                    "examples/apple_scab_leaf.png",
                    "examples/healthy_corn_leaf.png"
                ],
                inputs=image_input,
                label="Ảnh mẫu (Click để thử)"
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