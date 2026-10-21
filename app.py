import streamlit as st
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

from src.models import build_transfer_model
from src.gradcam import GradCAM, overlay_cam_on_image

# Page Setup
st.set_page_config(page_title="Chest X-Ray Pneumonia Screening", page_icon="🫁", layout="wide")

# Medical Disclaimer Banner
st.markdown("""
<div style="background-color:#ffebe9;padding:15px;border-radius:10px;border-left:6px solid #d9381e;margin-bottom:20px;">
    <h4 style="color:#d9381e;margin:0;">⚠️ CLINICAL TRIAGE DEMO — NOT FOR DIAGNOSTIC USE</h4>
    <p style="margin:5px 0 0 0;color:#333;font-size:14px;">
        This AI system provides pre-screening triage assistance to prioritize reading queues. 
        All predictions and heatmap visualizations require validation by a certified radiologist.
    </p>
</div>
""", unsafe_allow_html=True)

st.title("🫁 Chest X-Ray Pneumonia Triage Engine")

# Sidebar Threshold Slider
st.sidebar.header("⚙️ Triage Operating Point")
operating_threshold = st.sidebar.slider(
    "Screening Threshold (Lower = Higher Recall)",
    min_value=0.05, max_value=0.95, value=0.28, step=0.01,
    help="Default calibrated threshold (tau = 0.28) targets >= 95% Sensitivity to minimize false negatives."
)

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_transfer_model("resnet50", strategy="partial")
    model.eval().to(device)
    return model, device

model, device = load_model()
target_layer = model.layer4[-1]
gradcam_engine = GradCAM(model, target_layer)

uploaded_file = st.file_uploader("Upload Chest Radiograph (JPEG or PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    raw_pil = Image.open(uploaded_file).convert("RGB")
    raw_np = np.array(raw_pil)

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(raw_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = model(input_tensor).item()
        prob = 1 / (1 + np.exp(-logit))

    is_pneumonia = prob >= operating_threshold

    cam_map = gradcam_engine.generate_heatmap(input_tensor)
    overlay_img = overlay_cam_on_image(raw_np, cam_map)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input Radiograph")
        st.image(raw_pil, use_container_width=True)
    with col2:
        st.subheader("Grad-CAM Activation Map")
        st.image(overlay_img, use_container_width=True)

    st.markdown("---")
    res_col1, res_col2, res_col3 = st.columns(3)
    with res_col1:
        st.metric("Pneumonia Probability", f"{prob * 100:.2f}%")
    with res_col2:
        st.metric("Operating Threshold", f"{operating_threshold * 100:.1f}%")
    with res_col3:
        if is_pneumonia:
            st.error("🚨 URGENT RADIOLOGIST QUEUE")
        else:
            st.success("✅ ROUTINE QUEUE")