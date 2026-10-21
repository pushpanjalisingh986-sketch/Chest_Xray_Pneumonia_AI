import streamlit as st
import torch
import numpy as np
from PIL import Image

from src.models import build_transfer_model
from src.gradcam import GradCAMEngine
from src.guardrails import validate_radiograph
from src.inference import predict_with_uncertainty
from src.report import generate_pdf_report

st.set_page_config(page_title="Pneumonia AI Screening Engine", layout="wide")

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_transfer_model("resnet50", strategy="partial")
    
    weights_path = "resnet50_pneumonia.pth"
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(weights_path))
    else:
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        
    model.eval().to(device)
    return model, device

model, device = load_model()

st.title("🩺 Clinical Pneumonia AI Triage Engine")
st.markdown("Enterprise decision support pipeline featuring **Monte Carlo Uncertainty Estimation**, **Input Guardrails**, and **Grad-CAM Explainability**.")

# Sidebar Configuration
st.sidebar.header("Clinical Operating Parameters")
threshold = st.sidebar.slider("Operating Threshold (τ)", min_value=0.05, max_value=0.95, value=0.28, step=0.01)
st.sidebar.info("Operating at τ = 0.28 enforces Sensitivity ≥ 95.0% to minimize false negatives.")

uploaded_file = st.file_uploader("Upload Chest Radiograph (JPEG / PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    raw_img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(raw_img)

    # 1. Input Guardrail Check
    is_valid, msg = validate_radiograph(img_np)
    if not is_valid:
        st.error(f"🚨 Guardrail Rejection: {msg}")
    else:
        # Preprocessing
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        input_tensor = transform(image=img_np)['image'].unsqueeze(0).to(device)

        # 2. Monte Carlo Dropout Inference
        prob, uncertainty = predict_with_uncertainty(model, input_tensor, num_samples=15)

        # 3. Grad-CAM Activation Map Generation
        gradcam = GradCAMEngine(model, target_layer=model.layer4[-1])
        heatmap, overlay_np = gradcam.generate_overlay(input_tensor, img_np)
        overlay_pil = Image.fromarray(overlay_np)

        # UI Rendering
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Input Radiograph")
            st.image(raw_img, use_container_width=True)
        with col2:
            st.subheader("Grad-CAM Activation Map")
            st.image(overlay_pil, use_container_width=True)

        st.divider()

        # Decision Output Metrics
        t_col1, t_col2, t_col3 = st.columns(3)
        with t_col1:
            st.metric("Pneumonia Risk Score", f"{prob * 100:.2f}%", delta=f"± {uncertainty * 100:.2f}% Uncertainty")
        with t_col2:
            st.metric("Operating Threshold (τ)", f"{threshold * 100:.1f}%")
        with t_col3:
            if uncertainty > 0.15:
                triage_status = "⚠️ HIGH UNCERTAINTY — ESCALATED"
                st.error(triage_status)
            elif prob >= threshold:
                triage_status = "🚨 URGENT RADIOLOGIST QUEUE"
                st.error(triage_status)
            else:
                triage_status = "✅ ROUTINE QUEUE"
                st.success(triage_status)

        # 4. Clinical PDF Report Export
        pdf_bytes = generate_pdf_report(
            patient_id=uploaded_file.name.split('.')[0],
            prob=prob,
            uncertainty=uncertainty,
            threshold=threshold,
            triage_status=triage_status,
            overlay_img_pil=overlay_pil
        )

        st.download_button(
            label="📄 Download Clinical Diagnostic PDF Report",
            data=pdf_bytes,
            file_name=f"pneumonia_report_{uploaded_file.name.split('.')[0]}.pdf",
            mime="application/pdf"
        )