import streamlit as st
import requests
import base64
from io import BytesIO
from PIL import Image
import numpy as np

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Plant Disease Detector", layout="wide")

st.title("Plant Disease Detector — MobileNet Demo")
st.markdown("Upload a leaf image or take one with your webcam, choose a model, and get prediction + Grad-CAMs.")

models = {
    "MobileNetV2 (mobilenet_v1)": "mobilenet_v1",
    # Add more 
}
model_display = st.selectbox("Choose model", options=list(models.keys()))
model_key = models[model_display]

st.markdown("**Select Grad-CAM conv layers to visualize** (optional). Leave empty to use defaults.")
layers_input = st.text_input("Comma-separated layer names (e.g. Conv_1,block_3_project_BN)", value="")

col1, col2 = st.columns(2)

with col1:
    st.header("Input")
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    webcam_img = st.camera_input("Or take a photo with your webcam")

    image_file = None
    if webcam_img is not None:
        image_file = webcam_img
    elif uploaded is not None:
        image_file = uploaded

with col2:
    st.header("Controls")
    st.write("Model:", model_display)
    st.write("Layers:", layers_input if layers_input.strip() else "defaults")

    if st.button("Send to backend for prediction"):
        if image_file is None:
            st.error("Please upload an image or take one with the webcam.")
        else:
            with st.spinner("Sending image to backend..."):
                # Prepare multipart form
                files = {"file": ("input.jpg", image_file.getvalue(), "image/jpeg")}
                data = {"model_key": model_key, "layers_csv": layers_input}
                try:
                    resp = requests.post(f"{API_URL}/predict", files=files, data=data, timeout=60)
                    resp.raise_for_status()
                except Exception as e:
                    st.error(f"Request failed: {e}")
                    st.stop()

                result = resp.json()
                st.success("Prediction received")

                # Show prediction & top_k
                st.subheader("Prediction")
                st.write("Label (index):", result.get("predicted_label"))
                st.write("Confidence:", f"{result.get('confidence')*100:.2f}%")
                st.write("Top predictions (index + score):")
                st.table(result.get("top_k"))

                # Show Grad-CAMs
                gcs = result.get("gradcam_images_b64", [])
                if not gcs:
                    st.info("No Grad-CAM images returned (layer names may be invalid for this model).")
                else:
                    st.subheader("Grad-CAM visualizations")
                    # show each in a grid
                    cols = st.columns(min(len(gcs), 4))
                    for i, item in enumerate(gcs):
                        layer = item["layer"]
                        b64 = item["image"]
                        img_bytes = base64.b64decode(b64)
                        img = Image.open(BytesIO(img_bytes))
                        cols[i % len(cols)].image(img, caption=f"Layer: {layer}", use_column_width=True)
