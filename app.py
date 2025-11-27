import streamlit as st
import requests
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# ================================
#  BACKEND API URL
# ================================
API_URL = "http://127.0.0.1:8000"

# ================================
#  PAGE CONFIG
# ================================
st.set_page_config(page_title="Plant Disease Detector", layout="wide")

# ================================
#  SIDEBAR MENU (UPDATED)
# ================================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["MobileNetV2 (API)", "Custom CNN"])


# ==========================================================
#            PAGE 1 → MobileNetV2 (Backend API)
# ==========================================================
if page == "MobileNetV2 (API)":
    st.title("🌿 Plant Disease Detection — MobileNetV2 (FastAPI + Grad-CAM)")

    models = {"MobileNetV2 (mobilenet_v1)": "mobilenet_v1"}
    model_display = st.selectbox("Choose model", options=list(models.keys()))
    model_key = models[model_display]

    layers_input = st.text_input("Grad-CAM Layers (optional):", value="")

    col1, col2 = st.columns(2)

    with col1:
        uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        webcam_img = st.camera_input("Or take a photo")
        image_file = webcam_img if webcam_img else uploaded

    with col2:
        if st.button("Predict using API"):
            if image_file is None:
                st.error("Upload or capture an image.")
            else:
                files = {"file": ("input.jpg", image_file.getvalue(), "image/jpeg")}
                data = {"model_key": model_key, "layers_csv": layers_input}

                resp = requests.post(f"{API_URL}/predict", files=files, data=data)
                resp.raise_for_status()
                result = resp.json()

                st.success("Prediction received!")
                st.write("Label:", result["predicted_label"])
                st.write("Confidence:", f"{result['confidence']*100:.2f}%")
                st.table(result["top_k"])

                st.subheader("Grad-CAM Visualizations")
                cols = st.columns(4)
                for i, item in enumerate(result["gradcam_images_b64"]):
                    img = Image.open(BytesIO(base64.b64decode(item["image"])))
                    cols[i % 4].image(img, caption=item["layer"], use_column_width=True)



# ==========================================================
#            PAGE 2 → Custom CNN (Local Model)
# ==========================================================
if page == "Custom CNN":
    st.title(" Plant Disease Detection — Custom CNN")

    @st.cache_resource
    def load_custom_cnn():
        return tf.keras.models.load_model("final_best_model.h5")

    model = load_custom_cnn()

    class_labels = ["Healthy", "Powdery", "Rust"]

    color_map = {
        "Healthy": "#2ecc71",   # green
        "Powdery": "#f1c40f",   # yellow
        "Rust": "#e74c3c"       # red
    }

    suggestions = {
        "Healthy": [
            "Maintain watering and sunlight.",
            "Check leaves weekly.",
            "Use organic fertilizers."
        ],
        "Powdery": [
            "Remove infected leaves.",
            "Spray neem oil/fungicide.",
            "Increase plant spacing."
        ],
        "Rust": [
            "Trim rust-affected areas.",
            "Use sulfur fungicide.",
            "Avoid overhead watering."
        ]
    }

    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded Image", width=300)

        img_resized = img.resize((225, 225))
        arr = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

        if st.button("Predict with Custom CNN"):
            preds = model.predict(arr)[0]
            pred_index = np.argmax(preds)
            confidence = preds[pred_index]
            predicted_class = class_labels[pred_index]

            st.subheader("Prediction Result")
            st.write(f"**Leaf Type:** {predicted_class}")
            st.write(f"**Confidence:** {confidence*100:.2f}%")

            # ===============================
            # Horizontal Confidence Bars
            # ===============================
            st.subheader("Confidence Breakdown")

            for label, prob in zip(class_labels, preds):
                color = color_map[label]
                st.write(f"**{label}: {prob*100:.2f}%**")

                st.markdown(
                    f"""
                    <div style="
                        width: 100%;
                        background-color: #eaeaea;
                        border-radius: 6px;
                        height: 22px;
                        margin-bottom: 10px;">
                        <div style="
                            width: {prob*100}%;
                            background-color: {color};
                            height: 100%;
                            border-radius: 6px;">
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # ===============================
            # Additional Insights
            # ===============================
            st.subheader("Final Prediction")
            st.write(f"This is a **{predicted_class}** plant leaf.")

            st.markdown("###  Recommended Actions for Farmers")
            for tip in suggestions[predicted_class]:
                st.write(f"- {tip}")

            # ===============================
            # Probability Chart
            # ===============================
            st.subheader("Prediction Probability Chart")
            fig, ax = plt.subplots()
            ax.barh(class_labels, preds, color=[color_map[c] for c in class_labels])
            ax.set_xlabel("Probability")
            st.pyplot(fig)
