# backend.py
import io
import os
import base64
from typing import List
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from keras.models import load_model

app = FastAPI(title="Plant Disease API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_REGISTRY = {
    "mobilenet_v1": "mobilenet_model_1.h5",
}

DEFAULT_LAYERS = ["Conv_1", "block_16_project", "block_15_add", "block_3_project_BN"]

_loaded_models = {}

def load_or_get_model(key: str):
    if key in _loaded_models:
        return _loaded_models[key]
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Model '{key}' not registered on the backend.")
    path = MODEL_REGISTRY[key]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    model = load_model(path)
    _loaded_models[key] = model
    return model

def image_to_base64_png(cv_rgb_img):
    img_bgr = cv2.cvtColor(cv_rgb_img, cv2.COLOR_RGB2BGR)
    success, buffer = cv2.imencode(".png", img_bgr)
    if not success:
        raise RuntimeError("Could not encode image to PNG")
    b64 = base64.b64encode(buffer).decode("utf-8")
    return b64

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    heatmap = heatmap.numpy()
    return heatmap

def prepare_image_for_model(pil_img: Image.Image, target_size=(128,128)):
    pil_img = pil_img.convert("RGB")
    pil_img_resized = pil_img.resize(target_size)
    arr = np.array(pil_img_resized).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)  # batch dim
    return arr

class PredictResponse(BaseModel):
    predicted_label: str
    confidence: float
    top_k: List[dict]
    gradcam_images_b64: List[dict]

@app.post("/predict", response_model=PredictResponse)
async def predict(
    model_key: str = Form(...),
    layers_csv: str = Form(""),
    file: UploadFile = File(...),
):
    """
    Accepts multipart/form-data with:
    - model_key: key of model in MODEL_REGISTRY
    - layers_csv: comma separated conv layer names for Grad-CAM (optional)
    - file: uploaded image
    """
    model = load_or_get_model(model_key)

    if layers_csv.strip():
        layer_names = [n.strip() for n in layers_csv.split(",") if n.strip()]
    else:
        layer_names = DEFAULT_LAYERS

    contents = await file.read()
    pil_img = Image.open(io.BytesIO(contents))
    orig_np = np.array(pil_img.convert("RGB"))

    input_size = (128, 128)
    img_array = prepare_image_for_model(pil_img, target_size=input_size)

    # Predict
    preds = model.predict(img_array)
    pred_index = int(np.argmax(preds[0]))
    confidence = float(np.max(preds[0]))

    label = str(pred_index)
    topk = []
    top_idx = np.argsort(preds[0])[-5:][::-1]
    for idx in top_idx:
        topk.append({"index": int(idx), "score": float(preds[0][idx])})

    gradcam_images = []
    for layer_name in layer_names:
        try:
            # sanity: ensure layer exists
            _ = model.get_layer(layer_name)
        except Exception:
            # layer not present; skip
            continue

        heatmap = make_gradcam_heatmap(img_array, model, layer_name, pred_index)
        heatmap_resized = cv2.resize(heatmap, (orig_np.shape[1], orig_np.shape[0]))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)  # now RGB

        # Overlay heatmap onto original (blend)
        overlay = cv2.addWeighted(orig_np.astype("uint8"), 0.6, heatmap_colored, 0.4, 0)

        # Convert to base64
        img_b64 = image_to_base64_png(overlay)
        gradcam_images.append({"layer": layer_name, "image": img_b64})

    response = PredictResponse(
        predicted_label=label,
        confidence=confidence,
        top_k=topk,
        gradcam_images_b64=gradcam_images,
    )
    return response

# Simple ping
@app.get("/ping")
def ping():
    return {"status": "ok"}
