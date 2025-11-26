import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# model = load_model("mobilenet_model_1.h5")
model = load_model("second_best_mobilenet_model.h5")
print(model.summary())

last_conv_layer_name = "block_3_project_BN"

train_dir = r"archive/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train"

IMG_SIZE = (128, 128)
BATCH_SIZE = 16

datagen = ImageDataGenerator(
    rescale=1/255.0,
    validation_split=0.15,
    rotation_range=10,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

class_names = list(train_gen.class_indices.keys())
print("\nClasses loaded:", class_names, "\n")


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generate Grad-CAM heatmap with improved implementation
    """

    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        
        # Get the score for the predicted class
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    
    return heatmap.numpy()


def predict_with_gradcam(img_path, alpha=0.4):
    """
    Predict and visualize with Grad-CAM
    
    Args:
        img_path: Path to input image
        alpha: Transparency for heatmap overlay (0-1, lower = more transparent)
    """
    if not os.path.exists(img_path):
        print("❌ Error: File not found:", img_path)
        return None

    # Load original image
    original_img = cv2.imread(img_path)
    if original_img is None:
        print("❌ Error: Unable to load the image. Check format or path.")
        return None

    # Convert to RGB
    img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # Resize for model
    img_resized = cv2.resize(img_rgb, IMG_SIZE)
    
    # Preprocessing
    img_array = img_resized.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Get predictions
    preds = model.predict(img_array, verbose=0)
    pred_index = np.argmax(preds[0])
    confidence = float(preds[0][pred_index])

    print("\n" + "="*50)
    print(f"Prediction: {class_names[pred_index]}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print("="*50)
    
    # Show top 3 predictions
    top_3_idx = np.argsort(preds[0])[-3:][::-1]
    print("\nTop 3 Predictions:")
    for i, idx in enumerate(top_3_idx, 1):
        print(f"  {i}. {class_names[idx]}: {preds[0][idx]*100:.2f}%")
    print()

    # Generate Grad-CAM heatmap
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index)

    # Rescale heatmap to range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = plt.cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img_rgb.shape[1], img_rgb.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img_rgb
    superimposed_img = np.uint8(superimposed_img)

    # Also create overlay with OpenCV method for comparison
    heatmap_resized = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]))
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay_cv = cv2.addWeighted(img_rgb, 0.6, heatmap_colored, 0.4, 0)

    # Display results
    plt.figure(figsize=(16, 5))

    plt.subplot(1, 4, 1)
    plt.title("Original Image", fontsize=12, fontweight='bold')
    plt.imshow(img_rgb)
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.title("Grad-CAM Heatmap", fontsize=12, fontweight='bold')
    plt.imshow(heatmap, cmap='jet')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.title(f"Overlay (alpha={alpha})", fontsize=12, fontweight='bold')
    plt.imshow(superimposed_img)
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.title("Alternative Overlay", fontsize=12, fontweight='bold')
    plt.imshow(overlay_cv)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    return class_names[pred_index], confidence


def test_different_layers(img_path, layer_names):
    """
    Test Grad-CAM with different convolutional layers
    """
    print(f"\nTesting different layers for: {img_path}\n")
    
    for layer_name in layer_names:
        try:
            print(f"\n{'='*60}")
            print(f"Testing layer: {layer_name}")
            print(f"{'='*60}")
            
            global last_conv_layer_name
            last_conv_layer_name = layer_name
            
            predict_with_gradcam(img_path)
        except Exception as e:
            print(f"❌ Error with layer {layer_name}: {str(e)}")



# Single prediction with best layer
# predict_with_gradcam(r"archive/test/test/PotatoEarlyBlight2.JPG", alpha=0.4)
predict_with_gradcam(r"archive/test/test/PotatoEarlyBlight4.JPG", alpha=0.4)
# predict_with_gradcam(r"archive/test/test/TomatoEarlyBlight1.JPG")
# predict_with_gradcam(r"archive/test/test/AppleCedarRust3.JPG")

# test_layers = ["Conv_1", "block_16_project", "block_15_add", "block_14_add"]
# test_different_layers(r"archive/test/test/PotatoEarlyBlight2.JPG", test_layers)