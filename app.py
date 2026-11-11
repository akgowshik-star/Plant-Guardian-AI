import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

st.set_page_config(page_title="Plant Guardian AI", page_icon="🌱", layout="centered")
st.title("🌱 Plant Guardian AI")
st.write("Upload a clear plant-leaf photo to check it with your trained image-classification model.")

MODEL_PATH = Path("plant_guardian_model.keras")
LABELS_PATH = Path("class_names.txt")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

def load_labels():
    return [line.strip() for line in LABELS_PATH.read_text().splitlines() if line.strip()]

uploaded = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])

if not MODEL_PATH.exists() or not LABELS_PATH.exists():
    st.info("The app is ready, but the AI model has not been trained yet. Follow README.md to add the dataset and train it.")
elif uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded leaf image", use_container_width=True)
    if st.button("Analyse plant"):
        model = load_model()
        labels = load_labels()
        resized = image.resize((224, 224))
        arr = np.array(resized, dtype=np.float32)
        arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
        prediction = model.predict(np.expand_dims(arr, axis=0), verbose=0)[0]
        idx = int(np.argmax(prediction))
        label = labels[idx] if idx < len(labels) else f"Class {idx}"
        score = float(prediction[idx]) * 100

        st.subheader("Model result")
        st.write(f"**Predicted class:** {label}")
        st.write(f"**Model confidence:** {score:.1f}%")
        st.warning("This is an AI screening result, not a confirmed plant diagnosis. Check the plant carefully and consult a local agriculture expert if the plant is worsening.")

        lowered = label.lower()
        if "healthy" in lowered:
            st.success("Care tip: Continue regular watering suited to the plant, provide appropriate sunlight, and monitor new leaves.")
        else:
            st.write("General care tips:")
            st.write("- Avoid wetting leaves unnecessarily.")
            st.write("- Remove fallen plant debris using clean tools.")
            st.write("- Check nearby plants for similar symptoms.")
            st.write("- Confirm the result before applying any treatment.")
else:
    st.caption("For best results, use a well-lit photo with one leaf clearly visible.")
