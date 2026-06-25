from pathlib import Path
import sys

import torch
import streamlit as st
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "best_model.pt"

sys.path.insert(0, str(MODELS_DIR))

from model import build_model  # noqa: E402


@st.cache_resource
def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    class_names = checkpoint["class_names"]
    model = build_model(num_classes=len(class_names), use_pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_names


def preprocess_image(image):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return transform(image.convert("RGB")).unsqueeze(0)


def main():
    st.set_page_config(page_title="GeoVision AI", page_icon="🛰️", layout="wide")
    st.title("GeoVision AI")
    st.write("Upload a satellite image and get a class prediction from the trained model.")

    if not MODEL_PATH.exists():
        st.error(f"Model file not found: {MODEL_PATH}")
        return

    model, class_names = load_model()

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", use_container_width=True)

        input_tensor = preprocess_image(image)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            confidence, prediction_idx = torch.max(probabilities, dim=0)

        predicted_class = class_names[prediction_idx.item()]

        col1, col2 = st.columns(2)
        col1.metric("Predicted class", predicted_class)
        col2.metric("Confidence", f"{confidence.item() * 100:.2f}%")

        st.subheader("All class probabilities")
        prob_data = {
            "Class": class_names,
            "Probability": [float(p) * 100 for p in probabilities.tolist()],
        }
        st.bar_chart(prob_data, x="Class", y="Probability")

        st.subheader("Map view")
        st.info("Map display can be added next. For now, the dashboard focuses on prediction.")


if __name__ == "__main__":
    main()
