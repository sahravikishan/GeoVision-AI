from pathlib import Path
import sys

import pandas as pd
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "best_model.pt"

sys.path.insert(0, str(MODELS_DIR))

from model import build_model  # noqa: E402


CLASS_LABELS = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]


st.set_page_config(
    page_title="GeoVision AI",
    page_icon="G",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    .hero {
        padding: 0.5rem 0 1.25rem 0;
        border-bottom: 1px solid #e5e7eb;
        margin-bottom: 1.5rem;
    }
    .hero h1 {
        margin: 0;
        font-size: 2.1rem;
        color: #111827;
        letter-spacing: -0.03em;
    }
    .hero p {
        margin: 0.35rem 0 0 0;
        color: #6b7280;
        font-size: 0.95rem;
    }
    .panel {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 1rem;
    }
    .result {
        background: #111827;
        color: #ffffff;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        margin-bottom: 1rem;
    }
    .result .label {
        color: #9ca3af;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.25rem;
    }
    .result .value {
        font-size: 1.7rem;
        font-weight: 700;
        margin: 0;
    }
    .empty {
        border: 1px dashed #d1d5db;
        border-radius: 14px;
        padding: 2.2rem 1.2rem;
        text-align: center;
        color: #6b7280;
        background: #f9fafb;
    }
    .footer {
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #e5e7eb;
        color: #6b7280;
        font-size: 0.85rem;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None, None

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    class_names = checkpoint.get("class_names", CLASS_LABELS)
    model = build_model(num_classes=len(class_names), use_pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_names


def preprocess_image(image: Image.Image) -> torch.Tensor:
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


def predict(image: Image.Image, model, class_names):
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        confidence, prediction_idx = torch.max(probabilities, dim=0)

    predicted_class = class_names[prediction_idx.item()]
    return predicted_class, float(confidence.item()), probabilities.cpu().numpy()


def main():
    st.markdown(
        """
        <div class="hero">
            <h1>GeoVision AI</h1>
            <p>Satellite image classification with EfficientNet-B0.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model, class_names = load_model()

    if model is None:
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()

    left, right = st.columns([5, 5], gap="large")

    with left:
        st.subheader("Upload")
        uploaded_file = st.file_uploader(
            "Upload a satellite image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption=uploaded_file.name, use_container_width=True)
            st.caption(f"{image.size[0]} x {image.size[1]} pixels")
        else:
            st.markdown(
                """
                <div class="empty">
                    Upload a EuroSAT-style satellite image to see a prediction.
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right:
        st.subheader("Prediction")
        if uploaded_file is not None:
            predicted_class, confidence, probabilities = predict(image, model, class_names)

            st.markdown(
                f"""
                <div class="result">
                    <div class="label">Predicted Class</div>
                    <div class="value">{predicted_class}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.metric("Confidence", f"{confidence * 100:.2f}%")
            st.progress(confidence)

            prob_df = pd.DataFrame(
                {
                    "Class": class_names,
                    "Probability": [float(p) for p in probabilities],
                }
            ).sort_values("Probability", ascending=False)

            st.subheader("Class Probabilities")
            st.bar_chart(prob_df.set_index("Class"))
        else:
            st.markdown(
                """
                <div class="empty">
                    The prediction panel will appear here after you upload an image.
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="footer">
            EfficientNet-B0 model trained on the EuroSAT dataset.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
