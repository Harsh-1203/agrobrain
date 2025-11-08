import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from typing import Tuple, List
from PIL import Image  # PIL is safe to import, needed for image display

# ML
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# PyTorch will be imported lazily only when needed to avoid crashes
TORCH_AVAILABLE = None
torch = None
nn = None
transforms = None


# -----------------------------
# Paths and helpers
# -----------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DISEASE_DIR = os.path.join(APP_DIR, "disese detctor")
MODEL_TRAINING_DIR = os.path.join(APP_DIR, "model_traning")
DISEASE_MODEL_PATH = os.path.join(MODEL_TRAINING_DIR, "plant_disease_model.pt")
DISEASE_LABELS_PATH = os.path.join(MODEL_TRAINING_DIR, "disease_labels.json")


# -----------------------------
# Caching: Load models & data
# -----------------------------
 


@st.cache_data(show_spinner=False)
def load_crop_dataset() -> pd.DataFrame:
    csv_path = os.path.join(MODEL_TRAINING_DIR, "Crop_recommendation.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError("Crop_recommendation.csv not found in model_traning.")
    return pd.read_csv(csv_path)


@st.cache_resource(show_spinner=False)
def train_crop_model(df: pd.DataFrame) -> Tuple[Pipeline, List[str]]:
    feature_columns = [
        "N", "P", "K", "temperature", "humidity", "ph", "rainfall"
    ]
    target_column = "label"

    X = df[feature_columns]
    y = df[target_column]

    # Simple and fast model with scaling → RF works fine without scaling, but keep a consistent pipeline
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=300, random_state=42)),
        ]
    )

    pipeline.fit(X, y)
    return pipeline, feature_columns


@st.cache_resource(show_spinner=False)
def build_shap_explainer(_rf_model):
    # For tree models, TreeExplainer is efficient
    import shap  # lazy import to avoid slowing initial app load
    explainer = shap.TreeExplainer(_rf_model)
    return explainer


def _load_disease_labels() -> List[str]:
    """Load disease class labels saved during training."""
    if os.path.exists(DISEASE_LABELS_PATH):
        try:
            with open(DISEASE_LABELS_PATH, "r", encoding="utf-8") as f:
                labels = json.load(f)
            if isinstance(labels, list) and all(isinstance(lbl, str) for lbl in labels):
                return labels
        except Exception as exc:
            st.warning(f"Could not read disease_labels.json: {exc}")
    # Fallback to legacy 3-class setup
    return [
        "Corn_(maize)___Common_rust_",
        "Potato___Early_blight",
        "Tomato___Bacterial_spot",
    ]


def _format_label(label: str) -> str:
    """Convert raw folder label to a human readable string."""
    label = label.replace("_", " ").replace("___", " – ")
    label = label.replace("  ", " ")
    return label.strip()


def _lazy_import_torch():
    """Lazily import PyTorch only when needed to avoid startup penalties."""
    global TORCH_AVAILABLE, torch, nn, transforms
    if TORCH_AVAILABLE is not None:
        return TORCH_AVAILABLE

    try:
        import torch
        import torch.nn as nn
        from torchvision import transforms

        # Cache modules in globals so other functions can use them
        globals()["torch"] = torch
        globals()["nn"] = nn
        globals()["transforms"] = transforms

        TORCH_AVAILABLE = True
        return True
    except Exception as exc:
        TORCH_AVAILABLE = False
        st.warning(f"PyTorch could not be loaded: {exc}")
        return False


@st.cache_resource(show_spinner=False)
def load_disease_model():
    """Load the plant disease detection model. Creates model if not found."""
    if not _lazy_import_torch():
        return None, [], False

    disease_labels = _load_disease_labels()
    if not disease_labels:
        st.error("No disease labels found. Train the model and generate disease_labels.json.")
        return None, [], False

    class PlantDiseaseCNN(nn.Module):
        def __init__(self, num_classes: int):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes),
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = PlantDiseaseCNN(len(disease_labels)).to(device)
    model_trained = False

    if os.path.exists(DISEASE_MODEL_PATH):
        try:
            state_dict = torch.load(
                DISEASE_MODEL_PATH,
                map_location=device,
                weights_only=True,
            )
            model.load_state_dict(state_dict)
            model_trained = True
        except Exception as exc:
            st.warning(f"Could not load saved model: {exc}. Using untrained model structure.")

    model.eval()
    return model, disease_labels, model_trained


def preprocess_image_for_disease(image_file, target_size=(256, 256)):
    """Preprocess uploaded image for disease detection model."""
    if not _lazy_import_torch():
        return None, None

    try:
        preprocess = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        image = Image.open(image_file).convert("RGB")
        tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
        return tensor, image
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None, None


# -----------------------------
# UI Components
# -----------------------------
def sidebar_menu() -> str:
    st.sidebar.title("Smart Agri Dashboard")
    return st.sidebar.selectbox(
        "Select Page",
        [
            "Home",
            "Plant Disease Detection",
            "Crop Recommendation + SHAP",
            "About",
        ],
    )


def page_home():
    st.header("Smart Agriculture Assistant 🌱")
    st.image(os.path.join("home_page.jpeg"), width="stretch")
    st.markdown(
        """
        Use the sidebar to switch between:
        - **Plant Disease Detection**: Upload an image to detect plant diseases using CNN
        - **Crop Recommendation + SHAP**: Get crop recommendations with explainable AI
        """
    )


def page_about():
    st.header("About")
    st.markdown(
        """
        This dashboard combines two tools:
        - A CNN-based Plant Disease Detector powered by PyTorch
        - A classical ML Crop Recommendation model with SHAP explanations
        """
    )


def page_disease_detection():
    st.header("Plant Disease Detection 🔬")

    # Try to import PyTorch lazily
    if not _lazy_import_torch():
        st.error("⚠️ PyTorch is not installed or could not be loaded. Please install it to use disease detection.")
        st.code("pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu", language="bash")
        st.info("On Apple Silicon with Metal support you can also try: pip install torch torchvision torchaudio")
        return

    # Load model
    model, disease_labels, model_trained = load_disease_model()

    if model is None:
        st.error("⚠️ Could not load disease detection model.")
        return
    
    display_labels = [_format_label(lbl) for lbl in disease_labels]
    
    if not model_trained:
        st.warning("⚠️ **Model not trained yet.** The model architecture is loaded but needs to be trained.")
        st.info("""
        To train the model:
        1. Use the `Plant_Disease_Detection.ipynb` notebook in the `model_traning` folder
        2. Train the model in PyTorch and save it as `plant_disease_model.pt` and `disease_labels.json` in the `model_traning` folder
        3. The model will be automatically loaded when available
        """)

    st.markdown("""
    ### Upload Plant Image
    Upload an image of a plant leaf to detect potential diseases.
    """)
    if disease_labels:
        st.caption(
            "Supported diseases (showing up to 12): "
            + ", ".join(display_labels[:12])
            + ("…" if len(display_labels) > 12 else "")
        )

    # Image upload
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a plant leaf image (JPG, JPEG, or PNG format)"
    )

    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", width="stretch")

        with col2:
            st.subheader("Prediction Results")

            if model_trained:
                # Preprocess image
                tensor, processed_img = preprocess_image_for_disease(uploaded_file)

                if tensor is not None:
                    # Make prediction
                    with st.spinner("Analyzing image..."):
                        device = next(model.parameters()).device
                        tensor = tensor.to(device)
                        with torch.no_grad():
                            logits = model(tensor)
                            probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]

                        predicted_class_idx = int(np.argmax(probabilities))
                        confidence = probabilities[predicted_class_idx]

                        # Display results
                        st.success(f"**Predicted Disease:** {display_labels[predicted_class_idx]}")
                        st.metric("Confidence", f"{confidence * 100:.2f}%")

                        # Show all predictions
                        st.markdown("**All Predictions:**")
                        prediction_df = (
                            pd.DataFrame(
                                {
                                    "Disease": display_labels,
                                    "Probability (%)": probabilities * 100,
                                }
                            )
                            .sort_values(by="Probability (%)", ascending=False)
                            .reset_index(drop=True)
                        )
                        st.dataframe(
                            prediction_df.style.format({"Probability (%)": "{:.2f}"}),
                            width="stretch",
                        )

                        # Visualize predictions
                        fig, ax = plt.subplots(figsize=(8, 5))
                        colors = ['#2ecc71' if i == predicted_class_idx else '#95a5a6'
                                 for i in range(len(display_labels))]
                        ax.barh(display_labels, probabilities * 100, color=colors)
                        ax.set_xlabel('Probability (%)')
                        ax.set_title('Disease Detection Probabilities')
                        ax.set_xlim(0, 100)
                        for i, (label, prob) in enumerate(zip(display_labels, probabilities)):
                            ax.text(prob * 100 + 1, i, f'{prob*100:.1f}%',
                                   va='center', fontweight='bold' if i == predicted_class_idx else 'normal')
                        st.pyplot(fig, clear_figure=True)
            else:
                st.info("Please train the model first using the notebook before making predictions.")



 


def page_crop_recommendation_shap():
    st.header("Crop Recommendation + SHAP")

    df = load_crop_dataset()
    model, feature_cols = train_crop_model(df)

    # Inputs
    st.subheader("Input Features")
    col1, col2 = st.columns(2)
    with col1:
        N = st.number_input("Nitrogen (N)", min_value=0.0, max_value=200.0, value=50.0, step=1.0)
        P = st.number_input("Phosphorus (P)", min_value=0.0, max_value=200.0, value=50.0, step=1.0)
        K = st.number_input("Potassium (K)", min_value=0.0, max_value=200.0, value=50.0, step=1.0)
        ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
    with col2:
        temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=60.0, value=25.0, step=0.1)
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0, step=0.5)
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=1000.0, value=100.0, step=1.0)

    user_row = pd.DataFrame(
        [{
            "N": N,
            "P": P,
            "K": K,
            "temperature": temperature,
            "humidity": humidity,
            "ph": ph,
            "rainfall": rainfall,
        }],
        columns=feature_cols,
    )

    if st.button("Recommend Crop"):
        with st.spinner("Scoring and explaining..."):
            # Prediction & probabilities
            proba = model.predict_proba(user_row[feature_cols])[0]
            classes = model.named_steps["clf"].classes_
            top_idx = np.argsort(proba)[::-1][:5]

            st.subheader("Top Recommendations")
            for rank, i in enumerate(top_idx, start=1):
                st.write(f"{rank}. {classes[i]} — {proba[i]*100:.2f}%")

            # SHAP: use the same feature space as the trained model (scaled)
            scaler = model.named_steps["scaler"]
            rf_model = model.named_steps["clf"]
            background_df = df[feature_cols].sample(n=min(200, len(df)), random_state=42)
            background_scaled = scaler.transform(background_df)
            user_scaled = scaler.transform(user_row[feature_cols])

            explainer = build_shap_explainer(rf_model)
            # Lazy import shap right before use
            import shap
            shap_values = explainer.shap_values(user_scaled)

            # Find index for predicted class
            pred_idx = int(np.argmax(proba))
            # Support both list-of-classes and single-array outputs
            if isinstance(shap_values, list):
                shap_for_pred = shap_values[pred_idx][0]
            else:
                shap_for_pred = shap_values[0]
            # Ensure numpy arrays for advanced indexing
            shap_for_pred = np.asarray(shap_for_pred, dtype=float)

            st.subheader("Why this recommendation?")
            fig, ax = plt.subplots(figsize=(7, 4))
            shap_for_pred = np.asarray(shap_for_pred).ravel()
            # Pair feature names with SHAP values and sort by absolute impact
            pairs = list(zip(feature_cols, shap_for_pred))
            pairs_sorted = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)
            ordered_names = [name for name, _ in pairs_sorted]
            ordered_vals = np.array([val for _, val in pairs_sorted], dtype=float)
            ax.bar(ordered_names, ordered_vals, color=["#1f77b4" if v >= 0 else "#d62728" for v in ordered_vals])
            ax.set_ylabel("SHAP value (impact)")
            ax.set_title(f"Feature impact for predicted crop: {classes[pred_idx]}")
            plt.xticks(rotation=30, ha="right")
            st.pyplot(fig, clear_figure=True)

            # Global importance
            st.subheader("Global Feature Importance (mean |SHAP|)")
            # Compute on a small background set for speed (scaled)
            bg_vals = explainer.shap_values(background_scaled)
            # For multiclass, average absolute SHAP across classes then across samples
            if isinstance(bg_vals, list):
                mean_abs = np.mean([np.mean(np.abs(v), axis=0) for v in bg_vals], axis=0)
            else:
                mean_abs = np.mean(np.abs(bg_vals), axis=0)
            mean_abs = np.asarray(mean_abs).ravel()
            pairs_g = list(zip(feature_cols, mean_abs))
            pairs_g_sorted = sorted(pairs_g, key=lambda x: abs(x[1]), reverse=True)
            names_g = [n for n, _ in pairs_g_sorted]
            vals_g = np.array([v for _, v in pairs_g_sorted], dtype=float)
            fig2, ax2 = plt.subplots(figsize=(7, 4))
            ax2.bar(names_g, vals_g, color="#6baed6")
            ax2.set_ylabel("mean |SHAP value|")
            ax2.set_title("Global Feature Importance")
            plt.xticks(rotation=30, ha="right")
            st.pyplot(fig2, clear_figure=True)


# -----------------------------
# App Entrypoint
# -----------------------------
def main():
    st.set_page_config(page_title="Smart Agri Dashboard", layout="wide")
    page = sidebar_menu()
    if page == "Home":
        page_home()
    elif page == "Plant Disease Detection":
        page_disease_detection()
    elif page == "Crop Recommendation + SHAP":
        page_crop_recommendation_shap()
    elif page == "About":
        page_about()


if __name__ == "__main__":
    main()


