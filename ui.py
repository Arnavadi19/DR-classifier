"""Streamlit front-end for the DR Classification Lambda API."""
from typing import Optional

import requests
import streamlit as st
from PIL import Image

# --- CONFIG -----------------------------------------------------------------
API_URL = "https://tpfdx4w4z2j5yw2qk7cid7zs5q0jkplo.lambda-url.ap-south-1.on.aws/predict"
REQUEST_TIMEOUT = 120  # seconds

# --- UI SETUP ----------------------------------------------------------------
st.set_page_config(page_title="DR Classifier", layout="wide")
st.title("Diabetic Retinopathy (DR) Classifier 🩺")
st.write(
    "Upload a retinal fundus image to predict whether it shows no/mild DR or more than mild DR."
)

# Sidebar info
with st.sidebar:
    st.header("About")
    st.markdown(
        """
        This Streamlit dashboard sends images to a FastAPI service deployed on AWS Lambda.
        The response includes class probabilities, a confidence score, and a clinical
        interpretation matching the API output.
        """
    )
    st.markdown("**Cold start notice**: the first request after a period of inactivity may take up to 2 minutes")
    st.markdown("[API Health Check](https://tpfdx4w4z2j5yw2qk7cid7zs5q0jkplo.lambda-url.ap-south-1.on.aws/health)")

# --- HELPERS -----------------------------------------------------------------

def call_inference_api(file_name: str, file_bytes: bytes, mime_type: str) -> Optional[dict]:
    """Send the uploaded image to the inference API and return the parsed JSON."""
    files = {"file": (file_name, file_bytes, mime_type)}
    response = requests.post(API_URL, files=files, timeout=REQUEST_TIMEOUT)
    if response.status_code == 200:
        return response.json()

    st.error(f"API error {response.status_code}: {response.text}")
    return None


def render_result_card(result: dict) -> None:
    """Render the prediction block using Streamlit callouts."""
    prediction = result.get("prediction", "Unknown")
    confidence = result.get("confidence", 0.0)
    interpretation = result.get("interpretation", "")

    if "Positive" in prediction:
        st.error(f"**Result:** {interpretation}")
    else:
        st.success(f"**Result:** {interpretation}")

    st.metric("Confidence", f"{confidence * 100:.2f}%")

    class_probs = result.get("class_probabilities", {})
    st.subheader("Class Probabilities")
    st.write(
        {
            "Negative (No/Mild DR)": f"{class_probs.get('Negative', 0.0) * 100:.2f}%",
            "Positive (Moderate+ DR)": f"{class_probs.get('Positive', 0.0) * 100:.2f}%",
        }
    )


# --- MAIN INTERACTION --------------------------------------------------------

uploaded_file = st.file_uploader("Choose a fundus image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_column_width=True)
    st.caption(f"Filename: {uploaded_file.name}")

    if st.button("Classify Image", type="primary"):
        with st.spinner("Analyzing... please wait"):
            try:
                result = call_inference_api(
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type or "image/png",
                )
                if result:
                    render_result_card(result)
            except requests.exceptions.RequestException as exc:
                st.error(f"Failed to connect to the API: {exc}")
else:
    st.info("Upload a JPG or PNG fundus photograph to get started.")
