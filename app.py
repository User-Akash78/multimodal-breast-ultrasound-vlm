import streamlit as st
from PIL import Image

from src.inference import (
    load_models,
    run_full_pipeline
)


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Breast Ultrasound Analysis",
    page_icon="🩺",
    layout="wide"
)


# ---------------------------------------------------------
# Small UI styling
# ---------------------------------------------------------

st.markdown(
    """
    <style>
        /* Reduce excessive vertical spacing */
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1250px;
        }

        /* Keep result cards compact */
        div[data-testid="stMetric"] {
            padding: 0.5rem;
        }

        /* Make the generated report easier to read */
        .report-box {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: rgba(128, 128, 128, 0.08);
            line-height: 1.6;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# Load models once
# ---------------------------------------------------------

@st.cache_resource
def get_models():
    return load_models()


with st.spinner("Loading trained models..."):
    models = get_models()


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("Multimodal Breast Ultrasound Analysis")

st.warning(
    "Research prototype only — not intended for clinical diagnosis or treatment decisions."
)


# ---------------------------------------------------------
# Input section
# ---------------------------------------------------------

st.subheader("Case Input")

input_col1, input_col2 = st.columns(
    [1, 1.2],
    gap="large"
)

with input_col1:
    uploaded_file = st.file_uploader(
        "Upload breast ultrasound image",
        type=["png", "jpg", "jpeg"]
    )

    # Preview at a deliberately limited width.
    if uploaded_file is not None:
        preview_image = Image.open(
            uploaded_file
        ).convert("L")

        st.image(
            preview_image,
            caption="Uploaded ultrasound",
            width=320
        )

with input_col2:
    age = st.number_input(
        "Patient Age",
        min_value=18,
        max_value=120,
        value=50
    )

    tissue_composition = st.text_input(
        "Tissue Composition",
        placeholder="e.g. heterogeneous"
    )

    signs = st.text_input(
        "Clinical Signs",
        placeholder="e.g. palpable mass"
    )

    symptoms = st.text_input(
        "Symptoms",
        placeholder="e.g. pain"
    )

    analyse_button = st.button(
        "Analyse Case",
        type="primary",
        use_container_width=True
    )


# ---------------------------------------------------------
# Inference
# ---------------------------------------------------------

if analyse_button:

    if uploaded_file is None:
        st.error(
            "Please upload an ultrasound image before analysing the case."
        )

    else:
        image = Image.open(
            uploaded_file
        ).convert("L")

        with st.spinner(
            "Running segmentation, multimodal analysis, "
            "classification, and report generation..."
        ):
            result = run_full_pipeline(
                image=image,
                age=age,
                tissue_composition=tissue_composition,
                signs=signs,
                symptoms=symptoms,
                models=models
            )

        st.divider()

        st.subheader("Analysis Results")

        # -------------------------------------------------
        # Key prediction metrics
        # -------------------------------------------------

        metric_col1, metric_col2 = st.columns(2)

        with metric_col1:
            st.metric(
                "Classification",
                result["classification"].capitalize()
            )

        with metric_col2:
            st.metric(
                "Malignancy Probability",
                f"{result['malignancy_probability'] * 100:.1f}%"
            )

        # -------------------------------------------------
        # Compact image row
        # -------------------------------------------------

        st.markdown("#### Visual Analysis")

        image_col1, image_col2, image_col3 = st.columns(
            3,
            gap="medium"
        )

        with image_col1:
            st.image(
                image,
                caption="Original Ultrasound",
                width=280
            )

        with image_col2:
            st.image(
                result["predicted_mask"],
                caption="Predicted Lesion Mask",
                width=280,
                clamp=True
            )

        with image_col3:
            st.image(
                result["overlay"],
                caption="Lesion Overlay",
                width=280
            )

        # -------------------------------------------------
        # Generated report
        # -------------------------------------------------

        st.markdown("#### Generated Structured Report")

        st.markdown(
            f"""
            <div class="report-box">
                {result["report"]}
            </div>
            """,
            unsafe_allow_html=True
        )

        # Optional debugging/inspection information.
        with st.expander(
            "View ClinicalBERT input text"
        ):
            st.write(
                result["clinical_text"]
            )

        '''
        run the command- 
        python -m streamlit run app.py
        to run the streamlit website
        '''
