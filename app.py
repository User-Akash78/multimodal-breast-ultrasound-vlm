from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.inference import (
    load_models,
    run_full_pipeline
)


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
VAL_FILE = PROJECT_ROOT / "data" / "processed" / "val.csv"
IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "images"


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
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1250px;
        }

        div[data-testid="stMetric"] {
            padding: 0.5rem;
        }

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
# Helpers
# ---------------------------------------------------------

def clean_value(value):
    if pd.isna(value):
        return "not available"

    value = str(value).strip()

    if value == "":
        return "not available"

    return value


def format_case_id(value):
    """Display integer-like CaseIDs without a trailing .0."""
    if pd.isna(value):
        return "Unknown"

    try:
        number = float(value)
        if number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass

    return str(value)


def normalize_yes_no(value):
    value = clean_value(value).lower()

    if value == "yes":
        return "present"

    if value == "no":
        return "absent"

    return value


def normalize_posterior_features(value):
    value = clean_value(value)

    if value.lower() == "no":
        return "none"

    return value


def create_reference_report(row):
    """Create the same structured target report used during GPT-2 training."""
    return (
        "Findings: "
        f"Tissue composition: {clean_value(row['Tissue_composition']).replace('&', ', ')}. "
        f"Shape: {clean_value(row['Shape']).replace('&', ', ')}. "
        f"Margin: {clean_value(row['Margin']).replace('&', ', ')}. "
        f"Echogenicity: {clean_value(row['Echogenicity']).replace('&', ', ')}. "
        f"Posterior features: {normalize_posterior_features(row['Posterior_features']).replace('&', ', ')}. "
        f"Halo: {normalize_yes_no(row['Halo'])}. "
        f"Calcifications: {normalize_yes_no(row['Calcifications'])}. "
        f"Skin thickening: {normalize_yes_no(row['Skin_thickening'])}. "
        f"Assessment: BI-RADS {clean_value(row['BIRADS'])}. "
        f"Final classification: {clean_value(row['Classification'])}."
    )


def load_ground_truth_mask(row):
    """
    Load the expert tumour mask for a validation case.
    If multiple tumour masks are listed, combine them into one binary mask.
    """
    value = clean_value(row["Mask_tumor_filename"])

    if value == "not available":
        return None

    mask_files = [
        filename.strip()
        for filename in value.split("&")
        if filename.strip()
    ]

    combined_mask = None

    for filename in mask_files:
        mask_path = IMAGE_DIR / filename

        if not mask_path.exists():
            continue

        mask = Image.open(mask_path).convert("L")
        mask_array = np.array(mask) > 0

        if combined_mask is None:
            combined_mask = mask_array
        else:
            combined_mask = np.logical_or(
                combined_mask,
                mask_array
            )

    if combined_mask is None:
        return None

    return Image.fromarray(
        (combined_mask.astype(np.uint8) * 255)
    )


@st.cache_data
def load_validation_data():
    return pd.read_csv(VAL_FILE)


@st.cache_resource
def get_models():
    return load_models()


# ---------------------------------------------------------
# Load models
# ---------------------------------------------------------

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
# Input mode
# ---------------------------------------------------------

st.subheader("Case Input")

input_mode = st.radio(
    "Choose input source",
    ["Validation dataset", "Manual input"],
    horizontal=True
)

image = None
age = None
tissue_composition = None
signs = None
symptoms = None
selected_row = None
expected_report = None
ground_truth_mask = None


# ---------------------------------------------------------
# Validation dataset input
# ---------------------------------------------------------

if input_mode == "Validation dataset":

    if not VAL_FILE.exists():
        st.error(
            f"Validation file not found: {VAL_FILE}"
        )
        st.stop()

    if not IMAGE_DIR.exists():
        st.error(
            f"Image directory not found: {IMAGE_DIR}"
        )
        st.stop()

    val_df = load_validation_data()

    required_columns = [
        "CaseID",
        "Image_filename",
        "Mask_tumor_filename",
        "Age",
        "Tissue_composition",
        "Signs",
        "Symptoms",
        "Shape",
        "Margin",
        "Echogenicity",
        "Posterior_features",
        "Halo",
        "Calcifications",
        "Skin_thickening",
        "BIRADS",
        "Classification"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in val_df.columns
    ]

    if missing_columns:
        st.error(
            "The validation CSV is missing required columns: "
            + ", ".join(missing_columns)
        )
        st.stop()

    case_options = list(val_df.index)

    selected_index = st.selectbox(
        "Select Validation Case",
        options=case_options,
        format_func=lambda idx: (
            f"Case {format_case_id(val_df.loc[idx, 'CaseID'])}"
        )
    )

    selected_row = val_df.loc[selected_index]

    expected_report = create_reference_report(
        selected_row
    )

    ground_truth_mask = load_ground_truth_mask(
        selected_row
    )

    image_path = (
        IMAGE_DIR
        / str(selected_row["Image_filename"])
    )

    input_col1, input_col2 = st.columns(
        [1, 1.2],
        gap="large"
    )

    with input_col1:
        if image_path.exists():
            image = Image.open(
                image_path
            ).convert("L")

            st.image(
                image,
                caption=(
                    f"Case "
                    f"{format_case_id(selected_row['CaseID'])}"
                ),
                width=320
            )

            st.caption(
                f"Image file: {selected_row['Image_filename']}"
            )

            if ground_truth_mask is None:
                st.warning(
                    "Ground-truth tumour mask could not be loaded for this case."
                )

        else:
            st.error(
                f"Image not found: {image_path}"
            )

    with input_col2:
        age_value = selected_row["Age"]

        if pd.isna(age_value):
            age = 50
            age_display = "not available"
        else:
            age = int(float(age_value))
            age_display = str(age)

        tissue_composition = clean_value(
            selected_row["Tissue_composition"]
        )
        signs = clean_value(
            selected_row["Signs"]
        )
        symptoms = clean_value(
            selected_row["Symptoms"]
        )

        st.text_input(
            "Patient Age",
            value=age_display,
            disabled=True
        )

        st.text_input(
            "Tissue Composition",
            value=tissue_composition,
            disabled=True
        )

        st.text_input(
            "Clinical Signs",
            value=signs,
            disabled=True
        )

        st.text_input(
            "Symptoms",
            value=symptoms,
            disabled=True
        )

        analyse_button = st.button(
            "Analyse Case",
            type="primary",
            use_container_width=True
        )


# ---------------------------------------------------------
# Manual input
# ---------------------------------------------------------

else:

    input_col1, input_col2 = st.columns(
        [1, 1.2],
        gap="large"
    )

    with input_col1:
        uploaded_file = st.file_uploader(
            "Upload breast ultrasound image",
            type=["png", "jpg", "jpeg"]
        )

        if uploaded_file is not None:
            image = Image.open(
                uploaded_file
            ).convert("L")

            st.image(
                image,
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

        tissue_composition = st.selectbox(
            "Tissue Composition",
            [
                "heterogeneous: predominantly fat",
                "heterogeneous: predominantly fibroglandular",
                "homogeneous: fat",
                "homogeneous: fibroglandular",
                "not available",
            ],
            index=4
        )

        signs = st.selectbox(
            "Clinical Signs",
            [
                "no",
                "palpable",
                "skin retraction",
                "skin retraction&palpable",
                "breast scar",
                "not available"
            ],
            index=5
        )

        symptoms = st.selectbox(
            "Symptoms",
            [
                "no",
                "family history of breast/ovarian cancer",
                "HRT/hormonal contraception",
                "nipple discharge",
                "tenderness",
                "not available"
            ],
            index=5
        )
    
        analyse_button = st.button(
            "Analyse Case",
            type="primary",
            use_container_width=True
        )


# ---------------------------------------------------------
# For Free Text Input (If Needed))
# ---------------------------------------------------------
    # with input_col2:
    #     age = st.number_input(
    #         "Patient Age",
    #         min_value=18,
    #         max_value=120,
    #         value=50
    #     )

    #     tissue_composition = st.text_input(
    #         "Tissue Composition",
    #         placeholder="e.g. heterogeneous"
    #     )

    #     signs = st.text_input(
    #         "Clinical Signs",
    #         placeholder="e.g. palpable mass"
    #     )

    #     symptoms = st.text_input(
    #         "Symptoms",
    #         placeholder="e.g. pain"
    #     )

    #     analyse_button = st.button(
    #         "Analyse Case",
    #         type="primary",
    #         use_container_width=True
    #     )





# ---------------------------------------------------------
# Inference
# ---------------------------------------------------------

if analyse_button:

    if image is None:
        st.error(
            "Please provide a valid ultrasound image before analysing the case."
        )

    else:
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
        # Visual analysis
        # -------------------------------------------------

        st.markdown("#### Visual Analysis")

        if (
            input_mode == "Validation dataset"
            and ground_truth_mask is not None
        ):
            image_col1, image_col2, image_col3, image_col4 = st.columns(
                4,
                gap="medium"
            )

            with image_col1:
                st.image(
                    image,
                    caption="Original Ultrasound",
                    use_container_width=True
                )

            with image_col2:
                st.image(
                    ground_truth_mask,
                    caption="Ground Truth Mask",
                    use_container_width=True
                )

            with image_col3:
                st.image(
                    result["predicted_mask"],
                    caption="Predicted Lesion Mask",
                    use_container_width=True,
                    clamp=True
                )

            with image_col4:
                st.image(
                    result["overlay"],
                    caption="Predicted Lesion Overlay",
                    use_container_width=True
                )

        else:
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

        st.markdown("#### Structured Report")

        if (
            input_mode == "Validation dataset"
            and expected_report is not None
        ):
            report_col1, report_col2 = st.columns(
                2,
                gap="large"
            )

            with report_col1:
                st.markdown("##### Generated Report")
                st.markdown(
                    f"""
                    <div class="report-box">
                        {result["report"]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with report_col2:
                st.markdown("##### Expected Target Report")
                st.markdown(
                    f"""
                    <div class="report-box">
                        {expected_report}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        else:
            st.markdown("##### Generated Report")
            st.markdown(
                f"""
                <div class="report-box">
                    {result["report"]}
                </div>
                """,
                unsafe_allow_html=True
            )

        # Show the true validation label only after inference.
        if (
            input_mode == "Validation dataset"
            and selected_row is not None
            and "Classification" in selected_row.index
        ):
            st.caption(
                "Validation ground truth: "
                f"{clean_value(selected_row['Classification']).capitalize()}"
            )

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
