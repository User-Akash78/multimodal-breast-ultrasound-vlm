"""
Inference utilities for the multimodal breast ultrasound analysis system.

Pipeline
--------
1. U-Net predicts a lesion mask from the uploaded ultrasound image.
2. EfficientNet-B0 extracts lesion-guided visual tokens.
3. ClinicalBERT encodes the structured clinical information as text.
4. Bidirectional cross-attention fuses visual and clinical tokens.
5. The multimodal classifier predicts benign vs malignant.
6. GPT-2 generates the structured report.
7. The classifier prediction is appended as the final classification.

Designed to be imported by a local Streamlit app.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image

import torch
from transformers import AutoTokenizer, GPT2Tokenizer

from src.unet import UNet
from src.image_encoder import EfficientNetEncoder
from src.clinical_encoder import ClinicalBERTEncoder
from src.fusion import CrossAttentionFusion
from src.classifier import MultimodalClassifier
from src.report_generator import MultimodalReportGenerator


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"

IMAGE_SIZE = 224
CLINICAL_MAX_LENGTH = 64

CLINICAL_MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
GPT2_MODEL_NAME = "gpt2"

UNET_PATH = MODEL_DIR / "best_unet.pth"
MULTIMODAL_PATH = MODEL_DIR / "best_multimodal_classifier.pth"
REPORT_PATH = MODEL_DIR / "best_report_generator.pth"
SEGMENTATION_CONFIG_PATH = MODEL_DIR / "segmentation_config.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = torch.tensor(
    [0.485, 0.456, 0.406], dtype=torch.float32
).view(1, 3, 1, 1)

IMAGENET_STD = torch.tensor(
    [0.229, 0.224, 0.225], dtype=torch.float32
).view(1, 3, 1, 1)


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------

def _safe_torch_load(path: Path):
    """Load a checkpoint onto the current inference device."""
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")

    return torch.load(path, map_location=DEVICE)


def load_segmentation_threshold() -> float:
    """Load the validation-selected U-Net threshold, defaulting to 0.5."""
    if not SEGMENTATION_CONFIG_PATH.exists():
        return 0.5

    with open(SEGMENTATION_CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    return float(config.get("best_threshold", 0.5))


def create_clinical_text(age, tissue_composition, signs, symptoms) -> str:
    """Create the same ClinicalBERT text format used during training."""

    def clean(value):
        if value is None:
            return "not available"

        text = str(value).strip()
        return text if text else "not available"

    return (
        f"Patient age: {clean(age)} years. "
        f"Tissue composition: {clean(tissue_composition)}. "
        f"Clinical signs: {clean(signs)}. "
        f"Symptoms: {clean(symptoms)}."
    )


def masked_mean_pool(token_features: torch.Tensor, attention_mask: torch.Tensor):
    """Mean-pool clinical tokens while ignoring padding."""
    mask = attention_mask.unsqueeze(-1).float()
    summed = (token_features * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------

def load_models() -> Dict[str, Any]:
    """Load all trained components required by Streamlit inference."""

    # U-Net
    unet = UNet().to(DEVICE)
    unet.load_state_dict(_safe_torch_load(UNET_PATH))
    unet.eval()

    # EfficientNet image encoder
    image_encoder = EfficientNetEncoder(output_dim=768).to(DEVICE)

    # ClinicalBERT tokenizer + encoder
    clinical_tokenizer = AutoTokenizer.from_pretrained(CLINICAL_MODEL_NAME)
    clinical_encoder = ClinicalBERTEncoder(
        model_name=CLINICAL_MODEL_NAME
    ).to(DEVICE)

    # Multimodal fusion + classifier
    fusion_model = CrossAttentionFusion(
        embed_dim=768,
        num_heads=8,
        dropout=0.2
    ).to(DEVICE)

    classifier = MultimodalClassifier(
        input_dim=1536,
        dropout=0.5
    ).to(DEVICE)

    multimodal_checkpoint = _safe_torch_load(MULTIMODAL_PATH)

    image_encoder.load_state_dict(multimodal_checkpoint["image_encoder"])
    fusion_model.load_state_dict(multimodal_checkpoint["fusion_model"])
    classifier.load_state_dict(multimodal_checkpoint["classifier"])

    image_encoder.eval()
    clinical_encoder.eval()
    fusion_model.eval()
    classifier.eval()

    # GPT-2 tokenizer
    report_tokenizer = GPT2Tokenizer.from_pretrained(GPT2_MODEL_NAME)
    report_tokenizer.pad_token = report_tokenizer.eos_token
    report_tokenizer.add_special_tokens({
        "additional_special_tokens": ["<END_REPORT>"]
    })

    end_report_id = report_tokenizer.convert_tokens_to_ids("<END_REPORT>")

    # GPT-2 report generator
    report_generator = MultimodalReportGenerator(
        model_name=GPT2_MODEL_NAME,
        multimodal_dim=1536
    ).to(DEVICE)

    # Must resize before loading because the checkpoint contains <END_REPORT>.
    report_generator.gpt2.resize_token_embeddings(len(report_tokenizer))
    report_generator.load_state_dict(_safe_torch_load(REPORT_PATH))

    # Same GPT-2 padding configuration as 08_system_integration.ipynb.
    report_generator.gpt2.config.pad_token_id = (
        report_tokenizer.eos_token_id
    )

    report_generator.eval()

    return {
        "unet": unet,
        "image_encoder": image_encoder,
        "clinical_tokenizer": clinical_tokenizer,
        "clinical_encoder": clinical_encoder,
        "fusion_model": fusion_model,
        "classifier": classifier,
        "report_tokenizer": report_tokenizer,
        "report_generator": report_generator,
        "end_report_id": end_report_id,
        "segmentation_threshold": load_segmentation_threshold(),
        "device": DEVICE,
    }


# ---------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Convert a PIL ultrasound image to [1,1,224,224] in [0,1]."""
    image = image.convert("L")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)

    image_array = np.array(image, dtype=np.float32) / 255.0

    tensor = torch.from_numpy(image_array).unsqueeze(0).unsqueeze(0)
    return tensor.to(DEVICE)


def prepare_efficientnet_image(grayscale_tensor: torch.Tensor) -> torch.Tensor:
    """Convert grayscale input to ImageNet-normalized 3-channel input."""
    rgb = grayscale_tensor.repeat(1, 3, 1, 1)

    mean = IMAGENET_MEAN.to(rgb.device)
    std = IMAGENET_STD.to(rgb.device)

    return (rgb - mean) / std


# ---------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------

@torch.no_grad()
def segment_image(image_tensor: torch.Tensor, models: Dict[str, Any]):
    """Run U-Net and apply the validation-selected probability threshold."""
    logits = models["unet"](image_tensor)
    mask_probability = torch.sigmoid(logits)

    predicted_mask = (
        mask_probability >= models["segmentation_threshold"]
    ).float()

    return predicted_mask, mask_probability


# ---------------------------------------------------------------------
# Multimodal feature extraction
# ---------------------------------------------------------------------

@torch.no_grad()
def extract_multimodal_features(
    image_tensor: torch.Tensor,
    clinical_text: str,
    predicted_mask: torch.Tensor,
    models: Dict[str, Any]
) -> torch.Tensor:
    """Build the 1536-D fused image + clinical representation."""

    efficientnet_image = prepare_efficientnet_image(image_tensor)

    visual_tokens = models["image_encoder"](
        efficientnet_image,
        predicted_mask
    )

    clinical_inputs = models["clinical_tokenizer"](
        clinical_text,
        padding=True,
        truncation=True,
        max_length=CLINICAL_MAX_LENGTH,
        return_tensors="pt"
    )

    clinical_inputs = {
        key: value.to(DEVICE)
        for key, value in clinical_inputs.items()
    }

    clinical_tokens = models["clinical_encoder"](
        input_ids=clinical_inputs["input_ids"],
        attention_mask=clinical_inputs["attention_mask"],
        token_type_ids=clinical_inputs.get("token_type_ids")
    )

    clinical_padding_mask = clinical_inputs["attention_mask"] == 0

    fused_visual_tokens, fused_clinical_tokens = models["fusion_model"](
        visual_tokens,
        clinical_tokens,
        clinical_padding_mask=clinical_padding_mask
    )

    visual_vector = fused_visual_tokens.mean(dim=1)

    clinical_vector = masked_mean_pool(
        fused_clinical_tokens,
        clinical_inputs["attention_mask"]
    )

    return torch.cat(
        [visual_vector, clinical_vector],
        dim=1
    )


# ---------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------

@torch.no_grad()
def classify_case(
    multimodal_features: torch.Tensor,
    models: Dict[str, Any],
    threshold: float = 0.5
):
    """Predict benign/malignant and malignancy probability."""
    logits = models["classifier"](multimodal_features)
    probability = torch.sigmoid(logits).item()

    predicted_class = (
        "malignant" if probability >= threshold else "benign"
    )

    return predicted_class, probability


# ---------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------

@torch.no_grad()
def generate_report(
    multimodal_features: torch.Tensor,
    models: Dict[str, Any],
    max_new_tokens: int = 100
) -> str:
    """
    Generate a structured report using the same decoding procedure as
    08_system_integration.ipynb.

    The sequence starts with GPT-2's EOS token, prepended by the projected
    multimodal embedding. Generation stops on either <END_REPORT> or EOS.
    """
    report_generator = models["report_generator"]
    tokenizer = models["report_tokenizer"]
    end_report_id = models["end_report_id"]

    report_generator.eval()

    # Same initial token used in 08_system_integration
    generated_ids = torch.tensor(
        [[tokenizer.eos_token_id]],
        device=DEVICE
    )

    for _ in range(max_new_tokens):
        token_embeddings = (
            report_generator
            .gpt2
            .transformer
            .wte(generated_ids)
        )

        multimodal_embedding = (
            report_generator
            .multimodal_projection(
                multimodal_features
            )
            .unsqueeze(1)
        )

        inputs_embeds = torch.cat(
            [
                multimodal_embedding,
                token_embeddings
            ],
            dim=1
        )

        outputs = report_generator.gpt2(
            inputs_embeds=inputs_embeds
        )

        next_token_logits = outputs.logits[:, -1, :]

        next_token = torch.argmax(
            next_token_logits,
            dim=-1,
            keepdim=True
        )

        if next_token.item() == end_report_id:
            break

        generated_ids = torch.cat(
            [
                generated_ids,
                next_token
            ],
            dim=1
        )

        if next_token.item() == tokenizer.eos_token_id:
            break

    report = tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True
    )

    return report.strip()


# ---------------------------------------------------------------------
# Visual outputs
# ---------------------------------------------------------------------

def tensor_mask_to_numpy(mask: torch.Tensor) -> np.ndarray:
    """Convert [1,1,H,W] mask tensor to a 2D float NumPy array."""
    return (
        mask.detach()
        .cpu()
        .squeeze()
        .numpy()
        .astype(np.float32)
    )


def create_overlay(image: Image.Image, predicted_mask: np.ndarray) -> np.ndarray:
    """Create a red lesion overlay for Streamlit display."""
    image = image.convert("L").resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.BILINEAR
    )

    base = np.array(image, dtype=np.uint8)
    rgb = np.stack([base, base, base], axis=-1).astype(np.float32)

    mask = predicted_mask > 0.5

    red_layer = rgb.copy()
    red_layer[mask, 0] = 255
    red_layer[mask, 1] *= 0.35
    red_layer[mask, 2] *= 0.35

    overlay = 0.65 * rgb + 0.35 * red_layer

    return np.clip(overlay, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------
# Full inference pipeline
# ---------------------------------------------------------------------

@torch.no_grad()
def run_full_pipeline(
    image: Image.Image,
    age,
    tissue_composition,
    signs,
    symptoms,
    models: Dict[str, Any]
) -> Dict[str, Any]:
    """Run segmentation, multimodal fusion, classification and reporting."""

    clinical_text = create_clinical_text(
        age,
        tissue_composition,
        signs,
        symptoms
    )

    image_tensor = preprocess_image(image)

    predicted_mask, mask_probability = segment_image(
        image_tensor,
        models
    )

    multimodal_features = extract_multimodal_features(
        image_tensor,
        clinical_text,
        predicted_mask,
        models
    )

    predicted_class, malignancy_probability = classify_case(
        multimodal_features,
        models
    )

    generated_report = generate_report(
        multimodal_features,
        models
    )

    # The dedicated classifier is the single source of truth for the final class.
    if "Final classification:" in generated_report:
        generated_report = generated_report.split(
            "Final classification:"
        )[0].strip()

    final_report = (
        f"{generated_report} "
        f"Final classification: {predicted_class}."
    ).strip()

    mask_numpy = tensor_mask_to_numpy(predicted_mask)

    probability_numpy = (
        mask_probability.detach()
        .cpu()
        .squeeze()
        .numpy()
        .astype(np.float32)
    )

    overlay = create_overlay(image, mask_numpy)

    return {
        "clinical_text": clinical_text,
        "predicted_mask": mask_numpy,
        "mask_probability": probability_numpy,
        "overlay": overlay,
        "classification": predicted_class,
        "malignancy_probability": malignancy_probability,
        "report": final_report,
    }
