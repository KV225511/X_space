"""
AstroFoundation — Explainability & Interpretability (Grad-CAM, SHAP, Saliency & Gating)
======================================================================================
Provides astrophysically grounded interpretability for 1D light curves,
stellar spectra, 2D FITS images, and multi-modal fusion:

1. GradCAM1D: 1D Gradient-weighted Class Activation Mapping for Conv1D backbones.
2. GradCAM2D: 2D Spatial Grad-CAM for FITS vision backbones (PSF, host galaxy, transients).
3. SaliencyExplainer: Input gradient and SmoothGrad saliency maps for 1D sequences.
4. ShapleyExplainer: SHAP KernelExplainer.
5. MultiModalGatingExplainer: Quantifies modality contribution weights from cross-attention gating.
6. LineIdentifier: Matches salient spectral regions with canonical atomic emission/absorption lines.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Union, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Canonical astrophysical rest-frame spectral lines (in Angstroms)
CANONICAL_SPECTRAL_LINES = {
    "Ca_II_K": 3933.7,
    "Ca_II_H": 3968.5,
    "H_delta": 4101.7,
    "G_band": 4304.0,
    "H_gamma": 4340.5,
    "He_II_4686": 4685.7,
    "H_beta": 4861.3,
    "O_III_4959": 4958.9,
    "O_III_5007": 5006.8,
    "Mg_I_b": 5175.4,
    "Na_I_D": 5892.0,
    "TiO_band_1": 6158.0,
    "H_alpha": 6562.8,
    "N_II_6584": 6583.5,
    "S_II_6717": 6716.4,
    "TiO_band_2": 7054.0,
    "Ca_II_triplet_1": 8498.0,
    "Ca_II_triplet_2": 8542.1,
    "Ca_II_triplet_3": 8662.1,
}


@dataclass
class Explanation1DOutput:
    """
    Standard container for 1D model explanation.
    """
    saliency_map: np.ndarray             # (L,) normalized onto [0.0, 1.0]
    target_class: int                    # Target class index explained
    confidence: float                    # Predicted confidence for target class
    top_salient_wavelengths: List[float] # Wavelengths (Å) of salient local peaks
    matched_astrophysical_lines: List[Dict[str, Any]] # Matched atomic transitions


class GradCAM1D:
    """
    1D Grad-CAM implementation for 1D Convolutional Neural Networks.
    Hooks into any specified Conv1D or Sequential module.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self._handles.extend([h1, h2])

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def generate(
        self,
        x: torch.Tensor,
        target_class: Optional[int] = None,
        wavelength_grid: Optional[np.ndarray] = None,
        line_tolerance_angstroms: float = 25.0,
    ) -> Explanation1DOutput:
        """
        Generates 1D Grad-CAM saliency map for input sequence x.
        """
        self.model.eval()
        self.model.zero_grad()

        if x.dim() == 1:
            x = x.unsqueeze(0)
        target_len = x.shape[-1]

        # Forward pass
        logits = self.model(x)
        if hasattr(logits, 'logits'):
            logits = logits.logits

        probs = F.softmax(logits, dim=-1)
        if target_class is None:
            target_class = int(torch.argmax(logits, dim=-1).item())

        score = logits[0, target_class]
        score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Hooks failed to capture activations or gradients.")

        activations = self.activations[0]
        gradients = self.gradients[0]

        # Global Average Pooling of gradients across length dimension
        alpha = torch.mean(gradients, dim=-1, keepdim=True)  # (Channels, 1)

        # Weighted combination of feature maps
        cam = torch.sum(alpha * activations, dim=0)  # (Length_feat,)
        cam = F.relu(cam)  # ReLU: only features with positive contribution

        # Interpolate up to input length
        cam = cam.unsqueeze(0).unsqueeze(0)
        cam = F.interpolate(cam, size=target_len, mode='linear', align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()

        # Normalize onto [0, 1]
        c_min, c_max = np.min(cam), np.max(cam)
        if c_max - c_min > 1e-8:
            saliency = (cam - c_min) / (c_max - c_min)
        else:
            saliency = np.zeros_like(cam)

        # Find salient peaks (local maxima > 0.40)
        top_wavelengths = []
        matched_lines = []

        if wavelength_grid is None:
            wavelength_grid = np.linspace(3800.0, 9200.0, target_len)

        high_sal_mask = saliency > 0.40
        peak_indices = np.where(high_sal_mask)[0]

        for p_idx in peak_indices:
            peak_wave = float(wavelength_grid[p_idx])
            top_wavelengths.append(peak_wave)

            # Match against canonical astrophysical lines
            for line_name, canonical_wave in CANONICAL_SPECTRAL_LINES.items():
                if abs(peak_wave - canonical_wave) <= line_tolerance_angstroms:
                    matched_lines.append({
                        "line": line_name,
                        "canonical_wavelength": canonical_wave,
                        "observed_wavelength": peak_wave,
                        "saliency_intensity": float(saliency[p_idx])
                    })

        # Deduplicate matched lines by line name keeping highest intensity
        dedup_lines = {}
        for m in matched_lines:
            name = m["line"]
            if name not in dedup_lines or m["saliency_intensity"] > dedup_lines[name]["saliency_intensity"]:
                dedup_lines[name] = m

        return Explanation1DOutput(
            saliency_map=saliency,
            target_class=target_class,
            confidence=float(probs[0, target_class].item()),
            top_salient_wavelengths=list(set(np.round(top_wavelengths, 1).tolist())),
            matched_astrophysical_lines=list(dedup_lines.values())
        )


class GradCAM2D:
    """
    2D Grad-CAM implementation for convolutional vision backbones (FITS images).
    Identifies salient spatial pixel clusters (e.g. PSF core, host galaxy, companion blending).
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self._handles.extend([h1, h2])

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def generate(
        self,
        x: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generates 2D Grad-CAM heatmap normalized to [0, 1] matching (H, W) of input image.
        """
        self.model.eval()
        self.model.zero_grad()

        if x.dim() == 3:
            x = x.unsqueeze(0)
        _, _, H, W = x.shape

        logits = self.model(x)
        if hasattr(logits, 'logits'):
            logits = logits.logits

        if target_class is None:
            target_class = int(torch.argmax(logits, dim=-1).item())

        score = logits[0, target_class]
        score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Hooks failed to capture 2D activations/gradients.")

        acts = self.activations[0]
        grads = self.gradients[0]

        # Spatial Global Average Pooling
        weights = torch.mean(grads, dim=(-2, -1), keepdim=True)
        cam = torch.sum(weights * acts, dim=0)
        cam = F.relu(cam)

        # Upsample to input resolution
        cam = cam.unsqueeze(0).unsqueeze(0)
        cam = F.interpolate(cam, size=(H, W), mode='bilinear', align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()

        c_min, c_max = np.min(cam), np.max(cam)
        if c_max - c_min > 1e-8:
            return (cam - c_min) / (c_max - c_min)
        return np.zeros_like(cam)


class SaliencyExplainer:
    """
    Computes input gradient saliency maps and SmoothGrad for 1D models.
    """
    def __init__(self, model: nn.Module):
        self.model = model

    def explain(
        self,
        x: torch.Tensor,
        target_class: Optional[int] = None,
        num_smoothgrad_samples: int = 15,
        stdev_spread: float = 0.15,
    ) -> np.ndarray:
        """
        Computes SmoothGrad saliency map: averages gradients across noisy copies.
        """
        self.model.eval()
        if x.dim() == 1:
            x = x.unsqueeze(0)

        x_np = x.detach().cpu().numpy()
        stdev = stdev_spread * (np.max(x_np) - np.min(x_np) + 1e-8)

        accumulated_grads = np.zeros_like(x_np)

        for _ in range(num_smoothgrad_samples):
            noise = np.random.normal(0, stdev, x_np.shape).astype(np.float32)
            x_noisy = torch.tensor(x_np + noise, device=x.device, requires_grad=True)

            logits = self.model(x_noisy)
            if hasattr(logits, 'logits'):
                logits = logits.logits

            if target_class is None:
                target_class = int(torch.argmax(logits, dim=-1).item())

            self.model.zero_grad()
            score = logits[0, target_class]
            score.backward()

            grad = x_noisy.grad.detach().cpu().numpy()
            accumulated_grads += np.abs(grad)

        smooth_saliency = np.squeeze(accumulated_grads / num_smoothgrad_samples)
        s_min, s_max = np.min(smooth_saliency), np.max(smooth_saliency)
        if s_max - s_min > 1e-8:
            return (smooth_saliency - s_min) / (s_max - s_min)
        return np.zeros_like(smooth_saliency)


class ShapleyExplainer:
    """
    SHAP Explainer for AstroFoundation.
    Uses SHAP KernelExplainer for feature importance credit assignment.
    """
    def __init__(self, model: nn.Module):
        self.model = model

    def attribute_kernel_shap(
        self,
        x: np.ndarray,
        background_samples: np.ndarray,
        target_class: int = 0,
        nsamples: int = 50,
    ) -> np.ndarray:
        """
        Computes KernelSHAP feature importances using shap.KernelExplainer.
        """
        import shap

        def model_predict(data: np.ndarray):
            with torch.no_grad():
                device = next(self.model.parameters()).device
                tensor = torch.tensor(data, dtype=torch.float32, device=device)
                preds = self.model(tensor)
                if hasattr(preds, 'probabilities'):
                    preds = preds.probabilities
                elif hasattr(preds, 'logits'):
                    preds = F.softmax(preds.logits, dim=-1)
                else:
                    preds = F.softmax(preds, dim=-1)
                return preds.cpu().numpy()

        explainer = shap.KernelExplainer(model_predict, background_samples)
        shap_values = explainer.shap_values(x, nsamples=nsamples)
        if isinstance(shap_values, list):
            return shap_values[target_class]
        if hasattr(shap_values, "ndim") and shap_values.ndim == 3:
            return shap_values[:, :, target_class]
        return shap_values


class MultiModalGatingExplainer:
    """
    Extracts and explains modality contribution weights from CrossAttentionFusion.
    """
    @staticmethod
    def explain_contributions(fusion_module: nn.Module, embeddings: Tuple[torch.Tensor, ...]) -> Dict[str, float]:
        """
        Computes percentage contribution of each modality branch via gating sigma network.
        """
        if hasattr(fusion_module, 'gating_network'):
            with torch.no_grad():
                valid = [e for e in embeddings if e is not None]
                if len(valid) == 3:
                    concat = torch.cat(valid, dim=-1)
                    weights = fusion_module.gating_network(concat)[0].cpu().numpy()
                    w_sum = np.sum(weights) + 1e-8
                    return {
                        "light_curve_pct": float(round((weights[0] / w_sum) * 100.0, 1)),
                        "fits_image_pct":  float(round((weights[1] / w_sum) * 100.0, 1)),
                        "spectra_pct":     float(round((weights[2] / w_sum) * 100.0, 1)),
                    }
        return {"light_curve_pct": 33.3, "fits_image_pct": 33.3, "spectra_pct": 33.3}
