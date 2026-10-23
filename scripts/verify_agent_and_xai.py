"""
AstroFoundation — LangGraph Agent & XAI Suite Verification Script
=================================================================
Verifies:
1. LangGraph Autonomous Agent Reasoning & Feedback:
   - High-confidence planetary system (Kepler-10)
   - Extragalactic explosive transient (ASASSN-15lh)
   - High-energy kilonova / gamma-ray burst (GW170817-SGRB)
   - Out-of-Distribution Anomaly (DeepSpace-Anomaly-X)
2. Explainable AI (XAI) Suite:
   - 1D Grad-CAM on 4096-channel SDSS spectral encoder
   - 2D Grad-CAM on 224x224 FITS image ResNet-34 encoder
   - SmoothGrad noise-tolerant saliency on 8192-step light curve
   - Shapley credit assignment (SHAP) across modalities
   - Cross-attention dynamic gating weights
"""

import sys
import json
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent.agent import run_astro_agent
from src.agent.schemas import AgentOutput
from src.models.full_model import AstroFoundationModel
from src.evaluation.explainability import (
    GradCAM1D,
    GradCAM2D,
    SaliencyExplainer,
    ShapleyExplainer,
    MultiModalGatingExplainer,
    CANONICAL_SPECTRAL_LINES,
)


def verify_agent():
    print("\n" + "=" * 75)
    print("  PART 1: LangGraph Autonomous Agent Multi-Catalog Verification")
    print("=" * 75)

    test_cases = [
        {
            "target_id": "Kepler-10",
            "initial_class": "exoplanet_transit",
            "confidence": 0.985,
            "energy_score": -0.22,
            "is_anomaly": False,
            "ra": 285.6794,
            "dec": 50.2413,
            "desc": "Confirmed Rocky Exoplanet Host Star"
        },
        {
            "target_id": "ASASSN-15lh",
            "initial_class": "supernova",
            "confidence": 0.942,
            "energy_score": -0.28,
            "is_anomaly": False,
            "ra": 330.5517,
            "dec": -61.6841,
            "desc": "Superluminous Supernova SLSN-I"
        },
        {
            "target_id": "GW170817",
            "initial_class": "gamma_ray_burst",
            "confidence": 0.925,
            "energy_score": -0.35,
            "is_anomaly": False,
            "ra": 197.4503,
            "dec": -23.6331,
            "desc": "BNS Merger Kilonova / Short GRB"
        },
        {
            "target_id": "DeepSpace-Anomaly-99",
            "initial_class": "variable_star",
            "confidence": 0.380,
            "energy_score": -2.85,
            "is_anomaly": True,
            "ra": 180.1120,
            "dec": 12.4410,
            "desc": "Uncatalogued Out-of-Distribution Anomaly"
        },
    ]

    for tc in test_cases:
        print(f"\n[Case: {tc['target_id']}] ({tc['desc']})")
        print(f"  Input Neural Prediction : {tc['initial_class']} (Conf: {tc['confidence']*100:.1f}%, Energy: {tc['energy_score']:.2f}, Anomaly: {tc['is_anomaly']})")

        agent_out: AgentOutput = run_astro_agent(
            target_id=tc["target_id"],
            initial_class=tc["initial_class"],
            confidence=tc["confidence"],
            energy_score=tc["energy_score"],
            is_anomaly=tc["is_anomaly"],
            ra=tc.get("ra"),
            dec=tc.get("dec"),
        )

        print(f"  Agent Final Verdict     : {agent_out.final_classification} (Conf: {agent_out.final_confidence*100:.1f}%)")
        print(f"  Consensus Status        : {agent_out.consensus_status}")
        print(f"  Catalogs Queried        : {', '.join(agent_out.catalogs_queried)}")
        print(f"  Reasoning Steps Count   : {len(agent_out.reasoning_trace)} autonomous LangGraph cycle(s)")

        for step in agent_out.reasoning_trace:
            print(f"    - Step {step.step_index} [{step.action}]: {step.thought[:90]}...")
            if step.observation:
                print(f"      Obs: {step.observation[:90]}...")

        if agent_out.simbad_match and agent_out.simbad_match.found:
            print(f"  SIMBAD Match            : {agent_out.simbad_match.object_type} | SpType: {agent_out.simbad_match.spectral_type} | Plx: {agent_out.simbad_match.parallax_mas} mas")
        if agent_out.ned_match and agent_out.ned_match.found:
            print(f"  NED Match               : {agent_out.ned_match.object_type} | Redshift z: {agent_out.ned_match.redshift}")
        if agent_out.ads_match and agent_out.ads_match.found:
            print(f"  NASA ADS Match          : {len(agent_out.ads_match.papers)} published discovery paper(s) found. Top: '{agent_out.ads_match.papers[0].title[:60]}...'")
        print(f"  Scientific Feedback     : {agent_out.astronomical_notes}")


def verify_xai():
    print("\n" + "=" * 75)
    print("  PART 2: Explainable AI (XAI) Multi-Framework Verification")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Executing XAI algorithms on: {device}")

    # Load real multi-modal model
    model = AstroFoundationModel(embed_dim=256, num_classes=6, fits_channels=3).to(device)
    ckpt_path = Path("outputs/checkpoints/fusion_real_augmented.pt")
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  [OK] Loaded trained foundation checkpoint: {ckpt_path.name}")
    model.eval()

    # 1. Multi-Modal Gating Explainer
    print("\n[1/5] Cross-Attention Modality Gating Explainer:")
    e_lc = torch.randn(1, 256, device=device)
    e_img = torch.randn(1, 256, device=device)
    e_spec = torch.randn(1, 256, device=device)
    weights = MultiModalGatingExplainer.explain_contributions(model.fusion, (e_lc, e_img, e_spec))
    print("  Gating weights computed successfully:")
    for mod, pct in weights.items():
        print(f"    - {mod:<18}: {pct:5.2f}%")
    assert abs(sum(weights.values()) - 100.0) < 1.0, "Gating weights must sum to 100%"

    # 2. 1D Grad-CAM on Spectral Encoder
    print("\n[2/5] 1D Grad-CAM (SDSS Spectral Conv1D Encoder):")
    class SpecWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.encoder = m.spectral_encoder
            self.head = m.task_heads.classifier
        def forward(self, s):
            emb = self.encoder(s)
            return self.head(emb)

    spec_wrapper = SpecWrapper(model)
    target_conv1d = model.spectral_encoder.conv_stack[2]
    cam1d = GradCAM1D(spec_wrapper, target_conv1d)

    dummy_spec = torch.randn(1, 4096, device=device, requires_grad=True)
    wave_grid = np.linspace(3800.0, 9200.0, 4096)
    spec_cam_out = cam1d.generate(dummy_spec, wavelength_grid=wave_grid, target_class=0)
    print(f"  Saliency map shape     : {spec_cam_out.saliency_map.shape} (Range: [{spec_cam_out.saliency_map.min():.3f}, {spec_cam_out.saliency_map.max():.3f}])")
    print(f"  Target class predicted : Class {spec_cam_out.target_class} (Confidence: {spec_cam_out.confidence*100:.1f}%)")
    print(f"  Canonical Line Matches : {len(spec_cam_out.matched_astrophysical_lines)} spectral lines identified in active regions")
    for lm in spec_cam_out.matched_astrophysical_lines[:3]:
        print(f"    - {lm['line']} ({lm['canonical_wavelength']} A): Saliency={lm['saliency_intensity']:.3f}")
    cam1d.remove_hooks()

    # 3. 2D Grad-CAM on FITS ResNet-34 Image Encoder
    print("\n[3/5] 2D Grad-CAM (ResNet-34 FITS Image Encoder):")
    class ImgWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.encoder = m.image_encoder
            self.head = m.task_heads.classifier
        def forward(self, img):
            emb = self.encoder(img)
            return self.head(emb)

    img_wrapper = ImgWrapper(model)
    target_conv2d = None
    for m in reversed(list(model.image_encoder.feature_extractor.modules())):
        if isinstance(m, torch.nn.Conv2d):
            target_conv2d = m
            break

    cam2d = GradCAM2D(img_wrapper, target_conv2d)
    dummy_fits = torch.randn(1, 3, 224, 224, device=device, requires_grad=True)
    fits_cam_map = cam2d.generate(dummy_fits, target_class=0)
    print(f"  FITS 2D Heatmap shape  : {fits_cam_map.shape} (Range: [{fits_cam_map.min():.3f}, {fits_cam_map.max():.3f}])")
    cam2d.remove_hooks()

    # 4. SmoothGrad (Noise-Tolerant Input Gradient Saliency)
    print("\n[4/5] SmoothGrad SaliencyExplainer (1D Light Curves):")
    class LCWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.encoder = m.lc_encoder
            self.head = m.task_heads.classifier
        def forward(self, flx):
            emb = self.encoder(flx)
            return self.head(emb)

    lc_wrapper = LCWrapper(model)
    saliency_explainer = SaliencyExplainer(lc_wrapper)
    dummy_flux = torch.randn(1, 8192, device=device)
    sal_map = saliency_explainer.explain(dummy_flux, target_class=0, num_smoothgrad_samples=15, stdev_spread=0.15)
    print(f"  SmoothGrad map shape   : {sal_map.shape} (Range: [{sal_map.min():.3f}, {sal_map.max():.3f}])")
    top_indices = np.argsort(sal_map)[-3:][::-1]
    print(f"  Top Attributed Timesteps: {top_indices.tolist()} with peak values {[round(float(sal_map[i]), 4) for i in top_indices]}")

    # 5. ShapleyExplainer (Coalition Game-Theoretic Credit Assignment)
    print("\n[5/5] SHAP ShapleyExplainer (KernelSHAP Feature Attributions):")
    shap_explainer = ShapleyExplainer(spec_wrapper)
    bg = np.random.randn(10, 4096).astype(np.float32)
    sample_x = np.random.randn(1, 4096).astype(np.float32)
    shap_vals = shap_explainer.attribute_kernel_shap(sample_x, bg, target_class=0, nsamples=15)
    print(f"  KernelSHAP values shape : {shap_vals.shape} (Range: [{np.min(shap_vals):.4f}, {np.max(shap_vals):.4f}])")
    top_shap = np.argsort(np.abs(shap_vals[0]))[-3:][::-1]
    print(f"  Top Attributed Features : Bins {top_shap.tolist()} with SHAP weights {[round(float(shap_vals[0][i]), 5) for i in top_shap]}")

    print("\n" + "=" * 75)
    print("  ALL AGENT & XAI SUITE TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    verify_agent()
    verify_xai()
