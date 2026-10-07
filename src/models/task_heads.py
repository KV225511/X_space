"""
AstroFoundation — Task Heads & Anomaly Detection
================================================
Task heads applied to the 256-dimensional fused embedding:
1. Multi-Class Classification Head (6 astronomical event classes)
2. Energy-Based Out-of-Distribution (OOD) Anomaly Detector
3. TaskHeadOutput dataclass bundling all task outputs
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TaskHeadOutput:
    """
    Standardized prediction container produced by AstroFoundation task heads.
    """
    logits: torch.Tensor             # (BATCH, num_classes) raw unnormalized logits
    probabilities: torch.Tensor      # (BATCH, num_classes) normalized softmax distribution
    predicted_class: torch.Tensor    # (BATCH,) integer predicted class index
    energy_score: torch.Tensor       # (BATCH,) scalar free energy E(x)
    is_anomaly: torch.Tensor         # (BATCH,) bool flag: True if energy > threshold
    fused_embedding: torch.Tensor    # (BATCH, 256) L2-normalized representation


class ClassificationHead(nn.Module):
    """
    Classification head mapping 256-dim fused embedding to 6 event classes.
    Uses LayerNorm + Linear + GELU + Dropout + Linear.
    """
    def __init__(self, in_features: int = 256, num_classes: int = 6, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class AnomalyHead(nn.Module):
    """
    Energy-Based Out-of-Distribution (OOD) Anomaly Scoring Head.
    Computes free energy:
        E(x; T) = -T * log(sum(exp(logits / T)))
    Samples with higher energy are less likely to belong to the 6 known classes.
    """
    def __init__(self, temperature: float = 1.0, threshold: float = 0.0):
        super().__init__()
        self.temperature = temperature
        self.threshold = threshold

    def forward(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scaled_logits = logits / self.temperature
        energy = -self.temperature * torch.logsumexp(scaled_logits, dim=-1)
        is_anomaly = energy > self.threshold
        return energy, is_anomaly


class TaskHeads(nn.Module):
    """
    Unified task head container coordinating classification and anomaly detection.
    """
    def __init__(
        self,
        in_features: int = 256,
        num_classes: int = 6,
        temperature: float = 1.0,
        anomaly_threshold: float = 0.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.classifier = ClassificationHead(in_features, num_classes, dropout)
        self.anomaly_detector = AnomalyHead(temperature, anomaly_threshold)

    def forward(self, fused_embedding: torch.Tensor) -> TaskHeadOutput:
        logits = self.classifier(fused_embedding)
        probabilities = F.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1)
        energy_score, is_anomaly = self.anomaly_detector(logits)

        return TaskHeadOutput(
            logits=logits,
            probabilities=probabilities,
            predicted_class=predicted_class,
            energy_score=energy_score,
            is_anomaly=is_anomaly,
            fused_embedding=fused_embedding
        )
