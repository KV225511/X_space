"""
AstroFoundation — Training, Calibration & Active Learning Modules
"""

from .losses import FocalLoss, OODEnergyLoss
from .trainer import Trainer
from .schedulers import get_cosine_schedule_with_warmup
from .uncertainty import MCDropoutEstimator, UncertaintyEstimate, enable_dropout_only
from .calibration import TemperatureScaler, compute_ece, compute_brier_score
from .active_learning import ActiveLearningSelector, ActiveLearningBatch

__all__ = [
    "FocalLoss",
    "OODEnergyLoss",
    "Trainer",
    "get_cosine_schedule_with_warmup",
    "MCDropoutEstimator",
    "UncertaintyEstimate",
    "enable_dropout_only",
    "TemperatureScaler",
    "compute_ece",
    "compute_brier_score",
    "ActiveLearningSelector",
    "ActiveLearningBatch",
]
