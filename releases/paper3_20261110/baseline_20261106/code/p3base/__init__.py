"""P3 基础层：通信信道 + 卡尔曼-MMSE 信道估计，10/50/100 车通用。"""
from .config import (EstimationConfig, FoundationConfig, RFConfig, VLCConfig,
                     VehicleConfig)
from .scene import Scene, find_scene_root
from . import channel, contract, estimator, fading, geometry, link_eval, noise, pilots
from .kalman_mmse import KalmanMMSE, PredictiveBelief, paper_process_var

__all__ = [
    "EstimationConfig", "FoundationConfig", "RFConfig", "VLCConfig", "VehicleConfig",
    "Scene", "find_scene_root", "KalmanMMSE", "PredictiveBelief", "paper_process_var",
    "channel", "contract", "estimator", "fading", "geometry", "link_eval", "noise",
    "pilots",
]
__version__ = "0.1.0"
