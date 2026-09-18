import time
from typing import Dict, Any
from fedids_bench.models.base import IDSModel

class ResourceTracker:
    def __init__(self, model: IDSModel):
        self.model = model
        self.client_training_time: float = 0.0
        self.server_aggregation_time: float = 0.0
        self.inference_time: float = 0.0

    def add_client_training_time(self, duration: float):
        self.client_training_time += duration

    def add_server_aggregation_time(self, duration: float):
        self.server_aggregation_time += duration

    def add_inference_time(self, duration: float):
        self.inference_time += duration

    def get_summary(self) -> Dict[str, Any]:
        return {
            "num_parameters": self.model.num_parameters(),
            "model_size_bytes": self.model.parameter_bytes(),
            "client_training_time_sec": round(self.client_training_time, 4),
            "server_aggregation_time_sec": round(self.server_aggregation_time, 4),
            "inference_time_sec": round(self.inference_time, 4)
        }
