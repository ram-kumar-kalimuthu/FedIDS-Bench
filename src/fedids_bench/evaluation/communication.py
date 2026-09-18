from dataclasses import dataclass
from fedids_bench.models.base import IDSModel

@dataclass
class RoundCommunication:
    round_num: int
    upload_bytes_per_client: int
    download_bytes_per_client: int
    num_participating_clients: int

    @property
    def total_upload_bytes(self) -> int:
        return self.upload_bytes_per_client * self.num_participating_clients

    @property
    def total_download_bytes(self) -> int:
        return self.download_bytes_per_client * self.num_participating_clients

    @property
    def round_total_bytes(self) -> int:
        return self.total_upload_bytes + self.total_download_bytes


class CommunicationTracker:
    def __init__(self, model: IDSModel):
        self.param_bytes = model.parameter_bytes()
        self.history: list[RoundCommunication] = []

    def record_round(self, round_num: int, num_clients: int) -> RoundCommunication:
        # In standard FL (FedAvg):
        # Downlink: server sends global model to client -> param_bytes
        # Uplink: client sends updated model parameters to server -> param_bytes
        round_comm = RoundCommunication(
            round_num=round_num,
            upload_bytes_per_client=self.param_bytes,
            download_bytes_per_client=self.param_bytes,
            num_participating_clients=num_clients
        )
        self.history.append(round_comm)
        return round_comm

    def get_summary(self) -> dict:
        total_upload = sum(r.total_upload_bytes for r in self.history)
        total_download = sum(r.total_download_bytes for r in self.history)
        total_bytes = total_upload + total_download
        return {
            "parameter_bytes": self.param_bytes,
            "total_upload_bytes": total_upload,
            "total_download_bytes": total_download,
            "total_communication_bytes": total_bytes,
            "num_rounds_recorded": len(self.history)
        }
