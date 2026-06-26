import torch
import torch.nn as nn

class ONNXWrapper(nn.Module):
    """
    A lightweight wrapper for exporting trajectory prediction model to ONNX.
    Converts raw observation to expected model input format.
    """
    def __init__(self, model, pred_len: int):
        super().__init__()
        self.model = model
        self.pred_len = pred_len

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to export ONNX.
        Input: obs of shape (B, T_obs, 2)
        Output: pred of shape (B, T_pred, 2)
        """
        B, T_obs, _ = obs.shape
        data_dict = {
            "obs": obs,
            "gt": obs[:, -self.pred_len:, :],  # dummy
            "obs_st": obs,  # dummy standardized
            "gt_st": obs[:, -self.pred_len:, :],
            "first_history_index": torch.zeros((B,), dtype=torch.long, device=obs.device),
            "neighbors": [obs for _ in range(B)],  # dummy neighbors
            "neighbors_gt": [obs[:, -self.pred_len:, :] for _ in range(B)],
            "neighbors_st": [obs for _ in range(B)],
            "neighbors_gt_st": [obs[:, -self.pred_len:, :] for _ in range(B)],
            "neighbors_edge": [torch.ones((T_obs,), device=obs.device) for _ in range(B)],
            "map": None,
            "use_raw_input": True
        }
        out = self.model.predict(data_dict, return_prob=False)
        return out.get(("pred", 0), out.get("pred_traj"))

