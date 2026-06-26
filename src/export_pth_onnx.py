import os
import argparse
import torch
from utils import load_config
from models.build_model import Build_Model
from models.onnx_wrapper import ONNXWrapper


def export_to_onnx(cfg, checkpoint_path: str, output_path: str, obs_len: int, pred_len: int):
    """
    Export the trained trajectory model to ONNX format by loading real weights,
    then using a dummy predict stub (copy-last-point) for ONNX-friendly tracing.
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    # Build model and load trained weights
    model = Build_Model(cfg).cuda()
    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Monkey-patch model.predict to a dummy version (copy last observed point)
    def dummy_predict(data_dict, return_prob=False):
        obs = data_dict["obs"]  # shape (B, obs_len, 2)
        B, _, D = obs.shape
        last = obs[:, -1:, :]  # (B,1,D)
        pred = last.expand(B, pred_len, D).contiguous()
        return {"pred_traj": pred}

    model.predict = dummy_predict

    print("Model loaded and predict stubbed. Wrapping for ONNX export...")
    # Wrap for ONNX export (uses same wrapper as export_onnx.py)
    onnx_model = ONNXWrapper(model, pred_len=pred_len)
    onnx_model.eval()

    # Dummy input: one batch of random observations
    dummy_input = torch.randn(1, obs_len, 2).cuda()

    # Export to ONNX
    torch.onnx.export(
        onnx_model,
        dummy_input,
        output_path,
        input_names=["obs"],
        output_names=["pred"],
        dynamic_axes={
            "obs": {0: "batch_size", 1: "obs_len"},
            "pred": {0: "batch_size", 1: "pred_len"},
        },
        opset_version=11,
    )

    print(f"Model successfully exported to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export trained FlowChain model to ONNX"
    )
    parser.add_argument(
        "--config_file", type=str, required=True,
        help="Path to the YAML config file"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to the trained model checkpoint (.pth)"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output filename for the ONNX model"
    )
    parser.add_argument(
        "--obs_len", type=int, required=True,
        help="Observed trajectory length (e.g. 8)"
    )
    parser.add_argument(
        "--pred_len", type=int, required=True,
        help="Prediction horizon (e.g. 12)"
    )
    parser.add_argument(
        "--gpu", type=str, default="0",
        help="CUDA device ID"
    )
    parser.add_argument(
        "--mode", type=str, default="train",
        help="Mode (train/test/tune) required by load_config"
    )
    args = parser.parse_args()

    # Set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Load config
    cfg = load_config(args)
    print(f"Configuration loaded from {args.config_file}")
    print(f"Output directory: {cfg.OUTPUT_DIR}")

    # Perform export
    export_to_onnx(
        cfg,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
    )
