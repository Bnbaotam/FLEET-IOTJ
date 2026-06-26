import os
import argparse
import torch
from utils import load_config
from models.build_model import Build_Model
from models.onnx_wrapper import ONNXWrapper

def export_to_onnx(cfg, output_path: str, obs_len: int, pred_len: int):
    """
    Export the trajectory model to ONNX by stubbing out the model.predict
    so that it simply copies the last observed position for pred_len steps.
    """
    print("Exporting model to ONNX...")

    # Build and move model to GPU
    model = Build_Model(cfg).cuda()
    model.eval()

    # Monkey‐patch model.predict to a dummy version
    def dummy_predict(data_dict, return_prob=False):
        # data_dict["obs"] is (B, obs_len, 2)
        obs = data_dict["obs"]
        B, _, D = obs.shape
        # take last observed point, then repeat it pred_len times
        last = obs[:, -1:, :]                     # (B, 1, D)
        pred = last.expand(B, pred_len, D).contiguous()  # (B, pred_len, D)
        # match ONNXWrapper’s expected key
        return {"pred_traj": pred}

    model.predict = dummy_predict

    # Wrap for ONNX export
    onnx_model = ONNXWrapper(model, pred_len=pred_len)

    # Dummy input: one batch of random obs
    dummy_input = torch.randn(1, obs_len, 2).cuda()

    # Export
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
        description="Export Trajectory Prediction Model to ONNX"
    )
    parser.add_argument(
        "--config_file", type=str, required=True,
        help="Path to your YAML config"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Filename for the ONNX model"
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
        help="Mode (train/test/tune) – required by load_config"
    )

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Load your config (now parser has mode)
    cfg = load_config(args)
    print(f"Configuration loaded from {args.config_file}")
    print(f"Output directory: {cfg.OUTPUT_DIR}")

    export_to_onnx(cfg, args.output, obs_len=args.obs_len, pred_len=args.pred_len)
