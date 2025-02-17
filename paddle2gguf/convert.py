import argparse
import os
import subprocess
import torch
import onnx2torch


def paddle_to_onnx(
    paddle_model_dir,
    paddle_model_file,
    paddle_params_file,
    onnx_save_path,
    opset_version=9,
):
    # 调用 Paddle2ONNX API
    cmd = [
        "paddle2onnx",
        "--model_dir",
        paddle_model_dir,
        "--model_filename",
        paddle_model_file,
        "--params_filename",
        paddle_params_file,
        "--save_file",
        onnx_save_path,
        # "--opset_version",
        # str(opset_version),
    ]

    subprocess.run(cmd, check=True)


def onnx_to_torch(onnx_model_path, torch_save_path):
    # 加载 ONNX 模型并转换为 PyTorch 模型
    torch_model = onnx2torch.convert(onnx_model_path)
    torch.save(torch_model, torch_save_path)


def torch_to_gguf(torch_model_path, gguf_save_path, llama_cpp_dir, outtype="f16"):
    # 调用 llama.cpp 的 convert_hf_to_gguf.py 脚本
    script_path = os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py")
    cmd = [
        "python",
        script_path,
        torch_model_path,
        "--outtype",
        outtype,
        "--outfile",
        gguf_save_path,
    ]

    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Convert Paddle model to GGUF format.")
    parser.add_argument(
        "--paddle_model_dir", required=True, help="Directory of the Paddle model."
    )
    parser.add_argument(
        "--paddle_model_file", required=True, help="Filename of the Paddle model."
    )
    parser.add_argument(
        "--paddle_params_file",
        required=True,
        help="Filename of the Paddle model parameters.",
    )
    parser.add_argument(
        "--onnx_save_path",
        required=False,
        default="model.onnx",
        help="Path to save the ONNX model.",
    )
    parser.add_argument(
        "--torch_save_path",
        required=False,
        default="model.pth",
        help="Path to save the PyTorch model.",
    )
    parser.add_argument(
        "--gguf_save_path",
        required=False,
        default="model.gguf",
        help="Path to save the GGUF model.",
    )
    parser.add_argument(
        "--llama_cpp_dir", required=True, help="Directory of the llama.cpp repository."
    )
    parser.add_argument(
        "--opset_version", type=int, default=9, help="ONNX opset version."
    )
    parser.add_argument(
        "--outtype",
        type=str,
        choices=["f32", "f16", "bf16", "q8_0", "tq1_0", "tq2_0", "auto"],
        default="f16",
        help="output format - use f32 for float32, f16 for float16, bf16 for bfloat16, q8_0 for Q8_0, tq1_0 or tq2_0 for ternary, and auto for the highest-fidelity 16-bit float type depending on the first loaded tensor type",
    )

    args = parser.parse_args()

    # Step 1: Paddle to ONNX
    print(">>> Converting Paddle model to ONNX...")
    paddle_to_onnx(
        args.paddle_model_dir,
        args.paddle_model_file,
        args.paddle_params_file,
        args.onnx_save_path,
        args.opset_version,
    )
    print(">>> ONNX model saved to:", args.onnx_save_path)

    # Step 2: ONNX to Torch
    print(">>> Converting ONNX model to PyTorch...")
    onnx_to_torch(args.onnx_save_path, args.torch_save_path)
    print(">>> PyTorch model saved to:", args.torch_save_path)

    # Step 3: Torch to GGUF
    print(">>> Converting PyTorch model to GGUF...")
    torch_to_gguf(
        args.torch_save_path, args.gguf_save_path, args.llama_cpp_dir, args.outtype
    )
    print(">>> GGUF model saved to:", args.gguf_save_path)


if __name__ == "__main__":
    main()
