"""
Convert ONNX model to FP16 (Float16) for ~2x faster GPU inference.
Also optionally builds a TensorRT engine cache.

Usage:
    python scripts/convert_fp16.py
    python scripts/convert_fp16.py --input models/arcface_best_model_v3.onnx
    python scripts/convert_fp16.py --trt   # Also build TensorRT engine

Output:
    models/arcface_best_model_v3_fp16.onnx
    models/arcface_best_model_v3.trt  (optional, if --trt)
"""

import os
import sys
import argparse

# Setup paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "core"))
MODELS_DIR = os.path.join(ROOT_DIR, "models")


def convert_to_fp16(input_path, output_path=None):
    """Convert ONNX model from FP32 to FP16."""
    try:
        import onnx
        from onnxconverter_common import float16
    except ImportError:
        print("[ERROR] Missing dependencies. Install with:")
        print("  pip install onnx onnxconverter-common")
        return None

    print(f"[1/3] Loading model: {input_path}")
    model = onnx.load(input_path)
    input_size_mb = os.path.getsize(input_path) / (1024 * 1024)
    print(f"      Size: {input_size_mb:.1f} MB (FP32)")

    print("[2/3] Converting to FP16...")
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)

    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_fp16{ext}"

    print(f"[3/3] Saving: {output_path}")
    onnx.save(model_fp16, output_path)
    output_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    reduction = (1 - output_size_mb / input_size_mb) * 100
    print(f"\n{'='*50}")
    print(f"  FP32: {input_size_mb:.1f} MB")
    print(f"  FP16: {output_size_mb:.1f} MB ({reduction:.0f}% smaller)")
    print(f"  Expected speedup: ~1.5-2x on GPU")
    print(f"{'='*50}")

    return output_path


def build_trt_engine(onnx_path, trt_path=None):
    """Build TensorRT engine from ONNX model (NVIDIA GPU required)."""
    try:
        import tensorrt as trt
    except ImportError:
        print("[WARNING] TensorRT not installed. Skipping TRT engine build.")
        print("  Install: pip install tensorrt")
        return None

    if trt_path is None:
        base, _ = os.path.splitext(onnx_path)
        trt_path = f"{base}.trt"

    print(f"\n[TRT] Building engine from: {onnx_path}")

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  [TRT ERROR] {parser.get_error(i)}")
            return None

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

    profile = builder.create_optimization_profile()
    profile.set_shape("input", (1, 3, 112, 112), (1, 3, 112, 112), (8, 3, 112, 112))
    config.add_optimization_profile(profile)

    print("[TRT] Building... (this may take a few minutes)")
    engine = builder.build_serialized_network(network, config)

    if engine is None:
        print("[TRT ERROR] Failed to build engine")
        return None

    with open(trt_path, "wb") as f:
        f.write(engine)

    trt_size_mb = os.path.getsize(trt_path) / (1024 * 1024)
    print(f"[TRT] Engine saved: {trt_path} ({trt_size_mb:.1f} MB)")
    return trt_path


def verify_model(model_path):
    """Quick verification of the converted model."""
    import numpy as np
    import onnxruntime as ort

    print(f"\n[VERIFY] Testing: {model_path}")
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    
    input_name = session.get_inputs()[0].name
    dummy = np.random.randn(1, 3, 112, 112).astype(np.float32)
    
    import time
    times = []
    for _ in range(10):
        t0 = time.time()
        result = session.run(None, {input_name: dummy})
        times.append(time.time() - t0)
    
    output = result[0]
    avg_ms = np.mean(times) * 1000

    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")
    print(f"  Avg inference: {avg_ms:.1f}ms (CPU, 10 runs)")
    print(f"  [OK] Model is valid!")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ONNX model to FP16")
    parser.add_argument("--input", default=os.path.join(MODELS_DIR, "arcface_best_model_v3.onnx"),
                        help="Input ONNX model path")
    parser.add_argument("--output", default=None, help="Output FP16 model path")
    parser.add_argument("--trt", action="store_true", help="Also build TensorRT engine")
    parser.add_argument("--verify", action="store_true", default=True, help="Verify output model")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Model not found: {args.input}")
        sys.exit(1)

    fp16_path = convert_to_fp16(args.input, args.output)

    if fp16_path and args.verify:
        verify_model(fp16_path)

    if fp16_path and args.trt:
        build_trt_engine(fp16_path)

    if fp16_path:
        print(f"\n[NEXT STEPS]")
        print(f"  1. Update config.py:")
        print(f"     ARCFACE_PATH = os.path.join(MODELS_DIR, \"{os.path.basename(fp16_path)}\")")
        print(f"  2. Restart app.py")
        print(f"  3. Run benchmark to compare FP32 vs FP16 accuracy")
