#!/usr/bin/env python3
"""
Simple diagnostic script to check engine files and extract tensor info.
"""

import os
import sys
import subprocess

def check_engine_with_trtexec(engine_path):
    """Use trtexec utility to inspect engine."""
    print(f"\n=== Inspecting {engine_path} ===\n")

    if not os.path.exists(engine_path):
        print(f"ERROR: Engine file not found: {engine_path}")
        return

    # Try using trtexec if available
    try:
        result = subprocess.run(
            ['trtexec', '--loadEngine=' + engine_path, '--dumpProfile'],
            capture_output=True,
            text=True,
            timeout=10
        )
        print("--- trtexec output ---")
        print(result.stdout)
        if result.stderr:
            print("--- stderr ---")
            print(result.stderr)
    except FileNotFoundError:
        print("trtexec not found, trying direct Python inspection...")
        try_python_inspection(engine_path)
    except Exception as e:
        print(f"Error running trtexec: {e}")
        try_python_inspection(engine_path)

def try_python_inspection(engine_path):
    """Try direct Python tensorrt inspection."""
    try:
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())

        if engine is None:
            print("ERROR: Failed to deserialize engine")
            return

        print("\n--- TensorRT Inspection ---")

        # Try new API first
        if hasattr(engine, 'num_io_tensors'):
            print(f"Number of I/O Tensors: {engine.num_io_tensors}\n")

            inputs = []
            outputs = []

            for i in range(engine.num_io_tensors):
                name = engine.get_tensor_name(i)
                mode = engine.get_tensor_mode(name)
                shape = engine.get_tensor_shape(name)
                dtype = engine.get_tensor_dtype(name)

                if mode == trt.TensorIOMode.INPUT:
                    inputs.append((name, shape, str(dtype)))
                    print(f"INPUT:  {name} | Shape: {shape} | Type: {dtype}")
                else:
                    outputs.append((name, shape, str(dtype)))
                    print(f"OUTPUT: {name} | Shape: {shape} | Type: {dtype}")

            print(f"\n--- For config file ---")
            output_names = ';'.join([o[0] for o in outputs])
            print(f"output-blob-names={output_names}")
        else:
            print("Old TensorRT API (using bindings):")
            print(f"Num bindings: {engine.num_bindings}\n")

            outputs = []
            for i in range(engine.num_bindings):
                name = engine.get_binding_name(i)
                is_input = engine.binding_is_input(i)
                shape = engine.get_binding_shape(i)
                dtype = engine.get_binding_dtype(i)

                if is_input:
                    print(f"INPUT:  {name} | Shape: {shape} | Type: {dtype}")
                else:
                    outputs.append(name)
                    print(f"OUTPUT: {name} | Shape: {shape} | Type: {dtype}")

            print(f"\n--- For config file ---")
            output_names = ';'.join(outputs)
            print(f"output-blob-names={output_names}")

    except ImportError:
        print("ERROR: tensorrt module not available")
        print("Try: pip install tensorrt")
    except Exception as e:
        print(f"ERROR during inspection: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    engine_file = sys.argv[1] if len(sys.argv) > 1 else "./models/resnet34_peoplenet.engine"
    check_engine_with_trtexec(engine_file)
