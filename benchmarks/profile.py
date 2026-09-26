import argparse
import time

import mlx.core as mx

from rmsnorm.kernel_naive import rms_norm_naive
from rmsnorm.kernel_optimized import rms_norm_optimized
from rmsnorm.kernel_parallel import rms_norm_parallel
from rmsnorm.reference import rms_norm_mlx, rms_norm_reference


def main():
    parser = argparse.ArgumentParser(
        description="Run one RMSNorm configuration repeatedly for Metal profiling"
    )
    parser.add_argument(
        "--implementation",
        choices=("reference", "naive", "parallel", "optimized", "mlx"),
        default="optimized",
    )
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--hidden-dimension", type=int, default=8192)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--threadgroup-size", type=int, default=256)
    parser.add_argument("--vector-width", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument(
        "--accumulation", choices=("input", "fp32"), default="fp32"
    )
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="steady-state profiling duration in seconds; use 0 for --iterations",
    )
    parser.add_argument("--iterations", type=int, default=10000)
    args = parser.parse_args()

    dtype = mx.float16 if args.dtype == "float16" else mx.float32
    mx.random.seed(0)
    x = mx.random.normal((args.batch, args.hidden_dimension)).astype(dtype)
    weight = mx.random.normal((args.hidden_dimension,)).astype(dtype)

    implementations = {
        "reference": rms_norm_reference,
        "naive": rms_norm_naive,
        "parallel": lambda x, w: rms_norm_parallel(
            x, w, threadgroup_size=args.threadgroup_size
        ),
        "optimized": lambda x, w: rms_norm_optimized(
            x,
            w,
            threadgroup_size=args.threadgroup_size,
            vector_width=args.vector_width,
            accumulation=args.accumulation,
        ),
        "mlx": rms_norm_mlx,
    }
    implementation = implementations[args.implementation]

    for _ in range(args.warmup):
        mx.eval(implementation(x, weight))
    mx.synchronize()

    if args.duration > 0:
        deadline = time.perf_counter() + args.duration
        while time.perf_counter() < deadline:
            mx.eval(implementation(x, weight))
    else:
        for _ in range(args.iterations):
            mx.eval(implementation(x, weight))
    mx.synchronize()


if __name__ == "__main__":
    main()
