import argparse
import statistics
import time

import mlx.core as mx

from rmsnorm.kernel_naive import rms_norm_naive
from rmsnorm.kernel_optimized import rms_norm_optimized
from rmsnorm.kernel_parallel import rms_norm_parallel
from rmsnorm.reference import rms_norm_mlx, rms_norm_reference


BATCH_SIZES = [1, 16, 128, 1024]
HIDDEN_DIMENSIONS = [256, 512, 1024, 2048, 4096, 8192]
DTYPES = [mx.float16, mx.float32]
THREADGROUP_SIZES = [32, 64, 128, 256, 512]
VECTOR_WIDTHS = [1, 2, 4]
ACCUMULATION_MODES = ["input", "fp32"]
HEADER = (
    "suite,implementation,batch,hidden_dimension,dtype,threadgroup_size,"
    "vector_width,accumulation,median_us,min_us,max_us,max_abs_error,mean_abs_error"
)


def benchmark_once(fn, x, weight, warmup, iterations):
    for _ in range(warmup):
        output = fn(x, weight)
        mx.eval(output)

    mx.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        output = fn(x, weight)
        mx.eval(output)
    mx.synchronize()
    return (time.perf_counter() - start) / iterations


def benchmark_repeated(fn, x, weight, warmup, iterations, repeats):
    samples = [
        benchmark_once(fn, x, weight, warmup, iterations) * 1e6
        for _ in range(repeats)
    ]
    return statistics.median(samples), min(samples), max(samples)


def emit(
    suite,
    implementation,
    batch,
    hidden_dimension,
    dtype,
    timings,
    threadgroup_size="",
    vector_width="",
    accumulation="",
    max_abs_error="",
    mean_abs_error="",
):
    median_us, min_us, max_us = timings
    dtype_name = str(dtype).removeprefix("mlx.core.")
    error = "" if max_abs_error == "" else f"{max_abs_error:.8g}"
    mean_error = "" if mean_abs_error == "" else f"{mean_abs_error:.8g}"
    print(
        f"{suite},{implementation},{batch},{hidden_dimension},{dtype_name},"
        f"{threadgroup_size},{vector_width},{accumulation},"
        f"{median_us:.3f},{min_us:.3f},{max_us:.3f},{error},{mean_error}",
        flush=True,
    )


def run_implementations(args):
    implementations = [
        ("reference", rms_norm_reference),
        ("naive", rms_norm_naive),
        ("parallel", rms_norm_parallel),
        ("optimized", rms_norm_optimized),
        ("mlx", rms_norm_mlx),
    ]
    for dtype in DTYPES:
        for batch in BATCH_SIZES:
            for hidden_dimension in HIDDEN_DIMENSIONS:
                x = mx.random.normal((batch, hidden_dimension)).astype(dtype)
                weight = mx.random.normal((hidden_dimension,)).astype(dtype)
                for name, implementation in implementations:
                    timings = benchmark_repeated(
                        implementation,
                        x,
                        weight,
                        args.warmup,
                        args.iterations,
                        args.repeats,
                    )
                    emit(
                        "implementations",
                        name,
                        batch,
                        hidden_dimension,
                        dtype,
                        timings,
                    )


def run_threadgroups(args):
    batch = 128
    dtype = mx.float16
    for hidden_dimension in HIDDEN_DIMENSIONS:
        x = mx.random.normal((batch, hidden_dimension)).astype(dtype)
        weight = mx.random.normal((hidden_dimension,)).astype(dtype)
        for size in THREADGROUP_SIZES:
            implementations = [
                (
                    "parallel",
                    lambda x, w, size=size: rms_norm_parallel(
                        x, w, threadgroup_size=size
                    ),
                ),
                (
                    "optimized",
                    lambda x, w, size=size: rms_norm_optimized(
                        x, w, threadgroup_size=size
                    ),
                ),
            ]
            for name, implementation in implementations:
                timings = benchmark_repeated(
                    implementation,
                    x,
                    weight,
                    args.warmup,
                    args.iterations,
                    args.repeats,
                )
                emit(
                    "threadgroups",
                    name,
                    batch,
                    hidden_dimension,
                    dtype,
                    timings,
                    threadgroup_size=size,
                    vector_width=4 if name == "optimized" else "",
                    accumulation="fp32",
                )


def run_vectorization(args):
    batch = 128
    dtype = mx.float16
    for hidden_dimension in HIDDEN_DIMENSIONS:
        x = mx.random.normal((batch, hidden_dimension)).astype(dtype)
        weight = mx.random.normal((hidden_dimension,)).astype(dtype)
        for width in VECTOR_WIDTHS:
            implementation = lambda x, w, width=width: rms_norm_optimized(
                x,
                w,
                threadgroup_size=256,
                vector_width=width,
                accumulation="fp32",
            )
            timings = benchmark_repeated(
                implementation,
                x,
                weight,
                args.warmup,
                args.iterations,
                args.repeats,
            )
            emit(
                "vectorization",
                "optimized",
                batch,
                hidden_dimension,
                dtype,
                timings,
                threadgroup_size=256,
                vector_width=width,
                accumulation="fp32",
            )


def run_precision(args):
    batch = 128
    dtype = mx.float16
    for hidden_dimension in HIDDEN_DIMENSIONS:
        x = mx.random.normal((batch, hidden_dimension)).astype(dtype)
        weight = mx.random.normal((hidden_dimension,)).astype(dtype)
        expected = rms_norm_mlx(x, weight)
        mx.eval(expected)
        for accumulation in ACCUMULATION_MODES:
            implementation = lambda x, w, accumulation=accumulation: (
                rms_norm_optimized(
                    x,
                    w,
                    threadgroup_size=256,
                    vector_width=4,
                    accumulation=accumulation,
                )
            )
            actual = implementation(x, weight)
            absolute_error = mx.abs(
                actual.astype(mx.float32) - expected.astype(mx.float32)
            )
            max_error = mx.max(absolute_error)
            mean_error = mx.mean(absolute_error)
            mx.eval(max_error, mean_error)
            timings = benchmark_repeated(
                implementation,
                x,
                weight,
                args.warmup,
                args.iterations,
                args.repeats,
            )
            emit(
                "precision",
                "optimized",
                batch,
                hidden_dimension,
                dtype,
                timings,
                threadgroup_size=256,
                vector_width=4,
                accumulation=accumulation,
                max_abs_error=max_error.item(),
                mean_abs_error=mean_error.item(),
            )


def main():
    parser = argparse.ArgumentParser(description="Benchmark RMSNorm implementations")
    parser.add_argument(
        "--suite",
        choices=("all", "implementations", "threadgroups", "vectorization", "precision"),
        default="all",
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    if args.warmup < 0 or args.iterations < 1 or args.repeats < 1:
        parser.error("warmup must be nonnegative; iterations and repeats must be positive")

    mx.random.seed(0)
    print(HEADER, flush=True)
    suites = {
        "implementations": run_implementations,
        "threadgroups": run_threadgroups,
        "vectorization": run_vectorization,
        "precision": run_precision,
    }
    selected = suites if args.suite == "all" else {args.suite: suites[args.suite]}
    for run_suite in selected.values():
        run_suite(args)


if __name__ == "__main__":
    main()
