import mlx.core as mx
import pytest

from rmsnorm.kernel_naive import rms_norm_naive
from rmsnorm.kernel_optimized import rms_norm_optimized
from rmsnorm.kernel_parallel import rms_norm_parallel
from rmsnorm.reference import rms_norm_mlx, rms_norm_reference


SHAPES = [(1, 512), (16, 1024), (32, 4096), (2, 3, 8192)]
DTYPES = [mx.float16, mx.float32]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_reference_matches_mlx(shape, dtype):
    x = mx.random.normal(shape).astype(dtype)
    weight = mx.random.normal((shape[-1],)).astype(dtype)

    actual = rms_norm_reference(x, weight)
    expected = rms_norm_mlx(x, weight)
    mx.eval(actual, expected)

    assert mx.allclose(actual, expected, rtol=1e-2, atol=1e-3).item()


@pytest.mark.parametrize(
    "implementation",
    [rms_norm_naive, rms_norm_parallel, rms_norm_optimized],
)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_custom_kernel_matches_mlx(implementation, shape, dtype):
    x = mx.random.normal(shape).astype(dtype)
    weight = mx.random.normal((shape[-1],)).astype(dtype)

    actual = implementation(x, weight)
    expected = rms_norm_mlx(x, weight)
    mx.eval(actual, expected)

    assert mx.allclose(actual, expected, rtol=1e-2, atol=1e-3).item()


@pytest.mark.parametrize("threadgroup_size", [32, 64, 128, 256, 512])
@pytest.mark.parametrize("implementation", [rms_norm_parallel, rms_norm_optimized])
def test_threadgroup_sizes(implementation, threadgroup_size):
    x = mx.random.normal((16, 4096)).astype(mx.float16)
    weight = mx.random.normal((4096,)).astype(mx.float16)

    actual = implementation(x, weight, threadgroup_size=threadgroup_size)
    expected = rms_norm_mlx(x, weight)
    mx.eval(actual, expected)

    assert mx.allclose(actual, expected, rtol=1e-2, atol=1e-3).item()


@pytest.mark.parametrize("vector_width", [1, 2, 4])
def test_vector_widths(vector_width):
    x = mx.random.normal((16, 4096)).astype(mx.float16)
    weight = mx.random.normal((4096,)).astype(mx.float16)

    actual = rms_norm_optimized(x, weight, vector_width=vector_width)
    expected = rms_norm_mlx(x, weight)
    mx.eval(actual, expected)

    assert mx.allclose(actual, expected, rtol=1e-2, atol=1e-3).item()


@pytest.mark.parametrize("accumulation", ["input", "fp32"])
def test_accumulation_modes(accumulation):
    x = mx.random.normal((16, 4096)).astype(mx.float16)
    weight = mx.random.normal((4096,)).astype(mx.float16)

    actual = rms_norm_optimized(x, weight, accumulation=accumulation)
    expected = rms_norm_mlx(x, weight)
    mx.eval(actual, expected)

    assert mx.allclose(actual, expected, rtol=1e-2, atol=1e-3).item()
