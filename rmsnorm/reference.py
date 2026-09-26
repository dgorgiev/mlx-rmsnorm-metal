import mlx.core as mx


def rms_norm_reference(x, weight, eps=1e-5):
    """Compute RMSNorm with ordinary MLX operations."""
    variance = mx.mean(x * x, axis=-1, keepdims=True)
    inv_rms = mx.rsqrt(variance + eps)
    return x * inv_rms * weight


def rms_norm_mlx(x, weight, eps=1e-5):
    """Compute RMSNorm with MLX's optimized implementation."""
    return mx.fast.rms_norm(x, weight, eps)
