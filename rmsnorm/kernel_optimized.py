import mlx.core as mx


VECTOR_WIDTHS = (1, 2, 4)
ACCUMULATION_MODES = ("input", "fp32")


def _make_source(vector_width, accumulation):
    accumulator = "float" if accumulation == "fp32" else "T"
    vector_type = f"vec<T, {vector_width}>"
    accumulator_vector = f"vec<{accumulator}, {vector_width}>"

    if vector_width == 1:
        reduction_load = f"""
        {accumulator} value = static_cast<{accumulator}>(x[row_offset + j]);
        partial += value * value;
"""
        output_store = """
        float value = static_cast<float>(x[row_offset + j]);
        float scale = static_cast<float>(weight[j]);
        out[row_offset + j] = static_cast<T>(value * inv_rms * scale);
"""
    else:
        reduction_load = f"""
        {vector_type} values =
            *reinterpret_cast<const device {vector_type}*>(x + row_offset + j);
        {accumulator_vector} accumulator_values = {accumulator_vector}(values);
        partial += dot(accumulator_values, accumulator_values);
"""
        output_store = f"""
        {vector_type} values =
            *reinterpret_cast<const device {vector_type}*>(x + row_offset + j);
        {vector_type} scales =
            *reinterpret_cast<const device {vector_type}*>(weight + j);
        vec<float, {vector_width}> result =
            vec<float, {vector_width}>(values) * inv_rms *
            vec<float, {vector_width}>(scales);
        *reinterpret_cast<device {vector_type}*>(out + row_offset + j) =
            {vector_type}(result);
"""

    return f"""
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint simd_id = simdgroup_index_in_threadgroup;
    uint thread_count = threads_per_threadgroup.x;
    uint simd_width = threads_per_simdgroup;
    uint simd_count = (thread_count + simd_width - 1) / simd_width;
    uint D = x_shape[x_ndim - 1];
    uint row_offset = row * D;
    uint vectorized_D = (D % {vector_width} == 0) ? D : 0;

    {accumulator} partial = static_cast<{accumulator}>(0);
    for (uint j = tid * {vector_width};
         j < vectorized_D;
         j += thread_count * {vector_width}) {{
{reduction_load}
    }}
    for (uint j = vectorized_D + tid; j < D; j += thread_count) {{
        {accumulator} value =
            static_cast<{accumulator}>(x[row_offset + j]);
        partial += value * value;
    }}

    partial = simd_sum(partial);

    threadgroup {accumulator} simd_sums[32];
    if (lane == 0) {{
        simd_sums[simd_id] = partial;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {{
        {accumulator} simd_partial = tid < simd_count
            ? simd_sums[tid]
            : static_cast<{accumulator}>(0);
        simd_partial = simd_sum(simd_partial);
        if (lane == 0) {{
            simd_sums[0] = simd_partial;
        }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_rms = metal::rsqrt(
        static_cast<float>(simd_sums[0]) / static_cast<float>(D) + eps[0]);

    for (uint j = tid * {vector_width};
         j < vectorized_D;
         j += thread_count * {vector_width}) {{
{output_store}
    }}
    for (uint j = vectorized_D + tid; j < D; j += thread_count) {{
        float value = static_cast<float>(x[row_offset + j]);
        float scale = static_cast<float>(weight[j]);
        out[row_offset + j] = static_cast<T>(value * inv_rms * scale);
    }}
"""


_optimized_kernels = {
    (vector_width, accumulation): mx.fast.metal_kernel(
        name=f"rms_norm_optimized_v{vector_width}_{accumulation}",
        input_names=["x", "weight", "eps"],
        output_names=["out"],
        source=_make_source(vector_width, accumulation),
    )
    for vector_width in VECTOR_WIDTHS
    for accumulation in ACCUMULATION_MODES
}


def rms_norm_optimized(
    x,
    weight,
    eps=1e-5,
    threadgroup_size=256,
    vector_width=4,
    accumulation="fp32",
):
    """RMSNorm with selectable vector width and accumulation precision."""
    _validate_inputs(x, weight, threadgroup_size, vector_width, accumulation)
    rows = x.size // x.shape[-1]
    eps_array = mx.array([eps], dtype=mx.float32)
    kernel = _optimized_kernels[(vector_width, accumulation)]
    return kernel(
        inputs=[x, weight, eps_array],
        template=[("T", x.dtype)],
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
        grid=(rows * threadgroup_size, 1, 1),
        threadgroup=(threadgroup_size, 1, 1),
    )[0]


def _validate_inputs(x, weight, threadgroup_size, vector_width, accumulation):
    if x.ndim == 0:
        raise ValueError("x must have at least one dimension")
    if weight.ndim != 1 or weight.shape[0] != x.shape[-1]:
        raise ValueError("weight must be one-dimensional and match x.shape[-1]")
    if x.dtype != weight.dtype:
        raise ValueError("x and weight must have the same dtype")
    if threadgroup_size not in (32, 64, 128, 256, 512):
        raise ValueError("threadgroup_size must be 32, 64, 128, 256, or 512")
    if vector_width not in VECTOR_WIDTHS:
        raise ValueError("vector_width must be 1, 2, or 4")
    if accumulation not in ACCUMULATION_MODES:
        raise ValueError("accumulation must be 'input' or 'fp32'")
