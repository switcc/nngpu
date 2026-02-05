// Compute shader: normalise a float32 array to the [0, 1] range.
//
// This is the "WebGPU preprocessing" step in the demo pipeline.
// The result is then handed to WebNN for inference.

@group(0) @binding(0) var<storage, read_write> data: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    if (idx >= arrayLength(&data)) {
        return;
    }

    // Simple min-max normalisation.
    // For a real workload you would first reduce to find min/max in a
    // separate pass; here we hard-code the expected range for the demo.
    let min_val: f32 = 0.0;
    let max_val: f32 = 255.0;

    data[idx] = clamp((data[idx] - min_val) / (max_val - min_val), 0.0, 1.0);
}
