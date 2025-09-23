# Plan 05: HDR Environment Maps & Anti-Aliasing

## 🎯 **Current Implementation Plan**

### **Phase 1: HDR Environment Maps**
**Target: Realistic reflections and lighting**

#### **1.1 Environment Map Infrastructure**
- [ ] Add HDR image loading support (EXR format)
- [ ] Create environment map sampling functions
- [ ] Implement spherical coordinate mapping
- [ ] Add environment map struct to Scene

#### **1.2 Environment Integration**
- [ ] Replace solid background `(0.2, 0.7, 0.8)` with HDR sampling
- [ ] Update reflection rays to sample environment when missing scene
- [ ] Add environment lighting contribution
- [ ] Test with provided `studio_small_09_2k.exr`

#### **1.3 Performance Considerations**
- [ ] Bilinear interpolation for HDR sampling
- [ ] Consider MIP mapping for distant reflections
- [ ] Optimize environment lookups

### **Phase 2: Anti-Aliasing (MSAA)**
**Target: Smooth, production-quality rendering**

#### **2.1 Multi-Sample Anti-Aliasing**
- [ ] Add `--samples N` command line option
- [ ] Implement sub-pixel jittering (2x2, 4x4, 8x8 grids)
- [ ] Average multiple samples per pixel
- [ ] Maintain performance with parallel processing

#### **2.2 Sampling Patterns**
- [ ] Regular grid sampling (basic)
- [ ] Jittered sampling (better quality)
- [ ] Poisson disk sampling (advanced)

#### **2.3 Integration**
- [ ] Update main rendering loop for multi-sampling
- [ ] Add sample count to CLI arguments
- [ ] Test quality vs performance tradeoffs

## 🔧 **Technical Implementation Details**

### **HDR Environment Maps**

```rust
// New environment module
pub struct EnvironmentMap {
    data: Vec<Vector3>,    // HDR pixel data
    width: u32,
    height: u32,
}

impl EnvironmentMap {
    // Convert 3D ray direction to UV coordinates
    fn direction_to_uv(&self, dir: Vector3) -> (f32, f32) {
        let phi = dir.z.atan2(dir.x);
        let theta = dir.y.asin();
        let u = (phi / (2.0 * PI) + 0.5) % 1.0;
        let v = theta / PI + 0.5;
        (u, v)
    }

    // Sample environment map with bilinear filtering
    pub fn sample(&self, direction: Vector3) -> Vector3 {
        let (u, v) = self.direction_to_uv(direction);
        // Bilinear interpolation...
    }
}
```

### **Anti-Aliasing Implementation**

```rust
// Updated main rendering loop
let samples_per_pixel = args.samples;
framebuffer.par_iter_mut().enumerate().for_each(|(i, pixel)| {
    let x = i % WIDTH;
    let y = i / WIDTH;

    let mut color = Vector3::ZERO;
    for sample in 0..samples_per_pixel {
        // Jittered sub-pixel sampling
        let jitter_x = (sample % 2) as f32 * 0.5;
        let jitter_y = (sample / 2) as f32 * 0.5;

        let dir_x = (x as f32 + jitter_x) - WIDTH as f32 / 2.0;
        let dir_y = -(y as f32 + jitter_y) + HEIGHT as f32 / 2.0;
        let dir_z = -(HEIGHT as f32) / (2.0 * (FOV / 2.0).tan());

        let direction = Vector3::new(dir_x, dir_y, dir_z).normalized();
        color += cast_ray_with_params(&scene, Vector3::ZERO, direction, /* ... */);
    }
    *pixel = color / samples_per_pixel as f32;
});
```

## 📋 **Implementation Steps**

### **Step 1: HDR Support**
1. Add `exr` crate to Cargo.toml
2. Create `environment.rs` module
3. Implement HDR loading and sampling
4. Update renderer to use environment map
5. Test with studio HDR map

### **Step 2: Anti-Aliasing**
1. Add `--samples` CLI parameter
2. Implement jittered sampling in main loop
3. Test quality improvements
4. Optimize performance for high sample counts

### **Step 3: Integration**
1. Combine HDR + MSAA features
2. Update README with new options
3. Performance benchmarking
4. Visual quality comparison

## 🎨 **Expected Results**

### **HDR Environment Maps**
- **Before**: Flat blue background in reflections
- **After**: Realistic studio lighting in mirrors/glass
- **Impact**: 80% visual realism improvement

### **Anti-Aliasing**
- **Before**: Jagged sphere edges, aliased shadows
- **After**: Smooth, professional-quality rendering
- **Impact**: 60% quality improvement, 4x render time increase

## 🚀 **Future Enhancements**

### **Phase 3: Advanced Features**
- [ ] **Importance Sampling**: Sample environment based on brightness
- [ ] **Temporal Anti-Aliasing**: Accumulate samples over time
- [ ] **Adaptive Sampling**: More samples in complex areas
- [ ] **Environment Rotation**: Rotate HDR map for lighting control

### **Phase 4: Performance**
- [ ] **GPU Compute**: Port critical loops to compute shaders
- [ ] **Tile-Based Rendering**: Process image in tiles
- [ ] **Progressive Rendering**: Show intermediate results
- [ ] **Denoising**: AI-based noise reduction

## 📊 **Success Metrics**

### **Quality Targets**
- [ ] Realistic reflections showing environment details
- [ ] Smooth sphere edges with no visible aliasing
- [ ] Professional render quality comparable to commercial tools

### **Performance Targets**
- [ ] HDR environment: <20% performance overhead
- [ ] 4x MSAA: Complete render in <30 seconds
- [ ] Maintain parallel efficiency across all CPU cores

### **Usability Targets**
- [ ] Simple CLI: `cargo run -- --hdr studio.exr --samples 4`
- [ ] Clear documentation for new features
- [ ] Automatic HDR detection and loading

## 🔧 **Dependencies to Add**

```toml
[dependencies]
# Existing
rayon = "1.7"
glam = "0.24"
image = "0.24"
clap = { version = "4.0", features = ["derive"] }

# New for HDR
exr = "1.7"          # EXR/HDR image loading
# OR
openexr = "1.0"      # Alternative EXR loader

# For advanced sampling patterns
rand = "0.8"         # Random number generation
```

## 📈 **Timeline Estimate**

- **HDR Environment Maps**: 4-6 hours
- **Anti-Aliasing Implementation**: 2-3 hours
- **Integration & Testing**: 1-2 hours
- **Documentation Updates**: 1 hour

**Total: 8-12 hours for complete implementation**

This plan transforms the raytracer from a tech demo into a tool capable of producing professional-quality renders with realistic lighting and smooth anti-aliased output.