# Bug Hunt Report: nanotracer-rs

## Executive Summary

Comprehensive code analysis of the nanotracer-rs ray tracer revealed **no critical bugs**, but identified several areas for improvement including dead code, code duplication, excessive function parameters, and minor code style issues. The codebase is well-structured and compiles cleanly.

---

## 1. Dead/Unused Code

### 1.1 SIMD Renderer Module - COMPLETELY UNUSED
**Location:** `src/simd_renderer.rs`
**Status:** Dead code - never called
**Impact:** Low (no runtime effect)

```rust
// These functions are defined but never used anywhere in the codebase:
pub fn cast_ray_simd(...)     // line 8
pub fn cast_ray_simd8(...)    // line 45
```

**Analysis:** The module was created as part of optimization efforts (documented in `done.md`) but:
- Functions only wrap the scalar `cast_ray_with_params` in a loop
- No actual SIMD vectorization is performed
- Module is exported in `lib.rs` but never imported elsewhere

**Recommendation:**
- [ ] Option A: Remove module entirely (clean codebase)
- [ ] Option B: Implement real SIMD (requires significant work)

### 1.2 Unused Public Items

| Item | Location | Used? |
|------|----------|-------|
| `MATTE_WHITE` | `material.rs:51` | Never imported |
| `add_sphere()` | `scene.rs:98` | Never called (legacy API) |
| `cast_ray()` | `renderer.rs:39` | Never called (wrapper) |
| `Sphere::new()` | `geometry.rs:119` | Never called |
| `Object::new()` | `geometry.rs:25` | Never called |
| `tonemap_reinhard()` | `color.rs:5` | Only used internally |
| `linear_to_srgb()` | `color.rs:10` | Only used internally |
| `MAX_DEPTH` constants | `renderer.rs:10-14` | Used only as defaults |

**Recommendation:**
- [ ] Remove `MATTE_WHITE` or add usage
- [ ] Mark legacy APIs with `#[deprecated]` or remove
- [ ] Consider making internal functions `pub(crate)` instead of `pub`

---

## 2. Code Duplication

### 2.1 Tile Size Calculation - DUPLICATED
**Locations:** `main.rs:255-267` and `main.rs:303-315`

The exact same tile size auto-selection logic appears twice:

```rust
// First occurrence: lines 255-267
let tile_size = if args.tile_size == 0 {
    let cpu_cores = num_cpus::get();
    match cpu_cores {
        0..=2 => 32,
        // ...
    }
} else {
    args.tile_size.clamp(1, WIDTH.min(HEIGHT))
};

// Second occurrence: lines 303-315 (IDENTICAL)
let tile_size = if args.tile_size == 0 {
    // same code...
};
```

**Recommendation:**
- [ ] Extract into a helper function: `fn auto_tile_size(requested: usize, max: usize) -> usize`

### 2.2 Shadow Origin Offset Pattern - REPEATED 5 TIMES
**Locations:**
- `renderer.rs:132-136` (reflection)
- `renderer.rs:152-157` (refraction)
- `renderer.rs:184-188` (shadow check)
- `renderer.rs:285-289` (diffuse shadow)
- `splat/sampler.rs:272-277`, `308-312`, `328-332`

```rust
// This pattern appears 5+ times:
let shadow_orig = if dir.dot(normal) < 0.0 {
    point - normal * 1e-3
} else {
    point + normal * 1e-3
};
```

**Recommendation:**
- [ ] Extract helper: `fn offset_origin(point: Vec3, normal: Vec3, dir: Vec3) -> Vec3`

### 2.3 Light Sampling Code - DUPLICATED
**Locations:**
- `renderer.rs:176-205` (main renderer)
- `renderer.rs:277-296` (diffuse-only renderer)
- `splat/sampler.rs:268-294` (splat sampler)

**Recommendation:**
- [ ] Consider extracting common lighting calculation

---

## 3. Type System Issues

### 3.1 Vector3 Type Alias - CONFUSING
**Location:** `src/vec3.rs`

```rust
pub type Vector3 = Vec3;  // Just an alias for glam::Vec3
```

**Used in:** `environment.rs`, `utils.rs`
**Not used in:** All other modules (they use `Vec3` directly)

**Problems:**
- Inconsistent usage across modules
- No added value over using `glam::Vec3` directly
- Creates confusion about whether they're the same type

**Recommendation:**
- [ ] Replace all `Vector3` with `Vec3` for consistency
- [ ] Remove `vec3.rs` module entirely

---

## 4. Function Signature Issues

### 4.1 Too Many Arguments (Clippy Warning)
**Locations:**
- `cast_ray_with_params()` - 9 arguments
- `cast_ray_with_separate_depths()` - 9 arguments
- `adaptive_sample_pixel()` - 10 arguments

**Recommendation:**
Create a `RayConfig` struct:
```rust
pub struct RayConfig {
    pub max_depth: i32,
    pub reflection_depth: i32,
    pub refraction_depth: i32,
}
```

---

## 5. Clippy Warnings Summary

| Category | Count | Files |
|----------|-------|-------|
| Excessive float precision | 14 | `splat/sh.rs` |
| Collapsible if statements | 6 | `mesh.rs`, `scene.rs` |
| Manual div_ceil | 2 | `main.rs` |
| Too many arguments | 3 | `renderer.rs`, `main.rs` |
| Needless range loop | 4 | `splat/sh.rs` |
| While-let-on-iterator | 1 | `mesh.rs` |
| Manual memcpy | 1 | `splat/sh.rs` |
| len_zero | 1 | `splat/sampler.rs` |

**Fix Command:**
```bash
cargo clippy --fix --lib -p nanotracer-rs
cargo clippy --fix --bin "nanotracer-rs" -p nanotracer-rs
```

---

## 6. Dataflow Diagram

```
                              +----------------+
                              |   main.rs      |
                              |   CLI Args     |
                              +-------+--------+
                                      |
         +----------------------------+----------------------------+
         |                            |                            |
         v                            v                            v
+--------+--------+          +--------+--------+          +--------+--------+
|  Scene Setup    |          | Render Mode     |          | Splat Mode      |
|  add_object()   |          | cast_ray_*()    |          | generate_splats |
+--------+--------+          +--------+--------+          +--------+--------+
         |                            |                            |
         v                            v                            v
+--------+--------+          +--------+--------+          +--------+--------+
|  Scene          |<-------->| renderer.rs     |<-------->| splat/sampler   |
|  - objects[]    |          | - cast_ray()    |          | - sample_scene  |
|  - lights[]     |          | - reflect()     |          | - trace_incoming|
|  - environment  |          | - refract()     |          +--------+--------+
|  - scene_bvh    |          +--------+--------+                   |
+--------+--------+                   |                            v
         |                            v                   +--------+--------+
         v                   +--------+--------+          | splat/sh.rs     |
+--------+--------+          |  Intersection   |          | - fit_sh()      |
| geometry.rs     |          |  - hit          |          | - sh_basis()    |
| - Geometry enum |          |  - point        |          +--------+--------+
| - Object        |          |  - normal       |                   |
| - intersect()   |          |  - material     |                   v
+--------+--------+          +--------+--------+          +--------+--------+
         |                                                | splat/ply.rs    |
         v                                                | - write_ply()   |
+--------+--------+                                       +--------+--------+
| mesh.rs         |                                                |
| - Mesh          |                                                v
| - BVH (rtbvh)   |                                       +--------+--------+
| - cube/torus   |                                        |  output.ply     |
+-----------------+                                       +-----------------+

+--------+--------+          +--------+--------+
| environment.rs  |          | material.rs     |
| - EnvironmentMap|          | - Material      |
| - procedural_sky|          | - IVORY, GLASS  |
| - sample()      |          | - MIRROR, etc   |
+-----------------+          +-----------------+

+--------+--------+          +--------+--------+
| utils.rs        |          | color.rs        |
| - save_image()  |<---------| - tonemap       |
| - vec3_to_rgb() |          | - linear_to_srgb|
+-----------------+          +-----------------+

+--------+--------+
| simd_renderer   |  <-- DEAD CODE (never called)
| - cast_ray_simd |
+-----------------+
```

---

## 7. Architecture Issues

### 7.1 BVH Duplication
**Problem:** Two separate BVH systems exist:
1. **Per-mesh BVH** in `mesh.rs` using `rtbvh`
2. **Scene-level BVH** in `scene.rs` (custom implementation)

**Recommendation:**
- [ ] Consider unifying BVH implementation using `rtbvh` for both

### 7.2 Renderer Duplication
**Problem:** Three ray tracing code paths:
1. `cast_ray_with_separate_depths()` - full path
2. `cast_ray_diffuse_only()` - optimized path
3. `trace_incoming()` in sampler - duplicates lighting logic

**Recommendation:**
- [ ] Extract common lighting/shading into shared functions

---

## 8. Action Items

### High Priority (Code Quality)
- [ ] Remove or implement `simd_renderer.rs`
- [ ] Fix 28 clippy warnings
- [ ] Deduplicate tile_size calculation

### Medium Priority (Architecture)
- [ ] Unify Vector3/Vec3 usage
- [ ] Create RayConfig struct for too-many-arguments
- [ ] Extract offset_origin helper

### Low Priority (Cleanup)
- [ ] Remove unused MATTE_WHITE
- [ ] Remove legacy Sphere::new(), add_sphere() APIs
- [ ] Make internal functions pub(crate)

---

## 9. Memory Bank Entry

For context compactification survival, key findings:

```
nanotracer-rs Bug Hunt Summary:
- No critical bugs found
- simd_renderer.rs is DEAD CODE (never used)
- 28 clippy warnings (mostly style)
- Tile size calc duplicated in main.rs
- Shadow origin offset pattern repeated 5x
- Vector3 vs Vec3 inconsistency
- Too many function arguments (up to 10)
- Two BVH implementations (mesh + scene)
```

---

**Report Generated:** 2025-12-20
**Analyzed Files:** 15 source files
**Lines of Code:** ~2500 (excluding tests)
**Compile Status:** Clean (0 errors, 0 rustc warnings)
**Clippy Status:** 31 warnings (28 lib + 3 bin)
