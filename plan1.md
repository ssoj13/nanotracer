# NANOTRACER-RS: Bug Hunt Report & Refactoring Plan

**Date:** 2025-12-17
**Status:** Awaiting Approval
**Compiler:** Clean (no warnings)

---

## EXECUTIVE SUMMARY

Analysis identified **15 issues** across 6 categories:
- **3 HIGH** severity (bugs, major dead code)
- **7 MEDIUM** severity (dead code, duplication)
- **5 LOW** severity (minor cleanup)

Key findings:
1. **Critical Bug:** `refract()` in renderer.rs returns red vector instead of reflection
2. **Major Dead Code:** `ray.rs` module completely unused (~26 lines)
3. **Code Duplication:** `reflect()`/`refract()` duplicated between modules
4. **Architecture Debt:** Dual object storage system (legacy spheres + new objects)

---

## DATA FLOW DIAGRAM

```
                          main.rs
                             |
              +--------------+--------------+
              |                             |
         [Render Mode]              [Splat Mode]
              |                             |
              v                             v
    +-------------------+         +-------------------+
    |  Scene Setup      |         |  Scene Setup      |
    |  - add_sphere()   |         |  - add_sphere()   |
    |  - add_object()   |         |  - add_object()   |
    |  - add_light()    |         |  - add_light()    |
    +-------------------+         +-------------------+
              |                             |
              v                             v
    +-------------------+         +-------------------+
    |  Parallel Render  |         |  sample_scene()   |
    |  (rayon)          |         |  sampler.rs       |
    +-------------------+         +-------------------+
              |                             |
              v                             |
    +-------------------+                   |
    | cast_ray_with_    |<------------------+
    | params()          |                   |
    | renderer.rs       |                   |
    +-------------------+                   |
              |                             v
              v                   +-------------------+
    +-------------------+         |  fit_sh()         |
    | scene.intersect() |         |  sh.rs            |
    +-------------------+         +-------------------+
              |                             |
     +--------+--------+                    v
     |                 |          +-------------------+
     v                 v          |  Gaussian::from_  |
[objects]        [spheres]        |  sample()         |
(new API)        (legacy)         +-------------------+
     |                 |                    |
     +--------+--------+                    v
              |                   +-------------------+
              v                   |  write_ply()      |
    +-------------------+         |  ply.rs           |
    | Intersection      |         +-------------------+
    | result            |                   |
    +-------------------+                   v
              |                       [output.ply]
              v
    +-------------------+
    | save_image()      |
    | utils.rs          |
    +-------------------+
              |
              v
        [output.png]
```

---

## ISSUE DETAILS

### CATEGORY 1: CRITICAL BUGS

#### [BUG-001] refract() returns red vector on total internal reflection
- **File:** `src/renderer.rs:30`
- **Severity:** HIGH
- **Impact:** Incorrect rendering of glass/refractive materials

**Current Code:**
```rust
if k < 0.0 {
    Vec3::new(1.0, 0.0, 0.0) // Total internal reflection - WRONG!
}
```

**Correct Code (from sampler.rs:384-386):**
```rust
if k < 0.0 {
    reflect(i, n) // Total internal reflection - CORRECT
}
```

**Fix:** Replace line 30 with `reflect(i, n)`

---

### CATEGORY 2: DEAD CODE (HIGH PRIORITY)

#### [DEAD-001] ray.rs - Entire module unused
- **File:** `src/ray.rs` (26 lines)
- **Severity:** HIGH
- **Evidence:**
  - No `use crate::ray` anywhere in codebase
  - All ray operations use `(origin, direction)` tuple pairs
  - Only internal reference is self-definition

**Action:** Delete `src/ray.rs`, remove `pub mod ray;` from `lib.rs`

---

#### [DEAD-002] vec3.rs - Vec3Ext trait unused
- **File:** `src/vec3.rs:16-32`
- **Severity:** MEDIUM
- **Evidence:**
  - Only imported in dead `ray.rs`
  - Duplicates `glam::Vec3` methods: `length()`, `normalize()`

**Action:** Delete `Vec3Ext` trait and implementation

---

#### [DEAD-003] vec3.rs - Constants unused
- **File:** `src/vec3.rs:9-13`
- **Severity:** LOW
- **Evidence:**
  - Code uses `Vec3::ZERO`, `Vec3::ONE`, `Vec3::Y` directly from glam
  - `ZERO`, `ONE`, `X_AXIS`, `Y_AXIS`, `Z_AXIS` never referenced

**Action:** Delete constants

---

#### [DEAD-004] geometry.rs - ray_sphere_intersect() unused
- **File:** `src/geometry.rs:137-143`
- **Severity:** MEDIUM
- **Evidence:** Only definition found, no calls

**Action:** Delete function

---

#### [DEAD-005] scene.rs - all_objects() unused
- **File:** `src/scene.rs:190-196`
- **Severity:** MEDIUM
- **Evidence:** Only definition found, no calls

**Action:** Delete method

---

#### [DEAD-006] splat/mod.rs - rest_normalized() unused
- **File:** `src/splat/mod.rs:68-88`
- **Severity:** MEDIUM
- **Evidence:**
  - Only `rest_interleaved()` is called (line 121)
  - `rest_normalized()` never called

**Action:** Delete method

---

#### [DEAD-007] splat/sh.rs - stratified_hemisphere() unused
- **File:** `src/splat/sh.rs:265-293`
- **Severity:** MEDIUM
- **Evidence:**
  - Only `fibonacci_hemisphere()` is called
  - `stratified_hemisphere()` never called

**Action:** Delete function

---

#### [DEAD-008] splat/ply.rs - estimate_file_size() unused
- **File:** `src/splat/ply.rs:133-141`
- **Severity:** LOW
- **Evidence:** Only definition found, no calls

**Action:** Delete function (or keep for public API)

---

### CATEGORY 3: CODE DUPLICATION

#### [DUP-001] reflect() duplicated
- **Files:**
  - `src/renderer.rs:15-17` (pub)
  - `src/splat/sampler.rs:367-369` (private)
- **Severity:** MEDIUM

**Action:** Remove duplicate from `sampler.rs`, use `crate::renderer::reflect`

---

#### [DUP-002] refract() duplicated with DIFFERENT logic
- **Files:**
  - `src/renderer.rs:20-34` (has bug)
  - `src/splat/sampler.rs:372-389` (correct)
- **Severity:** HIGH (inconsistent behavior)

**Differences:**
| Aspect | renderer.rs | sampler.rs |
|--------|-------------|------------|
| TIR handling | Returns `(1,0,0)` BUG | Calls `reflect()` CORRECT |
| Recursion | Uses recursion | Uses variable shadowing |

**Action:** Fix `renderer.rs`, then remove duplicate from `sampler.rs`

---

#### [DUP-003] Checkerboard material duplicated
- **Files:**
  - `src/scene.rs:180-185`
  - `src/splat/sampler.rs:209-214`
- **Severity:** LOW

**Action:** Create `checkerboard_material(color: Vec3) -> Material` function in `material.rs`

---

### CATEGORY 4: ARCHITECTURE DEBT

#### [ARCH-001] Dual object storage system
- **File:** `src/scene.rs:59-70`
- **Severity:** HIGH (technical debt)

**Current:**
```rust
pub struct Scene {
    pub objects: Vec<Object>,     // New API
    pub spheres: Vec<Sphere>,     // Legacy API
    // ...
}
```

**Problems:**
1. Double iteration in `intersect()` (lines 116-137)
2. Double iteration in `sample_scene()` (sampler.rs:36-51)
3. Temporary `Geometry::Sphere` creation per ray (inefficient)
4. Confusing dual API

**Solution:** Migrate all spheres to `objects` vector:
```rust
pub fn add_sphere(&mut self, sphere: Sphere) {
    self.objects.push(sphere.to_object());
}
```

---

### CATEGORY 5: CODE STYLE / MINOR

#### [STYLE-001] Intersection::new() only used once
- **File:** `src/scene.rs:25-32`
- **Severity:** LOW
- **Note:** Used only on line 145. Consider inlining or keeping for clarity.

---

## REFACTORING PLAN

### Phase 1: Bug Fixes (Priority: IMMEDIATE)
- [ ] Fix `renderer.rs:30` - replace red vector with `reflect(i, n)`

### Phase 2: Dead Code Removal (Priority: HIGH)
- [ ] Delete `src/ray.rs`
- [ ] Remove `pub mod ray;` from `lib.rs`
- [ ] Delete `Vec3Ext` trait from `vec3.rs`
- [ ] Delete unused constants from `vec3.rs`
- [ ] Delete `ray_sphere_intersect()` from `geometry.rs`
- [ ] Delete `all_objects()` from `scene.rs`
- [ ] Delete `rest_normalized()` from `splat/mod.rs`
- [ ] Delete `stratified_hemisphere()` from `splat/sh.rs`
- [ ] Delete `estimate_file_size()` from `splat/ply.rs` (optional)

### Phase 3: Code Deduplication (Priority: MEDIUM)
- [ ] Remove `reflect()` from `sampler.rs`, import from `renderer`
- [ ] Remove `refract()` from `sampler.rs`, import from `renderer`
- [ ] Create `checkerboard_material()` function in `material.rs`
- [ ] Update `scene.rs` and `sampler.rs` to use shared function

### Phase 4: Architecture Cleanup (Priority: MEDIUM)
- [ ] Modify `add_sphere()` to add to `objects` instead of `spheres`
- [ ] Remove `spheres` field from `Scene` struct
- [ ] Simplify `intersect()` method
- [ ] Update `sample_scene()` to iterate only `objects`
- [ ] Run tests to verify no regressions

### Phase 5: Final Cleanup (Priority: LOW)
- [ ] Run `cargo clippy` for additional suggestions
- [ ] Run `cargo fmt` for formatting
- [ ] Update documentation if needed

---

## METRICS

| Metric | Before | After (Estimated) |
|--------|--------|-------------------|
| Total .rs files | 15 | 14 (-1) |
| Lines of dead code | ~180 | 0 |
| Duplicated functions | 3 | 0 |
| Critical bugs | 1 | 0 |

---

## IMPLEMENTATION COMPLETE

**Status:** All phases completed successfully!

### Changes Applied:

#### Phase 1: Bug Fix
- [x] Fixed `renderer.rs:30` - `refract()` now returns `reflect(i, n)` on TIR

#### Phase 2: Dead Code Removal
- [x] Deleted `src/ray.rs` (26 lines)
- [x] Removed `pub mod ray;` from `lib.rs`
- [x] Deleted `Vec3Ext` trait and constants from `vec3.rs`
- [x] Deleted `ray_sphere_intersect()` from `geometry.rs`
- [x] Deleted `all_objects()` from `scene.rs`
- [x] Deleted `rest_normalized()` from `splat/mod.rs`
- [x] Deleted `stratified_hemisphere()` from `splat/sh.rs`
- [x] Deleted `estimate_file_size()` from `splat/ply.rs`

#### Phase 3: Code Deduplication
- [x] Removed duplicate `reflect()` from `sampler.rs`
- [x] Removed duplicate `refract()` from `sampler.rs`
- [x] Created `checkerboard_material()` in `material.rs`
- [x] Updated `scene.rs` and `sampler.rs` to use shared function

#### Phase 4: Architecture Unification
- [x] Removed `spheres: Vec<Sphere>` field from `Scene`
- [x] Modified `add_sphere()` to use `objects` vector
- [x] Simplified `intersect()` - single iteration loop
- [x] Simplified `sample_scene()` - single iteration loop

#### Phase 5: Final Cleanup
- [x] Fixed unused import warning
- [x] All 11 tests passing
- [x] Clean compilation

### Final Metrics:

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Source files | 15 | 14 | -1 |
| Dead code lines | ~180 | 0 | -180 |
| Duplicated functions | 3 | 0 | -3 |
| Critical bugs | 1 | 0 | -1 |
| Tests passing | 11/11 | 11/11 | OK |

---

*Refactoring completed by Claude Code*
