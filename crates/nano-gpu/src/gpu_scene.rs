use bytemuck::{Pod, Zeroable};
use glam::Vec3;

use nano_core::geometry::Geometry;
use nano_core::material::{checkerboard_material, Material};
use nano_core::mesh::Mesh;
use nano_core::scene::{Light, Scene};

pub const MATERIAL_FLAG_CHECKERBOARD: u32 = 1;

/// GLSL `LIGHT_*` constants — keep these in sync with `nano-shaders`.
pub const LIGHT_KIND_POINT: u32 = 0;
pub const LIGHT_KIND_RECT: u32 = 1;
pub const LIGHT_KIND_SPHERE: u32 = 2;
pub const LIGHT_KIND_BOX: u32 = 3;
pub const LIGHT_KIND_ENV: u32 = 4;

#[repr(C)]
#[derive(Clone, Copy, Debug, Zeroable, Pod)]
pub struct GpuMaterial {
    pub diffuse: [f32; 3],
    pub specular_exponent: f32,
    pub albedo: [f32; 4],
    pub refractive_index: f32,
    pub flags: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Zeroable, Pod)]
pub struct GpuTriangle {
    pub v0: u32,
    pub v1: u32,
    pub v2: u32,
    pub _pad: u32,
}

/// 64-byte std430 light record. One per scene light; semantics are
/// driven by `kind` (see `LIGHT_KIND_*` constants). Per-light emissive
/// radiance lives in a *parallel* SSBO (`light_radiance`) — this keeps
/// the geometric record at exactly 64 bytes without bit-packing tricks.
///
/// Field layout by kind:
///
/// | Kind   | `center`            | `axis_u`         | `axis_v`            |
/// |--------|---------------------|------------------|---------------------|
/// | Point  | position.xyz, _     | unused           | unused              |
/// | Rect   | center.xyz, _       | u.xyz, _         | v.xyz, _            |
/// | Sphere | center.xyz, radius  | unused           | unused              |
/// | Box    | center.xyz, _       | rotation (quat)  | half_extents.xyz, _ |
/// | Env    | unused              | unused           | unused              |
///
/// `u` and `v` for `Rect` are the world-space half-extent vectors (not
/// unit vectors): a 2×3 rectangle whose long side is along +X is
/// `u = (1, 0, 0), v = (0, 1.5, 0)`.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Zeroable, Pod)]
pub struct GpuLight {
    pub kind: u32,
    pub two_sided: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub center: [f32; 4],
    pub axis_u: [f32; 4],
    pub axis_v: [f32; 4],
}

const _: () = assert!(core::mem::size_of::<GpuLight>() == 64);

#[derive(Debug)]
pub struct GpuSceneData {
    pub vertices: Vec<[f32; 4]>,
    pub normals: Vec<[f32; 4]>,
    pub triangles: Vec<GpuTriangle>,
    pub tri_materials: Vec<u32>,
    pub tri_cdf: Vec<f32>,
    pub tri_areas: Vec<f32>,
    pub materials: Vec<GpuMaterial>,
    /// Geometric records (one per scene light, or a single zero pad if
    /// the scene has no lights — Vulkan rejects zero-byte SSBOs).
    /// `lights[i]` shares the same index as `light_radiance[i]`.
    pub lights: Vec<GpuLight>,
    /// Emitted radiance per light (color · intensity, premultiplied).
    /// Parallel to `lights`; the w channel is unused / padding.
    pub light_radiance: Vec<[f32; 4]>,
    /// Logical light count — distinct from `lights.len()` because the
    /// buffer is padded to 1 entry when the scene has no lights and no
    /// environment. Shaders use this for the loop bound.
    pub light_count: u32,
}

impl GpuLight {
    /// Convert a CPU [`Light`] into its 64-byte GPU record. The matching
    /// radiance entry comes from [`Light::radiance`] on the caller side
    /// (see `build_gpu_scene_with_detail_boost`).
    pub fn from_light(light: &Light) -> Self {
        match light {
            Light::Point { position, .. } => Self {
                kind: LIGHT_KIND_POINT,
                two_sided: 0,
                _pad0: 0,
                _pad1: 0,
                center: [position.x, position.y, position.z, 0.0],
                axis_u: [0.0; 4],
                axis_v: [0.0; 4],
            },
            Light::Rect {
                center,
                u,
                v,
                two_sided,
                ..
            } => Self {
                kind: LIGHT_KIND_RECT,
                two_sided: u32::from(*two_sided),
                _pad0: 0,
                _pad1: 0,
                center: [center.x, center.y, center.z, 0.0],
                axis_u: [u.x, u.y, u.z, 0.0],
                axis_v: [v.x, v.y, v.z, 0.0],
            },
            Light::Sphere { center, radius, .. } => Self {
                kind: LIGHT_KIND_SPHERE,
                two_sided: 0,
                _pad0: 0,
                _pad1: 0,
                center: [center.x, center.y, center.z, *radius],
                axis_u: [0.0; 4],
                axis_v: [0.0; 4],
            },
            Light::Box {
                center,
                half_extents,
                rotation,
                ..
            } => Self {
                kind: LIGHT_KIND_BOX,
                two_sided: 0,
                _pad0: 0,
                _pad1: 0,
                center: [center.x, center.y, center.z, 0.0],
                axis_u: [rotation.x, rotation.y, rotation.z, rotation.w],
                axis_v: [half_extents.x, half_extents.y, half_extents.z, 0.0],
            },
            Light::Env { .. } => Self {
                kind: LIGHT_KIND_ENV,
                two_sided: 0,
                _pad0: 0,
                _pad1: 0,
                center: [0.0; 4],
                axis_u: [0.0; 4],
                axis_v: [0.0; 4],
            },
        }
    }
}

impl GpuMaterial {
    pub fn from_material(material: Material, flags: u32) -> Self {
        Self {
            diffuse: [material.diffuse_color.x, material.diffuse_color.y, material.diffuse_color.z],
            specular_exponent: material.specular_exponent,
            albedo: material.albedo,
            refractive_index: material.refractive_index,
            flags,
            _pad0: 0,
            _pad1: 0,
        }
    }
}

pub fn build_gpu_scene(scene: &Scene) -> GpuSceneData {
    build_gpu_scene_with_detail_boost(scene, 1.5, 3.0)
}

pub fn build_gpu_scene_with_detail_boost(
    scene: &Scene,
    detail_boost: f32,
    max_boost: f32,
) -> GpuSceneData {
    let mut vertices: Vec<[f32; 4]> = Vec::new();
    let mut normals: Vec<[f32; 4]> = Vec::new();
    let mut triangles: Vec<GpuTriangle> = Vec::new();
    let mut tri_materials: Vec<u32> = Vec::new();
    let mut tri_areas: Vec<f32> = Vec::new();
    let mut tri_weights: Vec<f32> = Vec::new();
    let mut materials: Vec<GpuMaterial> = Vec::new();

    for object in &scene.objects {
        let material_index = materials.len() as u32;
        materials.push(GpuMaterial::from_material(object.material, 0));

        match &object.geometry {
            Geometry::Sphere { center, radius } => {
                let mesh = sphere_mesh(*center, *radius, 24, 16);
                append_mesh(
                    &mesh,
                    material_index,
                    &mut vertices,
                    &mut normals,
                    &mut triangles,
                    &mut tri_materials,
                    &mut tri_areas,
                    detail_boost,
                    max_boost,
                    &mut tri_weights,
                );
            }
            Geometry::Mesh(mesh) => {
                append_mesh(
                    mesh,
                    material_index,
                    &mut vertices,
                    &mut normals,
                    &mut triangles,
                    &mut tri_materials,
                    &mut tri_areas,
                    detail_boost,
                    max_boost,
                    &mut tri_weights,
                );
            }
        }
    }

    if scene.checkerboard_enabled {
        let checker_material = checkerboard_material(Vec3::new(0.3, 0.3, 0.3));
        let material_index = materials.len() as u32;
        materials.push(GpuMaterial::from_material(
            checker_material,
            MATERIAL_FLAG_CHECKERBOARD,
        ));
        let plane = checkerboard_plane_mesh();
        append_mesh(
            &plane,
            material_index,
            &mut vertices,
            &mut normals,
            &mut triangles,
            &mut tri_materials,
            &mut tri_areas,
            detail_boost,
            max_boost,
            &mut tri_weights,
        );
    }

    // Build the GPU light table. If the scene has an environment map but
    // no explicit `Light::Env`, append an implicit unit-intensity one so
    // image-based lighting still contributes — matches the pre-refactor
    // unconditional `eval_env_irradiance` behaviour. The CPU `Scene` is
    // not mutated; this is a build-time policy.
    let auto_env = scene.environment.is_some() && !scene.has_env_light();
    let mut lights: Vec<GpuLight> = scene.lights.iter().map(GpuLight::from_light).collect();
    let mut light_radiance: Vec<[f32; 4]> = scene
        .lights
        .iter()
        .map(|l| {
            let r = l.radiance();
            [r.x, r.y, r.z, 0.0]
        })
        .collect();
    if auto_env {
        let env_light = Light::Env { intensity: 1.0 };
        lights.push(GpuLight::from_light(&env_light));
        let r = env_light.radiance();
        light_radiance.push([r.x, r.y, r.z, 0.0]);
    }
    let light_count = lights.len() as u32;
    // Vulkan rejects zero-byte SSBOs — pad to one zeroed slot when the
    // scene has no lights and no environment. `light_count` keeps the
    // logical count so the shader skips the dummy entry.
    if lights.is_empty() {
        lights.push(GpuLight::zeroed());
        light_radiance.push([0.0; 4]);
    }

    let mut tri_cdf = Vec::with_capacity(tri_weights.len());
    let total_weight: f32 = tri_weights.iter().sum();
    let mut accum = 0.0f32;
    if total_weight > 0.0 {
        for weight in &tri_weights {
            accum += *weight / total_weight;
            tri_cdf.push(accum);
        }
    }

    GpuSceneData {
        vertices,
        normals,
        triangles,
        tri_materials,
        tri_cdf,
        tri_areas,
        materials,
        lights,
        light_radiance,
        light_count,
    }
}

#[allow(clippy::too_many_arguments)]
fn append_mesh(
    mesh: &Mesh,
    material_index: u32,
    vertices: &mut Vec<[f32; 4]>,
    normals: &mut Vec<[f32; 4]>,
    triangles: &mut Vec<GpuTriangle>,
    tri_materials: &mut Vec<u32>,
    tri_areas: &mut Vec<f32>,
    detail_boost: f32,
    max_boost: f32,
    tri_weights: &mut Vec<f32>,
) {
    let base = vertices.len() as u32;
    vertices.extend(mesh.vertices.iter().map(|v| [v.x, v.y, v.z, 1.0]));
    let has_normals = mesh.normals.len() == mesh.vertices.len();
    if mesh.normals.len() == mesh.vertices.len() {
        normals.extend(mesh.normals.iter().map(|n| [n.x, n.y, n.z, 0.0]));
    } else {
        normals.extend(std::iter::repeat_n([0.0, 1.0, 0.0, 0.0], mesh.vertices.len()));
    }

    for tri in &mesh.indices {
        triangles.push(GpuTriangle {
            v0: base + tri[0],
            v1: base + tri[1],
            v2: base + tri[2],
            _pad: 0,
        });
        tri_materials.push(material_index);
        let v0 = mesh.vertices[tri[0] as usize];
        let v1 = mesh.vertices[tri[1] as usize];
        let v2 = mesh.vertices[tri[2] as usize];
        let face_vec = (v1 - v0).cross(v2 - v0);
        let area = face_vec.length() * 0.5;
        tri_areas.push(area.max(0.0));

        let mut weight = area.max(0.0);
        if has_normals && weight > 0.0 {
            // Curvature estimate: pairwise vertex-normal disagreement is
            // larger when the triangle bends sharply between its corners.
            // `1 - n_i · n_j` is 0 on a flat triangle, 1 at a 90° crease,
            // 2 at a fold — `max` over the three pairs picks up edges that
            // bend even if the other pair is flat. This catches "rough"
            // regions of a smoothed mesh better than the old face/vertex
            // delta which always reads 0 on convex-only spheres.
            let n0 = mesh.normals[tri[0] as usize].normalize_or_zero();
            let n1 = mesh.normals[tri[1] as usize].normalize_or_zero();
            let n2 = mesh.normals[tri[2] as usize].normalize_or_zero();
            let d01 = 1.0 - n0.dot(n1).clamp(-1.0, 1.0);
            let d02 = 1.0 - n0.dot(n2).clamp(-1.0, 1.0);
            let d12 = 1.0 - n1.dot(n2).clamp(-1.0, 1.0);
            let curvature = d01.max(d02).max(d12);
            let boost = (1.0 + curvature * detail_boost).clamp(0.5, max_boost);
            weight *= boost;
        }
        tri_weights.push(weight.max(0.0));
    }
}

fn sphere_mesh(center: Vec3, radius: f32, slices: u32, stacks: u32) -> Mesh {
    let mut vertices = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();

    for i in 0..=stacks {
        let v = i as f32 / stacks as f32;
        let phi = v * std::f32::consts::PI;
        let y = phi.cos();
        let r = phi.sin();

        for j in 0..=slices {
            let u = j as f32 / slices as f32;
            let theta = u * std::f32::consts::TAU;
            let x = r * theta.cos();
            let z = r * theta.sin();
            let n = Vec3::new(x, y, z).normalize_or_zero();
            let pos = center + n * radius;
            vertices.push(pos);
            normals.push(n);
        }
    }

    let stride = slices + 1;
    for i in 0..stacks {
        for j in 0..slices {
            let a = i * stride + j;
            let b = (i + 1) * stride + j;
            let c = (i + 1) * stride + j + 1;
            let d = i * stride + j + 1;
            indices.push([a, b, c]);
            indices.push([a, c, d]);
        }
    }

    Mesh::with_normals(vertices, indices, normals)
}

fn checkerboard_plane_mesh() -> Mesh {
    let y = -4.0;
    let x0 = -10.0;
    let x1 = 10.0;
    let z0 = -30.0;
    let z1 = -10.0;

    let vertices = vec![
        Vec3::new(x0, y, z0),
        Vec3::new(x1, y, z0),
        Vec3::new(x1, y, z1),
        Vec3::new(x0, y, z1),
    ];

    let normals = vec![Vec3::Y, Vec3::Y, Vec3::Y, Vec3::Y];
    let indices = vec![[0, 1, 2], [0, 2, 3]];

    Mesh::with_normals(vertices, indices, normals)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Quat, Vec3};

    #[test]
    fn gpu_light_layout_is_64_bytes_aligned_16() {
        assert_eq!(core::mem::size_of::<GpuLight>(), 64);
        assert_eq!(core::mem::align_of::<GpuLight>(), 16);
    }

    #[test]
    fn from_light_kind_dispatch() {
        let p = Light::Point {
            position: Vec3::new(1.0, 2.0, 3.0),
            color: Vec3::ONE,
            intensity: 1.0,
        };
        let g = GpuLight::from_light(&p);
        assert_eq!(g.kind, LIGHT_KIND_POINT);
        assert_eq!(g.center, [1.0, 2.0, 3.0, 0.0]);

        let r = Light::Rect {
            center: Vec3::new(0.0, 5.0, 0.0),
            u: Vec3::new(2.0, 0.0, 0.0),
            v: Vec3::new(0.0, 0.0, 3.0),
            color: Vec3::ONE,
            intensity: 1.0,
            two_sided: true,
        };
        let g = GpuLight::from_light(&r);
        assert_eq!(g.kind, LIGHT_KIND_RECT);
        assert_eq!(g.two_sided, 1);
        assert_eq!(g.axis_u, [2.0, 0.0, 0.0, 0.0]);
        assert_eq!(g.axis_v, [0.0, 0.0, 3.0, 0.0]);

        let s = Light::Sphere {
            center: Vec3::ZERO,
            radius: 4.0,
            color: Vec3::ONE,
            intensity: 1.0,
        };
        let g = GpuLight::from_light(&s);
        assert_eq!(g.kind, LIGHT_KIND_SPHERE);
        assert_eq!(g.center[3], 4.0);

        let b = Light::Box {
            center: Vec3::ZERO,
            half_extents: Vec3::new(1.0, 2.0, 3.0),
            rotation: Quat::IDENTITY,
            color: Vec3::ONE,
            intensity: 1.0,
        };
        let g = GpuLight::from_light(&b);
        assert_eq!(g.kind, LIGHT_KIND_BOX);
        assert_eq!(g.axis_u, [0.0, 0.0, 0.0, 1.0]); // identity quat (x,y,z,w)
        assert_eq!(g.axis_v, [1.0, 2.0, 3.0, 0.0]);

        let e = Light::Env { intensity: 2.0 };
        let g = GpuLight::from_light(&e);
        assert_eq!(g.kind, LIGHT_KIND_ENV);
    }
}
