use bytemuck::{Pod, Zeroable};
use glam::Vec3;

use nano_core::geometry::Geometry;
use nano_core::material::{checkerboard_material, Material};
use nano_core::mesh::Mesh;
use nano_core::scene::Scene;

pub const MATERIAL_FLAG_CHECKERBOARD: u32 = 1;

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

#[derive(Debug)]
pub struct GpuSceneData {
    pub vertices: Vec<[f32; 4]>,
    pub normals: Vec<[f32; 4]>,
    pub triangles: Vec<GpuTriangle>,
    pub tri_materials: Vec<u32>,
    pub tri_cdf: Vec<f32>,
    pub tri_areas: Vec<f32>,
    pub materials: Vec<GpuMaterial>,
    pub lights: Vec<[f32; 4]>,
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

    let lights = scene
        .lights
        .iter()
        .map(|l| [l.position.x, l.position.y, l.position.z, 1.0])
        .collect::<Vec<_>>();

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
