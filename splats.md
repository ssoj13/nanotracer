# Gaussian Splatting: Анализ возможностей для рендера

## Обзор технологии

3D Gaussian Splatting (3DGS) - революционная техника для воксельного представления 3D-сцен без использования нейронных сетей. Каждый "сплат" - это 3D-гауссиан с позицией, ковариацией (scale + rotation) и цветом (сферические гармоники).

## Rust-экосистема для Gaussian Splatting

### Основные крейты

| Крейт | Описание | GPU Backend | Зрелость |
|-------|----------|-------------|----------|
| **bevy_gaussian_splatting** | Полноценный плагин для Bevy, viewer + редактор | wgpu | Высокая (v4.6) |
| **wgpu-3dgs-viewer** | Standalone viewer на wgpu | wgpu | Средняя (v0.4) |
| **wgpu-3dgs-editor** | Редактор splats с wgpu | wgpu | Новый (2025) |
| **splatt-rs** | Ранний эксперимент с wgpu | wgpu | Низкая |
| **web-splat** | WebGPU renderer + WASM | WebGPU | Средняя |
| **gauzilla** | WASM viewer с WebGL, lock-free multithreading | WebGL | Средняя |

### bevy_gaussian_splatting (рекомендуется)
- GitHub: [mosure/bevy_gaussian_splatting](https://github.com/mosure/bevy_gaussian_splatting)
- Поддержка PLY, GCloud форматов
- Radix sort на GPU
- Инструменты: `ply_to_gcloud`, `viewer`, `test_gaussian`
- MIT/Apache-2.0

### wgpu-3dgs-viewer
- GitHub: [LioQing/wgpu-3dgs-viewer](https://github.com/LioQing/wgpu-3dgs-viewer)
- Чистый Rust + wgpu
- Хорошая документация
- Подходит как библиотека

## Форматы файлов

### PLY (основной)
Стандартный формат для 3DGS. Каждый сплат содержит:

```
vertex properties:
  - x, y, z               : позиция (float32)
  - scale_0, scale_1, scale_2 : масштаб по осям (float32, log-space)
  - rot_0..rot_3          : кватернион ориентации (float32)
  - f_dc_0, f_dc_1, f_dc_2 : базовый цвет RGB (SH degree 0)
  - f_rest_0..f_rest_44   : сферические гармоники степеней 1-3
  - opacity               : прозрачность (float32, sigmoid)
```

**Размеры:** ~250 байт на сплат, типичная сцена 1-5M сплатов = 250MB-1.25GB

### .splat (компактный)
Упрощённый бинарный формат без SH коэффициентов высших порядков:
- 32 байта на сплат
- Только position + scale + rotation + RGBA
- ~10x компактнее PLY

### .spz (Niantic/Scaniverse)
- [Open-source формат](https://scaniverse.com/news/spz-gaussian-splat-open-source-file-format)
- 90% сжатие относительно PLY
- Квантизация + entropy coding

### glTF + KHR_gaussian_splatting
- [Официальное расширение glTF](https://digitalproduction.com/2025/09/02/3d-gaussian-splats-officially-added-to-gltf-standard/) (сентябрь 2025)
- Интеграция с существующей 3D-экосистемой

## Структура данных Gaussian Splat

### Сферические гармоники (SH)
View-dependent цвет кодируется через SH-коэффициенты:

| Степень | Коэффициентов | RGB каналов | Всего |
|---------|---------------|-------------|-------|
| 0 (DC)  | 1             | 3           | 3     |
| 1       | 3             | 3           | 9     |
| 2       | 5             | 3           | 15    |
| 3       | 7             | 3           | 21    |
| **Итого** | 16          | 3           | **48** |

SH позволяют моделировать:
- Specular highlights
- View-dependent reflections
- Non-Lambertian поверхности

### Ковариационная матрица
3D-гауссиан определяется ковариационной матрицей 3x3:
```
Σ = R * S * S^T * R^T
где:
  R = матрица вращения из кватерниона
  S = диагональная матрица масштаба
```

## Техники сэмплинга и рендера

### Multi-view training
- [MVSplat](https://arxiv.org/abs/2403.14627): Sparse multi-view → 3DGS за один проход
- [FastGS](https://fastgs.github.io/): Тренировка за ~100 секунд
- [Efficient multi-view training](https://arxiv.org/abs/2506.12727): Batch training вместо single-view

### Rendering pipeline
1. **View-frustum culling** - отбросить невидимые сплаты
2. **Sorting** - сортировка по глубине (radix sort на GPU)
3. **Projection** - 3D gaussian → 2D gaussian
4. **Rasterization** - alpha-blending front-to-back
5. **SH evaluation** - вычисление цвета для view direction

### Synthetic data generation
- [Blender Python automation](https://www.youtube.com/watch?v=c3KZX8BMYBU): Golden spiral camera placement
- [Gaussian-Splatterer](https://github.com/osreboot/Gaussian-Splatterer): CUDA ray-tracing из mesh
- [Cut-and-Splat](https://arxiv.org/abs/2504.08473): Генерация synthetic training data

## Рендер со всех сторон (360° capture)

### Подходы к сэмплингу камер:
1. **Golden spiral** - равномерное распределение точек на сфере
2. **Icosphere subdivision** - вершины икосаэдра
3. **Spherical grid** - регулярная сетка (θ, φ)
4. **Turntable + dome** - круговые проходы на разных высотах

### Минимальное количество видов:
- Простые объекты: 20-50 изображений
- Сложная геометрия: 100-200 изображений
- Indoor сцены: 200-500 изображений

## Возможности Rust

### Преимущества:
- **wgpu** - кроссплатформенный GPU backend (Vulkan/Metal/DX12/WebGPU)
- **Bevy ecosystem** - готовая инфраструктура для 3D
- **Performance** - zero-cost abstractions, SIMD
- **WASM** - браузерный рендеринг

### Недостатки:
- Нет ready-to-use training library (только Python gsplat/nerfstudio)
- Ограниченная документация по GS-форматам
- Молодая экосистема

## Реализация записи PLY

```rust
use std::io::Write;

struct Gaussian {
    pos: [f32; 3],
    scale: [f32; 3],      // log-space
    rot: [f32; 4],        // quaternion (w, x, y, z)
    sh_dc: [f32; 3],      // degree 0 SH (RGB)
    sh_rest: [f32; 45],   // degrees 1-3
    opacity: f32,         // logit-space
}

fn write_ply(path: &str, gaussians: &[Gaussian]) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    
    // Header
    writeln!(f, "ply")?;
    writeln!(f, "format binary_little_endian 1.0")?;
    writeln!(f, "element vertex {}", gaussians.len())?;
    writeln!(f, "property float x")?;
    writeln!(f, "property float y")?;
    writeln!(f, "property float z")?;
    // ... scale, rotation, SH properties ...
    writeln!(f, "end_header")?;
    
    // Binary data
    for g in gaussians {
        f.write_all(bytemuck::bytes_of(&g.pos))?;
        // ... остальные поля ...
    }
    Ok(())
}
```

## Интеграция с nanotracer

### Возможные подходы:

1. **Export path-traced renders → train external 3DGS**
   - Рендерим сцену из множества ракурсов
   - Экспортируем изображения + camera poses
   - Тренируем в gsplat/nerfstudio (Python)

2. **Direct splat generation from geometry**
   - Сэмплируем поверхности mesh/SDF
   - Каждый sample → gaussian
   - Оптимизируем параметры

3. **Hybrid: path trace → fit gaussians**
   - Path trace → ground truth images
   - Differentiable gaussian fitting (нужен autograd)

## Рекомендации

### Для просмотра готовых splats:
- `bevy_gaussian_splatting` - лучший выбор для Rust

### Для генерации splats из 3D-сцен:
- Рендерить multi-view images в nanotracer
- Экспортировать camera poses (transforms.json)
- Использовать gsplat (Python) для training

### Для записи PLY:
- Использовать `ply-rs` крейт или писать вручную
- Формат простой, header + binary data

---

# Реализация в nanotracer-rs

## Архитектура: Direct Geometry-to-Splat

Вместо training loop используем прямую конвертацию:
```
Geometry → Surface Samples → SH Fitting → PLY Export
```

### Почему это работает?

Training loop в 3DGS решает **inverse problem**: изображения → 3D.
У нас **forward problem**: 3D геометрия уже есть.

Гауссиан = точка + ориентация + размер + view-dependent цвет.
Всё это можно получить напрямую из геометрии + path tracing.

## CLI Interface

```bash
nanotracer-rs.exe --splats output.ply [options]

Options:
  -S, --splats <FILE>     Export Gaussian splats to PLY file
  --sh-samples <N>        SH sampling directions per splat (default: 64)
  --splat-density <N>     Surface samples per unit area (default: 100)
  --sh-degree <0-3>       Max SH degree (default: 3)
```

## Модули

### 1. `src/splat/mod.rs` - Структуры данных

```rust
/// Single Gaussian splat with full SH coefficients
#[repr(C)]
pub struct Gaussian {
    pub pos: [f32; 3],          // world position
    pub normal: [f32; 3],       // surface normal (nx, ny, nz)
    pub sh_dc: [f32; 3],        // SH degree 0 (base RGB)
    pub sh_1: [[f32; 3]; 3],    // SH degree 1 (9 coeffs)
    pub sh_2: [[f32; 3]; 5],    // SH degree 2 (15 coeffs)
    pub sh_3: [[f32; 3]; 7],    // SH degree 3 (21 coeffs)
    pub opacity: f32,           // logit-space opacity
    pub scale: [f32; 3],        // log-space scale
    pub rotation: [f32; 4],     // quaternion (w, x, y, z)
}
```

### 2. `src/splat/sh.rs` - Spherical Harmonics

Real spherical harmonics до degree 3:

```rust
/// SH basis functions (real form, Condon-Shortley convention)
pub fn sh_basis(l: u32, m: i32, dir: Vec3) -> f32 {
    let (x, y, z) = (dir.x, dir.y, dir.z);
    match (l, m) {
        // Degree 0
        (0, 0) => 0.28209479,  // Y_0^0 = 1/(2*sqrt(pi))
        
        // Degree 1
        (1, -1) => 0.48860251 * y,           // Y_1^{-1}
        (1,  0) => 0.48860251 * z,           // Y_1^0
        (1,  1) => 0.48860251 * x,           // Y_1^1
        
        // Degree 2
        (2, -2) => 1.09254843 * x * y,                    // Y_2^{-2}
        (2, -1) => 1.09254843 * y * z,                    // Y_2^{-1}
        (2,  0) => 0.31539157 * (3.0*z*z - 1.0),          // Y_2^0
        (2,  1) => 1.09254843 * x * z,                    // Y_2^1
        (2,  2) => 0.54627421 * (x*x - y*y),              // Y_2^2
        
        // Degree 3
        (3, -3) => 0.59004358 * y * (3.0*x*x - y*y),
        (3, -2) => 2.89061144 * x * y * z,
        (3, -1) => 0.45704579 * y * (5.0*z*z - 1.0),
        (3,  0) => 0.37317633 * z * (5.0*z*z - 3.0),
        (3,  1) => 0.45704579 * x * (5.0*z*z - 1.0),
        (3,  2) => 1.44530572 * z * (x*x - y*y),
        (3,  3) => 0.59004358 * x * (x*x - 3.0*y*y),
        _ => 0.0,
    }
}

/// Project radiance samples onto SH basis
pub fn fit_sh(
    samples: &[(Vec3, Vec3)],  // (direction, radiance)
    max_degree: u32,
) -> ShCoeffs {
    let n_coeffs = ((max_degree + 1) * (max_degree + 1)) as usize;
    let mut coeffs = vec![[0.0f32; 3]; n_coeffs];
    
    // Monte Carlo integration: c_lm = (4*pi/N) * sum(L(w) * Y_lm(w))
    let weight = 4.0 * std::f32::consts::PI / samples.len() as f32;
    
    for (dir, radiance) in samples {
        let mut idx = 0;
        for l in 0..=max_degree {
            for m in -(l as i32)..=(l as i32) {
                let basis = sh_basis(l, m, *dir);
                coeffs[idx][0] += radiance.x * basis * weight;
                coeffs[idx][1] += radiance.y * basis * weight;
                coeffs[idx][2] += radiance.z * basis * weight;
                idx += 1;
            }
        }
    }
    
    ShCoeffs { coeffs, degree: max_degree }
}
```

### 3. `src/splat/sampler.rs` - Geometry Sampling

**Ключевой момент:** сэмплируем объекты СО ВСЕХ СТОРОН, не только с камеры.

```rust
/// Sample all surfaces in the scene
pub fn sample_scene_surfaces(
    scene: &Scene,
    density: f32,  // samples per unit area
    rng: &mut impl Rng,
) -> Vec<SurfaceSample> {
    let mut samples = Vec::new();
    
    // Sample spheres
    for sphere in &scene.spheres {
        let area = 4.0 * PI * sphere.radius * sphere.radius;
        let n_samples = (area * density) as usize;
        
        for _ in 0..n_samples {
            // Uniform point on sphere surface
            let dir = uniform_sphere(rng);
            let pos = sphere.center + dir * sphere.radius;
            let normal = dir;
            
            samples.push(SurfaceSample {
                pos,
                normal,
                material: sphere.material,
            });
        }
    }
    
    // Sample checkerboard plane
    // ... grid sampling ...
    
    samples
}

/// Generate SH samples for a surface point
pub fn sample_hemisphere_sh(
    scene: &Scene,
    point: Vec3,
    normal: Vec3,
    n_samples: usize,
    rng: &mut impl Rng,
) -> Vec<(Vec3, Vec3)> {
    let mut samples = Vec::with_capacity(n_samples);
    
    // Build tangent frame from normal
    let tangent = if normal.y.abs() < 0.9 {
        normal.cross(Vec3::Y).normalize()
    } else {
        normal.cross(Vec3::X).normalize()
    };
    let bitangent = normal.cross(tangent);
    
    for _ in 0..n_samples {
        // Uniform hemisphere sampling (можно stratified/importance)
        let local_dir = uniform_hemisphere(rng);
        
        // Transform to world space
        let world_dir = tangent * local_dir.x 
                      + normal * local_dir.y 
                      + bitangent * local_dir.z;
        
        // VIEW direction = opposite of ray direction
        // В 3DGS view_dir смотрит ОТ камеры К точке
        let view_dir = -world_dir;
        
        // Trace ray FROM the point TOWARD the view direction
        // (мы смотрим на точку с этого направления)
        let radiance = trace_incoming_radiance(scene, point, world_dir, rng);
        
        samples.push((view_dir, radiance));
    }
    
    samples
}
```

### 4. `src/splat/ply.rs` - PLY Writer

Бинарный PLY с полным набором атрибутов:

```rust
pub fn write_ply(path: &Path, gaussians: &[Gaussian]) -> io::Result<()> {
    let mut file = BufWriter::new(File::create(path)?);
    
    // ASCII header
    writeln!(file, "ply")?;
    writeln!(file, "format binary_little_endian 1.0")?;
    writeln!(file, "element vertex {}", gaussians.len())?;
    
    // Position
    writeln!(file, "property float x")?;
    writeln!(file, "property float y")?;
    writeln!(file, "property float z")?;
    
    // Normal
    writeln!(file, "property float nx")?;
    writeln!(file, "property float ny")?;
    writeln!(file, "property float nz")?;
    
    // SH DC (degree 0)
    for i in 0..3 {
        writeln!(file, "property float f_dc_{}", i)?;
    }
    
    // SH rest (degrees 1-3: 45 coefficients)
    for i in 0..45 {
        writeln!(file, "property float f_rest_{}", i)?;
    }
    
    // Opacity
    writeln!(file, "property float opacity")?;
    
    // Scale (log-space)
    for i in 0..3 {
        writeln!(file, "property float scale_{}", i)?;
    }
    
    // Rotation (quaternion)
    for i in 0..4 {
        writeln!(file, "property float rot_{}", i)?;
    }
    
    writeln!(file, "end_header")?;
    
    // Binary data
    for g in gaussians {
        write_f32_le(&mut file, &g.pos)?;
        write_f32_le(&mut file, &g.normal)?;
        write_f32_le(&mut file, &g.sh_dc)?;
        write_f32_le(&mut file, &flatten_sh_rest(g))?;
        write_f32_le(&mut file, &[g.opacity])?;
        write_f32_le(&mut file, &g.scale)?;
        write_f32_le(&mut file, &g.rotation)?;
    }
    
    Ok(())
}
```

## Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     SPLAT GENERATION PIPELINE                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. GEOMETRY SAMPLING                                           │
│     ├─ Spheres: uniform points on surface                       │
│     ├─ Plane: grid sampling with jitter                         │
│     └─ Output: Vec<SurfaceSample> (pos, normal, material)       │
│                                                                 │
│  2. SH FITTING (parallel per sample)                            │
│     ├─ Generate N directions on hemisphere                      │
│     ├─ For each direction:                                      │
│     │   └─ Trace ray, compute incoming radiance                 │
│     ├─ Project radiance onto SH basis                           │
│     └─ Output: ShCoeffs per sample                              │
│                                                                 │
│  3. GAUSSIAN CONSTRUCTION                                       │
│     ├─ pos = sample position                                    │
│     ├─ rotation = quaternion from normal                        │
│     ├─ scale = f(density, curvature) в log-space                │
│     ├─ opacity = 1.0 → logit(0.99) для непрозрачных             │
│     └─ sh_* = fitted coefficients                               │
│                                                                 │
│  4. PLY EXPORT                                                  │
│     └─ Binary little-endian PLY                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Сравнение подходов к сэмплированию

| Метод | Pros | Cons | Когда использовать |
|-------|------|------|--------------------|
| **Uniform random** | Простой, статистически корректный | Может быть clumping | Default choice |
| **Stratified** | Лучшее покрытие | Сложнее для сфер | Когда важна равномерность |
| **Blue noise** | Визуально лучше | Дорого генерировать | High-quality output |
| **Adaptive** | Больше сэмплов в сложных местах | Нужен критерий важности | Сцены с деталями |

## Сравнение SH sampling strategies

| Метод | Samples для degree 3 | Качество | Скорость |
|-------|---------------------|----------|----------|
| **Uniform hemisphere** | 64-128 | Хорошее | Быстро |
| **Cosine-weighted** | 32-64 | Хорошее для diffuse | Быстро |
| **Stratified** | 49 (7x7) | Отличное | Средне |
| **Fibonacci spiral** | 64 | Отличное покрытие | Быстро |

Рекомендация: **Fibonacci spiral** - детерминированное, равномерное покрытие:

```rust
/// Fibonacci spiral on hemisphere (deterministic, uniform)
fn fibonacci_hemisphere(n: usize) -> Vec<Vec3> {
    let golden_ratio = (1.0 + 5.0_f32.sqrt()) / 2.0;
    
    (0..n).map(|i| {
        let theta = 2.0 * PI * i as f32 / golden_ratio;
        let phi = (1.0 - 2.0 * (i as f32 + 0.5) / n as f32).acos();
        
        // Hemisphere: only positive z
        let phi = phi * 0.5;  // [0, pi/2]
        
        Vec3::new(
            phi.sin() * theta.cos(),
            phi.sin() * theta.sin(),
            phi.cos(),
        )
    }).collect()
}
```

## Scale estimation

Scale определяет размер гауссиана. Важно чтобы сплаты перекрывались:

```rust
/// Estimate splat scale from sampling density
fn estimate_scale(density: f32, curvature: Option<f32>) -> [f32; 3] {
    // Базовый размер из density: area_per_sample = 1/density
    // radius ~ sqrt(area_per_sample / pi)
    let base_radius = (1.0 / (density * PI)).sqrt();
    
    // Overlap factor (1.5-2.0 для хорошего покрытия)
    let overlap = 1.7;
    let scale = base_radius * overlap;
    
    // Log-space для PLY формата
    let log_scale = scale.ln();
    
    [log_scale, log_scale, log_scale]  // isotropic
}
```

## Формат вывода

**PLY binary** - стандарт де-факто:
- Читается всеми viewers (SuperSplat, Luma, bevy_gaussian_splatting)
- Полная информация (позиция, SH, scale, rotation)
- Размер: ~62 float32 на сплат = 248 байт

Альтернативы рассмотреть позже:
- `.splat` - компактнее, но теряем SH
- `.spz` - сжатый, но нужен encoder

## Ссылки

- [Original 3DGS Paper (INRIA)](https://github.com/graphdeco-inria/gaussian-splatting)
- [gsplat Python library](https://www.jmlr.org/papers/volume26/24-1476/24-1476.pdf)
- [PlayCanvas PLY Format Docs](https://developer.playcanvas.com/user-manual/gaussian-splatting/formats/ply/)
- [PlayCanvas Compression Blog](https://blog.playcanvas.com/compressing-gaussian-splats/)
- [SPZ Format Announcement](https://scaniverse.com/news/spz-gaussian-splat-open-source-file-format)
- [bevy_gaussian_splatting](https://docs.rs/bevy_gaussian_splatting/latest/bevy_gaussian_splatting)
- [wgpu-3dgs-viewer](https://github.com/LioQing/wgpu-3dgs-viewer)
- [LiteGS Training Framework](https://arxiv.org/html/2503.01199v1)
- [MVSplat Paper](https://arxiv.org/abs/2403.14627)
