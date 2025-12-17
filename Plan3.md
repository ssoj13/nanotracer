# Plan3.md — Mesh/BVH + Alembic (.abc) import + Splats

## Цели
1) Поддержать геометрию сложнее сфер: меши и процедурные примитивы (кубы/цилиндры/торусы/пирамиды/октаэдры) через меш.
2) Ускорить пересечения лучей по мешам через BVH.
3) Импортировать Alembic (`.abc`) через готовую библиотеку (Assimp) и конвертировать в наши меши.
4) Сохранить совместимость сплат-экспорта (PLY 3DGS) и корректные SH-конвенции.
5) Не деградировать рендер сфер/плоскости и текущий CLI.

## Неголы (сейчас)
- Полная поддержка анимации Alembic (time-sampling, topology varying) — только “статический кадр”, расширение позже.
- Полная PBR/текстуры/материалы из Alembic — только базовые цвета/простая модель (как сейчас).
- GPU path tracing или пересборка рендера под Bevy — не требуется.

---

## Выбор библиотек
### BVH
- **rtbvh** (crate `rtbvh`) — pure Rust, использует `glam`, несколько билд-алгоритмов (SAH/LOC/Spatial), traversal-итераторы, пакетная трассировка.
- Почему не `bvh` (svenstaro): требует `rust-version >= 1.87`, а у нас 1.85.

### Alembic
- **asset-importer** (crate `asset-importer`) — safe API поверх Assimp 6.x, default=prebuilt (без ручной сборки). Поддерживает Alembic через Assimp-импорт.
- Добавлять за feature-флагом (например `io_assimp`) чтобы не тащить нативные зависимости в минимальной сборке, если не нужно.

---

## Архитектурные изменения (high-level)

### 1) Единый объект сцены
Сейчас `Scene` хранит `spheres` и отдельный спец-кейс “checkerboard plane”.
Нужно перейти к:
- `Scene { objects: Vec<Object>, lights, environment }`
- `Object { geometry: Geometry, material: Material, transform?: Mat4 }`
- `Geometry`: `Sphere`, `Plane/Quad`, `Mesh` (и всё остальное как Mesh-generator).

Минимально: transform можно внедрить позже, но для Alembic почти сразу нужен хотя бы `Mat4` на объект.

### 2) Unified hit/shading contract
Ввести единый `Hit`:
- `t`, `point`, `normal`, `material`, `albedo_color` (если нужно отличать “материал” и “покрашенный результат”, например для checker).
Идея: `Scene::intersect()` возвращает `HitOption`, а шейдинг берёт данные только из `Hit`.

### 3) Mesh representation
- `Mesh { vertices: Vec<Vec3>, indices: Vec<[u32;3]>, normals?: Vec<Vec3>, uvs?: Vec<Vec2> }`
- Если normals отсутствуют — генерировать (face normals или smooth).
- Для рендера и сплатов достаточно: vertices + indices + normals.

### 4) BVH integration strategy
Два варианта (выбрать один):
A) BVH строится **на уровне меша**: каждый `Mesh` содержит свой BVH по треугольникам.
B) BVH строится **на уровне сцены**: единый BVH по “примитивам” (сферы+треугольники+…).

Рекомендация: начать с (A) — проще внедрение и меньше рефакторинга `Scene`. Дальше можно добавить “scene BVH” как оптимизацию.

---

## Roadmap / Мильстоуны

### Milestone 0 — Подготовка (0.5–1 день)
- Зафиксировать требования:
  - нужна ли трансформация объектов (scale/rotate/translate) уже сейчас? (для Alembic обычно да)
  - какие атрибуты нужны из Alembic: позиции, индексы, нормали, UV, материалы/цвет, инстансы?
- Выбрать initial scope Alembic: только PolyMesh, игнорировать камеры/лайты.

**Acceptance**
- Документ с “что поддерживаем в .abc v1”.

---

### Milestone 1 — Triangle + Mesh без BVH (1–2 дня)
- Реализовать пересечение `Ray` с треугольником (Möller–Trumbore).
- Добавить `Geometry::Mesh` и перебор треугольников (O(n)).
- Перенести “checkerboard plane” в `Geometry::Quad`/`Plane` как обычный объект.
- Обновить `Scene::intersect()` под список `objects`.

**Acceptance**
- Сцена с одним мешем (например куб из 12 треугольников) корректно рендерится.
- Юнит-тесты:
  - hit/miss на треугольнике
  - нормаль корректной ориентации
  - стабильность при почти параллельных лучах

---

### Milestone 2 — BVH через rtbvh (2–4 дня)
- Выбрать интеграцию (сначала BVH на меш).
- Сконвертировать треугольники меша в примитивы `rtbvh`:
  - AABB на треугольник
  - Ray для traversal
- Построить BVH билдом `construct_binned_sah()` (или другой, но один стабильный).
- В `Mesh::intersect()`:
  - traverse BVH → кандидаты треугольников → точное пересечение → выбрать ближайший `t`.

**Acceptance**
- Рендер меша на 50k+ треугольников работает приемлемо (без “минут на кадр”).
- Тест: BVH traversal выдаёт тот же nearest-hit что и brute-force (на небольшом меше).

**Риски/заметки**
- `rtbvh` использует собственные Ray/AABB типы; нужен слой адаптеров к `glam`.
- Важно: стабильная обработка `t_min=0.001` (self-intersection avoidance) сохраняется.

---

### Milestone 3 — Mesh surface sampling для splats (2–4 дня)
Сейчас `splat/sampler.rs` “знает” только сферы и плоскость и ищет материал эвристикой.
Нужно:
- Ввести trait/интерфейс:
  - `fn surface_area(&self) -> f32`
  - `fn sample_surface(&self, density, rng) -> impl Iterator<Item=SurfaceSample>`
- `SurfaceSample` расширить, чтобы не гадать материал:
  - `pos`, `normal`, `material_color`, `material_params` (albedo/specular/refract idx и т.п.) или прям `Material`.
- Для mesh:
  - предрасчёт площади треугольников
  - выбор треугольника по CDF (alias-table опционально)
  - равномерный барицентрический сэмплинг точки на треугольнике
  - нормаль: либо интерполяция вершинных нормалей, либо face normal

**Acceptance**
- Экспорт сплатов на кубе/торусе выглядит равномерно, без дыр при адекватных `--splat-density`.
- Тесты:
  - `sample_surface()` не возвращает NaN
  - количество сэмплов ~ `area*density` (погрешность в пределах)

---

### Milestone 4 — Procedural primitives via mesh (1–3 дня)
Добавить генераторы мешей:
- `cube(size)`
- `octahedron(size)`
- `pyramid(base, height)`
- `cylinder(radius, height, segments, caps)`
- `torus(R, r, seg_major, seg_minor)`

Рекомендация: все примитивы, кроме сферы, делать как меш (тор — особенно).
Сфера может остаться аналитической для скорости и качества.

**Acceptance**
- Для каждого примитива: рендер + сплат экспорт работают.
- Минимум один snapshot/manual check.

---

### Milestone 5 — Alembic import via asset-importer (Assimp) (2–6 дней)
- Добавить feature `io_assimp`.
- CLI:
  - `--import path.abc` (и/или `--import path.any`)
  - опционально `--import-scale`, `--import-center`, `--import-up-axis`
- Реализовать импорт:
  - прочитать сцену
  - собрать меши (vertices+indices)
  - применить node transforms (Mat4) к вершинам или хранить transform в объекте
  - normals: использовать если есть, иначе генерировать
  - материалы: минимум `diffuse_color`, опционально roughness/gloss игнорировать

**Acceptance**
- `.abc` из твоего пайплайна успешно импортится и рендерится.
- Экспорт сплатов из импортированного `.abc` открывается в viewer и выглядит ожидаемо.
- Тест: импорт маленького ассета (в репо не коммитить большие; можно мини-меш в `tests/data`).

**Риски/заметки**
- Assimp Alembic поддержка может быть неполной для некоторых файлов (особенно анимация/инстансы/UV sets).
- Prebuilt Assimp на Windows: важно совпадение CRT; `asset-importer` это учитывает, но надо проверить на твоей среде.

---

## Командный интерфейс (предложение)
- `--import <path>`: импорт мешей в сцену (abc/любое поддерживаемое Assimp).
- `--import-format-hint` (опционально, если Assimp путается).
- `--time <f32>` или `--frame <u32>` (заглушка до анимации).
- `--primitive <kind>` + параметры (для быстрых тестов без файлов).

---

## Вопросы (нужны ответы до реализации)
1) Alembic: нужен ли **временной** слайс сразу (frame/time), или достаточно “первого”?
2) Ожидаются ли инстансы (один меш много раз), или можно пока расплющить в уникальные меши?
3) Какие оси/up-axis в твоих `.abc`? (Y-up/Z-up), нужен ли конвертер?
4) Материалы: достаточно ли `diffuse_color`, или хочешь хотя бы roughness/metallic?
5) Ограничение по зависимостям: ок ли тащить `asset-importer` как optional feature?

---

## Проверка/качественные критерии
- `cargo test` проходит.
- `cargo fmt --all -- --check` проходит.
- Экспорт PLY остаётся совместимым (инварианты из AGENTS/done.md).
- На тестовых сценах:
  - без дыр при адекватных density/scale
  - без “психоделики” при `--sh-degree 0` (baseline)
  - приемлемая скорость на мешах (BVH работает)

---

## Этап 2 (после базовой версии)
- Scene-level BVH (общий по объектам/примитивам).
- Alembic time sampling + motion blur (если нужно).
- Текстуры/UV → выбор `material_color` из текстуры.
- Importance sampling для SH-лучей (снижение шума/лучше цвета).
