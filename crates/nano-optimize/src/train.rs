//! High-level training-loop driver.
//!
//! Phase A1 — **skeleton only**. This function bakes references, seeds
//! splats from the forward-fit, sets up Adam state per-attribute, and runs
//! a stub loop that prints diagnostics. The actual forward / backward /
//! loss / densify-and-prune pieces land in Phase A2+.
//!
//! The skeleton is the integration contract: by exercising the full
//! sequence (with a no-op step inside) we catch wiring problems early —
//! workspace deps, type plumbing between crates, configuration surfaces.

use nano_core::scene::Scene;
use nano_splat::SplatConfigGpu;

use crate::adam::{AdamConfig, AdamState};
use crate::reference::{BakeConfig, bake_references};
use crate::splat_store::SplatBuffer;

/// Top-level training configuration.
pub struct TrainConfig {
    pub iterations: u32,
    /// Hard cap on the splat count after densification. Inria's defaults
    /// for synthetic scenes sit around 3-5M; we cap at this so we don't
    /// blow VRAM on the wgpu rasteriser.
    pub max_splats: usize,
    /// Reference-baking configuration. The training resolution and view
    /// count are baked once at start.
    pub reference: BakeConfig,
    /// Forward-fit configuration used to seed the optimisation. Re-uses the
    /// same machinery as a stand-alone `--splats` run.
    pub seed: SplatConfigGpu,
    /// Adam hyperparameters for position channels.
    pub adam_pos: AdamConfig,
    /// Adam hyperparameters for the rotation / scale / opacity / SH channels.
    pub adam_attr: AdamConfig,
}

/// Run the optimisation loop and return the final splat buffer. Returns
/// the *seed* splats unchanged in Phase A1 because the inner step is a
/// no-op — the function still exercises every plumbing path.
pub fn train(
    scene: &Scene,
    cfg: &TrainConfig,
) -> Result<SplatBuffer, Box<dyn std::error::Error>> {
    eprintln!(
        "[train] baking {} reference views at {}×{}...",
        cfg.reference.views, cfg.reference.width, cfg.reference.height
    );
    let refs = bake_references(scene, &cfg.reference)?;
    eprintln!("[train] {} references ready", refs.len());

    eprintln!("[train] seeding splats via forward fit...");
    let seeds = nano_splat::generate_splats_gpu(scene, &cfg.seed)?;
    // `splats` will be mutated by densify / prune in Phase A5 — kept `mut`
    // even though Phase A1's stub loop does not yet touch it.
    #[allow(unused_mut)]
    let mut splats = SplatBuffer::from_gaussians(&seeds);
    eprintln!("[train] {} seed splats", splats.len());

    // Adam state: one slab per attribute kind. Position is 3 floats × len,
    // rotation 4, scale 3, opacity 1, sh_dc 3, sh_rest 45.
    let mut adam_pos = AdamState::new(cfg.adam_pos, splats.len() * 3);
    let mut adam_rot = AdamState::new(cfg.adam_attr, splats.len() * 4);
    let mut adam_scale = AdamState::new(cfg.adam_attr, splats.len() * 3);
    let mut adam_op = AdamState::new(cfg.adam_attr, splats.len());
    let mut adam_dc = AdamState::new(cfg.adam_attr, splats.len() * 3);
    let mut adam_rest = AdamState::new(cfg.adam_attr, splats.len() * 45);

    for iter in 0..cfg.iterations {
        // Phase A1 stub: forward / backward / loss live in A2-A4. Nothing
        // updates here yet, but advancing the Adam step counter lets us
        // observe correct bias-correction behaviour once gradients flow.
        adam_pos.t += 1;
        adam_rot.t += 1;
        adam_scale.t += 1;
        adam_op.t += 1;
        adam_dc.t += 1;
        adam_rest.t += 1;

        if iter % 500 == 0 {
            eprintln!(
                "[train] iter {}: {} splats (cap {})",
                iter,
                splats.len(),
                cfg.max_splats
            );
        }
    }

    eprintln!(
        "[train] complete. final splat count: {} ({} parameter floats)",
        splats.len(),
        adam_pos.m.len()
            + adam_rot.m.len()
            + adam_scale.m.len()
            + adam_op.m.len()
            + adam_dc.m.len()
            + adam_rest.m.len()
    );
    Ok(splats)
}
