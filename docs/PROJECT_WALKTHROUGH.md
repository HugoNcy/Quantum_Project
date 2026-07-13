# Project walkthrough: every phase, every file, every result

This document is the onboarding path for the repo. It explains what the project is trying to do, what was actually done step by step, why each Python file exists, what came out of each phase, and how to reproduce every number. The other documents in `docs/` each cover one slice in depth; this one ties them together. If you read only one document before touching the code, read this one.

Written July 2026, when phases 1, 1bis, 2 and 3 were done in simulation and phase 4 (hardware) was pending cloud access.

## 1. What this project is

We build a hybrid classical-quantum generative model: a small PyTorch network feeds a differentiable photonic circuit simulated through MerLin (Quandela's QML layer on top of Perceval), and a second small network maps the circuit's output distribution to samples. The whole thing is trained end to end with an MMD loss against a target distribution.

The research question is a comparison, not a speed claim: at a comparable trainable-parameter budget, do an MZI mesh (Clements decomposition) and a tritter mesh (3x3 DFT mixers) have the same generative expressivity, and does the ranking between the two survive realistic photonic noise (partially distinguishable photons, losses) and noise mismatch (training noise different from deployment noise)? This is explicitly not a quantum advantage claim. The project measures where things break under realistic noise and reports honestly, including flat or negative results.

Team of three. Hugo owns the generative model, the training loop, the MMD loss, the evaluation and the noise-robustness characterization. Niels and Tony own the photonic backend: Perceval circuits, the hand-built tritter mesh, and the planned Quandela Cloud execution.

The headline results so far, in one paragraph: the pipeline learns simple 2D targets down to the statistical floor of the MMD estimator. Partial distinguishability (down to 0.85) barely hurts. Photon loss is the mismatch axis that matters: a model trained without losses collapses when evaluated under losses, and training under the deployment noise repairs it. The tritter mesh beats the MZI mesh on every no-loss profile and the ranking inverts under losses. On a heavy-tailed 1D financial target the generator captures the bulk and truncates the tails, which the plan predicted. All of this is single-seed; seed replication is the declared next step.

## 2. How to read this repo

### The documents

| document | language | what it is |
|---|---|---|
| `README.md` | English | Overview, environment setup (Podman/Docker), conventions, roadmap, headline results. |
| `docs/PLAN.md` | French | The implementation plan written before the code: validated environment, the four known API traps, target architecture, phases 1 to 5 with acceptance criteria. The phase structure of everything below comes from here. |
| `docs/OFFLINE_KIT.md` | French | The API survival guide. Every snippet was executed on the pinned stack. Contains the output-dimension tables, the traps in detail, six crash-test exercises with solutions, and canonical reference implementations (tritter recipe, `copy_circuit_params`, MMD). |
| `docs/REVIEW.md` | French | Adversarial review of the phase 1 code and circuits (2026-07-05), ten findings F1 to F10. Several findings changed the code and the methodology; they are referenced as F1..F10 everywhere. |
| `docs/EXECUTION_LOG_PHASES_2_3_1BIS.md` | English | Diary of the 2026-07-07 session that executed phases 2, 3 and 1bis: gates, commands, numbers, decisions, commits. The most detailed record of how the results were produced. |
| `notebooks/results_noise_and_meshes.ipynb` | English | The interpretation write-up: 13 cells, analysis in Markdown, four code cells that read from `runs/` and `figures/`. Ships unexecuted; renders on GitHub through the committed figures. |
| `CLAUDE.md` | English | Binding conventions for humans and AI assistants working on the repo. |

### The file tree, and one trap

```
src/        the 11 Python modules (phases 1 to 3, detailed below)
tests/      3 gate suites, run as scripts, each writes tests/<name>_results.json
docs/       plan, offline kit, review, execution log, this file, exercises/
notebooks/  results_noise_and_meshes.ipynb
figures/    versioned result figures and compact CSV summaries
runs/       training outputs: model.pt, log.json, history.csv per run (GITIGNORED)
data/       local datasets (gitignored, currently unused: all targets are synthetic)
```

The trap: `runs/` is gitignored. A fresh clone has the figures and the CSV summaries (that is what `figures/` is for), but not the trained weights or run logs. The notebook's code cells read from `runs/`, so on a fresh clone they will not execute until you re-run the training scripts (section 10). The Markdown cells and the committed figures carry the full interpretation regardless.

## 3. Phase 0: environment validation and API traps

Before any model code, the stack itself was validated, because MerLin/Perceval have failure modes that are silent and would have poisoned every comparison later. This produced the initial scaffold (Dockerfile, `compose.yaml`, `requirements.txt`, README), `docs/OFFLINE_KIT.md`, and the six exercises in `docs/exercises/`. The versions were pinned after validation: `perceval-quandela==1.2.4`, `merlinquantum==0.4.0`, torch 2.5.1 in the container. (Historical note: the offline kit and the execution log were validated on a local venv with Perceval 1.2.3; the pin discrepancy is flagged in the execution log as a team decision. No API difference between the two was ever observed.)

The four traps, discovered by test and documented in `docs/PLAN.md` section 1.2, drive the whole design:

1. **`builder=` vs `circuit=`.** `CircuitBuilder.build()` returns an internal MerLin type that `QuantumLayer(circuit=...)` rejects. Pass the builder itself (route 1), or a raw `pcvl.Circuit` with manually declared parameter prefixes (route 2).
2. **Noise silently changes the output dimension.** With perfect photons MerLin restricts computation to the no-bunching subspace; any `indistinguishability < 1` switches it to the full Fock space, so clean and noisy layers return vectors of different sizes with no error. The non-negotiable project convention follows: every `QuantumLayer` forces `MeasurementStrategy.probs(ComputationSpace.FOCK)`, noiseless included.
3. **`transmittance < 1` changes the layer's internal structure.** Lossy noise models add internal keys, so a clean `state_dict` does not load into a lossy layer. Weight transfer between noise profiles must copy trainable phases by name (`copy_circuit_params`), never the full `state_dict`.
4. **No native tritter.** `add_entangling_layer` only knows `mzi` and `bell`; the tritter mesh has to be built by hand in pure Perceval.

The six exercises are runnable crash tests of exactly these points (solutions and outputs in `docs/OFFLINE_KIT.md`):

| exercise | what it proves |
|---|---|
| `ex1_perceval_circuit.py` | Pure Perceval basics: build a 3-mode circuit, name parameters, assign values, compute and check the unitary. |
| `ex2_circuit_builder.py` | Route 1: `CircuitBuilder` with angle encoding plus MZI mesh, and how input parameter prefixes are exposed. |
| `ex3_quantum_layer_fock.py` | Trap 2 live: the same circuit returns dimension C(m,n) by default and C(n+m-1,n) with FOCK forced, and noise flips the default silently. |
| `ex4_backward.py` | Gradients flow through the quantum layer, and a degenerate loss (mean of a probability row, which is constant 1) gives zero gradients. |
| `ex5_noise_state_dict.py` | Trap 3 live: the exact `state_dict` failure across noise profiles, and the `copy_circuit_params` fix. |
| `ex6_tritter_manual.py` | Trap 4: a hand-decomposed DFT3 tritter wrapped in a route 2 `QuantumLayer` with manual `input_parameters` / `trainable_parameters`. |

Two conventions born here matter everywhere below. Circuit structure is always a sandwich (entangling mesh, then angle encoding, then entangling mesh), because encoding phases applied directly to the input Fock state are a global phase and produce input-independent outputs. And parameter naming is the contract between backend and training loop: input encoding phases are `x0..x{n-1}`, trainable phases are `theta*`.

## 4. Phase 1: a minimal generator that learns

Goal (plan): prove the full pipeline learns a simple 2D target. Written and committed 2026-07-05, merged 2026-07-06 (PR #2). Five files:

- **`src/data.py`**: synthetic targets, each drawing from its own seeded `torch.Generator` so datasets are reproducible independently of global RNG state. `gaussian` (center (1,-1), std 0.5), `two_gaussians` (balanced mixture at (-1,0) and (1,0), std 0.3, the main target of the whole project), `ring` (unit circle, radial std 0.1), all shape (n, 2), registered in the `DATASETS` dict.
- **`src/losses.py`**: `mmd_loss(x, y)`, squared MMD with a mixture of Gaussian kernels, bandwidths (0.1, 0.5, 1.0, 2.0, 5.0) chosen to bracket the inter-point distances of the 2D targets. It is the biased V-statistic (diagonal kernel terms kept), so values are only comparable at a fixed batch size; every comparison in the project uses batch 256.
- **`src/model.py`**: `build_quantum_layer` (route 1, `CircuitBuilder`, sandwich mesh/encoding/mesh, FOCK forced) and `PhotonicGenerator`: z ~ N(0, I) with latent dim 6, a `Linear(6, 6)` encoder, `tanh` scaled to angles in (-pi, pi), the quantum layer (6 modes, 3 photons, output dim 56 = C(8,3)), then an adapter `Linear(56, 32) -> ReLU -> Linear(32, 2)`. The Fock dimension is discovered by a dummy forward because noise can change it. Also `copy_circuit_params`, the trap 3 utility, written in phase 1 as the plan required.
- **`src/train.py`**: the training loop. Adam on everything (encoder, circuit phases, adapter), defaults 800 steps, batch 256, lr 5e-3, seed 0, a pre-sampled pool of 20000 target points, real-vs-real MMD floor computed on one batch pair as reference. Writes `runs/baseline/model.pt` and `log.json` (full config plus per-step loss history). CLI: `--dataset --steps --seed --out-dir`.
- **`src/eval.py`**: rebuilds the generator from `log.json`, generates 2000 fresh samples with eval seed = train seed + 1, writes `figures/mmd_curve_<dataset>.png` (log scale, floor line) and `figures/scatter_<dataset>.png`. Known limitation, kept deliberately (review F3): it rebuilds without a noise argument, so it is phase 1 only; noisy runs are rebuilt through `mismatch_matrix.build_generator_from_log` instead.

Acceptance criteria from the plan, both met: MMD^2 falls from about 0.5 to the real-vs-real floor (near 4e-3) in about 150 of the 800 steps on `two_gaussians`, and the generated scatter covers both mixture modes. The committed evidence is `figures/mmd_curve_two_gaussians.png` and `figures/scatter_two_gaussians.png`.

Reproduce: `python src/train.py` then `python src/eval.py` (from the repo root).

## 5. The adversarial review: ten findings that shaped everything after

Same day as phase 1 (2026-07-05), the code was put through an adversarial review targeting silent physical and methodological bugs. `docs/REVIEW.md` records ten findings, each backed by a throwaway measurement script. This step is worth understanding because half of the phase 2/3 machinery exists as a response to it.

| finding | severity | what it says | what changed because of it |
|---|---|---|---|
| F1 | high | The "equal budget" between meshes was nominal only: measured Jacobian rank of output probabilities w.r.t. phases was 50/60 for MZI but 32/60 for the original tritter mesh (phases on unmixed modes collapse or end as unobservable output phases). | Tritter phase placement rewritten: phases only on modes the next layer mixes, minus one gauge reference per triplet. Post-fix: 56 nominal phases, rank 49 vs MZI 50. A rank-parity test (gap <= 10%) now enforces this. |
| F2 | medium | At 6 modes the tritter layer offsets drop a triplet: modes 0 and 5 are mixed in 1 layer out of 3, modes 2 and 3 in every layer. | Kept as a deliberate property of a planar (no mode crossing) tritter mesh, documented in the `circuits.py` docstrings and reported with the results. |
| F3 | medium | Forcing FOCK everywhere removes the dimension mismatch that would otherwise crash when evaluating a noisy run as clean; `eval.py` would do this silently. | Run logs record the profile name and params; `build_generator_from_log` rebuilds the noise model; grid scripts assert the output dimension against `expected_dim` before training. `eval.py` itself left as is, documented as phase 1 only. |
| F4 | medium | The MMD floor from a single batch pair varies by a factor 2.6 between draws. | Phase 2+ scripts average the floor over 16 disjoint pairs and report mean and std (`mmd_floor_stats`). |
| F5 | medium | The default bandwidths (0.1..5) are myopic on raw log-return scales of order 1e-2, attenuating the signal about 1000x. | The phase 1bis target is standardized to unit variance (affine, so the tail shape is unchanged), instead of touching the loss. |
| F6 | low | The V-statistic MMD is biased and not comparable across batch sizes. | Batch size fixed at 256 for every training and every evaluation. |
| F7 | low | The tritter decomposition is deterministic only empirically (about 1e-8 across processes), not by API contract. | A determinism test pins it (two `tritter()` calls must agree to 1e-8). |
| F8 | low | The input sensitivity test threshold (1e-3) was laxer than the CLAUDE.md contract (order 1e-2). | Strict test added: max per-dim std > 1e-2 and mean > 1e-3. |
| F9 | low | MerLin maps inputs to `x` parameters by insertion order, not by name. | A test locks the contract: the `x` parameters must appear in exact mode order `x0..x5`. |
| F10 | info | The default input state `[1,0,1,0,1,0]` was implicit and asymmetric w.r.t. the tritter triplets. | Phase 3 passes `input_state` explicitly and records it in every run log. |

`tests/test_review_findings.py` is the regression suite encoding F1, F2, F7, F8 and F9: Jacobian rank parity via SVD, strict sensitivity thresholds, tritter determinism, x-parameter ordering. Like all test files in this repo it runs as a plain script (`python tests/test_review_findings.py`) and regenerates `tests/test_review_findings_results.json`.

Frank remark: F1 is the finding that saved the project's central comparison. Without it, phase 3 would have compared a 50-direction mesh against a 32-direction mesh and called it fair.

## 6. Phase 2: noise grid and the mismatch matrix

Goal (plan): train the generator under a grid of noise profiles and measure what happens when training noise and deployment noise differ. The modules were written after the review; execution happened in the 2026-07-07 session recorded step by step in `docs/EXECUTION_LOG_PHASES_2_3_1BIS.md`.

### The new files

- **`src/noise.py`**: the profile grid as plain kwargs dicts (they serialize into run logs):

  | profile | indistinguishability | transmittance | FOCK output dim |
  |---|---|---|---|
  | P0 | 1.00 | 1.0 | 56 |
  | P1 | 0.95 | 1.0 | 56 |
  | P2 | 0.90 | 1.0 | 56 |
  | P3 | 0.85 | 1.0 | 56 |
  | P4 | 0.95 | 0.9 | 84 |

  Plus `make_noise` (builds the `pcvl.NoiseModel`), `is_lossy`, `expected_dim` (56 = C(8,3) clean; 84 = sum over k <= 3 of C(k+5,k) when losses allow photon counts below 3), and `transfer_generator`. That last function is the heart of the mismatch experiment: it copies every same-named same-shaped parameter, then realigns the first adapter Linear column by column through `QuantumLayer.output_keys`, matching Fock occupation tuples. States the source never saw (the 28 loss-only states of P4) get zero columns, which is the honest deployment semantics: the transferred model is blind to loss events.
- **`src/train_noise_grid.py`**: trains one generator per profile. Contains `mmd_floor_stats` (floor over 16 disjoint pairs, review F4) and `train_generator`, the generic loop that phases 2, 3 and 1bis all share: Adam, incremental `history.csv` flushed every 25 steps so long runs can be monitored from files, `model.pt` and a `log.json` carrying everything needed to rebuild the run (profile included, review F3). Each run reseeds torch (seed 0) so it is reproducible in isolation, and asserts its output dimension against `expected_dim` before training. CLI: `--profiles --dataset --steps --seed --out-root`.
- **`src/mismatch_matrix.py`**: the K x K experiment. For each pair (train profile A, eval profile B): rebuild the A-model from its log and load its weights, build a B-configured twin, move the weights with `transfer_generator`, and evaluate MMD against the target averaged over 8 fixed batches of 256 with deterministic latents (eval pool seed = train seed + 1, eval latent seeds = train seed + 100 + repeat). Writes `mismatch_<label>.json` and `.csv` next to the runs, plus `figures/mismatch_heatmap_<label>.png` and `figures/mmd_final_vs_indistinguishability_<label>.png`. It computes the plan's acceptance criterion itself: mean(diagonal) must not exceed mean(off-diagonal), else the JSON carries `suspect_bug: true`. Also home of `build_generator_from_log`, which phase 3 and the notebook reuse.
- **`tests/test_transfer.py`**: the gate suite written before any grid training, because the riskiest assumption in the design was that `output_keys[j]` really describes output column j. Five checks: the 56/84 key contract with the clean set strictly inside the lossy set; mechanical column alignment after a P0 to P4 transfer (exactly 28 zero columns); the decisive semantic check, transfer into an almost-clean lossy layer (transmittance 0.9999) changes outputs by at most 6.2e-6 on identical latents; a `MeshGenerator` smoke test; and a 2-step train / rebuild-from-log / evaluate round trip (this is what `runs/smoke/` is).

### What ran, and the gates before it

The session ran verification first: both existing test suites re-run (tritter vs DFT3 error 7.09e-9; sensitivity max std 0.0525 MZI / 0.0312 tritter; trainable counts 60/56; Jacobian ranks 50/49; determinism 4.19e-9), the new transfer gates, then 5-step timing pilots of both pipelines (19.5 s for the grid pilot, 89.2 s for the mesh pilot, about 0.08 s/step clean and 0.15 s/step lossy) to confirm the full runs fit in the time budget. Only then:

```bash
python src/train_noise_grid.py
python src/mismatch_matrix.py
```

Defaults throughout: `two_gaussians`, 800 steps, batch 256, lr 5e-3, latent 6, 6 modes, 3 photons, seed 0.

### The numbers and what they mean

MMD^2 means, rows = training profile, columns = evaluation profile (`figures/mismatch_mzi.csv`; full precision in `runs/noise_grid/mismatch_mzi.json` after a re-run):

| train \ eval | P0 | P1 | P2 | P3 | P4 |
|---|---|---|---|---|---|
| P0 | 0.0052 | 0.0053 | 0.0054 | 0.0056 | 0.0310 |
| P1 | 0.0052 | 0.0052 | 0.0053 | 0.0054 | 0.0308 |
| P2 | 0.0061 | 0.0059 | 0.0058 | 0.0058 | 0.0324 |
| P3 | 0.0059 | 0.0059 | 0.0059 | 0.0060 | 0.0372 |
| P4 | 0.0062 | 0.0062 | 0.0063 | 0.0064 | 0.0039 |

Diagonal mean 0.0052, off-diagonal mean 0.0112, real-vs-real floor 0.0033 +- 0.0014, `suspect_bug: false`. Three readings:

1. Partial distinguishability barely hurts on this target: the matched MMD moves from 0.0052 (P0) to just 0.0060 (P3). Frank remark from the write-up: `two_gaussians` is probably too easy to stress the interference resource, since the classical adapter can compensate a mildly blurred Fock distribution.
2. Losses are the real mismatch axis. Every loss-free-trained model collapses under P4 evaluation (0.031 to 0.037, six to eight times its own diagonal). The mechanism is structural: with transmittance 0.9 and 3 photons, 1 - 0.9^3 = 27.1% of events lose at least one photon, and the transferred model has zero adapter weight on all 28 low-photon states.
3. Training under the deployment noise repairs it: P4/P4 (0.0039) is the best cell of the whole matrix, at the floor. The reverse transfer (P4-trained, evaluated clean) degrades only mildly because the 56 three-photon columns survive the move intact.

## 7. Phase 3: MZI vs tritter at a fair budget

Goal (plan): the central comparison. The backend delivers `build_circuit(mesh_type, n_modes)` honoring the naming contract; the training side replays the phase 2 protocol for both meshes. Executed in the same 2026-07-07 session.

### The new files

- **`src/circuits.py`**: pure Perceval, both meshes as sandwich circuits.
  - `dft3_matrix` and `tritter()`: the ideal 3x3 DFT (cube roots of unity over sqrt(3)), decomposed by Reck into a fixed numeric-phase circuit; the decomposition template must be a full MZI cell (a bare BS template makes `Circuit.decomposition` return None silently, one of the offline kit's validated facts).
  - `_mzi_mesh`: Clements rectangle via `GenericInterferometer`, 15 cells of 2 trainable phases at 6 modes, so 30 thetas per mesh and 60 in the sandwich.
  - `_tritter_mesh` and helpers: layers of tritters on offset-cycled triplets (offsets 0, 1, 2), trainable phases before each layer except the first, only on modes that layer mixes, minus one gauge reference per triplet (the F1 fix). At 6 modes, 12 layers give 56 thetas in the sandwich and measured Jacobian rank 49 to 50 against the MZI's 50. The mode-coverage center bias (F2) is documented in the docstring.
  - `build_circuit`: mesh, then `x0..x5` encoding phases in mode order, then mesh.
- **`src/model.py` additions** (pure additions, per the project rule that existing code is completed, never modified): `build_mesh_layer` (route 2: raw circuit, explicit `input_parameters=["x"]` / `trainable_parameters=["theta"]`, explicit `input_state=[1,0,1,0,1,0]` per F10, FOCK forced) and `MeshGenerator`, the same pipeline as `PhotonicGenerator` over the explicit circuit.
- **`src/compare_meshes.py`**: trains every (mesh, profile) pair with the shared `train_generator` loop (10 runs), logs `n_thetas` per run so the fair-budget claim is auditable from the logs, then calls `mismatch_matrix` per mesh and writes `figures/final_table.csv` plus `figures/ranking_vs_noise.png` (matched-profile MMD per profile, one line per mesh, error bars, log scale). `--skip-train` rebuilds the table and figures from existing runs.
- **`tests/test_circuits.py`**: written alongside the circuits: tritter equals DFT3 up to global phase (< 1e-6), both meshes pass the input sensitivity test, trainable counts within ratio [0.8, 1.25].

### A consistency check that fell out for free

The route 2 MZI produced a mismatch matrix bit-for-bit identical (max cell difference 0.0 over all 25 cells) to the phase 2 route 1 matrix. Same Clements topology, same parameter count and order, same seed, hence the same initialization and trajectory. Two independent construction paths agreeing exactly is strong cross-validation of the raw-circuit route, and it means the MZI rows of phase 3 literally are the phase 2 model.

### The numbers

```bash
python src/compare_meshes.py
```

Matched-profile (train = eval) MMD^2, from `figures/final_table.csv` (floor 0.0033, per-cell eval stds 0.0007 to 0.0032 in the CSV):

| profile | mzi | tritter | better |
|---|---|---|---|
| P0 | 0.005160 | 0.004022 | tritter |
| P1 | 0.005202 | 0.004374 | tritter |
| P2 | 0.005815 | 0.004030 | tritter |
| P3 | 0.005981 | 0.004337 | tritter |
| P4 | 0.003889 | 0.005843 | mzi |

The answer to the research question: **the ranking is stable under partial distinguishability and inverts under losses.** On P0 to P3 the tritter is consistently better (0.0040-0.0044 vs 0.0052-0.0060) with four fewer trainable phases. Under P4 the MZI lands at the floor (0.0039) while the tritter degrades to 0.0058, and the tritter also transfers worse out of P4 (its P4-trained models evaluated clean: 0.0070-0.0074, vs 0.0062-0.0064 for the MZI; full matrices in `figures/mismatch_mesh_mzi.csv` and `figures/mismatch_mesh_tritter.csv`).

The honest caveat, attached wherever this result appears: one training seed, and the per-cell evaluation std is of the same order as the gaps, so each individual cell is a one-sigma statement. The weight comes from the direction being consistent across four profiles, not from any single cell. Seed replication (3 to 5 seeds) is the declared next step before claiming more.

## 8. Phase 1bis: heavy-tailed financial target

Goal (plan): swap the synthetic 2D target for a financial-style distribution and look at the tails, expecting honestly documented failure there. Numbered 1bis but executed last, in the same 2026-07-07 session.

Design decisions: no market data is reachable offline, so the target is synthetic but principled, a standardized Student-t with 4 degrees of freedom (leptokurtic like daily returns, infinite kurtosis). Two details worth knowing: `torch.distributions` samplers do not accept a `torch.Generator`, so the t variable is built from its definition t = z / sqrt(chi2/df) with the chi2(4) as a sum of 4 squared seeded normals; and standardizing to unit variance is what resolves review F5 (the default MMD bandwidths would be blind at raw return scale), without touching the loss, since an affine rescale leaves the tail shape unchanged.

New code, both pure additions: `log_returns(n, seed)` appended to `src/data.py` (shape (n, 1)), and **`src/train_financial.py`**: a `PhotonicGenerator` with `out_dim=1` on the clean profile, trained with the shared `train_generator` loop, then evaluated on 20000 fresh samples (seed 1). It writes `runs/financial/` plus three artifacts: `figures/hist_log_returns.png` (log density scale, otherwise the tails are invisible), `figures/qq_log_returns.png`, and `figures/tails_log_returns.csv`. The log records `out_dim: 1` because `build_generator_from_log` assumes 2D; a financial run is rebuilt manually.

```bash
python src/train_financial.py
```

The quantile report (`figures/tails_log_returns.csv`):

| quantile | target | generated |
|---|---|---|
| 0.001 | -4.846 | -2.448 |
| 0.01 | -2.649 | -2.143 |
| 0.05 | -1.506 | -1.479 |
| 0.5 | -0.002 | -0.017 |
| 0.95 | 1.503 | 1.466 |
| 0.99 | 2.627 | 2.098 |
| 0.999 | 4.859 | 2.325 |

The bulk is captured well (5% / 50% / 95% within a few percent); the extreme tails are not (target reaches +-4.85, the generator stops near 2.4, and the QQ plot flattens symmetrically at both ends). Why: the sample is an affine map of a 56-bin probability vector driven by tanh-bounded angles, so the output range is effectively bounded, and bandwidths of 0.1 and larger assign almost no loss to missing 0.1% tail mass. This is the outcome the plan predicted and asked to be documented as a result, not a failure. Plausible follow-ups are noted in the log: sub-0.1 bandwidths, a heavy-tailed latent, or an explicit quantile penalty.

## 9. Where the results live

`runs/` (gitignored, reproduced by the scripts):

| directory | what it is |
|---|---|
| `runs/smoke/P0` | 2-step pipeline round-trip from `tests/test_transfer.py`. |
| `runs/pilot_grid/`, `runs/pilot_mesh/` | 5-step timing pilots of the two big scripts (gate 1.3 of the execution log). |
| `runs/noise_grid/P0..P4` | Phase 2: 800-step runs plus `mismatch_mzi.json/.csv`. |
| `runs/mesh_compare/{mzi,tritter}/P0..P4` | Phase 3: 10 runs plus per-mesh mismatch summaries. |
| `runs/financial/` | Phase 1bis run. |

Every run directory contains `model.pt`, `log.json` (full config, profile params, Fock dim, input state, floor stats, loss history; phase 3 logs also carry `n_thetas`) and `history.csv` (per-step MMD, flushed during training).

`figures/` (versioned): the phase 1 curve and scatter, the three mismatch heatmaps and their `mmd_final_vs_indistinguishability_*` curves, `ranking_vs_noise.png`, the financial histogram and QQ plot, and the compact CSVs (`mismatch_mzi.csv`, `mismatch_mesh_mzi.csv`, `mismatch_mesh_tritter.csv`, `final_table.csv`, `tails_log_returns.csv`) that the tables above quote.

## 10. How to reproduce everything

Environment first (details in the README): build the container (`podman build -t photonic-gen .`, then `podman compose run --rm lab bash`), or the venv fallback (`python3.12 -m venv .venv`, install torch 2.5.1, then `pip install -r requirements.txt`). Everything below runs from the repository root and is CPU-only.

Gates, in the repo's script-style (no pytest needed; each regenerates its `tests/<name>_results.json`):

```bash
python tests/test_circuits.py
python tests/test_review_findings.py
python tests/test_transfer.py
```

Optional 5-step pilots to sanity-check the pipelines end to end in under two minutes:

```bash
python src/train_noise_grid.py --steps 5 --out-root runs/pilot_grid
python src/compare_meshes.py --steps 5 --out-root runs/pilot_mesh
```

The four production runs (about one hour total on a laptop CPU; all seeds fixed at 0, so the numbers reproduce exactly):

```bash
python src/train_noise_grid.py      # phase 2 training: runs/noise_grid/P0..P4
python src/mismatch_matrix.py       # phase 2 matrix: mismatch_mzi.*, 2 figures
python src/compare_meshes.py        # phase 3: 10 runs, both matrices, final_table.csv, ranking figure
python src/train_financial.py       # phase 1bis: runs/financial, 3 tail artifacts
```

Optionally the phase 1 baseline (`python src/train.py` then `python src/eval.py`), which phase 2 supersedes (its P0 run is the same model under the grid protocol).

What to expect if nothing broke: the mismatch diagonal mean near 0.0052 against an off-diagonal mean near 0.0112 with `suspect_bug: false`; a final table matching section 7 (tritter ahead on P0-P3, MZI ahead on P4); tail quantiles matching section 8. Seeds are deterministic on CPU with the pinned stack; small platform differences show up in the last digits, not in any conclusion.

## 11. Current state and what is left

Done in simulation: phases 1, 1bis, 2, 3, plus the write-up (notebook, README results section, this walkthrough). Remaining:

- **Phase 4**: inference of the best model on Quandela hardware (Belenos or Lucy), treating the QPU as an unknown noise profile inside exactly this mismatch framework. Pending cloud access; the `.env` `QUANDELA_TOKEN` mechanism is already documented in the README.
- **Seed replication** (3 to 5 seeds) to firm up or kill the mesh-ranking result, and a harder target (`ring` is already in `data.py`) to separate the architectures away from the floor.
- The Perceval pin question (validated on 1.2.3 locally, pinned 1.2.4) needs a team decision.
- **Phase 5** (this repo as a reproducible artifact) is effectively in progress; this document is part of it.

## 12. File-by-file inventory

| file | phase | purpose | writes |
|---|---|---|---|
| `src/data.py` | 1, 1bis | Synthetic targets: gaussian, two_gaussians, ring, log_returns; `DATASETS` registry | nothing |
| `src/losses.py` | 1 | Multi-bandwidth MMD (V-statistic, bandwidths 0.1..5) | nothing |
| `src/model.py` | 1, 3 | Route 1 layer + `PhotonicGenerator`; route 2 `build_mesh_layer` + `MeshGenerator`; `copy_circuit_params` | nothing |
| `src/train.py` | 1 | Baseline training CLI | `runs/baseline/` |
| `src/eval.py` | 1 | Baseline figures (clean runs only, F3 caveat) | `figures/mmd_curve_*.png`, `figures/scatter_*.png` |
| `src/noise.py` | 2 | Profile grid P0-P4, `expected_dim`, cross-profile `transfer_generator` | nothing |
| `src/train_noise_grid.py` | 2 | Grid training; shared `train_generator` loop and `mmd_floor_stats` | `runs/noise_grid/<P>/` |
| `src/mismatch_matrix.py` | 2, 3 | K x K train/eval matrix, acceptance gate, `build_generator_from_log` | `runs/<root>/mismatch_*.{json,csv}`, 2 figures per label |
| `src/compare_meshes.py` | 3 | Both meshes x all profiles, fair-budget audit, final table and ranking figure | `runs/mesh_compare/`, `figures/final_table.csv`, `figures/ranking_vs_noise.png` |
| `src/circuits.py` | 3 | Pure Perceval sandwich circuits: MZI (Clements) and tritter (DFT3) meshes | nothing |
| `src/train_financial.py` | 1bis | 1D Student-t run with tail report | `runs/financial/`, 3 `figures/*log_returns*` artifacts |
| `tests/test_circuits.py` | 3 gate | DFT3 parity, input sensitivity, trainable-count ratio | `tests/test_circuits_results.json` |
| `tests/test_review_findings.py` | review gate | Jacobian rank parity, strict sensitivity, tritter determinism, x-order (F1, F2, F7, F8, F9) | `tests/test_review_findings_results.json` |
| `tests/test_transfer.py` | 2 gate | output_keys contract, transfer alignment and semantics, `MeshGenerator` smoke, pipeline round trip | `tests/test_transfer_results.json`, `runs/smoke/` |
| `docs/exercises/ex1..ex6_*.py` | 0 | Runnable crash tests of the API traps (section 3) | nothing |
