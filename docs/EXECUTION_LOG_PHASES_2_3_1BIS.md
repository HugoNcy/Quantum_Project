# Execution log: phases 2, 3 and 1bis (session of 2026-07-07)

This document records, in full detail, the work session that took the repository
from "phase 2/3 code written but never executed, nothing committed" to "phases 2,
3 and 1bis run, verified, interpreted and committed". Everything happened on the
branch `tony/phase2-3-noise-mesh` (8 commits, listed at the end), on the local
fallback venv: Python 3.11.6, perceval-quandela 1.2.3, merlinquantum 0.4.0,
torch 2.12.1, Windows 11. Every command below was run from the repository root
with `.venv\Scripts\python.exe`. All seeds are 0 unless stated otherwise, so
every number in this file is reproducible.

Companion documents: `docs/PLAN.md` (the plan this executes), `docs/REVIEW.md`
(the 10 adversarial findings referenced as F1..F10), and
`notebooks/results_noise_and_meshes.ipynb` (the interpretation notebook written
at the end of this session).

---

## 0. Starting state

At session start the working tree held, uncommitted:

- the F1/F2 fix in `src/circuits.py` (tritter phase placement rewritten so the
  effective parameter budget matches the MZI mesh) plus its refreshed test
  values in `tests/test_circuits_results.json`;
- additive extensions to `src/model.py` (`build_mesh_layer` with an explicit
  input state, `MeshGenerator`);
- comment-only additions to the phase 1 files (`data.py`, `losses.py`,
  `train.py`, `eval.py`) and a provenance note in `docs/OFFLINE_KIT.md`;
- four complete but never-executed modules: `src/noise.py`,
  `src/train_noise_grid.py`, `src/mismatch_matrix.py`, `src/compare_meshes.py`;
- a new test suite `tests/test_review_findings.py` (already run once).

`runs/` did not exist, `results/` and `notebooks/` were empty: no phase 2 or 3
experiment had ever been executed. Phase 1 itself was committed and its
acceptance criteria verified (MMD^2 falls from about 0.5 to the real-vs-real
floor near 4e-3 in about 150 of 800 steps; the generated scatter covers both
modes of the two-Gaussian target).

Also found at HEAD: the tail of `.gitignore` had been appended in UTF-16 by
some earlier PowerShell redirect, every second byte a NUL, so the patterns
`.claude/`, `runs/`, `log.json` (and duplicates of `__pycache__/`, `.venv/`)
were inert.

## 1. Gates run before any training

The plan was verification-first: no grid training until the untested machinery
was proven. Four gates, in order.

### 1.1 Existing test suites (re-run as scripts)

`pytest` is not installed in the venv and PyPI is unreachable from the sandbox,
so both suites ran through their `__main__` blocks, which also regenerate the
results JSONs (the repository's established pattern):

```
python tests/test_circuits.py
python tests/test_review_findings.py
```

Measured values, all within thresholds:

| check | value | threshold |
|---|---|---|
| tritter vs DFT3 max error | 7.09e-9 | < 1e-6 |
| input sensitivity, max std (mzi / tritter) | 0.0525 / 0.0312 | > 1e-2 |
| input sensitivity, mean std (mzi / tritter) | 0.0170 / 0.0168 | > 1e-3 |
| trainable counts (mzi / tritter) | 60 / 56 | ratio in [0.8, 1.25] |
| effective Jacobian rank (mzi / tritter) | 50 / 49 | gap <= 10% |
| tritter decomposition determinism | 4.19e-9 | < 1e-8 |
| x parameters in mode order | true | true |

Note the determinism margin moves between process runs (6.65e-9 on the previous
session's run, 4.19e-9 here): the Perceval decomposition solver is only
deterministic to about 1e-8-1e-9 across processes. Still passing, worth
watching.

### 1.2 New gate suite: `tests/test_transfer.py`

The riskiest untested assumption in the whole phase 2 design sat in
`noise.transfer_generator`: it realigns the first adapter Linear column-by-column
through `QuantumLayer.output_keys`, assuming key j corresponds to output column
j. That attribute exists in MerLin 0.4.0 but its semantics had never been
exercised. Five checks were written and run (`python tests/test_transfer.py`,
results in `tests/test_transfer_results.json`):

1. **Key contract.** A clean 6-mode/3-photon layer exposes exactly 56
   occupation tuples all summing to 3 (C(8,3), full Fock space forced); a P4
   (lossy) layer exposes 84 tuples with 0 to 3 photons (sum over k <= 3 of
   C(k+5,k) = 56+21+6+1); the clean set is strictly contained in the lossy set
   (28 loss-only states). All confirmed.
2. **Mechanical alignment.** After `transfer_generator(P0 -> P4)`, every
   destination adapter column either equals the source column of the same
   occupation tuple or is one of exactly 28 zero columns (loss states the
   source never saw); bias and adapter tail copied. Confirmed.
3. **Semantic ordering (the decisive one).** Destination built with
   `pcvl.NoiseModel(transmittance=0.9999)`: lossy structure (84 states) but
   physically almost clean. After transfer, identical latents give
   max |out_src - out_dst| = 6.2e-6 (tolerance 1e-2). If `output_keys[j]` did
   not correspond to output column j this would diverge at the 0.05-0.1 level.
   The column semantics are proven.
4. **MeshGenerator smoke.** The class had never been instantiated. Both mesh
   types built under clean and lossy profiles (output dims 56 / 84 as
   expected), ran two finite Adam steps on the MMD, and survived the same
   tritter P0 -> P4 transfer path `mismatch_matrix.py` uses. The
   `input_state=[1,0,1,0,1,0]` kwarg (review F10) works.
5. **Pipeline round-trip.** A 2-step `train_generator` run under
   `runs/smoke/P0` was rebuilt from its own `log.json` by
   `build_generator_from_log`, weights loaded, and evaluated by `eval_mmd`
   with finite results.

### 1.3 Timing pilots

Per-step cost had been measured at planning time (about 0.08 s/step clean,
0.15 s/step lossy, batch 256, m=6, n=3). Two 5-step end-to-end pilots
confirmed the forecast and dry-ran the full pipelines including figures:

```
python src/train_noise_grid.py --steps 5 --out-root runs/pilot_grid    (19.5 s)
python src/compare_meshes.py  --steps 5 --out-root runs/pilot_mesh    (89.2 s)
```

Even the pessimistic forecast (pilot x 160) stayed under the 60-minute gate
per script, so the real runs used the spec defaults (800 steps). All four
production scripts together take about one hour on a laptop CPU.

### 1.4 Repository repair

`.gitignore` was fixed byte-level from Python (truncate at the first NUL,
append `.claude/`, `runs/`, `log.json` as UTF-8). The `*.png` pattern that the
mangled tail also intended was deliberately dropped: `figures/` is versioned
by design (the README declares it so, and the phase 2/3 result figures must be
committable). Verified with `git check-ignore`. This came first so that `runs/`
would never pollute `git status`.

## 2. Phase 2: noise grid and mismatch matrix

### 2.1 What ran

```
python src/train_noise_grid.py
python src/mismatch_matrix.py
```

Defaults from the scripts: profiles P0..P4, dataset `two_gaussians`, 800
steps, batch 256, lr 5e-3, latent dim 6, m=6 modes, n=3 photons, seed 0,
outputs under `runs/noise_grid/<profile>/` (`history.csv`, `model.pt`,
`log.json`). The grid profile definitions (from `src/noise.py`):

| profile | indistinguishability | transmittance | output dim (FOCK forced) |
|---|---|---|---|
| P0 | 1.00 | 1.0 | 56 |
| P1 | 0.95 | 1.0 | 56 |
| P2 | 0.90 | 1.0 | 56 |
| P3 | 0.85 | 1.0 | 56 |
| P4 | 0.95 | 0.9 | 84 |

Each run asserts its output dimension against `expected_dim` before training
(review F3 guard); all five held. Training was launched as one background
process and monitored through the incrementally flushed `history.csv` files.
Every profile converged; e.g. P0 ended its last two steps at MMD^2 0.0037 and
0.0054, P1 at 0.0048, right at the floor.

`mismatch_matrix.py` then evaluated every (train profile A, eval profile B)
pair: the A-model is rebuilt from its log, weights loaded directly (same
structure), a B-configured twin is built, weights moved with
`transfer_generator` (never a raw `state_dict` load across profiles: P4 has 84
output states and extra internal keys), and the MMD against the target is
averaged over 8 fixed evaluation batches of 256 with deterministic latents.
The real-vs-real floor is averaged over 16 disjoint batch pairs (review F4).

### 2.2 The numbers

MMD^2 means, rows = training profile, columns = evaluation profile
(`figures/mismatch_mzi.csv`, full precision in `runs/noise_grid/mismatch_mzi.json`):

| train \ eval | P0 | P1 | P2 | P3 | P4 |
|---|---|---|---|---|---|
| P0 | 0.0052 | 0.0053 | 0.0054 | 0.0056 | 0.0310 |
| P1 | 0.0052 | 0.0052 | 0.0053 | 0.0054 | 0.0308 |
| P2 | 0.0061 | 0.0059 | 0.0058 | 0.0058 | 0.0324 |
| P3 | 0.0059 | 0.0059 | 0.0059 | 0.0060 | 0.0372 |
| P4 | 0.0062 | 0.0062 | 0.0063 | 0.0064 | 0.0039 |

Diagonal mean 0.00521, off-diagonal mean 0.01120, floor 0.00333 +- 0.00135,
`suspect_bug: false`. The plan's acceptance criterion (diagonal at least as
good as off-diagonal on average) holds with a factor 2 margin.

### 2.3 What it means

1. **Partial distinguishability barely hurts on this target.** The matched
   (diagonal) MMD rises only from 0.0052 (P0) to 0.0060 (P3) against a floor
   of 0.0033. Pushing indistinguishability from 1.0 to 0.85 costs almost
   nothing. Frank remark: the two-Gaussian target is probably too easy to
   stress the interference resource, since the classical adapter can
   compensate a mildly blurred Fock distribution.
2. **Losses are the real mismatch axis.** Every loss-free-trained model
   collapses under P4 evaluation (0.031 to 0.037, six to eight times its own
   diagonal). The mechanism is structural, not subtle: with transmittance 0.9
   and 3 photons, a fraction 1 - 0.9^3 = 27.1% of events lose at least one
   photon; the transferred model has zero adapter weight on all 28 low-photon
   states, so 27% of the probability mass lands where the model was never
   trained.
3. **Training under the deployment noise repairs it.** The P4/P4 cell
   (0.0039) is the best value of the whole matrix, at the floor: the generator
   learns to use the enlarged output space. The reverse transfer (P4-trained,
   evaluated clean: 0.0062-0.0064) degrades only mildly, because the 56
   three-photon columns survive the move intact.

Figures: `figures/mismatch_heatmap_mzi.png`,
`figures/mmd_final_vs_indistinguishability_mzi.png`.

## 3. Phase 3: MZI vs tritter at fair budget

### 3.1 What ran

```
python src/compare_meshes.py
```

Defaults: both meshes, all five profiles, `two_gaussians`, 800 steps, batch
256, lr 5e-3, seed 0, outputs under `runs/mesh_compare/<mesh>/<profile>/`.
Ten trainings total, then a full 5x5 mismatch matrix per mesh and the final
table and ranking figure. Both meshes go through the identical route 2
construction (`model.build_mesh_layer`: raw Perceval circuit from
`circuits.build_circuit`, sandwich structure mesh/encoding/mesh, explicit
input state |1,0,1,0,1,0>, `input_parameters=["x"]`,
`trainable_parameters=["theta"]`, full Fock space forced).

Budget fairness, checked from the run logs: MZI 60 trainable phases, tritter
56 (12 layers, offset-cycled triplets, phases only on covered modes minus one
gauge reference per triplet, per the F1 fix), measured effective Jacobian
ranks 50 vs 49. Same input state, same photon number, same Fock dimension per
profile (56 clean, 84 lossy).

### 3.2 A consistency check that fell out for free

The route 2 MZI produced a mismatch matrix **bit-for-bit identical** (max
cell difference 0.0 over all 25 cells) to the phase 2 route 1 matrix
(CircuitBuilder MZI). Same Clements topology, same parameter count and order,
same seed, therefore the same initialization and the same training
trajectory. Two independent construction paths agreeing exactly is a strong
free cross-validation of the raw-circuit route; it also means the "mzi" rows
of phase 3 are literally the phase 2 model.

### 3.3 The numbers

Matched-profile (train = eval) MMD^2 from `figures/final_table.csv`:

| profile | mzi | tritter | better |
|---|---|---|---|
| P0 | 0.005160 | 0.004022 | tritter |
| P1 | 0.005202 | 0.004374 | tritter |
| P2 | 0.005815 | 0.004030 | tritter |
| P3 | 0.005981 | 0.004337 | tritter |
| P4 | 0.003889 | 0.005843 | mzi |

(floor 0.003329; per-cell evaluation stds 0.0007 to 0.0032 are in the CSV)

Full tritter mismatch matrix (`figures/mismatch_mesh_tritter.csv`), same
convention as above:

| train \ eval | P0 | P1 | P2 | P3 | P4 |
|---|---|---|---|---|---|
| P0 | 0.0040 | 0.0040 | 0.0040 | 0.0041 | 0.0300 |
| P1 | 0.0044 | 0.0044 | 0.0044 | 0.0044 | 0.0329 |
| P2 | 0.0040 | 0.0040 | 0.0040 | 0.0041 | 0.0327 |
| P3 | 0.0044 | 0.0044 | 0.0044 | 0.0043 | 0.0308 |
| P4 | 0.0070 | 0.0070 | 0.0072 | 0.0074 | 0.0058 |

Diagonal mean 0.00452, off-diagonal 0.01027, `suspect_bug: false` for both
meshes. Acceptance criteria of the plan all met: 10 rows in the final table,
both matrices sane, explicit ranking statement possible.

### 3.4 The answer to the research question

**The ranking is stable under partial distinguishability and it inverts under
losses.**

- On P0 through P3 the tritter mesh is consistently better (0.0040-0.0044 vs
  0.0052-0.0060) with four fewer trainable phases. Same direction on all four
  profiles.
- Under P4 the ranking flips: the MZI lands at the floor (0.0039) while the
  tritter degrades to 0.0058. The tritter also transfers worse out of P4
  (P4-trained evaluated clean: 0.0070-0.0074 vs 0.0062-0.0064 for the MZI).

Honest caveats, stated wherever this result appears: one training seed, and
the per-cell evaluation std (0.001-0.003) is of the same order as the gaps,
so each individual comparison is a one-sigma statement. What lends it weight
is the consistency of the direction across profiles, not any single cell.
Seed replication (3 to 5 seeds) is the declared next step before claiming
more. A flat result would also have been reportable; a noise-dependent
ranking is the more interesting outcome for the question asked.

Figures: `figures/ranking_vs_noise.png` (the two lines cross between P3 and
P4, with visibly overlapping error bars, which is the honest picture),
`figures/mismatch_heatmap_mesh_mzi.png`,
`figures/mismatch_heatmap_mesh_tritter.png`, plus the per-mesh
`mmd_final_vs_indistinguishability_mesh_*.png`.

## 4. Phase 1bis: heavy-tailed financial target

### 4.1 Design decisions

No market data is reachable from the offline environment, so the target is
synthetic but principled: a **standardized Student-t with 4 degrees of
freedom**, the classic stand-in for daily log-returns (leptokurtic, infinite
kurtosis at df=4, tail exponent in the empirical ballpark). Two implementation
details worth recording:

- `torch.distributions` samplers do not accept a `torch.Generator`, so the
  t variable is built from its definition t = z / sqrt(chi2/df), with the
  chi2(4) as the sum of 4 squared standard normals, everything drawn from one
  seeded generator. Sanity-checked on 200k samples: mean -0.004, std 0.999,
  sample excess kurtosis about 14.6 (large and unstable by construction at
  df=4, which is fine: heavy tails are the point), bit-reproducible.
- **Standardizing to unit variance resolves review finding F5** (the default
  MMD bandwidths 0.1..5 would be myopic on raw return scales of order 1e-2)
  without touching `losses.py`: standardization is affine, so the tail shape
  the phase is about is unchanged.

New code, both pure additions: `log_returns(n, seed)` appended to
`src/data.py` (registered as `DATASETS["log_returns"]`, shape (n, 1)), and
`src/train_financial.py` (PhotonicGenerator with `out_dim=1`, reuses
`train_generator`, then writes the evaluation artifacts itself). The run log
records `out_dim: 1` because `build_generator_from_log` assumes 2D; a
financial run must be rebuilt manually.

### 4.2 What ran and what came out

```
python src/train_financial.py
```

(800 steps, batch 256, seed 0, outputs `runs/financial/` plus three artifacts
in `figures/`). Training converged, final steps at MMD^2 0.002-0.004.
Evaluation on 20000 fresh samples (seed 1), quantile report
(`figures/tails_log_returns.csv`):

| quantile | target | generated |
|---|---|---|
| 0.001 | -4.846 | -2.448 |
| 0.01 | -2.649 | -2.143 |
| 0.05 | -1.506 | -1.479 |
| 0.5 | -0.002 | -0.017 |
| 0.95 | 1.503 | 1.466 |
| 0.99 | 2.627 | 2.098 |
| 0.999 | 4.859 | 2.325 |

The bulk is captured well (5% / 50% / 95% within a few percent). The tails
are not: at the 0.1% / 99.9% level the target reaches +-4.85 while the
generator stops near -2.45 / +2.33. The QQ plot
(`figures/qq_log_returns.png`) flattens symmetrically at both ends, and the
log-density histogram (`figures/hist_log_returns.png`) shows the generated
support ending around +-2.5 while the target extends past +-5 (with stray
draws to -20 in 20k samples, as an infinite-kurtosis distribution will).

Why this happens: the sample is an affine map of a 56-bin probability vector
driven by tanh-bounded encoder angles, so the output range is effectively
bounded, and with bandwidths of 0.1 and larger the MMD assigns almost no
loss to missing 0.1% tail mass. This is exactly the outcome the project plan
predicted and asked to be documented honestly: a generator at this scale does
not capture fat tails, and that is a result, not a failure. Plausible
follow-ups: sub-0.1 bandwidths, a heavy-tailed latent instead of the
Gaussian, or an explicit quantile penalty.

## 5. Write-up artifacts produced

- `notebooks/results_noise_and_meshes.ipynb`: 13 cells, interpretation in
  Markdown per the project convention, four code cells that only read from
  `runs/` and `figures/` (all four verified to execute exactly as Jupyter
  would, sequentially with the notebook directory as cwd; the notebook ships
  unexecuted because the local venv has no Jupyter, and it renders on GitHub
  through the committed figures). Includes a 3-panel matched-profile scatter
  cell rebuilding generators from their logs.
- README: new "Results (simulation, July 2026)" section with the headline
  numbers, the two key figures, the mechanism sentence for the P4 collapse,
  the reproduction commands, and a status line under the roadmap. Inserted as
  a pure addition, no existing line modified.
- The compact CSV summaries (`mismatch_mzi.csv`, `mismatch_mesh_mzi.csv`,
  `mismatch_mesh_tritter.csv`) were copied into `figures/` and versioned,
  next to `final_table.csv` which the comparison script already writes there.
  The full-precision JSONs stay in `runs/` (gitignored, reproduced by the
  scripts), consistent with the README's contract that `results/`-style data
  is reproduced, not versioned.

## 6. Comment pass (block level)

The working tree already carried block-level English comments on all of
`src/` from the previous session; this session's sweep verified coverage file
by file, confirmed zero em dashes and zero French in `src/`, and topped up
six remaining bare non-obvious spots: the `str(getattr(...))` serialization
of `input_state` in both training scripts, the shared-reference-log and
eval-pool sizing lines and the inverted x-axis in `mismatch_matrix.py`, and
the `n_thetas` audit field and shared-floor line in `compare_meshes.py`.

## 7. Deliberately not touched

- `docs/exercises/ex1_perceval_circuit.py`: carries the user's own
  uncommitted experiment (including a `u * u_conjugate` element-wise print
  that is not a unitarity check). Left strictly untouched and uncommitted by
  explicit user decision.
- `requirements.txt` pins `perceval-quandela==1.2.4` while the validated venv
  and the README both say 1.2.3. Flagged as a team decision, not changed.
- `main.pdf` at the repo root: left untracked.
- Nine pre-existing em dashes in the committed README predate the
  no-em-dash convention; only the one introduced this session was removed.
- `eval.py` still rebuilds without a noise argument (the documented F3
  caveat); noisy-run scatters go through `build_generator_from_log` in the
  notebook instead of an `eval.py` modification.

## 8. Commits (in order)

| hash | content |
|---|---|
| 9649f91 | fix gitignore tail corrupted by utf-16 redirect (.claude/, runs/, log.json) |
| 7844ad6 | fix tritter phase placement for effective budget parity and add review tests (F1 F2 F7 F8 F9) |
| 168bd25 | add route 2 mesh generator with explicit input state (review F10) |
| 0a0131a | complete block-level comment pass on phase 1 files and note dual validation env in offline kit |
| c738aec | add phase 2 noise grid: profiles, fock-key weight transfer, grid training, transfer gate tests |
| 41a0660 | add phase 1bis: standardized student-t log-returns target and 1d financial run with tail report |
| 9553a0b | add mismatch matrix and mzi vs tritter comparison scripts (phases 2 and 3) |
| 90710a2 | add phase 1bis/2/3 results: mismatch matrices, mesh ranking, tails report, notebook and readme results |

After the last commit, `git status` shows only the ex1 experiment (modified,
kept on purpose) and `main.pdf` (untracked). The branch has not been pushed;
merging via PR is the team's call.

## 9. What remains

- **Phase 4** (inference of the best model on Quandela hardware, treating the
  QPU as an unknown noise profile inside exactly this mismatch framework) is
  pending cloud access; the `.env` `QUANDELA_TOKEN` mechanism is already
  documented in the README.
- **Seed replication** (3 to 5 seeds) to firm up or kill the mesh-ranking
  result, and a harder target (ring) to separate the architectures away from
  the floor.
- The perceval version pin discrepancy above needs a team decision.
