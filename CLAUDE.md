# CLAUDE.md

Project: hybrid photonic generative models (MerLin/Perceval). Read PLAN.md for the full implementation plan, phases, and known API traps before writing any code.

## Code rules

- Comments in English.
- No decorative prints (e.g. print("=" * 80)) and no execution-tracking prints (e.g. print("Training...")). Logging goes to files (CSV/JSON).
- Never use em dashes anywhere, in code, comments, docs, or answers.
- Existing source files must never be modified, only completed. If a change to existing code seems necessary, stop and ask first.
- Code must stay simple, natural, and concise. No over-engineering, no config frameworks, no unnecessary classes.
- Seeds fixed in every script that produces a result.
- No emojis anywhere.

## Project-specific constraints (non-negotiable)

- Every QuantumLayer in this project forces the full Fock space: `merlin.MeasurementStrategy.probs(merlin.ComputationSpace.FOCK)`. Clean and noisy layers must always live in the same output basis.
- Circuit structure is always sandwich: entangling mesh, then angle encoding, then entangling mesh. Encoding placed before the first mesh acts as a global phase on the input Fock state and produces input-independent outputs.
- Every new circuit (including tritter variants) must pass the input sensitivity test before being used: std of the output distribution over a random input batch must be non-negligible (order 1e-2, not 1e-8).
- Never copy a full state_dict between layers with different noise models. Transmittance < 1 changes the internal structure. Use copy_circuit_params from src/model.py (copies trainable phases by name).
- Parameter naming convention for circuits: input encoding parameters prefixed "x", trainable parameters prefixed "theta". This is the contract between the photonic backend and the training loop.
- No quantum advantage claims anywhere in code, docs, or figures. The project measures where things break under realistic noise, honestly.

## Explanations and write-ups

- All explanation, interpretation, and answers to questions go in Markdown cells of notebooks, never in code comments.
- Academic write-up tone: a student explaining work to a classmate. Direct, natural, varied sentence lengths, frank critical remarks on results. No polished robotic academic tone.
