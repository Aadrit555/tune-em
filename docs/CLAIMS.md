# Claim-to-Evidence Matrix

Every substantive project claim traced to implementation, evidence, benchmarks,
assumptions, limitations, and a reproduction command. Status labels:
**STABLE** (tested, reproducible), **EXPERIMENTAL** (implemented, narrowly validated),
**HEURISTIC** (no formal guarantee), **NOT CLAIMED** (explicitly out of scope).

Conventions: all commands run from the repo root. `backend=mock` means the
deterministic hash-based test backend, NOT a real language model. Mock results
demonstrate machinery (plumbing, shapes, invariants), never model capability.

---

## 1. Core decision pipeline (STABLE)

- **Claim:** A typed pipeline maps prompt → candidates → tokenizer-aware
  representation → forward pass → candidate-conditional distribution →
  optional calibration → diagnostics → utility policy → decision/abstention
  with structured telemetry.
- **Implementation:** `anydecision/core/engine.py` (`decide`, `batch_decide`,
  `decide_adaptive`), `anydecision/core/question.py`, `anydecision/core/decision.py`.
- **Evidence:** `tests/test_question.py`, `tests/test_universal_von_api.py`,
  `tests/test_reproducibility.py` (127+ tests pass).
- **Benchmark:** `benchmarks/run_benchmark.py` → `benchmarks/results/benchmark_summary.json`
  (40 train / 40 test customer-escalation samples, mock backend).
- **Assumptions:** Open-weight causal LM exposing next-token logits (or mock backend).
- **Limitations:** Real-model numbers require a real backend; mock numbers only
  validate machinery.
- **Reproduce:** `python -m pytest tests/test_question.py -q`

## 2. Candidate-conditional vs raw vocabulary probability (STABLE)

- **Claim:** Every decision exposes `choice_probability` (candidate-conditional
  P(candidate | prompt, candidate set)), `raw_vocab_logprob` (unconstrained
  vocabulary log-probability), `model_token_probability`, `choice_margin`,
  and `predictive_entropy` as distinct quantities.
- **Implementation:** `anydecision/backends/transformers.py`
  (`next_token_logprobs_detailed`), `anydecision/scoring/`.
- **Evidence:** `tests/test_scoring.py` (raw vs conditional distinction,
  normalization).
- **Limitations:** vLLM top-k misses fall back to sequence scoring; documented
  as a bounded approximation path, not exact logprobs.
- **Reproduce:** `python -m pytest tests/test_scoring.py -q`

## 3. Tokenizer-aware candidate scoring (STABLE)

- **Claim:** Single- vs multi-token classification is decided by the tokenizer
  in prompt context, shared by all backends; joint vs length-normalized scores
  are exposed separately and never conflated.
- **Implementation:** `anydecision/scoring/candidate_tokenizer.py`,
  `anydecision/scoring/sequence.py`.
- **Evidence:** `tests/test_scoring.py` (yes/no, C++, New York, leading
  whitespace, punctuation, Unicode, shared first tokens, outside-top-k).
- **Reproduce:** `python -m pytest tests/test_scoring.py -q`

## 4. L1 option-order sensitivity (STABLE, effect-size qualified)

- **Claim:** L1 permutation debiasing measures and reduces option-order
  sensitivity (mean total-variation distance across deterministic permutations).
- **Implementation:** `anydecision/bias/permutation.py`, engine L1 path.
- **Evidence:** `tests/test_bias_permutation.py`, `tests/test_batch_async.py`
  (batch/single numerical equivalence).
- **Benchmark:** `benchmark_summary.json` records `mean_position_bias` per level.
- **Limitations:** Magnitude of bias reduction is model- and prompt-dependent;
  no universal percentage claimed.
- **Reproduce:** `python -m pytest tests/test_bias_permutation.py tests/test_batch_async.py -q`

## 5. Calibration with held-out validation (STABLE, mock-measured)

- **Claim:** L2 post-hoc calibration (temperature/vector/isotonic/Platt) is fit
  on validation data disjoint from test data and evaluated with accuracy, NLL,
  Brier, ECE, adaptive ECE, reliability curves, selective risk, coverage, and
  risk-coverage curves.
- **Implementation:** `anydecision/calibration/`, `anydecision/evaluation/benchmark.py`,
  `benchmarks/run_benchmark.py` (40 train / 40 held-out test).
- **Evidence:** `tests/test_calibration.py`; measured mock-backend deltas:
  L0 ECE 0.411 → L2 ECE 0.260; NLL 1.923 → 0.710; Brier 0.889 → 0.517
  (`benchmarks/results/benchmark_summary.json`).
- **Assumptions/Limitations:** Numbers above use `backend=mock` (synthetic
  labels); they validate the calibration machinery, not real-model gains.
  Real-model calibration must be re-measured per model/revision/seed and
  recorded with full metadata.
- **Reproduce:** `python benchmarks/run_benchmark.py`

## 6. Selective conformal / risk control (STABLE procedure, tiered claims)

- **Claim:** Risk-control outputs are mechanically tiered as `formal_conformal`
  (split threshold-selection + independent certification set),
  `high_confidence_empirical`, `heuristic`, or `unavailable`. The `formal`
  label is never applied to same-sample threshold searches.
- **Implementation:** `anydecision/calibration/selective_conformal.py`,
  `anydecision/calibration/conformal.py`.
- **Evidence:** `tests/test_conformal_hierarchical.py` (thousands of
  exchangeable simulation draws; malformed-record rejection).
- **Assumptions:** Exchangeability between calibration and test; stated
  alpha; documented multiple-threshold handling.
- **Limitations:** Guarantee holds under its assumptions only; distribution
  shift voids it (see §10).
- **Reproduce:** `python -m pytest tests/test_conformal_hierarchical.py -q`

## 7. Active calibration (HEURISTIC selection; NO fixed savings claim)

- **Claim:** Information-theoretic (entropy/margin/disagreement/shift-suspicion)
  sampling prioritizes label budgets; cost savings are ONLY reported from
  measured budget curves via `estimate_label_savings()`, never from a
  single-budget ECE delta. No fixed percentage (e.g. 75%) is claimed.
- **Implementation:** `anydecision/calibration/active.py`
  (`run_benchmark`, `evaluate_budget_curve`, `estimate_label_savings`).
- **Evidence:** `tests/test_conformal_hierarchical.py`
  (`test_active_calibration_no_cost_claim_without_curve`,
  `test_estimate_label_savings_requires_curve_evidence`).
- **Limitations:** Savings are dataset/model/budget-dependent; unmeasured until run.
- **Reproduce:** `python -m pytest tests/test_conformal_hierarchical.py -q -k active`

## 8. Expected utility (STABLE)

- **Claim:** Action selection maximizes EU(a) = Σ P(y|x)·U(a,y) with an
  EXPLICIT action→state mapping. No name guessing (`act.lower()==state.lower`
  removed); misclassification costs keyed by explicit `positive_class`, so
  semantics are invariant to class ordering.
- **Implementation:** `anydecision/theory/utility.py` (`from_action_costs` with
  `correct_action_for_state`, `standard_classification_costs` with
  `positive_class`).
- **Evidence:** `tests/test_theory.py` (explicit-mapping, permutation-invariance).
- **Reproduce:** `python -m pytest tests/test_theory.py -q`

## 9. Adaptive compute (STABLE accounting, enforced budget)

- **Claim:** Every adaptive decision records actual backend calls, token counts
  (exact when the backend tokenizer is available, otherwise labeled
  `tokens_estimated=True`), latency, compute path, and exit reason.
  `max_backend_calls` is a hard budget: stages whose predicted calls would
  exceed it are never started.
- **Implementation:** `anydecision/adaptive/router.py`,
  `BaseBackend.count_tokens` / `token_counts_exact`.
- **Evidence:** `tests/test_adaptive.py` (budget ceiling across budgets 1–5,
  estimated-flag on mock, exit reasons).
- **Reproduce:** `python -m pytest tests/test_adaptive.py -q`

## 10. Distribution-shift suspicion (HEURISTIC, not OOD detection)

- **Claim:** The `DistributionShiftHeuristic` reports shift-suspicion signals
  (normalized entropy, confidence collapse, order/template instability,
  embedding distance). It is NOT a validated OOD detector: no
  AUROC/AUPR/FPR95 evaluation on ID vs near-OOD vs semantic-OOD exists.
- **Implementation:** `anydecision/uncertainty/ood.py` (`OODDetector` kept as a
  deprecated alias).
- **Evidence:** `tests/test_uncertainty_abstention.py`.
- **Reproduce:** `python -m pytest tests/test_uncertainty_abstention.py -q`

## 11. Artifact integrity (STABLE tamper detection; OPTIONAL authenticity)

- **Claim:** Artifacts carry a SHA-256 tamper-detection hash over model,
  tokenizer, calibration, config, schema, and package-version fields plus
  compatibility checks. This is NOT cryptographic authenticity. Authenticity
  requires the optional Ed25519 signature verified against a trusted public key.
- **Implementation:** `anydecision/artifacts/saver.py` (`sign_payload_hash`,
  `verify_payload_signature`, `require_signature` / `trusted_public_key_hex`).
- **Evidence:** `tests/test_artifacts.py`.
- **Reproduce:** `python -m pytest tests/test_artifacts.py -q`

## 12. Serving (STABLE bounds; admin auth)

- **Claim:** Public decision API (`/decide`, `/batch`) is rate-limited,
  concurrency-bounded, input-bounded (question chars, options, batch size),
  timeout-guarded, request-ID traced, and never leaks raw exceptions.
  `/calibrate` is an admin endpoint requiring `X-API-Key`
  (`ANYDECISION_ADMIN_KEY`); disabled (403) without a key. `/model` always
  reports the active backend and flags `is_mock`.
- **Implementation:** `anydecision/serving/app.py`.
- **Evidence:** `tests/test_api.py` (oversize rejection, auth, safe errors,
  mock flag).
- **Reproduce:** `python -m pytest tests/test_api.py -q`

## 13. ViZDoom integration (STABLE harness; mock-measured policy runs)

- **Claim:** REAL ViZDoom state → observation (STATE/VISION/HYBRID) → typed
  anydecision candidate scoring → UtilityMatrix → `Decision.selected_action`
  → REAL game action → REAL transition → authoritative KILLCOUNT/DEATHCOUNT/
  ITEMCOUNT/DAMAGE_TAKEN. Executed action always equals the selected action
  (assertion-enforced). No hardcoded belief distributions. Victory is
  scenario-defined (combat maps: kills > 0; collection/navigation: survival).
  Scorecards report facts only.
- **Implementation:** `anydecision/games/vizdoom_env.py`.
- **Evidence:** `tests/test_vizdoom.py` (chain test, utility-sensitivity,
  model-sensitivity, abstention fallback, factual-report, baselines).
- **Benchmark (mock backend, `basic`, 3 episodes, HYBRID):**

  | policy | seed | won/3 | kills | decisions | mean lat (ms) |
  |---|---|---|---|---|---|
  | anydecision | 0 | 3 | 3 | 35 | 4.14 |
  | anydecision | 1 | 2 | 2 | 79 | 4.18 |
  | anydecision | 2 | 2 | 2 | 129 | 4.12 |
  | random | 0 | 2 | 2 | 79 | ~0.00 |
  | random | 1 | 3 | 3 | 77 | ~0.00 |
  | random | 2 | 3 | 3 | 11 | ~0.00 |
  | scripted | 0 | 3 | 3 | 17 | ~0.00 |
  | scripted | 1 | 3 | 3 | 16 | ~0.00 |
  | scripted | 2 | 3 | 3 | 8 | ~0.00 |

  Artifacts: `benchmarks/results/vizdoom_basic_<policy>_seed<seed>.json`.
  Interpretation: on trivial `basic`, everything scores; scripted dominates.
  With `backend=mock`, anydecision's "beliefs" are hash-based — this validates
  the control loop and metric plumbing, NOT model tactical skill. Real-model
  policy comparison is future work (see Limitations).
- **Assumptions:** ViZDoom ≥1.3 installed; bundled scenarios need no WAD.
- **Limitations:** Mock-backend policy skill is meaningless; VISION mode still
  reads numeric HUD values; E1M*-style WAD maps use a kills-or-survival proxy
  (exit-switch state untracked).
- **Reproduce:**
  `anydecision vizdoom --scenario basic --episodes 3 --seed 0 --policy anydecision --observation-mode HYBRID --output out.json`

## 14. Real-DOOM WAD evaluation (STABLE harness; mock-measured run)

- **Claim:** `anydecision real-doom --map E1M1 --wad DOOM.WAD` validates the
  WAD, parses genuine THINGS-lump entities (real initial state), and runs the
  policy in the live ViZDoom engine on that map. No damage/kills/pickups are
  fabricated. Missing WAD or missing vizdoom → actionable error, never a
  silent synthetic substitute.
- **Implementation:** `anydecision/games/ultimate_doom.py` (parser +
  ViZDoom-backed `RealDoomEvaluator`).
- **Evidence:** `tests/test_ultimate_doom.py` (parser + live evaluation +
  loud missing-WAD failure).
- **Benchmark (mock backend, E1M1, 2 episodes, seed 0):** WAD census 6
  monsters / 21 pickups; engine run 800 decisions, 0 kills, 0 deaths
  (`benchmarks/results/real_doom_E1M1_seed0.json`). The mock policy wanders a
  real map ineffectively — reported as-is.
- **Limitations:** WAD file required (not bundled); outcome uses the
  kills-or-survival proxy, not exit-switch clearance.
- **Reproduce:**
  `anydecision real-doom --map E1M1 --wad /path/to/DOOM.WAD --episodes 2 --seed 0 --output out.json`

## 14b. Learned state estimator (TRAINED, vision-only clone parity)

- **Claim:** A compact MLP (32→16, 6 vision-only features) trained on 936
  real scripted-demonstrator steps (basic + defend_the_center, kill-weighted +
  class-balanced, split by run) reproduces the demonstrator: train 1.000,
  held-out-run dev 1.000 (n=107). Closed-loop on fresh seeds it matches the
  scripted policy exactly (basic: 2/2 kills, identical action distribution;
  defend: identical 474-decision runs) while using VISION observations only.
- **Implementation:** `anydecision/games/doom_estimator.py`,
  `PolicyKind.LEARNED` in `vizdoom_env.py`; artifact
  `artifacts/doom_estimator_vision.{npz,json}` (weights + provenance).
- **Evidence:** `tests/test_vizdoom.py`
  (`test_learned_policy_executes_trained_estimator`: clone parity + kill parity);
  `benchmarks/results/vizdoom_basic_learned_seed0.json` (3/3 won, 3 kills).
- **Limitations:** Clone parity, not superhumanity: it cannot exceed the
  scripted demonstrator by construction (4-action repertoire). Retrain via
  trajectory logging (`record_trajectory=True`) + `train_estimator` for new
  scenarios; report holds only for the logged regime.
- **Reproduce:**
  `anydecision vizdoom --scenario basic --episodes 3 --seed 0 --policy learned --observation-mode VISION --output out.json`

## 15. Synthetic toy combat (SYNTHETIC, labeled)

- **Claim:** `anydecision/games/doom.py` and `anydecision doom` are a SEEDED
  SYNTHETIC simulator for deterministic tests/CI/policy unit tests — not real
  DOOM, not ViZDoom. Never compared against real-engine results.
- **Evidence:** `tests/test_doom.py`; module/CLI labels.
- **Reproduce:** `python -m pytest tests/test_doom.py -q`

## 16. Batching and backend parity (STABLE contracts)

- **Claim:** `batch_decide()` is numerically equivalent to sequential
  `decide()` up to documented float tolerance (ordering/counts preserved);
  Transformers and vLLM obey the same candidate-scoring contract, with
  top-k-miss handling exposed as an approximation, never exact logprobs.
- **Implementation:** `anydecision/core/engine.py`,
  `anydecision/backends/consistency.py`.
- **Evidence:** `tests/test_batch_async.py`, `tests/test_backends.py`.
- **Limitations:** vLLM parity needs a GPU rig; not measured in CI here.
- **Reproduce:** `python -m pytest tests/test_batch_async.py tests/test_backends.py -q`

## 17. Layer trajectory (EXPERIMENTAL)

- **Claim:** Inspects real hidden states where the backend exposes them;
  emergence layer is mathematically defined; mock-backend trajectories are
  seeded illustrations, not measurements.
- **Implementation:** `anydecision/representations/trajectory.py`.
- **Evidence:** `tests/test_trajectory.py` (incl. small-HF-model test where
  available).
- **Limitations:** Demonstrated on limited architectures; not a general
  cross-architecture capability claim.
- **Reproduce:** `python -m pytest tests/test_trajectory.py -q`

## 18. Explicitly NOT claimed

- No "Farama verified" status (ViZDoom integration only).
- No fixed annotation-cost percentage.
- No formal OOD detection (heuristics only).
- No SHA-256-as-authenticity (tamper detection only, unless Ed25519-signed).
- No universal victory predicate (scenario-defined only).
- No subjective scorecard ratings.
