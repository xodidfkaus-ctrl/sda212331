# Experiment Lifecycle — EXAONE 4.5 NoPE Research

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    EXPERIMENT LIFECYCLE (pre-registered)                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

  ┌─────────────────────────────────────────────────────────┐
  │  RESEARCH QUESTION (RQ1 / RQ2 / RQ3)                   │
  │  Source: HANDOVER.md §2, PREREGISTRATION.md             │
  └───────────────────────┬─────────────────────────────────┘
                          │
                          ▼
  ┌─────────────────────────────────────────────────────────┐
  │  PLAN.md                                                │
  │  ─────────────────────────────────────────────────────  │
  │  • Hypothesis (null + alternative)                      │
  │  • Method (script, corpus, n, seq lengths)              │
  │  • Decision Criteria (```criteria YAML block)           │
  │    ├── metric, direction                                │
  │    ├── accept: {min_abs_cohen_d, max_p_holm_corrected}  │
  │    ├── min_n_per_group, n_simultaneous_tests            │
  │    └── retrofitted: false ← REQUIRED for pre-reg       │
  │  • Stopping rules                                       │
  │                                                         │
  │  ⚠  COMMIT THIS BEFORE RUNNING ANY CODE ⚠             │
  │     git timestamp must precede first results.jsonl      │
  └───────────────────────┬─────────────────────────────────┘
                          │
                          │  git commit experiments/e{NNN}_{slug}/PLAN.md
                          │
                          ▼
  ┌─────────────────────────────────────────────────────────┐
  │  RUN (experiment script)                                │
  │  ─────────────────────────────────────────────────────  │
  │  nope_analysis/experiments/exp{N}_{name}.py             │
  │                                                         │
  │  Required at script top:                                │
  │    from nope_analysis.seeds import set_all_seeds        │
  │    set_all_seeds(get_seed("e{NNN}"))                    │
  │                                                         │
  │  Required output files:                                 │
  │    outputs/{slug}/results.jsonl  (append-only)          │
  │    outputs/{slug}/summary.json                          │
  │    outputs/{slug}/run_config.json                       │
  │    outputs/{slug}/*.png          (at least 1 chart)     │
  │                                                         │
  │  Model loading:                                         │
  │    from nope_analysis.loader import load_model_and_tokenizer
  │    (NOT AutoModelForCausalLM — config patch required)   │
  │    attn_implementation='eager' mandatory for hooks      │
  └───────────────────────┬─────────────────────────────────┘
                          │
                          ▼
  ┌─────────────────────────────────────────────────────────┐
  │  AUTO_VALIDATE  (nope_analysis/analysis/auto_validate.py│
  │  ─────────────────────────────────────────────────────  │
  │  Called at the END of every experiment script:          │
  │                                                         │
  │    validate_experiment(                                 │
  │        experiment_id="e{NNN}",                          │
  │        comparisons={                                    │
  │            "metric_name": (group_a, group_b),           │
  │        },                                               │
  │        output_dir=Path("outputs/{slug}"),               │
  │    )                                                    │
  │                                                         │
  │  Steps performed internally:                            │
  │    1. Locate experiments/e{NNN}_*/PLAN.md → ERROR if missing
  │    2. Parse ```criteria YAML block                      │
  │    3. Welch t-test + Cohen's d  (per metric)            │
  │    4. Bootstrap CI (n=10,000, seed=42, percentile)      │
  │    5. Holm-Bonferroni correction over k metrics         │
  │    6. Verdict per metric + overall:                     │
  │       ├── VALIDATED   : p_Holm ≤ 0.01 AND |d| ≥ 0.5 AND n ≥ min_n
  │       ├── FAILED      : p > 0.05  OR  |d| < 0.2       │
  │       └── INCONCLUSIVE: marginal / underpowered         │
  │    7. Write outputs/{slug}/stats.json                   │
  │    8. Warn if ANALYSIS.md missing                       │
  └───────────────────────┬─────────────────────────────────┘
                          │
                          │  git commit outputs/{slug}/stats.json
                          │             outputs/{slug}/results.jsonl
                          │             outputs/{slug}/summary.json
                          │
                          ▼
  ┌─────────────────────────────────────────────────────────┐
  │  ANALYSIS.md                                            │
  │  ─────────────────────────────────────────────────────  │
  │  Written by researcher after reviewing stats.json:      │
  │                                                         │
  │  • Results table (copy from summary.json)               │
  │  • Statistical tests (copy from stats.json)             │
  │  • Decision (copy overall_verdict from stats.json)      │
  │  • Deviations from Plan                                 │
  │    └── ⚠ HARKING RISK: flag EVERY post-hoc decision    │
  │  • Post-hoc Rationale (if any deviations exist)         │
  │  • Follow-up (what does this motivate?)                 │
  │                                                         │
  │  Update STATUS file:  DONE                              │
  │                                                         │
  │  ⚠  COMMIT THIS BEFORE MARKING EXPERIMENT COMPLETE ⚠  │
  └───────────────────────┬─────────────────────────────────┘
                          │
                          │  git commit experiments/e{NNN}_{slug}/ANALYSIS.md
                          │             experiments/e{NNN}_{slug}/STATUS
                          │
                          ▼
  ┌─────────────────────────────────────────────────────────┐
  │  EXPERIMENT_LOG.md update                               │
  │  ─────────────────────────────────────────────────────  │
  │  Add row: ID | RQ | Hypothesis | Verdict | Date | GPU   │
  │  Link to PLAN.md, ANALYSIS.md, stats.json               │
  └───────────────────────┬─────────────────────────────────┘
                          │
                    ┌─────┴──────┐
                    │            │
                    ▼            ▼
  ┌─────────────────┐      ┌─────────────────────────────────┐
  │ NEGATIVE_       │      │  FINDINGS.md                    │
  │ RESULTS.md      │      │  ──────────────────────────────  │
  │ ─────────────── │      │  Only if verdict = VALIDATED    │
  │ If FAILED or    │      │  OR as labeled PRELIMINARY if   │
  │ INCONCLUSIVE or │      │  INCONCLUSIVE (with upgrade path│
  │ aborted:        │      │                                 │
  │ • Add rejected  │      │  Each bullet MUST cite:         │
  │   hypothesis    │      │   [e{NNN}] experiment ID        │
  │   section       │      │                                 │
  │ • Record OOM /  │      │  NEVER contains conclusions     │
  │   bug details   │      │  from HANDOVER.md               │
  └─────────────────┘      └─────────────────────────────────┘


═══════════════════════════════════════════════════════════════
  FAST PATH: retrofit (e001–e004)      NORMAL PATH (e005–e008+)
  ────────────────────────────         ──────────────────────
  PLAN.md written after results        PLAN.md committed first
  retrofitted: true in criteria        retrofitted: false
  Verdict → EXPLORATORY only           Verdict → may be VALIDATED
  Cannot be primary paper claim        Can be primary paper claim
═══════════════════════════════════════════════════════════════


KEY FILES AND THEIR ROLES
─────────────────────────────────────────────────────────────
  PREREGISTRATION.md    — hypotheses + decision rules (source of truth)
  STATISTICAL_PROTOCOL.md — Holm-Bonferroni, effect size thresholds, sample sizes
  experiments/e{NNN}_*/PLAN.md    — per-experiment plan (commit BEFORE running)
  experiments/e{NNN}_*/ANALYSIS.md — per-experiment results + deviations
  experiments/e{NNN}_*/STATUS     — PENDING / DONE / ABORTED
  outputs/{slug}/stats.json       — auto_validate output (machine-readable verdict)
  nope_analysis/seeds.py          — global seed registry (never set seeds inline)
  environment.lock.yaml           — pip freeze + HF model SHA pin
  EXPERIMENT_LOG.md               — chronological table of all experiments
  NEGATIVE_RESULTS.md             — rejected hypotheses + aborted experiments
  FINDINGS.md                     — paper-grade findings (cite experiment IDs)
  HANDOVER.md                     — session setup + architecture facts (NO conclusions)
```

---

*Generated: 2026-04-26 | Applies from experiment e005 onward; e001–e004 retrofitted*
