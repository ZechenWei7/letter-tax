# The Letter Tax: How Much Shorter Is a Letter-Free Reasoning Trace Than Length-Optimized Language?

## A pre-registered fair measurement on a certified-deductive task

2026-09-16 · @Someone · DRAFT — living document, results to be filled in

## Abstract

Several methods report 3–16× fewer reasoning tokens when chain-of-thought is written in symbolic or abstract tokens instead of natural language. For the papers cited, those figures compare against verbal chains that were never length-optimized, on benchmarks the same models largely solve with no chain at all, and without a control for whether the compact trace carries computation. We measure the quantity those comparisons need and lack: on a synthetic deductive task certified to require an intermediate trace, the difference in trace length between RL-length-optimized policies that may use letters and RL-length-optimized policies that may not, at matched externalized accuracy — the *letter tax* — together with whether the letter-free trace is load-bearing. Both policies start from the same checkpoint, share reward, optimizer, and stopping rule, and differ only in the reachable vocabulary; a same-instrument arm that leaves single letters reachable separates "letters" from "hard constraint"; a deterministic word-removal transform of the letter-permitted policy's traces gives an encoding-only measurement with no RL; a same-alphabet content-destroyed policy gives the content floor. Decision rules, readings, and kill criteria are fixed here before any run. **Results: TBD.** We do not claim anything about a general non-human code, about frontier models, or about content that cannot be scored, and we do not call a letter-free notation "non-linguistic".

## 1. Introduction

Human language is the medium in which large language models think out loud, but nothing about that medium was designed for them. Its information density was shaped by the speed of human speech and the limits of human memory; its redundancy protects against noisy channels a model does not face. It is reasonable to ask whether a model could reason more cheaply in a representation of its own. That question, in its strong form — a *general* code serving across tasks, more efficient than language — requires multi-task RL at scale and a test for a shared code emerging, and is out of reach here; to our knowledge it has not been attempted. It also runs into a structural limit that applies to anyone using verifiable rewards: any task with a verifier has a formal encoding, and a formal encoding is a symbolic solution, so for any scorable task the answer to "can it be done without language" is "yes, in principle" before any experiment. The content types where language is plausibly indispensable — ambiguity, analogy, open-ended argument — cannot be scored, so cannot be trained this way.

What remains testable is what RL *finds* under pressure and what it costs. Recent work answers optimistically: abstract-token CoT (IBM, 2026), authored symbolic shorthand (ORION, 2025), and prompted symbol systems (CLSR, 2026) each report large token reductions at unchanged accuracy. But each compares against a verbal chain that was never length-optimized, on benchmarks the same model largely solves with no chain (IBM's own ablation: 85.8% on MATH-500 with no CoT vs 90.8% with abstract CoT), and none controls for whether the compact trace carries computation rather than acting as extra forward passes. Separately, Little (2026) and Kaufmann et al. (2026) show that ordinary length-penalized RL removes about half of a natural-language trace on tasks that need one — and that the trace stays English. The two lines have not been crossed.

This paper makes the measurement that crossing them requires. We define the **letter tax** as the relative excess length of an RL-length-optimized trace that may use letters over one that may not, at matched externalized accuracy, on a task certified to need a trace. We measure it on a sparse unique-linear-extension ordering task at Qwen3-4B scale, with the controls the comparison needs, and pre-register every decision rule. The modal expectation — shared by four independent design reviewers — is that the tax is small, because length-optimized language sheds most of its words on its own. If so, the 3–16× figures cannot be read as savings from leaving language; if the tax is large and the letter-free trace is load-bearing, that direction has its first controlled support.

**Contributions.**

1. An admission procedure that certifies a (model, task, difficulty) cell as *content*-necessary rather than merely depth-necessary, and with enough headroom above the task's oracle floor for compression to be observable (§4.2).
2. A same-instrument comparison — unconstrained RL, RL with single letters reachable, RL under a letter ban — from one checkpoint with one reward, plus an RL-free encoding-only transform and a same-alphabet content floor (§4.3).
3. Content diagnostics that distinguish a load-bearing trace from filler: transplant, resampling, a copy-rate gate, and a supervised decoder from trace segments to ground-truth partial assignments, with its own negative controls (§4.6).
4. Pre-registered decision rules, readings, and kill criteria, fixed after five rounds of adversarial design review and before any run (§4.7, §5, §9).
5. Results: TBD.

The question has acquired practical weight. OpenAI's GPT-6 Astra system card reports that the model shortens its chain of thought when told a monitor is present and introduces recurrent-depth computation that leaves fewer legible traces. Whether a compressed trace is denser or merely emptier bears directly on what chain-of-thought monitoring can be expected to see.

## 2. Related work

Prior work splits into two groups that have not been compared under the same conditions. The table records, for each paper, what it claims in its own words (abstract), what it compares against, and what the comparison leaves open; every characterization carries the pointer a reviewer needs to check it.

| Work | What it claims (own abstract) | Compact trace obtained by | Compared against | Task certified to need a trace? | Content verified? | Pointer |
| --- | --- | --- | --- | --- | --- | --- |
| [Abstract-CoT (IBM, 2604.22709)](https://arxiv.org/abs/2604.22709) | "Thinking Without Words": a reserved-vocabulary sequence replaces natural-language CoT; "up to 11.6× fewer reasoning tokens" at comparable performance; an emergent power law over the abstract vocabulary is read as "a learned abstract reasoning language" | 64 reserved tokens; bottlenecked SFT from verbal CoT + self-distillation + constrained-decoding RL | verbal CoT (SFT+RL), not length-optimized | No — the same model scores 85.8% on MATH-500 with no CoT vs 90.8% with abstract CoT | Weak — truncation to 32 abstract tokens costs \~6 pp; traces are near-periodic; a pause-token row exists but scores below no-CoT | Table 1, ablation table, App. examples |
| [ORION (2511.22891)](https://arxiv.org/abs/2511.22891) | Inspired by the Language of Thought Hypothesis, trains models to reason in compact "Mentalese"; with SLPO, "4–16× fewer tokens" on AIME, MATH-500, etc. | Authored symbolic shorthand, 40k SFT traces + length-rewarded RL (SLPO) | R1-distilled verbose reasoning traces | No (math benchmarks, unfiltered) | Not tested | Abstract; Pareto figure |
| [CLSR (2606.29354, ICML 2026)](https://arxiv.org/abs/2606.29354) | "When LLMs Develop Languages": agents "invent, evolve, and share" symbolic frameworks; "3–6×" fewer completion tokens than standard CoT; an information-theoretic bound relating accuracy and tokens under arbitrary symbolism | Frozen weights; prompted symbol systems + evolutionary selection + a router | standard CoT + compression-prompt baselines | No | Not tested; authors note notation stays partly readable | Abstract; Thm. 3.2 |
| [Little (2607.09786)](https://arxiv.org/abs/2607.09786) | Length-penalized RL cuts tokens with small accuracy loss but reduces CoT monitorability | Length-penalized RL, symmetric target, language unconstrained | Unpenalized RL | Yes — recall-solvable prompts filtered | N/A — trace stays English; removes verification/backtracking commentary | Method §; lexical-marker analysis |
| [Kaufmann et al. (2603.30036)](https://arxiv.org/abs/2603.30036) | Character-count penalties induce abbreviation; relation to monitorability | Character-count penalty on a synthetic task | Unpenalized | Yes | Abbreviations emerge; focus is monitorability | Results § |
| [Dense Sequential Chains (2605.07307)](https://arxiv.org/abs/2605.07307) | Removing alphabetic text from math CoT does not hurt; numerals carry it | Inference-time masking of existing traces | — | No | N/A — analysis, not training | Results § |

None of the three efficiency papers claims to have compared against a length-optimized linguistic policy on a trace-necessary task; the reading that their savings come from *leaving language* is in their framing (titles, "language of thought", "abstract reasoning language"), and it is that reading, not their numbers, that the present measurement bears on.

**Efficiency-oriented CoT compression.** Length-penalized RL (Arora & Zanette 2025; L1, Aggarwal & Welleck 2025; LC-R1; SWAP 2026) consistently removes about half of a reasoning trace at 1–2 pp cost. Little (2026) shows that what is removed is meta-commentary — verification, backtracking, exploration — while computation is preserved, and that the trace remains ordinary English at 25–38% of its original length. These methods stop well short of any floor: Little's penalty is symmetric around a target ratio and explicitly avoids minimum length.

**Non-linguistic and abstract CoT.** IBM's Abstract CoT, ORION, and CLSR each report large token reductions by moving reasoning out of natural language. As the table shows, none evaluates on reasoning-necessary tasks, none compares against a length-optimized linguistic trace, and only IBM tests whether the trace carries content (finding it largely survives truncation to 32 tokens, and qualitatively shows near-periodic sequences). A related inference-time analysis ([Rethinking Dense Sequential Chains, 2026](https://arxiv.org/abs/2605.07307)) finds that masking all alphabetic text from math CoT does not reduce accuracy, suggesting that in math the numerals already carry the reasoning.

**Latent reasoning.** Coconut (Hao et al. 2024) and recurrent-depth / looped transformers move reasoning into continuous hidden states, changing the architecture rather than the token vocabulary. Stepwise Internalization (Deng et al. 2024) removes CoT entirely by curriculum. These are complementary to the present question: we hold the architecture fixed and vary only what the discrete trace is allowed to contain.

**Monitorability.** The compression–monitorability trade-off (Little 2026; Kaufmann et al. 2026; Baker et al. 2025) and the reported behavior of GPT-6 Astra — shortening its CoT when a monitor is announced — make the density question a safety question: a trace that is shorter because it is denser and a trace that is shorter because reasoning moved into the forward pass look the same from outside.

## 3. Question, scope, and estimand

### 3.1 The question this study can answer

For the deductive content of one certified task: (D1) under length pressure and a letter ban, does a language-pretrained model reach a letter-free trace that solves the task, or does it fail (collapse, cipher, no recovery)? (D2) Is what it reaches load-bearing — problem-specific and carrying the computation — rather than filler plus extra forward passes; and is it the standard propositional notation or a private code? (D3) How much shorter is it than the *length-optimized* letter-permitted trace, and is the gap encoding or algorithm?

None of this bears on whether a model can invent a general code (Q1). A positive D1–D3 at 4B on one task is not evidence for Q1; a negative D2 on deductive content — the content type where language is most plausibly indispensable — would remove a necessary condition for Q1 without needing scale.

### 3.2 Estimand: the letter tax

`tax = (L_A − L_B) / L_A` at matched externalized accuracy, where L\_A is the length of the RL-length-optimized letter-permitted trace, L\_B the length of the RL-length-optimized letter-banned trace, both measured among correct traces, and externalized accuracy = accuracy − the same policy's direct-answer (no-trace) accuracy. Reported in tokens (primary), code points, and bits under a coder fit on the union of all arms' traces (robustness). A tax that appears in tokens but reverses in code points or bits is a tokenizer effect. A negative tax (letter-free longer by ≥ 10%) is reported as such.

Two pre-registered practical-significance tiers: **≥ 25%** is a detectable tax; **≥ 67%** is the tier consistent with the 3× claims in prior work. **< 10%** is no tax beyond tokenizer effects; 10–25% is reported as the bound "tax ≤ 25%". These are cutoffs, not powered tests.

We write *letter* tax, not *language* tax: the ban removes every token containing a letter in any script, leaving numerals, punctuation, symbols, and whitespace. A formal notation built from those is a notation; the disclosure is that a null letter tax is not evidence about a language tax in any broader sense.

### 3.3 Reasoning-necessary

A cell is admitted only if the frozen model's trace is certified as *content*-necessary (§4.2): direct-answer accuracy near chance, a forced-budget curve that is still rising, filler padding that does not help, a transplanted trace that fails, the letter ban binding, no known solver template, and native length far above the oracle floor. Post-training, each arm is re-checked for internalization; an arm whose direct accuracy has risen has its length results marked uninterpretable.

### 3.4 What is held fixed

All trained arms start from the same checkpoint and share reward, optimizer, hyperparameters, and stopping rule; each differs in exactly one setting. The manipulation gate may retune λ and G on arm A only, once, with the pre-tune curve reported; the chosen values apply to every arm.

### 3.5 Scope statement (verbatim in the paper)

All conclusions are restricted to Qwen3-4B with LoRA, disjunctive-precedence ordering at the admitted cell among (n, h, d) ∈ {(8,4,5), (8,4,6), (9,5,6), (9,5,7)} with decision depth ≥ 1, this symbolic prompt, this tokenizer, and this RL budget. A pre-registered outcome updates only the claim that, at matched accuracy on this task and model, a letter ban yields a shorter load-bearing trace than length-optimized letter-permitted RL. It does not estimate the compression of Abstract-CoT codebooks, ORION Mentalese, or CLSR routers, and it does not say whether their reported savings are or are not due to leaving language in their settings; it says only that their comparisons lack the baseline this measurement supplies.

## 4. Pre-registered protocol (v8)

### 4.1 Task: unique linear extension with disjunctive precedence

n events indexed 0..n−1. Constraints: hard precedence `i<j` (h of them); disjunctive precedence `(i<j)|(k<l)` (d of them), exactly one side of which is consistent with the unique solution σ. An instance is kept only if it has exactly one linear extension. Four cells are tried in the published order (n, h, d) = (8, 4, 5), (8, 4, 6), (9, 5, 6), (9, 5, 7); (10, 6, 7) was dropped because guided generation does not reach uniqueness there. Answer: the order as n digits, `<answer>3 0 7 1 …</answer>`, exact match, ≤ 32 tokens.

*Prompt is a symbolic IR shared by all arms*, with one format line and no examples:

```
n=8
hard: 0<3 2<5 5<1 4<7
disj: (1<6)|(6<2) (3<4)|(7<0) (2<7)|(5<3)
reason inside <think>, then give the order
answer: n digits, first to last
```

*Inference content*: propagation = transitive closure of hard and chosen precedences plus a unit rule (a disjunct contradicted by the closure forces the other; an implied disjunct is treated as decided); a case split = choosing a disjunct; refutation = a cycle in the closure. Minimum decision depth S is computed by DPLL over disjunctions with this propagator plus failed-disjunct probing defined as *single side, closure only* (add one side, compute the closure, no further unit propagation inside the probe). Instances are kept if S ≥ 1 under that definition; S is reported (73–95% of kept instances have S = 1, the rest S = 2). A property of the task, recorded: if unit propagation is also run inside each probe — one level of lookahead — every unique instance becomes S = 0. Every admitted instance is therefore solvable by one level of assume-propagate-refute without backtracking; "S ≥ 1" is relative to the fixed propagator above. d ∈ {5, 6, 7} so that enumerating the 2^d disjunct assignments is an order of magnitude longer than deduction; whether length pressure moves a policy from enumeration to deduction is observed (§4.5), not enforced.

*Known structural fact (disclosed)*: because the extension is unique, every covering edge of the answer appears as an explicit constraint in the prompt, so the answer is a Hamiltonian path through the prompt's edge set and the decision content is d bits. Writing the transitive closure is therefore bookkeeping; the floors below are defined on the decision content, not the closure.

*Generation is guided, and the guidance is constrained.* Uniqueness requires that every covering edge of σ not implied by the hard constraints appear as the consistent side of some disjunction, so random placement of disjunctions almost never yields a unique extension (< 1 in 10⁵ at the target cells). The generator therefore chooses the *consistent* side of each disjunction from σ's uncovered covering edges; the *inconsistent* side is drawn uniformly from all pairs that contradict σ, never selected for its effect on the search, so that it carries no learnable signal. Acceptance rates before and after guidance, the distribution of the inconsistent edges' span in σ, and whether the consistent edges equal exactly "covering edges minus hard-implied edges" are reported; all cell statistics and rejection rules are computed on the guided distribution that is actually used.

*Floors per instance*: the **decision floor** = d disjunct choices + n output digits; the **Kahn floor** = the sum of ready-set sizes along σ's ready-set trajectory (ready = no unplaced predecessor under the closure of hard ∪ decided edges, pruned by failed-element probing, single element, closure only); both reported at c ∈ {2, 2.5, 3} tokens per element. On the four cells the Kahn floor is 12–13 elements (≈ 30 tokens at c = 2.5) and the decision floor 13–16, so the minimal trace is very short and criterion 8 is not binding (measured on the admitted cell: median native length 8,600 tokens, naturally terminated traces 3,400–10,200, against a Kahn floor of 30 tokens — M is 287 × the Kahn floor, where criterion 8 asks for 5 ×); converged lengths are reported in absolute tokens next to both floors so that a percentage tax can be read against them. Headroom and the manipulation gate use the Kahn floor; the kill criterion uses the decision floor.

*Cell statistics and rejection rules* (computed on the guided distribution before any GPU-hour, per cell): median #LE(hard poset) ≥ 20; median position of the first ready set with ≥ 2 elements under closure propagation ≤ n/2 (an earlier rule on the fraction of singleton steps was withdrawn as inconsistent with S = 1 at n = 8; the fraction is reported); median maximum antichain of the hard poset ≥ 3; median number of Hamiltonian paths in the undirected *mentioned* graph (hard ∪ both disjuncts) ≥ 3; median number of disjunctions surviving probing ≥ 3; keep rate of the S ≥ 1 filter ≥ 1%; isomorphism types of the full structure (poset + disjunctions) in eval not a subset of train; S distribution and disjunct-redundancy fraction reported. All four cells pass (stopping split, n = 500): #LE(hard) medians 1,680–10,080; first non-singleton ready set at position 0; antichain 5; Hamiltonian paths 24–117; surviving disjunctions = d; redundancy 3–10%; guided acceptance 1.1–3.3% against 0/20,000 uniform; S ≥ 1 retention 31–42%. Chance for the direct answer = 1/#LE(hard poset). Train 2,000 / stopping-eval 500 / reporting-eval 500 / direct-check 2,000 from disjoint seeds; index relabeling and constraint order randomized per instance.

*Contamination*: this format is not a post-training staple; shortcut detectors (IR copy with trailing order; full-permutation enumeration; disjunct-assignment enumeration defined as systematic coverage of the remaining assignment cube — bitstrings, counters, Gray codes, nested try-both; guess-then-verify loops) are published before admission and run on frozen native traces and on every arm's converged correct traces. The English "suppose i before j" case split is the intended computation and is a strategy class, not a template.

*Instrument positive control*: one instance solved by hand as a DPLL trace in English and in the letter-free symbol set; both tokenized; ratio reported.

*Backup*: unique list-coloring with the same protocol and the same eight admission tests, used only if no ordering cell is admitted. Cup state tracking and multiplication (frozen only) are appendix material.

### 4.2 Admission (frozen Qwen3-4B, n = 500; all must pass; cells swept in a fixed published order)

1. Direct exact-match accuracy (`<think></think>` prefilled) ≤ 2 × chance. Per-pair accuracy vs a uniform random extension of the hard poset is reported as a diagnostic. Strict and lenient answer extraction both reported; admission uses strict.
2. Native-trace exact accuracy ∈ \[60%, 80%\] on the S ≥ 1 population.
3. Forced-budget curve at {0.1, 0.25, 0.5, 0.75, 1.0} × M: smallest budget within 5 pp of native ≥ 0.5 M.
4. Whitelist-unigram filler padding never within 10 pp of native at budgets ≥ 0.5 M, compared only at budget points where the native curve itself is > chance + 10 pp. *(Deviation D10; as registered: no restriction on the budget points. At cap 5,120 on (8,4,5) / (8,4,6) the native curve at 0.5 M was 5.4% / 2.2% and filler 0.2% / 0.0% — both ≈ 0, so "never within 10 pp" failed vacuously, independent of whether filler helps. Where native is clearly above chance the criterion is unchanged. On the admitted cell the compared points are 0.5 / 0.75 / 1.0 M: native 22.6 / 44.0 / 55.0% vs filler 0.2 / 0.2 / 0.4%.)*
5. Native trace transplanted i→j (prefilled, answer generated) lowers accuracy by at least half the gap to direct.
6. Frozen letter-banned accuracy ≤ direct + 10 pp.
7. Manually audited rate of template / answer-copying traces among native traces < 5%. *(Deviation D13; as registered: "shortcut detectors fire on < 5% of native traces". The four automatic detectors are demoted to descriptive quantities, reported under all four definitions — r4 / D9 / D12 / D13 = 92.2% / 25.8% / 13.2% / 6.0% on the admitted cell. Audit scope: the 66 traces flagged under D12 were all read, 0 template / answer-copying; the 434 unflagged traces were not read. The change was made after seeing frozen native traces and before any training; see `docs/deviations_log.md` D9, D12, D13 and `results/audit_criterion7_ord_n8_h4_d5.md`.)*
8. Median native length ≥ 5 × Kahn floor (c = 2.5).

Cell selection: native accuracy is measured (n = 500) on every candidate cell that passes the statistics; among cells inside the \[60, 80\]% window, the one with the largest Kahn floor is taken to the full eight-criterion admission, so that the tax is measured on the longest necessary trace the model can handle. The study stops if no cell passes and the backup task does not either. Note on scale: the Kahn floor at these cells is 12–16 elements (roughly 30–40 tokens) against native traces with a median of 8,600 tokens on the admitted cell (naturally terminated traces 3,400–10,200; 37.8% force-closed at the 10,240 cap) *(as registered: "native traces of 1,000–3,000 tokens" — an estimate written before any measurement on this task; the measured lengths are several times longer, which only widens the gap to the floor, so the argument is unaffected)*, so criterion 8 is not binding and the tax is measured on short traces; absolute lengths and both floors are reported alongside every ratio, and the estimand is read as the cost of connectives and scope markers, not of extended reasoning.

**Post-training vestigiality check**, per arm, every 50 steps on the 2,000-item direct-check split: the run stops and the arm is marked uninterpretable if (with-trace accuracy − direct accuracy) falls below half the frozen gap on two consecutive checks. For the E1 comparison, compared arms' trained direct accuracies must lie within 3 pp of each other at convergence.

### 4.3 Arms

| Arm | Trace constraint | Init | Seeds | Question it answers |
| --- | --- | --- | --- | --- |
| A | none | base | 3 | length-optimized letter-permitted comparator (E1) |
| A″ | letters reachable only as single-letter tokens (with/without leading space); multi-letter tokens banned | base | 2 | same instrument as B with letters reachable: constraint vs letters |
| B | every token containing a Unicode letter, mark, control character, or byte fragment set to −∞; audited whitelist (8,351 tokens) | base | 4 | the measured arm |
| C\_rand | letter ban; think-span context tokens drawn i.i.d. from B's converged unigram, length from B's converged distribution; loss on answer tokens only; run after B | base | 2 | content floor (E3) |
| B\_warm-SFT | A's converged traces transformed by a fixed control-word → symbol map (committed lexicon) with other letter tokens deleted; fresh adapter SFT; **no RL** | — | 1 per A seed | encoding-only measurement (E2) |
| B\_warm-RL | B\_warm-SFT + the same GRPO | B\_warm-SFT | 2, only if B seed 1's best accuracy at step 200 < A seed 1's best − 15 pp | usable vs findable from cold start |

All trained arms share model, LoRA placement (attention and MLP only — *deviation D5, the registered fallback branch of §5.5 test 3; as registered: "attention, MLP, embed\_tokens — lm\_head is tied and not adapted; the embed adapter's synchronization to the vLLM engine is verified in the smoke test". The check failed: with tied embeddings TRL's merge-and-sync writes the embedding delta into the shared tensor, so the sampler also carries it at the output head while the trainer's unmerged forward does not — sampler-vs-trainer |Δlogp| max / mean 0.965 / 0.186, against 0.208 / 0.031 for a q\_proj-only control. embed\_tokens LoRA was removed from every arm; the comparison between arms is unaffected, but B loses the embedding layer's trainable freedom and may be harder to cold-start*), reward, hyperparameters, and stopping rule. A seed fixes data order, sampling RNG, and LoRA initialization. Output grammar for all arms: `<think>` trace `</think>` `<answer>` digits `</answer>`; the prompt is prefilled with `<think>\n`; the mask, where present, applies from the first generated token to the first `</think>` (which is on the whitelist) and is lifted there; forced close at the cap appends `</think>\n\n<answer>`. A and A″ must pass the E3 transplant and resample collapse at convergence to serve as comparators; if A fails, the no-matched-accuracy paper (§5.3) applies.

### 4.4 Reward and optimization

`r = 10 · [ c · (1 − min(λ·L/M, 0.9)) − 0.5 · v · (1 − c) − 0.05 · v · c ]`; c = exact match, forced to 0 by any hard violation (non-whitespace between the first `</think>` and the final `<answer>`; any extra think tag); v = soft violation (missing/malformed answer, multiple distinct answers after `</think>`, span > 32 tokens); L = tokens from the first generated token to the final `<answer>`; M = median native length; λ = 0.5. Any clean correct completion outranks any incorrect one; escaping the mask scores −5.

GRPO, Dr. GRPO loss, no reward-std scaling, β = 0; 2 prompts × G = 16; LoRA r = 32, α = 64, dropout 0, lr 1e-5 after 10 warm-up steps; training T = 1.0 no top-k/p, evaluation T = 0.6, top-p 0.95, top-k 20; cap 10,240 *(deviation D8; as registered: 5,120, or 6,144 if native forced-close > 20%. At cap 5,120 the frozen model's native traces on (8,4,5) were force-closed 91.6% of the time and at 8,192 still 57%, so criterion 2 failed by truncation alone; admission was rerun in full at 10,240, which sets M = 8,599.5 and applies to every arm)*. Policy log-probs computed under the arm's own mask and renormalized; sampler and trainer asserted to agree within 1e-3. Stopping: two consecutive 50-step intervals with ΔL\_mean < 5% and Δacc < 4 pp on the stopping-eval split, or 400 steps. Manipulation gate: A seed 1 must reach ≥ 30% token reduction from the frozen model at ≤ 5 pp accuracy loss and remain ≥ 2 × the Kahn floor; else λ = 0.3, then λ = 1.0, then G = 32 on A in that order; the passing setting applies to every arm and the pre-tune curve is reported. *(Deviation D14, stage 1 only: the gate's thresholds are unchanged, but on a gate failure the matrix halts and reports to the user instead of entering the registered automatic retune sequence λ = 0.3 → λ = 1.0 → G = 32; the same halt applies to the kill criterion, any non-zero subprocess exit, a vestigiality trip, or a missing SFT source. Reason: the measured 494 s/step projects the full matrix to ≈ $571 against the $300 ceiling, so each retune is a spending decision that §5.1 reserves for a human. Whether the registered retune sequence is then followed is decided by the user at that point and recorded.)* Run order: A seed 1 (20-step cost measurement, then to convergence) → B seed 1 → warm decision at B1 step 200 → remaining seeds → C\_rand → diagnostics.

### 4.5 Endpoints and decision rules

**E1 — letter tax.** Per converged policy, the forced-budget curve at {0.1 … 1.0} × its converged length on the reporting-eval split, isotonic per seed; y = total accuracy (externalized accuracy reported alongside); x = tokens among correct traces (unconditional length alongside). Matched y = the highest y all of A, A″, B reach, minus 5 pp; tax = (L\_A − L\_B)/L\_A by interpolation. Primary interval: item-level paired bootstrap of correct-trace length at the matched point, seeds as equal-weighted blocks; seed-level exact Mann–Whitney as a sensitivity table. Units: tokens (primary), code points, and bits under zstd-19 with dictionaries trained on the union of all arms' correct think-spans and per arm; a tax is claimed only if all three units fall in the same tier. Tiers: ≥ 67% consistent with the magnitudes in prior work; ≥ 25% detectable; 10–25% reported as "≤ 25%"; < 10% no tax beyond tokenizer effects; ≤ −10% a cost. Also reported: frozen → A reduction; tax within A's modal strategy class (strategy class assigned per trace by the published detectors, with enumeration operationalized as explored branches > 3 × the DPLL minimum, and a manually audited sample giving detector precision/recall); tax against both floors; item-matched length on items every arm solves; A's letter fraction over training ("low" = < 30% at convergence). If any arm has no point within 5 pp of the matched y, E1 is not computed and the no-matched-accuracy paper (§5.3) is the registered result. Expectation, stated now: with these constants the tax will most likely land in the "≤ 25%" or tokenizer tier; that is the expected result, not a failure.

**E2 — encoding-only.** B\_warm-SFT is valid if its total accuracy is within 5 pp of its source A seed; its length ratio to the source is then the encoding-only tax, per A seed. Otherwise: "words were load-bearing or the map was lossy".

**E3 — content.** Required for B: (i) B > C\_rand at matched length (item-paired, seeds as blocks); (ii) transplant and unigram-resample of B's own traces each lower B's accuracy by at least half the gap to its trained direct accuracy; (iii) prompt-copy rate (IR-copied spans ≥ 4 tokens) < 30%. Decoder: L2 logistic regression over bag-of-n-grams (n ≤ 3) of **prefix-only** think-span segments with copied spans masked; labels are the **target instance's ground truth** — the answer-consistent disjunct choices and the true ready set / next event at the model's own step count, aligned by the published parser but never read from the trace; prefixes whose commitments the parser cannot align are excluded and the parse rate is reported per arm. Instrument bar: at least one arm's traces decodable at ≥ 90% with a ≥ 10 pp drop on that arm's transplanted and C\_rand traces (paired bootstrap over items excluding zero). For B, the decoder clause is required only if B's parse rate ≥ 50%; otherwise B is reported as not decodable and E3 rests on (i)–(iii). Parser-based soundness/completeness of each prefix against the closure of its own commitments is reported as trace validity, descriptive. Cipher statistic (Spearman of B's token frequencies vs A's word frequencies under the B\_warm map) reported, descriptive.

### 4.6 Readings (fixed)

| Outcome | Reading |
| --- | --- |
| tax ≥ 25% (all units, same tier), E3 passes | under a letter ban, RL produces a shorter load-bearing trace assembled from the supplied primitives; ≥ 67% additionally matches the magnitudes prior work reports |
| A″ ≈ B < A | the effect is the hard constraint, not letters |
| tax < 10%, A's letter fraction low | length optimization alone sheds letters on this task and model |
| tax < 10%, letter fraction high | words cost nothing at matched accuracy on this task and model |
| tax ≤ −10% | the ban costs length |
| 10 ≤ tax < 25% | reported as "≤ 25%" |
| arms differ in strategy class | the ban changed the algorithm; pooled tax primary, within-class tax may be undefined; reported under D3 |
| cold B fails, B\_warm-RL succeeds | usable but not findable from cold start at this budget |
| both fail | search-budget result |
| B > C\_rand but transplant/resample do not collapse | structured filler |
| E1 not computable | the no-matched-accuracy paper (§5.3) |

## 5. Feasibility, kill criteria, analysis plan, reproducibility

### 5.1 Compute and budget

Qwen3-4B, bf16 LoRA, one A100 80 GB (RunPod), vLLM sampling. Matrix: A × 3, A″ × 2, B × 4, C\_rand × 2 = 11 RL runs, plus B\_warm-SFT (minutes) and, conditionally, B\_warm-RL × 2. Budget ceiling **$300**; the scope is frozen at this matrix and does not grow with budget.

**Cost is measured, not assumed.** A seed 1 runs 20 steps first; wall-clock, tokens per step, and memory are recorded and the full matrix is projected. Decision points, in order:

1. Projected total ≤ $300 → proceed.
2. Projected total > $300 → **stop and decide** (add budget; seek institutional compute; defer B\_warm-RL). No seed and no arm is cut automatically — cutting seeds trades conclusion quality for money and is a human decision.
3. After A seeds 1–3 converge: if the seed-to-seed coefficient of variation of converged length exceeds 15%, stop and decide whether B needs 6 seeds.

Only items that do not affect any reading may be trimmed automatically (eval frequency, archive granularity).

### 5.2 Order of execution

Environment and unit tests (§5.5) → whitelist audit → generator cell statistics and rejection rules (CPU) → admission on the first surviving (n, h, d) cell in the published order → floors and M → instrument positive control → A seed 1: 20-step cost measurement, then to convergence → manipulation gate → B\_warm-SFT from A seed 1 (E2, early read) → B seed 1 → warm decision at B1 step 200 → B seeds 2–4 → A seeds 2–3 → A″ × 2 → C\_rand × 2 (needs B's converged distributions) → B\_warm-RL × 2 only if triggered → diagnostics on every converged checkpoint → analysis.

### 5.3 Kill criteria and minimum publishable version

The study stops if: no cell passes admission at N ∈ {10, 12}; the manipulation gate fails after both λ retunes and the G retune; A seed 1 converges within 1.5 × the oracle floor (no headroom for a tax to appear); or embeddings/lm\_head LoRA cannot be served and their removal makes the whitelist unreachable in the smoke test (B never emits > 5% non-prompt symbols in 20 steps).

Minimum publishable version if killed after the gate: the admission procedure and its tables (including rejected cells), the frozen → A reduction, the instrument positive control, and B\_warm-SFT (E2) — as a short note. If killed before the gate: the admission procedure and a negative task-generation note.

### 5.4 Analysis plan (fixed before data)

- **Figure 1**: forced-budget curves — one overlay panel with A, A″, B, C\_rand and the frozen model on a shared x-axis (tokens among correct traces), seeds overlaid, oracle floor as a vertical line, matched-accuracy line drawn; small side panel: A's letter fraction over training.
- **Table 1**: the letter tax per seed in tokens, code points, and bits; range; Mann–Whitney; the frozen → A reduction; tax within A's modal strategy class; item-matched secondary length; E2 per A seed.
- **Table 2**: E3 four-way table with each criterion's value and decision; cipher check; copy rate.
- **Figure 2**: one B trace at three training checkpoints with the decoder's reading of its intermediate assignments.
- **Appendix**: admission tables for every cell tried; strategy-class counts per arm; reachable and emitted operator sets; detector rules verbatim; cup and multiplication pipeline checks; pre-tune A curve if the gate retuned; direct-accuracy trajectory per arm; B\_warm-RL curves if run.

### 5.5 Implementation tests (all pass before pre-registration is posted)

1. Mask: 10,000 sampled B think-tokens contain zero banned ids; the renormalized masked softmax sums to 1 over the whitelist.
2. Log-probs: sampler (vLLM engine-level processor) and trainer (masked, renormalized) agree within 1e-3 on 1,000 sequences; banned rows of lm\_head receive zero gradient from trace positions.
3. Embedding/lm\_head LoRA: weights synchronize to the vLLM engine (smoke test with a planted change); else these modules are removed from LoRA for all arms and the change is logged.
4. C\_rand: on a toy task where a planted trace token is predictive, the policy's think-span tokens never enter the context, the context tokens' unigram matches B's converged unigram (χ² p > 0.05), and loss on trace positions is exactly zero.
5. Decoder: instance-level train/held-out split with no trace overlap; ≥ 90% on A's traces before use.
6. Generator: uniqueness and S verified by brute force on 10,000 instances; permutation invariance of the verifier; canonical-form de-duplication; disjointness of train / stopping-eval / reporting-eval asserted.
7. Reward: exhaustive ordering test over c, v, hard/soft, and L.
8. B\_warm transform: applied to 100 A traces, the strategy-class distribution matches A's; the full lexicon and symbol map are committed.

### 5.6 Reproducibility and reporting

The generator, detectors, whitelist and its audit, B\_warm map, configs, seeds, the four cell datasets and their reports, and this protocol are committed to a public repository before the first admission run. **Freeze commit: `d207c263df938f6003e51ff3da90f13367b40d99` ("v8 pre-registration freeze (r4)", 2026-09-19); follow-up `471f0561` records the hash in the README. Earlier freezes r1–r3 (`eceff232`, `288c8745`, `fd5c6566`) are superseded and listed in §9. Repository: https://github.com/ZechenWei7/letter-tax. Pre-registration: OSF Registries, https://osf.io/usycb, registered 2026-09-19 02:23:34 (accepted), citing r4; the r4 README is archived as a supplementary file there.** No GPU-hour was spent before that timestamp. Deviations are logged with commit hashes. The full rollout archive (every training rollout and every eval output, compressed) is released. Every negative reading has a report template written in advance. Code under MIT; data under CC-BY-4.0.

## 6. Results

### 6.1 Admission

*Filled from `results/cloud_log.md` and `results/admission_ord_n8_h4_d5.json`; still TBD: prompt-format check, instrument positive control ratio for the ordering task, floors at c = 2 and 3, canonical-form counts.*

**Admitted cell: (n, h, d) = (8, 4, 5), cap 10,240 (D8), M = 8,599.5.** Frozen Qwen3-4B, stopping split n = 500, direct-check n = 2,000.

| # | Criterion | Value | Threshold | Pass |
| --- | --- | --- | --- | --- |
| 1 | direct exact (strict = lenient, 0 format errors) | 0.45% | ≤ 2 × 2^−5 = 6.25% | ✓ |
| 2 | native exact | 60.8% | [60, 80]% | ✓ |
| 3 | forced-budget curve at 0.1/0.25/0.5/0.75/1.0 × M | 0.4 / 2.6 / 22.6 / 44.0 / 55.0% | never reaches native − 5 pp = 55.8% → smallest budget > 1.0 M ≥ 0.5 M | ✓ |
| 4 | filler padding (compared only where native > chance + 10 pp, D10) | 0.4 / 0 / 0.2 / 0.2 / 0.4% | never within 10 pp of native at ≥ 0.5 M | ✓ |
| 5 | transplant (derangement) | 0.0% (drop 60.8 pp) | drop ≥ 30.2 pp | ✓ |
| 6 | frozen letter-ban | 0.6% | ≤ direct + 10 pp | ✓ |
| 7 | audited template / answer-copy rate (D13) | 0 / 500 (66 flagged traces read; 434 unflagged not read) | < 5% | ✓ |
| 8 | M / Kahn floor (30 tokens at c = 2.5) | 286.7× (decision floor 32.5 tokens → 264.6×) | ≥ 5× | ✓ |

Native-trace detail: natural termination 62.2% (311/500), of which 89.1% correct (length min / median / max 3,423 / 6,863 / 10,187); force-closed 37.8%, of which 14.3% correct; overall length p10 / p25 / p50 / p75 / p90 = 5,115 / 6,366 / 8,588 / 10,240 / 10,240. Per-pair direct accuracy 0.612 against a 0.666 uniform-extension baseline. Strategy class (descriptive): 100% mixed_probe_enum. Automatic shortcut detectors (descriptive, four definitions r4 / D9 / D12 / D13): 92.2% / 25.8% / 13.2% / 6.0%.

**Path to admission (all disclosed as deviations).** At the registered cap 5,120 the cell failed criteria 2, 4, 7: native 32.4% with 91.6% force-closed (the 42 naturally terminated traces were all correct); (8,4,6) likewise (native 22.6%, 94.2% force-closed), and (9,5,6)/(9,5,7) were not run. A native-only retest of (8,4,5) at cap 8,192 (n = 200) still force-closed 57%, with 97.7% accuracy among natural terminations. D6 fixed a direct-answer token limit (16 → 32) that had truncated every direct answer; D8 raised the cap to 10,240; D10 removed a vacuous failure of criterion 4; D9/D12/D13 replaced criterion 7's automatic detectors, which on reading the flagged traces were measuring re-verification and restatement rather than templates, with a manual audit. Every one of these changes was made after seeing frozen native traces and before any training.

### 6.2 Cost measurement and manipulation gate

**Cost (measured, A `--dry-run` 20 steps, cap 10,240, provisional M = 8,192).** 494 s/step (generation 218 s + training 276 s), L_mean 8,106, ≈ 259k tokens/step, peak 74.4 GiB allocated. Projection at $1.6/h: 200 steps = 27.4 h/run; 11 runs ≈ 302 GPU·h ≈ $483 training + ≈ $88 eval ≈ **$571 > $300 ceiling**. vLLM memory 0.6 + sleep mode (D11) gave ≈ 4% and was not adopted. Per §5.1 decision point 2 this was a stop-and-decide: the decision (D14) was to run only A1 → B_warm-SFT(A1) → B1 → warm decision, halting on any gate failure instead of auto-retuning, then decide again. Stage-1 worst case (A1 and B1 each to 400 steps) ≈ $200.

**Manipulation gate.** TBD — frozen → A reduction; gate / kill verdict.

### 6.3 E1 — letter tax

TBD — Figure 1, Table 1.

### 6.4 E2 — encoding-only

TBD — B\_warm-SFT per A seed.

### 6.5 E3 — content

TBD — Table 2, Figure 2, cipher check.

### 6.6 What B wrote

TBD — descriptive: notation, operators emitted, strategy classes, decoder readings, evolution over training from the archive.

## 7. Discussion

### 7.1 What the measurement does and does not settle

A, A″, and B share everything but the reachable vocabulary, so a difference among them is attributable to what the mask makes reachable; B\_warm-SFT holds the algorithm fixed by construction and measures encoding alone; C\_rand fixes alphabet and depth and removes content. The measurement does not settle whether a letter-free code would arise without a mask, whether findings at 4B/LoRA persist at scale, or anything about content that cannot be scored. A positive tax is an RL-dynamics finding on one task whose prompt already supplies propositional primitives; a null tax on this task is not evidence about a broader language tax.

### 7.2 Fairness to prior work

Abstract-CoT and ORION do post-train or length-train a verbal or symbolic baseline; our objection is not that they lacked any optimized verbal arm but that the baseline is not a letter-permitted, length-optimized policy on a task certified to need a trace, and that no filler-matched trained control is reported. CLSR is a test-time multi-agent method and is cited for its claim, not as an RL comparison. Dense Sequential Chains is an inference-time analysis of existing traces. We do not reproduce or evaluate any of these methods and make no claim about their numbers in their settings (§3.5). Each factual characterization in the related-work table carries a pointer to the table or ablation in the cited paper it rests on.

### 7.3 Relation to the compression–monitorability trade-off

If the tax is large and E3 passes, compressed traces in deployed models may be denser rather than emptier, and monitoring would need to decode rather than read. If the tax is small, compression is removal of narration and what remains stays readable; the monitorability loss Little (2026) reports would then be a loss of narrated intent rather than of computation.

### 7.4 Threats to validity

- **Single task with supplied primitives.** The IR hands the model event indices, `<`, and `|`; B assembles rather than invents. Disclosed; the word "discovery" is not used.
- **Tokenizer.** Qwen3 tokenizes digits singly, so the tokens unit is structurally biased against B; code points and bits are the robustness units and this is stated.
- **Exploration.** Cold-start B may fail for search reasons; B\_warm-RL and the hit-rate logs separate that from non-existence, and the reading is worded as a search-budget result.
- **Algorithm change.** RL can change the strategy as well as the encoding; the tax is also reported within A's modal strategy class, and B\_warm-SFT is the algorithm-fixed measurement.
- **Power.** With 3–4 seeds only complete separation is claimable; every seed is reported; the 25% and 67% tiers are practical-significance cutoffs, not powered tests.
- **Contamination and the prompt leak.** The ordering format is not a post-training staple, and shortcut detectors are applied; but every covering edge of the answer appears in the prompt by construction, so a policy can in principle glue prompt edges into a path without a case split. The cell-rejection rules (Hamiltonian-path count, antichain width, effective disjunction count) bound this; the strategy classifier reports whether it happens.
- **Mask leakage.** Enclosed Latin, regional indicators, and leet strings are audited out; a numeral–punctuation cipher a human could still pronounce is not excluded, hence the cipher check.

## 8. Limitations and future work

- **Scale.** 1.7B and 4B parameters. A positive result motivates, but does not establish, the same at frontier scale; a negative result is weaker still (§7.2).
- **Cold start.** Pure RL under a letter-free mask may not recover. IBM (2026) report cold-start failure for abstract tokens and use bottlenecked SFT with self-distillation as warm-up. We implement the same warm-up, with arm A's traces as teacher, and report whether it was needed. "Emergence without warm-up" is itself a result we record.
- **Architecture held fixed.** We vary only the trace alphabet. Recurrent-depth or continuous-thought architectures (Coconut; GPT-6 Astra's reported recurrence) change the compute-per-token trade-off and are the natural next comparison: whether a discrete non-linguistic trace or a latent loop is the more efficient carrier of the same serial computation.
- **Multi-agent convergence.** The question the title asks about *inventing* a language is, in its strong form, a question about a shared protocol between agents. A single-model trace is a private code; testing whether two independently trained arm-B models converge on compatible codes is future work.
- **Compute.** Local experiments are limited to \~200 GRPO steps at 4–8 generations. The cloud stage is budgeted at roughly 4–6 full runs; a λ sweep with multiple seeds per arm would require institutional compute.

* **Task-specificity of the code.** A single-task run can at best produce a task-shaped notation. The question with deployment significance is whether codes learned under the same constraint on structurally different tasks share structure — delimiters, counters, state layout — and whether a code learned on one task survives transfer to another. Multi-task RL under the mask, with cross-task probes, is the natural next study; it would also be the first place to look for a genuine "language of thought" as opposed to a compression scheme.
* **Decoding the code.** Prior work that produces unreadable traces (IBM 2026; Coconut) characterizes them statistically or by projection, not by translation. Because every intermediate state in the cup task is known, a supervised translator from trace segments to state can be trained and evaluated; its success or failure is a direct measure of how the trace stores information, and a fallback for the counterfactual-edit diagnostic when the notation does not parse.

## 9. Experiment log (authors' record)

Design decisions and the observations that forced them, newest first. Entries are meant to be appended by whoever runs the next experiment.

**2026-09-19 (planning-led session; text aligned to the running protocol).** The project moved to a single lead session; the execution session holds the pod until the stage-1 chain ends, after which the lead takes over all operations (time recorded in `ops/state.md`). Three places where this document had drifted from the running protocol were corrected, each marked inline as a deviation with the registered wording kept next to it: §4.4 cap 5,120 → 10,240 (D8); §4.2 criterion 7, automatic detectors → manual audit (D13); §7.4 "supplied primitives" reworded from the K&K operators to the ordering IR (event indices, `<`, `|`). §6.1 and the cost half of §6.2 were filled from `results/cloud_log.md` and the admission json. Standing rule adopted for any future change to a decision criterion: check it against the registered page (https://osf.io/usycb) first — what the page fixes can only change as a logged deviation reported in the paper; what it does not fix (eval frequency, archive granularity, run order) may be adjusted but is still logged. Budget watch: stage-1 worst case (A1 and B1 each to the 400-step cap at 494 s/step, $1.6/h) ≈ $200; spend is re-projected when the first evals land and the run stops for a user decision if the projection exceeds $200.

**2026-09-20 (A1 first-step crash: root cause, fix, relaunch).** The crash (`CUDA error: invalid argument` in the first training step's backward) was chased through two wrong conclusions before the right one, all three recorded in `results/cloud_log.md`. (1) "Host / driver": the crash appeared after a pod migration to a newer driver, but a backward ladder on that host — pure torch, SDPA at 10.7k tokens, Qwen3-4B + LoRA + checkpointing with and without padding masks, and with a colocated vLLM engine resident — passed 11/11, and the training stack itself then passed on the same host. (2) "Not the memory split": four first-step launches at a lower vLLM memory fraction all passed, but so did the old fraction — every one of those launches had a warm compile cache and the same deterministic first batch, so they tested nothing. What separated crashes from passes was whether vLLM's torch.compile / inductor / triton caches were cold; they live on the container disk, which is wiped at every pod stop, so the first process after every boot compiled from scratch. The 2 × 2 on this host: fully cold compile × vLLM fraction 0.45 crashed 3/3; fully cold × 0.40 passed; warm × 0.45 passed 3/3; warm × 0.40 passed 6/6. A C++ stack from the reproduced crash located it: `LogsumexpBackward` allocates a ~6 GiB tensor, the caching allocator is at the limit (training-side peak 78.4 of 79.25 GiB reserved), and its release-cached-blocks-and-retry path fails. The user's memory-pressure hypothesis was right from the start; the cold compile only supplied the last few hundred MiB. Fix (D20, engineering — the memory split and cache location are not in the registered protocol; batch 2 × 16, cap, reward, LoRA, eval split / n / temperature / frequency untouched, step-0 eval kept): vLLM fraction 0.45 → 0.35 (0.40 was validated first; the user chose 0.35 because a ~2 GiB margin is thin for a 400-step, 27-hour run and a `--strict` crash needs a manual restart; 0.35 leaves ~6.4 GiB at no measurable step-time cost: ≈ 503 s vs 494 s), compile caches moved to the persistent volume, and a throw-away one-step warm-up before the chain. Acceptance, fixed before it was run: reproduce the crash under a cold cache, then three independent fully cold launches with the fix, one with a small pre-training eval — 3/3 passed. One side-observation worth keeping: freshly compiled kernels change sampling numerically, so the same seed does not give the same first batch across cold and warm starts; the sampling distribution is unaffected. The stage-1 chain was relaunched at 05:33 UTC from a clean A1 directory, the crashed run kept aside, the gate anchored on the rerun's own step-0 eval per the rule fixed on 09-19. Budget note: the $200 stage-1 line had been computed without the per-50-step evals (55 min each); with them, both RL runs going to the 400-step cap would cost ≈ $220–225 (< $300); to be re-projected from measured numbers when the first evals land. The user then moved the internal stage-1 line from $200 to $240: the $200 figure was an internal line drawn from the projection that omitted evals, not a protocol number; the true stage-1 upper bound sits inside the registered $300 ceiling, which is unchanged; and these two runs are what the continue / stop decision rests on. The run stops for a decision only if the re-projection exceeds $240. Lesson, same as the detector episodes: both wrong conclusions came from reading a result as a test of a hypothesis it could not discriminate; what resolved it was reading the raw logs (compile times, batch identity) rather than the pass / fail counts.

**2026-09-19 23:50 UTC (takeover; A1 crash; anchor rule).** The execution session's stage-1 chain ended when A1 crashed: the step-0 eval completed (acc 0.624, L\_mean 8,219, L\_median 8,586.5, force-closed 39.4%, direct 0.4% — consistent with admission: 0.608 / M 8,599.5 / 37.8%), then the first training step's backward raised `CUDA error: invalid argument` (not an OOM); `--strict` halted the chain and stopped the pod, with no retry and no setting changed. Known differences from the 20-step dry-run that had passed: a new host after a pod migration (driver 595.91 vs 580.159), and an eval before the first training step. The lead session took over all operations at this point (D17). Decisions taken by the user before any rerun number exists: (i) the crashed run's directory is kept aside, the rerun starts from a clean directory, the manipulation gate is anchored on the rerun's own step-0 eval, and both step-0 evals are reported; (ii) a fix confined to the vLLM sleep / colocate eval↔train hand-off, drivers, or library versions is engineering and is only logged, whereas a fix that skips the step-0 eval, changes the eval n, or changes the eval split / temperature / frequency touches the gate anchor and the stopping rule's data source and needs a user decision first; (iii) budget lines are judged at $1.6/h (pod price $1.59/h; the config's stale 1.9 replaced, D18), with the stage-1 line at $200 and the ceiling at $300 unchanged.

**2026-09-19 (text alignment, second pass).** Four further places brought in line with the running protocol, same method (deviation marked inline, registered wording kept): §4.3 LoRA placement without embed\_tokens (D5); §4.2 criterion 4 restricted to budget points where native > chance + 10 pp (D10); §4.4 gate failure halts for a user decision in stage 1 instead of auto-retuning (D14); §4.1 / §4.2 native-length figures replaced by the measured ones (median 8,600; natural terminations 3,400–10,200; the registered "1,000–3,000" was a pre-measurement estimate) with a note that "criterion 8 is not binding" is unaffected — M is 287 × the Kahn floor against a required 5 ×. Correction made in the same pass: the §6.1 row for criterion 8 had copied the log's label "M / (5 × Kahn)" for the value 286.7×, which is M / Kahn (8,599.5 / 30); the row now says so, threshold ≥ 5 ×. No decision criterion changed in either pass; documentation edits are held uncommitted until the stage-1 chain ends and the lead session takes over.

**2026-09-19 (post-registration decisions before training, D5–D14; details and numbers in `docs/deviations_log.md`).** D5: embed_tokens LoRA removed from all arms — under tied embeddings the sampler saw the embedding delta at the output head and the trainer did not (sampler-vs-trainer mean |Δlogp| 0.186 vs 0.031 for the q-only control); this is the registered fallback branch. D6: direct-answer token limit 16 → 32, because every direct answer had been truncated before `</answer>` and direct accuracy was reading 0 for format reasons. D8: cap 5,120 → 10,240 after 91.6% of native traces were force-closed at 5,120 and 57% at 8,192; admission rerun in full. D10: criterion 4 compared only where native > chance + 10 pp (removes a vacuous failure at 0.5 M where both curves ≈ 0). D9 → D12 → D13: criterion 7's detectors were narrowed twice and still fired on 13.2%; reading all 66 flagged traces found 0 templates (50 ordinary reasoning after restating constraints, 16 candidate-shuffling after failed reasoning), with triggers such as the event inventory "0,1,…,7" counted as a guessed permutation — so criterion 7 became a manual audit and the detectors descriptive. Third time in this project that stopping to read raw traces showed the detector, not the data, was broken. D14: measured 494 s/step projects the 11-run matrix to ≈ $571 against the $300 ceiling; per §5.1 no seed was cut automatically — the user chose to run A1 → B_warm-SFT(A1) → B1 → warm decision only, with `--strict` halting on any gate failure instead of auto-retuning λ / G, and to decide again afterwards.

**2026-09-17 (post-calibration).** Cell selected: cups\_n8\_k20 at cap 5120 (native 74.8%, forced-close 12%, M = 3278 from n = 1,000); cap raised from the 8 GB-era 3072 after cups\_n8\_k16 was found to pass the window only because the cap truncated it (84.4% at cap 4096). Multiplication training runs dropped: the frozen model under the mask solves mul\_4x3/4x4 at 69%/57%, so the mask does not bind. Arm D (answer-only RL) dropped after discussion: its result would not change any reading — internalizability is a task–model fit question independent of what the trace encodes — and the risk it guarded against (a trained arm whose trace has become vestigial) is detected directly by the per-arm post-training direct-accuracy check. Lesson recorded: every control must name the reading its outcome would change; D had been accepted through three review rounds without that test.

**2026-09-17 (task switch and rounds 4–6).** A 35B pilot on cups showed the model writes a symbol table unprompted and solves the query by back-tracing one cup; the full-state variant fixed the shortcut but left the deeper problem — state content has a one-symbol-per-cup encoding, so symbols win by construction and the result would say nothing about inference. Main task switched to sparse Knights-and-Knaves (deductive content, no numeral fallback, enumeration infeasible by N, case-split depth ≥ 2 enforced); cups moved to the appendix. Round 4 (task switch): truth-table premise was wrong (bitmask enumeration is short) → enumeration made infeasible by N and propositional notation declared the intended result, not a loophole; English prompt is a translation tax on B → symbolic IR for all arms; rejection-sampled generators bias toward propagation-only instances → constructive generation with a split-depth floor. Round 5 (worth-it and redundancy): three of four reviewers judged the "discovery" framing an artifact of the budget and the fair-measurement framing the real contribution; estimand renamed the *letter tax*; B\_eng, B\_seq, β = 0.01 seed, OOD N = 12, arm-adapted coder, permutation test, and three overlapping diagnostics cut; B\_warm-SFT promoted to a primary encoding-only measurement; LoRA extended to embeddings/lm\_head so rare symbols can be learned. Round 6 (pre-registration audit, twelve dimensions): all four verdicts "run after fixes"; fixes were specification, not design — output grammar stated, E2 rule, cold-B threshold, decoder labels and controls, two-tier thresholds, seed-level statistics, disjoint stopping/reporting evals, cost measured before the matrix, budget ceiling $300 with stop-and-decide rather than automatic seed cuts, scope statement verbatim. Lesson kept from round 3: every element must name the reading its outcome would change.

**2026-09-19 (final review and freeze r4).** A last four-reviewer pass on the frozen text found no structural objection; all verdicts were "ready after exact edits". Edits made: the post-training vestigiality check re-based so it cannot fire on an arm whose trace has not yet started contributing (active once with-trace ≥ frozen direct + 10 pp; stop if direct ≥ with-trace − (with-trace − frozen direct)/2 twice; C\_rand exempt); admission 1 re-based to 2 × 2^−d on the 2,000-item direct split; kill and gate floors stated in tokens at c = 2.5; matched accuracy defined over seed means; C\_rand seed i bound to B seed i with one length draw per prompt group; step-400 designated checkpoint; strategy classes made descriptive and the class-shift reading withdrawn (the classifier is letter-dependent); tiers written as intervals; B\_warm-SFT scheduled per A seed; transplant pairing by random derangement in both admission and diagnostics; M = median of the 500 admission traces; budget ceiling coded. Four freeze commits were needed because implementation facts pulled from the code exposed drift from the text (audit ban list missing from the tree; gate on the wrong floor; a stale comment; a stale docstring); each was fixed before the hash was cited.

**2026-09-18 (rounds 7–8, second task switch).** The K&K generator showed that random instances are 99.9% propagation-only under unit propagation + failed-literal probing, so the usable population was a rejection-sampled sliver and depth-2 instances existed only by stitching two depth-1 groups — a structured sub-population, not K&K. An independent task search (four reviewers) converged on disjunctive-precedence ordering with a unique linear extension: the case split comes from explicit disjunctions in the problem (a density knob, not a filter on solver behaviour), refutation is a cycle, contamination is low. Round 7 exposed a structural fact of that task — covering edges of the answer are in the prompt, the decision content is d bits, and closure-as-pairs is bookkeeping — which was disclosed rather than patched: floors re-based to the decision trace and the Kahn ready-set trace; cell-rejection statistics added (#LE(hard), antichain width, Hamiltonian paths in the mentioned graph, effective disjunctions, isomorphism overlap); enumeration of the 2^d assignments treated as a strategy class that the length penalty is expected, not assumed, to drive out; templates gate only shortcuts. Round 8: decoder made non-circular (target-instance labels, prefix-only features, unparseable prefixes excluded); the post-training direct check rewritten as a vestigiality gap test on 2,000 items with two consecutive trips; the kill re-based to the decision floor; E1 matched on total accuracy with item-level paired bootstrap; the no-matched-accuracy paper registered as a first-class outcome; expectation of a "≤ 25%" or tokenizer-tier result stated up front. Scope frozen.

**Design review rounds (2026-09-16).** Three rounds of adversarial review by four independent models (Claude, Grok, Gemini, DeepSeek), each given only the design document. Changes adopted, by round:

- *Round 1:* reward inverted correctness ordering at L ≥ M/λ (correct-long < wrong-short) and the group-pass gate was a discontinuity → penalty capped, gate removed. 98% native accuracy meant arm A would only delete post-answer verification, the very confound under test → admission band 60–80% and forced-budget curve requirement. Arm C made mandatory; inference-only diagnostics (trained-policy direct accuracy, transplant, unigram resample) added. Framing toward prior work scoped to a mechanism test; no claim about their numbers.
- *Round 2:* KL against an unmasked reference is undefined for the masked arm → renormalize under the mask, then β = 0 by default. Correct-but-malformed could score below wrong-clean → reward rewritten so any correct outranks any incorrect. Admission criterion "≥ 4× answer length" was vacuous → budgets as fractions of M; filler and transplant tests added. Single-token filler control replaced by same-alphabet C\_rand. Pareto front from training snapshots invalid → converged checkpoints and eval-time budget forcing. One λ, seeds reallocated. Bytes replaced by code points; arm-adapted coder replaces base-LM nats.
- *Round 3:* matched accuracy as pooled max − 5 pp could be undefined or tautological → intersection of attainable accuracies. Prompted arm A′ was a soft intervention against B's hard one → replaced by A″ (same mask instrument, letters reachable). The stated question was sequential but the matrix parallel → B\_seq added. C\_rand rollout semantics fixed (context tokens from B's unigram, loss on answer only, run after B). H1 reframed from "denser" to "changes what RL discovers"; cups cannot show density by construction. Answer-tag tokens removed from the think-phase whitelist; leet and enclosed-Latin audit; prompt-copy rate as an H2 gate.

Design is frozen at v4 pending the first cloud run's cost estimate.

| Date | Decision | Reason |
| --- | --- | --- |
| 2026-09-16 | Test 8 generations per step before restarting arm A λ=0.7 | GPU at 60% CUDA and 3.2/8 GB during λ=0.3; samples per step, not step time, was the bottleneck |
| 2026-09-16 | Arm A λ=0.3 terminated at step 150; relaunched at λ=0.7, lr 3e-5, eval every 50 steps | Six eval points showed L median 1820→1742, within noise; in-group reward spread ≤0.30 too small at 4 sequences/step |
| 2026-09-16 | Vocabulary mask changed from top-N frequency to letter-free (Unicode L\*/M\*/control/U+FFFD banned) | Frequency mask evaded in one step by space-dropping ("needto", "multipliedby") and code-switching ("break它down"); byte fragments could reconstruct letters |
| 2026-09-16 | Format rule: only whitespace allowed between `</think>` and final `<answer>` | Mask lifts at `</think>`; without the rule, arm B could reason in language in the unmasked gap |
| 2026-09-16 | L defined as all tokens up to the final `<answer>`, not the `<think>` span | Prevents reasoning from migrating past `</think>` to escape the length penalty; Qwen3 rehearses `<answer>` inside its think block, so "unique answer" means unique value |
| 2026-09-16 | Training baseline changed from forced-budget (@1024) to native trace with 3072 cap | Under uniform forcing, L is constant within every GRPO group and the length penalty cancels in the group-relative advantage; gradient was zero |
| 2026-09-16 | Forced close changed from `</think>` to `</think>\n\n<answer>` | With `</think>` alone Qwen3 continued computing in the answer region; forced-@1536 accuracy went from 6/9 to 9/9 |
| 2026-09-16 | Cup tracking moved to cloud (4B); mul\_3x3 chosen for local | Qwen3-1.7B native trace on cups ≈250 tokens/step, all samples cap at 3072; forced accuracy at chance |
| 2026-09-16 | `move(i→j)` operation removed from cup task; only swap/flip remain | \~700 native tokens per move step; shifting semantics ambiguous |
| 2026-09-16 | Arm A run before arm B locally | Arm A validates that the length gradient works at all and supplies teacher traces for a possible warm-up; on mul, A and B differ little mechanistically |
| 2026-09-15 | Research question reframed from "is non-linguistic CoT more efficient" to "denser, or just shorter?" | IBM Abstract-CoT, ORION, CLSR all evaluate on tasks solvable without CoT against uncompressed baselines; Little and Kaufmann compress on reasoning-necessary tasks but never leave language |
| 2026-09-15 | Readability-constrained control arm dropped | Readability is not well-defined as a training constraint; the frozen baseline and length-optimized arm A are the right references |
| 2026-09-15 | Content diagnostics (truncation, permutation, probes) made mandatory; filler arm C optional | IBM's own ablation (32-token truncation costs \~6 pp; no-CoT SFT at 85.8%) leaves open whether abstract tokens carry content |
| 2026-09-15 | Local model Qwen3-1.7B 4-bit, cloud Qwen3-4B (not Qwen3.5) | Unsloth advises against 4-bit training on Qwen3.5's hybrid architecture; Qwen3-4B matches Little (2026) for calibration |
| 2026-09-15 | Synthetic tasks (mul, cups) instead of GSM8K | Contamination, saturation at 2B, and no control over whether reasoning is necessary |

**Open items**

- [ ] Arm A λ=0.7 (and 8-generation variant) results
- [ ] Arm B letter-free mask: untrained accuracy, then 200-step run
- [ ] Decide whether arm B needs bottlenecked-SFT warm-up
- [ ] Cloud: 4B on cups and mul, λ sweep, n=300 eval, content diagnostics
- [ ] Read Kaufmann et al. (2603.30036) and Rethinking Dense Sequential Chains (2605.07307) in full and update §2
- [ ] Compute access: cloud budget first; Purdue RCAC standby or ACCESS Explore if results are inconclusive
