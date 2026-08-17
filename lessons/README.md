# Twelve pitfalls in empirical multi-armed bandit evaluation

*Found by auditing my own experiments, not someone else's.*

This directory is an unusual thing to publish: a catalogue of the mistakes I made building
this benchmark, why each one produced a *plausible but wrong* conclusion, and how I detected
it. I am publishing it because none of these traps are specific to me — every one of them is
easy to fall into, and most of them fail **silently**: the experiment still runs, the numbers
still look reasonable, and the paper still reads well.

Pitfalls 1–11 came from an audit run on 2026-07-17 against `src/`, the raw results, and the
report draft. **Pitfall 12 was found on 2026-08-17, in this repository's own `README.md`,
four weeks after the rest of this list was published** — it is a repeat of #10, and it is the
reason the list is not closed. Every number quoted below is reproducible from the committed
code.

Two of the tables below (#1, #3) quote values as the audit measured them at the time, against
the draft pipeline. Where the final committed results differ, the current numbers are noted
inline — the conclusions are unchanged, and in #1 the current data is the stronger case.

---

## 1. A self-referential metric rewards the *worst* policy

**The trap.** I measured "adaptation speed" as: after the breakpoint at `t = 5000`, how many
rounds until instantaneous regret returns to normal — where *normal* was defined as
`η = 1.1 × ρ_min`, and `ρ_min` was that policy's **own** average regret over the last 500
rounds.

**Why it fails.** Two students take a make-up exam after being ill. The rule is "recovered =
back to your own usual average." The strong student (usually 9/10) must reach 9 again. The
weak student (usually 3/10) only needs 3. The weak student always "recovers faster" — not
because they are healthier, but because their bar is lower.

**How I caught it.** I ran the metric on Round Robin — an algorithm that cannot adapt to
anything, by construction:

| Policy | τ_adapt (rounds) | baseline regret ρ_min |
|---|---|---|
| **Round Robin** | **1.95** ← "fastest adapter" | 0.03124 (worst) |
| Least Connections | 3.41 | 0.02764 |
| UCB | 233.63 | 0.00414 |
| Thompson Sampling | 2314.83 | 0.00266 (best) |

Spearman correlation between τ_adapt and baseline regret across 9 methods: **ρ = −0.867,
p = 0.0025**. The metric was measuring how lenient each policy's own threshold was, not how
fast it adapted. A conclusion built on it had to be withdrawn entirely.

*Current committed data (the legacy metric is retained in `results_summary.json` as
`legacy_tau_adapt_mean` precisely so this stays checkable): the effect reproduces at
**ρ = −0.836, p = 0.0014** across all 11 policy configurations, and across the 7 tuned
configurations alone it is **ρ = −1.000** — a perfect rank inversion. On the set of policies
anyone would actually compare, the "adaptation speed" metric ranked them in exactly reverse
order of how well they performed.*

> **Rule.** Before trusting a metric, run it on an algorithm that *cannot possibly* have the
> property you are measuring. If that algorithm scores well, the metric is broken — not the
> algorithm. I now call this the **absurd control group** check and it is a mandatory step.

---

## 2. Two contradictory accounts of where the hyperparameters came from

Section 4.4 of the draft said hyperparameters were "set to the theoretically recommended
values from Chapter 3." Section 4.7, a few pages later, showed they were the output of a grid
search — and the "theoretical default" was a *different* pair of values. Chapter 3, when
checked, recommended no numeric values at all; it said the opposite, that these parameters
"require prior knowledge of the drift frequency."

The cause was mundane: section 4.4 was written before the tuning protocol was finalised, and
never updated.

> **Rule.** Every configuration constant in the prose must be traceable to a file. If the
> answer to "where did this number come from?" is "I remember it being that," you have found
> where the contradiction will grow. Ideally the paper *includes* generated values rather than
> restating them by hand.

---

## 3. A grid-search optimum on the boundary means the grid is wrong

| τ (SW-UCB) | regret | | γ (D-UCB) | regret |
|---|---|---|---|---|
| 50 | 319.43 | | 0.900 | 343.39 |
| 100 | 305.96 | | 0.950 | 337.32 |
| 200 | 285.67 | | 0.990 | 319.67 |
| 500 | 244.09 | | 0.995 | 307.45 |
| **1000 — grid edge** | **199.86** | | **0.999 — grid edge** | **249.32** |

Monotone across every point on both grids, with the optimum at the edge. Reporting that as
"the optimal value" is like turning a radio dial to its stop while the signal is still getting
stronger and concluding the best frequency is the end of the dial.

The logical consequence was the interesting part, and I had avoided it: **SW-UCB with τ → ∞
*is* UCB**, and **D-UCB with γ → 1 *is* UCB**. My own tuning was saying that the best way to
configure these non-stationary algorithms, in this benchmark, is to make them as close to the
stationary algorithm as possible. That is a finding — but only if you say it out loud.

> **Rule.** Optimum at the boundary + monotone trend = the grid is wrong, not the result. And
> when tuning "wants" to turn algorithm A into algorithm B, ask whether the benchmark is
> telling you A is unnecessary here.

*What happened after: the grid was extended to 8 points per algorithm (`results/table_4_7.txt`),
and the optimum kept moving outward until it reached the only place it could stop — `τ = 10000`
(= T) and `γ = 1.0`, the degenerate limits themselves. There is nothing past that boundary to
extend into: a window longer than the horizon is the horizon.*

---

## 4. No statistical test, absolute language — and N was 5, not 150

The results chapter reported `mean ± std` and then concluded with phrases like "fully
confirms H4" and "clearly superior." There was not a single p-value or confidence interval.

Worse, the sample size was wrong. 150 trajectories = **5 instances × 30 noise seeds**. Thirty
runs on the same instance share the same gap structure between arms; they are strongly
correlated. The number of independent units is **5**. Evidence of this was sitting in my own
table: a non-greedy rate with `±22.89%` standard deviation — that dispersion comes from
*between-instance* variation, not Monte Carlo noise.

Two distributions were also badly skewed, making `mean ± std` actively misleading:
ε-greedy stationary regret `116.38 ± 194.10` (std 1.7× the mean — a bimodal distribution where
most runs converge and a minority get stuck on a wrong arm) and τ_adapt `7.8 ± 38.2`.

**Fix:** average the 30 runs within each instance first, giving 5 independent values per
(algorithm, scenario); then run paired tests across instances, report bootstrap CIs, and use
median + IQR for skewed quantities.

> **Rule.** `mean ± std` is description, not evidence. And before counting N, ask *"N of what,
> independent of each other?"* — 150 trajectories from 5 environments is N = 5.

---

## 5. A "verification" number that contradicted its own closed form

The draft claimed to have empirically verified the Jensen gap between `E[1/L]` and `1/E[L]`
and reported it "varying from 12.5% to 14.8% depending on the skew of the distribution."

But both formulas were already in the paper: `E[L] = kθ` and `E[1/L] = 1/((k−1)θ)`. Multiply
them and θ cancels:

```
E[1/L] × E[L] = k/(k−1) = 5/4 = 1.25        for k = 5
```

The gap is a **closed-form constant** — 25.0% relative to `1/E[L]`, or 20.0% relative to
`E[1/L]` — independent of θ, therefore identical for every arm, every timestep, every
scenario. A 2M-sample Monte Carlo check confirmed 25.02% and 24.97% for θ = 5 and θ = 15.

The same paragraph also managed to contradict itself: the gap "varies depending on the skew"
and "is constant across arms" cannot both be true.

> **Rule.** When a quantity has a closed form, check the measurement against the formula
> *before* putting it in a section titled "verification." A verification section that
> contradicts itself is worse than having none.

---

## 6. The metric defined in the text was not the metric computed in the code

Two genuinely different quantities were being used interchangeably:

- **non-optimal rate** — % of rounds choosing an arm other than the environment's true optimum
  `a*(t)`. Requires ground truth. Measures *how often you were wrong*.
- **non-greedy rate** — % of rounds choosing an arm other than the one the policy itself
  currently believes is best. Measures *how much you explored*.

A policy can be 0% non-greedy and 100% wrong: diligently exploiting a mistaken belief.

The table defined metric (1). `simulator.py` computed metric (2) — from the agent's own
empirical means, never touching `env.optimal_arms`. The result caption used the Vietnamese
name for (1) and the English name for (2) over the same column of numbers.

This produced a real invalid inference: a 53.74% figure was used to argue that SW-UCB "must
sustain a very high non-optimal selection rate, causing regret loss." Under definition (2)
that number only says it explored a lot — and exploring a *near-optimal* arm costs almost
nothing.

> **Rule.** Exploration and error are different things. Each metric needs exactly one
> mathematical definition, and that sentence must match the line of code that computes it,
> word for word.

---

## 7. One algorithm was handicapped by its configuration, then judged on its "mechanism"

H1 claimed directed exploration (UCB/TS) beats random exploration (ε-greedy). But ε-greedy ran
with `c = 0.1, d = 1.0, K = 10`, giving `ε_t = min(1, 1/t)` and therefore:

```
Σ ε_t ≈ 1 + ln(10000) ≈ 9.8 exploration rounds — out of 10,000
```

With 10 arms, that is roughly **one random exploration per arm** across the entire horizon.
The measured non-greedy rate of 0.36% confirmed it.

Auer et al. (2002), the theorem being cited to justify this schedule, requires **c > 5**. At
c = 0.1 — fifty times smaller — the bound does not apply, and the "premature convergence to a
suboptimal arm" the report described is the textbook consequence of that choice, not an
empirical discovery.

The tuning budget was also unequal: SW-UCB and D-UCB each got a 5-point grid; ε-greedy got no
tuning at all. The comparison put two optimised algorithms next to a disadvantaged one and drew
conclusions about *mechanisms*.

> **Rule.** Comparative claims are only meaningful under a declared, equal tuning budget.
> "A beats B" without that qualifier is not a result — and when re-run fairly, this particular
> ranking reversed.

---

## 8. Oracle leakage through reward normalisation constants

Every policy learned on rewards normalised to [0,1] using two hard-coded constants:

```python
r_min = 0.005   # = 1/175.12 ms — 99th percentile of latency
r_max = 0.150   # = 1/6.70 ms   — 1st  percentile of latency
```

Those percentiles come from the latency distribution of **the very environment being
evaluated**. In a real deployment nobody knows them in advance — to know your cluster's latency
percentiles you must first measure them, i.e. partly solve the problem before running the
algorithm that solves it.

This is mild oracle leakage. Mitigating factor: all MAB policies received the same normalised
reward, so MAB-vs-MAB comparison is not directly biased. But the H4 conclusion ("MAB beats
classical baselines in real deployments") weakens, because Round Robin and Least Connections
need no such constants while the bandits do — and the cost of estimating that scale online was
never charged to them.

Worse, the normalisation layer was added *after* observing that UCB over-explored on this
benchmark: a design decision made in response to results on the same data used to score it.

> **Rule.** Any constant derived from the evaluation environment is oracle information until
> proven otherwise. Ask of every preprocessing step: *could I compute this before deployment?*

*Fixed in the committed pipeline: `simulator.py` now normalises from each policy's own observed
history — running min/max for the first 100 rounds, then 1st/99th percentiles of its own
observations recomputed every 500 rounds — and the hard-coded `r_min = 0.005`, `r_max = 0.150`
constants are gone. The residual asymmetry is not fixed and is not fixable this way: the bandits
still carry a scale-estimation mechanism that Round Robin and Least Connections do not need, and
its cost is not charged to them anywhere in the results.*

---

## 9. Half of a two-part hypothesis quietly disappeared

H1 had two clauses: (a) UCB ≈ Thompson Sampling, **and** (b) both > ε-greedy. The results
section tested and celebrated (b), and simply stopped mentioning (a) — while the data
contradicted it clearly: TS `40.74 ± 8.26` vs UCB `65.25 ± 13.10` on the stationary scenario,
60% apart with non-overlapping ±1 std intervals.

This is exactly the behaviour the pre-registration protocol — which the report proudly claimed
to follow — exists to prevent. Violating your own stated commitment is worse than never making
one.

And the dropped half was *interesting*: TS wins when stationary but loses to UCB under both
gradual and abrupt drift. "TS wins when stable, UCB is more robust when things move" is a
finding. Discarding it was pure waste.

> **Rule.** A hypothesis of the form "A and B" requires reporting both A and B, including the
> half that fails. A hypothesis honestly reported as half-rejected is far more credible than
> one "confirmed" by silence.

---

## 10. Hand-typed summary numbers that contradicted my own tables

Two headline sentences — one closing the tuning section, one in the overall conclusion — both
contained ranges that did not match the tables directly above them.

**"Tuning reduces dynamic regret by 30–50% across *all* scenarios":** 3 of 6 cells fell
outside, one as low as 17.9%. The word *all* is dangerous — a single cell breaks the sentence.

**"MAB policies achieve 3–8× lower cumulative regret":** 11 of 15 cells outside the range, in
both directions (13.1× in one cell, 1.2× in another). The 1.2–1.7× cells are precisely those
that might not survive the statistical test from pitfall #4 — so the report's first conclusion
was anchored to its weakest evidence.

Both errors were "glance at the table and estimate." Both take a reviewer thirty seconds with a
calculator to catch — on numbers I published myself. Conclusions are the section a committee
reads *most* carefully, often before the results.

**Fix:** `make_summary_numbers.py` now regenerates every summary figure directly from the raw
results. No summary number in the final report was typed by hand.

> **Rule.** Never hand-type an aggregate. If a number appears in prose, a script should have
> produced it.

---

## 11. An unverifiable constant inside a derivation

The argument for why no sliding window τ is usable at high drift density rested on:

```
√(2 ln τ · K / τ)  ≲  gap     ⟹     τ ≳ 1200
```

The draft used "normalised gap ≈ 0.35–0.4" with **no formula, script, or reference** for where
that came from. Three different quantities were all being called "gap":

| | quantity | value | unit |
|---|---|---|---|
| 1 | theoretical relative gap `1/(k−1)` | 25.0% — closed form | dimensionless |
| 2 | relative latency gap `(L₂−L₁)/L₁` | 6.7%–33.8% | milliseconds |
| 3 | **normalised-reward gap** — the one the formula needs | **≈ 0.082** | online-normalised reward |

Quantity (2) had been substituted for quantity (3). They happen to be the same order of
magnitude but live in different spaces: SW-UCB's confidence bound applies to normalised reward
estimates, not to milliseconds.

Unlike the Jensen gap, this one has **no closed form** — the online normaliser re-estimates
`(r̂_min, r̂_max)` from the policy's own observed history every 500 rounds, so the scale depends
on which policy is running and how far it has converged. The only way to get it is to run the
real simulator.

Measured properly: **0.0823**, which solves to **τ ≈ 30,481** — larger than the entire horizon
`T = 10,000`. The qualitative conclusion survived; in fact it got *stronger*.

> **Rule.** If a symbol appears in a derivation, it needs a formula or a measurement script.
> "Approximately 0.35" with no provenance is a guess wearing the costume of a constant.

---

## 12. The catalogue did not stop pitfall #10 from happening again

**Found 2026-08-17**, four weeks after this list was published, while re-deriving every number
in the top-level `README.md` for a write-up rather than trusting it. Three of its summary
sentences did not survive the check — all three the same failure as #10, in the one document
written *after* the fix for #10 was already in place.

| Claim in `README.md` | Recomputed from `results/` |
|---|---|
| regret improvement "median 8.1×" | **median 5.2×** — 8.10× is the value of a single cell (D-UCB, gradual drift), relabelled as an aggregate |
| "TS beats UCB consistently, `p = 0.007`" | holds on stationary (5/5, `p = 0.0067`) and abrupt drift (5/5, `p = 0.0066`); **fails on gradual drift** (4/5, `p = 0.206`) |
| default D-UCB "statistically indistinguishable" from Least Connections | distinguishable in all three scenarios (`p = 0.0087`, `0.0088`, `0.0367`) — small differences, but consistent in sign across instances, which is exactly what a paired test detects. The right word is ***practically*** indistinguishable |

No underlying result changed; the corrected numbers argue the same direction. What changed is
what I now think #10's fix actually accomplished.

The fix for #10 was `make_summary_numbers.py`, and it worked — no summary number in the
*report* was typed by hand after it existed. But the fix was scoped to the artefact where the
bug was found, not to the failure mode. `README.md` is a different published surface, written
by hand, outside that pipeline, and the same failure walked straight back in through the one
door nobody had thought to cover. A catalogue of your own mistakes is not a vaccine; it
documents where you have already been careless, which is not the same as where you will be
careless next.

The three claims above are also, individually, the most persuasive sentences in the document —
the headline ratio, the significance claim, the honest-sounding caveat. That is not a
coincidence. Summary prose is where a number is doing the most rhetorical work and receiving
the least scrutiny, because by then the author is describing something they already believe.

> **Rule.** When you fix a "no number without a source" bug, fix it for **every** surface that
> publishes numbers — report, README, slides, abstract — not just the one that was caught. And
> re-derive, rather than re-read, when quoting your own prior work: reading your own summary
> checks whether you remember it, not whether it is true.

---

## Four meta-lessons

**1. Test every metric against an absurd control group.** Pitfall #1 was invisible for weeks
and became obvious in one line the moment Round Robin was included in the table.

**2. Generate every aggregate; type none of them.** Pitfalls #5, #10 and #11 are the same
failure in three costumes — a number that entered the prose without a source.

**3. Write the methods section *after* the method is frozen, from a single source of truth.**
Pitfall #2 exists purely because prose was written ahead of a decision and never revisited.

**4. Fix the failure mode, not the artefact it was found in.** Pitfall #12 is #10 walking back
in through a document the fix for #10 did not cover. Every fix above should be read as a
question — *what else does this apply to that I have not checked yet?* — rather than as a
line item that has been closed.

---

## Why publish this

Three of four pre-registered hypotheses in this study were rejected, one of them in the
direction opposite to what I predicted. That is not a failed experiment — a benchmark that can
only confirm its author's expectations is not measuring anything.

The same applies to this list. An audit you run on yourself and then hide is a rehearsal; one
you publish is a result.

And the list stays open. Pitfall #12 was added four weeks after the first eleven, and it is a
repeat of one of them — which is the most useful thing this document has done so far, because
a catalogue that stopped growing would only mean I had stopped re-deriving my own numbers.
