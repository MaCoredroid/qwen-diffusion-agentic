# Cross-instance answer-leakage audit — swe_datagen_s1

**Question answered (verbatim from the ask):** *how similar are the trained
Verified-adjacent instances to the held-out 113?* — measured (scan) + adjudicated
(eyeball) with quoted evidence, in **measure-and-report mode**: the DROP-LIST below
is *listed, not executed*. The drop decision runs at pool freeze.

Generated `2026-07-07T06:15:08Z` · CPU-only · `.venv` python · 0.79 s ·
text-sim backend `tfidf_cosine_sklearn` (TF-IDF cosine).
Re-runnable verbatim at pool freeze: `runs/swe_datagen_s1/leakage_audit.py`.

---

## POOL-FREEZE RE-AUDIT — 2026-07-09 (Task #87, gates the initial SFT set)

Re-ran `leakage_audit.py` **verbatim** (baseline commit `db8d269`) on the current
**311-keeper** pool at snapshot time. This is the pre-registered pool-freeze gate,
triggered early because SFT begins tonight on the current pool (chunked-resume
training extends the set later — each extension re-triggers this audit).

**Integrity pins (all PASS):** holdout = **113** sha-asserted ids
(`inner5 ∪ tier0_20 ∪ tier1_100`), sha256 `c56f473a…168e` pin-assert **PASS**;
keepers snapshot sha256 `2aeecce1…`; SWE-Gym disjoint-repo assert **PASS**.

**Pool shape at snapshot:** 311 keepers = **193 SWE-bench_Verified + 118 SWE-Gym**
by dataset source; by **teacher/generator**: **294 stock-Qwen3.5-9B-AR + 17
Qwen3.6-27B-NVFP4+MTP**. The **Opus-4.8 pilot keepers are NOT yet in the file**
(4 pending promotion) — they enter at a later chunk and will **re-trigger this
audit verbatim before joining**. So this stamp gates a **two-teacher** initial set.

**Topline (measured):** same-repo keeper×holdout pairs **5816**; flagged **368**;
keepers with a GOLD-file overlap **65**; keepers with function-level overlap **12**
(up from 1 at the n=114 baseline); issue-text sim ≥ p95 **116**.

### Adjudication of NEW flagged pairs since the n=114 baseline

All 12 function-level-overlap pairs were read by hand (keeper final_patch + own gold
patch + full trajectory/PS vs holdout gold+test), plus a **systematic verbatim
leak-line scan** over all 110 gold-file-overlap pairs: for each pair, every
substantive `+` line of the holdout's gold patch (restricted to shared files) was
searched in the keeper's full training text, then **split by field** (final_patch /
problem-statement / assistant text / whole trajectory) and by chronological
**direction** (keeper-is-follow-up `K>H` vs keeper-is-predecessor `K<H`).

**Key discriminator (direction × surface):** a keeper that is a *chronological
follow-up* to a same-file holdout has the holdout's fix **already merged into its
base repo**, so those lines appear as *ambient file content the model read* — benign
(this is exactly why the baseline kept `xarray-6599→4687`, which surfaces only a
generic 1-line call). A leak requires the holdout's **distinctive** added lines to
be **surfaced in the trained-to-produce final_patch or framed in the problem
statement** — not merely read mid-trajectory.

Only **3 of 110** pairs put any holdout-added line in the keeper's `final_patch`:

| keeper → holdout | dir | in final_patch | verdict |
|---|---|--:|---|
| `pydata__xarray-6461` → `pydata__xarray-4687` | follow-up | 5–10 distinctive | **ANSWER_LEAK** |
| `django__django-13128` → `django__django-13121` | follow-up | 7 (idioms) | SUBSYSTEM_ADJACENT (tight) |
| `scikit-learn-11578` → `scikit-learn-14087` | predecessor | 1 (generic) | BENIGN |

- **`xarray-6461 → 4687` = ANSWER_LEAK.** 6461 is a direct follow-up *bug report on
  the very `where(keep_attrs=…)` feature 4687 introduced*. Its problem-statement
  traceback quotes 4687's added lines verbatim, and its `final_patch` edits `where()`
  itself, carrying **`def where(cond, x, y, keep_attrs=None):`**, the explanatory
  comment `# keep the attributes of x, the second parameter…`,
  **`keep_attrs = lambda attrs, context: attrs[1]`**, **`keep_attrs = _get_keep_attrs(default=False)`**,
  and **`keep_attrs=keep_attrs,`** as context. A model trained on 6461 has seen
  essentially all of held-out 4687's gold implementation. (Materially different from
  the baseline-cleared `6599→4687`, which edits `_ensure_numeric` and never surfaces
  the `where()` body — confirmed: 6599 shows only 1 generic call line, mid-trajectory.)
- **`django-13128 → 13121` = SUBSYSTEM_ADJACENT (tightest non-leak).** The 7 matched
  lines are django duration-type-detection **idioms** (`…output_field.get_internal_type()`,
  `fields.DurationField()`, `datetime_fields = {…}`) re-used inside a *new*
  `_resolve_output_field` method. It does **not** reproduce 13121's distinctive
  SQL-generation fix (`DurationExpression.as_sql` / `format_for_duration_arithmetic`
  / `DurationValue` removal) that 13121's FAIL_TO_PASS actually tests. Flagged for
  monitor visibility; **not** on the drop-list.
- The remaining 13 verbatim-match pairs (incl. `sympy-24213→24066`, `django-13012/13449→13121`,
  `django-15467→12713`) match **only in mid-trajectory ambient file reads** or on
  generic idioms — the benign merged-neighbor pattern the baseline already accepts.
- All ~98 gold-file-only pairs (no function overlap) are same-file / different-function
  in django/matplotlib/sympy **hub files** (`query.py`, `expressions.py`,
  `fields/__init__.py`, `base.py`, `admin/options.py`, `_axes.py`) — SUBSYSTEM_ADJACENT.
  Spot-checked top-5 by text-sim (`django-15380→15104` autodetector, `sympy-22714→17655`
  point.py, `matplotlib-26113→13989` _axes.py, `matplotlib-25311→20859` legend.py,
  `django-13670→14373` dateformat.py): all disjoint-fix. Note `django-13670→14373` is
  the **safe direction** — the keeper is the *predecessor* (`y()` 2-digit fix), which
  appears as unchanged **context** in 14373's patch; 14373's actual added line
  `return '%04d' % self.data.year` (`Y()` method) is **not** in the keeper → benign.

### The 2 sev-10 django frontier flags from the baseline — RESOLVED CLEAN

Baseline forward-guard flagged `django-11138 → 13121` and `django-16032 → 11734` as
sev-10 *frontier* ids to re-examine **if** they became keepers. **Neither became a
keeper** (0 rows in `keepers.jsonl` / snapshot as an `instance_id`; the 7 incidental
`16032` string hits are trajectory-text substrings, and `11734` appears only as a
held-out id). Forward-guard prediction holds.

### DROP-LIST (measure-and-report — LISTED, not executed)

> **`["pydata__xarray-6461"]`** — one keeper adjudicated `ANSWER_LEAK`.
> **Do NOT delete** — surfaced for the monitor's drop decision.

Remedy options for the monitor (either fully restores slice cleanliness): **(a)**
drop keeper `pydata__xarray-6461` from the training set (recommended; 1 of 311), or
**(b)** excise held-out `pydata__xarray-4687` from the 113-id eval ring. Because both
training arms consume the same pool, the exposure inflates **both** arms' score on
4687 **equally** — it is a shared confound that does **not** bias the arm-vs-arm delta
(the promotion currency); it touches only the absolute 113-ring number, which the
standing rule already treats as train-adjacent, not a promotion signal. The leaked
keeper is a **stock-9B-AR** Verified instance (unrelated to any teacher swap).

### Opus-teacher firewall (for the pending Opus-4.8 pilot keepers)

Opus-teacher keepers, when promoted, get the **identical instance-level firewall**:
`expand_frontier.py` hash-asserts that **no eval-ring id ever enters training**, and
this cross-instance audit re-runs verbatim over them before they join. State plainly:
**Opus may have memorized public GitHub repos, but that affects TEACHER STRENGTH
(how good the demonstrations are), not EVAL LEAKAGE — because the evaluated 113
instances never train, on any teacher.** Teacher pretraining-memorization is a
capability-attribution caveat for the datagen arm, not a contamination of the held-out
slice.

### GATE VERDICT: **CONDITIONAL PASS** for tonight's SFT launch

310 of 311 keepers are clean; the single `ANSWER_LEAK` (`pydata__xarray-6461`) is
surgically removable and does not corrupt the arm-vs-arm currency. Training launch on
the 311-pool is **authorized to proceed**, with the standing condition that the
monitor applies one of the two remedies above (drop `xarray-6461` from train, or
`xarray-4687` from eval) at or before the next chunked-resume extension, and that the
pending Opus-4.8 pilot keepers re-trigger this audit verbatim before promotion.

_Re-audit generated `2026-07-09T17:15:18Z` · CPU-only · 2.08 s · sha pins PASS._

---

## Standing rule (read this before quoting any number here)

- **The absolute numbers against the 113-ring are `train-adjacent`, not a
  promotion signal.** This audit measures *whether a trained keeper hands a
  held-out task its fix* (a firewall question). Similarity/overlap counts against
  the eval ring describe the training pool's shape; they are **not** evidence that
  the diffusion arm is better or worse.
- **Paired arm-vs-arm reads are the promotion currency.** A capability is promoted
  only on a raw/constrained *model-only* gain measured **arm vs arm on the same
  held-out slice** — never on an absolute 113-ring score. This audit exists to keep
  that slice clean, not to score on it.

The two audit layers are independent:
- **Instance-level holdout** is already airtight upstream — `expand_frontier.py`
  hash-asserts that no eval-ring id ever enters training.
- **This audit** is the *cross-instance* layer: a trained keeper whose patch (or
  whose issue text) is close enough to a held-out instance that solving the keeper
  could reveal the held-out fix.

---

## Integrity pins (all asserted PASS)

| pin | value | status |
|---|---|---|
| eval holdout | **113 distinct ids** (`inner5 ∪ tier0_20 ∪ tier1_100`) | rebuilt byte-identical to `expand_frontier.py` |
| holdout sha256 | `c56f473ad31e52bee0f85151562f4e2122e4815dfa3f1b776b15fe121e8d168e` | **pin-assert PASS** vs `.eval_holdout_sha256` |
| keepers snapshot | **114 rows** — 54 Verified-source + 60 SWE-Gym | sha256 `61e1f89092b34e6f…` |
| SWE-Gym disjoint-repo | 0 repo collisions with any holdout repo | **assert PASS** |

Because the 60 SWE-Gym keepers share **no repo** with any holdout id, the *entire*
cross-instance leakage surface is the **54 Verified-source keepers**.

---

## Topline (measured)

| metric | value |
|---|--:|
| same-repo keeper×holdout pairs | **720** |
| flagged pairs (severity-ranked) | **49** |
| keepers with a same-repo holdout neighbor | **53** |
| keepers with ANY file overlap vs holdout gold+test | **15** |
| keepers overlapping a holdout **GOLD** file | **15** |
| keepers with **function-level** symbol overlap | **1** |
| keepers with issue-text sim ≥ p95 (p95 = 0.16608, p99 = 0.365) | **18** |

**Reading of the topline:** all 15 file co-locations are *same-file / different-
function*; exactly **1** keeper (`sympy-12481`) shares a function-level symbol
(`Permutation.__new__`) with any holdout, and even that one is a co-location, not a
shared fix (adjudicated below). Highest text-sim pair (`xarray-3305 → 4687`, 0.648)
has **zero** file overlap — it is boilerplate MCVE bug-report similarity.

---

## Eyeball verdicts — the 16 highest-signal pairs (all with GOLD-file overlap)

Every keeper that touches a held-out **gold** file was read by hand against the
held-out gold+test patch. **Verdict legend:** `ANSWER_LEAK` (keeper reveals the
held-out fix — would be dropped) · `SUBSYSTEM_ADJACENT` (same file/subsystem, but
disjoint functions & logic — solving the keeper hands over nothing) · `BENIGN`
(repo-only, no shared gold file).

**Result: 16 of 16 = `SUBSYSTEM_ADJACENT`. Zero `ANSWER_LEAK`.**

| # | keeper | holdout | shared gold file | func∩ | text-sim | verdict |
|--:|---|---|---|---|--:|---|
| 1 | `sympy__sympy-12481` | `sympy__sympy-12489` | `combinatorics/permutations.py` | `Permutation.__new__` | 0.166 | SUBSYSTEM_ADJACENT |
| 2 | `pydata__xarray-6599` | `pydata__xarray-4687` | `xarray/core/computation.py` | — | 0.392 | SUBSYSTEM_ADJACENT |
| 3 | `pytest-dev__pytest-10081` | `pytest-dev__pytest-8399` | `src/_pytest/unittest.py` | — | 0.221 | SUBSYSTEM_ADJACENT |
| 4 | `pytest-dev__pytest-7236` | `pytest-dev__pytest-8399` | `src/_pytest/unittest.py` | — | 0.152 | SUBSYSTEM_ADJACENT |
| 5 | `scikit-learn__scikit-learn-11578` | `scikit-learn__scikit-learn-14087` | `sklearn/linear_model/logistic.py` | — | 0.142 | SUBSYSTEM_ADJACENT |
| 6 | `psf__requests-1142` | `psf__requests-5414` | `requests/models.py` | — | 0.139 | SUBSYSTEM_ADJACENT |
| 7 | `psf__requests-1724` | `psf__requests-5414` | `requests/models.py` | — | 0.123 | SUBSYSTEM_ADJACENT |
| 8 | `psf__requests-1921` | `psf__requests-5414` | `requests/models.py` | — | 0.118 | SUBSYSTEM_ADJACENT |
| 9 | `pytest-dev__pytest-6202` | `pytest-dev__pytest-8399` | `src/_pytest/python.py` | — | 0.112 | SUBSYSTEM_ADJACENT |
| 10 | `matplotlib__matplotlib-24149` | `matplotlib__matplotlib-13989` | `lib/matplotlib/axes/_axes.py` | — | 0.104 | SUBSYSTEM_ADJACENT |
| 11 | `sympy__sympy-12419` | `sympy__sympy-17630` | `matrices/expressions/matexpr.py` | — | 0.081 | SUBSYSTEM_ADJACENT |
| 12 | `sympy__sympy-11618` | `sympy__sympy-17655` | `sympy/geometry/point.py` | — | 0.079 | SUBSYSTEM_ADJACENT |
| 13 | `django__django-11095` | `django__django-12713` | `contrib/admin/options.py` | — | 0.078 | SUBSYSTEM_ADJACENT |
| 14 | `django__django-11133` | `django__django-13195` | `django/http/response.py` | — | 0.074 | SUBSYSTEM_ADJACENT |
| 15 | `pytest-dev__pytest-7982` | `pytest-dev__pytest-5840` | `src/_pytest/pathlib.py` | — | 0.060 | SUBSYSTEM_ADJACENT |
| 16 | `django__django-11095` | `django__django-16100` | `contrib/admin/options.py` | — | 0.032 | SUBSYSTEM_ADJACENT |

### Quoted evidence (load-bearing diffs)

**#1 `sympy-12481 → 12489` — the single function-level hit, and it is co-location only.**
Both patches touch `permutations.py::Permutation.__new__`, but in disjoint regions
solving unrelated problems. Keeper (12481, "constructor fails with non-disjoint
cycles") adds exactly one line — `if not is_cycle:` — deleting the `ValueError`
branch in the `has_dups` block so non-disjoint cycles apply left-to-right. Holdout
(12489, "Permutation can't be subclassed") is a subclassing refactor:
`@staticmethod def _af_new(perm)` → `@classmethod def _af_new(cls, perm)`,
`Basic.__new__(Perm, perm)` → `Basic.__new__(cls, perm)`, tail
`obj = Basic.__new__(cls, aform)` → `return cls._af_new(aform)`. **0 of the 34
distinct added lines in the holdout gold patch appear anywhere in the keeper row**
(patch, prompt, or 69-message trajectory); the keeper's added line does not appear
in the gold patch. `"subclass"`/`"classmethod"` appear 0 times in the trajectory.

**#2 `xarray-6599 → 4687` — same gold file, disjoint functions; hazard checked.**
Keeper edits only `_ensure_numeric` (polyval dtype): `if x.dtype.kind in "mM":` →
`== "M"` plus an `elif x.dtype.kind == "m": … astype("float64")`. Holdout edits only
`where()`: adds `keep_attrs=None`, `_get_keep_attrs(default=False)`, the decisive
`keep_attrs = lambda attrs, context: attrs[1]`. The 6599 base already contains the
merged 4687 fix, so all 61 trajectory messages were grepped: the trajectory **never
reveals** the `where()` signature change, the `_get_keep_attrs` defaulting, or the
`attrs[1]` line the `test_where_attrs` FAIL_TO_PASS requires. Elevated text-sim
(0.392) is boilerplate `xr.show_versions()` dumps + adjacent test insertion.

**#3/#4/#9 `pytest-10081, -7236, -6202 → 8399`.** All three share a pytest gold file
but touch disjoint functions. 10081/7236 edit `TestCaseFunction.runtest`/`.teardown`
(`--pdb` + class-level `unittest.skip` teardown timing); 6202 edits
`PyobjMixin.getmodpath` (`return s.replace(".[","[")` → `return s`). Held-out 8399
is a fixture-visibility rename in `_make_xunit_fixture` /
`_inject_setup_*_fixture` — `name=f"unittest_…"` → `f"_unittest_…"` (leading
underscore) plus four analogous renames in `src/_pytest/python.py`. None of the three
keepers contains any fixture-naming logic or the leading-underscore convention.

**#5 `sklearn-11578 → 14087`.** Keeper: one kwarg in `_log_reg_scoring_path` —
`LogisticRegression(fit_intercept=fit_intercept)` → `…, multi_class=multi_class)`.
Holdout: inside `LogisticRegressionCV.fit` — `if self.multi_class=='ovr':` →
`if multi_class=='ovr':` plus elasticnet-only `l1_ratio_` bookkeeping. Different
functions, different bug classes; 11578's 0.20-era base predates the elasticnet code
14087 fixes.

**#6/#7/#8 `requests-1142, -1724, -1921 → 5414`.** All three edit
`requests/models.py::PreparedRequest` but disjoint methods: 1142 `prepare_content_length`
(body/`Content-Length`), 1724 `prepare_method` (`to_native_string(...)`), 1921
`prepare_headers` (None-filtering). Held-out 5414 edits `prepare_url`:
`host.startswith(u'*')` → `host.startswith((u'*', u'.'))` guarding
`raise InvalidURL('URL has an invalid label.')`. No keeper mentions `InvalidURL`,
idna, hosts, or `startswith`; the idna host-label path did not exist in the keepers'
years-older base trees.

**#10 `matplotlib-24149 → 13989`.** Keeper edits only `cbook/__init__.py::_safe_first_finite`
(wraps `next(...)` in `try/except StopIteration`). Held-out edits one line in
`_axes.py::hist` (`hist_kwargs = dict(density=density)` → `hist_kwargs['density'] =
density`). Grep of the 82 KB trajectory: **0** occurrences of `hist_kwargs`,
`def hist`, `density`, `range=`; 68 of `_safe_first_finite`. The flag is pure
overlap on the ~8000-line `_axes.py` hub (keeper only appears there via a traceback).

**#11 `sympy-12419 → 17630`** (`matexpr.py`): keeper edits `Identity._entry`
(`KroneckerDelta(i, j)`); holdout edits module-level `_postprocessor` (`MatAdd`
dispatch). **#12 `sympy-11618 → 17655`** (`geometry/point.py`): keeper edits
`Point.distance` (zero-pad mismatched dims); holdout adds `Point.__rmul__`.
**#13/#16 `django-11095 → 12713 / 16100`** (`admin/options.py`): keeper adds
`get_inlines` hook; holdouts edit `formfield_for_manytomany` (widget guard) and
`changelist_view` (`transaction.atomic`) respectively — ~1400 lines apart, `func∩=0`.
**#14 `django-11133 → 13195`** (`http/response.py`): keeper adds `memoryview`
coercion to `make_bytes`; holdout rewrites `delete_cookie` (`samesite`). **#15
`pytest-7982 → 5840`** (`pathlib.py`): keeper flips `follow_symlinks=False→True` in
`visit()`; holdout deletes `unique_path()` and re-keys the conftest cache in
`config/__init__.py`. In every case the trained diff and its problem statement never
mention the identifiers the held-out fix and its FAIL_TO_PASS tests require.

*(Full per-pair evidence with exact diff hunks is preserved in the campaign's eyeball
record and `leakage_audit_report.json`.)*

---

## DROP-LIST (measure-and-report — LISTED, not executed)

> _(This is the **n=114 baseline** drop-list. Superseded at pool freeze — see the
> **POOL-FREEZE RE-AUDIT** section above, whose current drop-list is
> `["pydata__xarray-6461"]`.)_

> **`[]` — empty. No keeper was adjudicated `ANSWER_LEAK`.**

No trained keeper's patch or trajectory hands a held-out task its fix. The 15
gold-file co-locations are same-file / different-function and warrant only spot
review (done above), **not exclusion**. If the drop decision is executed at pool
freeze, **zero keepers** are removed on leakage grounds. Should any of the 54
Verified keepers be re-derived or the pool grow, re-run `leakage_audit.py` verbatim
and re-adjudicate any *new* GOLD-file-overlap pair before promotion.

---

## Frontier pre-screen (not-yet-collected Verified-adjacent ids)

A forward guard over the **333** not-yet-collected same-repo frontier ids vs holdout
gold files (gold-patch files only): **221** gold-file-overlap pairs across **114**
frontier ids — django-heavy, driven by shared hub files (`db/models/sql/query.py`,
`db/models/expressions.py`, `utils/autoreload.py`). Two django ids reach sev=10 (two
shared gold files each): `django-11138 → 13121` and `django-16032 → 11734`. These are
**not** keepers today — they are a re-examine list **if** they later become keepers,
not a current leak.

---

## Method & re-run

For every keeper × same-repo held-out id the audit computes: **(a)** file overlap of
`{keeper.final_patch ∪ keeper gold patch}` vs `{holdout gold ∪ holdout test}`;
**(b)** issue-text TF-IDF cosine (sklearn; token-Jaccard fallback); **(c)** enclosing
def/class overlap from unified-diff hunk headers, restricted to shared files.
Deterministic, CPU-only, no network/GPU. Rebuilds the 113-id holdout byte-identically
and asserts its sha256 before doing anything.

```
.venv/bin/python runs/swe_datagen_s1/leakage_audit.py
```

Artifacts: `leakage_audit_report.json` · `leakage_audit_report.md` ·
`leakage_audit_keepers_snapshot.jsonl` · this file.
