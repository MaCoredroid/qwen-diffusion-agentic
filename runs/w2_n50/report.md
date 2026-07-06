# W2 N=50 stock-AR vs diffusion — resolve@1 (paired)

> **STATUS: RUN NOT COMPLETE — no verdict.** Insufficient data (arms not fully scored: ar, diffusion). All `0/50` cells below mean *not run*, NOT *failed*. Launch: `setsid bash runs/w2_n50/run_all.sh &`. See ANOMALIES at bottom.

pool_sha256 `fe1973937dfb500b…`  N=50  scoring: AR=MISSING diffusion=MISSING

## PRIMARY — paired resolve@1 McNemar
- AR resolved: **0/50**   diffusion resolved: **0/50**
- both=0  AR-only(b)=0  diffusion-only(c)=0  neither=50
- net (diffusion−AR) = **+0**   McNemar exact 2-sided p = **1.0000**
- **PARITY CLAIM (|net|<=2 AND p>=0.05): INSUFFICIENT DATA**

## SECONDARY
### Throughput (episodes / GPU-h at run concurrency)
- ar: (timing missing)
- diffusion: (timing missing)
### Tokens + loop-halt covariates
| arm | eps | resolved | input_tok | output_tok | loop_halts | budget | timeouts | empty | med_turns |
|---|---|---|---|---|---|---|---|---|---|
| ar | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | None |
| diffusion | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | None |
### Loop-halt resolve split (covariate, pre-registered)
- ar: post-resolve halts=0  pre-resolve halts=0
- diffusion: post-resolve halts=0  pre-resolve halts=0
### Per-repo resolved (arm: resolved/total)
| repo | AR | diffusion |
|---|---|---|
| astropy/astropy | 0/2 | 0/2 |
| django/django | 0/22 | 0/22 |
| matplotlib/matplotlib | 0/3 | 0/3 |
| mwaskom/seaborn | 0/1 | 0/1 |
| psf/requests | 0/1 | 0/1 |
| pydata/xarray | 0/2 | 0/2 |
| pylint-dev/pylint | 0/1 | 0/1 |
| pytest-dev/pytest | 0/2 | 0/2 |
| scikit-learn/scikit-learn | 0/3 | 0/3 |
| sphinx-doc/sphinx | 0/5 | 0/5 |
| sympy/sympy | 0/8 | 0/8 |

## ANOMALIES
- ar: 50 episodes MISSING metadata: ['astropy__astropy-14182', 'astropy__astropy-14539', 'django__django-11333', 'django__django-11400', 'django__django-11734', 'django__django-12713']
- ar: only 0/50 episodes present
- ar: scoring report MISSING
- diffusion: 50 episodes MISSING metadata: ['astropy__astropy-14182', 'astropy__astropy-14539', 'django__django-11333', 'django__django-11400', 'django__django-11734', 'django__django-12713']
- diffusion: only 0/50 episodes present
- diffusion: scoring report MISSING
