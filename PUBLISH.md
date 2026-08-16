# Publishing the consolidated repository

The working tree is ready. Git object writes fail through the Cowork device
bridge (`unlink: Operation not permitted`), so run these in a normal terminal
on your machine. Nothing has been staged or committed — the tree is untouched
and coherent.

## 1. Rename the GitHub repo (once, in the browser)

`github.com/rishovnag/arbnet-option-pricing` → **Settings → Repository name** →
`arbnet-densitynet-option-pricing` → Rename.

GitHub redirects the old URL permanently, so the v3 clone anyone already has
keeps working. The local remote is **already repointed** to the new name, and
it now matches the `\repourl` printed in the paper's data-availability
statement.

## 2. Clear the orphan objects left by the blocked staging

```bash
cd C:\Users\risho\arbnet
git gc --prune=now
```

(1,472 zero-byte `tmp_obj_*` files under `.git/objects`. Git ignores them, but
`gc` removes them cleanly. Harmless either way.)

## 3. Delete the scratch you approved

```bash
rmdir /s /q _to_delete
```

64 MB: 28 superseded result files, 9 stale working docs, `.pytest_cache`,
a `.DS_Store`, `main_v4_preview.pdf`, the literal-backslash junk directory,
and the stale `index.lock`.

## 4. Commit and push

```bash
git add -A
git status --short | findstr /v "data/"     # sanity-check the non-data changes
git commit -m "v4: skew clock, dispersion frontier, consolidated artifacts

DensityNet gains a skew clock m_i(T) = 1 + h(T)(m_i - 1), h(T) = 1 - e^{-beta T}:
the guarantee is preserved verbatim (the mixing law contracts toward its own
mean, so the convex order survives) while the expiry kink becomes exact and the
short-dated Var(log m) offset vanishes. On the 1,359-day NSE study this makes
DensityNet the best compliant model in volatility space as well as in price
(held-out IV RMSE 0.058 vs BS 0.063, DM -5.6; price 37.70 vs 38.99 fixed-mean).

- arbnet/models/density.py: skew_clock and fixed_V_m modes, mixture_diagnostics
- scripts/train_nse.py, run_study.py: arbnet_density_skew model + diagnostics
- scripts/vm_sweep.py: the mixing-dispersion frontier (paper Sec. 9.7.3)
- scripts/make_figures.py: regenerates Figures 3, 4, 6, 7, 8, 9
- tests/test_density.py: 34 tests (skew clock, dispersion pinning, reparam identity)
- results/: v4 artifacts replace the superseded v3 set; .gitignore whitelist updated
- README: v4 results, artifact-to-paper map, reproduction notes"
git push -u origin main
```

## 5. Zenodo (for the paper's \doiurl placeholder)

Create a release from the pushed commit, connect the repo in Zenodo, and paste
the minted DOI into `\doiurl` in `main.tex`. The repo is 460 MB (429 MB of it
bundled bhavcopy data), which is within Zenodo's 50 GB per-record limit.

## What is now in the repo

| | |
|---|---|
| `arbnet/` | the package (39 tracked files) |
| `scripts/` | 15 scripts incl. the three new ones |
| `tests/` | pytest suite, 34 density tests |
| `data/` | 1,377 files, 429 MB — the bundled NSE data |
| `results/` | exactly the 14 artifacts behind the paper's tables and figures |
| `README.md` | consolidated: results, artifact map, reproduction |
| `DATASETS.md`, `LICENSE`, `requirements.txt`, `setup.py` | unchanged |

`research-paper/` stays gitignored, as before.
