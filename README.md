# Congestion-Adaptive Betweenness Centrality (CABC) — Repository

This repository contains Python code to generate synthetic network topologies (an Abilene-like Internet model and a configurable logistical model), compute traditional graph centralities, and compute congestion-aware centralities using the CABC models (System-Optimal and User-Equilibrium variants) implemented with CVXPY.

This codebase implements the simulation and analysis used in the IEEE Transactions on Network Science and Engineering article submission titled
"Congestion-Adaptive Betweenness Centrality via Queue-Theoretic Routing Dynamics".

## Contents

- `generate_network_abilene.py` — builds a deterministic, Abilene-like topology with clients, servers, and PoP routers and assigns capacities and layout positions.
- `CABC_abilene.py` — implements the core CABC algorithms: `SO_CABC` (system-optimal) and `UE_CABC` (user-equilibrium) using CVXPY. Also contains flow-matrix construction helpers.
- `main_abilene.py` — high-level driver that generates the Abilene graph, computes traditional centralities (degree, betweenness, closeness, harmonic) and the CABC centralities, and writes per-centrality plots plus combined figures (PNG files).
- `generate_network_logistical.py`, `CABC_logistical.py`, `main_logistical.py`, `draw_logistical.py` — analogous scripts for the logistical network model and plotting (same overall flow; see file headers for differences).
- `test.py`, `test_problem.py` — small scripts / quick checks (useful for experiments or for extending tests).
- A number of PNG artifacts are included (example outputs), e.g. `topology.png`, `all_centralities_with_CABC.png`, and per-centrality images.

## Requirements

- Python 3.8+ (tested with Python 3.8–3.11)
- numpy
- networkx
- matplotlib
- cvxpy
- scipy
- ecos (or another CVXPY-compatible solver such as SCS)

A `requirements.txt` is included in this repository for convenience. To create an isolated environment and install the dependencies, run:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

If you prefer to install packages individually, you can also run:

```bash
pip install numpy networkx matplotlib cvxpy scipy ecos
```

## Quick start — Abilene workflow

1. Activate your Python virtualenv (see above).
2. From the repository root run:

```bash
python3 main_abilene.py
```

3. Outputs are saved in the repository root, examples include:

- `topology.png` — visual layout of the Abilene topology
- `Degree_centrality.png`, `Betweenness_centrality.png`, `Closeness_centrality.png`, `Harmonic_centrality.png` — per-centrality visualizations
- `SO_CABC_centrality.png`, `UE_CABC_centrality.png` — congestion-adaptive centralities
- `all_centralities_with_CABC.png` — combined subplot overview

The script prints or raises errors from CVXPY if the solver fails to find an optimal solution.

## Quick start — Logistical workflow

To run the logistical model and plotting (analogous to the Abilene flow):

```bash
python3 main_logistical.py
```

Check the generated PNG files (filenames include `_logistical` where applicable).

## Notes on CVXPY / Solvers

- The CABC implementations use CVXPY with `ECOS` by default. If `ECOS` is not available or you prefer another solver (e.g., `SCS`, `OSQP`, or commercial solvers), modify the `solver=` argument in the `SO_CABC` / `UE_CABC` calls or install a recommended solver.
- Large graphs or very tight capacity constraints can make the convex programs harder to solve or cause numerical issues; adjust `client_rate_mbps` and capacity assignments if you encounter infeasible or slow solves.

## Files and where to look to modify behavior

- To change topology parameters (counts, seed, capacities): edit `generate_network_abilene.py` or `generate_network_logistical.py`.
- To tweak CABC model parameters (K, cost constants, rho_max): edit `CABC_abilene.py` or `CABC_logistical.py` (top of file contains `c_a`, `c_s`, and `rho_max`).
- To change plotting styles, colormaps, export resolution, or which centralities are shown: edit `main_abilene.py` / `main_logistical.py`.

## Tests / Quick checks

- `test.py` and `test_problem.py` are small scripts in the repo. Read their headers to see what they run; they can be useful starting points for unit tests or examples.

## Reproducibility and assumptions

- The Abilene generator uses deterministic random seeds (`RANDOM_SEED = 3`) for reproducible topology builds. If you want randomized runs, change or remove the seed in `generate_network_abilene.py` or the logistical equivalent.
- Node capacity units in the code are Mbps; client/server rates are expressed in Mbps.

## Recommended next steps / improvements

- Add a `requirements.txt` or `pyproject.toml` for reproducible installs.
- Add small unit tests for the flow-matrix builder (`_build_flow_mats`) and CVXPY objective construction with mocked small graphs.
- Add a CLI wrapper (argparse) to the `main_*.py` drivers to easily change rates, output folder, and which plots to produce.

## License & Contact

This repository does not include a license file. For questions about the code, open an issue or contact the repository owner.

---


## Citation

If you use this code in published work, please cite the associated IEEE Transactions on Network Science and Engineering article submission (currently under review)

"Congestion-Adaptive Betweenness Centrality via Queue-Theoretic Routing Dynamics"

Suggested BibTeX (will fill in year, DOI when available):

```bibtex
@article{yourkey2026cabc,
	title = {Congestion-Adaptive Betweenness Centrality via Queue-Theoretic Routing Dynamics},
	author = {Rishi Rani, Massimo Franceschetti},
	journal = {IEEE Transactions on Network Science and Engineering},
	year = {2026},
	doi = {<will insert DOI when available>}
}
```

