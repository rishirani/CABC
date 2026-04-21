import time
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

try:
    import cvxpy as cp
except ImportError as exc:
    raise ImportError(
        "This script requires cvxpy. Install it with `pip install cvxpy`."
    ) from exc


# ---------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------
GRID_ROWS = 7
GRID_COLS = 7
NODE_CAPACITY = 100.0
SOURCE_SINK_CAPACITY = 500.0
RHO_BAR = 0.99
DEMANDS = [10.0, 50.0, 100.0]

LINEAR_CA = 0.0
LINEAR_CS = 0.0
CABC_CA = 1.0
CABC_CS = 1.0

SOLVER = cp.SCS  # broadly available; switch to cp.ECOS if preferred
FIG_DPI = 300

# Dynamic figure scaling
FIG_WIDTH_PER_COL = 5
FIG_HEIGHT_PER_ROW = 5


# ---------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------

def build_grid_graph(
    rows: int = GRID_ROWS,
    cols: int = GRID_COLS,
    capacity: float = NODE_CAPACITY,
    source_sink_capacity: float = SOURCE_SINK_CAPACITY,
) -> Tuple[nx.Graph, Dict[Tuple[int, int], Tuple[int, int]]]:
    """Build an undirected grid with integer lattice positions."""
    G = nx.grid_2d_graph(rows, cols)

    source = (rows // 2, 0)
    sink = (rows // 2, cols - 1)

    for node in G.nodes():
        G.nodes[node]["capacity"] = float(capacity)

    G.nodes[source]["capacity"] = float(source_sink_capacity)
    G.nodes[sink]["capacity"] = float(source_sink_capacity)

    # Positions laid out like a matrix: left->right, top->bottom.
    pos = {(r, c): (c, -r) for r, c in G.nodes()}
    return G, pos


# ---------------------------------------------------------------------
# CABC / linear routing solver
# ---------------------------------------------------------------------

def _build_flow_problem(G: nx.Graph, demand: float):
    """
    Builds the vectorized one-commodity flow problem.

    Source = center node on leftmost column
    Sink   = center node on rightmost column
    """
    nodes = list(G.nodes())
    n_index = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)

    edges = []
    for u, v in G.edges():
        edges.append((u, v))
        edges.append((v, u))
    E = len(edges)

    rows = max(r for r, _ in G.nodes()) + 1
    cols = max(c for _, c in G.nodes()) + 1
    source = (rows // 2, 0)
    sink = (rows // 2, cols - 1)

    b = np.zeros(N, dtype=float)
    b[n_index[source]] = -float(demand)
    b[n_index[sink]] = +float(demand)

    mu = np.array([float(G.nodes[n]["capacity"]) for n in nodes], dtype=float)

    Pin = np.zeros((N, E), dtype=float)
    Pout = np.zeros((N, E), dtype=float)
    for e_idx, (u, v) in enumerate(edges):
        Pout[n_index[u], e_idx] = 1.0
        Pin[n_index[v], e_idx] = 1.0

    A = Pin - Pout
    f = cp.Variable(E, nonneg=True)

    return f, edges, nodes, Pin, Pout, A, b, mu, source, sink


def solve_routing_centrality(
    G: nx.Graph,
    demand: float,
    c_a: float,
    c_s: float,
    mode: str,
    rho_bar: float = RHO_BAR,
    solver=SOLVER,
    verbose: bool = False,
):
    """
    mode:
        - 'linear' : linear minimum-cost routing (K = 0)
        - 'so'     : system-optimal CABC
        - 'ue'     : user-equilibrium CABC
    """
    f, edges, nodes, Pin, Pout, A, b, mu, source, sink = _build_flow_problem(G, demand)

    lam = Pin @ f
    rho = cp.multiply(1.0 / mu, lam)
    K = 0.5 * (c_a**2 + c_s**2)

    constraints = [
        A @ f - b == 0,
        rho >= 0,
        rho <= rho_bar,
    ]

    if mode == "linear":
        # With equal capacities this reduces to shortest-path-style linear routing.
        objective = cp.sum(rho)
    elif mode == "so":
        # rho + K rho^2/(1-rho) == K/(1-rho) + (1-K)rho - K
        objective = cp.sum(K * cp.inv_pos(1 - rho) + (1.0 - K) * rho)
    elif mode == "ue":
        # Integral of delay / Beckmann-type potential.
        objective = cp.sum(K * (-cp.log1p(-rho)) + (1.0 - K) * rho)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'linear', 'so', or 'ue'.")

    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver=solver, verbose=verbose)

    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"{mode} solve failed with status: {problem.status}")

    f_opt = np.asarray(f.value).reshape(-1)
    inflow = Pin @ f_opt
    outflow = Pout @ f_opt

    # Internal/transit importance proxy used in your other scripts.
    node_flow_vec = 0.5 * (inflow + outflow)
    node_flow = {n: float(node_flow_vec[i]) for i, n in enumerate(nodes)}

    edge_flow = {edges[i]: float(f_opt[i]) for i in range(len(edges))}
    utilization = {n: float(inflow[i] / mu[i]) for i, n in enumerate(nodes)}

    return {
        "node_flow": node_flow,
        "edge_flow": edge_flow,
        "utilization": utilization,
        "objective_value": float(problem.value),
        "status": problem.status,
        "source": source,
        "sink": sink,
    }


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def normalize_dict(values: Dict, eps: float = 1e-12) -> Dict:
    arr = np.array(list(values.values()), dtype=float)
    vmax = float(np.max(arr))
    if vmax <= eps:
        return {k: 0.0 for k in values}
    return {k: float(v) / vmax for k, v in values.items()}


def draw_single_panel(ax, G, pos, centrality, title, source, sink):
    cent_norm = normalize_dict(centrality)
    node_vals = [cent_norm[n] for n in G.nodes()]

    nx.draw_networkx_edges(G, pos=pos, ax=ax, width=1.25, alpha=0.50)
    nx.draw_networkx_nodes(
        G,
        pos=pos,
        ax=ax,
        node_color=node_vals,
        cmap=plt.cm.Spectral,
        vmin=0,
        vmax=1,
        node_size=700,
        edgecolors="k",
        linewidths=0.8,
    )

    labels = {n: f"{cent_norm[n]:.2f}" for n in G.nodes()}
    nx.draw_networkx_labels(G, pos=pos, labels=labels, font_size=12, ax=ax)

    # Highlight source and sink.
    nx.draw_networkx_nodes(
        G,
        pos=pos,
        nodelist=[source],
        node_size=760,
        node_color="none",
        edgecolors="lime",
        linewidths=2.5,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        G,
        pos=pos,
        nodelist=[sink],
        node_size=760,
        node_color="none",
        edgecolors="cyan",
        linewidths=2.5,
        ax=ax,
    )

    ax.set_title(title, fontsize=14)
    ax.set_axis_off()

    sm = plt.cm.ScalarMappable(cmap=plt.cm.Spectral, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized centrality", fontsize=14)
    cbar.ax.tick_params(labelsize=14)


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------

def main():
    G, pos = build_grid_graph()

    mode_specs = [
        ("linear", LINEAR_CA, LINEAR_CS, "Linear Routing"),
        ("so", CABC_CA, CABC_CS, "SO-CABC"),
        ("ue", CABC_CA, CABC_CS, "UE-CABC"),
    ]

    results = {}
    runtimes = {}

    print(f"Running {GRID_ROWS}x{GRID_COLS} grid traffic-regime experiment...\n")
    for mode, c_a, c_s, pretty_name in mode_specs:
        results[mode] = {}
        runtimes[mode] = {}
        for demand in DEMANDS:
            print(f"Computing {pretty_name} for demand = {demand:.0f} ...")
            t0 = time.perf_counter()
            out = solve_routing_centrality(
                G=G,
                demand=demand,
                c_a=c_a,
                c_s=c_s,
                mode=mode,
                rho_bar=RHO_BAR,
                solver=SOLVER,
                verbose=False,
            )
            elapsed = time.perf_counter() - t0
            results[mode][demand] = out
            runtimes[mode][demand] = elapsed
            print(
                f"  done in {elapsed:.4f}s | objective = {out['objective_value']:.6f}"
            )

    # Dynamic subplot layout
    nrows = len(mode_specs)
    ncols = len(DEMANDS)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(FIG_WIDTH_PER_COL * ncols, FIG_HEIGHT_PER_ROW * nrows),
        squeeze=False,
    )

    for row_idx, (mode, _, _, pretty_name) in enumerate(mode_specs):
        for col_idx, demand in enumerate(DEMANDS):
            ax = axes[row_idx, col_idx]
            out = results[mode][demand]
            draw_single_panel(
                ax=ax,
                G=G,
                pos=pos,
                centrality=out["node_flow"],
                title=f"{pretty_name}, demand={int(demand)}",
                source=out["source"],
                sink=out["sink"],
            )

    fig.suptitle(
        f"Traffic Regime Analysis on a {GRID_ROWS}x{GRID_COLS} Grid",
        fontsize=15,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    output_name = f"grid_traffic_regime_{nrows}x{ncols}.png"
    fig.savefig(output_name, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {output_name}")

    # Save a compact runtime / objective summary.
    with open("grid_traffic_regime_summary.txt", "w") as f:
        f.write(f"{GRID_ROWS}x{GRID_COLS} Grid Traffic Regime Analysis\n")
        f.write("================================\n")
        f.write(f"Source node: {(GRID_ROWS // 2, 0)}\n")
        f.write(f"Sink node: {(GRID_ROWS // 2, GRID_COLS - 1)}\n")
        f.write(f"Interior node capacity: {NODE_CAPACITY}\n")
        f.write(f"Source/sink capacity: {SOURCE_SINK_CAPACITY}\n")
        f.write(f"rho_bar: {RHO_BAR}\n\n")

        for mode, _, _, pretty_name in mode_specs:
            f.write(f"{pretty_name}\n")
            f.write("-" * len(pretty_name) + "\n")
            for demand in DEMANDS:
                out = results[mode][demand]
                f.write(
                    f"Demand {int(demand):>3d}: runtime = {runtimes[mode][demand]:.6f}s, "
                    f"objective = {out['objective_value']:.6f}, status = {out['status']}\n"
                )
            f.write("\n")

    print("Saved grid_traffic_regime_summary.txt")


if __name__ == "__main__":
    main()