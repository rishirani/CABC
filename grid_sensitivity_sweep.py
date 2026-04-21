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

INTERIOR_NODE_CAPACITY = 100.0
BOUNDARY_COLUMN_CAPACITY = 1000.0

SOLVER = cp.SCS  # switch to cp.ECOS if preferred/available
FIG_DPI = 300

FIG_WIDTH_PER_COL = 5
FIG_HEIGHT_PER_ROW = 5

# K-sweep: demand fixed at 50, rho_bar fixed at 0.99
K_SWEEP_DEMAND = 10.0
K_SWEEP_RHO_BAR = 0.99
K_VALUES = [0.0, 1, 2, 3]

# rho_bar-sweep: demand fixed at 100, K fixed at 1
RHO_SWEEP_DEMAND = 100.0
RHO_SWEEP_K = 0.0
RHO_BAR_VALUES = [1.0, 0.6, 0.3, 0.15]


# ---------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------

def build_grid_graph(
    rows: int = GRID_ROWS,
    cols: int = GRID_COLS,
    interior_capacity: float = INTERIOR_NODE_CAPACITY,
    boundary_column_capacity: float = BOUNDARY_COLUMN_CAPACITY,
) -> Tuple[nx.Graph, Dict[Tuple[int, int], Tuple[int, int]]]:
    """
    Build an undirected grid with integer lattice positions.

    All nodes in the leftmost and rightmost columns are assigned the
    boundary-column capacity. All interior columns use the interior capacity.
    """
    G = nx.grid_2d_graph(rows, cols)

    for r, c in G.nodes():
        if c == 0 or c == cols - 1:
            G.nodes[(r, c)]["capacity"] = float(boundary_column_capacity)
        else:
            G.nodes[(r, c)]["capacity"] = float(interior_capacity)

    pos = {(r, c): (c, -r) for r, c in G.nodes()}
    return G, pos


# ---------------------------------------------------------------------
# Flow problem
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


def solve_routing_centrality_from_K(
    G: nx.Graph,
    demand: float,
    K: float,
    mode: str,
    rho_bar: float,
    solver=SOLVER,
    verbose: bool = False,
):
    """
    Solve routing using a direct K parameter rather than c_a, c_s.

    mode:
        - 'so' : system-optimal CABC
        - 'ue' : user-equilibrium CABC
    """
    f, edges, nodes, Pin, Pout, A, b, mu, source, sink = _build_flow_problem(G, demand)

    lam = Pin @ f
    rho = cp.multiply(1.0 / mu, lam)

    # rho_bar constraint applies to all nodes, including source and sink
    constraints = [
        A @ f - b == 0,
        rho >= 0,
        rho <= rho_bar,
    ]

    if mode == "so":
        # rho + K rho^2/(1-rho) == K/(1-rho) + (1-K)rho - K
        objective = cp.sum(K * cp.inv_pos(1 - rho) + (1.0 - K) * rho)
    elif mode == "ue":
        # Integral of delay / Beckmann-type potential
        objective = cp.sum(K * (-cp.log1p(-rho)) + (1.0 - K) * rho)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'so' or 'ue'.")

    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver=solver, verbose=verbose)

    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(
            f"{mode} solve failed for demand={demand}, K={K}, rho_bar={rho_bar} "
            f"with status: {problem.status}"
        )

    f_opt = np.asarray(f.value).reshape(-1)
    inflow = Pin @ f_opt
    outflow = Pout @ f_opt

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
# Plotting helpers
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

    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.Spectral,
        norm=plt.Normalize(vmin=0, vmax=1)
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized centrality", fontsize=14)
    cbar.ax.tick_params(labelsize=14)


def make_2x4_plot(
    G,
    pos,
    top_results,
    bottom_results,
    top_titles,
    bottom_titles,
    output_name,
    suptitle,
):
    fig, axes = plt.subplots(
        nrows=2,
        ncols=4,
        figsize=(4 * FIG_WIDTH_PER_COL, 2 * FIG_HEIGHT_PER_ROW),
        squeeze=False,
    )

    for col_idx in range(4):
        top_ax = axes[0, col_idx]
        top_out = top_results[col_idx]
        draw_single_panel(
            ax=top_ax,
            G=G,
            pos=pos,
            centrality=top_out["node_flow"],
            title=top_titles[col_idx],
            source=top_out["source"],
            sink=top_out["sink"],
        )

        bot_ax = axes[1, col_idx]
        bot_out = bottom_results[col_idx]
        draw_single_panel(
            ax=bot_ax,
            G=G,
            pos=pos,
            centrality=bot_out["node_flow"],
            title=bottom_titles[col_idx],
            source=bot_out["source"],
            sink=bot_out["sink"],
        )

    fig.suptitle(suptitle, fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_name, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_name}")


# ---------------------------------------------------------------------
# Main experiments
# ---------------------------------------------------------------------

def main():
    G, pos = build_grid_graph()

    results = {
        "k_sweep": {"so": [], "ue": []},
        "rho_sweep": {"so": [], "ue": []},
    }
    runtimes = {
        "k_sweep": {"so": {}, "ue": {}},
        "rho_sweep": {"so": {}, "ue": {}},
    }

    # ------------------------------------------------------------
    # K sweep
    # ------------------------------------------------------------
    print(f"Running K-sweep on {GRID_ROWS}x{GRID_COLS} grid...")
    print(
        f"Demand fixed at {K_SWEEP_DEMAND:.0f}, rho_bar fixed at {K_SWEEP_RHO_BAR:.2f}, "
        f"K in {K_VALUES}\n"
    )

    for mode, pretty_name in [("so", "SO-CABC"), ("ue", "UE-CABC")]:
        print(f"{pretty_name}:")
        for K in K_VALUES:
            print(f"  Computing {pretty_name} for K = {K:.1f} ...")
            t0 = time.perf_counter()
            out = solve_routing_centrality_from_K(
                G=G,
                demand=K_SWEEP_DEMAND,
                K=K,
                mode=mode,
                rho_bar=K_SWEEP_RHO_BAR,
                solver=SOLVER,
                verbose=False,
            )
            elapsed = time.perf_counter() - t0
            results["k_sweep"][mode].append(out)
            runtimes["k_sweep"][mode][K] = elapsed
            print(
                f"    done in {elapsed:.4f}s | objective = {out['objective_value']:.6f}"
            )
        print()

    # ------------------------------------------------------------
    # rho_bar sweep
    # ------------------------------------------------------------
    print("Running rho_bar-sweep...")
    print(
        f"Demand fixed at {RHO_SWEEP_DEMAND:.0f}, K fixed at {RHO_SWEEP_K:.1f}, "
        f"rho_bar in {RHO_BAR_VALUES}\n"
    )

    for mode, pretty_name in [("so", "SO-CABC"), ("ue", "UE-CABC")]:
        print(f"{pretty_name}:")
        for rho_bar in RHO_BAR_VALUES:
            print(f"  Computing {pretty_name} for rho_bar = {rho_bar:.1f} ...")
            t0 = time.perf_counter()
            out = solve_routing_centrality_from_K(
                G=G,
                demand=RHO_SWEEP_DEMAND,
                K=RHO_SWEEP_K,
                mode=mode,
                rho_bar=rho_bar,
                solver=SOLVER,
                verbose=False,
            )
            elapsed = time.perf_counter() - t0
            results["rho_sweep"][mode].append(out)
            runtimes["rho_sweep"][mode][rho_bar] = elapsed
            print(
                f"    done in {elapsed:.4f}s | objective = {out['objective_value']:.6f}"
            )
        print()

    # ------------------------------------------------------------
    # Plot K sweep
    # ------------------------------------------------------------
    k_top_titles = [rf"SO-CABC, $K={K:g}$" for K in K_VALUES]
    k_bottom_titles = [rf"UE-CABC, $K={K:g}$" for K in K_VALUES]

    make_2x4_plot(
        G=G,
        pos=pos,
        top_results=results["k_sweep"]["so"],
        bottom_results=results["k_sweep"]["ue"],
        top_titles=k_top_titles,
        bottom_titles=k_bottom_titles,
        output_name="grid_K_sweep_2x4_SO_UE.png",
        suptitle=(
            rf"$K$-Sweep on a ${GRID_ROWS}\times {GRID_COLS}$ Grid "
            rf"($d={int(K_SWEEP_DEMAND)}$, $\bar{{\rho}}={K_SWEEP_RHO_BAR}$)"
        ),
    )

    # ------------------------------------------------------------
    # Plot rho_bar sweep
    # ------------------------------------------------------------
    rho_top_titles = [
        rf"SO-CABC, $\bar{{\rho}}={rho_bar:.2g}$" for rho_bar in RHO_BAR_VALUES
    ]
    rho_bottom_titles = [
        rf"UE-CABC, $\bar{{\rho}}={rho_bar:.2g}$" for rho_bar in RHO_BAR_VALUES
    ]

    make_2x4_plot(
        G=G,
        pos=pos,
        top_results=results["rho_sweep"]["so"],
        bottom_results=results["rho_sweep"]["ue"],
        top_titles=rho_top_titles,
        bottom_titles=rho_bottom_titles,
        output_name="grid_rhobar_sweep_2x4_SO_UE.png",
        suptitle=(
            rf"$\bar{{\rho}}$-Sweep on a ${GRID_ROWS}\times {GRID_COLS}$ Grid "
            rf"($d={int(RHO_SWEEP_DEMAND)}$, $K={RHO_SWEEP_K:g}$)"
        ),
    )

    # ------------------------------------------------------------
    # Save summary
    # ------------------------------------------------------------
    with open("grid_sensitivity_summary_SO_UE.txt", "w") as f:
        f.write(f"{GRID_ROWS}x{GRID_COLS} Grid Sensitivity Analysis (SO and UE)\n")
        f.write("=========================================================\n\n")

        f.write("Capacities:\n")
        f.write(f"  Interior columns capacity: {INTERIOR_NODE_CAPACITY}\n")
        f.write(f"  Left/right boundary columns capacity: {BOUNDARY_COLUMN_CAPACITY}\n")
        f.write("  rho_bar constraint applies to all nodes, including source and sink\n\n")

        f.write("K-sweep\n")
        f.write("-------\n")
        f.write(
            f"Demand fixed at {K_SWEEP_DEMAND}, rho_bar fixed at {K_SWEEP_RHO_BAR}\n\n"
        )

        for mode, pretty_name in [("so", "SO-CABC"), ("ue", "UE-CABC")]:
            f.write(f"{pretty_name}\n")
            f.write("-" * len(pretty_name) + "\n")
            for K in K_VALUES:
                result = results["k_sweep"][mode][K_VALUES.index(K)]
                f.write(
                    f"K = {K:>3.1f}: runtime = {runtimes['k_sweep'][mode][K]:.6f}s, "
                    f"objective = {result['objective_value']:.6f}, "
                    f"status = {result['status']}\n"
                )
            f.write("\n")

        f.write("rho_bar-sweep\n")
        f.write("-------------\n")
        f.write(
            f"Demand fixed at {RHO_SWEEP_DEMAND}, K fixed at {RHO_SWEEP_K}\n\n"
        )

        for mode, pretty_name in [("so", "SO-CABC"), ("ue", "UE-CABC")]:
            f.write(f"{pretty_name}\n")
            f.write("-" * len(pretty_name) + "\n")
            for rho_bar in RHO_BAR_VALUES:
                result = results["rho_sweep"][mode][RHO_BAR_VALUES.index(rho_bar)]
                f.write(
                    f"rho_bar = {rho_bar:>3.1f}: runtime = {runtimes['rho_sweep'][mode][rho_bar]:.6f}s, "
                    f"objective = {result['objective_value']:.6f}, "
                    f"status = {result['status']}\n"
                )
            f.write("\n")

    print("Saved grid_sensitivity_summary_SO_UE.txt")


if __name__ == "__main__":
    main()