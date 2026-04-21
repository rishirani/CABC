import cvxpy as cp
import numpy as np
import scipy.sparse as sp
import time
import networkx as nx

# ------------------------------------------------------------------
# Global CABC parameters (same as Abilene)
# ------------------------------------------------------------------
c_a = 5.0
c_s = 0.5
K_CABC = (c_a**2 + c_s**2) / 2.0
rho_max = 0.9


def _build_flow_mats(G):
    """
    Build flow matrices for logistical network.

    Differences vs Abilene:
      - Graph is already directed
      - Node attributes provide:
          * supply  (positive = source)
          * demand  (positive = sink)
          * capacity
      - Exogenous balance uses the existing logistical convention:
        b_i = demand_i - supply_i
    """
    nodes = list(G.nodes())
    n_index = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)

    # -----------------------------
    # Exogenous flow vector b
    # -----------------------------
    b = np.zeros(N, dtype=float)

    for i, n in enumerate(nodes):
        data = G.nodes[n]
        supply = float(data.get("supply", 0.0))
        demand = float(data.get("demand", 0.0))
        b[i] = demand - supply

    # Sanity check: global balance
    if abs(b.sum()) > 1e-8:
        total_positive = b[b > 0].sum()
        total_negative = -b[b < 0].sum()
        raise ValueError(
            f"Total positive external balance ({total_positive:.3f}) "
            f"!= total negative external balance ({total_negative:.3f})"
        )

    # -----------------------------
    # Capacity vector μ
    # -----------------------------
    mu = np.empty(N, dtype=float)
    for i, n in enumerate(nodes):
        if "capacity" not in G.nodes[n]:
            raise KeyError(f"Node {n} missing 'capacity' attribute")
        mu[i] = float(G.nodes[n]["capacity"])
        if mu[i] <= 0:
            raise ValueError(f"Node {n} has non-positive capacity {mu[i]}")

    # -----------------------------
    # Directed edges (already directed)
    # -----------------------------
    edges = list(G.edges())
    E = len(edges)

    row_in, col_in, data_in = [], [], []
    row_out, col_out, data_out = [], [], []

    for e, (u, v) in enumerate(edges):
        iu = n_index[u]
        iv = n_index[v]
        row_out.append(iu); col_out.append(e); data_out.append(1.0)
        row_in.append(iv);  col_in.append(e);  data_in.append(1.0)

    Pin = sp.csr_matrix((data_in, (row_in, col_in)), shape=(N, E))
    Pout = sp.csr_matrix((data_out, (row_out, col_out)), shape=(N, E))
    A = Pin - Pout  # inflow - outflow

    f = cp.Variable(E, nonneg=True)

    return f, edges, nodes, Pin, Pout, A, b, mu



def SO_CABC(G, verbose=False, solver=cp.ECOS):
    """
    System-Optimal CABC for logistical networks:
        sum_i [ rho_i + K * rho_i^2/(1 - rho_i) ]

    Algebraically equivalent vectorized form:
        rho + K * rho^2/(1-rho)
      = K * inv_pos(1-rho) + (1-K) * rho - K

    The additive constant -K does not affect the optimizer, so we solve the
    fully vectorized objective
        sum_i [ K * inv_pos(1-rho_i) + (1-K) * rho_i ]
    """
    f, edges, nodes, Pin, Pout, A, b, mu = _build_flow_mats(G)

    constraints = [A @ f - b == 0]

    lam = Pin @ f
    rho = cp.multiply(1.0 / mu, lam)

    constraints += [rho >= 0, rho <= rho_max]

    term = K_CABC * cp.inv_pos(1 - rho) + (1.0 - K_CABC) * rho

    prob = cp.Problem(cp.Minimize(cp.sum(term)), constraints)
    prob.solve(solver=solver, verbose=verbose)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"SO_CABC: solver status {prob.status}")

    f_opt = np.array(f.value).reshape(-1)

    edge_flow = {e: float(f_opt[i]) for i, e in enumerate(edges)}

    infl = Pin @ f_opt
    out = Pout @ f_opt
    node_flow_vec = np.maximum(infl, out)
    node_flow = {n: float(node_flow_vec[i]) for i, n in enumerate(nodes)}

    return node_flow, edge_flow, float(prob.value)



def UE_CABC(G, verbose=False, solver=cp.ECOS):
    """
    User-Equilibrium CABC for logistical networks:
        K * (-log(1 - rho)) + rho * (1 - K)
    """
    f, edges, nodes, Pin, Pout, A, b, mu = _build_flow_mats(G)

    constraints = [A @ f - b == 0]

    lam = Pin @ f
    rho = cp.multiply(1.0 / mu, lam)

    constraints += [rho >= 0, rho <= rho_max]

    term = K_CABC * (-cp.log1p(-rho)) + rho * (1.0 - K_CABC)

    prob = cp.Problem(cp.Minimize(cp.sum(term)), constraints)
    prob.solve(solver=solver, verbose=verbose)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"UE_CABC: solver status {prob.status}")

    f_opt = np.array(f.value).reshape(-1)

    edge_flow = {e: float(f_opt[i]) for i, e in enumerate(edges)}

    infl = Pin @ f_opt
    out = Pout @ f_opt
    node_flow_vec = np.maximum(infl, out)
    node_flow = {n: float(node_flow_vec[i]) for i, n in enumerate(nodes)}

    return node_flow, edge_flow, float(prob.value)



def _build_node_split_digraph(G):
    """Build a directed node-split graph for node-capacitated max-flow."""
    H = nx.DiGraph()
    big_cap = 0.0

    for v, data in G.nodes(data=True):
        cap_v = float(data.get("capacity", 1.0))
        if cap_v <= 0:
            raise ValueError(f"Node {v} has non-positive capacity {cap_v}.")
        big_cap += cap_v
        H.add_node((v, "in"))
        H.add_node((v, "out"))
        H.add_edge((v, "in"), (v, "out"), capacity=cap_v)

    big_cap = max(big_cap, 1.0)

    for u, v in G.edges():
        H.add_edge((u, "out"), (v, "in"), capacity=big_cap)

    return H, big_cap



def flow_betweenness_centrality(
    G,
    normalized=False,
    flow_func=nx.algorithms.flow.preflow_push,
    progress=True,
    print_every=25,
):
    """
    Exact full all-pairs node-capacitated flow betweenness for directed graphs.

    This keeps a single persistent node-split graph in memory and only
    adds/removes the two pair-specific source/sink edges per solve.
    """
    nodes = list(G.nodes())
    centrality = {v: 0.0 for v in nodes}

    H, big_cap = _build_node_split_digraph(G)
    src = ("__super_source__",)
    snk = ("__super_sink__",)
    H.add_node(src)
    H.add_node(snk)

    total_pairs = len(nodes) * (len(nodes) - 1)
    if progress:
        print(f"[Flow Betweenness] Starting exact all-pairs computation on {len(nodes)} nodes.")
        print(f"[Flow Betweenness] Total ordered source-target pairs: {total_pairs}")
        t0 = time.time()
    else:
        t0 = None

    idx = 0
    for s in nodes:
        s_in = (s, "in")
        for t in nodes:
            if s == t:
                continue
            t_out = (t, "out")
            idx += 1

            H.add_edge(src, s_in, capacity=big_cap)
            H.add_edge(t_out, snk, capacity=big_cap)

            try:
                max_flow_value, flow_dict = nx.maximum_flow(
                    H, src, snk, capacity="capacity", flow_func=flow_func
                )
            except nx.NetworkXUnbounded:
                max_flow_value = 0.0
                flow_dict = {}
            finally:
                if H.has_edge(src, s_in):
                    H.remove_edge(src, s_in)
                if H.has_edge(t_out, snk):
                    H.remove_edge(t_out, snk)

            if max_flow_value > 1e-12:
                inv_flow = 1.0 / float(max_flow_value)
                for v in nodes:
                    if v == s or v == t:
                        continue
                    transit_flow = flow_dict.get((v, "in"), {}).get((v, "out"), 0.0)
                    centrality[v] += float(transit_flow) * inv_flow

            if progress and (idx == 1 or idx % print_every == 0 or idx == total_pairs):
                elapsed = time.time() - t0
                avg = elapsed / idx
                remaining = total_pairs - idx
                eta = remaining * avg
                pct = 100.0 * idx / total_pairs
                print(
                    f"[Flow Betweenness] {idx}/{total_pairs} pairs "
                    f"({pct:.1f}%) | elapsed {elapsed:.1f}s | "
                    f"avg/pair {avg:.3f}s | ETA {eta:.1f}s"
                )

    if normalized and total_pairs > 0:
        scale = 1.0 / total_pairs
        centrality = {v: scale * score for v, score in centrality.items()}

    if progress:
        total_elapsed = time.time() - t0
        print(f"[Flow Betweenness] Done in {total_elapsed:.1f}s")

    return centrality
