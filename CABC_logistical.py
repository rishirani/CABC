import cvxpy as cp
import numpy as np
import scipy.sparse as sp

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
      - Exogenous balance: b_i = supply_i - demand_i
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
        raise ValueError(
            f"Total supply ({b[b>0].sum():.3f}) "
            f"!= total demand ({-b[b<0].sum():.3f})"
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
    """
    eps = 1e-6

    f, edges, nodes, Pin, Pout, A, b, mu = _build_flow_mats(G)

    # Flow balance
    constraints = [A @ f - b == 0]

    lam = Pin @ f
    rho = cp.multiply(1.0 / mu, lam)

    constraints += [rho >= 0, rho <= rho_max]

    # DCP-safe per-node objective
    obj_terms = [
        rho[i] + K_CABC * cp.quad_over_lin(rho[i], 1 - rho[i])
        for i in range(len(nodes))
    ]

    prob = cp.Problem(cp.Minimize(cp.sum(obj_terms)), constraints)
    prob.solve(solver=solver, verbose=verbose)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"SO_CABC: solver status {prob.status}")

    f_opt = np.array(f.value).reshape(-1)

    edge_flow = {e: float(f_opt[i]) for i, e in enumerate(edges)}

    infl = Pin @ f_opt
    out  = Pout @ f_opt
    node_flow_vec = np.maximum(infl, out)
    node_flow = {n: float(node_flow_vec[i]) for i, n in enumerate(nodes)}

    return node_flow, edge_flow, float(prob.value)


def UE_CABC(G, verbose=False, solver=cp.ECOS):
    """
    User-Equilibrium CABC for logistical networks:
        K * (-log(1 - rho)) + rho * (1 - K)
    """
    eps = 1e-6

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
    out  = Pout @ f_opt
    node_flow_vec = np.maximum(infl, out)
    node_flow = {n: float(node_flow_vec[i]) for i, n in enumerate(nodes)}

    return node_flow, edge_flow, float(prob.value)
