import cvxpy as cp
import numpy as np
import scipy.sparse as sp

# ------------------------------------------------------------------
# Global CABC parameters
# ------------------------------------------------------------------
c_a = 5.0
c_s = 0.5
K_CABC = (c_a**2 + c_s**2) / 2.0  # (c_a^2 + c_s^2)/2
rho_max = 0.9


def _build_flow_mats(G, client_rate_mbps):
    """
    Vectorized build:
      - directed edge list (u->v for each undirected edge, both directions)
      - sparse Pin/Pout so that Pin@f = inflow, Pout@f = outflow
      - A = Pin - Pout gives (inflow - outflow)
      - b is exogenous demand (clients negative, servers positive, routers 0)
      - mu is capacity vector aligned with node ordering
    """
    nodes = list(G.nodes())
    n_index = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)

    # Identify node types
    clients = [n for n, d in G.nodes(data=True) if d.get("type") == "client"]
    servers = [n for n, d in G.nodes(data=True) if d.get("type") == "server"]
    routers = [n for n, d in G.nodes(data=True)
               if d.get("type") in ["client_router", "server_router", "backbone_router"]]

    if not clients or not servers:
        raise ValueError("Graph must have clients and servers.")

    # Exogenous demand vector b (aligned with nodes order)
    demand = {n: 0.0 for n in nodes}
    for c in clients:
        demand[c] = -float(client_rate_mbps)
    per_server = float(client_rate_mbps) * len(clients) / len(servers)
    for s in servers:
        demand[s] = +per_server

    b = np.array([demand[n] for n in nodes], dtype=float)

    # Capacity vector mu (aligned with nodes order)
    mu = np.empty(N, dtype=float)
    for i, n in enumerate(nodes):
        if "capacity" not in G.nodes[n]:
            raise KeyError(f"Node {n} is missing 'capacity' attribute.")
        mu[i] = float(G.nodes[n]["capacity"])
        if mu[i] <= 0:
            raise ValueError(f"Node {n} has non-positive capacity: {mu[i]}")

    # Optional consistency check: routers should have zero demand
    routers_set = set(routers)
    routers_mask = np.array([n in routers_set for n in nodes], dtype=bool)
    if np.any(np.abs(b[routers_mask]) > 1e-12):
        raise ValueError("Routers should have zero exogenous demand, but b is nonzero on some routers.")

    # Directed edges
    edges = []
    for u, v in G.edges():
        edges.append((u, v))
        edges.append((v, u))
    E = len(edges)

    # Build sparse Pin, Pout
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


def SO_CABC(G, client_rate_mbps=50.0, verbose=False, solver=cp.ECOS):
    """
    System-Optimal CABC:
        sum_i [ rho_i + K * rho_i^2/(1 - rho_i) ]

    DCP-safe implementation:
      - Vectorize constraints (A@f == b, rho bounds)
      - Objective uses quad_over_lin per node because denom must be scalar in CVXPY.
    """
    eps = 1e-6

    f, edges, nodes, Pin, Pout, A, b, mu = _build_flow_mats(G, client_rate_mbps)

    # Vector flow balance
    constraints = [A @ f - b == 0]

    lam = Pin @ f                      # inflow vector (N,)
    rho = cp.multiply(1.0 / mu, lam)   # utilization vector (N,)

    # Domain: 0 <= rho <= 1-eps
    constraints += [rho >= 0, rho <= rho_max]

    # DCP-safe objective: sum_i rho_i + K * quad_over_lin(rho_i, 1-rho_i)
    # Need scalar denom -> per-component loop (cheap: N ~ 171 for Abilene build)
    obj_terms = []
    for i in range(len(nodes)):
        obj_terms.append(rho[i] + K_CABC * cp.quad_over_lin(rho[i], 1 - rho[i]))

    prob = cp.Problem(cp.Minimize(cp.sum(obj_terms)), constraints)
    prob.solve(solver=solver, verbose=verbose)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"SO_CABC: solver status {prob.status}")

    f_opt = np.array(f.value).reshape(-1)

    edge_flow = {e: float(f_opt[i]) for i, e in enumerate(edges)}

    infl_val = (Pin @ f_opt)
    out_val  = (Pout @ f_opt)
    node_flow_vec = 0.5 * (infl_val + out_val)
    node_flow = {n: float(node_flow_vec[i]) for i, n in enumerate(nodes)}

    return node_flow, edge_flow, float(prob.value)


def UE_CABC(G, client_rate_mbps=50.0, verbose=False, solver=cp.ECOS):
    """
    User-Equilibrium CABC:
        term_i = K * (-log(1-rho_i)) + rho_i*(1-K)
    """
    eps = 1e-6

    f, edges, nodes, Pin, Pout, A, b, mu = _build_flow_mats(G, client_rate_mbps)

    constraints = [A @ f - b == 0]

    lam = Pin @ f
    rho = cp.multiply(1.0 / mu, lam)

    constraints += [rho >= 0, rho <= rho_max]

    # Stable: -log(1-rho) = -log1p(-rho)
    term = K_CABC * (-cp.log1p(-rho)) + rho * (1.0 - K_CABC)

    prob = cp.Problem(cp.Minimize(cp.sum(term)), constraints)
    prob.solve(solver=solver, verbose=verbose)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"UE_CABC: solver status {prob.status}")

    f_opt = np.array(f.value).reshape(-1)

    edge_flow = {e: float(f_opt[i]) for i, e in enumerate(edges)}

    infl_val = (Pin @ f_opt)
    out_val  = (Pout @ f_opt)
    node_flow_vec = 0.5 * (infl_val + out_val)
    node_flow = {n: float(node_flow_vec[i]) for i, n in enumerate(nodes)}

    return node_flow, edge_flow, float(prob.value)
