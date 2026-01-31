import networkx as nx
import numpy as np

def generate_network_logistical(
    SEED=4,
    REGIONS=("W", "C", "E"),
    # sizes per region
    M_PER=2,
    RD_PER=4,
    LD_PER=8,
    R_PER=20,
    # attachment params
    K_M2RD=2,
    K_RD2LD_IN=2,
    K_LD2R_IN=2,
    K_LD_LATERAL=1,
    # gateways
    N_GATE_WC_RD=1,
    N_GATE_CE_RD=1,
    MAKE_W_E_DIRECT=False,
    # capacities/supply/demand
    SUPPLY_M_CENTRAL=7500,
    CAP_M_CENTRAL=15000,
    SUPPLY_M_OTHER=3750,
    CAP_M_OTHER=5000,
    CAP_RD=7500,
    CAP_LD=3000,
    DEMAND_PER_RETAILER=500,
    CAP_RETAILER_MULT=1.5,
    # layout params
    GRID=20,
    MIN_SEP=0.050,
    MAX_TRIES_PER_NODE=4000,
    JITTER=0.85,
    # tier y-bands
    Y_M=(0.9, 1.00),
    Y_RD=(0.7, 0.8),
    Y_LD=(0.4, 0.6),
    Y_R=(0.00, 0.3),
):
    """
    Generate the 4-tier regional logistics network.
    Returns: (G, pos)
      - G is a directed graph with node attrs: layer, region, capacity, demand, supply
      - pos is a dict: node -> (x,y) in normalized [0,1]x[0,1] layout
    """
    rng = np.random.default_rng(SEED)

    # ----------------------------
    # Helpers: region bands + positions
    # ----------------------------
    def region_x_band(region):
        if region == "W":
            return (0.00, 1/3)
        if region == "C":
            return (1/3, 2/3)
        if region == "E":
            return (2/3, 1.00)
        raise ValueError("Unknown region")

    def sample_point_in_sector(ix, iy, grid=GRID, jitter=JITTER):
        cell = 1.0 / grid
        x0, y0 = ix * cell, iy * cell
        cx, cy = x0 + 0.5 * cell, y0 + 0.5 * cell
        dx = (rng.uniform(-0.4, 0.4) * cell) * jitter
        dy = (rng.uniform(-0.4, 0.4) * cell) * jitter
        return (float(cx + dx), float(cy + dy))

    def allowed_cells_for_band(region, y_band, grid=GRID):
        x_low, x_high = region_x_band(region)
        y_low, y_high = y_band

        x_cells = [i for i in range(grid)
                   if (i/grid) >= x_low - 1e-9 and ((i+1)/grid) <= x_high + 1e-9]
        y_cells = [j for j in range(grid)
                   if (j/grid) >= y_low - 1e-9 and ((j+1)/grid) <= y_high + 1e-9]
        return x_cells, y_cells

    def far_enough(pt, placed_pts, min_sep):
        px, py = pt
        for qx, qy in placed_pts:
            dx = px - qx
            dy = py - qy
            if (dx*dx + dy*dy) < (min_sep * min_sep):
                return False
        return True

    def place_nodes_in_region_minsep(nodes, region, y_band,
                                     min_sep=MIN_SEP,
                                     max_tries_per_node=MAX_TRIES_PER_NODE):
        x_cells, y_cells = allowed_cells_for_band(region, y_band, grid=GRID)
        placed_pts = []
        pos_local = {}

        for n in nodes:
            placed = False
            for _ in range(max_tries_per_node):
                ix = int(rng.choice(x_cells))
                iy = int(rng.choice(y_cells))
                pt = sample_point_in_sector(ix, iy, grid=GRID, jitter=JITTER)
                if far_enough(pt, placed_pts, min_sep):
                    pos_local[n] = pt
                    placed_pts.append(pt)
                    placed = True
                    break

            if not placed:
                raise RuntimeError(
                    f"Could not place node {n} in region={region}, y_band={y_band} "
                    f"with min_sep={min_sep}. Try reducing MIN_SEP or increasing GRID."
                )
        return pos_local

    def rng_sample(items, k):
        items = list(items)
        if k <= 0:
            return []
        k = min(k, len(items))
        idx = rng.choice(len(items), size=k, replace=False)
        return [items[i] for i in idx]

    def rng_shuffle(items):
        items = list(items)
        perm = rng.permutation(len(items))
        return [items[i] for i in perm]

    # ----------------------------
    # Build node lists (internal IDs include region)
    # ----------------------------
    manufacturers, rds, lds, retailers = [], [], [], []
    for reg in REGIONS:
        manufacturers += [f"M{reg}{i}"  for i in range(M_PER)]
        rds          += [f"RD{reg}{i}" for i in range(RD_PER)]
        lds          += [f"LD{reg}{i}" for i in range(LD_PER)]
        retailers     += [f"R{reg}{i}"  for i in range(R_PER)]

    layer_of = (
        {n: "manufacturer" for n in manufacturers} |
        {n: "rd"           for n in rds} |
        {n: "ld"           for n in lds} |
        {n: "retailer"     for n in retailers}
    )
    region_of = {
        n: next(ch for ch in n if ch in REGIONS)
        for n in manufacturers + rds + lds + retailers}
    # ----------------------------
    # Create directed graph + attach capacities/demand/supply
    # ----------------------------
    CAP_RETAILER = CAP_RETAILER_MULT * DEMAND_PER_RETAILER

    G = nx.DiGraph()
    for n in manufacturers + rds + lds + retailers:
        reg = region_of[n]
        layer = layer_of[n]

        if layer == "manufacturer":
            cap = CAP_M_CENTRAL if reg == "C" else CAP_M_OTHER
            demand = 0.0
            supply = SUPPLY_M_CENTRAL if reg == "C" else SUPPLY_M_OTHER
        elif layer == "rd":
            cap = CAP_RD
            demand = 0.0
            supply = 0.0
        elif layer == "ld":
            cap = CAP_LD
            demand = 0.0
            supply = 0.0
        elif layer == "retailer":
            cap = CAP_RETAILER
            demand = DEMAND_PER_RETAILER
            supply = 0.0
        else:
            raise ValueError("Unknown layer")

        G.add_node(
            n,
            layer=layer,
            region=reg,
            capacity=float(cap),
            demand=float(demand),
            supply=float(supply),
        )

    # ----------------------------
    # Positions with guaranteed minimum separation per (region × tier band)
    # ----------------------------
    pos = {}
    for reg in REGIONS:
        pos.update(place_nodes_in_region_minsep([n for n in manufacturers if region_of[n] == reg], reg, y_band=Y_M))
        pos.update(place_nodes_in_region_minsep([n for n in rds          if region_of[n] == reg], reg, y_band=Y_RD))
        pos.update(place_nodes_in_region_minsep([n for n in lds          if region_of[n] == reg], reg, y_band=Y_LD))
        pos.update(place_nodes_in_region_minsep([n for n in retailers     if region_of[n] == reg], reg, y_band=Y_R))

    # ----------------------------
    # Edge construction (topology randomized by SEED)
    # ----------------------------
    M_by_reg  = {reg: [n for n in manufacturers if region_of[n] == reg] for reg in REGIONS}
    RD_by_reg = {reg: [n for n in rds          if region_of[n] == reg] for reg in REGIONS}
    LD_by_reg = {reg: [n for n in lds          if region_of[n] == reg] for reg in REGIONS}
    R_by_reg  = {reg: [n for n in retailers     if region_of[n] == reg] for reg in REGIONS}

    # 1) M -> RD (manufacturer-driven sparsity)
    for reg in REGIONS:
        for m in M_by_reg[reg]:
            chosen = rng_sample(RD_by_reg[reg], K_M2RD)
            for rd in chosen:
                G.add_edge(m, rd, etype="M2RD")

    # Ensure each RD has at least 1 inbound manufacturer edge (coverage)
    for reg in REGIONS:
        for rd in RD_by_reg[reg]:
            in_ms = [u for u, _ in G.in_edges(rd)]
            if len(in_ms) == 0:
                m = rng.choice(M_by_reg[reg])
                G.add_edge(m, rd, etype="M2RD")

    # 2) RD -> LD (each LD has exactly K_RD2LD_IN inbound RDs)
    for reg in REGIONS:
        for ld in LD_by_reg[reg]:
            chosen = rng_sample(RD_by_reg[reg], K_RD2LD_IN)
            for rd in chosen:
                G.add_edge(rd, ld, etype="RD2LD")

    # 3) LD -> Retailer (each retailer has exactly K_LD2R_IN inbound LDs)
    for reg in REGIONS:
        for r in R_by_reg[reg]:
            chosen = rng_sample(LD_by_reg[reg], K_LD2R_IN)
            for ld in chosen:
                G.add_edge(ld, r, etype="LD2R")

    # 4) LD -> LD lateral transfers (each LD has exactly K_LD_LATERAL outgoing)
    for reg in REGIONS:
        if K_LD_LATERAL > 0:
            for ld in LD_by_reg[reg]:
                others = [x for x in LD_by_reg[reg] if x != ld]
                chosen = rng_sample(others, min(K_LD_LATERAL, len(others)))
                for ld2 in chosen:
                    G.add_edge(ld, ld2, etype="LD2LD_local")

    # 5) Cross-region gateways at RD layer (bidirectional pairs)
    def add_gateway_pairs_rd(regA, regB, n_pairs):
        A = rng_shuffle(RD_by_reg[regA])
        B = rng_shuffle(RD_by_reg[regB])
        pairs = list(zip(A[:n_pairs], B[:n_pairs]))
        for a, b in pairs:
            G.add_edge(a, b, etype="RD2RD_gateway")
            G.add_edge(b, a, etype="RD2RD_gateway")
        return pairs

    add_gateway_pairs_rd("W", "C", N_GATE_WC_RD)
    add_gateway_pairs_rd("C", "E", N_GATE_CE_RD)
    if MAKE_W_E_DIRECT:
        add_gateway_pairs_rd("W", "E", 1)

    return G, pos
