import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# ----------------------------
# Reproducibility (change this and BOTH topology + layout will change)
# ----------------------------
SEED = 4
rng = np.random.default_rng(SEED)

# ----------------------------
# Regions
# ----------------------------
REGIONS = ["W", "C", "E"]

# ----------------------------
# New per-region sizes (4-tier)
# ----------------------------
M_PER  = 2    # Manufacturers per region
RD_PER = 4    # Regional Distributors per region
LD_PER = 8    # Local Distributors per region
R_PER  = 20   # Retailers per region

# ----------------------------
# Edge-attachment parameters (new structure)
# ----------------------------
K_M2RD         = 2   # each Manufacturer connects to exactly this many RD nodes (in-region)
K_RD2LD_IN     = 2   # each LD has exactly this many inbound RD nodes (in-region)
K_LD2R_IN      = 2   # each Retailer has exactly this many inbound LD nodes (in-region)
K_LD_LATERAL   = 1   # each LD has exactly this many outgoing LD->LD local edges (in-region). Set 0 to disable.

# Cross-region gateways at the RD layer
N_GATE_WC_RD = 1
N_GATE_CE_RD = 1
MAKE_W_E_DIRECT = False

# ----------------------------
# Capacities and demand (units/week)
# ----------------------------
SUPPLY_M_CENTRAL = 7500
CAP_M_CENTRAL = 15000

SUPPLY_M_OTHER = 3750
CAP_M_OTHER   = 5000

CAP_RD = 7500
CAP_LD = 3000

DEMAND_PER_RETAILER = 500
CAP_RETAILER = 1.5*DEMAND_PER_RETAILER

# ----------------------------
# Layout grid on [0,1]x[0,1]
# ----------------------------
GRID = 20

# ----------------------------
# Position separation settings (visualization only)
# ----------------------------
MIN_SEP = 0.050
MAX_TRIES_PER_NODE = 4000
JITTER = 0.85

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

def sample_point_in_sector(rng, ix, iy, grid=GRID, jitter=JITTER):
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

def place_nodes_in_region_minsep(rng, nodes, region, y_band,
                                 min_sep=MIN_SEP,
                                 max_tries_per_node=MAX_TRIES_PER_NODE):
    """
    Place nodes within a (region × tier y_band) using sector sampling,
    guaranteeing min_sep between nodes in THIS band.
    """
    x_cells, y_cells = allowed_cells_for_band(region, y_band, grid=GRID)
    placed_pts = []
    pos = {}

    for n in nodes:
        placed = False
        for _ in range(max_tries_per_node):
            ix = int(rng.choice(x_cells))
            iy = int(rng.choice(y_cells))
            pt = sample_point_in_sector(rng, ix, iy, grid=GRID, jitter=JITTER)
            if far_enough(pt, placed_pts, min_sep):
                pos[n] = pt
                placed_pts.append(pt)
                placed = True
                break

        if not placed:
            raise RuntimeError(
                f"Could not place node {n} in region={region}, y_band={y_band} "
                f"with min_sep={min_sep}. Try reducing MIN_SEP or increasing GRID."
            )
    return pos

def rng_sample(rng, items, k):
    items = list(items)
    if k <= 0:
        return []
    k = min(k, len(items))
    idx = rng.choice(len(items), size=k, replace=False)
    return [items[i] for i in idx]

def rng_shuffle(rng, items):
    items = list(items)
    perm = rng.permutation(len(items))
    return [items[i] for i in perm]

# ----------------------------
# Build node lists (internal IDs include region)
# ----------------------------
manufacturers, rds, lds, retailers = [], [], [], []
for reg in REGIONS:
    manufacturers += [f"M_{reg}{i}"  for i in range(M_PER)]
    rds          += [f"RD_{reg}{i}" for i in range(RD_PER)]
    lds          += [f"LD_{reg}{i}" for i in range(LD_PER)]
    retailers     += [f"R_{reg}{i}"  for i in range(R_PER)]

layer_of = (
    {n: "manufacturer" for n in manufacturers} |
    {n: "rd"           for n in rds} |
    {n: "ld"           for n in lds} |
    {n: "retailer"     for n in retailers}
)
region_of = {n: n.split("_")[1][0] for n in manufacturers + rds + lds + retailers}

# ----------------------------
# Create directed graph + attach capacities/demand
# ----------------------------
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

    G.add_node(n, layer=layer, region=reg, capacity=float(cap), demand=float(demand), supply=float(supply))

# ----------------------------
# Display labels (plotting only)
# ----------------------------
display_label = {}
for i, n in enumerate(manufacturers, start=1):
    display_label[n] = f"M{i}"
for i, n in enumerate(rds, start=1):
    display_label[n] = f"RD{i}"
for i, n in enumerate(lds, start=1):
    display_label[n] = f"LD{i}"
for i, n in enumerate(retailers, start=1):
    display_label[n] = f"R{i}"

# ----------------------------
# Positions with guaranteed minimum separation per (region × tier band)
# 4 tiers: M (top), RD, LD, Retailer (bottom)
# ----------------------------
pos = {}
Y_M  = (0.9, 1.00)
Y_RD = (0.7, 0.8)
Y_LD = (0.4, 0.6)
Y_R  = (0.00, 0.3)

for reg in REGIONS:
    pos.update(place_nodes_in_region_minsep(rng, [n for n in manufacturers if region_of[n] == reg], reg, y_band=Y_M))
    pos.update(place_nodes_in_region_minsep(rng, [n for n in rds          if region_of[n] == reg], reg, y_band=Y_RD))
    pos.update(place_nodes_in_region_minsep(rng, [n for n in lds          if region_of[n] == reg], reg, y_band=Y_LD))
    pos.update(place_nodes_in_region_minsep(rng, [n for n in retailers     if region_of[n] == reg], reg, y_band=Y_R))

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
        chosen = rng_sample(rng, RD_by_reg[reg], K_M2RD)
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
        chosen = rng_sample(rng, RD_by_reg[reg], K_RD2LD_IN)
        for rd in chosen:
            G.add_edge(rd, ld, etype="RD2LD")

# 3) LD -> Retailer (each retailer has exactly K_LD2R_IN inbound LDs)
for reg in REGIONS:
    for r in R_by_reg[reg]:
        chosen = rng_sample(rng, LD_by_reg[reg], K_LD2R_IN)
        for ld in chosen:
            G.add_edge(ld, r, etype="LD2R")

# 4) LD -> LD lateral transfers (each LD has exactly K_LD_LATERAL outgoing)
for reg in REGIONS:
    if K_LD_LATERAL > 0:
        for ld in LD_by_reg[reg]:
            others = [x for x in LD_by_reg[reg] if x != ld]
            chosen = rng_sample(rng, others, min(K_LD_LATERAL, len(others)))
            for ld2 in chosen:
                G.add_edge(ld, ld2, etype="LD2LD_local")

# 5) Cross-region gateways at RD layer (bidirectional pairs)
def add_gateway_pairs_rd(regA, regB, n_pairs):
    A = rng_shuffle(rng, RD_by_reg[regA])
    B = rng_shuffle(rng, RD_by_reg[regB])
    pairs = list(zip(A[:n_pairs], B[:n_pairs]))
    for a, b in pairs:
        G.add_edge(a, b, etype="RD2RD_gateway")
        G.add_edge(b, a, etype="RD2RD_gateway")
    return pairs

gw_WC = add_gateway_pairs_rd("W", "C", N_GATE_WC_RD)
gw_CE = add_gateway_pairs_rd("C", "E", N_GATE_CE_RD)

gw_WE = []
if MAKE_W_E_DIRECT:
    gw_WE = add_gateway_pairs_rd("W", "E", 1)

gateway_rds = set([n for pair in (gw_WC + gw_CE + gw_WE) for n in pair])

# ----------------------------
# Plot
# ----------------------------
fig, ax = plt.subplots(figsize=(15, 8))

edges_m2rd  = [(u, v) for u, v, dat in G.edges(data=True) if dat.get("etype") == "M2RD"]
edges_rd2ld = [(u, v) for u, v, dat in G.edges(data=True) if dat.get("etype") == "RD2LD"]
edges_ld2r  = [(u, v) for u, v, dat in G.edges(data=True) if dat.get("etype") == "LD2R"]
edges_ld2ld = [(u, v) for u, v, dat in G.edges(data=True) if dat.get("etype") == "LD2LD_local"]
edges_gate  = [(u, v) for u, v, dat in G.edges(data=True) if dat.get("etype") == "RD2RD_gateway"]

nx.draw_networkx_edges(G, pos, edgelist=edges_m2rd,  alpha=0.20, arrowsize=7,  width=1.0, ax=ax)
nx.draw_networkx_edges(G, pos, edgelist=edges_rd2ld, alpha=0.20, arrowsize=7,  width=1.0, ax=ax)
nx.draw_networkx_edges(G, pos, edgelist=edges_ld2r,  alpha=0.20, arrowsize=7,  width=1.0, ax=ax)
nx.draw_networkx_edges(G, pos, edgelist=edges_ld2ld, alpha=0.18, arrowsize=7,  width=1.0, ax=ax)
nx.draw_networkx_edges(G, pos, edgelist=edges_gate,  alpha=0.70, arrowsize=7, width=1.5, ax=ax)

color_map = {"W": "tab:blue", "C": "tab:orange", "E": "tab:green"}

def draw_nodes(nodelist, shape, size):
    for reg in REGIONS:
        nodes = [n for n in nodelist if region_of[n] == reg]
        if nodes:
            nx.draw_networkx_nodes(
                G, pos, nodelist=nodes, node_shape=shape,
                node_color=color_map[reg], node_size=size, ax=ax
            )

draw_nodes(manufacturers, "s", 250)  # Manufacturer
draw_nodes(rds,           "o", 250)  # Regional Distributor
draw_nodes(lds,           "D", 250)  # Local Distributor
draw_nodes(retailers,     "^", 250)  # Retailer

# Gateway highlight (RD gateways)
nx.draw_networkx_nodes(
    G, pos, nodelist=list(gateway_rds), node_shape="o",
    node_size=300, node_color="none", edgecolors="black",
    linewidths=1, ax=ax
)

nx.draw_networkx_labels(G, pos, labels=display_label, font_size=7, font_color="black", ax=ax)

shape_legend = [
    Line2D([0], [0], marker='s', color='w', label='Manufacturer',         markerfacecolor='gray', markersize=9),
    Line2D([0], [0], marker='o', color='w', label='Regional Distributor', markerfacecolor='gray', markersize=9),
    Line2D([0], [0], marker='D', color='w', label='Local Distributor',    markerfacecolor='gray', markersize=9),
    Line2D([0], [0], marker='^', color='w', label='Retailer',             markerfacecolor='gray', markersize=9),
]
leg1 = ax.legend(handles=shape_legend, title="Node Type", loc="upper left", frameon=True)
ax.add_artist(leg1)

color_legend = [
    Line2D([0], [0], marker='o', color='w', label='West',    markerfacecolor='tab:blue',   markersize=9),
    Line2D([0], [0], marker='o', color='w', label='Central', markerfacecolor='tab:orange', markersize=9),
    Line2D([0], [0], marker='o', color='w', label='East',    markerfacecolor='tab:green',  markersize=9),
]
ax.legend(handles=color_legend, title="Region", loc="upper right", frameon=True)

ax.set_axis_off()
ax.set_title(
    f"4-Tier Regional Logistics Network (SEED={SEED}) | "
    f"M/RD/LD/R per region = {M_PER}/{RD_PER}/{LD_PER}/{R_PER} | min_sep={MIN_SEP}"
)
fig.tight_layout()
plt.show()

# ----------------------------
# Debug prints
# ----------------------------
print("Gateway pairs (RD) W<->C:", [(display_label[a], display_label[b]) for a, b in gw_WC])
print("Gateway pairs (RD) C<->E:", [(display_label[a], display_label[b]) for a, b in gw_CE])
if gw_WE:
    print("Gateway pairs (RD) W<->E:", [(display_label[a], display_label[b]) for a, b in gw_WE])

# Capacity sanity summary
tot_supply = 0.0
tot_demand = 0.0
for _, dat in G.nodes(data=True):
    if dat["layer"] == "manufacturer":
        tot_supply += dat["supply"]
    if dat["layer"] == "retailer":
        tot_demand += dat["demand"]

print(f"Total manufacturer capacity (all regions): {tot_supply:,.0f} units/week")
print(f"Total retailer demand (all regions):       {tot_demand:,.0f} units/week")
print(f"Supply/Demand ratio:                       {tot_supply/tot_demand:.2f}x")
