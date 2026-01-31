import networkx as nx
import random
import math
import numpy as np

RANDOM_SEED = 3
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# -----------------------------
# Parameters
# -----------------------------
N_clients = 130
N_servers = 8
Backbone_Scale = 0.9

# How many client routers per client PoP
CR_PER_POP_MIN, CR_PER_POP_MAX = 2, 2
# How many clients per client router
CLIENTS_PER_CR_MIN, CLIENTS_PER_CR_MAX = 4, 10

# How many server routers per server PoP
SR_PER_POP_MIN, SR_PER_POP_MAX = 1, 2
# How many servers per server router
SERVERS_PER_SR_MIN, SERVERS_PER_SR_MAX = 2, 4


# -----------------------------
# Helper: allocate counts per bin
# -----------------------------
def allocate_counts(total, n_bins, min_val, max_val, rng):
    """
    Randomly allocate 'total' items into 'n_bins' bins,
    each between [min_val, max_val], inclusive.
    """
    if n_bins * min_val > total or n_bins * max_val < total:
        raise ValueError("Cannot allocate with given constraints")

    counts = [min_val] * n_bins
    remaining = total - n_bins * min_val
    indices = list(range(n_bins))

    while remaining > 0:
        rng.shuffle(indices)
        for i in indices:
            if remaining <= 0:
                break
            room = max_val - counts[i]
            if room <= 0:
                continue
            add = rng.randint(1, min(room, remaining))
            counts[i] += add
            remaining -= add

    return counts


# -----------------------------
# Graph generation
# -----------------------------
def generate_graph():
    rng = random.Random(RANDOM_SEED)

        # -------------------------
    # Hardcoded Abilene backbone (1–11 indexing)
    # -------------------------

    abilene_nodes = {
        1: {"city": "New York",       "lat": 40, "lon": -74},
        2: {"city": "Chicago",        "lat": 41, "lon": -83},
        3: {"city": "Washington DC",  "lat": 38, "lon": -77},
        4: {"city": "Seattle",        "lat": 42, "lon": -122},
        5: {"city": "Sunnyvale",      "lat": 37, "lon": -122},
        6: {"city": "Los Angeles",    "lat": 32, "lon": -114},
        7: {"city": "Denver",         "lat": 39, "lon": -108},
        8: {"city": "Kansas City",    "lat": 39, "lon": -100},
        9: {"city": "Houston",        "lat": 31, "lon": -95},
        10: {"city": "Atlanta",       "lat": 33, "lon": -84},
        11: {"city": "Indianapolis",  "lat": 39, "lon": -90},
    }

    # Edges rewritten to match new 1–11 numbering
    abilene_edges = [
        (1, 2),
        (1, 3),
        (2, 11),
        (3, 10),
        (4, 5),
        (4, 7),
        (5, 6),
        (5, 7),
        (6, 9),
        (7, 8),
        (8, 9),
        (8, 11),
        (9, 10),
        (10, 11),
    ]

    # Sort keys to ensure deterministic ordering
    raw_nodes_sorted = sorted(abilene_nodes.keys())  # [1..11]

    # Assign A,B,C,... naming
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if len(raw_nodes_sorted) > len(letters):
        raise ValueError("Not enough letters for backbone nodes")

    id_to_pop = {
        node_id: letters[i]  # 1→A, 2→B, ...
        for i, node_id in enumerate(raw_nodes_sorted)
    }

    backbone_nodes = list(id_to_pop.values())

    # Create the graph
    G = nx.Graph()
    pop_latlon = {}

    for node_id in raw_nodes_sorted:
        pop = id_to_pop[node_id]
        data = abilene_nodes[node_id]
        city = data["city"]
        lat = data["lat"]
        lon = data["lon"]

        G.add_node(pop, type="backbone_router", city=city)
        pop_latlon[pop] = (lon, lat)

    # Add edges with new naming
    for u, v in abilene_edges:
        G.add_edge(id_to_pop[u], id_to_pop[v])


    # -------------------------
    # Choose client vs server POPs
    # -------------------------
    pops = backbone_nodes.copy()
    rng.shuffle(pops)
    client_pops = pops[:9]
    server_pops = pops[9:]  # 2 PoPs

    # Per-PoP index for PoP routers (client+server routers)
    router_index_per_pop = {pop: 0 for pop in backbone_nodes}
    # Per-router index for leaves (clients+servers)
    leaf_index_per_router = {}

    # -------------------------
    # Build client routers + clients
    # -------------------------
    # First decide how many CR per client pop
    cr_per_pop = {}
    total_cr = 0
    for pop in client_pops:
        n_cr = rng.randint(CR_PER_POP_MIN, CR_PER_POP_MAX)
        cr_per_pop[pop] = n_cr
        total_cr += n_cr

    # Allocate N_clients across all CRs with per-CR bounds
    client_counts_per_cr = allocate_counts(
        total=N_clients,
        n_bins=total_cr,
        min_val=CLIENTS_PER_CR_MIN,
        max_val=CLIENTS_PER_CR_MAX,
        rng=rng,
    )

    client_idx = 0
    cr_idx = 0
    client_routers = []
    clients = []

    # For plotting/layout bookkeeping: which nodes hang off which PoP
    client_tree_nodes = {pop: set([pop]) for pop in client_pops}
    server_tree_nodes = {pop: set([pop]) for pop in server_pops}

    for pop in client_pops:
        n_cr = cr_per_pop[pop]
        for _ in range(n_cr):
            # Name like A1, A2, ... for this PoP
            router_index_per_pop[pop] += 1
            local_idx = router_index_per_pop[pop]
            cr_name = f"{pop}{local_idx}"

            cr_idx += 1
            client_routers.append(cr_name)
            G.add_node(cr_name, type="client_router")
            G.add_edge(pop, cr_name)
            client_tree_nodes[pop].add(cr_name)

            # Initialize leaf index for this router
            leaf_index_per_router[cr_name] = 0

            # attach clients for this CR
            n_clients_here = client_counts_per_cr[len(client_routers) - 1]
            for _ in range(n_clients_here):
                leaf_idx = leaf_index_per_router[cr_name]
                # A1a, A1b, ...
                c_name = f"{cr_name}{chr(ord('a') + leaf_idx)}"
                leaf_index_per_router[cr_name] += 1

                client_idx += 1
                clients.append(c_name)
                G.add_node(c_name, type="client")
                G.add_edge(cr_name, c_name)
                client_tree_nodes[pop].add(c_name)

    # -------------------------
    # Build server routers + servers
    # -------------------------
    # Decide server routers per server PoP
    sr_per_pop = {}
    total_sr = 0
    for pop in server_pops:
        n_sr = rng.randint(SR_PER_POP_MIN, SR_PER_POP_MAX)
        sr_per_pop[pop] = n_sr
        total_sr += n_sr

    # Allocate N_servers across SRs
    server_counts_per_sr = allocate_counts(
        total=N_servers,
        n_bins=total_sr,
        min_val=SERVERS_PER_SR_MIN,
        max_val=SERVERS_PER_SR_MAX,
        rng=rng,
    )

    server_idx = 0
    sr_idx = 0
    server_routers = []
    servers = []

    for pop in server_pops:
        n_sr = sr_per_pop[pop]
        for _ in range(n_sr):
            # Name like A1, A2, ... continuing from any client routers on same PoP
            router_index_per_pop[pop] += 1
            local_idx = router_index_per_pop[pop]
            sr_name = f"{pop}{local_idx}"

            sr_idx += 1
            server_routers.append(sr_name)
            G.add_node(sr_name, type="server_router")
            G.add_edge(pop, sr_name)
            server_tree_nodes[pop].add(sr_name)

            # Initialize leaf index for this router
            leaf_index_per_router[sr_name] = 0

            n_servers_here = server_counts_per_sr[len(server_routers) - 1]
            for _ in range(n_servers_here):
                leaf_idx = leaf_index_per_router[sr_name]
                # A1a, A1b, ... (same scheme for servers)
                s_name = f"{sr_name}{chr(ord('a') + leaf_idx)}"
                leaf_index_per_router[sr_name] += 1

                server_idx += 1
                servers.append(s_name)
                G.add_node(s_name, type="server")
                G.add_edge(sr_name, s_name)
                server_tree_nodes[pop].add(s_name)
    # -------------------------
    # Assign capacity to nodes
    # -------------------------
    C0 = 100  # Mbps

    capacity_map = {
        "client": 1 * C0,
        "client_router": 20 * C0,
        "backbone_router": 200 * C0,
        "server_router": 100 * C0,
        "server": 50 * C0,
    }

    for n, data in G.nodes(data=True):
        ntype = data.get("type")
        if ntype in capacity_map:
            G.nodes[n]["capacity"] = capacity_map[ntype]
        else:
            G.nodes[n]["capacity"] = None  # fallback in case future node types appear


    # -------------------------
    # Layout as before
    # -------------------------

    pos = {}

    # Backbone (PoP) nodes
    backbone_nodes = [n for n, data in G.nodes(data=True)
                      if data.get("type") == "backbone_router"]

    # 1) Place backbone routers at fixed geographic coordinates
    for pop, (lon, lat) in pop_latlon.items():
        pos[pop] = (lon, lat)

    # Global center of backbone (fallback direction if no backbone neighbors)
    cx = sum(lon for lon, _ in pop_latlon.values()) / len(pop_latlon)
    cy = sum(lat for _, lat in pop_latlon.values()) / len(pop_latlon)

    # Geometry parameters
    ROUTER_RADIUS     = 2                 # distance from PoP to each CR/SR
    LEAF_RADIUS       = 2                 # distance from router to each client/server
    GAP_SAFETY_FRACT  = 1                 # use 100% of each free gap
    IDEAL_SECTOR      = 2 * math.pi / 4   # 180° per router (as in your code)
    MIN_SECTOR        = math.pi / 2.      # 90° per router
    LEAF_SECTOR_FRACT = 1                 # leaves use full router sector
    MIN_LEAF_SEP      = math.radians(24)  # 20 degrees in radians

    def compute_gaps(pop):
        """
        For a PoP, compute all angular gaps between backbone neighbors.
        Returns a list of (mid_angle, gap_width) in radians.
        """
        px, py = pos[pop]
        neighbors = [nb for nb in G.neighbors(pop)
                     if G.nodes[nb].get("type") == "backbone_router"]

        # No backbone neighbors: treat whole circle as a single gap
        if not neighbors:
            base = math.atan2(py - cy, px - cx)
            return [(base, 2 * math.pi)]

        blocked = []
        for nb in neighbors:
            x_nb, y_nb = pos[nb]
            blocked.append(math.atan2(y_nb - py, x_nb - px))
        blocked.sort()

        extended = blocked + [blocked[0] + 2 * math.pi]

        gaps = []
        for a, b in zip(extended[:-1], extended[1:]):
            gap = b - a
            mid = (a + b) / 2.0
            gaps.append((mid, gap))
        return gaps

    # 2) For each PoP, assign router sectors across multiple gaps
    for pop in backbone_nodes:
        px, py = pos[pop]

        # Routers directly attached to this PoP
        crs = [nb for nb in G.neighbors(pop)
               if G.nodes[nb].get("type") == "client_router"]
        srs = [nb for nb in G.neighbors(pop)
               if G.nodes[nb].get("type") == "server_router"]

        routers = crs + srs
        if not routers:
            continue

        K = len(routers)

        # --- compute all usable gaps ---
        raw_gaps = compute_gaps(pop)  # list of (mid_angle, full_gap_width)
        usable_gaps = []
        total_usable = 0.0

        for mid, full_w in raw_gaps:
            w = full_w * GAP_SAFETY_FRACT
            if w <= 0:
                continue
            usable_gaps.append([mid, w])  # mutable width
            total_usable += w

        if not usable_gaps:
            # Fallback: whole circle
            usable_gaps = [[math.atan2(py - cy, px - cx), 2 * math.pi]]
            total_usable = 2 * math.pi

        # --- choose sector size per router ---
        # try ideal, then min, else equal split of total_usable
        if K * IDEAL_SECTOR <= total_usable:
            sector = IDEAL_SECTOR
        elif K * MIN_SECTOR <= total_usable:
            sector = MIN_SECTOR
        else:
            sector = total_usable / K  # may be < 90° if geometry too tight

        # --- compute capacities per gap with this sector width ---
        capacities = [int(w // sector) for _mid, w in usable_gaps]
        if sum(capacities) < K:
            # relax to proportional allocation
            capacities = [0] * len(usable_gaps)
            remaining = K
            for i, (_mid, w) in enumerate(usable_gaps):
                if remaining <= 0:
                    break
                share = max(1, int(round(K * (w / total_usable))))
                share = min(share, remaining)
                capacities[i] = share
                remaining -= share
            if remaining > 0:
                idx_max = max(range(len(usable_gaps)),
                              key=lambda i: usable_gaps[i][1])
                capacities[idx_max] += remaining

        # --- assign routers to gaps, and compute each router's sector center ---
        router_centers = []
        router_idx = 0

        for (gap_idx, (mid, w)) in enumerate(usable_gaps):
            cap = capacities[gap_idx]
            if cap <= 0:
                continue
            n_here = min(cap, K - router_idx)
            if n_here <= 0:
                continue

            span = sector * n_here
            if span > w:
                span = w  # last resort, sectors may be slightly compressed

            if n_here ==1:
                start_center = mid
            else:
                start_center = mid - sector / 2.0
            for j in range(n_here):
                if router_idx >= K:
                    break
                theta_c = start_center + j * sector
                router_centers.append(theta_c)
                router_idx += 1

            if router_idx >= K:
                break

        while len(router_centers) < K:
            router_centers.append(raw_gaps[0][0])

        # Now we have a center angle for each router (aligned with routers list)
        for router, theta_c in zip(routers, router_centers):
            # Router position
            rx = px + ROUTER_RADIUS * math.cos(theta_c)
            ry = py + ROUTER_RADIUS * math.sin(theta_c)
            pos[router] = (rx, ry)

            # Leaves (clients or servers) attached to this router
            leaves = [
                nb for nb in G.neighbors(router)
                if G.nodes[nb].get("type") in ("client", "server")
            ]
            n_leaf = len(leaves)
            if n_leaf == 0:
                continue

            # Leaves use a fraction of the router sector (to keep margin),
            # but at least enough to ensure 20° separation between neighbors.
            if n_leaf == 1:
                leaf_angles = [theta_c]
            else:
                base_leaf_sector = sector * LEAF_SECTOR_FRACT
                required_leaf_sector = (n_leaf - 1) * MIN_LEAF_SEP
                leaf_sector = max(base_leaf_sector, required_leaf_sector)

                #start_leaf = theta_c - (leaf_sector / 2.0) + MIN_LEAF_SEP / 2.0
                start_leaf = theta_c - MIN_LEAF_SEP* n_leaf/2.
                leaf_angles = [
                    start_leaf + MIN_LEAF_SEP * j
                    for j in range(n_leaf)
                ]

            for leaf, phi in zip(leaves, leaf_angles):
                lx = rx + LEAF_RADIUS * math.cos(phi)
                ly = ry + LEAF_RADIUS * math.sin(phi)
                pos[leaf] = (lx, ly)

    return G, pos
