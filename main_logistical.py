import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import numpy as np
import time
from matplotlib.lines import Line2D

from generate_network_logistical import generate_network_logistical
from CABC_logistical import SO_CABC, UE_CABC, flow_betweenness_centrality


# -----------------------------
# Plot parameters
# -----------------------------
node_sz = 100
row_figsize = 12
col_figsize = 6
plot_mk_sz = 8
plot_node_ft = 4



def main():
    # -----------------------------
    # 1) Generate the logistical network
    # -----------------------------
    G, pos = generate_network_logistical()

    # Layer → shape mapping
    layer_shapes = {
        "manufacturer": "s",
        "rd": "o",
        "ld": "D",
        "retailer": "^",
    }

    # -----------------------------
    # 2) Compute centralities + runtimes
    # -----------------------------
    def timed_compute(name, fn, *args, **kwargs):
        print(f"Computing {name}...")
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        print(f"{name} done in {elapsed:.6f}s.")
        return result, elapsed

    centrality_times = {}

    degree_cent, elapsed = timed_compute("Degree centrality", nx.degree_centrality, G)
    centrality_times["Degree"] = elapsed

    bet_cent, elapsed = timed_compute("Betweenness centrality", nx.betweenness_centrality, G)
    centrality_times["Betweenness"] = elapsed

    close_cent, elapsed = timed_compute("Closeness centrality", nx.closeness_centrality, G)
    centrality_times["Closeness"] = elapsed

    flow_bet_cent, elapsed = timed_compute(
        "Flow Betweenness centrality",
        flow_betweenness_centrality,
        G,
        progress=True,
        print_every=25,
    )
    centrality_times["Flow_Betweenness"] = elapsed

    centralities = {
        "Degree": degree_cent,
        "Betweenness": bet_cent,
        "Closeness": close_cent,
        "Flow_Betweenness": flow_bet_cent,
    }

    print("Computing SO-CABC centrality...")
    t0_so = time.perf_counter()
    so_node_flow, so_flow_edges, so_total_cost = SO_CABC(G)
    so_elapsed = time.perf_counter() - t0_so
    centrality_times["SO_CABC"] = so_elapsed
    print(f"SO-CABC done in {so_elapsed:.6f}s. Total cost = {so_total_cost:.6f}")

    print("Computing UE-CABC centrality...")
    t0_ue = time.perf_counter()
    ue_node_flow, ue_flow_edges, ue_total_cost = UE_CABC(G)
    ue_elapsed = time.perf_counter() - t0_ue
    centrality_times["UE_CABC"] = ue_elapsed
    print(f"UE-CABC done in {ue_elapsed:.6f}s. Total cost = {ue_total_cost:.6f}")

    centralities["SO_CABC"] = so_node_flow
    centralities["UE_CABC"] = ue_node_flow

    print("\n=== Centrality Compute Times ===")
    for name in ["Degree", "Closeness", "Betweenness", "Flow_Betweenness", "SO_CABC", "UE_CABC"]:
        print(f"{name:20s}: {centrality_times[name]:.6f} s")

    # Normalize to [0,1]
    for name, cent in centralities.items():
        vals = np.array(list(cent.values()))
        if vals.max() > 0:
            centralities[name] = {n: v / vals.max() for n, v in cent.items()}
        else:
            centralities[name] = {n: 0.0 for n in cent}

    # -----------------------------
    # 3) Legends
    # -----------------------------
    cmap = cm.Spectral

    layer_pretty = {
        "manufacturer": "Manufacturer",
        "rd": "Reg. Distributor",
        "ld": "Loc. Distributor",
        "retailer": "Retailer",
    }

    legend_elements_centrality = [
        Line2D(
            [0], [0],
            marker=shape,
            color="k",
            label=layer_pretty[layer],
            markerfacecolor="white",
            markersize=10,
            markeredgewidth=1.5,
            linestyle="None"
        )
        for layer, shape in layer_shapes.items()
    ]

    # -----------------------------
    # 4) Topology plot (region-colored + two legends)
    # -----------------------------
    fig, ax = plt.subplots(figsize=(row_figsize, col_figsize))

    region_colors = {"W": "tab:blue", "C": "tab:orange", "E": "tab:green"}

    for layer, shape in layer_shapes.items():
        for reg, col in region_colors.items():
            nodes = [n for n, d in G.nodes(data=True)
                    if d["layer"] == layer and d.get("region") == reg]
            if nodes:
                nx.draw_networkx_nodes(
                    G, pos,
                    nodelist=nodes,
                    node_color=col,
                    node_shape=shape,
                    node_size=node_sz,
                    ax=ax
                )

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=plot_node_ft, ax=ax)

    leg1 = ax.legend(handles=legend_elements_centrality, loc="upper left", title="Node Type")
    ax.add_artist(leg1)

    region_legend = [
        Line2D([0], [0], marker='o', color='w', label='West',
            markerfacecolor=region_colors["W"], markersize=plot_mk_sz),
        Line2D([0], [0], marker='o', color='w', label='Central',
            markerfacecolor=region_colors["C"], markersize=plot_mk_sz),
        Line2D([0], [0], marker='o', color='w', label='East',
            markerfacecolor=region_colors["E"], markersize=plot_mk_sz),
    ]
    ax.legend(handles=region_legend, loc="upper right", title="Region")

    ax.set_title("Regional Logistics Network")
    ax.axis("off")

    print("Saving logistical_topology.png ...")
    plt.savefig("logistical_topology.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # -----------------------------
    # 5) Individual centrality plots
    # -----------------------------
    for name, cent in centralities.items():
        fig, ax = plt.subplots(figsize=(row_figsize, col_figsize))

        for layer, shape in layer_shapes.items():
            nodes = [n for n, d in G.nodes(data=True) if d["layer"] == layer]
            node_colors = [cmap(cent[n]) for n in nodes]

            nx.draw_networkx_nodes(
                G, pos,
                nodelist=nodes,
                node_color=node_colors,
                node_shape=shape,
                node_size=node_sz,
                edgecolors="k",
                linewidths=0.5,
                ax=ax
            )

        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.5)
        nx.draw_networkx_labels(G, pos, font_size=plot_node_ft, ax=ax)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(f"{name} Centrality (normalized)")

        ax.set_title(f"{name} Centrality")
        ax.axis("off")
        ax.legend(handles=legend_elements_centrality, loc="lower right", title="Node Type")

        outfile = f"{name}_centrality_logistical.png"
        print(f"Saving {outfile} ...")
        plt.savefig(outfile, dpi=300, bbox_inches="tight")
        plt.close(fig)

    # -----------------------------
    # 6) Combined 2×3 subplot
    # -----------------------------
    ordered_names = [
        "Degree",
        "Closeness",
        "Betweenness",
        "Flow_Betweenness",
        "SO_CABC",
        "UE_CABC",
    ]

    pretty_titles = {
        "Degree": "Degree Centrality",
        "Betweenness": "Betweenness Centrality",
        "Closeness": "Closeness Centrality",
        "Flow_Betweenness": "Flow Betweenness Centrality",
        "SO_CABC": "System-Optimal Congestion Adaptive Betweenness Centrality (SO-CABC)",
        "UE_CABC": "User-Equilibrium Congestion Adaptive Betweenness Centrality (UE-CABC)",
    }

    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(row_figsize * 3, col_figsize * 2)
    )
    axes = axes.flatten()

    for i, name in enumerate(ordered_names):
        ax = axes[i]
        cent = centralities[name]

        for layer, shape in layer_shapes.items():
            nodes = [n for n, d in G.nodes(data=True) if d["layer"] == layer]
            node_colors = [cmap(cent[n]) for n in nodes]

            nx.draw_networkx_nodes(
                G, pos,
                nodelist=nodes,
                node_color=node_colors,
                node_shape=shape,
                node_size=node_sz,
                edgecolors="k",
                linewidths=0.5,
                ax=ax
            )

        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.5)
        nx.draw_networkx_labels(G, pos, font_size=plot_node_ft, ax=ax)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(f"{name} (norm.)")

        ax.set_title(pretty_titles[name])
        ax.axis("off")

    fig.legend(handles=legend_elements_centrality, loc="lower right", title="Node Type")
    fig.suptitle("Logistical Network Centralities", fontsize=16)
    fig.tight_layout()

    print("Saving all_logistical_centralities.png ...")
    fig.savefig("all_logistical_centralities.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # -----------------------------
    # 7) Save timing results
    # -----------------------------
    with open("centrality_compute_times_logistical.txt", "w") as f:
        f.write("Centrality Compute Times (seconds)\n")
        f.write("----------------------------------\n")
        for name in ["Degree", "Closeness", "Betweenness", "Flow_Betweenness", "SO_CABC", "UE_CABC"]:
            f.write(f"{name}: {centrality_times[name]:.6f}\n")

    print("Saving centrality_compute_times_logistical.txt ...")


if __name__ == "__main__":
    main()
