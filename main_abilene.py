import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import numpy as np
import time
from matplotlib.lines import Line2D

from generate_network_abilene import generate_graph  # imports function to generate graph
from CABC_abilene import SO_CABC, UE_CABC, flow_betweenness_centrality

node_sz = 50
row_figsize = 12
col_figsize = 6
plot_mk_sz = 8
plot_node_ft = 4

def main():
    # -----------------------------
    # 1) Generate the Abilene Internet Topology
    # -----------------------------
    G, pos_kk = generate_graph()  # Generating Random Mini Model of Internet

    # Node type to shape mapping
    type_shapes = {
        "client": "o",
        "client_router": "s",
        "server": "D",
        "server_router": "^",
        "backbone_router": "v"
    }

    # Node type to color mapping for general topology
    type_colors = {
        "client": "skyblue",
        "client_router": "dodgerblue",
        "server": "green",
        "server_router": "limegreen",
        "backbone_router": "orange"
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
        print_every=25
    )
    centrality_times["Flow_Betweenness"] = elapsed

    centralities = {
        "Degree": degree_cent,
        "Betweenness": bet_cent,
        "Closeness": close_cent,
        "Flow_Betweenness": flow_bet_cent,
    }

    # CABC centralities
    print("Computing SO-CABC centrality...")
    t0_so = time.perf_counter()
    so_node_flow, so_flow_edges, so_total_cost = SO_CABC(G, client_rate_mbps=50.0)
    so_elapsed = time.perf_counter() - t0_so
    centrality_times["SO_CABC"] = so_elapsed
    print(f"SO-CABC done in {so_elapsed:.6f}s. Total cost = {so_total_cost:.6f}")

    print("Computing UE-CABC centrality...")
    t0_ue = time.perf_counter()
    ue_node_flow, ue_flow_edges, ue_total_cost = UE_CABC(G, client_rate_mbps=50.0)
    ue_elapsed = time.perf_counter() - t0_ue
    centrality_times["UE_CABC"] = ue_elapsed
    print(f"UE-CABC done in {ue_elapsed:.6f}s. Total cost = {ue_total_cost:.6f}")

    # Add SO_CABC and UE_CABC node centralities (will normalize below)
    centralities["SO_CABC"] = so_node_flow
    centralities["UE_CABC"] = ue_node_flow

    # Print summary table
    print("\n=== Centrality Compute Times ===")
    for name in ["Degree", "Closeness", "Betweenness", "Flow_Betweenness", "SO_CABC", "UE_CABC"]:
        print(f"{name:20s}: {centrality_times[name]:.6f} s")

    # Normalize all centralities to [0,1]
    for name, cent in centralities.items():
        vals = np.array(list(cent.values()))
        if vals.max() > 0:
            centralities[name] = {n: v / vals.max() for n, v in cent.items()}
        else:
            centralities[name] = {n: 0.0 for n in cent}

    # -----------------------------
    # 3) Individual plots
    # -----------------------------
    cmap = cm.Spectral

    # Colored legend for topology
    legend_elements_topology = [
        Line2D(
            [0], [0],
            marker=type_shapes[node_type],
            color='w',
            label=node_type.replace('_', ' ').title(),
            markerfacecolor=type_colors[node_type],
            markersize=plot_mk_sz
        )
        for node_type in type_shapes
    ]

    # Shape-only legend for centrality plots
    legend_elements_centrality = [
        Line2D(
            [0], [0],
            marker=shape,
            color='k',
            label=node_type.replace('_', ' ').title(),
            markerfacecolor='white',
            markersize=10,
            markeredgewidth=1.5,
            linestyle='None'
        )
        for node_type, shape in type_shapes.items()
    ]

    # General topology plot
    fig, ax = plt.subplots(figsize=(row_figsize, col_figsize))
    for node_type, shape in type_shapes.items():
        nodes_of_type = [n for n, d in G.nodes(data=True) if d["type"] == node_type]
        nx.draw_networkx_nodes(
            G, pos=pos_kk, nodelist=nodes_of_type,
            node_color=type_colors[node_type],
            node_shape=shape,
            node_size=node_sz,
            ax=ax
        )
    nx.draw_networkx_edges(G, pos=pos_kk, ax=ax)
    nx.draw_networkx_labels(G, pos=pos_kk, font_size=plot_node_ft, ax=ax)
    ax.set_title("The Abilene Internet Topology")
    ax.axis('off')
    ax.legend(handles=legend_elements_topology, loc='lower right', title="Node Type")
    print("Saving topology.png ...")
    plt.savefig("topology.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Individual centrality plots (now includes UE_CABC as well)
    for name, cent in centralities.items():
        fig, ax = plt.subplots(figsize=(row_figsize, col_figsize))
        for node_type, shape in type_shapes.items():
            nodes_of_type = [n for n, d in G.nodes(data=True) if d["type"] == node_type]
            node_colors = [cmap(cent[n]) for n in nodes_of_type]
            nx.draw_networkx_nodes(
                G, pos=pos_kk,
                nodelist=nodes_of_type,
                node_color=node_colors,
                node_shape=shape,
                node_size=node_sz,
                edgecolors='k',
                linewidths=0.5,
                ax=ax
            )
        nx.draw_networkx_edges(G, pos=pos_kk, ax=ax, alpha=0.5)
        nx.draw_networkx_labels(G, pos=pos_kk, font_size=plot_node_ft, ax=ax)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(f"{name} Centrality (normalized)")
        ax.set_title(f"{name} Centrality")
        ax.axis('off')
        ax.legend(handles=legend_elements_centrality, loc='lower right', title="Node Type")
        outfile = f"{name}_centrality.png"
        print(f"Saving {outfile} ...")
        plt.savefig(outfile, dpi=300, bbox_inches="tight")
        plt.close(fig)

        # -----------------------------
    # 4) Combined subplot figure (2x3) with
    #    Degree, Closeness, Betweenness, Flow_Betweenness, SO_CABC, UE_CABC
    # -----------------------------
    ordered_names = ["Degree", "Closeness", "Betweenness",
                     "Flow_Betweenness", "SO_CABC", "UE_CABC"]
    
    # Pretty titles for display
    pretty_titles = {
        "Degree": "Degree Centrality",
        "Betweenness": "Betweenness Centrality",
        "Closeness": "Closeness Centrality",
        "Flow_Betweenness": "Flow Betweenness Centrality",
        "SO_CABC": "System-Optimal Congestion Adaptive Betweenness Centrality (SO-CABC)",
        "UE_CABC": "User-Equilibrium Congestion Adaptive Betweenness Centrality (UE-CABC)",
    }

    all_plots = [(name, centralities[name]) for name in ordered_names]

    ncols, nrows = 3, 2
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(row_figsize * ncols, col_figsize * nrows)
    )
    axes = axes.flatten()

    for i, (name, cent) in enumerate(all_plots):
        ax = axes[i]

        for node_type, shape in type_shapes.items():
            nodes_of_type = [n for n, d in G.nodes(data=True) if d.get("type") == node_type]
            node_colors = [cmap(cent[n]) for n in nodes_of_type]
            nx.draw_networkx_nodes(
                G, pos=pos_kk, nodelist=nodes_of_type,
                node_color=node_colors,
                node_shape=shape,
                node_size=node_sz,
                edgecolors='k',
                linewidths=0.5,
                ax=ax
            )

        nx.draw_networkx_edges(G, pos=pos_kk, ax=ax, alpha=0.5)
        nx.draw_networkx_labels(G, pos=pos_kk, font_size=plot_node_ft, ax=ax)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(f"{name} (norm.)")

        # Use pretty, expanded titles (with special cases for SO_CABC, UE_CABC)
        ax.set_title(pretty_titles.get(name, name), fontsize=12)
        ax.axis("off")

    # Hide unused axes if any (shouldn't be, but just in case)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.legend(handles=legend_elements_centrality, loc="lower right", title="Node Type")
    fig.suptitle("Abilene Internet Centralities", fontsize=16)
    fig.tight_layout()
    print("Saving all_centralities_with_CABC.png ...")
    fig.savefig("all_centralities_with_CABC.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # -----------------------------
    # 5) Save timing results
    # -----------------------------
    with open("centrality_compute_times.txt", "w") as f:
        f.write("Centrality Compute Times (seconds)\n")
        f.write("----------------------------------\n")
        for name in ["Degree", "Closeness", "Betweenness", "Flow_Betweenness", "SO_CABC", "UE_CABC"]:
            f.write(f"{name}: {centrality_times[name]:.6f}\n")

    print("Saving centrality_compute_times.txt ...")


if __name__ == "__main__":
    main()
