import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D

from generate_network_logistical import generate_network_logistical
from CABC_logistical import SO_CABC, UE_CABC


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

    # Layer → color mapping (topology plot)
    layer_colors = {
        "manufacturer": "firebrick",
        "rd": "orange",
        "ld": "gold",
        "retailer": "steelblue",
    }

    # -----------------------------
    # 2) Compute centralities
    # -----------------------------
    centralities = {
        "Degree": nx.degree_centrality(G),
        "Betweenness": nx.betweenness_centrality(G),
        "Closeness": nx.closeness_centrality(G),
        "Harmonic": nx.harmonic_centrality(G),
    }

    # CABC centralities
    so_node_flow, so_flow_edges, so_total_cost = SO_CABC(G)
    ue_node_flow, ue_flow_edges, ue_total_cost = UE_CABC(G)

    centralities["SO_CABC"] = so_node_flow
    centralities["UE_CABC"] = ue_node_flow

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

    # nodes: shape = layer, color = region
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

    # Legend 1: node type (shapes)
    leg1 = ax.legend(handles=legend_elements_centrality, loc="upper left", title="Node Type")
    ax.add_artist(leg1)

    # Legend 2: region (colors)
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

        plt.savefig(f"{name}_centrality_logistical.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # -----------------------------
    # 6) Combined 2×3 subplot
    # -----------------------------
    ordered_names = [
        "Degree",
        "Betweenness",
        "Closeness",
        "Harmonic",
        "SO_CABC",
        "UE_CABC",
    ]

    pretty_titles = {
        "Degree": "Degree Centrality",
        "Betweenness": "Betweenness Centrality",
        "Closeness": "Closeness Centrality",
        "Harmonic": "Harmonic Centrality",
        "SO_CABC": "System-Optimal Congestion Adaptive Betweenness Centrality",
        "UE_CABC": "User-Equilibrium Congestion Adaptive Betweenness Centrality",
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
        fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)

        ax.set_title(pretty_titles[name])
        ax.axis("off")

    fig.legend(handles=legend_elements_centrality, loc="lower right", title="Node Type")
    fig.suptitle("Logistical Network Centralities", fontsize=16)
    fig.tight_layout()

    fig.savefig("all_logistical_centralities.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
