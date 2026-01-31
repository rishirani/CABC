import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import numpy as np

from generate_network import generate_graph  # imports function to generate graph
from flow_centrality import flow_centrality # the function we just wrote

def main():
    # 1) Generate the mini-Internet topology
    G, pos_kk= generate_graph() #Generating Random Mini Model of Internet
    # 2) Run flow centrality
    node_flow, flow_edges, total_cost = flow_centrality(G, client_rate_mbps=100.0)

    # 3) Print results
    print("=== Flow Centrality Test ===")
    print(f"Total min-cost flow cost = {total_cost:.2f}")
    print("\nTop 50 nodes by throughput (Mbps):")
    top_nodes = sorted(node_flow.items(), key=lambda kv: kv[1], reverse=True)[:50]
    for node, flow in top_nodes:
        print(f"{node:<15} {flow:>10.1f} Mbps")

    # 4) Plot flow centrality
    raw_values = np.array([(node_flow[n]) for n in G.nodes()])

    # Normalize to [0, 1]
    if raw_values.max() > 0:  # avoid divide-by-zero
        values = raw_values / raw_values.max()
    else:
        values = np.zeros_like(raw_values)

    cmap = cm.Spectral

    fig, ax = plt.subplots(figsize=(12, 8))
    nx.draw_networkx_edges(G, pos=pos_kk, alpha=0.3, ax=ax)

    nodes = nx.draw_networkx_nodes(
        G, pos=pos_kk,
        node_color=values,
        node_size=300,
        cmap=cmap,
        edgecolors="k",
        ax=ax
    )
    nx.draw_networkx_labels(G, pos=pos_kk, font_size=7, ax=ax)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Normalized Flow Centrality")

    ax.set_title("Flow Centrality Test Plot (Normalized)")
    ax.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
