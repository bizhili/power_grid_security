import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
import json


def plot_2d_RGG(G: nx.Graph, pos: dict) -> None:
    """Plot a 2D RGGmetric graph.
    Args:
        G: A networkx Graph object.
        pos: A dictionary mapping node IDs to (x, y) coordinates.
    """
    plt.figure(figsize=(8, 6))
    plt.subplot(1, 2, 1)
    nx.draw_networkx_edges(G, pos, alpha=0.4)
    nx.draw_networkx_nodes(G, pos, node_size=80)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Random RGGmetric graph")
    plt.subplot(1, 2, 2)
    # Get the degrees of all nodes
    degrees = [d for n, d in G.degree()]
    plt.hist(degrees, bins=np.arange(min(degrees), max(degrees) + 1) - 0.5, density=True, alpha=0.75)
    plt.title("Degree Distribution Histogram")
    plt.xlabel("Degree")
    plt.ylabel("Probability")
    plt.grid(True)
    plt.show()

def plot_degree_distribution(G: nx.graph, Gpre= None, fixMin= 0, fixMax= 30, ax= None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    degrees = [d for n, d in G.degree()]
    refMin= min(degrees)
    refMax= max(degrees)+1
    fontZise= 25
    if Gpre is not None:
        degreesRef= [d for n, d in Gpre.degree()]
        refMin= min(refMin, min(degreesRef))
        refMax= max(refMax, max(degreesRef)+1)
        if fixMin is not None:
            refMin= fixMin
            refMax= fixMax
        ax.hist(degrees, bins=np.arange(refMin, refMax)-0.5, density=True, color="royalblue", alpha=0.5, label="Ture Graph")
        ax.hist(degreesRef, bins=np.arange(refMin, refMax)-0.5, density=True,color="orange", alpha=0.3, label="Predicted Graph")
        
    else:
        if fixMin is not None:
            refMin= fixMin
            refMax= fixMax
        ax.hist(degrees, bins=np.arange(refMin, refMax)-0.5, density=True, color="royalblue", alpha=0.5, label="True Graph")
    #ax.set_title(f"Degree distribution", fontsize=fontZise)
    #ax.legend(prop = { "size": fontZise }, loc ="upper right")
    ax.set_xlabel("Degree", fontsize=fontZise)
    ax.set_ylabel("Probability", fontsize=fontZise)
    ax.tick_params(axis='both', labelsize=fontZise)
    ax.grid(True)

def read_posDic_from_json(filename= ""):
    with open(filename, 'r') as file:
        StatesCoor= json.load(file)
    posDic={int(i):[StatesCoor[i][1][1], StatesCoor[i][1][0]] for i in StatesCoor }
    return posDic

def plot_spring_layout(G= None, GPre= None, pos= None, ax= None, label= False, stringT= ""):
    fontZise= 25
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    # Calculate node positions using spring layout
    if G is not None:
        n= nx.number_of_nodes(G)
    else:
        n= nx.number_of_nodes(GPre)
    if pos is None:
        pos = nx.spring_layout(G, k= 2/n, iterations= 100)
    if G is not None:
        edgesList= nx.to_edgelist(G)
        for edge in edgesList:
            node0= pos[edge[0]]
            node1= pos[edge[1]]
            ax.plot([node0[0], node1[0]], 
                    [node0[1], node1[1]], marker = '', color="royalblue", alpha=0.5)
        ax.plot([node0[0], node1[0]], 
                    [node0[1], node1[1]], marker = '', color="royalblue", alpha=0.5, label= "True links")  
    if GPre is not None:
        edgesList= nx.to_edgelist(GPre)
        for edge in edgesList:
            node0= pos[edge[0]]
            node1= pos[edge[1]]
            ax.plot([node0[0], node1[0]], 
                    [node0[1], node1[1]], marker = '', color="orange", alpha=0.3)
        ax.plot([node0[0], node1[0]], 
                        [node0[1], node1[1]], marker = '', color="orange", alpha=0.3, label= "Predicted  links")
    for i in pos:
        ax.plot([pos[i][0]], [pos[i][1]], marker='.', color="black", alpha=1 )
        if label:
            ax.text(pos[i][0], pos[i][1], f"{i}")
    #ax.legend(prop = { "size": fontZise }, loc ="upper right")
    ax.set_title(stringT, fontsize=fontZise)
    ax.axis('off')
def plot_spring_and_degree(G: nx.graph, GPre= None, pos= None, fixMin= 0, fixMax= 30, stringT=""):
    f, (ax1, ax2) = plt.subplots(1, 2, sharey= False, figsize=(16, 6))
    plot_degree_distribution(G, GPre, fixMin, fixMax, ax2)
    plot_spring_layout(G, GPre, pos, ax1, label= False, stringT= stringT)
    plt.tight_layout(pad=0.2, w_pad=0.2, h_pad=0.2)
        
def plot_adjacenty(trueGraph, preGraphs=[], campThis= "viridis"):
    maxValue= 0
    fontsize= 25

    for preGraph in preGraphs:
        maxTemp= np.max(preGraph)
        maxValue= maxTemp if maxTemp>maxValue else maxValue

    plt.figure(figsize=(8*(len(preGraphs)+1), 8))
    plt.subplot(1, len(preGraphs)+1, 1)
    plt.imshow(trueGraph, cmap= campThis, norm=Normalize(vmin=0, vmax= maxValue))
    colorbar =plt.colorbar(shrink=0.01)
    colorbar.ax.set_axis_off()
    plt.axis('off')
    plt.title('Real graph', fontsize= fontsize)

    for i in range(len(preGraphs)-1):
        plt.subplot(1, len(preGraphs)+1, i+2)
        plt.imshow(preGraphs[2], cmap= campThis, norm=Normalize(vmin=0, vmax= maxValue))
        plt.title(f'{i+1} strains')
        colorbar =plt.colorbar(shrink=0.01)
        colorbar.ax.set_axis_off()
    
    plt.subplot(1, len(preGraphs)+1, len(preGraphs)+1)
    plt.axis('off')
    plt.imshow(preGraphs[-1], cmap= campThis, norm=Normalize(vmin=0, vmax= maxValue))
    plt.title(f'Predicted graph', fontsize= fontsize)

    mycolorbar= plt.colorbar(label='',  shrink=0.4)
    mycolorbar.ax.tick_params(labelsize=fontsize-8)
    # Adjust layout for better spacing
    plt.tight_layout()
    # Show the plot
    plt.show()
