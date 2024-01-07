import networkx as nx
import matplotlib.pyplot as plt

class Knowledge_graph:
    def __init__(self, df: str = None):
        self.df = df
    
    def create_graph(self):
        self.G = nx.from_pandas_edgelist(self.df, 'node_1', 'node_2', edge_attr='edge', create_using=nx.MultiGraph())
        nx.draw(self.G, with_labels=True)
    
    def query_sub_graph(self, query_node):
        neighbors = list(self.G.neighbors(query_node)) + [query_node]
        subgraph = self.G.subgraph(neighbors)

        pos = nx.spring_layout(subgraph)

        plt.figure(figsize=(8, 8))

        node_size = 2000
        node_color = 'lightblue'
        font_color = 'black'
        font_weight = 'bold'
        font_size = 8
        edge_color = 'gray'
        edge_style = 'dashed'

        # Draw the subgraph
        nx.draw(subgraph, pos, with_labels=True, node_size=node_size, node_color=node_color, font_color=font_color, font_size = font_size,
                font_weight=font_weight, edge_color=edge_color, style=edge_style)

        # Add additional customizations
        plt.title(f"Graph of Node: {query_node}")

        # Save the plot to a file
        plt.savefig('subgraph.png')
        #plt.show()