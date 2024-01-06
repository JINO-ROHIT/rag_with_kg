import networkx as nx
import matplotlib.pyplot as plt

class Knowledge_graph:
    def __init__(self, df: str = None):
        self.df = df
    
    def create_graph(self):
        self.G = nx.from_pandas_edgelist(self.df, 'node_1', 'node_2', edge_attr='edge', create_using=nx.Multiself.G())
        nx.draw(self.G, with_labels=True)
    
    def query_sub_graph(self, query_node):
        neighbors = list(self.G.neighbors(query_node)) + [query_node]
        subgraph = self.G.subgraph(neighbors)

        # Plot the subself.G
        pos = nx.spring_layout(subgraph)
        nx.draw(subgraph, pos, with_labels=True, node_color='lightblue', font_weight='bold', node_size=700)
        plt.title(f"Subself.G of Node {query_node} and its Neighbors")
        #plt.show()
        plt.savefig('subgraph.png')