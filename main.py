import torch
import torch.nn as tnn
from torch_geometric.data import Data as gData
from torch_geometric.data import Batch as gBatch
import torch_geometric.nn as gnn
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import cmath


def main():
    batch_size = 16
    num_nodes = 4
    num_in_node_features = 16
    num_out_node_features = 64
    num_in_edge_features = 4
    num_out_edge_features = 8

    # Define batch of example graph
    edge_index = torch.tensor([[0, 1, 2, 0, 3, 2, 3, 0],
                               [1, 0, 0, 2, 2, 3, 0, 3]], dtype=torch.long)





    # Node features
    batch_x = torch.randn((batch_size, num_nodes, num_in_node_features), dtype=torch.float)

    # Edge features -- batch_edge_features has shape: torch.Size([4, 42, 8])
    batch_edge_attr = torch.randn((batch_size, edge_index.size(1), num_in_edge_features), dtype=torch.float)

    # Wrap input node and edge features, along with the single edge_index, into a `torch_geometric.data.Batch` instance
    l = []
    for i in range(batch_size):
        l.append(gData(x=batch_x[i], edge_index=edge_index, edge_attr=batch_edge_attr[i]))
    batch = gBatch.from_data_list(l)

    # Thus,
    # batch.x          -- shape: torch.Size([28, 16])
    # batch.edge_index -- shape: torch.Size([2, 168])
    # batch.edge_attr  -- shape: torch.Size([168, 8])

    # Define NNConv layer
    nn = tnn.Sequential(tnn.Linear(num_in_edge_features, 25), tnn.ReLU(),
                        tnn.Linear(25, num_in_node_features * num_out_node_features))
    gconv = gnn.NNConv(in_channels=num_in_node_features, out_channels=num_out_node_features, nn=nn, aggr='mean')

    # Forward pass
    y = gconv(x=batch.x, edge_index=batch.edge_index, edge_attr=batch.edge_attr)


if __name__ == '__main__':
    main()
