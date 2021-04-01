import numpy as np
import scipy as sp


def gcn_receptive_field_mask(adj, dim_label, num_layers, n_nodes):
    """
    Calculate a matrix indicating which nodes / edges should be summed over
    to determine one dimension of the node-specific perturbation radius

    Parameters
    ----------
    adj: scipy.spare_matrix [n_nodes, n_nodes]
        Binary sparse undirected adjacency matrix of graph
    dim_label : str
        The label of the dimension we're calculating a mask for
        (adj_del, attr_add or attr_del)
    num_layers : int
        Number of layers / propagation hops in the underlying GNN
    n_nodes : int
        Number of nodes in the graph

    Returns
    -------
    scipy.sparse_matrix [2, ?]
    """

    if dim_label not in ['adj_del', 'attr_add', 'attr_del']:
        raise NotImplementedError(f'Dim-label {dim_label} not supported')

    if dim_label == 'adj_del':
        edge_idx = np.stack(adj.nonzero())
        edge_idx = edge_idx[:, edge_idx[0] < edge_idx[1]]

        k_hop_reachable = [np.eye(n_nodes)]
        for _ in range(num_layers):
            k_hop_reachable.append(k_hop_reachable[-1] + k_hop_reachable[-1] @ adj)

        mask = np.zeros((n_nodes, edge_idx.shape[1]))

        for node in range(n_nodes):
            for i, edge in enumerate(edge_idx.T):
                start, end = edge[0], edge[1]
                if ((k_hop_reachable[num_layers - 1][node, start] > 0
                     and k_hop_reachable[num_layers][node, end] > 0)
                        or
                        (k_hop_reachable[num_layers - 1][node, end] > 0
                         and k_hop_reachable[num_layers][node, start] > 0)):
                    mask[node, i] = 1

    if dim_label in ['attr_add', 'attr_del']:

        mask = np.zeros((n_nodes, n_nodes))

        reachable = np.eye(n_nodes)
        for _ in range(num_layers):
            reachable += reachable @ adj

        mask[reachable.nonzero()] = 1

    return sp.sparse.csr_matrix(mask)
