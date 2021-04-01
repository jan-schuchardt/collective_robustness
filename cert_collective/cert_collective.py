import itertools

import cvxpy as cp
import numpy as np
import scipy as sp

from sparse_smoothing.cert import binary_certificate, joint_binary_certficate


def calc_max_rads(pareto_points, dim_labels):
    """
    Calculate the largest number of edge deletions,
    attr additions and attribute deletions that some node is robust to.

    Parameters
    ----------
    pareto_points: array_like [P, ?]: 
        Pareto-points of base certificates of all predictions,
        where P is number of pareto points.
    dim_labels : list(str)
        List indicating for each budget-dimension if it is related to edge deletion,
        attribute addition or attribute deletion

    Returns
    -------
    array_like [3]
    """

    max_rads = np.zeros(3)

    edge_del_dims = np.array([label.startswith('adj_del') for label in dim_labels])
    attr_add_dims = np.array([label.startswith('attr_add') for label in dim_labels])
    attr_del_dims = np.array([label.startswith('attr_del') for label in dim_labels])

    if edge_del_dims.sum() > 0:
        max_rads[0] = np.maximum(0, np.max(pareto_points[:, edge_del_dims]))
    if attr_add_dims.sum() > 0:
        max_rads[1] = np.maximum(0, np.max(pareto_points[:, attr_add_dims]))
    if attr_del_dims.sum() > 0:
        max_rads[2] = np.maximum(0, np.max(pareto_points[:, attr_del_dims]))

    return max_rads


def calc_local_budgets(local_budget_descriptor, local_budget_mode, attr, adj):
    """
    For each node, calculate how many edge deletions / attr additions / attr deletions
    may be performed on it, based on the provided local budget descriptor

    Parameters
    ----------
    local_budget_descriptor : array_like [3]
        Absolute number or percentage of allowed edge deletions / attr additions / attr deletions
        at each node
    local_budget_mode : str in ['relative', 'absolute']
        Whether the local budgets should be seen as absolute or as a percentage of the number of existing
        nodes / edges at a node
    attr_dense : array_like [n_nodes, n_attributes]
        Attribute matrix of the graph
    A_dense : array_like [n_nodes, n_nodes]
        Adjacency matrix of the graph

    Returns
    -------
    array_like [n_nodes, 3]
    """

    attr_dense = np.array(attr.todense())
    A_dense = np.array(adj.todense())

    num_edges = np.count_nonzero(A_dense, axis=1)
    num_attr = np.count_nonzero(attr_dense, axis=1)
    potential_attr_add = attr_dense.shape[1] - num_attr

    num_existing = np.array([num_edges, num_attr, num_attr])
    potential_change = np.array([num_edges, potential_attr_add, num_attr])

    if local_budget_mode == 'relative':
        local_budgets = np.floor(local_budget_descriptor[:, np.newaxis] * num_existing)
    elif local_budget_mode == 'absolute':
        local_budgets = np.floor(np.minimum(local_budget_descriptor[:, np.newaxis], potential_change))
    else:
        raise ValueError(f'Local budget mode - {local_budget_mode} - not supported')

    return local_budgets


def calc_sparse_directed_incidence_masks(edge_idx, n_nodes):
    """
    For each node and symmetric edge in the graph, calculate a matrix indicating if the directed
    version of that edge originates at that node.
    Calculate one matrix for directed edges going from smaller to larger node indeces
    and one for directed edges going from larger to smaller node indices

    Parameters
    ----------
    edge_idx : array_like [2, ?]
        Sparse representation of the adjacency matrix
    n_nodes : int
        Number of nodes in the graph

    Returns
    -------
    list(array_like [n_nodes, ?])
    """

    edge_idx_directed = edge_idx[:, edge_idx[0] < edge_idx[1]]
    mask = np.zeros((2, n_nodes, edge_idx_directed.shape[1]))
    for node in range(n_nodes):
        mask[0, node, (edge_idx_directed[0] == node)] = 1
        mask[1, node, (edge_idx_directed[1] == node)] = 1

    return sp.sparse.csr_matrix(mask[0]), sp.sparse.csr_matrix(mask[1]), 


def calc_grid_pareto_points(grid):
    """
    Calculate for each node a set of points s.t. if the perceived perturbation
    exceeds any of the points, the corresponding classifier is successfully attacked.
    All nodes are pareto-optimal, i.e. there is no point with smaller coordinates
    in each dimension that also fulfills the property from the previous sentence.
    Returns matrix of these points and mask indicating which matrix
    row corresponds to which node.

    Parameters
    ----------
    grid : array_like
        0-1-valued certification grid of the base certificate

    Returns
    -------
    array_like [P, ?]
    array_like [n_nodes, P]
    """

    padded_grid = np.pad(grid, (0, 1))[:grid.shape[0]]  # Don't pad node dimension

    pareto_points_dict = {}
    num_pareto_points = 0

    for node in range(grid.shape[0]):
        candidates = np.vstack((~padded_grid[node]).nonzero()).T
        order = np.argsort(candidates.sum(axis=1))  # low-norm points should dominate many others
        candidates = candidates[order]

        dominated_mask = np.zeros(candidates.shape[0], dtype='bool')

        for i, candidate in enumerate(candidates):
            if not dominated_mask[i]:
                dominated_mask |= np.all(candidates >= candidate, axis=1)
                dominated_mask[i] = False

        num_pareto_points += np.sum(~dominated_mask)
        pareto_points_dict[node] = candidates[~dominated_mask]

    pareto_points = np.zeros((num_pareto_points, grid.ndim - 1))
    pareto_points_node_mask = np.zeros((grid.shape[0], num_pareto_points))

    c = 0
    for node in range(grid.shape[0]):
        points = pareto_points_dict[node]
        l = len(points)
        pareto_points[c:(c+l)] = points
        pareto_points_node_mask[node, c:(c+l)] = 1
        c += l

    return pareto_points, sp.sparse.csr_matrix(pareto_points_node_mask)


def calc_pareto_points_reachable_mask(pareto_points, dim_labels, radius):
    """
    Calculate a mask indicating which pareto-optimal points exist and can be reached
    at a given radius

    Parameters
    ----------
    pareto_points: array_like [P, ?]: 
        Pareto-points of base certificates of all predictions,
        where P is number of pareto points.
    dim_labels : list(str)
        List indicating for each budget-dimension if it is related to edge deletion,
        attribute addition or attribute deletion
    radius : array_like [3]
        Allowed number of edge deletions / attribute additions / attribute deletions

    Returns
    -------
    array_like [P]
    """

    edge_del_dims = np.array([label.startswith('adj_del') for label in dim_labels])
    attr_add_dims = np.array([label.startswith('attr_add') for label in dim_labels])
    attr_del_dims = np.array([label.startswith('attr_del') for label in dim_labels])

    per_dim_radius = np.zeros(len(dim_labels))
    per_dim_radius[edge_del_dims] = radius[0]
    per_dim_radius[attr_add_dims] = radius[1]
    per_dim_radius[attr_del_dims] = radius[2]

    return np.all(per_dim_radius >= pareto_points, axis=1).astype(int)


def generate_collective_problem(n_nodes, n_edges,
                                dim_labels, receptive_field_masks,
                                pareto_points, pareto_points_node_mask,
                                local_budgets=None,
                                sparse_directed_incidence_masks=None,
                                num_attackers=None,
                                radius=None, idx_test=None):

    """
    Generate the convex optimization describing the collective certificate.
    Returns radius and pareto-point-reachability parameters for efficient repeated evaluation.

    Parameters
    ----------
    n_nodes : int
        Number of nodes in the graph
    n_edges : int
        Number of undirected edges in the graph
    dim_labels : list(str)
        List indicating for each budget-dimension if it is related to edge deletion,
        attribute addition or attribute deletion
    receptive_field_masks : list(array_like [n_nodes, ?])
        List of equal length as dim_labels.
        Elements are binary matrices indicating for each node which nodes / edges
        should be summed over to determine perceived perturbation for each prediction.
    pareto_points: array_like [P, ?]: 
        Pareto-points of base certificates of all predictions,
        where P is number of pareto points.
    pareto_points_node_mask : array_like [n_nodes, ?]
        Indicates which pareto points belong to which node classifier
    local_budgets : array_like [3, n_nodes]
        How many edge deletions / attr additions / attr deletions are allowed at a specific node
    sparse_directed_incidence_masks : list(array_like [n_nodes, ?]) [2]
        Which directed edges in the graph originate at which nodes
    num_attackers : int
        Number of nodes the attacker has access to
        If None: All nodes are attacker-controlled
    radius : array_like [3]
        Number of allowed edge deletions / attribute additions / attribute deletions.
        If None, a cvxpy parameter is generated for repeated evaluation with different values
    idx_test: array_like [?]
        Node indices of predictions to be certified / targeted by the adversary..
        If None: All predictions are certified.

    Returns
    -------
    int
        Number of robust nodes
    cvxpy.Parameter [3]
        Radius parameter
    cvxpy.Parameter [P]
        Pareto-point reachability mask
    """

    num_pareto, n_dims = pareto_points.shape

    if radius is None:
        radius = cp.Parameter(3)

    # Budget allocation variables
    if local_budgets is not None:
        edge_del_directed = cp.Variable((2, n_edges), pos=True)
        edge_del = cp.sum(edge_del_directed, axis=0)
    else:
        edge_del = cp.Variable(n_edges, pos=True)
    attr_perturb = cp.Variable((n_nodes, 2), pos=True)

    # For each budget dimension: Sum up perturbation perceived by each prediction
    perceived_perturbs = []

    for dim_label, mask in zip(dim_labels, receptive_field_masks):
        if dim_label.startswith('attr_add'):
            perceived = mask @ attr_perturb[:, 0]
        elif dim_label.startswith('attr_del'):
            perceived = mask @ attr_perturb[:, 1]
        elif dim_label.startswith('adj_del'):
            perceived = mask @ edge_del
        perceived_perturbs.append(perceived)

    perceived_perturbs = cp.vstack(perceived_perturbs)

    # Check which pareto points are reached 
    # (i.e. perceived perturbation in each dimension matches/exceeds value of pareto point)
    pareto_point_dim_indicators = cp.Variable((perceived_perturbs.shape[0], num_pareto), pos=True)
    pareto_point_indicators = cp.Variable(num_pareto, pos=True)

    # Mask out pareto points that are guaranteed unreachable
    pareto_points_reachable_mask = cp.Parameter(num_pareto, boolean=True)

    attacked = cp.Variable(n_nodes, pos=True)

    constraints = [
        edge_del <= 1,
        cp.sum(edge_del) <= radius[0],
        cp.sum(attr_perturb, axis=0) <= radius[1:],
        pareto_point_dim_indicators <= 1,
        *[pareto_point_indicators <= pareto_point_dim_indicators[d] for d in range(n_dims)],
        pareto_points_node_mask @ cp.multiply(pareto_point_indicators, pareto_points_reachable_mask) >= attacked,
        attacked <= 1,
        perceived_perturbs @ pareto_points_node_mask >= cp.multiply(pareto_point_dim_indicators, pareto_points.T)
    ]

    if local_budgets is not None:
        if num_attackers is not None:

            attackers = cp.Variable(n_nodes, pos=True)

            constraints.extend([
                # Only attackers may manipulate edges
                ((sparse_directed_incidence_masks[0] @ edge_del_directed[0]
                 + sparse_directed_incidence_masks[1] @ edge_del_directed[1])
                 <= cp.multiply(local_budgets[0], attackers)),

                # For all nodes, edge changes must be within local budget constraint
                ((sparse_directed_incidence_masks[0] + sparse_directed_incidence_masks[1]) @ edge_del
                 <= local_budgets[0]),

                attr_perturb[:, 0] <= cp.multiply(local_budgets[1], attackers),
                attr_perturb[:, 1] <= cp.multiply(local_budgets[2], attackers),
                attackers <= 1,
                cp.sum(attackers) == num_attackers
            ])
        else:
            constraints.extend([
                ((sparse_directed_incidence_masks[0] + sparse_directed_incidence_masks[1]) @ edge_del
                 <= local_budgets[0]),
                attr_perturb[:, 0] <= local_budgets[1],
                attr_perturb[:, 1] <= local_budgets[2]
            ])

    if idx_test is None:
        obj = cp.sum(attacked)
    else:
        obj = cp.sum(attacked[idx_test])

    problem = cp.Problem(cp.Maximize(obj), constraints)

    return problem, radius, pareto_points_reachable_mask


def eval_collective_cert_problem(problem, radius_param, pareto_points_reachable_param,
                                 radius, pareto_points, dim_labels, n_nodes, solver='MOSEK', idx_test=None):
    """
    Evaluates collective certificate for a specific radius,
    returns number of certifiably robust nodes.

    Parameters
    ----------
    problem : cvxpy.Problem
        Optimization problem underlying the collective certificate
    radius_param: cvxpy.Parameter [3]:
        Problem parameter for deletions, attribute additions, attribute deletions
    pareto_points_reachable_param: cvxpy.Parameter [?]:
        Problem parameter to indicate which pareto points are unreachable
        (i.e. greater than radius / global budget)
    radius: array_like [3]
        Global budget for edge deletions, attribute additions, attribute deletions
    pareto_points: array_like [P, ?]: 
        Pareto-points of base certificates of all predictions,
        where P is number of pareto points.
    dim_labels : list(str)
        List indicating for each perceived-budget-dimension if it is related to edge deletion,
        attribute addition or attribute deletion
    n_nodes : int
        Number of nodes in the graph
    solver: str
        String specifying solver to be used by cvxpy
    idx_test: array_like [?]
        Node indices of predictions to be certified / targeted by the adversary..
        If None: All predictions are certified.

    Returns
    -------
    int
    """

    pareto_points_reachable_mask = calc_pareto_points_reachable_mask(pareto_points, dim_labels, radius)

    radius_param.value = radius
    pareto_points_reachable_param.value = pareto_points_reachable_mask

    if solver == 'MOSEK':
        import mosek

        num_attacked = problem.solve(
                solver='MOSEK',
                mosek_params={mosek.iparam.intpnt_solve_form: mosek.solveform.primal})

    else:
        num_attacked = problem.solve(solver=solver)

    if idx_test is None:
        if num_attacked > (n_nodes - 0.01):
            num_attacked = n_nodes
        num_robust = n_nodes - np.floor(num_attacked)
    else:
        if num_attacked > len(idx_test):
            num_attacked = len(idx_test)
        num_robust = len(idx_test) - np.floor(num_attacked)

    return num_robust


def collective_certificate_grid(base_grid, dim_labels,
                                receptive_field_masks,
                                attr, adj,
                                max_rad=None,
                                rad_stepsize=np.array([1, 1, 1]),
                                local_budget_descriptor=None, local_budget_mode='relative', num_attackers=None,
                                idx_test=None,
                                solver='MOSEK'):
    """
    Evaluate collective certificate on regular grid
    of global adversarial budgets.

    Parameters
    ----------
    base_grid : array_like [n_nodes, [? / ?, ? / ?, ?, ?]]
        Two- to four-dimensional boolean array of base certificate of each
        predictions. Each trailing dimension corresponds to one type of
        perturbation.
    dim_labels : list(str)
        List indicating for each budget-dimension if it is related to edge deletion,
        attribute addition or attribute deletion
    receptive_field_masks : list(array_like [n_nodes, ?])
        List of equal length as dim_labels.
        Elements are binary matrices indicating for each node which nodes / edges
        should be summed over to determine perceived perturbation for each prediction.
    attr: scipy.sparse_matrix [n_nodes, ?]
        Binary sparse attribute matrix of graph
    adj: scipy.spare_matrix [n_nodes, n_nodes]
        Binary sparse undirected adjacency matrix of graph
    max_rad: array_like [3]
        Maximum number of deletions / attr additions / attr deletions
        to evaluate the collective certificate at.
        If None: All elements are inferred from base certificate
        If entry is -1: Specific element is inferred from base certificate.
    rad_stepsize:
        Step size of grid between [0, 0, 0] and max_rad on which
        the collective certificate is evaluated.
    local_budget_descriptor : array_like [3]
        Absolute number or percentage of allowed edge deletions / attr additions / attr deletions
        at each node.
        If None: No local budget constraints are imposed
    local_budget_mode : str in ['relative', 'absolute']
        Whether the local budgets should be seen as absolute or as a percentage of the number of existing
        nodes / edges at a node
    num_attackers : int
        The number of nodes an attacker can have access to at once.
        If None: All nodes are attacker-controlled.
    idx_test: array_like [?]
        Node indices of predictions to be certified / targeted by the adversary.
        If None: All predictions are certified.

    Returns
    -------
    array_like [?, ?, ?]
    """

    edge_idx = np.stack(adj.nonzero())
    n_edges = (edge_idx[0] < edge_idx[1]).sum()
    n_nodes = adj.shape[0]

    # Pareto points characterizing each node's base certificate
    pareto_points, pareto_points_node_mask = calc_grid_pareto_points(base_grid)

    # Extra data for local budgets / limited #attackers
    if num_attackers is not None and local_budget_descriptor is None:
        local_budgets = np.ones(3)
        local_budget_mode = 'relative'

    if local_budget_descriptor is not None:
        local_budgets = calc_local_budgets(local_budget_descriptor, local_budget_mode, attr, adj)
        sparse_directed_incidence_masks = calc_sparse_directed_incidence_masks(edge_idx, n_nodes)
    else:
        local_budgets = None
        sparse_directed_incidence_masks = None

    # Set maximum global radii if not fully specified by user
    if max_rad is None:
        max_rad = calc_max_rads(pareto_points, dim_labels)
    elif -1 in max_rad: # If -1: Infer from largest radius in base cert
        max_rad_base = calc_max_rads(pareto_points, dim_labels)
        for i, m in enumerate(max_rad):
            if m != -1:
                max_rad_base[i] = max_rad[i]
        max_rad = max_rad_base
    max_rad = np.array(max_rad)

    # Eliminate pareto points that can never be reached within max_rad
    pareto_points_reachable_mask = calc_pareto_points_reachable_mask(pareto_points, dim_labels, max_rad)
    pareto_points = pareto_points[pareto_points_reachable_mask != 0]
    pareto_points_node_mask = pareto_points_node_mask[:, pareto_points_reachable_mask != 0]

    # Set up grid of budgets to iterate over
    edge_del_vals = np.arange(0, max_rad[0] + 1, rad_stepsize[0])
    attr_add_vals = np.arange(0, max_rad[1] + 1, rad_stepsize[1])
    attr_del_vals = np.arange(0, max_rad[2] + 1, rad_stepsize[2])

    cert_grid = np.empty((len(edge_del_vals), len(attr_add_vals), len(attr_del_vals)), dtype='int')

    # Define parametric problem for more efficient repeated evaluation
    problem, radius_param, pareto_points_reachable_param = generate_collective_problem(
        n_nodes, n_edges, dim_labels, receptive_field_masks,
        pareto_points, pareto_points_node_mask,
        local_budgets=local_budgets, num_attackers=num_attackers,
        sparse_directed_incidence_masks=sparse_directed_incidence_masks,
        idx_test=idx_test)

    # Evaluate collective cert for every value in cert_grid
    for edge_del, attr_add, attr_del in itertools.product(edge_del_vals, attr_add_vals, attr_del_vals):
        radius = np.array([edge_del, attr_add, attr_del])

        num_robust = eval_collective_cert_problem(problem,
                                                  radius_param, pareto_points_reachable_param,
                                                  radius, pareto_points, dim_labels,
                                                  n_nodes, solver=solver, idx_test=None)

        cert_grid[edge_del_vals == edge_del,
                  attr_add_vals == attr_add,
                  attr_del_vals == attr_del] = num_robust

    return cert_grid
