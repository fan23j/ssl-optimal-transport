# Reference: https://github.com/PythonOT/POT/blob/5faa4fbdb1a64351a42d31dd6f54f0402c29c405/ot/gromov/_bregman.py#L22
# Translated to pytorch

import torch
from .utils import tensor_product, gwgrad, init_matrix

def entropic_gromov_wasserstein(C1, C2, p, q, loss_fun, epsilon, symmetric=None, G0=None,
                                max_iter=1000, sinkhorn_max_iter=1000, tol=1e-9, verbose=False, log=False):
    r"""
    Returns the gromov-wasserstein transport between :math:`(\mathbf{C_1}, \mathbf{p})` and :math:`(\mathbf{C_2}, \mathbf{q})`

    The function solves the following optimization problem:

    .. math::
        \mathbf{GW} = \mathop{\arg\min}_\mathbf{T} \quad \sum_{i,j,k,l} L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l} - \epsilon(H(\mathbf{T}))

        s.t. \ \mathbf{T} \mathbf{1} &= \mathbf{p}

             \mathbf{T}^T \mathbf{1} &= \mathbf{q}

             \mathbf{T} &\geq 0

    Where :

    - :math:`\mathbf{C_1}`: Metric cost matrix in the source space
    - :math:`\mathbf{C_2}`: Metric cost matrix in the target space
    - :math:`\mathbf{p}`: distribution in the source space
    - :math:`\mathbf{q}`: distribution in the target space
    - `L`: loss function to account for the misfit between the similarity matrices
    - `H`: entropy

    .. note:: If the inner solver `ot.sinkhorn` did not convergence, the
        optimal coupling :math:`\mathbf{T}` returned by this function does not
        necessarily satisfy the marginal constraints
        :math:`\mathbf{T}\mathbf{1}=\mathbf{p}` and
        :math:`\mathbf{T}^T\mathbf{1}=\mathbf{q}`. So the returned
        Gromov-Wasserstein loss does not necessarily satisfy distance
        properties and may be negative.

    Parameters
    ----------
    C1 : array-like, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : array-like, shape (nt, nt)
        Metric cost matrix in the target space
    p :  array-like, shape (ns,)
        Distribution in the source space
    q :  array-like, shape (nt,)
        Distribution in the target space
    loss_fun :  string
        Loss function used for the solver either 'square_loss' or 'kl_loss'
    epsilon : float
        Regularization term >0
    symmetric : bool, optional
        Either C1 and C2 are to be assumed symmetric or not.
        If let to its default None value, a symmetry test will be conducted.
        Else if set to True (resp. False), C1 and C2 will be assumed symmetric (resp. asymmetric).
    G0: array-like, shape (ns,nt), optional
        If None the initial transport plan of the solver is pq^T.
        Otherwise G0 must satisfy marginal constraints and will be used as initial transport of the solver.
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        Record log if True.

    Returns
    -------
    T : array-like, shape (`ns`, `nt`)
        Optimal coupling between the two spaces

    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    .. [47] Chowdhury, S., & Mémoli, F. (2019). The gromov–wasserstein
        distance between networks and stable network invariants.
        Information and Inference: A Journal of the IMA, 8(4), 757-787.
    """
    if G0 is None:
        G0 = torch.outer(p, q)
    else:
        G0 = torch.tensor(G0) if not torch.is_tensor(G0) else G0

    T = G0
    constC, hC1, hC2 = init_matrix(C1, C2, p, q, loss_fun)

    cpt = 0
    err = 1

    while (err > tol and cpt < max_iter):

        Tprev = T

        tens = gwgrad(constC, hC1, hC2, T)
        T = sinkhorn_knopp(p, q, tens, epsilon, numItermax=sinkhorn_max_iter, verbose=True)

        if cpt % 10 == 0:
            err = torch.norm(T - Tprev)

        cpt += 1

    return T

def sinkhorn_knopp(a, b, M, reg, numItermax=1000, stopThr=1e-9, verbose=False, log=False, warn=True, warmstart=None, **kwargs):
    if len(a) == 0:
        a = torch.full((M.shape[0],), 1.0 / M.shape[0], dtype=M.dtype, device=M.device)
    if len(b) == 0:
        b = torch.full((M.shape[1],), 1.0 / M.shape[1], dtype=M.dtype, device=M.device)

    # init data
    dim_a = len(a)
    dim_b = b.shape[0]

    u = torch.ones(dim_a, dtype=M.dtype, device=M.device) / dim_a
    v = torch.ones(dim_b, dtype=M.dtype, device=M.device) / dim_b

    K = torch.exp(M / (-reg))

    Kp = (1 / a).reshape(-1, 1) * K

    err = 1
    for ii in range(numItermax):
        uprev = u
        vprev = v
        KtransposeU = K.t() @ u
        v = b / KtransposeU
        u = 1. / (Kp @ v)

        if (torch.any(KtransposeU == 0)
                or torch.any(torch.isnan(u)) or torch.any(torch.isnan(v))
                or torch.any(torch.isinf(u)) or torch.any(torch.isinf(v))):
            u = uprev
            v = vprev
            break
        if ii % 10 == 0:
            tmp2 = torch.einsum('i,ij,j->j', u, K, v)
            err = torch.norm(tmp2 - b)  # violation of marginal

            if err < stopThr:
                break
    
    return u.reshape((-1, 1)) * K * v.reshape((1, -1))