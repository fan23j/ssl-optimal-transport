import torch

def tensor_product(constC, hC1, hC2, T):
    r"""Return the tensor for Gromov-Wasserstein fast computation

    The tensor is computed as described in Proposition 1 Eq. (6) in :ref:`[12] <references-tensor-product>`

    Parameters
    ----------
    constC : tensor, shape (ns, nt)
        Constant :math:`\mathbf{C}` matrix in Eq. (6)
    hC1 : tensor, shape (ns, ns)
        :math:`\mathbf{h1}(\mathbf{C1})` matrix in Eq. (6)
    hC2 : tensor, shape (nt, nt)
        :math:`\mathbf{h2}(\mathbf{C2})` matrix in Eq. (6)
    T : tensor
        Tensor T used in tensor-matrix multiplication
    Returns
    -------
    tens : tensor, shape (`ns`, `nt`)
        :math:`\mathcal{L}(\mathbf{C_1}, \mathbf{C_2}) \otimes \mathbf{T}` tensor-matrix multiplication result

    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    """
    
    A = - torch.mm(
        torch.mm(hC1, T), hC2.t()
    )
    tens = constC + A
    return tens

def gwgrad(constC, hC1, hC2, T):
    r"""Return the gradient for Gromov-Wasserstein

    The gradient is computed as described in Proposition 2 in :ref:`[12] <references-gwggrad>`

    Parameters
    ----------
    constC : tensor, shape (ns, nt)
        Constant :math:`\mathbf{C}` matrix in Eq. (6)
    hC1 : tensor, shape (ns, ns)
        :math:`\mathbf{h1}(\mathbf{C1})` matrix in Eq. (6)
    hC2 : tensor, shape (nt, nt)
        :math:`\mathbf{h2}(\mathbf{C2})` matrix in Eq. (6)
    T : tensor, shape (ns, nt)
        Current value of transport matrix :math:`\mathbf{T}`

    Returns
    -------
    grad : tensor, shape (`ns`, `nt`)
        Gromov-Wasserstein gradient

    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    """
    return 2 * tensor_product(constC, hC1, hC2, T)  # [12] Prop. 2 misses a 2 factor

def init_matrix(C1, C2, p, q, loss_fun='square_loss'):
    r"""Return loss matrices and tensors for Gromov-Wasserstein fast computation

    Returns the value of :math:`\mathcal{L}(\mathbf{C_1}, \mathbf{C_2}) \otimes \mathbf{T}` with the
    selected loss function as the loss function of Gromov-Wasserstein discrepancy.

    The matrices are computed as described in Proposition 1 in :ref:`[12] <references-init-matrix>`

    Where :

    - :math:`\mathbf{C_1}`: Metric cost matrix in the source space
    - :math:`\mathbf{C_2}`: Metric cost matrix in the target space
    - :math:`\mathbf{T}`: A coupling between those two spaces

    The square-loss function :math:`L(a, b) = |a - b|^2` is read as :

    .. math::

        L(a, b) = f_1(a) + f_2(b) - h_1(a) h_2(b)

        \mathrm{with} \ f_1(a) &= a^2

                        f_2(b) &= b^2

                        h_1(a) &= a

                        h_2(b) &= 2b

    The kl-loss function :math:`L(a, b) = a \log\left(\frac{a}{b}\right) - a + b` is read as :

    .. math::

        L(a, b) = f_1(a) + f_2(b) - h_1(a) h_2(b)

        \mathrm{with} \ f_1(a) &= a \log(a) - a

                        f_2(b) &= b

                        h_1(a) &= a

                        h_2(b) &= \log(b)

    Parameters
    ----------
    C1 : array-like, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : array-like, shape (nt, nt)
        Metric cost matrix in the target space
    p : array-like, shape (ns,)
        Probability distribution in the source space
    q : array-like, shape (nt,)
        Probability distribution in the target space
    loss_fun : str, optional
        Name of loss function to use: either 'square_loss' or 'kl_loss' (default='square_loss')
    nx : backend, optional
        If let to its default value None, a backend test will be conducted.
    Returns
    -------
    constC : array-like, shape (ns, nt)
        Constant :math:`\mathbf{C}` matrix in Eq. (6)
    hC1 : array-like, shape (ns, ns)
        :math:`\mathbf{h1}(\mathbf{C1})` matrix in Eq. (6)
    hC2 : array-like, shape (nt, nt)
        :math:`\mathbf{h2}(\mathbf{C2})` matrix in Eq. (6)


    .. _references-init-matrix:
    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    """

    if loss_fun == 'square_loss':
        f1 = lambda a: a**2
        f2 = lambda b: b**2
        h1 = lambda a: a
        h2 = lambda b: 2*b
    elif loss_fun == 'kl_loss':
        f1 = lambda a: a * torch.log(a + 1e-15) - a
        f2 = lambda b: b
        h1 = lambda a: a
        h2 = lambda b: torch.log(b + 1e-15)
    
    constC1 = torch.matmul(f1(C1), torch.matmul(torch.reshape(p, (-1, 1)), torch.ones(1, len(q)).type_as(q)))
    constC2 = torch.matmul(torch.ones(len(p), 1).type_as(p), torch.matmul(torch.reshape(q, (1, -1)), f2(C2).T))
    constC = constC1 + constC2
    hC1 = h1(C1)
    hC2 = h2(C2)

    return constC, hC1, hC2
