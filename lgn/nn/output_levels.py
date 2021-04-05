import torch
import torch.nn as nn

from lgn.nn import BasicMLP
from lgn.g_lib import cat
from lgn.cg_lib import normsq


################################# Get Scalars #################################
class GetScalarsNode(nn.Module):
    """
    Construct a set of scalar features for each node by using the
    covariant node GVec representations at various levels.

    Parameters
    ----------
    tau_levels : list of GTau
        Multiplicities of the output GVec at each level.
    full_scalars : bool
        Optional, default: True
        If True, we will construct a more complete set of scalar invariants from the full
        GVec by constructing its norm squared (using the normsq function).
        If False, we just extract the (0,0) component of node features.
    device : torch.device
        Optional, default: None, in which case we will use
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        The device to which the module is initialized.
    dtype : torch.dtype
        Optional, default: torch.float64
        The data type to which the module is initialized.
    """
    def __init__(self, tau_levels, full_scalars=True, device=None, dtype=torch.float64):

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        super().__init__()

        self.device = device
        self.dtype = dtype

        self.maxdim = max(len(tau) for tau in tau_levels) - 1

        self.full_scalars = full_scalars
        if full_scalars:
            self.split = [sum(tau.values()) for tau in tau_levels]
            self.num_scalars = sum(self.split)
        else:
            self.split = [tau[(0, 0)] for tau in tau_levels]  # multiplicity of scalar features
            self.num_scalars = sum(self.split)

    """
    Forward step for :class:`GetScalarsNode`

    Parameters
    ----------
    reps_all_levels list of GVec
        List of scalar node features at each level

    Returns
    -------
    scalars : torch.Tensor
        Invariant scalar node features constructed from reps_all_levels
    """
    def forward(self, reps_all_levels):
        reps = cat(reps_all_levels)
        scalars = reps.pop((0, 0))

        if self.full_scalars and len(reps.keys())>0:
            scalars_full = list(normsq(reps).values())
            scalars = [scalars] + scalars_full
            scalars = torch.cat(scalars, dim=reps.cdim)
        return scalars


############################### Output of network ###############################
class OutputLinear(nn.Module):
    """
    Module to create prediction based upon a set of scalar node features.
    This is used when full_scalars is False.
    Node features are summed over to ensure permutation invariance.

    Parameters
    ----------
    num_scalars : int
        Number scalars that will be used in the prediction at the output
        of the network.
    num_classes : int
        Optional, default: 5
        The number of classes to classify the jet.
        In this case, the number of classes is 5: g/q/t/w/z with one-hot encoding:
            - g jet: [1, 0, 0, 0, 0]
            - q jet: [0, 1, 0, 0, 0]
            - t jet: [0, 0, 1, 0, 0]
            - w jet: [0, 0, 0, 1, 0]
            - z jet: [0, 0, 0, 0, 1]
    bias : bool
        Optional, default: True
        Whether to include a bias term in the linear mixing level.
    device : torch.device
        Optional, default: None, in which case it will be set to
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        The device to which the module is initialized.
    dtype : torch.dtype
        Optional, default: torch.float64
        The data type to which the module is initialized.
    """
    def __init__(self, num_scalars, num_classes=5, bias=True, device=None, dtype=torch.float64):

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        super(OutputLinear, self).__init__()

        self.num_scalars = num_scalars
        self.bias = bias
        if num_scalars > 0:
            self.lin = nn.Linear(2 * num_scalars, num_classes, bias=bias)
            self.lin.to(device=device, dtype=dtype)
        self.softmax = nn.Softmax(dim=1)
        self.zero = torch.tensor(0, dtype=dtype, device=device)

    def forward(self, scalars):
        """
        Forward step for Outputlinear

        Parameter
        ----------
        node_scalars : torch.Tensor
            Scalar features for each node used to predict the final learning target.

        Return
        -------
        predict : torch.Tensor
            Tensor used for predictions. One-hot encoding is used
        """
        if self.num_scalars > 0:
            batch_size = scalars.shape[1]
            # sum over all nodes to ensure permutation invariance
            scalars = scalars.sum(2).permute(1, 2, 3, 0)
            # put the complex dimension at the end and collapse into one dimension of scalars
            scalars = scalars.contiguous().view((batch_size, -1))
            # apply linear mixing to scalars in each event
            predict = self.lin(scalars)
        else:
            predict = scalars

        predict = self.softmax(predict)

        return predict


class OutputPMLP(nn.Module):
    """
    Module to create prediction based upon scalars constructed from full node features.

    This is peformed in a three-step process:
        (1) A MLP is applied to each set of scalar node-features.
        (2) The environments are summed up.
        (3) Another MLP is applied to the output to predict a single learning target.

    Parameters
    ----------
    num_scalars : int
        Number scalars that will be used in the prediction at the output
        of the network.
    num_classes : int
        Optional, default: 5
        The number of classes to classify the jet.
        In this case, the number of classes is 5: g/q/t/w/z with one-hot encoding:
            - g jet: [1, 0, 0, 0, 0]
            - q jet: [0, 1, 0, 0, 0]
            - t jet: [0, 0, 1, 0, 0]
            - w jet: [0, 0, 0, 1, 0]
            - z jet: [0, 0, 0, 0, 1]
    num_mixed : int
        Optional, default: 2
    activation : str
    device : torch.device
        Optional, default: None, in which case it will be set to
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        The device to which the module is initialized.
    dtype : torch.dtype
        Optional, default: torch.float64
        The data type to which the module is initialized.
    """
    def __init__(self, num_scalars, num_classes=5, num_mixed=2, activation='leakyrelu', device=None, dtype=torch.float64):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        super(OutputPMLP, self).__init__()

        self.num_scalars = num_scalars
        self.num_mixed = num_mixed

        # mlp applied on scalar features
        self.mlp1 = BasicMLP(2*num_scalars, num_scalars * num_mixed, num_hidden=1, activation=activation, device=device, dtype=dtype)
        self.mlp2 = BasicMLP(num_scalars * num_mixed, num_classes, num_hidden=1, activation=activation, device=device, dtype=dtype)
        self.softmax = nn.Softmax(dim=1)

        self.zero = torch.tensor(0, device=device, dtype=dtype)

    def forward(self, node_scalars, node_mask):
        """
        Forward pass for OutputPMLP.

        Parameters
        ----------
        node_scalars : torch.Tensor
            Scalar features for each node used to predict the final learning target.
        node_mask : torch.Tensor
            The mask for node features.
            This is unused and is included only for pedagogical purposes.

        Return
        -------
        predict : torch.Tensor
            Tensor used for predictions. One-hot encoding is used.
        """
        # Reshape
        node_shape = node_scalars.permute(1,2,3,4,0).shape[:2] + (2*self.num_scalars,)
        node_scalars = node_scalars.view(node_shape)

        # First MLP applied to each node
        x = self.mlp1(node_scalars)

        # Reshape to sum over each node in graphs, setting non-existent nodes to zero.
        node_mask = node_mask.unsqueeze(-1)
        x = torch.where(node_mask, x, self.zero).sum(1)

        # Prediction on permutation invariant representation of graphs
        predict = self.mlp2(x)

        predict = self.softmax(predict)

        return predict
