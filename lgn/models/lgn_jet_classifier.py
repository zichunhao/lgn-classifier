import torch
import logging
import sys
sys.path.insert(1, '../..')

from lgn.cg_lib import CGModule, ZonalFunctionsRel, ZonalFunctions, normsq4
from lgn.g_lib import GTau

from lgn.models.lgn_cg import LGNCG

from lgn.nn import RadialFilters
from lgn.nn import InputLinear, MixReps
from lgn.nn import OutputLinear, OutputPMLP, GetScalarsNode
from lgn.nn import NoLayer

class LGNJetClassifier(CGModule):

    """
    LGN Network for classify jets in hls4ml data.

    Parameters
    ----------
    maxdim : list of int
        Maximum weight in the output of CG products. (Expanded to list of
        length num_cg_levels)
    max_zf : list of int
        Maximum weight in the output of the spherical harmonics  (Expanded to list of
        length num_cg_levels)
    num_cg_levels : int
        Number of cg levels to use.
    num_channels : list of int
        Number of channels that the output of each CG are mixed to (Expanded to list of
        length num_cg_levels)
    weight_init : str
        The type of weight initialization. The choices are 'randn' and 'rand'.
    level_gain : list of floats
        The gain at each level. (args.level_gain = [1.])
    num_basis_fn : int
        The number of basis function to use.
    output_layer : str
        The output layer to use. Choices are
            - linear: for lgn.OutputLinear. (default in args)
            - MLP: for lgn.OutputMLP.
    num_mpnn_layers : int
        The number of MPNN layers
    num_classes : int
        Optional, default: 5, for g/q/t/w/z
        One-hot encoding:
            - g jet: [1, 0, 0, 0, 0]
            - q jet: [0, 1, 0, 0, 0]
            - t jet: [0, 0, 1, 0, 0]
            - w jet: [0, 0, 0, 1, 0]
            - z jet: [0, 0, 0, 0, 1]
        The number of jet classes to classify.
    activation : str
        Optional, default: 'leakyrelu'
        The activation function for lgn.LGNCG
    p4_into_CG : bool
        Optional, default: False
        Whether or not to feed in 4-momenta themselves to the first CG layer,
        in addition to scalars.
            - If true, MixReps will be used for the input linear layer of the model.
            - If false, IntputLinear will be used.
    add_beams : bool
        Optional, default: False
        Append two proton beams of the form (m^2,0,0,+-1) to each event
    scale : float or int
        Scaling parameter for node features.
    full_scalars : bool
        Optional, default: True
        If True, we will construct a more complete set of scalar invariants from the full
        GVec by constructing its norm squared (using the normsq function).
        If False, we just extract the (0,0) component of node features.
    mlp : bool
        Optional, default: True
        Whether to include the extra MLP layer on scalar features in nodes.
    mlp_depth : int
        Optional, default: None
        The number of hidden layers in CGMLP.
    mlp_width : list of int
        Optional, default: None
        The number of perceptrons in each CGMLP layer
    device : torch.device
        Optional, default: None, in which case it will be set to
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        The device to which the module is initialized.
    dtype : torch.dtype
        Optional, default: torch.float64
        The data type to which the module is initialized.
    cg_dict : CGDict
        Optional, default: None
        Clebsch-gordan dictionary for taking the CG decomposition.
    """
    def __init__(self, maxdim, max_zf, num_cg_levels, num_channels,
                 weight_init, level_gain, num_basis_fn,
                 output_layer, num_mpnn_layers, num_classes=5,
                 activation='leakyrelu', p4_into_CG=True, add_beams=False,
                 scale=1., full_scalars=True, mlp=True, mlp_depth=None, mlp_width=None,
                 device=None, dtype=torch.float64, cg_dict=None):

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if dtype is None:
            dtype = torch.float64

        logging.info(f'Initializing network with device: {device} and dtype: {dtype}')

        level_gain = expand_var_list(level_gain, num_cg_levels)
        maxdim = expand_var_list(maxdim, num_cg_levels)
        max_zf = expand_var_list(max_zf, num_cg_levels)
        num_channels = expand_var_list(num_channels, num_cg_levels)

        # loggings
        if output_layer.lower().startswith('lin'):
            output_layer_str = 'OutputLinear'
        elif output_layer.lower().startswith('pmlp'):
            output_layer_str = 'OutputPMLP'
        else:
            output_layer_str = 'Failed to implement'

        logging.info(f'maxdim: {maxdim}')
        logging.info(f'max_zf: {max_zf}')
        logging.info(f'num_cg_levels: {num_cg_levels}')
        logging.info(f'num_channels: {num_channels}')
        logging.info(f'number of basis function for radial functions: {num_basis_fn}')
        logging.info(f"input layer type: {'InputLinear' if not p4_into_CG else 'MixReps'}")
        logging.info(f"output layer type: {output_layer_str}")

        super().__init__(maxdim=max(maxdim + max_zf), device=device, dtype=dtype, cg_dict=cg_dict)

        self.num_cg_levels = num_cg_levels
        self.num_channels = num_channels
        self.scale = scale
        self.num_classes = num_classes
        self.full_scalars = full_scalars
        self.p4_into_CG = p4_into_CG

        # spherical harmonics
        if p4_into_CG:
            # Express input momenta in the bases of spherical harmonics
            self.zonal_fns_in = ZonalFunctions(max(max_zf), dtype=dtype,
                                               device=device, cg_dict=cg_dict)
        # relative position in momentum space
        self.zonal_fns = ZonalFunctionsRel(max(max_zf), dtype=dtype,
                                           device=device, cg_dict=cg_dict)

        # Position functions
        self.rad_funcs = RadialFilters(max_zf, num_basis_fn, num_channels, num_cg_levels,
                                       device=device, dtype=dtype)
        tau_pos = self.rad_funcs.tau

        if num_cg_levels:
            if add_beams:
                num_scalars_in = 2
            else:
                num_scalars_in = 1
        else:
            num_scalars_in = 150+2  # number of particles per jet (after padding)

        num_scalars_out = num_channels[0]

        # Input linear layer: self.input_func_node
        if not num_cg_levels:
            self.input_func_node = InputLinear(num_scalars_in, num_scalars_out,
                                               device=device, dtype=dtype)
        else:
            tau_in = GTau({**{(0,0): num_scalars_in}, **{(l,l): 1 for l in range(1, max_zf[0] + 1)}})
            tau_out = GTau({(l,l): num_scalars_out for l in range(max_zf[0] + 1)})
            self.input_func_node = MixReps(tau_in, tau_out, device=device, dtype=dtype)

        tau_input_node = self.input_func_node.tau

        # CG layers
        self.lgn_cg = LGNCG(maxdim, max_zf, tau_input_node, tau_pos,
                            num_cg_levels, num_channels, level_gain, weight_init,
                            mlp=mlp, mlp_depth=mlp_depth, mlp_width=mlp_width,
                            activation=activation, device=self.device, dtype=self.dtype, cg_dict=self.cg_dict)

        tau_cg_levels_node = self.lgn_cg.tau_levels_node

        self.get_scalars_node = GetScalarsNode(tau_cg_levels_node, device=self.device, dtype=self.dtype)

        num_scalars_node = self.get_scalars_node.num_scalars

        # Output linear level
        if output_layer.lower().startswith('lin'):
            self.output_layer_node = OutputLinear(num_scalars_node, num_classes=self.num_classes,
                                                  bias=True, device=self.device, dtype=self.dtype)
        elif output_layer.lower().startswith('pmlp'):
            self.output_layer_node = OutputPMLP(num_scalars_node, num_classes=num_classes,
                                                num_mixed=mlp_width, device=self.device, dtype=self.dtype)

        # logging
        logging.info(f'Model initialized. Number of parameters: {sum(p.nelement() for p in self.parameters())}')


    """
    Forward pass of the classifier

    Parameters
    ----------
    data : dict
        Dictionary of data to pass to the network.
    covariance_test : bool
        Optional, default: False
        If False, return prediction (scalar reps) only.
        If True, return both predictions and full node features, where the full node features
        will be used to test Lorentz covariance.

    Returns
    -------
    prediction : torch.Tensor
        The one-hot encoding prediction for jet type.
    """
    def forward(self, data, covariance_test=False):
        # Get data
        node_scalars, node_ps, node_mask, edge_mask = self.prepare_input(data, self.num_cg_levels)

        # Calculate Zonal functions
        if self.p4_into_CG:
            zonal_functions_in, _, _ = self.zonal_fns_in(node_ps)
            # all input are so far reals, so [real, imaginary] = [scalars, 0]
            zonal_functions_in[(0, 0)] = torch.stack([node_scalars.unsqueeze(-1), torch.zeros_like(node_scalars.unsqueeze(-1))])
        zonal_functions, norms, sq_norms = self.zonal_fns(node_ps, node_ps)

        # Input layer
        if self.num_cg_levels > 0:
            rad_func_levels = self.rad_funcs(norms, edge_mask * (norms != 0).byte())
            # Feed scalars only
            if not self.p4_into_CG:
                node_reps_in = self.input_func_node(node_scalars, node_mask)
            # Feed both scalars and 4-momenta
            else:
                node_reps_in = self.input_func_node(zonal_functions_in)
        else:
            rad_func_levels = []
            node_reps_in = self.input_func_node(node_scalars, node_mask)


        # CG layer
        nodes_all = self.lgn_cg(node_reps_in, node_mask, rad_func_levels, zonal_functions)
        # Project reps into scalars
        node_scalars = self.get_scalars_node(nodes_all)

        # Output layer
        prediction = self.output_layer_node(node_scalars)

        if covariance_test:
            return prediction, nodes_all
        else:
            return prediction

    """
    Extract input from data.

    Parameters
    ----------
    data : dict
        The jet data.

    Returns
    -------
    node_scalars : torch.Tensor
        Tensor of scalars for each node.
    node_ps: :torch.Tensor
        Momenta of the nodes
    node_mask : torch.Tensor
        Node mask used for batching data.
    edge_mask: torch.Tensor
        Edge mask used for batching data.
    """
    def prepare_input(self, data, cg_levels=True):

        node_ps = data['p4'].to(device=self.device, dtype=self.dtype) * self.scale

        data['p4'].requires_grad_(True)

        node_mask = data['node_mask'].to(device=self.device, dtype=torch.uint8)
        edge_mask = data['edge_mask'].to(device=self.device, dtype=torch.uint8)

        scalars = torch.ones_like(node_ps[:,:, 0]).unsqueeze(-1)
        scalars = normsq4(node_ps).abs().sqrt().unsqueeze(-1)

        if 'scalars' in data.keys():
            scalars = torch.cat([scalars, data['scalars'].to(device=self.device, dtype=self.dtype)], dim=-1)

        if not cg_levels:
            scalars = torch.stack(tuple(scalars for _ in range(scalars.shape[-1])), -2).to(device=self.device, dtype=self.dtype)

        return scalars, node_ps, node_mask, edge_mask

"""
Expand variables in a list

Parameters
----------
var : list, int, or float
    The variables
num_cg_levels : int
    Number of cg levels to use.

Return
------
var_list : list
    The list of variables. The length will be num_cg_levels.
"""
def expand_var_list(var, num_cg_levels):
    if type(var) == list:
        var_list = var + (num_cg_levels - len(var)) * [var[-1]]
    elif type(var) in [float, int]:
        var_list = [var] * num_cg_levels
    else:
        raise ValueError(f'Incorrect type of variables: {type(var)}. \
                         The allowed data types are list, float, or int')
    return var_list
