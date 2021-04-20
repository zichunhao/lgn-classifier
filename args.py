import argparse
from math import inf

def setup_argparse():
    parser = argparse.ArgumentParser(description='LGN classifier options')

    # loading data
    parser.add_argument('--file-path', type=str, default='hls4ml.pt', metavar='N',
                        help='The path of the data.')
    parser.add_argument('--num-classes', type=int, default=5, metavar='N',
                        help='The number of jet types. Default: 5.')
    parser.add_argument('--class-labels', type=str, default=['g', 'q', 't', 'w', 'z'], metavar='N',
                        help='The jet types.')
    parser.add_argument('--num-train', type=int, default=528000, metavar='N',
                        help='Number of samples to train on. (default: 528000)')
    parser.add_argument('--num-val', type=int, default=-1, metavar='N',
                        help='Number of samples to validate on. (default: -1)')
    parser.add_argument('--num-test', type=int, default=-1, metavar='N',
                        help='Number of samples to test eqvuivariance on. (default: -1)')

    # training parameters
    parser.add_argument('--num-epochs', type=int, default=64, metavar='N',
                        help='Number of epochs to train. Default: 64')
    parser.add_argument('--batch-size', '-bs', type=int, default=16, metavar='N',
                        help='The batch size. Default: 32')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='N',
                        help='Initial learning rate. Default: 0.001')

    # neural network constructions
    parser.add_argument('--maxdim', nargs='*', type=int, default=[4], metavar='N',
                        help='Cutoff (maximum weight) in the Clebsch-Gordon operations. Default: [3]')
    parser.add_argument('--max-zf', nargs='*', type=int, default=[1], metavar='N',
                        help='Maximum weight of spherical harmonics to use. Default: [1]')
    parser.add_argument('--num-cg-levels', type=int, default=3, metavar='N',
                        help='Number of CG layers (default: 3)')
    parser.add_argument('--num-channels', nargs='*', type=int, default=[2, 3, 4, 3], metavar='N',
                        help='Number of channels to allow after mixing. Default: [2, 3, 4, 3]')
    parser.add_argument('--weight-init', type=str, default='randn', metavar='str',
                        help='Weight initialization function to use. Default: randn')
    parser.add_argument('--num-basis-fn', type=int, default=16, metavar='N',
                        help='Number of basis functons (default: 10)')
    parser.add_argument('--level-gain', nargs='*', type=float, default=[1.], metavar='N',
                        help='Gain at each level (default: [1.])')
    parser.add_argument('--output-layer', type=str, default='linear', metavar='N',
                        help='The output layer type to use: linear or mlp. Default: linear')
    parser.add_argument('--num-mpnn-layers', type=int, default=1,
                        help='Number layers to use in InputMPNN layer. Default: 1')
    parser.add_argument('--activation', type=str, default='leakyrelu',
                        help='Activation function used in MLP layers. Options: (relu, elu, leakyrelu, sigmoid, logsigmoid). Default: elu.')
    parser.add_argument('--p4_into_CG', action=BoolArg, default=True,
                        help='Feed 4-momenta themselves to the first CG layer, in addition to scalars. Default: True')
    parser.add_argument('--add-beams', action=BoolArg, default=False,
                        help='Whether to two proton beams of the form (m^2,0,0,+-1) to each event. Default: False')
    parser.add_argument('--full-scalars', action=BoolArg, default=True,
                        help='Wehther to feed the norms of ALL irrep tensors (as opposed to just the Loretnz scalar irreps) at each level into the output layer. Default: True')
    parser.add_argument('--mlp', action=BoolArg, default=True,
                        help='Whether to insert a perceptron acting on invariant scalars inside each CG level. Default: True')
    parser.add_argument('--mlp-depth', type=int, default=3, metavar='N',
                        help='Number of hidden layers in each MLP. Default: 3')
    parser.add_argument('--mlp-width', type=int, default=2, metavar='N',
                        help='Width of hidden layers in each MLP in units of the number of inputs. Default: 2')
    parser.add_argument('--dtype', type=str, default='float64', metavar='N',
                        help='The data type the model is initialized to. Default: float64')


    # trainin options
    parser.add_argument("--load-to-train", action="store_true", default=False,
                        help="Whether to load the trained model to continue training.")
    parser.add_argument("--load-model-path", type=str, default=None,
                        help="The path of the model to load.")
    parser.add_argument("--load-epoch", type=int, default=1,
                        help="The epoch number to load.")
    parser.add_argument("--test-equivariance", action=BoolArg, default=True,
                        help="Whether to test the equivariance of the trainined model. Default: True")
    parser.add_argument("--test-over-all-epochs", action=BoolArg, default=True,
                        help="Whether to test the equivariance in all epochs. If False, only the last epoch model will be tested. Default: False")
    parser.add_argument("--outpath", type=str, default = 'trained_models',
                        help="Output folder, in which loggings, models, and figures are stored.")
    parser.add_argument("--patience", type=int, default=10,
                        help="Patience before early stopping. Default: 10")
    parser.add_argument('--print-logging', action=BoolArg, default=True,
                        help='Whether to print all logging info. Default: False')
    """
    parser.add_argument("--train", action=BoolArg, default=True,
                        help="Whether to train the model. Default: True")
    parser.add_argument("--load-to-test", action=BoolArg, default=False,
                        help="Whether to load the trained model for testing.")
    """

    # results saving options
    parser.add_argument('--fig-format', type=str, default='pdf', metavar='str',
                        help='The format of output figures. Default: pdf')

    args = parser.parse_args()

    return args

### From https://github.com/fizisist/LorentzGroupNetwork/blob/master/src/lgn/engine/args.py
# Take an argparse argument that is either a bool or a str and return a boolean.
class BoolArg(argparse.Action):

    def __init__(self, default=None, nargs=None, *args, **kwargs):
        if nargs is not None:
            raise ValueError("nargs us allowed")

        # Set default
        if default is None:
            raise ValueError("Default must be set!")

        default = _arg_to_bool(default)

        super().__init__(*args, default=default, nargs='?', **kwargs)

    def __call__(self, parser, namespace, argstring, option_string):

        if argstring is not None:
            # If called with an argument, convert to bool
            argval = _arg_to_bool(argstring)
        else:
            # BoolArg will invert default option
            argval = True

        setattr(namespace, self.dest, argval)


# Convert argument to boolean
def _arg_to_bool(arg):

    if type(arg) is bool:
        return arg

    elif type(arg) is str:
        # If string, convert to true/false
        arg = arg.lower()
        if arg in ['true', 't', '1']:
            return True
        elif arg in ['false', 'f', '0']:
            return False
        else:
            return ValueError('Could not parse a True/False boolean')
    else:
        raise ValueError(f'Input must be boolean or string! {type(arg)}')

# From https://stackoverflow.com/questions/12116685/how-can-i-require-my-python-scripts-argument-to-be-a-float-between-0-0-1-0-usin
class Range(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
    def __eq__(self, other):
        return self.start <= other <= self.end
