import argparse

def setup_argparse():
    parser = argparse.ArgumentParser(description='LGN classifier options')

    # loading data
    parser.add_argument('--file-path', type=str, default='./hls4ml/hls4ml.pt', metavar='N',
                        help='The path of the data.')
    parser.add_argument('--num-train', type=int, default=10, metavar='N',
                        help='Number of samples to train on. (default: 528000)')
    parser.add_argument('--num-test', type=int, default=10, metavar='N',
                        help='Number of samples to test on. (default: -1)')
    parser.add_argument('--num-val', type=int, default=10, metavar='N',
                        help='Number of samples to validate on. (default: -1)')

    # neural network constructions
    parser.add_argument('--maxdim', nargs='*', type=int, default=[3], metavar='N',
                        help='Cutoff (maximum weight) in the Clebsch-Gordon operations. Default: [3]')
    parser.add_argument('--max-zf', nargs='*', type=int, default=[1], metavar='N',
                        help='Maximum weight of spherical harmonics to use. Default: [1]')
    parser.add_argument('--num-cg-levels', type=int, default=3, metavar='N',
                        help='Number of CG layers (default: 3)')
    parser.add_argument('--num-channels', nargs='*', type=int, default=[2, 3, 4, 3], metavar='N',
                        help='Number of channels to allow after mixing. Default: [2, 3, 4, 3]')
    parser.add_argument('--weight-init', type=str, default='randn', metavar='str',
                        help='Weight initialization function to use. Default: randn')
    parser.add_argument('--num-basis-fn', type=int, default=10, metavar='N',
                        help='Number of basis functons(default: 10)')
    parser.add_argument('--level-gain', nargs='*', type=float, default=[1.], metavar='N',
                        help='Gain at each level (default: [1.])')
    parser.add_argument('--output-layer', type=str, default='linear', metavar='N',
                        help='The output layer type to use: linear or mlp. Default: linear')
    parser.add_argument('--num-mpnn-levels', type=int, default=1,
                        help='Number levels to use in InputMPNN layer. Default: 1')
    parser.add_argument('--activation', type=str, default='leakyrelu',
                        help='Activation function used in MLP layers. Options: (relu, elu, leakyrelu, sigmoid, logsigmoid). Default: elu.')
    parser.add_argument('--p4_into_CG', type=bool, default=True,
                        help='Feed 4-momenta themselves to the first CG layer, in addition to scalars. Default: True')
    parser.add_argument('--add-beams', type=bool, default=False,
                        help='Whether to two proton beams of the form (m^2,0,0,+-1) to each event. Default: False')
    parser.add_argument('--full-scalars', type=bool, default=True,
                        help='Wehther to feed the norms of ALL irrep tensors (as opposed to just the Loretnz scalar irreps) \
                        at each level into the output layer. Default: True')
    parser.add_argument('--mlp', type=bool, default=True,
                        help='Whether to insert a perceptron acting on invariant scalars inside each CG level. Default: True')
    parser.add_argument('--mlp-depth', type=int, default=3, metavar='N',
                        help='Number of hidden layers in each MLP. Default: 3')
    parser.add_argument('--mlp-width', type=int, default=2, metavar='N',
                        help='Width of hidden layers in each MLP in units of the number of inputs. Default: 2')


    # training parameters
    parser.add_argument('--num-epoch', type=int, default=1, metavar='N',
                        help='Number of epochs to train. Default: 53')
    parser.add_argument('--batch-size', '-bs', type=int, default=32, metavar='N',
                        help='The batch size. Default: 32')
    parser.add_argument('--lr-init', type=float, default=0.001, metavar='N',
                        help='Initial learning rate. Default: 0.005')
    parser.add_argument('--lr-final', type=float, default=1e-5, metavar='N',
                        help='Final (held) learning rate. Default: 1e-5')
    parser.add_argument('--lr-decay', type=int, default=inf, metavar='N',
                        help='Timescale over which to decay the learning rate. Default: inf')
    parser.add_argument('--lr-decay-type', type=str, default='cos', metavar='str',
                        help='Type of learning rate decay. Options: (cos | linear | exponential | pow | restart). Default: cos')
