'''Commandline Argument Parser'''
import argparse
import enum
import copy


class ArgsEnum(enum.Enum):
    REQUIRED = 1


class Args(object):
    def __init__(self, description='', **arguments):
        self.args = dict()

        for key, default in arguments.items():
            details = dict()
            if key in _DEFAULT_ARGS:
                details = copy.deepcopy(_DEFAULT_ARGS[key])

            details['default'] = default

            if default is ArgsEnum.REQUIRED:
                del details['default']
                details['required'] = True
    
            self.args[key] = details

    def parse(self, description=''):
        parser = argparse.ArgumentParser(
            description=description
        )
        for var, details in self.args.items():
            # set option_string to --<arg_name>
            parser.add_argument('--' + var, **details)
        return parser.parse_args()


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# The option_string is always gonna be --<arg_name>
# And everything is always assumed to be required if the default is not present
_DEFAULT_ARGS = {
    'desc': {
        'help': 'Description of what the experiment is trying to accomplish',
    },
    'nb_epochs': {
        'type': int,
        'help': 'Number of epochs to run the algorithm',
    },
    'net_layer_size': {
        'type': int,
        'help': 'The size of each hidden layer',
    },
    'net_nb_layers': {
        'type': int,
        'help': 'The number of layers for the network',
    },
    'net_forecast_steps': {
        'type': int,
        'help': 'Forecast steps',
    },
    'window_size': {
        'type': int,
        'help': 'Window size',
    },
    'batch_size': {
        'type': int,
        'help': 'The batch size',
    },
    'load_path': {
        'help': 'The path for loading a previously saved network and continuing the training',
    },
    'store_path': {
        'help': 'The path for storing the model',
    },
    'data_path': {
        'help': 'The path for loading the input data',
    },
    'store': {
        'type': str2bool,
        'help': 'Whether or not to store the result and logs (tensorboard, variants, etc)',
    },
    'shuffle': {
        'type': str2bool,
        'help': 'Whether or not to shuffle the data',
    },
    'use_dropout': {
        'type': str2bool,
        'help': 'Whether or not to use dropout',
    },
    'lr': {
        'type': float,
        'help': 'The learning rate',
    },
}