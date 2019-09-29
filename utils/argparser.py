import argparse
import os
import yaml


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Parser(object):
    def __init__(self):
        self.desc = "Simple GCN Experiment Pipelines"
        self.parser = argparse.ArgumentParser(
                description=self.desc)

    def create_parser(self):
        self.parser.add_argument(
            '--config',
            default='/public/lhy/wms/TMD_Project_2019/Code/SimpleGCN/utils/config.yaml',
            help='absolute path of config file')
        self.parser.add_argument(
            '--mode',
            default='data_preprocess',
            help='script running mode'
        )
        self.parser.add_argument(
            '--data_preprocess_args',
            default=dict(),
            type=dict,
            help='arguments for data preprocess'
        )
        self.parser.add_argument(
            '--train_args',
            default=dict(),
            type=dict,
            help='arguments for train'
        )
        self.parser.add_argument(
            '--save-score',
            type=str2bool,
            default=False,
            help='if True, the classification score will be stored')

    def dump_args(self, args, work_dir):
        arg_dict = vars(args)
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
        with open('{}/config.yaml'.format(work_dir), 'w') as f:
            yaml.dump(arg_dict, f)

