import yaml

from utils import Parser, Trainer
from utils import save_data


if __name__ == '__main__':
    parser = Parser()
    parser.create_parser()
    pargs = parser.parser.parse_args()
    if pargs.config is not None:
        with open(pargs.config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        key = vars(pargs).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.parser.set_defaults(**default_arg)
    args = parser.parser.parse_args()
    mode = args.mode
    if 'data_preprocess' == mode:
        save_data(args.data_preprocess_args['window_size'], args.data_preprocess_args['overlap'], args.data_preprocess_args['label_path'])
    elif 'train' == mode:
        trainer = Trainer(args.train_args)
        trainer.train()
