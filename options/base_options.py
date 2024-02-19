import argparse

class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--dataroot', default='/data2/datasets/OCTA-500/3M/', help='path to data')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids')
        parser.add_argument('--train_ids',type=list,default=[0,140],help='train id number')
        parser.add_argument('--val_ids',type=list,default=[140,150],help='val id number') 
        parser.add_argument('--test_ids',type=list,default=[150,200],help='test id number')
        parser.add_argument('--modality_filename', type=list, default=['OCT','OCTA','Label_RV'], help='dataset filename, last name is label filename')
        parser.add_argument('--data_size', type=list, default=[640,304,304], help='input data size separated with comma') 
        parser.add_argument('--block_size', type=list, default=[160,76,76], help='crop size separated with comma') 
        parser.add_argument('--in_channels', type=int, default=2, help='input channels')
        parser.add_argument('--channels', type=int, default=64, help='channels') 
        parser.add_argument('--plane_perceptron_channels', type=int, default=64, help='post_channels')
        parser.add_argument('--saveroot', default='logs', help='path to save results')
        parser.add_argument('--n_classes', type=int, default=2, help='fianl class number for classification')
        parser.add_argument('--feature_dir', default='./logs/Features_V2',help='feature_dir')

        self.initialized = True
        return parser

    def gather_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        opt, _ = parser.parse_known_args()
        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)
        print('')

    def parse(self):
        opt = self.gather_options()
        self.print_options(opt)
        self.opt = opt
        return self.opt



