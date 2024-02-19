from .base_options import BaseOptions
import os

class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--load', type=str, default=False, help='whether restore or not')
        parser.add_argument('--batch_size', type=int, default=3, help='input batch size')
        parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
        parser.add_argument('--max_iteration', type=int, default=20001, help='iterations for batch_size samples')
        parser.add_argument('--max_iteration_post', type=int, default=3001, help='iterations for batch_size samples')
        parser.add_argument('--save_interval', type=int, default=300, help='frequency of printing train result') 
        parser.add_argument('--save_interval_post', type=int, default=50, help='frequency of printing train result')
        parser.add_argument('--mode', type=str, default='train')
        parser.add_argument('--optimizer', type=str, default='Adam')

        return parser
