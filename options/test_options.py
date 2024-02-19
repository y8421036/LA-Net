from .base_options import BaseOptions


class TestOptions(BaseOptions):

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  
        parser.add_argument('--mode', type=str, default='test')

        return parser
