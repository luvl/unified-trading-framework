from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect

from environment import DirectReinforcement
from processdata import prep_data
from lib.config import init_config

from lib.visualization import BacktestingVisualization
from lib.parser import parse_args

def main():
    args = parse_args()
    dat_config = prep_data(remake=False)
    env = DirectReinforcement(dat_config, *init_config(args))

    from models.dqn import init_dqn
    render_file_location = init_dqn(env)

    # render_file_location = "/home/linhdn/Developer/unified-framework-for-trading/log/3_s:500_w:100_3.6_14:39/Test_1/position_[243]_render.pkl"
    backtest_viz = BacktestingVisualization(render_file_location).run()

    # for key, data in inspect.getmembers(env):
    #     print('{}: {!r}'.format(key, data))

if __name__ == "__main__":
    main()
        

