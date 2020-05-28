from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect

from environment import DirectReinforcement
from processdata import prep_data
from config import init_config

def main():    
    dat_config = prep_data(remake=False)
    env = DirectReinforcement(dat_config, *init_config())

    from models.dqn import init_dqn
    init_dqn(env)

    # for key, data in inspect.getmembers(env):
    #     print('{}: {!r}'.format(key, data))

if __name__ == "__main__":
    main()
        

