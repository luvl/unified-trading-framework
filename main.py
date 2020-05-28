from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect, random

def main():
    from processdata import prep_data
    dat_config = prep_data(remake=False)

    from tensorflow.keras.optimizers import Adam
    from models.dqn import deep_q_network, train, train_w_validation, test, describe_stats
    from rl.memory import SequentialMemory
    from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
    from rl.agents.dqn import DQNAgent
    from environment import DirectReinforcement
    from config import EnvConfig, TrainConfig, TestConfig, ValConfig, HyperparamConfig, DefaultMacro

    env_config = EnvConfig()
    tra_config = TrainConfig()
    tes_config = TestConfig()
    val_config = ValConfig()
    par_config = HyperparamConfig()
    default_macro = DefaultMacro()

    env = DirectReinforcement(env_config, dat_config, tra_config, tes_config, val_config, par_config, default_macro)
    model = deep_q_network(env)
    memory = SequentialMemory(limit=env.mem_size, window_length=env.window_length)
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(), 
        attr='eps',
        value_max=1.0, 
        value_min=0.1, 
        value_test=0.05, 
        nb_steps=env.explore_steps
    )
    nb_actions = env.action_space.n  # set up number of actions (outputs)
    dqn = DQNAgent(
        model=model, 
        gamma=env.gamma, 
        nb_actions=nb_actions, 
        memory=memory,
        batch_size=env.batch_size, 
        nb_steps_warmup=1000,
        target_model_update=env.tar_mod_up, 
        policy=policy, 
        delta_clip=env.delta_clip
    )

    dqn.compile(
        Adam(lr=env.learning_rate, decay=env.learning_rate_decay), 
        metrics=['mse']
    )

    if default_macro.TRAIN_W_VALIDATION:
        train_w_validation(env, dqn)
    else:
        train(env, dqn)

    describe_stats(env, env.steps)
    test(env, dqn)

    # for key, data in inspect.getmembers(env):
    #     print('{}: {!r}'.format(key, data))

if __name__ == "__main__":
    main()
        

