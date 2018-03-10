import numpy as np
import random
import tensorflow as tf

from config import LinearQL_Config, DeepQL_Config
import hanabi_sim as hs
from models.linear_ql import LinearQL_Model
from models.deep_ql import DQL_Model


if __name__=='__main__':
    np.random.seed(1339)
    random.seed(1340)
    tf.set_random_seed(1341)
    config = LinearQL_Config()
    # config = DeepQL_Config()
    train_simulator = hs.RegularHanabiGameEasyFeatures(config.num_players)
    test_simulator = hs.RegularHanabiGameEasyFeatures(config.num_players)

    def eps_decay(step):
        if step >= config.eps_nsteps:
            return config.eps_end
        else:
            eps_increment = float(config.eps_begin - config.eps_end) / config.eps_nsteps
            return config.eps_begin - step * eps_increment

#    def eps_decay(step):
#        return float(config.eps_begin) * config.eps_delay / (config.eps_delay + step)

    model = LinearQL_Model(config, train_simulator, test_simulator, eps_decay)
    # model = DQL_Model(config, train_simulator, test_simulator, eps_decay)
    model.train()
