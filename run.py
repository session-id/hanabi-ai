import numpy as np
import random
import tensorflow as tf

from config import Config
import hanabi_sim as hs
from models.ql import MLP_QL_Model
# from models.recurrent_ql import RecurrentQL_Model
from hanabi_expert import HanabiExpert

if __name__=='__main__':
    np.random.seed(1339)
    random.seed(1340)
    tf.set_random_seed(1341)
    config = Config()


    
    train_simulator = hs.RegularHanabiGameEasyFeatures(config.num_players,
            config.colors, config.cards_per_player, config.max_number, config.number_counts)
    expert = HanabiExpert(train_simulator)
    test_simulator = train_simulator
    print("State size:", train_simulator.get_state_vector_size())

    def eps_decay(step):
        if step >= config.eps_nsteps:
            return config.eps_end
        else:
            eps_increment = float(config.eps_begin - config.eps_end) / config.eps_nsteps
            return config.eps_begin - step * eps_increment

#    def eps_decay(step):
#        return float(config.eps_begin) * config.eps_delay / (config.eps_delay + step)

    model = MLP_QL_Model(config, train_simulator, test_simulator, eps_decay, expert)
    model.train()
