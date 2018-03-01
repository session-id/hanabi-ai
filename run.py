import numpy as np
import random
import tensorflow as tf

from config import LinearQL_Config
import hanabi_sim as hs
from models.linear_ql import LinearQL_Model

if __name__=='__main__':
    np.random.seed(1339)
    random.seed(1340)
    tf.set_random_seed(1341)
    config = LinearQL_Config()
    train_simulator = hs.RegularHanabiGameEasyFeatures(2)
    test_simulator = hs.RegularHanabiGameEasyFeatures(2)
    model = LinearQL_Model(None, config, train_simulator, test_simulator)
    model.train()
