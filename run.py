from config import LinearQL_Config
import hanabi_sim as hs
from models.linear_ql import LinearQL_Model

if __name__=='__main__':
    config = LinearQL_Config()
    train_simulator = hs.RegularHanabiGameEasyFeatures(2)
    test_simulator = hs.RegularHanabiGameEasyFeatures(2)
    model = LinearQL_Model(None, config, train_simulator, test_simulator)
    model.train()
