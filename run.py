from config import LinearQL_Config
import hanabi_sim as hs
from models.linear_ql import LinearQL_Model

if __name__=='__main__':
    config = LinearQL_Config()
    simulator = hs.RegularHanabiGameEasyFeatures(2)
    model = LinearQL_Model(None, config, simulator, simulator)
    model.train()
