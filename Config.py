DATASET_DIR = "./datasets"
COIN_PEAR   = "BTCUSDT"
# COIN_PEAR   = "ADAUSDT"
# COIN_PEAR   = "BNBUSDT"
# COIN_PEAR   = "BCHUSDT"
# COIN_PEAR   = "LTCUSDT"
# COIN_PEAR   = "ETHUSDT"

LOOK_BACK_LEN = 360       # Attach Last 6 Hours of data with sequence
FUTURE_PREDICT = 60 # 60 minuts
DO_ACTION_MIN_CHANCE = 0.01 # 1% change

# Dataframe for testing
START_TS = 1617300690000
END_TS   = 1627668690000

# Date Fraim For Training
TRAINING_START_TS = 1514745000000
TRAINING_END_TS   = 1609439400000
# TRAINING_START_TS = 1590949800000
# TRAINING_END_TS   = 1609439400000