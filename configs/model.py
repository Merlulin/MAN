import datetime

from common import *

# Wandb setting
use_wandb = True
project = "FinalTest"
name = "MyModel_V0"
run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Network setting
model_target = "src.models.model_V0.Network"  # to use original DCC model


# For saving model
save_path = "./saved_models/model_V0"

##### Note #####
# If you want to override some parameters from the common/ folder, just
# override down here. For examples:
# active_agent_radius = 1   


use_wandb_test = False
test_folder = "./test_set"

density = 0.3
test_env_settings = (
    (40, 4, density),
    (40, 8, density),
    (40, 16, density),
    (40, 32, density),
    (40, 64, density),
    (80, 4, density),
    (80, 8, density),
    (80, 16, density),
    (80, 32, density),
    (80, 64, density),
    (80, 128, density),
)  # map length, number of agents, density


# 我的创新点开关
open_closed_goal = False
congestion_radius = 2
congestion_expend = 1
block_rate_threshold = 0.5
congestion_dispair = 3