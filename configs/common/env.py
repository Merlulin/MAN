obs_radius = 4
active_agent_radius = 3
reward_fn = dict(
    move=-0.055, stay_on_goal=0, stay_off_goal=-0.055, collision=-0.5, finish=3, goin_congestion=-0.1, stay_congestion=-0.005, has_go=-0.003
)


# obs_shape = (6, 2 * obs_radius + 1, 2 * obs_radius + 1) # (6, 9, 9), 分别表示6个不同类型的输入观测视野 : (障碍物, 其他智能体, 四个启发式通道)
obs_shape = (8, 2 * obs_radius + 1, 2 * obs_radius + 1) # (7, 9, 9), 分别表示7个不同类型的输入观测视野 : (障碍物, 其他智能体, 四个启发式通道，拥塞信息启发通道, 已访问通道)
action_dim = 5
