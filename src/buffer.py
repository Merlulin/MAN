import math

from dataclasses import dataclass

import numpy as np

from src.config import config


@dataclass
class EpisodeData:
    __slots__ = (
        "actor_id",
        "num_agents",
        "map_len",
        "obs",
        "last_act",
        "actions",
        "rewards",
        "hiddens",
        "relative_pos",
        "comm_mask",
        "gammas",
        "td_errors",
        "sizes",
        "done",
    )
    actor_id: int
    num_agents: int
    map_len: int
    obs: np.ndarray
    last_act: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    hiddens: np.ndarray
    relative_pos: np.ndarray
    comm_mask: np.ndarray
    gammas: np.ndarray
    td_errors: np.ndarray
    sizes: np.ndarray
    done: bool


class SumTree:
    """used for prioritized experience replay"""

    def __init__(self, capacity: int):
        layer = 1
        while 2 ** (layer - 1) < capacity:
            layer += 1
        assert 2 ** (layer - 1) == capacity, "capacity only allow n**2 size"
        self.layer = layer
        self.tree = np.zeros(2**layer - 1, dtype=np.float64)
        self.capacity = capacity
        self.size = 0

    def sum(self):
        assert (
            np.sum(self.tree[-self.capacity :]) - self.tree[0] < 0.1
        ), "sum is {} but root is {}".format(
            np.sum(self.tree[-self.capacity :]), self.tree[0]
        )
        return self.tree[0]

    def __getitem__(self, idx: int):
        assert 0 <= idx < self.capacity

        return self.tree[self.capacity - 1 + idx]

    def batch_sample(self, batch_size: int):
        p_sum = self.tree[0]
        interval = p_sum / batch_size

        prefixsums = np.arange(0, p_sum, interval, dtype=np.float64) + np.random.uniform(
            0, interval, batch_size
        )

        idxes = np.zeros(batch_size, dtype=int)
        for _ in range(self.layer - 1):
            nodes = self.tree[idxes * 2 + 1]
            idxes = np.where(prefixsums < nodes, idxes * 2 + 1, idxes * 2 + 2)
            prefixsums = np.where(
                idxes % 2 == 0, prefixsums - self.tree[idxes - 1], prefixsums
            )

        priorities = self.tree[idxes]
        idxes -= self.capacity - 1

        assert np.all(priorities > 0), "idx: {}, priority: {}".format(idxes, priorities)
        assert np.all(idxes >= 0) and np.all(idxes < self.capacity)

        return idxes, priorities

    def batch_update(self, idxes: np.ndarray, priorities: np.ndarray):
        assert idxes.shape[0] == priorities.shape[0]
        idxes += self.capacity - 1
        self.tree[idxes] = priorities

        for _ in range(self.layer - 1):
            idxes = (idxes - 1) // 2
            idxes = np.unique(idxes)
            self.tree[idxes] = self.tree[2 * idxes + 1] + self.tree[2 * idxes + 2]

        # check
        assert (
            np.sum(self.tree[-self.capacity :]) - self.tree[0] < 0.1
        ), "sum is {} but root is {}".format(
            np.sum(self.tree[-self.capacity :]), self.tree[0]
        )


class LocalBuffer:
    __slots__ = (
        "actor_id",
        "map_len",
        "num_agents",
        "obs_buf",
        "act_buf",
        "rew_buf",
        "hidden_buf",
        "forward_steps",
        "relative_pos_buf",
        "q_buf",
        "capacity",
        "size",
        "done",
        "burn_in_steps",
        "chunk_capacity",
        "last_act_buf",
        "comm_mask_buf",
    )

    def __init__(
        self,
        actor_id: int,
        num_agents: int,
        map_len: int,
        init_obs: np.ndarray,
        forward_steps=config.forward_steps,
        capacity: int = config.max_episode_length,
        burn_in_steps=config.burn_in_steps,
        obs_shape=config.obs_shape,
        hidden_dim=config.hidden_dim,
        action_dim=config.action_dim,
    ):
        """
        buffer for each episode
        args:
            actor_id: id of actor
            num_agents: number of agents
            map_len: length of map
            init_obs: initial observation (num_agents, 6, 2 * self.obs_radius + 1, 2 * self.obs_radius + 1)
            forward_steps: n-step forward
            capacity: max length of episode
            burn_in_steps: 预热步长
            obs_shape: shape of observation
            hidden_dim: dimension of hidden state
            action_dim: dimension of action
        """
        self.actor_id = actor_id
        self.num_agents = num_agents
        self.map_len = map_len

        self.burn_in_steps = burn_in_steps
        self.forward_steps = forward_steps

        # chunk_capacity 表示一个数据块的容量
        self.chunk_capacity = config.chunk_capacity

        # obs_buf 用于存放观测数据：初始化为 (预热步长 + 经验最大长度 + 1, 智能体数量, (6, 2 * obs_radius + 1, 2 * obs_radius + 1))
        self.obs_buf = np.zeros(
            (burn_in_steps + capacity + 1, num_agents, *obs_shape), dtype=bool
        )
        # last_act_buf 用于存放上一步的动作数据：初始化为 (预热步长 + 经验最大长度 + 1, 智能体数量, 5)
        self.last_act_buf = np.zeros(
            (burn_in_steps + capacity + 1, num_agents, 5), dtype=bool
        )
        # act_buf 用于存放动作数据：初始化为 (经验最大长度)
        self.act_buf = np.zeros((capacity), dtype=np.uint8)
        # rew_buf 用于存放奖励数据：初始化为 (经验最大长度 + n-step forward - 1)
        self.rew_buf = np.zeros((capacity + forward_steps - 1), dtype=np.float16)
        # hidden_buf 用于存放隐藏状态数据：初始化为 (预热步长 + 经验最大长度 + 1, 智能体数量, 隐藏状态维度)
        self.hidden_buf = np.zeros(
            (burn_in_steps + capacity + 1, num_agents, hidden_dim), dtype=np.float16
        )
        # relative_pos_buf 用于存放相对位置数据：初始化为 (预热步长 + 经验最大长度 + 1, 智能体数量, 智能体数量, 2)
        self.relative_pos_buf = np.zeros(
            (burn_in_steps + capacity + 1, num_agents, num_agents, 2), dtype=np.int8
        )
        # comm_mask_buf 用于存放通信掩码数据：初始化为 (预热步长 + 经验最大长度 + 1, 智能体数量, 智能体数量)
        self.comm_mask_buf = np.zeros(
            (burn_in_steps + capacity + 1, num_agents, num_agents), dtype=bool
        )
        # q_buf 用于存放Q值数据：初始化为 (经验最大长度, 动作维度)
        self.q_buf = np.zeros((capacity + 1, action_dim), dtype=np.float32)

        self.capacity = capacity
        self.size = 0   # 表示真实的步长

        # 将初始观测数据作为预热数据    
        self.obs_buf[: burn_in_steps + 1] = init_obs

    
    def add(
        self,
        q_val,
        action: int,
        last_act,
        reward: float,
        next_obs,
        hidden,
        relative_pos,
        comm_mask,
    ):
        '''
        将经验中的某一步加入到buffer中
        '''
        assert self.size < self.capacity

        self.act_buf[self.size] = action
        self.rew_buf[self.size] = reward
        self.obs_buf[self.burn_in_steps + self.size + 1] = next_obs
        self.last_act_buf[self.burn_in_steps + self.size + 1] = last_act
        self.q_buf[self.size] = q_val
        self.hidden_buf[self.burn_in_steps + self.size + 1] = hidden
        self.relative_pos_buf[self.burn_in_steps + self.size] = relative_pos
        self.comm_mask_buf[self.burn_in_steps + self.size] = comm_mask

        self.size += 1

    def finish(self, last_q_val=None, last_relative_pos=None, last_comm_mask=None):
        '''
        表示完成一个episode的采样, 返回一个EpisodeData对象
        '''
        # forward_steps 取forward_steps和size的最小值，表示获取此前的多少步长，需要对size求min是因为确保数据存在
        forward_steps = min(self.size, self.forward_steps)
        # cumulated_gamma 表示累积的gamma值（累计折扣因子），所以可以推断出这列的forward_step表示在计算Q值时只考虑前向forward_step步长的回报
        cumulated_gamma = [
            config.gamma**forward_steps for _ in range(self.size - forward_steps)
        ]

        # last q value is None if done
        if last_q_val is None:
            done = True
            # 如果last_q_val为None，表示已经结束，则将最后的gamma值赋值为0加入到累加gamma中
            cumulated_gamma.extend([0 for _ in range(forward_steps)])

        else:
            done = False
            # 如果last_q_val不为None，表示还没有结束，需要将最后的q值加入到q_buf中
            self.q_buf[self.size] = last_q_val
            self.relative_pos_buf[self.burn_in_steps + self.size] = last_relative_pos
            self.comm_mask_buf[self.burn_in_steps + self.size] = last_comm_mask
            #  将最后的gamma值加入到累加gamma中
            cumulated_gamma.extend(
                [config.gamma**i for i in reversed(range(1, forward_steps + 1))]
            )

        # 计算chunk（数据库）的数量， ceil表示向上取整
        num_chunks = math.ceil(self.size / config.chunk_capacity)
        cumulated_gamma = np.array(cumulated_gamma, dtype=np.float16)
        self.obs_buf = self.obs_buf[: self.burn_in_steps + self.size + 1]
        self.last_act_buf = self.last_act_buf[: self.burn_in_steps + self.size + 1]
        self.act_buf = self.act_buf[: self.size]
        self.rew_buf = self.rew_buf[: self.size + self.forward_steps - 1]
        self.hidden_buf = self.hidden_buf[: self.size]
        self.relative_pos_buf = self.relative_pos_buf[
            : self.burn_in_steps + self.size + 1
        ]
        self.comm_mask_buf = self.comm_mask_buf[: self.burn_in_steps + self.size + 1]

        # 使用卷积核对奖励进行卷积操作
        self.rew_buf = np.convolve(
            self.rew_buf,
            [
                config.gamma ** (self.forward_steps - 1 - i)
                for i in range(self.forward_steps)
            ],
            "valid",
        ).astype(np.float16)

        # caculate td errors for prioritized experience replay

        max_q = np.max(self.q_buf[forward_steps : self.size + 1], axis=1)
        # 这一行代码通过复制最后一个最大Q值forward_steps-1次并追加到max_q数组的末尾，来扩展max_q数组，
        # 确保它与奖励和行动数组具有相同的长度。这样做是为了在计算TD误差时可以有一个对应的未来最大Q值。
        max_q = np.concatenate(
            (max_q, np.array([max_q[-1] for _ in range(forward_steps - 1)]))
        )
        # 提取了对应于已采取的行动的Q值。
        target_q = self.q_buf[np.arange(self.size), self.act_buf]
        # 计算时间差分误差，初始化TD误差数组（num_chunks * self.chunk_capacity）> self.size
        td_errors = np.zeros(num_chunks * self.chunk_capacity, dtype=np.float32)
        # 计算实际TD误差为奖励加上折扣后的未来最大Q值减去当前状态的Q值，使用.clip(1e-6)确保误差不会因数值下溢而变成0
        td_errors[: self.size] = np.abs(
            self.rew_buf + max_q * cumulated_gamma - target_q
        ).clip(1e-6)
        # 计算每个数据块的大小，并且确保最后一个数据块不会超过实际的size，所以这里可以推断出sizes的长度就是数据块的数量，数据块就是chunk_capacity大小的一段经验行为。
        sizes = np.array(
            [
                min(self.chunk_capacity, self.size - i * self.chunk_capacity)
                for i in range(num_chunks)
            ],
            dtype=np.uint8,
        )

        data = EpisodeData(
            self.actor_id,
            self.num_agents,
            self.map_len,
            self.obs_buf,
            self.last_act_buf,
            self.act_buf,
            self.rew_buf,
            self.hidden_buf,
            self.relative_pos_buf,
            self.comm_mask_buf,
            cumulated_gamma,
            td_errors,
            sizes,
            done,
        )

        return data
