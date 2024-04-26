import random

from typing import List
from collections import deque

import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform

from src.config import config
from src.od_mstar3 import od_mstar
from src.od_mstar3.col_set_addition import NoSolutionError, OutOfTimeError

ACTION_LIST = np.array([[-1, 0], [1, 0], [0, -1], [0, 1], [0, 0]], dtype=int)
DIRECTION_TO_ACTION = {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3, (0, 0): 4}


class Environment:
    def __init__(
        self,
        num_agents: int = config.init_env_settings[0],
        map_length: int = config.init_env_settings[1],
        obs_radius: int = config.obs_radius,
        reward_fn: dict = config.reward_fn,
        fix_density=None,
        curriculum=False,
        init_env_settings_set=config.init_env_settings,
    ):

        self.curriculum = curriculum
        if curriculum:
            self.env_set = [init_env_settings_set]
            self.num_agents = init_env_settings_set[0]
            self.map_size = (init_env_settings_set[1], init_env_settings_set[1])
        else:
            self.num_agents = num_agents
            self.map_size = (map_length, map_length)

        # set as same as in PRIMAL
        if fix_density is None:
            self.fix_density = False
            self.obstacle_density = np.random.triangular(0, 0.33, 0.5)
        else:
            self.fix_density = True
            self.obstacle_density = fix_density

        self.map = np.random.choice(
            2, self.map_size, p=[1 - self.obstacle_density, self.obstacle_density]
        ).astype(int)

        partition_list = self._map_partition()

        while len(partition_list) == 0:
            self.map = np.random.choice(
                2, self.map_size, p=[1 - self.obstacle_density, self.obstacle_density]
            ).astype(int)
            partition_list = self._map_partition()

        self.agents_pos = np.empty((self.num_agents, 2), dtype=int)
        self.goals_pos = np.empty((self.num_agents, 2), dtype=int)

        pos_num = sum([len(partition) for partition in partition_list])

        # loop to assign agent original position and goal position for each agent
        for i in range(self.num_agents):

            pos_idx = random.randint(0, pos_num - 1)
            partition_idx = 0
            for partition in partition_list:
                if pos_idx >= len(partition):
                    pos_idx -= len(partition)
                    partition_idx += 1
                else:
                    break

            pos = random.choice(partition_list[partition_idx])
            partition_list[partition_idx].remove(pos)
            self.agents_pos[i] = np.asarray(pos, dtype=int)

            pos = random.choice(partition_list[partition_idx])
            partition_list[partition_idx].remove(pos)
            self.goals_pos[i] = np.asarray(pos, dtype=int)

            partition_list = [
                partition for partition in partition_list if len(partition) >= 2
            ]
            pos_num = sum([len(partition) for partition in partition_list])

        self.obs_radius = obs_radius

        self.reward_fn = reward_fn
        self._get_heuri_map()
        self.steps = 0

        self.last_actions = np.zeros((self.num_agents, 5), dtype=bool)

        # Inference Strategy
        self.astar_type = config.astar_type
        self.active_agent_radius = config.active_agent_radius
        self.use_aep = config.use_aep
        self.aep_astar_type = config.aep_astar_type
        
        #####################################################################################
        # 创新点1：拥塞区域检测，如果发生碰撞则说明这个区域存在拥塞可能
        #####################################################################################
        # 拥塞区域判断图
        self.conflict_group = []
        self.block_area = np.zeros_like(self.map, dtype=bool)
        # 位于拥塞区域的时间步数
        # self.stay_congestion_time = [0] * self.num_agents

        # 每个智能体已走过的路线路
        self.has_go_map = np.zeros((self.num_agents, self.map_size[0], self.map_size[1]), dtype=bool)
        for i in range(self.num_agents):
            x, y = self.agents_pos[i][0], self.agents_pos[i][1]
            self.has_go_map[i][x][y] = True

    def update_env_settings_set(self, new_env_settings_set):
        self.env_set = new_env_settings_set

    def reset(self, num_agents=None, map_length=None):

        if self.curriculum:
            rand = random.choice(self.env_set)
            self.num_agents = rand[0]
            self.map_size = (rand[1], rand[1])

        elif num_agents is not None and map_length is not None:
            self.num_agents = num_agents
            self.map_size = (map_length, map_length)

        if not self.fix_density:
            self.obstacle_density = np.random.triangular(0, 0.33, 0.5)

        self.map = np.random.choice(
            2, self.map_size, p=[1 - self.obstacle_density, self.obstacle_density]
        ).astype(np.float32)

        partition_list = self._map_partition()

        while len(partition_list) == 0:
            self.map = np.random.choice(
                2, self.map_size, p=[1 - self.obstacle_density, self.obstacle_density]
            ).astype(np.float32)
            partition_list = self._map_partition()

        self.agents_pos = np.empty((self.num_agents, 2), dtype=int)
        self.goals_pos = np.empty((self.num_agents, 2), dtype=int)

        pos_num = sum([len(partition) for partition in partition_list])

        for i in range(self.num_agents):

            pos_idx = random.randint(0, pos_num - 1)
            partition_idx = 0
            for partition in partition_list:
                if pos_idx >= len(partition):
                    pos_idx -= len(partition)
                    partition_idx += 1
                else:
                    break

            pos = random.choice(partition_list[partition_idx])
            partition_list[partition_idx].remove(pos)
            self.agents_pos[i] = np.asarray(pos, dtype=int)

            pos = random.choice(partition_list[partition_idx])
            partition_list[partition_idx].remove(pos)
            self.goals_pos[i] = np.asarray(pos, dtype=int)

            partition_list = [
                partition for partition in partition_list if len(partition) >= 2
            ]
            pos_num = sum([len(partition) for partition in partition_list])

        self.steps = 0
        self._get_heuri_map()

        self.last_actions = np.zeros((self.num_agents, 5), dtype=bool)


        self.block_area = np.zeros_like(self.map, dtype=bool)
        self.conflict_group = []
        self.has_go_map = np.zeros((self.num_agents, self.map_size[0], self.map_size[1]), dtype=bool)
        for i in range(self.num_agents):
            x, y = self.agents_pos[i][0], self.agents_pos[i][1]
            self.has_go_map[i][x][y] = True
        
        return self.observe()

    def load(self, map: np.ndarray, agents_pos: np.ndarray, goals_pos: np.ndarray):

        self.map = np.copy(map)
        self.agents_pos = np.copy(agents_pos)
        self.goals_pos = np.copy(goals_pos)

        self.num_agents = agents_pos.shape[0]
        self.map_size = (self.map.shape[0], self.map.shape[1])

        self.steps = 0

        self._get_heuri_map()

        self.last_actions = np.zeros((self.num_agents, 5), dtype=bool)
        
        self.block_area = np.zeros_like(self.map, dtype=bool)
        self.conflict_group = []
        self.has_go_map = np.zeros((self.num_agents, self.map_size[0], self.map_size[1]), dtype=bool)
        for i in range(self.num_agents):
            x, y = self.agents_pos[i][0], self.agents_pos[i][1]
            self.has_go_map[i][x][y] = True
        

    def _get_heuri_map(self):
        dist_map = (
            np.ones((self.num_agents, *self.map_size), dtype=np.int32)
            * np.iinfo(np.int32).max
        )

        empty_pos = np.argwhere(self.map == 0).tolist()
        empty_pos = set([tuple(pos) for pos in empty_pos])

        for i in range(self.num_agents):
            open_list = set()
            x, y = tuple(self.goals_pos[i])
            open_list.add((x, y))
            dist_map[i, x, y] = 0

            while open_list:
                x, y = open_list.pop()
                dist = dist_map[i, x, y]

                up = x - 1, y
                if up in empty_pos and dist_map[i, x - 1, y] > dist + 1:
                    dist_map[i, x - 1, y] = dist + 1
                    open_list.add(up)

                down = x + 1, y
                if down in empty_pos and dist_map[i, x + 1, y] > dist + 1:
                    dist_map[i, x + 1, y] = dist + 1
                    open_list.add(down)

                left = x, y - 1
                if left in empty_pos and dist_map[i, x, y - 1] > dist + 1:
                    dist_map[i, x, y - 1] = dist + 1
                    open_list.add(left)

                right = x, y + 1
                if right in empty_pos and dist_map[i, x, y + 1] > dist + 1:
                    dist_map[i, x, y + 1] = dist + 1
                    open_list.add(right)

        self.heuri_map = np.zeros((self.num_agents, 4, *self.map_size), dtype=bool)
        self.heuri_dist_map = dist_map
        
        for x, y in empty_pos:
            for i in range(self.num_agents):

                if x > 0 and dist_map[i, x - 1, y] < dist_map[i, x, y]:
                    self.heuri_map[i, 0, x, y] = 1

                if x < self.map_size[0] - 1 and dist_map[i, x + 1, y] < dist_map[i, x, y]:
                    self.heuri_map[i, 1, x, y] = 1

                if y > 0 and dist_map[i, x, y - 1] < dist_map[i, x, y]:
                    self.heuri_map[i, 2, x, y] = 1

                if y < self.map_size[1] - 1 and dist_map[i, x, y + 1] < dist_map[i, x, y]:
                    self.heuri_map[i, 3, x, y] = 1

        self.heuri_map = np.pad(
            self.heuri_map,
            (
                (0, 0),
                (0, 0),
                (self.obs_radius, self.obs_radius),
                (self.obs_radius, self.obs_radius),
            ),
        )

    def _map_partition(self):
        """
        partitioning map into independent partitions
        """
        empty_list = np.argwhere(self.map == 0).tolist()

        empty_pos = set([tuple(pos) for pos in empty_list])

        if not empty_pos:
            raise RuntimeError("no empty position")

        partition_list = list()
        while empty_pos:

            start_pos = empty_pos.pop()

            open_list = list()
            open_list.append(start_pos)
            close_list = list()

            while open_list:
                x, y = open_list.pop(0)

                up = x - 1, y
                if up in empty_pos:
                    empty_pos.remove(up)
                    open_list.append(up)

                down = x + 1, y
                if down in empty_pos:
                    empty_pos.remove(down)
                    open_list.append(down)

                left = x, y - 1
                if left in empty_pos:
                    empty_pos.remove(left)
                    open_list.append(left)

                right = x, y + 1
                if right in empty_pos:
                    empty_pos.remove(right)
                    open_list.append(right)

                close_list.append((x, y))

            if len(close_list) >= 2:
                partition_list.append(close_list)

        return partition_list

    def step(self, actions: List[int]):
        """
        actions:
            list of indices
                0 up
                1 down
                2 left
                3 right
                4 stay
        """

        assert (
            len(actions) == self.num_agents
        ), "only {} actions as input while {} agents in environment".format(
            len(actions), self.num_agents
        )
        assert all(
            [action_idx < 5 and action_idx >= 0 for action_idx in actions]
        ), "action index out of range"

        checking_list = [i for i in range(self.num_agents)]

        rewards = []
        next_pos = np.copy(self.agents_pos)

        # remove unmoving agent id
        for agent_id in checking_list.copy():
            if actions[agent_id] == 4:
                # unmoving
                if np.array_equal(self.agents_pos[agent_id], self.goals_pos[agent_id]):
                    rewards.append(self.reward_fn["stay_on_goal"])
                else:
                    rewards.append(self.reward_fn["stay_off_goal"])
                checking_list.remove(agent_id)
            else:
                # move
                next_pos[agent_id] += ACTION_LIST[actions[agent_id]]
                rewards.append(self.reward_fn["move"])

        # first round check, these two conflicts have the highest priority
        for agent_id in checking_list.copy():

            # if np.any(next_pos[agent_id]<0) or np.any(next_pos[agent_id]>=self.map_size[0]):
            if (
                np.any(next_pos[agent_id] < 0)
                or next_pos[agent_id][0] >= self.map_size[0]
                or next_pos[agent_id][1] >= self.map_size[1]
            ):
                # agent out of map range
                rewards[agent_id] = self.reward_fn["collision"]
                next_pos[agent_id] = self.agents_pos[agent_id]
                checking_list.remove(agent_id)

            elif self.map[tuple(next_pos[agent_id])] == 1:
                # collide obstacle
                rewards[agent_id] = self.reward_fn["collision"]
                next_pos[agent_id] = self.agents_pos[agent_id]
                checking_list.remove(agent_id)

        # second round check, agent swapping conflict
        no_conflict = False
        while not no_conflict:

            no_conflict = True
            for agent_id in checking_list:

                target_agent_id = np.where(
                    np.all(next_pos[agent_id] == self.agents_pos, axis=1)
                )[0]

                if target_agent_id:

                    target_agent_id = target_agent_id.item()

                    if np.array_equal(
                        next_pos[target_agent_id], self.agents_pos[agent_id]
                    ):
                        assert (
                            target_agent_id in checking_list
                        ), "target_agent_id should be in checking list"

                        next_pos[agent_id] = self.agents_pos[agent_id]
                        rewards[agent_id] = self.reward_fn["collision"]

                        next_pos[target_agent_id] = self.agents_pos[target_agent_id]
                        rewards[target_agent_id] = self.reward_fn["collision"]

                        checking_list.remove(agent_id)
                        checking_list.remove(target_agent_id)
                        
                        ####################################
                        # 创新点1：碰撞对象加入到冲突组
                        ####################################
                        self.conflict_group.append([agent_id, target_agent_id])
                        
                        no_conflict = False
                        break

        # third round check, agent collision conflict
        no_conflict = False
        while not no_conflict:
            no_conflict = True
            for agent_id in checking_list:
                
                temp_conflict_group = []
                
                collide_agent_id = np.where(
                    np.all(next_pos == next_pos[agent_id], axis=1)
                )[0].tolist()
                if len(collide_agent_id) > 1:
                    # collide agent
                    for id in collide_agent_id:
                        temp_conflict_group.append(id)

                    # if all agents in collide agent are in checking list
                    all_in_checking = True
                    for id in collide_agent_id.copy():
                        if id not in checking_list:
                            all_in_checking = False
                            collide_agent_id.remove(id)

                    if all_in_checking:

                        collide_agent_pos = next_pos[collide_agent_id].tolist()
                        for pos, id in zip(collide_agent_pos, collide_agent_id):
                            pos.append(id)
                        collide_agent_pos.sort(
                            key=lambda x: x[0] * self.map_size[0] + x[1]
                        )  # TODO (or actually fine)
                        collide_agent_id.remove(collide_agent_pos[0][2])

                    next_pos[collide_agent_id] = self.agents_pos[collide_agent_id]
                    for id in collide_agent_id:
                        rewards[id] = self.reward_fn["collision"]


                    for id in collide_agent_id:
                        checking_list.remove(id)

                    self.conflict_group.append(temp_conflict_group)
                    no_conflict = False
                    break

        ####################################################################
        # 创新点 1的补救：检查移动的倾向的下一步是否发生碰撞
        ####################################################################

        next_next_pos = np.copy(next_pos)
        for i in range(self.num_agents):
            next_next_pos[i] += ACTION_LIST[actions[i]]
        next_checking_list = [i for i in range(self.num_agents)]

        # remove unmoving agent id
        for agent_id in next_checking_list.copy():
            if actions[agent_id] == 4:
                next_checking_list.remove(agent_id)


        # second round check, agent swapping conflict
        next_no_conflict = False
        while not next_no_conflict:

            next_no_conflict = True
            for agent_id in next_checking_list:

                target_agent_id = np.where(
                    np.all(next_next_pos[agent_id] == next_pos, axis=1)
                )[0]

                if target_agent_id:

                    target_agent_id = target_agent_id.item()

                    if np.array_equal(
                            next_next_pos[target_agent_id], next_pos[agent_id]
                    ):
                        assert (
                                target_agent_id in next_checking_list
                        ), "target_agent_id should be in checking list"

                        next_checking_list.remove(agent_id)
                        next_checking_list.remove(target_agent_id)

                        ####################################
                        # 创新点1：将预计碰撞对象加入到冲突组
                        ####################################
                        self.conflict_group.append([agent_id, target_agent_id])

                        next_no_conflict = False
                        break

        # third round check, agent collision conflict
        next_no_conflict = False
        while not next_no_conflict:
            next_no_conflict = True
            for agent_id in next_checking_list:

                temp_conflict_group = []

                collide_agent_id = np.where(
                    np.all(next_next_pos == next_next_pos[agent_id], axis=1)
                )[0].tolist()
                if len(collide_agent_id) > 1:
                    # collide agent
                    for id in collide_agent_id:
                        temp_conflict_group.append(id)

                    # if all agents in collide agent are in checking list
                    all_in_checking = True
                    for id in collide_agent_id.copy():
                        if id not in next_checking_list:
                            all_in_checking = False
                            collide_agent_id.remove(id)

                    if all_in_checking:

                        collide_agent_pos = next_next_pos[collide_agent_id].tolist()
                        for pos, id in zip(collide_agent_pos, collide_agent_id):
                            pos.append(id)
                        collide_agent_pos.sort(
                            key=lambda x: x[0] * self.map_size[0] + x[1]
                        )  # TODO (or actually fine)
                        collide_agent_id.remove(collide_agent_pos[0][2])

                    next_next_pos[collide_agent_id] = next_pos[collide_agent_id]
                    
                    for id in collide_agent_id:
                        next_checking_list.remove(id)
                    
                    self.conflict_group.append(temp_conflict_group)
                    next_no_conflict = False
                    break


        #####################################################################
        # 创新点 1： 每一次行动后，根据冲突组的信息更新拥塞区域，并给予行为的变化进行惩罚
        #####################################################################
        self.block_area = self.check_congestion_area(k=config.congestion_expend)
        agent_map = np.zeros((self.map_size), dtype=bool)
        agent_map[self.agents_pos[:, 0], self.agents_pos[:, 1]] = 1
        agent_map = np.pad(agent_map, self.obs_radius, "constant", constant_values=0)
        for id in range(self.num_agents):
            # 如果新的坐标是已经走过的，则给予极少量的惩罚，少于呆在拥塞区域内的惩罚，迫使智能体走向一个未探索过的区域
            n_x, n_y = next_pos[id]
            if self.has_go_map[id][n_x][n_y]:
                rewards[id] += self.reward_fn["has_go"]
            if self.block_area[tuple(self.agents_pos[id])] == False and self.block_area[tuple(next_pos[id])] == True:
                rewards[id] += self.reward_fn["goin_congestion"]
            elif self.block_area[tuple(self.agents_pos[id])] == True and self.block_area[tuple(next_pos[id])] == True:
                x, y = self.agents_pos[id]
                # alpha表示了当前智能体FOV范围内的智能体数量
                alpha = agent_map[x : x + 2 * self.obs_radius + 1, y : y + 2 * self.obs_radius + 1].sum()
                rewards[id] += self.reward_fn["stay_congestion"]

        self.agents_pos = np.copy(next_pos)

        self.steps += 1

        #####################################################
        # 根据行为将智能体的走过的位置进行标记
        #####################################################
        for i in range(self.num_agents):
            x, y = self.agents_pos[i][0], self.agents_pos[i][1]
            self.has_go_map[i][x][y] = True

        # check done
        if np.array_equal(self.agents_pos, self.goals_pos):
            done = True
            rewards = [self.reward_fn["finish"] for _ in range(self.num_agents)]
        else:
            done = False

        info = {"step": self.steps - 1}

        # make sure no overlapping agents
        assert np.unique(self.agents_pos, axis=0).shape[0] == self.num_agents

        # update last actions
        self.last_actions = np.zeros((self.num_agents, 5), dtype=bool)
        self.last_actions[np.arange(self.num_agents), np.array(actions)] = 1
        
        return self.observe(), rewards, done, info

    def conflict_checker(self, actions, q_val):
        # check the availability of all actions across all agents
        for agent_id in range(self.num_agents):
            if actions[agent_id] == 4:
                continue
            count = 0  # Avoid infinite loop in certain test cases
            initial_action = actions[agent_id]
            while self.is_unavailable_action(
                self.agents_pos[agent_id][0],
                self.agents_pos[agent_id][1],
                actions[agent_id],
                self.map,
            ):
                count += 1
                if count > 4:
                    actions[agent_id] = initial_action
                    # generate a warning
                    print(
                        "Warning: agent {} is stuck in an islolated cell".format(agent_id)
                    )
                    break
                q_val[agent_id][actions[agent_id]] = -np.inf
                actions[agent_id] = torch.argmax(q_val[agent_id]).item()
                # rewards[agent_id] = self.reward_fn['collision']

        checking_list = [i for i in range(self.num_agents)]

        # rewards = []
        next_pos = np.copy(self.agents_pos)

        # remove unmoving agent id from checking list
        for agent_id in checking_list.copy():
            if actions[agent_id] == 4:
                # unmoving
                # if np.array_equal(self.agents_pos[agent_id], self.goals_pos[agent_id]):
                # rewards.append(self.reward_fn['stay_on_goal'])
                # else:
                # rewards.append(self.reward_fn['stay_off_goal'])
                checking_list.remove(agent_id)
            else:
                # move
                next_pos[agent_id] += ACTION_LIST[actions[agent_id]]
                # rewards.append(self.reward_fn['move'])

        # first round check, agent swapping conflict
        no_conflict = False
        while not no_conflict:

            no_conflict = True
            for agent_id in checking_list:

                target_agent_id = np.where(
                    np.all(next_pos[agent_id] == self.agents_pos, axis=1)
                )[0]

                if target_agent_id.size > 0:

                    target_agent_id = target_agent_id.item()

                    if np.array_equal(
                        next_pos[target_agent_id], self.agents_pos[agent_id]
                    ):
                        assert (
                            target_agent_id in checking_list
                        ), "target_agent_id should be in checking list"

                        # compare the q_val of two agents and for the one with lower q_val, mask the previous action and choose the second best action.
                        if (
                            q_val[agent_id][actions[agent_id]]
                            < q_val[target_agent_id][actions[target_agent_id]]
                        ):
                            q_val[agent_id][actions[agent_id]] = -np.inf
                            actions[agent_id] = torch.argmax(q_val[agent_id]).item()
                            # move agent_id
                            next_pos[agent_id] = (
                                self.agents_pos[agent_id] + ACTION_LIST[actions[agent_id]]
                            )

                            # checking_list.remove(target_agent_id)
                        else:
                            q_val[target_agent_id][actions[target_agent_id]] = -np.inf
                            actions[target_agent_id] = torch.argmax(
                                q_val[target_agent_id]
                            ).item()
                            # move target_agent_id
                            next_pos[target_agent_id] = (
                                self.agents_pos[target_agent_id]
                                + ACTION_LIST[actions[target_agent_id]]
                            )

                            # checking_list.remove(agent_id)

                        # next_pos[agent_id] = self.agents_pos[agent_id]
                        # rewards[agent_id] = self.reward_fn['collision']

                        # next_pos[target_agent_id] = self.agents_pos[target_agent_id]
                        # rewards[target_agent_id] = self.reward_fn['collision']

                        # checking_list.remove(agent_id)
                        # checking_list.remove(target_agent_id)

                        no_conflict = False

                        break

        # second round check, agent collision conflict
        MAX_ITERS = 100
        count = 0
        no_conflict = False
        while not no_conflict:
            no_conflict = True
            count += 1
            if count > MAX_ITERS:
                break
            for agent_id in checking_list:

                collide_agent_id = np.where(
                    np.all(next_pos == next_pos[agent_id], axis=1)
                )[0].tolist()
                if len(collide_agent_id) > 1:
                    # collide agent

                    # if all agents in collide agent are in checking list
                    all_in_checking = True
                    for id in collide_agent_id.copy():
                        if id not in checking_list:
                            all_in_checking = False
                            collide_agent_id.remove(id)

                    if all_in_checking:

                        q_val_dict = {}
                        for id in collide_agent_id.copy():
                            q_val_dict[id] = q_val[id][actions[id]]
                        # get the agent_id with highest q_val
                        winner_id = max(q_val_dict, key=q_val_dict.get)

                        # resample the action for the other agents
                        for id in collide_agent_id.copy():
                            if id != winner_id:
                                q_val[id][actions[id]] = -np.inf
                                actions[id] = torch.argmax(q_val[id]).item()
                                # move agent_id
                                next_pos[id] = (
                                    self.agents_pos[id] + ACTION_LIST[actions[id]]
                                )
                            # rewards[id] = self.reward_fn['collision']
                        # checking_list.remove(winner_id)

                    else:
                        for id in collide_agent_id.copy():
                            q_val[id][actions[id]] = -np.inf
                            actions[id] = torch.argmax(q_val[id]).item()
                            # move agent_id
                            next_pos[id] = self.agents_pos[id] + ACTION_LIST[actions[id]]
                            # rewards[id] = self.reward_fn['collision']

                    # for id in collide_agent_id:
                    #     rewards[id] = self.reward_fn['collision']

                    # for id in collide_agent_id:
                    #     checking_list.remove(id)

                    no_conflict = False
                    break

        # third round check, solve final conflict in the checking_list by making all the remaining agents involved in conflicts stay

        # in the original code: (first round check, these two conflicts have the highest priority)
        for agent_id in checking_list.copy():

            # if np.any(next_pos[agent_id]<0) or np.any(next_pos[agent_id]>=self.map_size[0]):
            if (
                np.any(next_pos[agent_id] < 0)
                or next_pos[agent_id][0] >= self.map_size[0]
                or next_pos[agent_id][1] >= self.map_size[1]
            ):
                # agent out of map range
                # rewards[agent_id] = self.reward_fn['collision']  #NOTE: if want to use reward, use culmulative reward here
                next_pos[agent_id] = self.agents_pos[agent_id]
                checking_list.remove(agent_id)

            elif self.map[tuple(next_pos[agent_id])] == 1:
                # collide obstacle
                # rewards[agent_id] = self.reward_fn['collision']  #NOTE: if want to use reward, use culmulative reward here
                next_pos[agent_id] = self.agents_pos[agent_id]
                checking_list.remove(agent_id)

            if actions[agent_id] == 4:
                # remove the agent that stays, these agents decide to stay because they resampled stay actions after losing in the q value comparison.
                checking_list.remove(agent_id)

        no_conflict = False
        while not no_conflict:

            no_conflict = True
            for agent_id in checking_list:

                target_agent_id = np.where(
                    np.all(next_pos[agent_id] == self.agents_pos, axis=1)
                )[0]

                if target_agent_id.size > 0:

                    target_agent_id = target_agent_id.item()

                    if np.array_equal(
                        next_pos[target_agent_id], self.agents_pos[agent_id]
                    ):  # and target_agent_id != agent_id:
                        assert (
                            target_agent_id in checking_list
                        ), "target_agent_id should be in checking list"

                        next_pos[agent_id] = self.agents_pos[agent_id]
                        # rewards[agent_id] = self.reward_fn['collision']

                        next_pos[target_agent_id] = self.agents_pos[target_agent_id]
                        # rewards[target_agent_id] = self.reward_fn['collision']
                        try:
                            checking_list.remove(agent_id)
                            checking_list.remove(target_agent_id)
                        except:
                            print("agent ID: ", agent_id)
                            print("target agent ID: ", target_agent_id)
                            print("checking_list:", checking_list)
                            raise ValueError("target_agent_id not in checking_list")

                        no_conflict = False
                        break

        if np.unique(next_pos, axis=0).shape[0] != self.num_agents:
            # (third round check, agent collision conflict)
            no_conflict = False
            while not no_conflict:
                no_conflict = True
                for agent_id in checking_list:

                    collide_agent_id = np.where(
                        np.all(next_pos == next_pos[agent_id], axis=1)
                    )[0].tolist()
                    if len(collide_agent_id) > 1:
                        # collide agent

                        # if all agents in collide agent are in checking list
                        all_in_checking = True
                        for id in collide_agent_id.copy():
                            if id not in checking_list:
                                all_in_checking = False
                                collide_agent_id.remove(id)

                        if all_in_checking:

                            collide_agent_pos = next_pos[collide_agent_id].tolist()
                            for pos, id in zip(collide_agent_pos, collide_agent_id):
                                pos.append(id)
                            collide_agent_pos.sort(
                                key=lambda x: x[0] * self.map_size[0] + x[1]
                            )

                            collide_agent_id.remove(collide_agent_pos[0][2])

                        next_pos[collide_agent_id] = self.agents_pos[collide_agent_id]
                        # for id in collide_agent_id:
                        #     rewards[id] = self.reward_fn['collision']

                        for id in collide_agent_id:
                            checking_list.remove(id)

                        no_conflict = False
                        break

        return next_pos

    def advanced_escape_policy(self, next_pos, q_val, loc_history, actions):
        """
        Advanced escape policy to avoid deadlocks
        """

        # mask agents at goal position
        was_at_goal = (self.agents_pos == self.goals_pos).all(axis=-1)
        is_active_agent = ~(next_pos == self.goals_pos).all(axis=-1)
        # if agent was not active and is active, then it is a new active agent
        new_active_agent = was_at_goal & is_active_agent
        is_active_agent = is_active_agent & ~new_active_agent
        # check if history[-1] and history[-3] are the same
        mask_history_1 = (loc_history[:, 0] == loc_history[:, 2]).all(axis=-1)
        # check if history[-2] and history[-4] are the same
        mask_history_2 = (loc_history[:, 1] == loc_history[:, 3]).all(axis=-1)
        # now, final mask is the logical and of all the masks
        mask_final = is_active_agent & mask_history_1 & mask_history_2

        # Re-order Q-values in descending order
        q_val = q_val.clone()
        q_val_best_action = torch.max(q_val, dim=1)[0]
        agent_idx_priority_escape = torch.argsort(q_val_best_action, descending=True)

        for agent_id, iter_ in enumerate(agent_idx_priority_escape):
            if mask_final[agent_id]:
                # If distance between agent and goal is less than 3, continue and next_pos is current decided

                if config.aep_distance_threshold is not None:
                    if (
                        np.linalg.norm(
                            self.agents_pos[agent_id] - self.goals_pos[agent_id]
                        )
                        < config.aep_distance_threshold
                    ):
                        continue
                # retake current map such that obstacles and agents are obstacles
                upd_map = np.copy(self.map)

                # Only mask out previously chosen (by priority) next positions
                for id_ in agent_idx_priority_escape[:iter_]:
                    upd_map[tuple(next_pos[id_])] = 1

                # Force keeping current position
                keep_pos_flag = False

                # Check if we should only use the q value
                if config.aep_use_q_value_only:
                    use_q_value = True
                else:
                    path = self.astar(
                        upd_map,
                        self.agents_pos[agent_id],
                        self.goals_pos[agent_id],
                        next_pos,
                        self.aep_astar_type,
                    )
                    try:
                        _ = path[1][0]
                        use_q_value = False
                    except:
                        use_q_value = True

                if not use_q_value:
                    assert path is not None, "Path should not be None"
                    pos = path[0][0]
                    new_pos = path[1][0]
                    direction = (new_pos[0] - pos[0], new_pos[1] - pos[1])
                    actions[agent_id] = DIRECTION_TO_ACTION[direction]
                else:
                    # mask all obstacles and next position of other agents
                    available_actions = []
                    for action in range(5):
                        if not self.is_unavailable_action(
                            self.agents_pos[agent_id][0],
                            self.agents_pos[agent_id][1],
                            action,
                            upd_map,
                        ):
                            available_actions.append(action)
                    # choose action among available actions with highest q_val
                    try:
                        # take q val of the available actions
                        q_val_ = q_val[agent_id][available_actions]
                        # take index of largest q val
                        idx = torch.argmax(q_val_).item()
                        actions[agent_id] = available_actions[idx]
                    except:
                        keep_pos_flag = True

                # Update next pos
                if not keep_pos_flag:
                    next_pos[agent_id] = (
                        self.agents_pos[agent_id] + ACTION_LIST[actions[agent_id]]
                    )
                else:
                    next_pos[agent_id] = self.agents_pos[agent_id]

            # # Update the next position of the current agent as an obstacle
            # upd_map[tuple(next_pos[agent_id])] = 1

        # DIRECTION_TO_ACTION
        # compare next pos with current pos to get the direction
        actions = [
            DIRECTION_TO_ACTION[
                (
                    next_pos[i][0] - self.agents_pos[i][0],
                    next_pos[i][1] - self.agents_pos[i][1],
                )
            ]
            for i in range(self.num_agents)
        ]
        next_pos = self.conflict_checker(actions, q_val)

        # Update the history of the agents' locations
        # add the next_pos to the history (beginning) and remove the last element
        loc_history = np.roll(loc_history, 1, axis=1)
        loc_history[:, 0] = next_pos
        return next_pos, loc_history

    #########################################
    # Step function for testing

    def step_infer(self, q_val, no_active_agent_nearby, loc_history):
        """
        location history: [num_agents, 4] # 4 previous steps
        actions:
            list of indices
                0 up
                1 down
                2 left
                3 right
                4 stay
        Step function for testing phase, where q_val is the output of the network.

        no_active_agent_nearby: a list of bool values with each value indicating whether there is other agents within agent i's observation
        """
        # NOTE: rewards are not used in testing phase, if want to use rewards, need to modify the code. for example, use culmulative rewards instead of a single action reward.
        q_val = torch.as_tensor(q_val)
        q_val_init = q_val.clone()
        actions = torch.argmax(q_val, 1).tolist()

        # Note that A* type can be set to None, then the A* path will not be used
        if self.astar_type is not None:
            # if no active agent nearby, then follow the A* path to goal for one step
            for agent_id in range(self.num_agents):
                if no_active_agent_nearby[agent_id][0] and not np.array_equal(
                    self.agents_pos[agent_id], self.goals_pos[agent_id]
                ):
                    path = self.astar(
                        self.map,
                        self.agents_pos[agent_id],
                        self.goals_pos[agent_id],
                        self.agents_pos,
                    )
                    if path is not None:
                        pos = path[0][0]
                        new_pos = path[1][0]
                        direction = (new_pos[0] - pos[0], new_pos[1] - pos[1])
                        actions[agent_id] = DIRECTION_TO_ACTION[direction]

        # If using A*1: not necessary to check the validity of actions from A* path
        # WARNING: if using A*2, necessary to check the validity of actions from A*2 path

        assert (
            len(actions) == self.num_agents
        ), "only {} actions as input while {} agents in environment".format(
            len(actions), self.num_agents
        )
        assert all(
            [action_idx < 5 and action_idx >= 0 for action_idx in actions]
        ), "action index out of range"

        # # (second round check, agent swapping conflict)
        # next_pos, checking_list = self.conflict_checker(next_pos, checking_list)

        # Get next position with current actions and q values for priorities
        next_pos = self.conflict_checker(actions, q_val)

        # Advanced Escape Policy
        if self.use_aep:
            next_pos, loc_history = self.advanced_escape_policy(
                next_pos, q_val_init, loc_history, actions
            )

        #### UPDATE THE AGENT POSITION (after all the checks and updates)
        self.agents_pos = np.copy(next_pos)

        self.steps += 1

        # check done
        if np.array_equal(self.agents_pos, self.goals_pos):
            done = True
            # rewards = [self.reward_fn['finish'] for _ in range(self.num_agents)]
        else:
            done = False

        info = {"step": self.steps - 1}

        assert (
            np.unique(self.agents_pos, axis=0).shape[0] == self.num_agents
        ), "agent overlapping"

        # update last actions
        self.last_actions = np.zeros((self.num_agents, 5), dtype=bool)
        self.last_actions[np.arange(self.num_agents), np.array(actions)] = 1

        # return self.observe(), done, info
        return self.observe_infer(), done, info, loc_history

    def is_unavailable_action(self, curr_x, curr_y, action, map_=None):
        if map_ is None:
            map_ = self.map
        next_x = curr_x + ACTION_LIST[action][0]
        next_y = curr_y + ACTION_LIST[action][1]
        if (
            next_x < 0
            or next_x >= self.map_size[0]
            or next_y < 0
            or next_y >= self.map_size[1]
        ):
            return True
        elif map_[next_x, next_y] == 1:
            return True
        else:
            return False

    def observe(self):
        """
        return observation and position for each agent

        obs: shape (num_agents, 7, 2*obs_radius+1, 2*obs_radius+1)
            layer 1: agent map
            layer 2: obstacle map
            layer 3-6: heuristic map

        last_act: agents' last step action

        pos: current position of each agent, used for caculating communication mask

        """
        obs = np.zeros(
            (self.num_agents, 8, 2 * self.obs_radius + 1, 2 * self.obs_radius + 1),
            dtype=float,
        )

        obstacle_map = np.pad(self.map, self.obs_radius, "constant", constant_values=0)
        block_area_padded = np.pad(self.block_area, self.obs_radius, "constant", constant_values=True)
        has_go_padded = np.pad(self.has_go_map, ((0, 0), (self.obs_radius, self.obs_radius), (self.obs_radius, self.obs_radius)), "constant", constant_values=True)

        agent_map = np.zeros((self.map_size), dtype=bool)
        agent_map[self.agents_pos[:, 0], self.agents_pos[:, 1]] = 1
        agent_map = np.pad(agent_map, self.obs_radius, "constant", constant_values=0)

        for i, agent_pos in enumerate(self.agents_pos):
            x, y = agent_pos

            obs[i, 0] = agent_map[
                x : x + 2 * self.obs_radius + 1, y : y + 2 * self.obs_radius + 1
            ]
            obs[i, 0, self.obs_radius, self.obs_radius] = 0
            obs[i, 1] = obstacle_map[
                x : x + 2 * self.obs_radius + 1, y : y + 2 * self.obs_radius + 1
            ]
            obs[i, 2:-2] = self.heuri_map[
                i, :, x : x + 2 * self.obs_radius + 1, y : y + 2 * self.obs_radius + 1
            ]
            obs[i, -2] = block_area_padded[
                x : x + 2 * self.obs_radius + 1, y : y + 2 * self.obs_radius + 1
            ]
            obs[i, -1] = has_go_padded[
                i, x: x + 2 * self.obs_radius + 1, y: y + 2 * self.obs_radius + 1
            ]
            
        return obs, np.copy(self.last_actions), np.copy(self.agents_pos)

    def observe_infer(self):
        """
        return observation and position for each agent, plus a mask indicating whether there are other live agents within agent i's observation range

        obs: shape (num_agents, 7, 2*obs_radius+1, 2*obs_radius+1)
            layer 1: agent map
            layer 2: obstacle map
            layer 3-6: heuristic map

        last_act: agents' last step action

        pos: current position of each agent, used for caculating communication mask

        """
        obs = np.zeros(
            (self.num_agents, 8, 2 * self.obs_radius + 1, 2 * self.obs_radius + 1),
            dtype=bool,
        )

        # obs_finished_or_no = np.zeros((self.num_agents, 2*self.obs_radius+1, 2*self.obs_radius+1), dtype=bool)
        obs_finished_or_no = np.zeros(
            (
                self.num_agents,
                2 * self.active_agent_radius + 1,
                2 * self.active_agent_radius + 1,
            ),
            dtype=bool,
        )

        obstacle_map = np.pad(self.map, self.obs_radius, "constant", constant_values=0)
        # pad with 0, so later the boundary is [x:x+2*self.obs_radius+1, y:y+2*self.obs_radius+1]
        # instead of [x-self.obs_radius:x+self.obs_radius+1, y-self.obs_radius:y+self.obs_radius+1] 
        block_area_padded = np.pad(self.block_area, self.obs_radius, "constant", constant_values=True)
        has_go_padded = np.pad(self.has_go_map, ((0, 0), (self.obs_radius, self.obs_radius), (self.obs_radius, self.obs_radius)), "constant", constant_values=True)

        
        agent_map = np.zeros((self.map_size), dtype=bool)
        agent_map[self.agents_pos[:, 0], self.agents_pos[:, 1]] = 1
        agent_map = np.pad(
            agent_map, self.obs_radius, "constant", constant_values=0
        )  # pad with 0, so later...

        unfinished_agents_map = np.zeros((self.map_size), dtype=bool)
        for i in range(self.num_agents):
            if not np.array_equal(self.agents_pos[i], self.goals_pos[i]):
                unfinished_agents_map[self.agents_pos[i][0], self.agents_pos[i][1]] = 1
        # unfinished_agents_map = np.pad(unfinished_agents_map, self.obs_radius, 'constant', constant_values=0)
        unfinished_agents_map = np.pad(
            unfinished_agents_map, self.active_agent_radius, "constant", constant_values=0
        )

        for i, agent_pos in enumerate(self.agents_pos):
            x, y = agent_pos

            obs[i, 0] = agent_map[
                x : x + 2 * self.obs_radius + 1, y : y + 2 * self.obs_radius + 1
            ]
            obs[i, 0, self.obs_radius, self.obs_radius] = 0
            obs[i, 1] = obstacle_map[
                x : x + 2 * self.obs_radius + 1, y : y + 2 * self.obs_radius + 1
            ]
            obs[i, 2:-2] = self.heuri_map[
                           i, :, x: x + 2 * self.obs_radius + 1, y: y + 2 * self.obs_radius + 1
                           ]
            obs[i, -2] = block_area_padded[
                         x: x + 2 * self.obs_radius + 1, y: y + 2 * self.obs_radius + 1
                         ]
            obs[i, -1] = has_go_padded[
                         i, x: x + 2 * self.obs_radius + 1, y: y + 2 * self.obs_radius + 1
                         ]

            obs_finished_or_no[i] = unfinished_agents_map[
                x : x + 2 * self.active_agent_radius + 1,
                y : y + 2 * self.active_agent_radius + 1,
            ]
            obs_finished_or_no[i, self.active_agent_radius, self.active_agent_radius] = 0

        # create a list of bool values with each value indicating whether there is other agents within agent i's observation range
        no_active_agent_nearby = np.ones((self.num_agents, 1), dtype=bool)
        for i in range(self.num_agents):
            if np.any(obs_finished_or_no[i]):
                no_active_agent_nearby[i][0] = False

        return (
            obs,
            np.copy(self.last_actions),
            np.copy(self.agents_pos),
            no_active_agent_nearby,
        )

    def astar(self, map, start, goal, agent_pos, astar_type=None):
        """Types:
        - 1: consider obstacles all agents
        - 2: consider obstacles terminated agents only
        - Set to None only in self.astar_type to not use A* path
        """
        if astar_type is None:
            astar_type = (
                self.astar_type if self.astar_type is not None else 1
            )  # default to 1
        if astar_type == 1:
            return self._astar1(map, start, goal, agent_pos)
        elif astar_type == 2:
            return self._astar2(map, start, goal, agent_pos)
        elif astar_type == 3:
            return self._astar3(map, start, goal, agent_pos)
        else:
            raise ValueError(
                "Invalid astar_type: {}".format(self.astar_type),
                "astar_type should be 1 (consider obstacles all agents) or 2 (consider obstacles terminated agents only)",
            )

    def _astar1(self, map, start, goal, agent_pos):
        """A* function for single agent, suitable for maps with more agents
        robots: all other agents except the current agent
        """
        world = np.copy(map)
        robots = np.copy(agent_pos)
        # delete start from robots
        robots = np.delete(robots, np.where(np.all(robots == start, axis=1)), axis=0)

        for i, j in robots:
            world[i, j] = 1
        return self._astar_main(world, start, goal)

    def _astar2(self, map, start, goal, agent_pos):
        """A* function for single agent, suitable for maps with few agents
        robots: agents that have completed their tasks
        """
        world = np.copy(map)
        robots = np.copy(agent_pos)

        agents_to_remove = []
        for i in range(self.num_agents):
            if np.array_equal(start, robots[i]) or not np.array_equal(
                self.goals_pos[i], robots[i]
            ):
                agents_to_remove.append(i)

        # Create a new list that contains only the agents you want to keep
        filtered_robots = [
            robots[i] for i in range(len(robots)) if i not in agents_to_remove
        ]
        robots = filtered_robots

        for i, j in robots:  # robots: agents that have completed their tasks
            world[i, j] = 1

        return self._astar_main(world, start, goal)

    def _astar3(self, map, start, goal, agent_pos):
        # simply call astar main without map modification
        world = np.copy(map)
        return self._astar_main(world, start, goal)

    def _astar_main(self, world, start, goal):
        try:
            path = od_mstar.find_path(world, [start], [goal], inflation=1, time_limit=5)
        except Exception as exception:
            if isinstance(exception, NoSolutionError) or isinstance(
                exception, OutOfTimeError
            ):
                path = None
            else:
                path = None
                print("Unknown Exception Detected: ", exception)
        return path
        
    def check_congestion_area(self, k=config.congestion_expend):
        '''
        统计整个地图中有多少个拥塞分区, 每个分区按照config.congestion_radius进行划分
        返回拥塞区域图, block_area[i][j] 表示(i, j)位置是否是拥塞区域
        '''
        block_area = np.zeros_like(self.map, dtype=np.bool)
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        # 清除已经非拥塞的对象
        for group in self.conflict_group:
            group_pos = [self.agents_pos[id] for id in group]
            average_distance = np.mean(pdist(group_pos, 'euclidean'))
            arrived_goal = np.array([self.agents_pos[id] == self.goals_pos[id] for id in group])
            # 如果相对平均距离大于阈值 或者 所有的agent都到达了目标点，那么就不再是拥塞组
            if average_distance >= config.congestion_dispair or arrived_goal.all():
                self.conflict_group.remove(group)
        
        def dfs(x, y, depth, is_single_road):
            if depth > k or block_area[x][y] or self.map[x][y] == 1:
                return
            
            block_area[x][y] = True
            
            moveable_directions = sum(
                1 for dx, dy in directions
                if 0 <= x + dx < self.map_size[0] and 0 <= y + dy < self.map_size[1] and self.map[x + dx][y + dy] == 0
            )
            
            if moveable_directions < 3:
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.map_size[0] and 0 <= ny < self.map_size[1]:
                        dfs(nx, ny, depth, True)
            else:
                if is_single_road:
                    return
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.map_size[0] and 0 <= ny < self.map_size[1]:
                        dfs(nx, ny, depth + 1, False)
        
        for group in self.conflict_group:
            for id in group:
                x, y = self.agents_pos[id]
                block_area[x][y] = False
                dfs(x, y, 0, False)
        
        return block_area