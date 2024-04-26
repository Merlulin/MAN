import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import config


class CommLayer(nn.Module):
    def __init__(
        self, input_dim=config.hidden_dim, message_dim=32, pos_embed_dim=16, num_heads=4
    ):
        '''
        args:
            input_dim: 输入维度 256
            message_dim: 消息维度 32
            pos_embed_dim: 位置嵌入维度 16
            num_heads: 多头注意力机制的头数 4
        '''
        super().__init__()
        self.input_dim = input_dim
        self.message_dim = message_dim
        self.pos_embed_dim = pos_embed_dim
        self.num_heads = num_heads

        self.norm = nn.LayerNorm(input_dim)
        
        # 位置嵌入层
        self.position_embeddings = nn.Linear(
            (2 * config.obs_radius + 1) ** 2, pos_embed_dim
        )

        # KV的维度均是从input_dim + pos_embed_dim到message_dim * num_heads，Q作为询问并不需要加入位置嵌入
        self.message_key = nn.Linear(input_dim + pos_embed_dim, message_dim * num_heads)
        self.message_value = nn.Linear(input_dim + pos_embed_dim, message_dim * num_heads)
        self.hidden_query = nn.Linear(input_dim, message_dim * num_heads)

        # 多头汇集
        self.head_agg = nn.Linear(message_dim * num_heads, message_dim * num_heads)

        # 使用GRU门限单元控制信息流
        self.update = nn.GRUCell(num_heads * message_dim, input_dim)

    def position_embed(self, relative_pos, dtype, device):
        '''
        位置嵌入
        '''
        batch_size, num_agents, _, _ = relative_pos.size()
        # 将所有在观测范围外的智能体的相对位置信息置为0
        relative_pos[(relative_pos.abs() > config.obs_radius).any(3)] = 0
        # 一维位置编码
        one_hot_position = torch.zeros(
            (batch_size * num_agents * num_agents, 9 * 9), dtype=dtype, device=device
        )
        relative_pos += config.obs_radius
        relative_pos = relative_pos.reshape(batch_size * num_agents * num_agents, 2)
        # 将二维相对位置信息转变为一维位置索引
        relative_pos_idx = relative_pos[:, 0] + relative_pos[:, 1] * 9

        one_hot_position[
            torch.arange(batch_size * num_agents * num_agents), relative_pos_idx.long()
        ] = 1
        position_embedding = self.position_embeddings(one_hot_position)

        return (
            position_embedding  # size: batch_size*num_agents*num_agents x pos_embed_dim
        )

    def forward(self, hidden, relative_pos, comm_mask):
        """
        hidden shape: batch_size x num_agents x latent_dim
        relative_pos shape: batch_size x num_agents x num_agents x 2
        comm_mask shape: batch_size x num_agents x num_agents
        """
        batch_size, num_agents, hidden_dim = hidden.size()
        # 根据comm_mask生成注意力掩码，同时在第四个和第五个维度上增加维度， shape：batch_size x num_agents x num_agents x 1 x 1
        attn_mask = (comm_mask == False).unsqueeze(3).unsqueeze(4)
        relative_pos = relative_pos.clone()

        # postion embedding (batch_size*num_agents*num_agents x pos_embed_dim)
        position_embedding = self.position_embed(
            relative_pos, hidden.dtype, hidden.device
        )

        input = hidden

        hidden = self.norm(hidden)

        hidden_q = self.hidden_query(hidden).view(
            batch_size, 1, num_agents, self.num_heads, self.message_dim
        )  # batch_size x num_agents x message_dim*num_heads

        # 通过repeat_interleave重复张量，用于后续与位置编码进行维度一致，便于拼接
        message_input = hidden.repeat_interleave(num_agents, dim=1).view(
            batch_size * num_agents * num_agents, hidden_dim
        )
        # 将位置嵌入与输入拼接
        message_input = torch.cat((message_input, position_embedding), dim=1)
        message_input = message_input.view(
            batch_size, num_agents, num_agents, self.input_dim + self.pos_embed_dim
        )
        # 生成key和value
        message_k = self.message_key(message_input).view(
            batch_size, num_agents, num_agents, self.num_heads, self.message_dim
        )
        message_v = self.message_value(message_input).view(
            batch_size, num_agents, num_agents, self.num_heads, self.message_dim
        )

        # attention: batch_size x num_agents x num_agents x self.num_heads x 1
        attn_score = (hidden_q * message_k).sum(
            4, keepdim=True
        ) / self.message_dim**0.5  
        # 利用attn_mask进行掩码, 将不需要的交互位置的注意力值设置为负无穷，因为attn_mask是基于num_agents 和 num_agents
        # 所以在两个智能体如果不需要交互，则对应的attn_mask值为True，同时逐一attn_mask在第四和第五个维度会自动扩展
        attn_score.masked_fill_(attn_mask, torch.finfo(attn_score.dtype).min)
        attn_weights = F.softmax(attn_score, dim=1)

        # agg
        agg_message = (
            (message_v * attn_weights)
            .sum(1)
            .view(batch_size, num_agents, self.num_heads * self.message_dim)
        )
        agg_message = self.head_agg(agg_message)

        # update hidden with request message
        input = input.view(-1, hidden_dim)
        agg_message = agg_message.view(
            batch_size * num_agents, self.num_heads * self.message_dim
        )
        updated_hidden = self.update(agg_message, input)

        # some agents may not receive message, keep it as original
        update_mask = comm_mask.any(1).view(-1, 1)
        hidden = torch.where(update_mask, updated_hidden, input)
        hidden = hidden.view(batch_size, num_agents, hidden_dim)

        return hidden


class CommBlock(nn.Module):
    def __init__(self, hidden_dim=config.hidden_dim, message_dim=128, pos_embed_dim=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.message_dim = message_dim
        self.pos_embed_dim = pos_embed_dim

        self.request_comm = CommLayer()
        self.reply_comm = CommLayer()

    def forward(self, latent, relative_pos, comm_mask):
        """
        latent shape: batch_size x num_agents x latent_dim
        relative_pos shape: batch_size x num_agents x num_agents x 2
        comm_mask shape: batch_size x num_agents x num_agents
        """

        batch_size, num_agents, latent_dim = latent.size()

        assert relative_pos.size() == (
            batch_size,
            num_agents,
            num_agents,
            2,
        ), relative_pos.size()
        assert comm_mask.size() == (batch_size, num_agents, num_agents), comm_mask.size()

        if torch.sum(comm_mask).item() == 0:
            # 如果没有通信对象，则直接返回
            return latent

        hidden = self.request_comm(latent, relative_pos, comm_mask)

        comm_mask = torch.transpose(comm_mask, 1, 2)

        hidden = self.reply_comm(hidden, relative_pos, comm_mask)

        return hidden


class Network(nn.Module):
    def __init__(
        self, input_shape=config.obs_shape, selective_comm=config.selective_comm
    ):
        super().__init__()

        self.hidden_dim = config.hidden_dim
        self.latent_dim = self.hidden_dim + 5
        self.obs_shape = input_shape
        self.selective_comm = selective_comm

        self.obs_encoder = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, 3, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 3, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 192, 3, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(192, 256, 3, 1),
            nn.LeakyReLU(0.2, True),
            nn.Flatten(),
        )
        
        self.recurrent = nn.GRUCell(self.latent_dim, self.hidden_dim)
        self.comm = CommBlock(self.hidden_dim)

        self.hidden = None

        # dueling q structure
        self.adv = nn.Linear(self.hidden_dim, 5)
        self.state = nn.Linear(self.hidden_dim, 1)

    @torch.no_grad()
    def step(self, obs, last_act, pos, goal):
        '''
        last_act: 上一步的动作Q值, shape: (num_agents, config.action_dim)
        pos: 目标位置信息, shape: (num_agents, 2)
        '''
        num_agents = obs.size(0)                                 # 智能体的总数
        agent_indexing = torch.arange(num_agents).to(obs.device) # 智能体的索引
        
        # 通过unsqueeze插入维度，然后通过pytorch的广播功能实现智能体之间的相对位置计算
        # pos.unsqueeze(0)的形状为(1, num_agents, 2)，pos.unsqueeze(1)的形状为(num_agents, 1, 2)
        # realative_pos的形状为(num_agents, num_agents, 2)
        relative_pos = pos.unsqueeze(0) - pos.unsqueeze(1) 

        # 首先根据配置的obs_radius获取所有在观测范围内智能体： 先对relative_pos取绝对值
        # .all(2)检查判断后为[dif_x, dif_y]是否都在观测半径内。结果是一个形状为(n, n)的二维张量，每一项标识当前列索引的智能体是否在当前行索引的智能体的观测范围内
        in_obs_mask = (relative_pos.abs() <= config.obs_radius).all(2) 
        in_obs_mask[agent_indexing, agent_indexing] = 0 # 排除自己

        # 如果开启了selective_comm，则进行范围内的选择性通信
        if self.selective_comm:
            test_mask = in_obs_mask.clone()                                 # 复制一个in_obs_mask, 标识FOV内的所有可见的智能体
            test_mask[agent_indexing, agent_indexing] = 1                   # 将自己标记为可见
            num_in_obs_agents = test_mask.sum(1)                            # 计算每个智能体在FOV内的可见智能体数量
            origin_agent_idx = torch.zeros(num_agents, dtype=torch.long)    # 用于记录每个智能体在FOV内的可见智能体的索引
            for i in range(num_agents - 1):
                origin_agent_idx[i + 1] = (                                 # 计算每个智能体在所有FOV内的原始索引值,因为代码中所有的智能体编号是根据总的FOV顺从0开始一个一个存的
                    test_mask[i, i:].sum()                                  # 所以必须得到每个FOV中智能体编号的偏置
                    + test_mask[i + 1, : i + 1].sum()                       
                    + origin_agent_idx[i]
                )
            # repeat_interleave函数会在指定的维度上重复张量的元素, 然后通过view函数将张量的形状转换为(num_agents, num_agents, *config.obs_shape)
            # 最后通过[test_mask]索引获取每个智能体个体中所有在各自FOV内的智能体的观测信息
            # test_obs的形状为(num_in_obs_agents.sum(), *config.obs_shape)
            test_obs = torch.repeat_interleave(obs, num_agents, dim=0).view(
                num_agents, num_agents, *config.obs_shape
            )[test_mask]

            test_relative_pos = relative_pos[test_mask] 
            test_relative_pos += config.obs_radius # += obs_radius, 主要是obs中存信息的时候,为了方便表示范围是0 ~ map_size + 2*obs_radius, 所以这里要加上obs_radius

            # 把自己的位置信息置为0, 因为第0号obs是agent map
            test_obs[
                torch.arange(num_in_obs_agents.sum()),
                0,
                test_relative_pos[:, 0],
                test_relative_pos[:, 1],
            ] = 0

            # 获取所有智能体上一次的操作Q值, test_last_act的形状为(num_agent x *(num_in_obs_agents), config.action_dim)
            test_last_act = torch.repeat_interleave(last_act, num_in_obs_agents, dim=0)
            if self.hidden is None:
                # 如果hidden为空, 则初始化为0
                test_hidden = torch.zeros((num_in_obs_agents.sum(), self.hidden_dim)).to(
                    obs.device
                )
            else:
                test_hidden = torch.repeat_interleave(
                    self.hidden, num_in_obs_agents, dim=0
                ).to(obs.device)

            # 选择性通信的第一步,先计算全部的通信对Q值的影响
            test_latent = self.obs_encoder(test_obs)
            test_latent = torch.cat((test_latent, test_last_act), dim=1)

            test_hidden = self.recurrent(test_latent, test_hidden)
            # 将GRU输出值对应的每个智能体的值取出保存在hidden中
            self.hidden = test_hidden[origin_agent_idx]
            
            # 通过Q network计算Q值
            adv_val = self.adv(test_hidden)
            state_val = self.state(test_hidden)
            test_q_val = state_val + adv_val - adv_val.mean(1, keepdim=True)
            test_actions = torch.argmax(test_q_val, 1)
            
            # actions_mat的形状为(num_agents, num_agents), 用于记录每个智能体的动作
            actions_mat = (
                torch.ones(
                    (num_agents, num_agents), dtype=test_actions.dtype, device=obs.device
                )
                * -1
            )
            # 将每个智能体的动作填入actions_mat中
            actions_mat[test_mask] = test_actions
            # 将所有加入通信后的行为与原始行为进行比较, 如果不一样则标记为1, 说明这些智能体需要进行通信
            diff_action_mask = actions_mat != actions_mat[
                agent_indexing, agent_indexing
            ].unsqueeze(1)
            

            assert (in_obs_mask[agent_indexing, agent_indexing] == 0).all()
            # 执行了一个按位与（bitwise AND）操作，对in_obs_mask和diff_action_mask两个张量进行元素级别的逻辑与操作。
            # 及comm_mask必须同时满足两个条件
            comm_mask = torch.bitwise_and(in_obs_mask, diff_action_mask)

        else:

            latent = self.obs_encoder(obs)
            latent = torch.cat((latent, last_act), dim=1)

            # mask out agents that are far away
            dist_mat = relative_pos[:, :, 0] ** 2 + relative_pos[:, :, 1] ** 2
            _, ranking = dist_mat.topk(
                min(config.max_comm_agents, num_agents), dim=1, largest=False
            )
            dist_mask = torch.zeros((num_agents, num_agents), dtype=torch.bool)
            dist_mask.scatter_(1, ranking, True)
            comm_mask = torch.bitwise_and(in_obs_mask, dist_mask)
            # comm_mask[torch.arange(num_agents), torch.arange(num_agents)] = 0

            if self.hidden is None:
                self.hidden = self.recurrent(latent)
            else:
                self.hidden = self.recurrent(latent, self.hidden)

        assert (comm_mask[agent_indexing, agent_indexing] == 0).all()

        self.hidden = self.comm(
            self.hidden.unsqueeze(0), relative_pos.unsqueeze(0), comm_mask.unsqueeze(0)
        )
        self.hidden = self.hidden.squeeze(0)

        adv_val = self.adv(self.hidden)
        state_val = self.state(self.hidden)

        q_val = state_val + adv_val - adv_val.mean(1, keepdim=True)

        actions = torch.argmax(q_val, 1).tolist()

        return (
            actions,
            q_val.cpu().numpy(),
            self.hidden.squeeze(0).cpu().numpy(),
            relative_pos.cpu().numpy(),
            comm_mask.cpu().numpy(),
        )

    def reset(self):
        self.hidden = None

    @torch.autocast(device_type="cuda")
    def forward(self, obs, last_act, steps, hidden, relative_pos, comm_mask):
        """
        used for training
        block_area: (seq_len, batch_size, maplen, maplen)
        """
        # obs shape: seq_len（预热步 + 前向步长：用于标识是否发现）, batch_size, num_agents, obs_shape
        # relative_pos shape: batch_size, seq_len, num_agents, num_agents, 2
        seq_len, batch_size, num_agents, *_ = obs.size()

        # obs: (seq_len, batch_size, num_agents, 6, 2 * obs_radius + 1, 2 * obs_radius + 1) -> (seq_len * batch_size * num_agents, 6, 2 * obs_radius + 1, 2 * obs_radius + 1)
        obs = obs.view(seq_len * batch_size * num_agents, *self.obs_shape)
        last_act = last_act.view(seq_len * batch_size * num_agents, config.action_dim)
        
        # latent: (seq_len * batch_size * num_agents, 256 * (2 * obs_radius + 1) ^ 2)
        latent = self.obs_encoder(obs)
        # latent: (seq_len * batch_size * num_agents, 256 * (2 * obs_radius + 1) ^ 2 + 5)
        latent = torch.cat((latent, last_act), dim=1)
        # latent: (seq_len, batch_size * num_agents, 256 * (2 * obs_radius + 1) ^ 2 + 5)
        latent = latent.view(seq_len, batch_size * num_agents, self.latent_dim)

        hidden_buffer = []
        for i in range(seq_len):
            # hidden size: batch_size*num_agents x self.hidden_dim(256)
            hidden = self.recurrent(latent[i], hidden)
            # hidden size: (batch_size, num_agents, self.hidden_dim(256))
            hidden = hidden.view(batch_size, num_agents, self.hidden_dim)
            hidden = self.comm(hidden, relative_pos[:, i], comm_mask[:, i])
            # only hidden from agent 0
            hidden_buffer.append(hidden[:, 0])
            hidden = hidden.view(batch_size * num_agents, self.hidden_dim)

        # hidden buffer size: batch_size x seq_len x self.hidden_dim
        hidden_buffer = torch.stack(hidden_buffer).transpose(0, 1)

        # hidden size: batch_size x self.hidden_dim
        hidden = hidden_buffer[torch.arange(config.batch_size), steps - 1]

        adv_val = self.adv(hidden)
        state_val = self.state(hidden)

        q_val = state_val + adv_val - adv_val.mean(1, keepdim=True)

        return q_val
