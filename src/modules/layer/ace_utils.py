import torch
import torch.nn as nn
from typing import Callable, Optional

def build_normalization(norm_type: str, dim: Optional[int] = None) -> nn.Module:
    r"""
    Overview:
        Build the corresponding normalization module
    Arguments:
        - norm_type (:obj:`str`): type of the normaliztion, now support ['BN', 'IN', 'SyncBN', 'AdaptiveIN']
        - dim (:obj:`int`): dimension of the normalization, when norm_type is in [BN, IN]
    Returns:
        - norm_func (:obj:`nn.Module`): the corresponding batch normalization function

    .. note::
        For beginers, you can refer to <https://zhuanlan.zhihu.com/p/34879333> to learn more about batch normalization.
    """
    if dim is None:
        key = norm_type
    else:
        if norm_type in ['BN', 'IN', 'SyncBN']:
            key = norm_type + str(dim)
        elif norm_type in ['LN']:
            key = norm_type
        else:
            raise NotImplementedError("not support indicated dim when creates {}".format(norm_type))
    norm_func = {
        'BN1': nn.BatchNorm1d,
        'BN2': nn.BatchNorm2d,
        'LN': nn.LayerNorm,
        'IN2': nn.InstanceNorm2d,
    }
    if key in norm_func.keys():
        return norm_func[key]
    else:
        raise KeyError("invalid norm type: {}".format(key))
    
def sequential_pack(layers: list) -> nn.Sequential:
    r"""
    Overview:
        Pack the layers in the input list to a `nn.Sequential` module.
        If there is a convolutional layer in module, an extra attribute `out_channels` will be added
        to the module and set to the out_channel of the conv layer.
    Arguments:
        - layers (:obj:`list`): the input list
    Returns:
        - seq (:obj:`nn.Sequential`): packed sequential container
    """
    assert isinstance(layers, list)
    seq = nn.Sequential(*layers)
    for item in reversed(layers):
        if isinstance(item, nn.Conv2d) or isinstance(item, nn.ConvTranspose2d):
            seq.out_channels = item.out_channels
            break
        elif isinstance(item, nn.Conv1d):
            seq.out_channels = item.out_channels
            break
    return seq

class RelationAggregator(nn.Module):
    """Relation Aggregator from ACE for processing state-relation data"""
    def __init__(
            self,
            state_len: int,
            relation_len: int,
    ) -> None:
        super(RelationAggregator, self).__init__()
        self._state_encoder = nn.Sequential(
            nn.Linear(state_len + relation_len, state_len),
            nn.ReLU(inplace=True),
        )

    def forward(self, state, relation, alive_mask):
        relation_avr, relation_max = relation.chunk(2, dim=-1)
        relation_avr = (relation_avr * alive_mask.unsqueeze(1).unsqueeze(-1)).mean(-2)
        relation_max = (relation_max * alive_mask.unsqueeze(1).unsqueeze(-1)).max(-2).values
        state = self._state_encoder(torch.cat([state, relation_avr, relation_max], dim=-1))
        return state

def MLP(
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    layer_num: int,
    layer_fn: Callable = None,
    activation: nn.Module = None,
    norm_type: str = None,
    use_dropout: bool = False,
    dropout_probability: float = 0.5
):
    r"""
    Overview:
        create a multi-layer perceptron using fully-connected blocks with activation, normalization and dropout,
        optional normalization can be done to the dim 1 (across the channels)
        x -> fc -> norm -> act -> dropout -> out
    Arguments:
        - in_channels (:obj:`int`): Number of channels in the input tensor
        - hidden_channels (:obj:`int`): Number of channels in the hidden tensor
        - out_channels (:obj:`int`): Number of channels in the output tensor
        - layer_num (:obj:`int`): Number of layers
        - layer_fn (:obj:`Callable`): layer function
        - activation (:obj:`nn.Module`): the optional activation function
        - norm_type (:obj:`str`): type of the normalization
        - use_dropout (:obj:`bool`): whether to use dropout in the fully-connected block
        - dropout_probability (:obj:`float`): probability of an element to be zeroed in the dropout. Default: 0.5
    Returns:
        - block (:obj:`nn.Sequential`): a sequential list containing the torch layers of the fully-connected block
    .. note::

        you can refer to nn.linear (https://pytorch.org/docs/master/generated/torch.nn.Linear.html)
    """
    assert layer_num >= 0, layer_num
    if layer_num == 0:
        return sequential_pack([nn.Identity()])

    channels = [in_channels] + [hidden_channels] * (layer_num - 1) + [out_channels]
    if layer_fn is None:
        layer_fn = nn.Linear
    block = []
    for i, (in_channels, out_channels) in enumerate(zip(channels[:-1], channels[1:])):
        block.append(layer_fn(in_channels, out_channels))
        if norm_type is not None:
            block.append(build_normalization(norm_type, dim=1)(out_channels))
        if activation is not None:
            block.append(activation)
        if use_dropout:
            block.append(nn.Dropout(dropout_probability))
    return sequential_pack(block)

def ActionSampler(logit, action_mask, cfg):
    if cfg['type'] == 'arg_max':
        return (logit - 1e9 * (~action_mask.bool())).max(-1).indices
    elif cfg['type'] == 'eps_greedy':
        action_max = (logit - 1e9 * (~action_mask.bool())).max(-1).indices
        action_rnd = torch.multinomial(action_mask.float(), 1).squeeze(-1)
        rand_mask = (torch.rand(logit.shape[:-1]) < cfg['eps']).to(logit.device)
        return (rand_mask.float() * action_rnd + (1 - rand_mask.float()) * action_max).long()
    elif cfg['type'] == 'boltzman':
        action_max = (logit - 1e9 * (~action_mask.bool())).max(-1).indices
        action_rnd = torch.multinomial(action_mask.float(), 1).squeeze(-1)
        action_bzm = torch.multinomial(logit.softmax(-1) * action_mask.float(), 1).squeeze(-1)
        rand_mask = (torch.rand(logit.shape[:-1]) < cfg['eps']).to(logit.device)
        btzm_mask = (torch.rand(logit.shape[:-1]) < cfg['bzm']).to(logit.device)
        return (rand_mask.float() * (btzm_mask.float() * action_bzm + (1 - btzm_mask.float()) * action_rnd) + (
                1 - rand_mask.float()) * action_max).long()

class DecisionEncoder(nn.Module):
    def __init__(
            self,
            embed_num: int,  # 6 [dead, stop, move_n, move_s, move_e, move_w]
            hidden_len: int,
            update_state: bool = True,
    ) -> None:
        super(DecisionEncoder, self).__init__()
        self.embed_num = embed_num
        self.update_state = update_state
        self._action_embed = nn.Parameter(
            torch.zeros(1, 1, embed_num, hidden_len))  # [batch, agent, no_attack_action_num, hidden]
        nn.init.kaiming_normal_(self._action_embed, mode='fan_out')
        self._decision_encoder = MLP(hidden_len, hidden_len, 2 * hidden_len, 2, activation=nn.ReLU(inplace=True))
        self._logit_encoder = nn.Sequential(
            nn.Linear(2 * hidden_len, hidden_len),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_len, 1),
        )

    def forward(self, i, state, action_embed, alive_mask, action_mask, action):
        batch_size, agent_num, hidden_len, embed_num, device = state.shape[0], state.shape[1], \
                                                               state.shape[2], self._action_embed.shape[
                                                                   2], state.device
        agent_id = torch.LongTensor([i]).to(device)
        active_self_embed = self._action_embed.expand(batch_size, -1, -1, -1)
        passive_none_embed = torch.zeros(batch_size, agent_num, 1, hidden_len).to(device)
        passive_map = torch.cat(
            [torch.zeros(agent_num, embed_num).long().to(device),
             torch.diag(torch.ones(agent_num)).long().to(device)],
            dim=1).view(1, -1, embed_num + agent_num, 1).expand(batch_size, -1, -1,
                                                                hidden_len*2)  # [batch, agent, action_num, hidden]
        # get action embed
        active_embed, passive_embed = action_embed.chunk(2, dim=-1)  # [batch, agent, hidden], [batch, agent, hidden]
        active_embed_ = torch.cat([active_self_embed, active_embed.unsqueeze(1)],
                                  dim=2)  # [batch, 1, active_action, hidden] embed for active agent
        active_embed = active_embed_.scatter_add(2,
                                                 embed_num + agent_id.view(1, 1, 1, 1).expand(batch_size, 1, 1, hidden_len),
                                                 passive_embed.unsqueeze(1).index_select(2, agent_id))
        passive_embed = torch.cat([passive_none_embed, passive_embed.unsqueeze(2)],
                                  dim=2)  # [batch, agent, passive_action(2), hidden] embed for passive agent
        # get decision embed
        active_state = state.index_select(1, agent_id).unsqueeze(2) + active_embed
        passive_state = state.unsqueeze(2) + passive_embed
        active_decision = self._decision_encoder(active_state)    # [batch, 1, active_action, hidden]
        passive_decision = self._decision_encoder(passive_state)  # [batch, agent, passive_action(2), hidden]
        decision = passive_decision.gather(2, passive_map)        # [batch, agent, action_number, hidden]
        decision.scatter_(1, agent_id.view(1, 1, 1, 1).expand(batch_size, 1, embed_num + agent_num, 2 * hidden_len),
                          active_decision)
        # get logit
        decision_avr, decision_max = decision.chunk(2, dim=-1)  # [batch, agent, action_num, hidden]
        decision_avr = (decision_avr * alive_mask.unsqueeze(-1).unsqueeze(-1)).mean(1)  # [batch, action_num, hidden]
        decision_max = (decision_max * alive_mask.unsqueeze(-1).unsqueeze(-1)).max(1).values  # [batch, action_num, hidden]
        decision = torch.cat([decision_avr, decision_max], dim=-1)
        logit = self._logit_encoder(decision).squeeze(-1)  # [batch, action_num]
        # get action
        if isinstance(action, dict):
            action = ActionSampler(logit, action_mask, action)  # [batch]
        # get updated state
        if self.update_state:
            active_embed = active_embed_.gather(2, action.view(-1, 1, 1, 1).expand(-1, -1, -1, hidden_len))
            state = state.scatter_add(1, agent_id.view(1, 1, 1).expand(batch_size, 1, hidden_len), active_embed.squeeze(2))
            passive_map = passive_map.gather(2, action.view(-1, 1, 1, 1).expand(-1, agent_num, 1,
                                                                                hidden_len))  # [batch, agent, 1, hidden]
            passive_embed = passive_embed.gather(2, passive_map).squeeze(2)
            state = state + passive_embed
        decision = decision.gather(1, action.view(-1, 1, 1).expand(-1, -1, hidden_len*2))
        return state, decision, logit, action

class StateActionEncoder(nn.Module):
    """State-Action Encoder adapted from ACE SMAC model"""
    def __init__(
            self,
            agent_num: int,
            state_len: int,
            relation_len: int,
            hidden_len: int,
    ):
        super(StateActionEncoder, self).__init__()
        self.agent_num = agent_num
        self._action_encoder = MLP(2 * hidden_len + hidden_len + hidden_len, hidden_len, 2 * hidden_len, 2,
                                   activation=nn.ReLU(inplace=True))
        self._state_encoder = MLP(state_len, hidden_len, hidden_len, 2, activation=nn.ReLU(inplace=True))
        self._relation_encoder = MLP(hidden_len + relation_len, hidden_len, 2 * hidden_len, 2,
                                     activation=nn.ReLU(inplace=True))
        self._relation_aggregator = RelationAggregator(hidden_len, 2 * hidden_len)

    
    def forward(self, obs) -> dict:
        state = obs['states']  # [batch, entity_num, state_len]
        relation = obs['relations']  # [batch, entity_num, relation_len]
        alive_mask = obs['alive_mask']  # [batch, entity_num]
        state = self._state_encoder(state)
        relation = self._relation_encoder(
            torch.cat([relation, state.unsqueeze(1).expand(-1, relation.shape[1], -1, -1)], dim=-1))
        state = self._relation_aggregator(state, relation, alive_mask)
        action_embed = self._action_encoder(torch.cat(
            [relation, state.unsqueeze(1).expand(-1, relation.shape[1], -1, -1),
             state.unsqueeze(2).expand(-1, -1, relation.shape[1], -1)], dim=-1))
        
        return torch.cat([state, action_embed], dim=1)



    