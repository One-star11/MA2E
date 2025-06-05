import torch
import torch.nn as nn
import torch.nn.functional as F
from .ace_utils import StateActionEncoder
# Importing the MLP and RelationAggregator from ACE

#should be able to handle episode data. 
#episode data is (total_episode_size,timestep, feature_dim). where feature_dim = (state_dim + action_dim)


class CrossAttention(nn.Module):
    def __init__(self, input_size, heads, embed_size, offline_keys=None, offline_values=None,
                 state_len=None, relation_len=None, action_dim=None, use_ace_encoder=True):
        super().__init__()
        self.input_size = input_size
        self.heads = heads
        self.emb_size = embed_size
        self.use_ace_encoder = use_ace_encoder
        
        # Initialize ACE encoder if requested
        if use_ace_encoder and state_len is not None and relation_len is not None:
            self.state_action_encoder = StateActionEncoder(
                state_len=state_len,
                relation_len=relation_len,
                hidden_len=embed_size,
                action_dim=action_dim
            )
            # Adjust input size for encoded features
            actual_input_size = embed_size * 2 if action_dim is not None else embed_size
        else:
            self.state_action_encoder = None
            actual_input_size = input_size

        self.toqueries = nn.Linear(actual_input_size, self.emb_size * heads, bias=False)
        
        # offline parameters
        if offline_keys is not None and offline_values is not None:
            # offline_keys: (#episodes, state_action_dim) or raw obs format
            # offline_values: (#episodes, reward_steps)
            if isinstance(offline_keys, dict):
                # Process offline keys through encoder if they're in raw format
                if self.state_action_encoder is not None:
                    with torch.no_grad():
                        processed_keys = []
                        for episode_obs in offline_keys:
                            encoded = self.state_action_encoder(episode_obs)
                            # Take mean across entities if multi-agent
                            if encoded.dim() == 3:
                                encoded = encoded.mean(dim=1)
                            processed_keys.append(encoded)
                        offline_keys = torch.stack(processed_keys)
            
            n_episodes = offline_keys.size(0)
            self.n_episodes = n_episodes
            
            # Pre-expand keys for heads: (heads, n_episodes, embed_size)
            self.register_buffer('keys', 
                offline_keys.unsqueeze(0).repeat(heads, 1, 1)
            )
            # Convert step-wise rewards to episode-wise absolute mean rewards: (n_episodes,)
            self.register_buffer('values',
                offline_values.abs().mean(dim=-1)  # Take absolute mean across time steps
            )
        else:
            raise ValueError("Offline keys and values must be provided")

    def forward(self, x, curiosity_score=None):
        # Handle raw observation input
        if isinstance(x, dict) and self.state_action_encoder is not None:
            # x is raw observation dict
            x = self.state_action_encoder(x)
            # Take mean across entities if multi-agent output
            if x.dim() == 3:
                x = x.mean(dim=1)
        
        b, hin = x.size()    # (batch_size, input_size)
        
        h = self.heads
        e = self.emb_size
        
        # Transform to queries: (b, h, e)
        queries = self.toqueries(x).view(b, h, e)
        queries = queries / (e ** (1/4))

        # Compute attention scores for each batch and head
        # queries: (b, h, e), keys: (h, n_episodes, e)
        # -> attention: (b, h, n_episodes)
        dot = torch.matmul(queries, self.keys.transpose(-2, -1))
        
        # Add episode values to attention scores: (b, h, n_episodes)
        dot = dot + self.values.view(1, 1, -1)
        
        # Apply softmax over episodes dimension: (b, h, n_episodes)
        dot = F.softmax(dot, dim=-1)
        
        if curiosity_score is not None:
            # curiosity_score: (b, score) -> (b, 1, 1)
            curiosity = torch.exp(curiosity_score).view(b, 1, 1)
            dot = dot * curiosity
            dot = F.softmax(dot, dim=-1)  # Renormalize
        
        # Average attention scores across heads: (b, n_episodes)
        out = dot.mean(dim=1)
        
        return out