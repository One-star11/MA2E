import torch as th
import numpy as np
from types import SimpleNamespace as SN
from .segment_tree import SumSegmentTree, MinSegmentTree
import random
from modules.layer.cross_atten import CrossAttention
import jax
import jax.numpy as jnp
import flashbax as fbx
from flashbax.vault import Vault
class EpisodeBatch:
    def __init__(self,
                 scheme,
                 groups,
                 batch_size,
                 max_seq_length,
                 data=None,
                 preprocess=None,
                 device="cpu"):
        self.scheme = scheme.copy()
        self.groups = groups
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.preprocess = {} if preprocess is None else preprocess
        self.device = device

        if data is not None:
            self.data = data
        else:
            self.data = SN()
            self.data.transition_data = {}
            self.data.episode_data = {}
            self._setup_data(self.scheme, self.groups, batch_size, max_seq_length, self.preprocess)

    def _setup_data(self, scheme, groups, batch_size, max_seq_length, preprocess):
        if preprocess is not None:
            for k in preprocess:
                assert k in scheme
                new_k = preprocess[k][0]
                transforms = preprocess[k][1]

                vshape = self.scheme[k]["vshape"]
                dtype = self.scheme[k]["dtype"]
                for transform in transforms:
                    vshape, dtype = transform.infer_output_info(vshape, dtype)

                self.scheme[new_k] = {
                    "vshape": vshape,
                    "dtype": dtype
                }
                if "group" in self.scheme[k]:
                    self.scheme[new_k]["group"] = self.scheme[k]["group"]
                if "episode_const" in self.scheme[k]:
                    self.scheme[new_k]["episode_const"] = self.scheme[k]["episode_const"]

        assert "filled" not in scheme, '"filled" is a reserved key for masking.'
        scheme.update({
            "filled": {"vshape": (1,), "dtype": th.long},
        })

        for field_key, field_info in scheme.items():
            assert "vshape" in field_info, "Scheme must define vshape for {}".format(field_key)
            vshape = field_info["vshape"]
            episode_const = field_info.get("episode_const", False)
            group = field_info.get("group", None)
            dtype = field_info.get("dtype", th.float32)

            if isinstance(vshape, int):
                vshape = (vshape,)

            if group:
                assert group in groups, "Group {} must have its number of members defined in _groups_".format(group)
                shape = (groups[group], *vshape)
            else:
                shape = vshape

            if episode_const:
                self.data.episode_data[field_key] = th.zeros((batch_size, *shape), dtype=dtype, device=self.device)
            else:
                self.data.transition_data[field_key] = th.zeros((batch_size, max_seq_length, *shape), dtype=dtype, device=self.device)

    def extend(self, scheme, groups=None):
        self._setup_data(scheme, self.groups if groups is None else groups, self.batch_size, self.max_seq_length)

    def to(self, device):
        for k, v in self.data.transition_data.items():
            self.data.transition_data[k] = v.to(device)
        for k, v in self.data.episode_data.items():
            self.data.episode_data[k] = v.to(device)
        self.device = device

    def update(self, data, bs=slice(None), ts=slice(None), mark_filled=True):
        slices = self._parse_slices((bs, ts))
        for k, v in data.items():

            if k in self.data.transition_data:
                target = self.data.transition_data
                if mark_filled:
                    target["filled"][slices] = 1
                    mark_filled = False
                _slices = slices
            elif k in self.data.episode_data:
                target = self.data.episode_data
                _slices = slices[0]
            else:
                raise KeyError("{} not found in transition or episode data".format(k))

            dtype = self.scheme[k].get("dtype", th.float32)
            v = th.tensor(v, dtype=dtype, device=self.device)
            self._check_safe_view(v, target[k][_slices])
            target[k][_slices] = v.view_as(target[k][_slices])

            if k in self.preprocess:
                new_k = self.preprocess[k][0]
                v = target[k][_slices]
                for transform in self.preprocess[k][1]:
                    v = transform.transform(v)
                target[new_k][_slices] = v.view_as(target[new_k][_slices])
    def _check_safe_view(self, v, dest):
        idx = len(v.shape) - 1
        for s in dest.shape[::-1]:
            if v.shape[idx] != s:
                if s != 1:
                    raise ValueError("Unsafe reshape of {} to {}".format(v.shape, dest.shape))
            else:
                idx -= 1

    def __getitem__(self, item):
        if isinstance(item, str):
            if item in self.data.episode_data:
                return self.data.episode_data[item]
            elif item in self.data.transition_data:
                return self.data.transition_data[item]
            else:
                raise ValueError
        elif isinstance(item, tuple) and all([isinstance(it, str) for it in item]):
            new_data = self._new_data_sn()
            for key in item:
                if key in self.data.transition_data:
                    new_data.transition_data[key] = self.data.transition_data[key]
                elif key in self.data.episode_data:
                    new_data.episode_data[key] = self.data.episode_data[key]
                else:
                    raise KeyError("Unrecognised key {}".format(key))

            # Update the scheme to only have the requested keys
            new_scheme = {key: self.scheme[key] for key in item}
            new_groups = {self.scheme[key]["group"]: self.groups[self.scheme[key]["group"]]
                          for key in item if "group" in self.scheme[key]}
            ret = EpisodeBatch(new_scheme, new_groups, self.batch_size, self.max_seq_length, data=new_data, device=self.device)
            return ret
        else:
            item = self._parse_slices(item)
            new_data = self._new_data_sn()
            for k, v in self.data.transition_data.items():
                new_data.transition_data[k] = v[item]
            for k, v in self.data.episode_data.items():
                new_data.episode_data[k] = v[item[0]]

            ret_bs = self._get_num_items(item[0], self.batch_size)
            ret_max_t = self._get_num_items(item[1], self.max_seq_length)

            ret = EpisodeBatch(self.scheme, self.groups, ret_bs, ret_max_t, data=new_data, device=self.device)
            return ret

    def _get_num_items(self, indexing_item, max_size):
        if isinstance(indexing_item, list) or isinstance(indexing_item, np.ndarray):
            return len(indexing_item)
        elif isinstance(indexing_item, slice):
            _range = indexing_item.indices(max_size)
            return 1 + (_range[1] - _range[0] - 1)//_range[2]

    def _new_data_sn(self):
        new_data = SN()
        new_data.transition_data = {}
        new_data.episode_data = {}
        return new_data

    def _parse_slices(self, items):
        parsed = []
        # Only batch slice given, add full time slice
        if (isinstance(items, slice)  # slice a:b
            or isinstance(items, int)  # int i
            or (isinstance(items, (list, np.ndarray, th.LongTensor, th.cuda.LongTensor)))  # [a,b,c]
            ):
            items = (items, slice(None))

        # Need the time indexing to be contiguous
        if isinstance(items[1], list):
            raise IndexError("Indexing across Time must be contiguous")

        for item in items:
            #TODO: stronger checks to ensure only supported options get through
            if isinstance(item, int):
                # Convert single indices to slices
                parsed.append(slice(item, item+1))
            else:
                # Leave slices and lists as is
                parsed.append(item)
        return parsed

    def max_t_filled(self):
        return th.sum(self.data.transition_data["filled"], 1).max(0)[0]

    def __repr__(self):
        return "EpisodeBatch. Batch Size:{} Max_seq_len:{} Keys:{} Groups:{}".format(self.batch_size,
                                                                                     self.max_seq_length,
                                                                                     self.scheme.keys(),
                                                                                     self.groups.keys())


class ReplayBuffer(EpisodeBatch):
    def __init__(self, scheme, groups, buffer_size, max_seq_length, preprocess=None, device="cpu", args=None, env_info=None):
        super(ReplayBuffer, self).__init__(scheme, groups, buffer_size, max_seq_length, preprocess=preprocess, device=device)
        self.buffer_size = buffer_size  # same as self.batch_size but more explicit
        self.buffer_index = 0
        self.episodes_in_buffer = 0
        self.heads = 1
        self.args = args  # Store args for agent ID encoding
        self.env_info = env_info  # Store env_info
        self.n_agents = groups["agents"]  # Number of agents
        self.n_actions = scheme["avail_actions"]["vshape"][0]  # Number of possible actions
        self.device = device
        self.max_ep_len = 71

        # Load data from all three vaults
        self.vlt_good = Vault(rel_dir = "vaults", vault_name = "5m_vs_6m.vlt", vault_uid = "Good")
        self.vlt_medium = Vault(rel_dir = "vaults", vault_name = "5m_vs_6m.vlt", vault_uid = "Medium")
        self.vlt_poor = Vault(rel_dir = "vaults", vault_name = "5m_vs_6m.vlt", vault_uid = "Poor")
        
        # Store episode metadata and prepare transformer features
        self.offline_features = []  # List to store episode features
        self.offline_values = []    # List to store episode values (rewards)
        self.episode_metadata = []  # List to store episode metadata for later retrieval

        # Store experiences in memory to avoid repeated reads
        self.experiences = {
            "good": self.vlt_good.read().experience,
            "medium": self.vlt_medium.read().experience,
            "poor": self.vlt_poor.read().experience
        }
        
        def process_vault_data(vault, vault_type):
            print(f"Processing {vault_type} vault...")
            experience = self.experiences[vault_type]
            
            # Convert only essential data to torch tensors directly
            terminals = th.from_numpy(np.array(experience["terminals"][0]))
            truncations = th.from_numpy(np.array(experience["truncations"][0]))
            states = th.from_numpy(np.array(experience["infos"]["state"][0]))
            actions = th.from_numpy(np.array(experience["actions"][0]))
            rewards = th.from_numpy(np.array(experience["rewards"][0]))
            
            # Find episode boundaries
            episode_ends = (terminals > 0.5) | (truncations > 0.5)
            end_indices = th.unique(th.where(episode_ends)[0])
            
            # Process each episode
            episode_start = 0
            for ep_idx, episode_end in enumerate(end_indices):
                episode_length = episode_end - episode_start + 1
                
                # Store episode metadata for later retrieval
                self.episode_metadata.append({
                    'vault_type': vault_type,
                    'vault': vault,
                    'start_idx': episode_start,
                    'end_idx': episode_end,
                    'length': episode_length
                })
                
                # Get episode data
                episode_states = states[episode_start:episode_end+1]
                episode_actions = actions[episode_start:episode_end+1]
                episode_rewards = rewards[episode_start:episode_end+1, 0:1]
                
                # Create feature vector for transformer
                states_actions = th.cat([
                    episode_states.reshape(episode_length, -1),
                    episode_actions.reshape(episode_length, -1)
                ], dim=-1)
                
                # Pad if necessary
                if episode_length < self.max_ep_len:
                    pad_length = self.max_ep_len - episode_length
                    # Pad in the timestep dimension (first dimension)
                    states_actions = th.cat([
                        states_actions,
                        th.zeros(pad_length, states_actions.size(-1), dtype=states_actions.dtype)
                    ], dim=0)
                    episode_rewards = th.cat([
                        episode_rewards,
                        th.zeros(pad_length, episode_rewards.size(-1), dtype=episode_rewards.dtype)
                    ], dim=0)
                
                # Store features and values for transformer
                self.offline_features.append(states_actions.flatten())
                self.offline_values.append(episode_rewards.flatten())
                
                episode_start = episode_end + 1
            
            print(f"Processed {len(end_indices)} episodes from {vault_type} vault")
        
        # Process all vaults
        process_vault_data("good", "good")
        process_vault_data("medium", "medium")
        process_vault_data("poor", "poor")

        
        # Stack features and values
        self.offline_features = th.stack(self.offline_features)
        self.offline_values = th.stack(self.offline_values)
        
        print(f"offline_features shape: {self.offline_features.shape}")
        print(f"offline_values shape: {self.offline_values.shape}")
        
        # Create the transformer with the prepared data
        self.transformer = CrossAttention(
            self.offline_features.shape[1],
            self.heads,
            self.offline_features.shape[1],
            offline_keys=self.offline_features,
            offline_values=self.offline_values
        )
        if device != "cpu":
            self.transformer = self.transformer.to(device)

    def _load_full_episode(self, episode_meta):
        """Load full episode data from vault when needed."""
        vault = episode_meta['vault']
        start_idx = episode_meta['start_idx']
        end_idx = episode_meta['end_idx']
        episode_length = episode_meta['length']
        
        # Read vault data for this episode
        experience = self.experiences[episode_meta['vault_type']]
        
        # Initialize obs_all and action_all for this episode
        obs_all = th.zeros((self.max_ep_len, self.n_agents, self.args.MT_traj_length, self.args.input_shape), dtype=th.float32)
        action_all = th.zeros((self.max_ep_len, self.n_agents, self.args.MT_traj_length, 1), dtype=th.float32)
        
        # Get episode data directly as torch tensors
        obs = th.from_numpy(np.array(experience["observations"][0]))[start_idx:end_idx+1]
        actions = th.from_numpy(np.array(experience["actions"][0]))[start_idx:end_idx+1]
        
        # Set agent IDs in obs_all if needed
        if self.args.obs_agent_id:
            if self.args.obs_last_action:
                # Agent ID after obs and last action
                for idx in range(self.n_agents):
                    obs_all[:, idx, :, self.env_info["obs_shape"] + self.env_info["n_actions"] + idx] = 1
            else:
                # Agent ID after obs
                for idx in range(self.n_agents):
                    obs_all[:, idx, :, self.env_info["obs_shape"] + idx] = 1
        
        # Fill in the observations and actions for each timestep
        for t in range(episode_length):
            # For each timestep, shift the window and add new data
            if t > 0:
                obs_all[t, :, :self.args.MT_traj_length-1] = obs_all[t-1, :, 1:]
                action_all[t, :, :self.args.MT_traj_length-1] = action_all[t-1, :, 1:]
            
            # Add new observation
            obs_all[t, :, -1, :self.env_info["obs_shape"]] = obs[t]
            
            # Add last action to observation if needed
            if self.args.obs_last_action and t > 0:
                for agent_id in range(self.n_agents):
                    obs_all[t, agent_id, -1, self.env_info["obs_shape"] + int(actions[t - 1, agent_id])] = 1
            
            # Add new action
            if t < episode_length:
                action_all[t, :, -1, 0] = actions[t].reshape(-1)
        
        # Create episode dictionary with full data
        episode = {
            'state': th.from_numpy(np.array(experience["infos"]["state"][0]))[start_idx:end_idx+1],
            'obs': obs,
            'actions': actions,
            'avail_actions': th.from_numpy(np.array(experience["infos"]["legals"][0]))[start_idx:end_idx+1],
            'reward': th.from_numpy(np.array(experience["rewards"][0]))[start_idx:end_idx+1,0:1],
            'terminated': th.from_numpy(np.array(experience["terminals"][0]))[start_idx:end_idx+1,0:1],
            'length': episode_length,
            'vault_type': episode_meta['vault_type'],
            'probs': th.zeros((episode_length, self.n_agents, self.env_info["n_actions"]), dtype=th.float),
            'obs_all': obs_all,
            'action_all': action_all
        }
        
        # Pad episode to max_ep_len if needed
        if episode_length < self.max_ep_len:
            padding_len = self.max_ep_len - episode_length
            for key in ['state', 'obs', 'actions', 'avail_actions', 'reward', 'terminated', 'probs']:
                if key in episode:
                    shape = list(episode[key].shape)
                    shape[0] = padding_len
                    padding = th.zeros(shape, dtype=episode[key].dtype)
                    episode[key] = th.cat([episode[key], padding], dim=0)
        
        return episode

    def _get_episode_attention(self, states, actions, max_time):
        """Get attention scores for a full episode."""
        batch_size = states.shape[0] if len(states.shape) > 2 else 1
        if len(states.shape) == 2:
            states = states.unsqueeze(0)
            actions = actions.unsqueeze(0)
            
        # Create feature vector for transformer exactly as in __init__
        states_actions = th.cat([
            states.reshape(batch_size, max_time, -1),
            actions.reshape(batch_size, max_time, -1)
        ], dim=-1)
        
        # Flatten for transformer input
        states_actions = states_actions.reshape(batch_size, -1)
        
        # Move to device
        states_actions = states_actions.to(self.device)
        
        # Get attention scores - returns (batch_size, n_episodes)
        return self.transformer(states_actions)  # Keep (batch, n_episodes) shape for top-k selection

    def _create_expanded_batch(self, ep_batch, k):
        """Create a new batch with space for original and retrieved episodes.
        
        Args:
            ep_batch: Original episode batch
            k: Number of similar episodes to retrieve per original episode
            
        The expanded batch will have the following structure:
        - For each original episode, we allocate k+1 spaces:
            - 1 for the original episode
            - k for the similar episodes
        - Original episodes are placed at indices: 0, k+1, 2(k+1), 3(k+1), ...
        - Similar episodes will be placed in between
        """
        expanded_batch = {
            'transition_data': {},
            'episode_data': {}
        }
        
        # Initialize expanded batch with same structure as original
        for key in ep_batch.data.transition_data.keys():
            if key != 'filled':
                shape = list(ep_batch.data.transition_data[key].shape)
                shape[0] *= (k + 1)  # Multiply batch dimension by k+1
                expanded_batch['transition_data'][key] = th.zeros(shape, 
                dtype=ep_batch.data.transition_data[key].dtype,
                device=ep_batch.data.transition_data[key].device)
        
        for key in ep_batch.data.episode_data.keys():
            shape = list(ep_batch.data.episode_data[key].shape)
            shape[0] *= (k + 1)  # Multiply batch dimension by k+1
            expanded_batch['episode_data'][key] = th.zeros(shape,
                dtype=ep_batch.data.episode_data[key].dtype,
                device=ep_batch.data.episode_data[key].device)
        
        # Place original episodes at indices 0, k+1, 2(k+1), ...
        stride = k + 1  # Distance between original episodes in expanded batch
        for key in ep_batch.data.transition_data.keys():
            if key != 'filled':
                expanded_batch['transition_data'][key][::stride] = ep_batch.data.transition_data[key]
        
        for key in ep_batch.data.episode_data.keys():
            expanded_batch['episode_data'][key][::stride] = ep_batch.data.episode_data[key]
            
        return expanded_batch

    def _retrieve_similar_episodes(self, ep_batch, k=3):
        """Retrieve k most similar episodes for each episode in batch."""
        # Create expanded batch
        # print("ep_batch.batch_size", ep_batch.batch_size)
        expanded_batch = self._create_expanded_batch(ep_batch, k)
        
        # Get state and action data from the episode batch
        states = ep_batch["state"]   # shape: (batch, time, state_dim)
        actions = ep_batch["actions"]  # shape: (batch, time, action_dim)
        max_time = states.shape[1]
        
        # Compute attention scores for the whole batch at once
        attention_scores = self._get_episode_attention(states, actions, max_time)  # shape: (batch, n_episodes)
        _, top_k_indices = th.topk(attention_scores, k=k)  # shape: (batch, k)

        # Create a new scheme without 'filled' key for the new batch
        new_scheme = {k: v for k, v in ep_batch.scheme.items() if k != 'filled'}

        # Prepare raw data dicts for update
        transition_data = {}
        episode_data = {}
        
        # Initialize transition_data with zeros for all keys except 'filled'
        for key in ep_batch.data.transition_data.keys():
            if key != 'filled':
                shape = list(expanded_batch['transition_data'][key].shape)
                transition_data[key] = th.zeros(shape, 
                    dtype=ep_batch.data.transition_data[key].dtype,
                    device=ep_batch.data.transition_data[key].device)
        
        # Initialize episode_data
        for key in ep_batch.data.episode_data.keys():
            shape = list(expanded_batch['episode_data'][key].shape)
            episode_data[key] = th.zeros(shape,
                dtype=ep_batch.data.episode_data[key].dtype,
                device=ep_batch.data.episode_data[key].device)

        # Copy original episodes to their positions
        stride = k + 1
        for key in transition_data.keys():
            transition_data[key][::stride] = ep_batch.data.transition_data[key]
        for key in episode_data.keys():
            episode_data[key][::stride] = ep_batch.data.episode_data[key]

        # Group episodes by vault type to minimize vault reads
        vault_episodes = {}
        for b in range(ep_batch.batch_size):
            for i, top_idx in enumerate(top_k_indices[b]):
                episode_meta = self.episode_metadata[top_idx.item()]
                vault_type = episode_meta['vault_type']
                if vault_type not in vault_episodes:
                    vault_episodes[vault_type] = []
                vault_episodes[vault_type].append((b, i, top_idx.item()))

        # Load episodes by vault type
        for vault_type, episodes in vault_episodes.items():
            # Get all unique episode indices for this vault
            unique_indices = list(set(idx for _, _, idx in episodes))
            
            # Load all episodes for this vault at once
            loaded_episodes = {}
            for idx in unique_indices:
                loaded_episodes[idx] = self._load_full_episode(self.episode_metadata[idx])
            
            # Update the expanded batch with loaded episodes
            for b, i, idx in episodes:
                expanded_idx = b * (k + 1) + i + 1
                episode = loaded_episodes[idx]
                print(f"Episode {b}, Retrieved episode {idx} from {episode['vault_type']} vault")
                
                # Update transition data
                for key in transition_data.keys():
                    if key in episode:
                        transition_data[key][expanded_idx] = episode[key].view(*transition_data[key][expanded_idx].shape)

        # Create new EpisodeBatch with the modified scheme
        new_ep_batch = EpisodeBatch(
            scheme=new_scheme,
            groups=ep_batch.groups,
            batch_size=ep_batch.batch_size * (k + 1),
            max_seq_length=ep_batch.max_seq_length,
            device=ep_batch.device,
            preprocess=ep_batch.preprocess
        )

        # Update the new batch with our prepared data
        new_ep_batch.update(transition_data)
        new_ep_batch.update(episode_data)
        return new_ep_batch

    def insert_episode_batch(self, ep_batch, recursive=False):
        """Insert episode batch into buffer with similar episode retrieval."""
        if not recursive:
            # Retrieve similar episodes and expand batch
            ep_batch = self._retrieve_similar_episodes(ep_batch, k=3)
        
        # Continue with original episode insertion logic
        if self.buffer_index + ep_batch.batch_size <= self.buffer_size:
            self.update(ep_batch.data.transition_data,
                        slice(self.buffer_index, self.buffer_index + ep_batch.batch_size),
                        slice(0, ep_batch.max_seq_length),
                        mark_filled=False)
            self.update(ep_batch.data.episode_data,
                        slice(self.buffer_index, self.buffer_index + ep_batch.batch_size))
            self.buffer_index = (self.buffer_index + ep_batch.batch_size)
            self.episodes_in_buffer = max(self.episodes_in_buffer, self.buffer_index)
            self.buffer_index = self.buffer_index % self.buffer_size
            assert self.buffer_index < self.buffer_size
        else:
            buffer_left = self.buffer_size - self.buffer_index
            self.insert_episode_batch(ep_batch[0:buffer_left, :], recursive=True)
            self.insert_episode_batch(ep_batch[buffer_left:, :], recursive=True)

    def can_sample(self, batch_size):
        return self.episodes_in_buffer >= batch_size

    def sample(self, batch_size):
        assert self.can_sample(batch_size)
        if self.episodes_in_buffer == batch_size:
            return self[:batch_size]
        else:
            # Uniform sampling only atm
            ep_ids = np.random.choice(self.episodes_in_buffer, batch_size, replace=False)
            return self[ep_ids]

    def uni_sample(self, batch_size):
        return self.sample(batch_size)

    def sample_latest(self, batch_size):
        assert self.can_sample(batch_size)
        if self.buffer_index - batch_size < 0:
            #Uniform sampling
            return self.uni_sample(batch_size)
        else:
            # Return the latest
            return self[self.buffer_index - batch_size : self.buffer_index]

    def __repr__(self):
        return "ReplayBuffer. {}/{} episodes. Keys:{} Groups:{}".format(self.episodes_in_buffer,
                                                                        self.buffer_size,
                                                                        self.scheme.keys(),
                                                                        self.groups.keys())


# Adapted from the OpenAI Baseline implementations (https://github.com/openai/baselines)
class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, scheme, groups, buffer_size, max_seq_length, alpha, beta, t_max, preprocess=None, device="cpu"):
        super(PrioritizedReplayBuffer, self).__init__(scheme, groups, buffer_size, max_seq_length,
                                                      preprocess=preprocess, device="cpu")
        self.alpha = alpha
        self.beta_original = beta
        self.beta = beta
        self.beta_increment = (1.0 - beta) / t_max
        self.max_priority = 1.0

        it_capacity = 1
        while it_capacity < buffer_size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)

    def insert_episode_batch(self, ep_batch):
        # TODO: convert batch/episode to idx?
        pre_idx = self.buffer_index
        super().insert_episode_batch(ep_batch)
        idx = self.buffer_index
        if idx >= pre_idx:
            for i in range(idx - pre_idx):
                self._it_sum[pre_idx + i] = self.max_priority ** self.alpha
                self._it_min[pre_idx + i] = self.max_priority ** self.alpha
        else:
            for i in range(self.buffer_size - pre_idx):
                self._it_sum[pre_idx + i] = self.max_priority ** self.alpha
                self._it_min[pre_idx + i] = self.max_priority ** self.alpha
            for i in range(self.buffer_index):
                self._it_sum[i] = self.max_priority ** self.alpha
                self._it_min[i] = self.max_priority ** self.alpha

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, self.episodes_in_buffer - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, t):
        assert self.can_sample(batch_size)
        self.beta = self.beta_original + (t * self.beta_increment)

        idxes = self._sample_proportional(batch_size)
        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * self.episodes_in_buffer) ** (-self.beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * self.episodes_in_buffer) ** (-self.beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)

        return self[idxes], idxes, weights

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < self.episodes_in_buffer
            self._it_sum[idx] = priority ** self.alpha
            self._it_min[idx] = priority ** self.alpha
            self.max_priority = max(self.max_priority, priority)