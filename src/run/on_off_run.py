import datetime
import os
import pprint
import time
import math as mth
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath
import numpy as np

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot


def run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(dirname(abspath(__file__)))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)
    
    group_name = ''
    if args.use_wandb : 
        if args.use_MT_mode : 
            #group_name = '{}_MT_{}_{}'.format(args.wandb_group_info, args.env_args['map_name'], args.name)
            if args.MT_finetune :
                group_name = '{}_MT_{}_{}_{}_{}_{}'.format(args.wandb_group_info, args.env_args['map_name'], args.name,args.masking_type,args.MT_traj_length,args.positional_encoding_target)
            else : 
                group_name = '{}_MT_without_FT_{}_{}_{}_{}_{}'.format(args.wandb_group_info, args.env_args['map_name'], args.name,args.masking_type,args.MT_traj_length,args.positional_encoding_target)
            #args.rnn_hidden_dim = args.rnn_hidden_dim * 2
        else :
            group_name = '{}_{}_{}'.format(args.wandb_group_info, args.env_args['map_name'], args.name)
        scenario_name = '{}_{}'.format(args.env_args['map_name'],unique_token)
        project_name = args.project_name
        logger.setup_wandb(project_name,group_name,scenario_name)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger,group_name=group_name)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def evaluate_sequential(args, runner):

    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()

def get_input_shape(args) :
    input_shape = args.obs_shape
    if args.obs_last_action:
        input_shape += args.n_actions
    if args.obs_agent_id:
        input_shape += args.n_agents

    return input_shape

def run_sequential(args, logger,group_name):

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    # args.unit_type_bits = env_info["unit_type_bits"]
    # args.shield_bits_ally = env_info["shield_bits_ally"]
    # args.shield_bits_enemy = env_info["shield_bits_enemy"]
    # args.n_enemies = env_info["n_enemies"]
    args.obs_shape = env_info["obs_shape"]
    args.accumulated_episodes = getattr(args, "accumulated_episodes", None)
    args.input_shape = get_input_shape(args)

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
        "action_all" : {"vshape" :(args.MT_traj_length,1),"dtype":th.long,"group":"agents"},
        "obs_all" : {"vshape" :(args.MT_traj_length,args.input_shape),"group":"agents"},
        #"policy": {"vshape": (env_info["n_agents"],)}
    }
    
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    on_buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu",
                          args=args)
    off_buffer = ReplayBuffer(scheme, groups, args.off_buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu",
                          args=args)
    buffer_for_MT = ReplayBuffer(scheme, groups, args.batch_size_run, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)


    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](on_buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, on_buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            evaluate_sequential(args, runner)
            return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))
    
    ##################################################################
    ######################### MT PRETRAINING #########################
    ##################################################################
    
    if args.use_MT_mode :
        cnt = 0
        cnt2= 0 
        n_repeat = args.MT_train_n_repeat
        avg_mae_loss = list()
        while cnt <= int(args.MT_max_pretraining_episode/args.batch_size_run) :
            
            with th.no_grad() :
                episode_batch = runner.run(test_mode=False,MT_train_mode=True)
                on_buffer.insert_episode_batch(episode_batch)
                del episode_batch
                
            if on_buffer.can_sample(args.MT_batchsize) :
                episode_sample = on_buffer.sample(args.MT_batchsize)

                # Truncate batch to only filled timesteps
                max_ep_t = episode_sample.max_t_filled()
                episode_sample = episode_sample[:, :max_ep_t]

                if episode_sample.device != args.device:
                    episode_sample.to(args.device)

                avg_loss = learner.MT_train(episode_sample,runner.t_env,episode,logger=logger,batch_size=args.MT_batchsize,n_repeat = n_repeat,write_log=True)
                del episode_sample
                
                if len(avg_mae_loss) < 100 :
                    avg_mae_loss.append(avg_loss)
                else :
                    avg_mae_loss[0:99] = avg_mae_loss[1:100]
                    avg_mae_loss[99] = avg_loss
                print('MAE_loss : {}'.format(np.mean(avg_mae_loss).item()))
                if cnt2 % 100 == 0 and args.save_MT :
                    cnt2 += 1
                    filename = 'results/params/{}_Pretraining_{}.pt'.format(group_name,cnt)
                    learner.save_MT(filename=filename)
            cnt += 1

        
        del on_buffer
        on_buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu",
                          args=args)
        
        
        n_repeat = 2
    
    fine_tune_cnt = 0
    fine_tune_step = 0 
    
    
    
    while runner.t_env <= args.t_max:

        # Run for a whole episode at a time
        episode_batch = runner.run(test_mode=False)
        on_buffer.insert_episode_batch(episode_batch)
        off_buffer.insert_episode_batch(episode_batch)
        buffer_for_MT.insert_episode_batch(episode_batch)

        if off_buffer.can_sample(args.buffer_size):
            # off samples
            episode_sample = off_buffer.uni_sample(args.off_batch_size)
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = process_batch(episode_sample[:, :max_ep_t], args)
            learner.train(episode_sample, runner.t_env, episode, off=True)
            del episode_sample

        if on_buffer.can_sample(args.buffer_size):
            # on samples
            episode_sample = on_buffer.sample_latest(args.batch_size)
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = process_batch(episode_sample[:, :max_ep_t], args)
            learner.train(episode_sample, runner.t_env, episode, off=False)
            del episode_sample
            
            
        #########################################       
        ############ MT FINETUNING ############## 
        
        if args.use_MT_mode and fine_tune_cnt <= args.MT_fine_tune_total and fine_tune_step <= runner.t_env and args.MT_finetune :
            fine_tune_step += args.MT_fine_tune_per_step
            fine_tune_cnt += 1
            if args.MT_fine_tune_total*0.5 <= fine_tune_cnt :
                n_repeat = 1
                
            if buffer_for_MT.can_sample(args.batch_size_run) :
                episode_sample_for_MT = buffer_for_MT.sample(args.batch_size_run)

                max_ep_t  = episode_sample_for_MT.max_t_filled()
                episode_sample_for_MT = episode_sample_for_MT[:,:max_ep_t]

                if episode_sample_for_MT.device != args.device :
                    episode_sample_for_MT.to(args.device)

                avg_loss = learner.MT_train(episode_sample_for_MT,runner.t_env,episode,logger=logger,batch_size=args.batch_size_run,n_repeat=2,write_log=True)
                del episode_sample_for_MT


        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
            #"results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config


def process_batch(batch, args):

    if batch.device != args.device:
        batch.to(args.device)
    return batch


