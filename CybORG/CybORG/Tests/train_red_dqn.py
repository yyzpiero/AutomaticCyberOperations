import inspect
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime
import json
from typing import Union
from pprint import pprint
from CybORG import CybORG
from CybORG.Agents.Wrappers import *
from stable_baselines3 import DQN, PPO
from CybORG.Agents.SimpleAgents.B_line import B_lineAgent
from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent
from CybORG.Agents.Wrappers.TrueTableWrapper import true_obs_to_table
import torch
from CybORG.Agents.MyDQNAgent.agent import AgentBuild
from CybORG.Agents.MyDQNAgent.my_utils import get_device
from tensorboardX import SummaryWriter
import gym

path = str(inspect.getfile(CybORG))
path = path[:-10] + "/Shared/Scenarios/Scenario1b.yaml"

# action_space = env.action_space

DQN_HYPERPARAMS = {
    "c51": False,
    "dueling": True,
    "noisy_net": False,
    "double_DQN": True,
    "buffer_start_size": 1000,
    "buffer_capacity": 50000,
    "n_multi_step": 5,
    "epsilon_start": 0.95,
    "epsilon_decay": 200000,
    "epsilon_final": 0.05,
    "optimizer_type": "Adam",
    "learning_rate": 1e-4,
    "gamma": 0.95,
    "n_iter_update_target": 200,
    "batch_size": 256,
    "net_arch": [512, 512],
    "env_name": "CartPole-v1",
    "Vmin": -100,
    "Vmax": +100,
    "num_atoms": 51
    # "env_name": "nasim:Tiny-v0"
}

MAX_N_Iter = 500000
DEVICE = torch.device(get_device("auto"))
writer = None
TEST_PER_N_ITER = int(2000)
TRAIN_FREQ = 1
TEST_N_EPISODE = 5
BATCH_SIZE = int(DQN_HYPERPARAMS["batch_size"])
seed = 0
LOG_DIR = "./content/runs/C51+D3QN"
SUMMARY_WRITER = False
tb_log_dir = LOG_DIR + "/" + datetime.datetime.now().strftime("%m-%d-%H-%M")


def test_episode(env, agent, n_eval_episodes):
    all_episode_reward = []
    for _ in range(n_eval_episodes):
        obs = env.reset()
        episode_rewards = []
        done = False
        # rewards = 0
        while not done:

            action = agent.act(obs)
            next_obs, reward, done, _ = env.step(action)
            obs = next_obs
            # print("action:{},reward{}".format(action, reward))
            episode_rewards.append(reward)

        all_episode_reward.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_reward)
    mean_episode_reward_std = np.std(all_episode_reward)

    return mean_episode_reward, mean_episode_reward_std


if __name__ == "__main__":

    cyborg = CybORG(path, "sim")
    wrappers = EnumActionWrapper(FixedFlatWrapper(cyborg))
    env = OpenAIGymWrapper(env=wrappers, agent_name="Red")

    cyborg_test = CybORG(path, "sim")
    wrappers_test = EnumActionWrapper(FixedFlatWrapper(cyborg))
    env_test = OpenAIGymWrapper(env=wrappers, agent_name="Red")

    # env = gym.make("nasim:Small-v0")
    # env_test = gym.make("nasim:Small-v0")

    print("Ovservation Shape:{}".format(np.shape(env.observation_space)))

    if SUMMARY_WRITER:
        writer = SummaryWriter("%s/seed-%s" % (tb_log_dir, seed))
        with open(tb_log_dir + "/" + "HyperParas.json", mode="w") as fp:
            json.dump(DQN_HYPERPARAMS, fp)
    else:
        writer = None

    agent = AgentBuild(
        env, device=DEVICE, summary_writer=writer, hyperparameters=DQN_HYPERPARAMS
    )
    # agent = DQN("MlpPolicy", env)
    # agent.learn(total_timesteps=10000, log_interval=4)
    n_games = 0
    x_loop_must_break = False

    for n_iter in range(MAX_N_Iter):
        # print("---trainning at {}---".format(agent.n_iter))
        obs = env.reset()
        done = False

        while not done:
            if agent.n_iter >= MAX_N_Iter:
                x_loop_must_break = True
                break

            action = agent.e_greedy_act(obs)
            # action, _state = agent.predict(obs)
            # print(action)
            new_obs, reward, done, _ = env.step(action)
            agent.add_env_feedback(obs, action, new_obs, reward, done)
            obs = new_obs

            if agent.n_iter % TRAIN_FREQ == 0:
                agent.train(BATCH_SIZE)

            if agent.n_iter % TEST_PER_N_ITER == 0:
                test_mean_reward, test_mean_reward_std = test_episode(
                    env_test, agent, TEST_N_EPISODE
                )

                agent.print_info(verbose=0)
                print(
                    "Test: reward_per_episode_mean:{:.2f} at episode:{} at itertaion:{}".format(
                        test_mean_reward, agent.n_games + 1, agent.n_iter
                    )
                )
                if writer is not None:
                    writer.add_scalar(
                        "test_total_reward", test_mean_reward, agent.n_games
                    )
                    writer.add_scalar(
                        "test_total_reward_iter", test_mean_reward, agent.n_iter
                    )
                    writer.add_scalar(
                        "test_total_reward_iter_std", test_mean_reward_std, agent.n_iter
                    )
                    writer.add_scalars(
                        "rewards", {"test_total_reward": test_mean_reward}, agent.n_iter
                    )
        if x_loop_must_break == True:
            break

        agent.reset_stats()
