import torch
from rlhive.rl_envs import RoboHiveEnv
from torchrl.envs.utils import set_exploration_mode
from torchrl.envs.transforms import RewardScaling, TransformedEnv, FlattenObservation
from torchrl.envs import TransformedEnv, R3MTransform
from torchrl.envs import (
    CatTensors,
    DoubleToFloat,
    EnvCreator,
    ObservationNorm,
)

def make_transformed_env_new(
    env,
    stats=None,
):
    env = TransformedEnv(env, R3MTransform('resnet50', in_keys=["pixels"], download=True))
    env.append_transform(FlattenObservation(first_dim=0, in_keys=['r3m_vec']))
    return env

def make_transformed_env(
    env,
    stats=None,
):
    env = TransformedEnv(env, R3MTransform('resnet50', in_keys=["pixels"], download=True))
    env.append_transform(FlattenObservation(first_dim=0))
    return env

base_env = RoboHiveEnv("visual_franka_slide_random-v3", device=torch.device('cuda:0'))
env = make_transformed_env(base_env)
with torch.no_grad(), set_exploration_mode("random"):
    td = env.reset()
    #print(td['observation_vector'].shape)
    print(td['r3m_vec'].shape)

env = make_transformed_env_new(base_env)
with torch.no_grad(), set_exploration_mode("random"):
    td = env.reset()
    #print(td['observation_vector'].shape)
    print(td['r3m_vec'].shape)
