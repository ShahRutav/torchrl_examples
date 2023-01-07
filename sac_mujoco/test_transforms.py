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

def make_transformed_env(
    env,
    stats=None,
):
    """
    Apply transforms to the env (such as reward scaling and state normalization)
    """

    #env = TransformedEnv(env)
    env = TransformedEnv(env, R3MTransform('resnet50', in_keys=["pixels"], download=True))
    env.append_transform(FlattenObservation(first_dim=0))
    env.append_transform(RewardScaling(loc=0.0, scale=5.0))
    selected_keys = list(env.observation_spec.keys())
    out_key = "observation_vector"
    env.append_transform(CatTensors(in_keys=selected_keys, out_key=out_key))

    ##  we normalize the states
    if stats is None:
        _stats = {"loc": 0.0, "scale": 1.0}
    else:
        _stats = stats
    env.append_transform(
        ObservationNorm(**_stats, in_keys=[out_key], standard_normal=True)
    )
    env.append_transform(DoubleToFloat(in_keys=[out_key], in_keys_inv=[]))
    return env

base_env = RoboHiveEnv("visual_franka_slide_random-v3", device=torch.device('cuda:0'))
env = make_transformed_env(base_env)
with torch.no_grad(), set_exploration_mode("random"):
    td = env.reset()
