# torchrl_examples
Training examples using TorchRL repo

## Installation instructions
```
git clone --branch=robohive_sac_r3m https://github.com/ShahRutav/torchrl_examples.git
cd torchrl_examples
conda create -n rlhive -y python=3.8
conda activate rlhive
bash scripts/installation.sh
python3 -mpip install git+https://github.com/facebookresearch/rlhive.git
```

## Launching Experiments
[NOTE] Set ulimit for your shell (default 1024): `ulimit -n 4096`
```
cd sac_mujoco
sim_backend=MUJOCO MUJOCO_GL=egl python sac.py -m hydra/launcher=slurm hydra/output=slurm
```

To run a small experiment for testing, run the following command:
```
cd sac_mujoco
sim_backend=MUJOCO MUJOCO_GL=egl python sac.py -m total_frames=2000 init_random_frames=25 buffer_size=2000 hydra/launcher=slurm hydra/output=slurm
```
