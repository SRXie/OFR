python slot_attention/hydra_train.py hydra/launcher=submitit_slurm hydra.launcher.timeout_min=400 hydra.launcher.gpus_per_node=2 hydra.launcher.max_num_timeout=100 hydra.launcher.partition="learnfair" -m
