python slot_attention/hydra_train.py sweep_name="corr_new" data_mix_idx="range(0, 24)" lr=0.0002 hydra/launcher=submitit_slurm hydra.launcher.timeout_min=3200 hydra.launcher.gpus_per_task=2 hydra.launcher.max_num_timeout=100 hydra.launcher.partition="learnlab" hydra.launcher.cpus_per_gpu=10 -m
