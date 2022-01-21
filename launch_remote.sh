python hydra_train.py sweep_name="sa_final" data_mix_idx=1,2,3,4,5,13,14,15,16,17 hydra/launcher=submitit_slurm hydra.launcher.timeout_min=3200 hydra.launcher.gpus_per_task=2 hydra.launcher.max_num_timeout=100 hydra.launcher.partition="learnlab" hydra.launcher.cpus_per_gpu=10 -m

