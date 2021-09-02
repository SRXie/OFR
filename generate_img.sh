#! /bin/bash

#SBATCH --job-name=clevr_data_gen

# Use to change the number of replicates
#SBATCH --array=0-1000

#SBATCH --output=/private/home/%u/runs/ad_hoc_categories/logs/%x_%A_%a.out

#SBATCH --partition=uninterrupted

#SBATCH --ntasks=1

#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=1

#SBATCH --time=40:00:00

#SBATCH --signal=USR1@60

#SBATCH --open-mode=append
job_name=${SLURM_JOB_NAME}
job_date=$(date +%y%m%d:%H)
experiment_name="${job_name}_${job_date}_${SLURM_ARRAY_JOB_ID}"

echo "Experiment name: ${experiment_name}"
echo "Job array ID: ${SLURM_ARRAY_TASK_ID}"
echo "Job array Count: ${SLURM_ARRAY_TASK_COUNT}"
echo "Job array Max: ${SLURM_ARRAY_TASK_MAX}"
echo "Job array Min: ${SLURM_ARRAY_TASK_MIN}"

USE_GPU=1

if [ ${USE_GPU} == 1 ]; then
  module load cuda/10.0
fi

if [ ! -e "/checkpoint/siruixie/clevr_corr" ]; then
  mkdir "/checkpoint/siruixie/clevr_corr"
fi

OUTPUT_DIR="/checkpoint/siruixie/clevr_corr/"

if [ ! -e ${OUTPUT_DIR} ]; then
  mkdir ${OUTPUT_DIR}
  mkdir ${OUTPUT_DIR}/images
  mkdir ${OUTPUT_DIR}/scenes
fi

if [ ! -e ${OUTPUT_DIR}/images/${SLURM_ARRAY_TASK_ID} ]; then
  mkdir ${OUTPUT_DIR}/images/${SLURM_ARRAY_TASK_ID}
  mkdir ${OUTPUT_DIR}/scenes/${SLURM_ARRAY_TASK_ID}
fi

srun blender --background -noaudio --python \
    clevr_obj_test/image_generation/render_images.py -- \
    --num_images ${RAW_DATASET_NUM_IMAGES} \
    --output_image_dir "${OUTPUT_DIR}/images/${SLURM_ARRAY_TASK_ID}"\
    --output_scene_dir "${OUTPUT_DIR}/scenes/${SLURM_ARRAY_TASK_ID}"\
    --output_scene_file "${OUTPUT_DIR}/ADHOC_scenes_${SLURM_ARRAY_TASK_ID}.json"\
    --use_gpu ${USE_GPU}\
    --color_correlated
    --en_sigma 0.1
    --max_job_id ${SLURM_ARRAY_TASK_MAX}\
    --current_job_id ${SLURM_ARRAY_TASK_ID}

# Uncomment to render.

# These lines render locally if you uncomment thte for-loop
# SLURM_ARRAY_TASK_MAX=1000
#SLURM_ARRAY_TASK_ID=1
# Takes 3 seconds to render 1 image.
# for SLURM_ARRAY_TASK_ID in {0..1000}; do
#   blender --background -noaudio --python \
#     clevr-dataset-gen/image_generation/render_images.py --\
#     --num_images ${RAW_DATASET_NUM_IMAGES}\
#     --render_or_sample 1\
#     --filename_prefix ADHOC\
#     --split "train" \
#     --output_image_dir "${OUTPUT_DIR}/images/${SLURM_ARRAY_TASK_ID}"\
#     --output_scene_dir "${OUTPUT_DIR}/scenes/${SLURM_ARRAY_TASK_ID}"\
#     --output_scene_file "${OUTPUT_DIR}/ADHOC_scenes_${SLURM_ARRAY_TASK_ID}.json"\
#     --use_gpu ${USE_GPU}\
#     --max_job_id ${SLURM_ARRAY_TASK_MAX}\
#     --current_job_id ${SLURM_ARRAY_TASK_ID}\
#     --render_num_samples ${RENDER_NUM_SAMPLES}

#   echo "Done"
# done
