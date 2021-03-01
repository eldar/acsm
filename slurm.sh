#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 48:00:00
#SBATCH -o "/scratch/inf0/user/insafutdin/slurm-%A.out"
#SBATCH -e "/scratch/inf0/user/insafutdin/slurm-%A.err"
#SBATCH --gres gpu:1
echo "Starting job"

eval "$(conda shell.bash hook)"
conda activate acsm

#EXPDIR=/BS/eldar-3d2/work/src2/depth-exp

cd /BS/eldar-3d2/work/src2/acsm

python -m acsm.experiments.pascal.csp \
--name acsm_bird_0parts_kp_False_eldar \
--batch_size=6 \
--n_data_workers=8 \
--display_port=8098 \
--display_visuals \
--display_freq=100 \
--save_visuals \
--save_visual_freq=100 \
--use_html \
--kp_loss_wt=0.0 \
--save_epoch_freq=50 \
--save_visual_count=1 \
--single_axis_pred=True \
--dl_out_pascal=True \
--dl_out_imnet=True \
--num_epochs=400 \
--warmup_pose_iter=500 \
--warmup_deform_iter=10000 \
--warmup_semi_supv=0 \
--multiple_cam=True \
--flip_train=True \
--ent_loss_wt=0.05 \
--scale_bias=0.75 \
--num_hypo_cams=8 \
--parts_file=acsm/part_files/bird_0.txt \
--reproject_loss_wt=1.0 \
--mask_loss_wt=0.0 \
--no_trans=False \
--cov_mask_loss_wt=10 \
--con_mask_loss_wt=0.1 \
--n_contour=1000 \
--nmr_uv_loss_wt=0.0 \
--resnet_style_decoder=True \
--resnet_blocks=4 \
--depth_loss_wt=1.0 \
--category=bird \
--el_euler_range=20 \
--cyc_euler_range=20