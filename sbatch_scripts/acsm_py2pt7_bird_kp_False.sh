#!/bin/sh
#SBATCH --partition=vl-fb-gtx1080
#SBATCH --job-name=acsm_py2pt7_bird_kp_False
#SBATCH --output=/home/nileshk/DeformParts/refactor/dcsm/cachedir/slurm_logs/acsm_py2pt7_bird_kp_False.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=10-00:00:00
#SBATCH --mem=16G
set -x 

module load cuda/9.2
cd /home/nileshk/DeformParts/refactor/
source ~/anaconda3/etc/profile.d/conda.sh
conda activate dcsm2
hostname
srun python -m dcsm.experiments.pascal.csp --name acsm_py2pt7_bird_kp_False --batch_size=6 --n_data_workers=8 --display_port=8098 --display_visuals --display_freq=100 --save_visuals --save_visual_freq=100 --use_html --kp_loss_wt=0.0 --save_epoch_freq=50 --save_visual_count=1 --single_axis_pred=True --dl_out_pascal=True --dl_out_imnet=True --num_epochs=400 --warmup_pose_iter=500 --warmup_deform_iter=10000 --warmup_semi_supv=0 --multiple_cam=True --flip_train=True --ent_loss_wt=0.05 --scale_bias=0.75 --num_hypo_cams=8 --parts_file=acsm/part_files/bird_0.txt --reproject_loss_wt=1.0 --mask_loss_wt=0.0 --no_trans=False --cov_mask_loss_wt=10 --con_mask_loss_wt=0.1 --n_contour=1000  --nmr_uv_loss_wt=0.0 --resnet_style_decoder=True --resnet_blocks=4 --depth_loss_wt=1.0 --category=bird --el_euler_range=20 --cyc_euler_range=20 
