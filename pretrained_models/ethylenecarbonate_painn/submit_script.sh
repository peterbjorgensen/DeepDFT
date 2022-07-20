#!/bin/bash -ex
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=sm3090
#SBATCH -N 1      # Minimum of 1 node
#SBATCH -n 8     # 7 MPI processes per node
#SBATCH --time=7-00:00:00
#SBATCH --mem=15G     # 10 GB RAM per node
#SBATCH --gres=gpu:RTX3090:1
module load foss
module load Python/3.8.6-GCCcore-10.2.0
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
source ~/graphnn_env/bin/activate

    cd ~/densitynet_revisions/2021-06-09T17:05:43+02:00-b7b414c
    git fetch && git checkout b7b414c
    python -u runner.py --dataset /home/niflheim2/pbjo/ethylenecarbonate/ethylenecarbonate.txt --split_file /home/niflheim2/pbjo/ethylenecarbonate/splits.json --cutoff 4 --num_interactions 3 --max_steps 10000000 --node_size 128 --use_painn_model --output_dir ~/densitynet_runs/2021-06-10T18:14:58.096877
    