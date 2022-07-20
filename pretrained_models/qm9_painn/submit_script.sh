#!/bin/bash -ex
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=sm3090
#SBATCH -N 1      # Minimum of 1 node
#SBATCH -n 8     # 7 MPI processes per node
#SBATCH --time=7-00:00:00
#SBATCH --mem=15G     # 10 GB RAM per node
#SBATCH --gres=gpu:RTX3090:1
module load Python
module load foss
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
source ~/graphnn_env/bin/activate

    cd ~/densitynet_revisions/2021-05-18T17:49:52+02:00-953ae0e
    git fetch && git checkout 953ae0e
    python -u runner.py --dataset /home/niflheim2/pbjo/qm9vasp/qm9vasp.txt --split_file /home/niflheim2/pbjo/qm9vasp/datasplits.json  --cutoff 4 --num_interactions 3 --max_steps 10000000 --node_size 128 --use_painn_model --output_dir ~/densitynet_runs/2021-05-18T17:52:04.873627
    