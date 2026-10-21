#!/usr/bin/env bash

#SBATCH -J launch.sh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20gb
#SBATCH --time=4:00:00

#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1

#SBATCH --error=train.%J.err
#SBATCH --output=train.%J.out

module load python
conda activate rsna

export MPLBACKEND=agg
export PYTORCH_ALLOC_CONF=expandable_segments:True

python main.py 
