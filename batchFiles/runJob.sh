#!/bin/bash
                           
#SBATCH --no-requeue
#SBATCH --partition=alpha
#SBATCH --nodes=1                   
#SBATCH --gres=gpu:1
#SBATCH --mem=8000
#SBATCH --mincpus=1
#SBATCH --time=24:00:00                             
#SBATCH --job-name=DeepaugBigRunPath
#SBATCH --mail-type=ALL
#SBATCH --mail-user=colin.simon@mailbox.tu-dresden.de
#SBATCH --output=output-%j.out
#SBATCH --error=error-%j.out



module --force purge                                          
module load modenv/hiera GCC/10.3.0 OpenMPI/4.1.1 TensorFlow                 

COMPUTE_WS_NAME=deepaug_$SLURM_JOB_ID
COMPUTE_WS_PATH=$(ws_allocate -F ssd $COMPUTE_WS_NAME 7)
echo WS_Name: $COMPUTE_WS_NAME
echo WS_Path: $COMPUTE_WS_PATH

virtualenv $COMPUTE_WS_PATH/pyenv
source $COMPUTE_WS_PATH/pyenv/bin/activate

cd /scratch/ws/0/cosi765e-python_virtual_environment/deepaugment


pip install -r requirements.txt

python run_policy_generation.py

deactivate

ws_release -F ssd $COMPUTE_WS_NAME