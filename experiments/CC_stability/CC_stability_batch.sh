#!/bin/bash

#SBATCH --array=1-1000%25 	    # job array (limit up to 25)
#SBATCH --nodes=1               # number of nodes to use
#SBATCH --ntasks-per-node=4     # number of tasks
#SBATCH --cpus-per-task=2       # number of CPU cores per task
#SBATCH --gres=gpu:4            # number of GPUs to use
#SBATCH --mem=8G     	    	# memory per node (0 = use all of it)
#SBATCH --time=00:45:00         # time (DD-HH:MM)
#SBATCH --account=def-adcockb

# set up environment
module load python/3.10
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

cd $SLURM_TMPDIR
git clone https://github.com/mneyrane/MSc-thesis-NESTAnets.git
cd MSc-thesis-NESTAnets
pip install --no-index -e . 
cd experiments/CC_stability

PROJECT_DIR="$HOME/projects/def-adcockb/mneyrane"
ARCHIVE_NAME="CC_stability_$SLURM_ARRAY_TASK_ID.tar"

# experiment parameters
IMAGE_PATH="$SLURM_TMPDIR/MSc-thesis-NESTAnets/experiments/images/brain_512.png"
ETA="1e-2"
ETA_PERT_VALUES=("1e-2" "1e-1" "1e0" "1e1")

for i in ${!ETA_PERT_VALUES[@]}; do
    PARAMS=(
        --save-dir "RESULTS-$SLURM_ARRAY_TASK_ID-$i" 
        --image-path $IMAGE_PATH 
        --eta $ETA 
        --eta-pert ${ETA_PERT_VALUES[$i]}
    )
    CUDA_VISIBLE_DEVICES=$i python CC_stability.py "${PARAMS[@]}" &
done

wait # for tasks to finish

tar -cf $ARCHIVE_NAME RESULTS-*

mv $ARCHIVE_NAME $PROJECT_DIR
