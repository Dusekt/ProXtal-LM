#!/bin/bash
MYDIR=`pwd`
RUN_FILE=`pwd`/run.$$

echo "#!/bin/sh" >> $RUN_FILE
echo "#PBS -l walltime=240:00:00" >> $RUN_FILE
echo "#PBS -q gpu_long" >> $RUN_FILE
echo "#PBS -l select=1:ncpus=8:ngpus=1:gpu_mem=80gb:mem=120gb:scratch_local=10gb" >> $RUN_FILE
echo "cd $MYDIR" >> $RUN_FILE

echo "singularity exec -B \$PWD:/scratch --pwd /scratch --nv /storage/brno12-cerit/home/spiwokv/esmfold/esmfold_latest.sif python scripts/train.py --config small_og --n-hypotheses 3 --checkpoint-dir checkpoints/v1_small_og --matching-mode greedy --crystal-og-weight 0.1 --diversity-weight 0.1 --name v1_small_og --max-epochs 120
" >> $RUN_FILE

qsub $RUN_FILE



