#!/bin/bash
#SBATCH -J VGG16_1a                # jobname
#SBATCH -o VGG16_1a.o%A.%a         # jobname.o%j for single (non array) jobs jobname.o%A.%a for array jobs
#SBATCH -e VGG16_1a.e%A.%a         # error file name A is the jobid and a is the arraytaskid
#SBATCH -a 1-15 #45%15
###SBATCH -c 64                    # start and stop of the array start-end
#SBATCH -A TG-MCB180035
#SBATCH -N 1
#SBATCH -p gpu-shared  #p1,gcluster           # queue (partition) -- compute, shared, large-shared, debug (30mins)
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=4
###SBATCH --cpus-per-task=4
###SBATCH --mem-per-cpu=20000
#SBATCH -t 10:00:00             # run time (dd:hh:mm:ss) - 1.5 hours
#SBATCH --mail-user=2maitreyab@gmail.com
###SBATCH --mail-type=begin       # email me when the job starts
###SBATCH --mail-type=end         # email me when the job finishes
#SBATCH --mail-type=fail
#SBATCH --export=ALL

# Set the number of threads per task(Default=1)
#export OMP_NUM_THREADS=1
#export MV2_SHOW_CPU_BINDING=1
#CPUS=$(($SLURM_CPUS_ON_NODE/2))
#echo  $SLURM_CPUS_PER_TASK
#

cd  /expanse/lustre/projects/uci141/mbanerjee

module purge
module load gpu
module load slurm
module load singularitypro

export SINGULARITYENV_TINI_SUBREAPER=1
#export SINGULARITY_BIND="$PWD:/tf"

#document me!
JOBFILE=jobfile.txt
OPTIONS=1a$(head -n ${SLURM_ARRAY_TASK_ID} ${JOBFILE} | tail -n 1)
PYSCRIPT=./mbanerjee/VGG16_1a.py
HOSTDIR=$HOME/uci141
MOUNTDIR=/home/$USER

hostname
echo ${PWD} ${UID} ${PYSCRIPT} ${OPTIONS}
#singularity run --nv tensorflownojup.sif
#singularity exec --nv ../tensorflow.sif python ${PYSCRIPT} ${OPTIONS}
singularity exec --bind ${HOSTDIR}:${MOUNTDIR} --nv tensorflow.2.16.1.sif python ${PYSCRIPT} ${OPTIONS}
