#!/bin/bash
#SBATCH --time=15:00
#SBATCH --account=an-tr043
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=Roomman-%j.out


# add modules
module load python/3.11.5
module list


# begin bash instructions
python ISP_Hotel_Cancellation_Final.py
srun hostname
# end bash instructions
# Memory Utilized: 513.23 MB Memory Efficiency: 12.53% of 4.00 GB.
# Based on the results in the efficiency report the memory utilised was only 514 MB.
# therefore it is recommended to request memory to around 550 MB to improve memory efficiency.
 
