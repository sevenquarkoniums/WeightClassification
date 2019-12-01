#!/usr/bin/env python
from subprocess import call
def submit(script='name.py', shfile='run.sh', outfile='output.out', system='cori'):
    '''
    submit a batch job.
    '''
    fsh = open(shfile, 'w')
    line = '#!/bin/bash -l\n'
    if system == 'cori':
        line += 'module load python/3.6-anaconda-5.2\n'
        line += 'export HDF5_USE_FILE_LOCKING=FALSE\n'
    elif system == 'scc':
        line += 'module load anaconda3\n'
    line += '%s\n' % script
    fsh.write(line)
    fsh.close()

    command = 'chmod +x \'%s\'\n' % shfile
    call(command, shell=True)

    if system == 'cori':
        command = 'sbatch -N 1 -C haswell --qos=premium -t 9:00:00 -o %s %s' % (outfile, shfile)
    elif system == 'scc':
        command = 'qsub -j y -o %s %s' % (outfile, shfile)
    call(command, shell=True)

for seed in range(3):
    submit('./main.py %d' % seed, 'sh/%d.sh' % seed, 'out/%d.out' % seed, 'scc')
    