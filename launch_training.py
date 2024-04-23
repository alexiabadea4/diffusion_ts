import sys
import os
import itertools
import json
import argparse
import time
def safe_write_and_verify(file_name, content):
    try:
        with open(file_name, 'w') as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        if not os.path.exists(file_name) or os.path.getsize(file_name) == 0:
            raise ValueError("File does not exist or is empty after writing.")
        time.sleep(1)  
        return True
    except Exception as e:
        print(f"Failed to write and verify {file_name}: {e}")

        return False
def load_json(filename):
    with open(filename) as f:
        data = json.load(f)
    return data

class SafeDict(dict):
    def __missing__(self, key):
        return '{' + key + '}'

# def dict_product(dicts):
#     return (dict(zip(dicts.keys(), x)) for x in itertools.product(*dicts.values()))
def dict_product(dicts):
  
    keys = dicts.keys()
    values = dicts.values()
    for combination in itertools.product(*values):
        yield dict(zip(keys, combination))



hp = {
    'common': {
        'epochs': [500],
        'batch_size': [32,64,128],
        'learning_rate': [1e-2, 1e-3, 1e-4],
        'beta_end': [0.1,0.5,0.2,0.3,0.4,0.6,0.7,0.8,0.9],
        'diff_steps': [10,50,90,100,150,200],
        'loss_type': ["l1","huber","l2"],
        'beta_schedule': ["linear","quad","const","jsd","sigmoid","cosine"],
    }
}

# scp monitor_jobs.py ab1320@login.hpc.imperial.ac.uk:/rds/general/user/ab1320/home/diffusion_ts/diffusion_ts


pbs_directory = '/rds/general/user/ab1320/home/diffusion_ts/diffusion_ts/pbs_files_order'
if not os.path.exists(pbs_directory):
    os.makedirs(pbs_directory)

configurations = list(dict_product(hp['common']))
print(hp)



print(f'Launching {len(configurations)} experiments')
for config in configurations:
    print(config)

if input("Proceed? (y/n): ").strip().lower() != 'y':
    print('Aborted.')
    sys.exit()


for i, config in enumerate(configurations):
    command = f"python trainer_hpc.py --epochs {config['epochs']} --batch_size {config['batch_size']} --learning_rate {config['learning_rate']} --beta_end {config['beta_end']} --diff_steps {config['diff_steps']} --loss_type {config['loss_type']} --beta_schedule {config['beta_schedule']}"
    job_script = f"""
    #!/bin/bash
    
    #PBS -l select=1:ncpus=1:mem=4gb
    #PBS -l walltime=03:00:00
    #PBS -N job_{str(i).zfill(4)}



    module load anaconda3/personal
    source /rds/general/user/ab1320/home/anaconda3/etc/profile.d/conda.sh
    conda activate myenv

    cd /rds/general/user/ab1320/home/diffusion_ts/diffusion_ts
    export WANDB_API_KEY='ea7b35f3ac6447d69c7e329e7da9ab67693e75f3'

    {command}
    """
    file_name = os.path.join(pbs_directory, f'job_{str(i).zfill(4)}.pbs')
    
    if safe_write_and_verify(file_name, job_script):
        print(f"Submitting job script {file_name}.")
    #     os.system(f"qsub {file_name}")
    # else:
    #     print("File writing or verification failed.")

