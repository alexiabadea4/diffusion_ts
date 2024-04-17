import sys
import os
import itertools
import json
import argparse

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
        'batch_size': [64],
        'learning_rate': [1e-3],
        'beta_end': [0.1,0.2,0.01,0.05,0.001],
        'diff_steps': [100,50,500,1000],
        'loss_type': ["l2"],
        'beta_schedule': ["linear","quad","const","jsd","sigmoid","cosine"],
    }
}





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
    #PBS -N job_{i}



    module load anaconda3/personal
    source /rds/general/user/ab1320/home/anaconda3/etc/profile.d/conda.sh
    conda activate myenv

    cd /rds/general/user/ab1320/home/diffusion_ts/diffusion_ts
    export WANDB_API_KEY='ea7b35f3ac6447d69c7e329e7da9ab67693e75f3'

    {command}
    """
    with open(f'job_{i}.pbs', 'w') as f:
        f.write(job_script)
    print(f'Job script job_{i}.pbs created.')

    os.system("qsub job_{i}.pbs")