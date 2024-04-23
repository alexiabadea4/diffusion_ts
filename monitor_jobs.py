# import time
# import os
# import glob
# import subprocess

# def is_job_queued_or_running(job_name):
#     try:
#         # This command gets the current queued and running jobs
#         result = subprocess.check_output(['/opt/pbs/bin/qstat', '-f'], stderr=subprocess.STDOUT)
#         jobs_status = result.decode('utf-8')
#         return job_name in jobs_status
#     except subprocess.CalledProcessError as e:
#         print(f"Failed to fetch jobs status: {e.output.decode()}")
#         return False
# import subprocess

# def submit_job(pbs_file):
#     try:
#         # Capture the output and errors of the subprocess
#         result = subprocess.run(['/opt/pbs/bin/qsub', pbs_file], check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#         print(f"Submitted: {pbs_file}")
#     except subprocess.CalledProcessError as e:
#         # Now the output can be safely accessed
#         print(f"Failed to submit job {pbs_file}: Return Code: {e.returncode}")
#         if e.stdout:
#             print("STDOUT:", e.stdout)
#         if e.stderr:
#             print("STDERR:", e.stderr)

# def monitor_and_queue_jobs():
#     # Set the working directory where job scripts are located
#     os.chdir('/rds/general/user/ab1320/home/diffusion_ts/diffusion_ts/pbs_files')
    
#     while True:
#         pbs_files = sorted(glob.glob('*.pbs'))
#         print(f"Checking {len(pbs_files)} job scripts...")

#         for pbs_file in pbs_files:
#             if not is_job_queued_or_running(pbs_file):
#                 print(f"{pbs_file} will be queued")
#                 submit_job(pbs_file)
#             else:
#                 print(f"{pbs_file} is already queued or running")

#         # Sleep before the next check
#         sleep_seconds = 180
#         print(f"Waiting {sleep_seconds} seconds before next check...")
#         time.sleep(sleep_seconds)

# if __name__ == '__main__':
#     monitor_and_queue_jobs()


# import subprocess
# import os
# import glob
# import time

# def is_queue_full():
#     try:
#         # Execute qstat and pipe to wc -l to count lines
#         result = subprocess.run('qstat | wc -l', shell=True, capture_output=True, text=True)
#         # Check if the number of lines equals 52 (adjust this number if needed)
#         return int(result.stdout.strip()) == 52
#     except subprocess.CalledProcessError as e:
#         print("Error checking queue:", e)
#         return False  # Assume queue is not full if there's an error

# def has_job_been_queued(job_number):
#     # Check for the existence of an output file for the job in the current directory
#     output_files = glob.glob(f'job_{job_number}.o*')
#     return len(output_files) > 0

# def submit_job(pbs_file):
#     try:
#         # Submit the job using qsub
#         subprocess.run(['qsub', pbs_file], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#         print(f"Submitted: {pbs_file}")
#     except subprocess.CalledProcessError as e:
#         print(f"Failed to submit job {pbs_file}: {e}")
#         if e.stdout:
#             print("STDOUT:", e.stdout)
#         if e.stderr:
#             print("STDERR:", e.stderr)

# def monitor_and_queue_jobs():
#     # Set the working directory where job scripts are located
#     os.chdir('/rds/general/user/ab1320/home/diffusion_ts/diffusion_ts/pbs_files')

#     while True:
#         pbs_files = sorted(glob.glob('*.pbs'))
#         print(f"Checking {len(pbs_files)} job scripts...")

#         if not is_queue_full():
#             print("not full")
#             for pbs_file in pbs_files:
#                 job_number = pbs_file.split('.')[0].split('_')[1]  # Assuming file format is 'job_X.pbs'
#                 if not has_job_been_queued(job_number):
#                     print(f"{pbs_file} will be queued")
#         #             submit_job(pbs_file)
#         #         else:
#         #             print(f"{pbs_file} has already been queued or is running")
#         # else:
#         #     print("Queue is full, waiting to submit new jobs.")

#         # Sleep before the next check
#         sleep_seconds = 180
#         print(f"Waiting {sleep_seconds} seconds before next check...")
#         time.sleep(sleep_seconds)

# if __name__ == '__main__':
#     monitor_and_queue_jobs()

import subprocess
import os
import glob
import time

def rename_files(directory):
    # Change the working directory to where the files are located
    os.chdir(directory)

    # List all files that match the pattern 'job_*.pbs' and extract their numbers
    files = [f for f in os.listdir() if f.startswith('job_') and f.endswith('.pbs')]
    files_sorted = sorted(files, key=lambda x: int(x.split('_')[1].split('.')[0]))

    # Rename files to have consecutive numbering with zero-padding
    for index, filename in enumerate(files_sorted, start=0):  # Start from 0 for correct zero-padding
        new_filename = f"job_{str(index).zfill(4)}.pbs"  # zfill pads the string with zeros on the left to fill width
        os.rename(filename, new_filename)

        print(f"Renamed {filename} to {new_filename}")

def get_current_queue_count():
    try:
        # Execute qstat, pipe to wc -l to count lines, and get the number of current jobs
        result = subprocess.run('qstat | wc -l', shell=True, capture_output=True, text=True)
        return int(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        print("Error checking queue:", e)
        return 0  # Assume zero if there's an error to safely handle unexpected issues

def has_job_been_queued(job_number):
    # Check for the existence of an output file for the job in the current directory
    output_files = glob.glob(f'job_{job_number}.o*')
    error_files = glob.glob(f'job_{job_number}.e*')
    return len(output_files) > 0 or len(error_files) > 0 

def submit_job(pbs_file):
    try:
    
        #subprocess.run(['qsub', pbs_file], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Submitted: {pbs_file}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to submit job {pbs_file}: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)

def monitor_and_queue_jobs():
    # Set the working directory where job scripts are located
    os.chdir('/rds/general/user/ab1320/home/diffusion_ts/diffusion_ts/pbs_files_order')
    directory = '/rds/general/user/ab1320/home/diffusion_ts/diffusion_ts/pbs_files_order'
    #rename_files(directory)
    max_jobs = 52  # Maximum number of jobs that can be in the queue

    while True:
        current_queue_count = get_current_queue_count()
        available_slots = max_jobs - current_queue_count
        if available_slots == 52:
            available_slots = 50
        if available_slots > 0:
            print(f"{available_slots} slots available in the queue.")
            pbs_files = sorted(glob.glob('*.pbs'))
            queued_jobs = 0

            for pbs_file in pbs_files:
                print(f"{available_slots} slots available in the queue.")
                if available_slots<=1:
                    break  # Stop queuing if no slots are available
                job_number = pbs_file.split('.')[0].split('_')[1]  # Extract job number from filename
                
                if not has_job_been_queued(job_number):
                    print(f"{pbs_file} will be queued")
                    available_slots = available_slots-1
                    subprocess.run(['qsub', pbs_file], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                    # if submit_job(pbs_file):
                    #     print(f"{pbs_file} has been queued")
                    #     queued_jobs += 1
                        
                else:
                    print(f"{pbs_file} has already been queued or is running")
        else:
            print("Queue is full, waiting to submit new jobs.")

        # Sleep before the next check
        sleep_seconds = 180
        print(f"Waiting {sleep_seconds} seconds before next check...")
        time.sleep(sleep_seconds)

if __name__ == '__main__':
    monitor_and_queue_jobs()
