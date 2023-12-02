import subprocess

# Path to your virtual environment's activate script
# virtual_env_activate = '/work/sbajaj_umass_edu/GNNEnv/bin/activate'

# # Command to activate the virtual environment
# activate_command = f'source {virtual_env_activate}'

# Command to run your Python script within the virtual environment
script_to_run = '/work/sbajaj_umass_edu/GNNEnv/bin/python3 process_init_group.py'

try:
    # Activate the virtual environment
    # subprocess.run(activate_command, shell=True, check=True)

    # Run the Python script within the virtual environment
    subprocess.run(script_to_run, shell=True, check=True)
except subprocess.CalledProcessError as e:
    print(f"An error occurred: {e}")
