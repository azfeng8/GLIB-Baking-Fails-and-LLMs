import argparse
import os

from cluster_utils import DEFAULT_BRANCH, SingleSeedRunConfig, \
    config_to_cmd_flags, config_to_logfile, generate_run_configs, \
    get_cmds_to_prep_repo, run_cmds_on_machine


def _main() -> None:
    # Set up argparse.
    parser = argparse.ArgumentParser()
    parser.add_argument("--machines", required=True, type=str)
    args = parser.parse_args()
    openstack_dir = os.path.dirname(os.path.realpath(__file__))
    # Load the machine IPs.
    machine_file = os.path.join(openstack_dir, args.machines)
    with open(machine_file, "r", encoding="utf-8") as f:
        machines = f.read().splitlines()
    for machine in machines:
        clear_tmp(machine)
 

def clear_tmp(machine: str):
    print(f"Launching on machine {machine}")
    # Enter the repo and activate conda.
    server_cmds = ["cd /tmp", "find . -maxdepth 1 -name '*.pddl' -delete &"]
    # Prepare the repo.
    # Run the main command.
    run_cmds_on_machine(server_cmds, "ubuntu", machine)

if __name__ == '__main__':
    _main()
