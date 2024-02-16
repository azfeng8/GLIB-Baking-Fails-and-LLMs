"""Launch script for openstack experiments.

Requires a file that contains a list of IP addresses for instances that are:
    - Turned on
    - Accessible via ssh for the user of this file
    - Configured with an llm_glib image (current snapshot name: llm_glib-v1)
    - Sufficient in number to run all of the experiments in the config file
Make sure to place this file within the openstack_scripts folder.
    
Usage example:
    python openstack_scripts/launch.py --config example.yaml \
        --machines machines.txt

The default branch can be overridden with the --branch flag.
"""

import argparse
import os

from cluster_utils import DEFAULT_BRANCH, SingleSeedRunConfig, \
    config_to_cmd_flags, config_to_logfile, generate_run_configs, \
    get_cmds_to_prep_repo, run_cmds_on_machine


def _main() -> None:
    # Set up argparse.
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--machines", required=True, type=str)
    parser.add_argument("--sshkey", required=False, type=str, default=None)
    parser.add_argument("--branch", type=str, default=DEFAULT_BRANCH)
    args = parser.parse_args()
    openstack_dir = os.path.dirname(os.path.realpath(__file__))
    # Load the machine IPs.
    machine_file = os.path.join(openstack_dir, args.machines)
    with open(machine_file, "r", encoding="utf-8") as f:
        machines = f.read().splitlines()
    # Make sure that the ssh key exists.
    if args.sshkey is not None:
        assert os.path.exists(args.sshkey)
    # Generate all of the run configs.
    run_configs = list(generate_run_configs(args.config))
    num_machines = len(machines)

    if len(run_configs) > num_machines:
        # Divide the runs between the machines.
        runs_per_machine = [len(run_configs)  // num_machines] * num_machines
        extra_runs = len(run_configs) % num_machines
        for i in range(len(runs_per_machine)):
            if i < extra_runs:
                runs_per_machine[i] += 1

        assert sum(runs_per_machine) == len(run_configs)

        cfg_i = 0
        for machine_i, runs in enumerate(runs_per_machine):
            for _ in range(runs):
                cfg = run_configs[cfg_i]
                assert isinstance(cfg, SingleSeedRunConfig)
                logfile = os.path.join("logs", config_to_logfile(cfg))
                cmd_flags = config_to_cmd_flags(cfg)
                cmd = f"python main.py {cmd_flags}"
                _launch_experiment(cmd, machines[machine_i], logfile, args.sshkey, args.branch)
                cfg_i += 1

    else:
        # Launch the runs.
        for machine,cfg in zip(machines, run_configs):
            assert isinstance(cfg, SingleSeedRunConfig)
            logfile = os.path.join("logs", config_to_logfile(cfg))
            cmd_flags = config_to_cmd_flags(cfg)
            cmd = f"python main.py {cmd_flags}"
            _launch_experiment(cmd, machine, logfile, args.sshkey, args.branch)


def _launch_experiment(cmd: str, machine: str, logfile: str, ssh_key: str,
                       branch: str) -> None:
    print(f"Launching on machine {machine}: {cmd}")
    # Enter the repo and activate conda.
    server_cmds = ["llm_glib"]
    # Prepare the repo.
    server_cmds.extend(get_cmds_to_prep_repo(branch))
    # Run the main command.
    server_cmds.append(f"{cmd} &> {logfile} &")
    run_cmds_on_machine(server_cmds, "ubuntu", machine, ssh_key=ssh_key)


if __name__ == "__main__":
    _main()
