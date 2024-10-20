"""Download results from openstack experiments.

Requires a file that contains a list of IP addresses for instances that are:
    - Turned on
    - Accessible via ssh for the user of this file
    - Configured with an llm_glib image (current snapshot name: llm_glib-v1)
    - Sufficient in number to run all of the experiments in the config file
Make sure to place this file within the openstack_scripts folder.

The dir flag should point to a directory where the results, logs, and llm_cache
subdirectories will be downloaded.

Usage example:
    python scripts/openstack/download.py --dir "$PWD" --machines machines.txt \
        --sshkey ~/.ssh/cloud.key
"""

import argparse
import os

from cluster_utils import SAVE_DIRS


def _main() -> None:
    # Set up argparse.
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, type=str)
    parser.add_argument("--machines", required=True, type=str)
    parser.add_argument("--sshkey", required=False, type=str, default=None)
    args = parser.parse_args()
    openstack_dir = os.path.dirname(os.path.realpath(__file__))
    # Load the machine IPs.
    machine_file = os.path.join(openstack_dir, args.machines)
    with open(machine_file, "r", encoding="utf-8") as f:
        machines = f.read().splitlines()
    # Make sure that the ssh key exists.
    if args.sshkey is not None:
        assert os.path.exists(args.sshkey)
    # Create the download directory if it doesn't exist.
    os.makedirs(args.dir, exist_ok=True)
    # Loop over machines.
    for machine in machines:
        _download_from_machine(machine, args.dir, args.sshkey)


def _download_from_machine(machine: str, download_dir: str,
                           ssh_key: str) -> None:
    print(f"Downloading from machine {machine}")
    try:
        for save_dir in SAVE_DIRS:
            local_save_dir = os.path.join(download_dir, save_dir)
            os.makedirs(local_save_dir, exist_ok=True)
            cmd = f"scp -r " 
            if ssh_key is not None:
                cmd += f"-i {ssh_key} "
            cmd += "-o StrictHostKeyChecking=no " + \
                f"ubuntu@{machine}:~/GLIB-Baking-Fails-and-LLMs/{save_dir}/* {local_save_dir}"
            retcode = os.system(cmd)
            if retcode != 0:
                print(f"WARNING: command failed: {cmd}")
    except KeyboardInterrupt:
        return

if __name__ == "__main__":
    _main()
