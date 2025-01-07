import argparse
import logging
from datetime import datetime
from settings import AgentConfig as ac
from settings import EnvConfig as ec
from settings import GeneralConfig as gc

def parse_flags() -> None:
    """Set the configs in settings.py using the commandline arguments."""
    parser = argparse.ArgumentParser()

    parse_env_config(parser)
    parse_agent_config(parser)
    parse_general_config(parser)

    args = parser.parse_args()

    gc.verbosity = args.loglevel
    gc.start_seed = args.start_seed
    gc.num_seeds = args.num_seeds
    gc.results_dir = args.results_dir
    gc.timings_dir = args.timings_dir
    gc.agent = args.agent

    ac.curiosity_methods_to_run = args.curiosity_methods
    ac.learning_name = args.learning_name
    ac.max_zpk_learning_time = args.max_zpk_learning_time
    ac.operator_fail_limit = int(args.operator_fail_limit)
    ac.temperature = str(args.temperature)
    ac.init_ops_method = args.init_ops_method
    ac.local_minima_method = args.local_minima_method
    ac.oracle_max_depth = args.oracle_max_depth
    ac.alpha = args.alpha
    ac.p_min = args.p_min
    
    def gen(file):
        if file is None:
            lines = []
        else:
            with open(file, 'r') as f:
                lines = f.readlines()
        for line in lines:
            if not line.strip(): continue
            yield line
        
    ac.input_generator = gen(args.inputs_file)

    ec.domain_names = args.domains

def parse_general_config(parser:argparse.ArgumentParser):
    parser.add_argument('--start_seed', type=int, required=True)
    parser.add_argument('--num_seeds', type=int, required=True)
    parser.add_argument('--dataset_logging', action="store_true", default=False)

    parser.add_argument("--debug", action="store_const", dest="loglevel", const=logging.DEBUG, default=logging.INFO)
    parser.add_argument('--data_dir', type=str, default='data', help="Path to variational distance transition pickles")
    parser.add_argument("--results_dir", type=str, default='results')
    parser.add_argument("--timings_dir", type=str, default='results/timings')
    parser.add_argument("--planning_results_dir", type=str, default='results/planning_results')
    parser.add_argument("--agent", type=str, choices=['create_demos', 'use_demos', 'student', 'vanilla'], default='vanilla')

def parse_env_config(parser:argparse.ArgumentParser):
    parser.add_argument("--domains", required=True, nargs='+')

def parse_agent_config(parser:argparse.ArgumentParser):
    parser.add_argument('--curiosity_methods', required=True, nargs='+')
    parser.add_argument('--learning_name', required=True, type=str)
    parser.add_argument('--max_zpk_learning_time', type=int, default=180, help='seconds before timeout ZPK')
    parser.add_argument('--operator_fail_limit', required=False, default=0, help='# times before deleting the operator')
    parser.add_argument('--temperature', required=False, default=1, help='LLM temperature')
    parser.add_argument('--init_ops_method', required=False, default='skill-conditioned', choices=['goal-conditioned', 'skill-conditioned', 'combined-todo-goal', 'skill-conditioned-two-stage', 'combined-all'])
    parser.add_argument('--local_minima_method', required=False, default='delete-operator', choices=['precond-relax', 'delete-operator'])
    parser.add_argument('--inputs_file', required=False, type=str, help="File to inputs for StudentAgentSubgoals")
    parser.add_argument("--oracle_max_depth", type=int, default=2)
    parser.add_argument('--oracle_max_neighbors', type=int, default=50)
    parser.add_argument("--alpha", type=float, default=.5)
    parser.add_argument("--p_min", type=float, default=1e-11)