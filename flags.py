import argparse
import logging
from datetime import datetime
from settings import AgentConfig as ac
from settings import EnvConfig as ec
from settings import GeneralConfig as gc
from settings import LLMConfig as lc

def parse_flags() -> None:
    """Set the configs in settings.py using the commandline arguments."""
    parser = argparse.ArgumentParser()

    parse_llm_config(parser)
    parse_env_config(parser)
    parse_agent_config(parser)
    parse_general_config(parser)

    args = parser.parse_args()

    lc.iterative_log_path = args.llm_iterative_log
    lc.cache_dir = args.llm_cache_dir
    lc.model = args.llm_model_name
    lc.max_tokens = args.llm_max_tokens

    gc.verbosity = args.loglevel
    gc.start_seed = args.start_seed
    gc.num_seeds = args.num_seeds
    gc.vardisttrans_dir = args.data_dir
    gc.results_dir = args.results_dir
    gc.timings_dir = args.timings_dir
    gc.planning_results_dir = args.planning_results_dir
    gc.dataset_logging = args.dataset_logging

    ac.curiosity_methods_to_run = args.curiosity_methods
    ac.learning_name = args.learning_name
    ac.max_zpk_learning_time = args.max_zpk_learning_time
    ac.max_traj_len = args.max_traj_len
    ac.operator_fail_limit = int(args.operator_fail_limit)
    ac.temperature = str(args.temperature)
    ac.init_ops_method = args.init_ops_method

    ec.domain_name = args.domains

 
def parse_llm_config(parser:argparse.ArgumentParser):
    parser.add_argument("--llm_iterative_log", type=str, default='llm_iterative_log')
    parser.add_argument("--llm_cache_dir", type=str, default="llm_cache")
    parser.add_argument("--llm_model_name", type=str, default='gpt-4')
    parser.add_argument("--llm_max_tokens", type=int, default=4096)

    # LLM Iterative method hyperparameters
    parser.add_argument('--max_traj_len', type=int, default=10)

def parse_general_config(parser:argparse.ArgumentParser):
    parser.add_argument('--start_seed', type=int, required=True)
    parser.add_argument('--num_seeds', type=int, required=True)
    parser.add_argument('--dataset_logging', action="store_true", default=False)

    parser.add_argument("--debug", action="store_const", dest="loglevel", const=logging.DEBUG, default=logging.INFO)
    parser.add_argument('--data_dir', type=str, default='data', help="Path to variational distance transition pickles")
    parser.add_argument("--results_dir", type=str, default='results')
    parser.add_argument("--timings_dir", type=str, default='results/timings')
    parser.add_argument("--planning_results_dir", type=str, default='results/planning_results')

def parse_env_config(parser:argparse.ArgumentParser):
    parser.add_argument("--domains", required=True, nargs='+')

def parse_agent_config(parser:argparse.ArgumentParser):
    parser.add_argument('--curiosity_methods', required=True, nargs='+')
    parser.add_argument('--learning_name', required=True, type=str)
    parser.add_argument('--max_zpk_learning_time', type=int, default=180, help='seconds before timeout ZPK')
    parser.add_argument('--operator_fail_limit', required=False, default=0, help='# times before deleting the operator')
    parser.add_argument('--temperature', required=False, default=1, help='LLM temperature')
    parser.add_argument('--init_ops_method', required=False, default='skill-conditioned', choices=['goal-conditioned', 'skill-conditioned', 'combined'])
