"""Settings used throughout the directory.
"""


class EnvConfig:
    """Environment-specific constants.
    """
    domain_names = []

class AgentConfig:
    """Agent-specific constants.
    """
    curiosity_methods_to_run = []

    learning_name = "LNDR"

    planner_name = {
        "Blocks": "ff",
        "Easygripper": "ff",
        "Glibdoors": "ff",
        "Bakinglarge": "fd",
    }

    # Maximum trajectory length
    max_traj_len = 10

    # How often to learn operators.
    learning_interval = {
        "Blocks": 1,
        "Easygripper": 1,
        "Glibdoors": 1,
        "Bakinglarge": 1
    }

    # Max training episode length.
    max_train_episode_length = {
        "Blocks": 25,
        "Easygripper": 25,
        "Glibdoors": 25,
        "Bakinglarge": 50,
    }
    # Max test episode length.
    max_test_episode_length = {
        "Blocks": 25,
        "Easygripper": 100,
        "Glibdoors": 25,
        "Bakinglarge": 50
    }
    # Timeout for planner.
    planner_timeout = None  # set in main.py

    # Number of training iterations.
    num_train_iters = {
        "Blocks": 501,
        "Easygripper": 301,
        "Glibdoors": 2501,
        "Bakinglarge": 2000
    }

    ## Constants for curiosity modules. ##
    max_sampling_tries = 20
    max_planning_tries = 20 
    oracle_max_depth = 2 #14
    oracle_max_neighbors = 50 # 100

    ## Constants for mutex detection. ##
    mutex_num_episodes = {
        "Blocks": 35,
        "Easygripper": 35,
        "Glibdoors": 35,
        "Bakinglarge": 50
    }
    mutex_episode_len = {
        "Blocks": 35,
        "Easygripper": 35,
        "Glibdoors": 35,
        "Bakinglarge": 50
    }
    mutex_num_action_samples = 10

    ## Constants for LNDR (also called ZPK throughout code). ##
    max_zpk_learning_time = 1800
    max_zpk_explain_examples_transitions = {
        "Blocks": 25,
        "Easygripper": 25,
        "Glibdoors": 25,
        "Bakinglarge": 50
    }
    max_zpk_action_batch_size = {
        "Blocks": None,
        "Easygripper": None,
        "Glibdoors": None,
        "Bakinglarge": None
    }
    zpk_initialize_from_previous_rule_set = {
        "Blocks": False,
        "Easygripper": False,
        "Glibdoors": False,
        "Bakinglarge": False
    }

    local_minima_method = 'delete-operator'

    p_min = 1e-11
    alpha = 0.5
    # Major hacks to access predicates.
    train_env = None


class GeneralConfig:
    """General configuration constants.
    """
    verbosity = 1
    start_seed = 1
    num_seeds = 1
    results_dir = 'results/'
    timings_dir = results_dir + 'timings/'

class PlottingConfig:
    """Plotting from cached results.
    """
    domain = "Bakinglarge"
    seeds =  range(1,11)
    agent_learner_explorer = [
        ('demoagent', "LNDR", "GLIB_L2"),
        ("demoagent", "LNDR", "oracle"), 
        ("agent", "LNDR", "oracle"),
        ('agent', "LNDR", "GLIB_L2"),
    ]