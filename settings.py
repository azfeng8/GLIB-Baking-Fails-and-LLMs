"""Settings used throughout the directory.
"""


class EnvConfig:
    """Environment-specific constants.
    """
    # domain_name = ["Rearrangement"]
    domain_name = ["Minecraft"]
    # domain_name = ["Baking"]
    # domain_name = ['Travel']
    # domain_name = ["Rearrangement","Baking", "Minecraft", "Travel"]
    
    # Number of test problems. Only needed for non-PDDLGym envs.
    num_test_problems = {}

    # Number of transitions to use for variational distance computation.
    num_var_dist_trans = {
        'Baking': 1000,
        "Blocks": 1000,
        "Glibblocks": 1000,
        "Tsp": 1000,
        "Rearrangement": 4000,
        "Glibrearrangement": 1000,
        "Easygripper": 1000,
        "Gripper": 1000,
        "Doors": 1000,
        "Glibdoors": 1000,
        "Tireworld": 1000,
        "Explodingblocks": 1000,
        "River": 1000,
        "NDRBlocks": 100,
        "Minecraft": 80000,
        "Travel": 1000
    }


class AgentConfig:
    """Agent-specific constants.
    """
    curiosity_methods_to_run = [
        "LLM+GLIB_L2",
        "LLM+GLIB_G1",
        # "GLIB_L2",
        # "GLIB_G1",
        # "oracle",
        # "random",
    ]

    # learning_name = "LNDR"
    # learning_name = "LLMIterative+LNDR"
    # learning_name = "LLMWarmStart+LNDR"
    learning_name = "LLM+LNDR"

    planner_name = {
        "Blocks": "ff",
        "Glibblocks": "ff",
        "Tsp": "ff",
        "Rearrangement": "ff",
        "Easygripper": "ff",
        "Gripper": "ff",
        "Glibrearrangement": "ff",
        "Doors": "ff",
        "Glibdoors": "ff",
        "NDRBlocks": "ffreplan",
        "Tireworld": "ffreplan",
        "Explodingblocks": "ffreplan",
        "River": "ffreplan",
        "Baking": "ff",
        "Minecraft": "ff",
        "Travel": "ff"
    }

    # Maximum trajectory length
    max_traj_len = 10

    # How often to learn operators.
    learning_interval = {
        "Blocks": 1,
        "Glibblocks": 1,
        "Tsp": 1,
        "Rearrangement": 1,
        "Glibrearrangement": 1,
        "Easygripper": 1,
        "Gripper": 1,
        "Doors": 1,
        "Glibdoors": 1,
        "Tireworld": 10,
        "Explodingblocks": 10,
        "River": 10,
        "NDRBlocks": 25,
        "Baking": 1,
        "Minecraft": 1,
        "Travel": 1
    }

    # How often to use the LLM to learn operators. Interval units are iterations.
    LLM_learn_interval = {
        "Baking": 300,
        "Minecraft": 300,
        "Travel": 300,
        "Rearrangement": 300,
        "Glibdoors": 300,
    }
    LLM_trajectory_length = {
        "Baking": 10,
        "Minecraft": 10,
        "Travel": 10,
        "Rearrangement": 15,
        "Glibdoors": 10
    }
    LLM_start_interval = {
        "Baking": 50,
        "Minecraft": 50,
        "Travel": 50,
    }

    # Max training episode length.
    max_train_episode_length = {
        "Blocks": 25,
        "Glibblocks": 25,
        "Tsp": 25,
        "Rearrangement": 25,
        "Glibrearrangement": 25,
        "Easygripper": 25,
        "Gripper": 25,
        "Doors": 25,
        "Glibdoors": 25,
        "Tireworld": 8,
        "Explodingblocks": 25,
        "River": 25,
        "PybulletBlocks" : 10,
        "NDRBlocks" : 25,
        "Baking": 25,
        "Minecraft": 30,
        "Travel": 35
    }
    # Max test episode length.
    max_test_episode_length = {
        "Blocks": 25,
        "Glibblocks": 25,
        "Tsp": 25,
        "Rearrangement": 25,
        "Glibrearrangement": 25,
        "Easygripper": 100,
        "Gripper": 100,
        "Doors": 25,
        "Glibdoors": 25,
        "Tireworld": 25,
        "Explodingblocks": 25,
        "River": 25,
        "PybulletBlocks" : 25,
        "NDRBlocks" : 25,
        "Baking": 25,
        "Travel": 25,
        "Minecraft": 25
    }
    # Timeout for planner.
    planner_timeout = None  # set in main.py

    # Number of training iterations.
    num_train_iters = {
        "Blocks": 501,
        "Glibblocks": 501,
        "Tsp": 501,
        "Rearrangement": 1001,
        "Glibrearrangement": 1501,
        "Easygripper": 3001,
        "Gripper": 3001,
        "Doors": 2501,
        "Glibdoors": 2501,
        "Tireworld": 401,
        "Explodingblocks": 501,
        "River": 1001,
        "PybulletBlocks" : 501,
        "NDRBlocks" : 1501,
        "Baking": 1799,
        "Travel": 1501,
        "Minecraft": 1799
    }

    ## Constants for curiosity modules. ##
    max_sampling_tries = 100
    max_planning_tries = 50
    oracle_max_depth = 2

    ## Constants for mutex detection. ##
    mutex_num_episodes = {
        "Blocks": 35,
        "Glibblocks": 35,
        "Tsp": 10,
        "Rearrangement": 35,
        "Glibrearrangement": 35,
        "Easygripper": 35,
        "Gripper": 35,
        "Doors": 35,
        "Glibdoors": 35,
        "Tireworld": 35,
        "Explodingblocks": 35,
        "River": 35,
        "PybulletBlocks": 35,
        "NDRBlocks": 35,
        "Baking": 35,
        "Travel": 35,
        "Minecraft": 35
    }
    mutex_episode_len = {
        "Blocks": 35,
        "Glibblocks": 35,
        "Tsp": 10,
        "Rearrangement": 35,
        "Glibrearrangement": 35,
        "Easygripper": 35,
        "Gripper": 35,
        "Doors": 35,
        "Glibdoors": 35,
        "Tireworld": 35,
        "Explodingblocks": 35,
        "River": 35,
        "PybulletBlocks": 35,
        "NDRBlocks": 35,
        "Baking": 35,
        "Minecraft": 35,
        "Travel": 35
    }
    mutex_num_action_samples = 10

    ## Constants for TILDE (also called FOLDT throughout code). ##
    max_foldt_feature_length = 10e8
    max_foldt_learning_time = 180
    max_foldt_exceeded_strategy = "fail" # 'fail' or 'early_stopping' or 'pdb'

    ## Constants for LNDR (also called ZPK throughout code). ##
    max_zpk_learning_time = 180
    max_zpk_explain_examples_transitions = {
        "Blocks": 25,
        "Glibblocks": 25,
        "Tsp": 25,
        "Rearrangement": 25,
        "Glibrearrangement": 25,
        "Easygripper": 25,
        "Gripper": 25,
        "Doors": 25,
        "Glibdoors": 25,
        "Tireworld": float("inf"),
        "Explodingblocks": 25,
        "River": 25,
        "PybulletBlocks": float("inf"),
        "NDRBlocks": float("inf"),
        "Baking": 25,
        "Minecraft": 25,
        "Travel": 25
    }
    max_zpk_action_batch_size = {
        "Blocks": None,
        "Glibblocks": None,
        "Tsp": None,
        "Rearrangement": None,
        "Glibrearrangement": None,
        "Easygripper": None,
        "Gripper": None,
        "Doors": None,
        "Glibdoors": None,
        "Tireworld": None,
        "Explodingblocks": None,
        "River": None,
        "PybulletBlocks": None,
        "NDRBlocks": None,
        "Baking": None,
        "Minecraft": None,
        "Travel": None
    }
    zpk_initialize_from_previous_rule_set = {
        "Blocks": False,
        "Glibblocks": False,
        "Tsp": False,
        "Rearrangement": False,
        "Glibrearrangement": False,
        "Easygripper": False,
        "Gripper": False,
        "Doors": False,
        "Glibdoors": False,
        "Tireworld": True,
        "Explodingblocks": False,
        "River": False,
        "PybulletBlocks": False,
        "NDRBlocks": False,
        "Baking": False,
        "Minecraft": False,
        "Travel": False
    }

    # Major hacks. Only used by oracle_curiosity.py and LLM methods.
    train_env = None


class GeneralConfig:
    """General configuration constants.
    """
    verbosity = 1
    start_seed = 40
    num_seeds = 1
    vardisttrans_dir = 'data/'
    results_dir = 'results/'
    timings_dir = results_dir + 'timings/'
    planning_results_dir = results_dir + "planning_results/"

class LLMConfig:
    """LLM Configuration."""
    model = "gpt-4"
    cache_dir = "./llm_cache"
    iterative_log_path = 'llm_iterative_log'
    max_tokens = 4096

class PlottingConfig:
    """Plotting from cached results.
    """
    # learner_explorer = [("LLMIterative+LNDR", "LLM+GLIB_G1"), ("LNDR", "GLIB_G1")]
    # learner_explorer = [("LLMIterative+LNDR", "LLM+GLIB_L2"), ("LNDR", "GLIB_L2")]
    
    # learner_explorer = [("LLMWarmStart+LNDR", "GLIB_G1"), ("LNDR", "GLIB_G1"), ("LLMWarmStart+LNDR", "GLIB_L2"), ("LNDR", "GLIB_L2"),]
    # learner_explorer = [("LLMWarmStart+LNDR", "GLIB_L2"), ("LNDR", "GLIB_L2")]

    # learner_explorer = [  ("LNDR", "GLIB_L2"), ("LNDR", "GLIB_G1"), ("LNDR", "random")]

    # learner_explorer = [("LLM+LNDR", "LLM+GLIB_L2"), ("LNDR", "GLIB_L2")]
    # learner_explorer = [("LLM+LNDR", "LLM+GLIB_L2")]#, ("LNDR", "GLIB_G1")]
    learner_explorer = [("LLMWarmStart+LNDR", "GLIB_L2"),  ("LNDR", "GLIB_L2")]
    learner_explorer = [("LLMWarmStart+LNDR", "GLIB_L1"),  ("LNDR", "GLIB_L1")]
    # learner_explorer = [("LLMWarmStart+LNDR", "GLIB_G1"), ("LNDR", "GLIB_G1")]
    # seeds = [range(60, 70)]  + [range(1, 11)]

    # learner_explorer= [("LNDR", "GLIB_L2"), ("LNDR", "GLIB_G1"), ("LNDR", "random")]
    # learner_explorer = [("LNDR", "random")]
    # seeds = [range(1, 11), range(1,11)]
    # seeds = [range(50,60), range(50,60)]
    # seeds = [range(60,70), range(60,70)]
    # seeds = [range(70,80), range(70,80)]
    # seeds = [range(150, 160)] + [range(100, 110)]
    seeds = [range(160, 170), range(160, 170)]

    # learner_explorer = [("LLM+LNDR", "LLM+GLIB_G1"), ("LLM+LNDR", "LLM+GLIB_L2"), ("LNDR", "GLIB_L2"), ("LNDR", "GLIB_G1")]
    # seeds = [range(12, 22)] * 2 + [range(1, 11)]  * 2

    # domains = ["Baking",  "Minecraft", "Travel"]
    # domains = ["Baking", "Rearrangement", "Travel", "Minecraft", "Glibblocks", "Doors", "Easygripper"]
    # domains = ["Baking", "Minecraft", "Travel", "Blocks"]
    domains = ["Minecraft"]
