"""Settings used throughout the directory.
"""


class EnvConfig:
    """Environment-specific constants.
    """
    domain_name = ["Baking"]#["Minecraft", "Rearrangement", "Travel", "Baking"]
    # domain_name = ['Tireworld']
    # domain_name = ["Glibdoors", "Tireworld", "Glibblocks", "Explodingblocks"]
    # domain_name = ["Gripper", "Travel"]
    
    # Number of test problems. Only needed for non-PDDLGym envs.
    num_test_problems = {}

    # Number of transitions to use for variational distance computation.
    num_var_dist_trans = {
        'Baking': 1000,
        "Blocks": 1000,
        "Glibblocks": 1000,
        "Tsp": 1000,
        "Rearrangement": 1000,
        "Glibrearrangement": 1000,
        "Easygripper": 1000,
        "Gripper": 1000,
        "Doors": 1000,
        "Glibdoors": 1000,
        "Tireworld": 1000,
        "Explodingblocks": 1000,
        "River": 1000,
        "NDRBlocks": 100,
        "Minecraft": 1000,
        "Travel": 1000
    }
    logging = True


class AgentConfig:
    """Agent-specific constants.
    """
    curiosity_methods_to_run = [
        # "LLM+GLIB_L2"
        # "GLIB_L2",
        "LLM+GLIB_G1"
        # "GLIB_G1",
        # "oracle",
        # "random",
        # "GLIB_Seq"
        # "LLMOracle"
    ]

    cached_results_to_load = [
        # "GLIB_L2",
        # "GLIB_G1",
        # "oracle",
        # "random",
    ]
    # learning_name = "TILDE"
    # learning_name = "LNDR"
    # learning_name = "LLM+LNDR"
    learning_name = "LLMIterative+ZPK"
    # learning_name = "groundtruth-PDDLEnv"+EnvConfig.domain_name+"-v0"

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

    # Random seed optionally used by curiosity modules.
    seed = 0
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
        "Baking": 300
    }
    LLM_trajectory_length = {
        "Baking": 10
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
        "Travel": 25
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
        "Minecraft": 100
    }
    # Timeout for planner.
    planner_timeout = None  # set in main.py

    # Number of training iterations.
    num_train_iters = {
        "Blocks": 501,
        "Glibblocks": 501,
        "Tsp": 501,
        "Rearrangement": 1501,
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
        "Minecraft": 2501
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

    # Major hacks. Only used by oracle_curiosity.py.
    train_env = None


class GeneralConfig:
    """General configuration constants.
    """
    verbosity = 1
    start_seed = 24
    num_seeds = 1   

class LLMConfig:
    """LLM Configuration."""
    model = "gpt-4"
    cache_dir = "/home/catalan/llm_cache"
    max_tokens = 4096

class PlottingConfig:
    """Plotting from cached results.
    """
    domains = ["Baking"]#,"Baking", "Glibblocks", ]
    learner_explorer = [("LLMIterative+ZPK", "LLM+GLIB_G1"), ("LNDR", "LLM+GLIB_G1"), ("LNDR", "GLIB_G1")]
