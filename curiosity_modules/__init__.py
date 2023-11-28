from .curiosity_base import BaseCuriosityModule
from .oracle_curiosity import OracleCuriosityModule
from .random_actions import RandomCuriosityModule
from .llm_explorer import LLMOracle
from .GLIB_grounded import *
from .GLIB_lifted import *
from .replay_sampled_lifted_sequences import *


def create_curiosity_module(
    curiosity_module_name,
    action_space,
    observation_space,
    planning_module,
    learned_operators,
    operator_learning_module,
    domain_name,
    experiment_log_path,
):
    module = None
    if curiosity_module_name == "oracle":
        module = OracleCuriosityModule
    elif curiosity_module_name == "random":
        module = RandomCuriosityModule
    elif curiosity_module_name == "GLIB_G1":
    # Grounded is not supported for replay
        module = GLIBG1CuriosityModule
    elif curiosity_module_name == "GLIB_L2":
    # Lifted needs more support for replay
        module = GLIBL2CuriosityModule
    elif curiosity_module_name == "GLIB_Seq":
        module = GLIBLSCuriosityModule
    elif curiosity_module_name == "LLMOracle":
        module = LLMOracle
    else:
        raise Exception(
            "Unrecognized curiosity module or module not supported for logging yet '{}'".format(
                curiosity_module_name
            )
        )
    return module(
        action_space,
        observation_space,
        planning_module,
        learned_operators,
        operator_learning_module,
        domain_name,
        experiment_log_path,
    )
