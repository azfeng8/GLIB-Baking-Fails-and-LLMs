from .zpk import ZPKOperatorLearningModule, LLMZPKOperatorLearningModule
from .llm_plus import LLMZPKIterativeOperatorLearningModule
from .foldt import FOLDTOperatorLearningModule
from .groundtruth import GroundTruthOperatorLearningModule

<<<<<<< HEAD
def create_operator_learning_module(operator_learning_name, learned_operators, domain_name, llm, llm_precondition_goal_ops, log_llm):
=======
def create_operator_learning_module(operator_learning_name, learned_operators, domain_name, llm, llm_precondition_goal_ops, planning_ops, log_llm_path):
>>>>>>> fc0ccb3 (update primitive logging for openstack)
    if operator_learning_name.startswith("groundtruth"):
        env_name = operator_learning_name[len("groundtruth-"):]
        return GroundTruthOperatorLearningModule(env_name, learned_operators)
    if operator_learning_name == "LNDR":
        return ZPKOperatorLearningModule(learned_operators, domain_name)
    if operator_learning_name == "LLM+LNDR":
        return LLMZPKOperatorLearningModule(learned_operators, domain_name, llm, planning_ops)
    if operator_learning_name == "TILDE":
        return FOLDTOperatorLearningModule(learned_operators)
    if operator_learning_name == "LLMIterative+ZPK":
<<<<<<< HEAD
<<<<<<< HEAD
        return LLMZPKIterativeOperatorLearningModule(learned_operators, domain_name, llm, llm_precondition_goal_ops)
=======
        return LLMZPKIterativeOperatorLearningModule(learned_operators, domain_name, llm, llm_precondition_goal_ops, log_llm)
>>>>>>> df99738 (Add argument parsing for logging LLM and plotting options)
=======
        return LLMZPKIterativeOperatorLearningModule(learned_operators, domain_name, llm, llm_precondition_goal_ops, planning_ops, log_llm_path)
>>>>>>> fc0ccb3 (update primitive logging for openstack)
    raise Exception("Unrecognized operator learning module '{}'".format(operator_learning_name))
