from .zpk import ZPKOperatorLearningModule, LLMZPKWarmStartOperatorLearningModule
from .llm_plus import LLMZPKIterativeOperatorLearningModule, LLMZPKOperatorLearningModule
from .foldt import FOLDTOperatorLearningModule
from .groundtruth import GroundTruthOperatorLearningModule

def create_operator_learning_module(operator_learning_name, learned_operators, domain_name, llm, llm_precondition_goal_ops, planning_ops, log_llm_path):
    if operator_learning_name.startswith("groundtruth"):
        env_name = operator_learning_name[len("groundtruth-"):]
        return GroundTruthOperatorLearningModule(env_name, learned_operators)
    if operator_learning_name == "LNDR":
        return ZPKOperatorLearningModule(learned_operators, domain_name, planning_ops)
    if operator_learning_name == "LLMWarmStart+LNDR":
        return LLMZPKWarmStartOperatorLearningModule(learned_operators, domain_name, llm, planning_ops)
    if operator_learning_name == "LLMIterative+LNDR":
        return LLMZPKIterativeOperatorLearningModule(learned_operators, domain_name, llm, llm_precondition_goal_ops, planning_ops, log_llm_path)
    if operator_learning_name == "LLM+LNDR":
        return LLMZPKOperatorLearningModule(learned_operators, domain_name, llm, llm_precondition_goal_ops, planning_ops, log_llm_path)
    if operator_learning_name == "TILDE":
        return FOLDTOperatorLearningModule(learned_operators)
    raise Exception("Unrecognized operator learning module '{}'".format(operator_learning_name))
