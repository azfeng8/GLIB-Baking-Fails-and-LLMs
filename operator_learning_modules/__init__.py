from .zpk import ZPKOperatorLearningModule, LLMZPKWarmStartOperatorLearningModule
from .llm_plus import LLMZPKIterativeOperatorLearningModule, LLMZPKOperatorLearningModule
from .foldt import FOLDTOperatorLearningModule
from .groundtruth import GroundTruthOperatorLearningModule

def create_operator_learning_module(operator_learning_name, planning_operators, learned_operators, domain_name, llm, llm_precondition_goal_ops, log_llm_path):
    if operator_learning_name.startswith("groundtruth"):
        env_name = operator_learning_name[len("groundtruth-"):]
        return GroundTruthOperatorLearningModule(env_name, planning_operators)
    if operator_learning_name == "LNDR":
        return ZPKOperatorLearningModule(planning_operators, learned_operators, domain_name)
    if operator_learning_name == "LLMWarmStart+LNDR":
        return LLMZPKWarmStartOperatorLearningModule(planning_operators, learned_operators, domain_name, llm)
    if operator_learning_name == "LLMIterative+LNDR":
        raise NotImplementedError("Learning and planning ops are not separate")
        return LLMZPKIterativeOperatorLearningModule(planning_operators, domain_name, llm, llm_precondition_goal_ops, log_llm_path)
    if operator_learning_name == "LLM+LNDR":
        raise NotImplementedError("Learning and planning ops are not separate")
        return LLMZPKOperatorLearningModule(planning_operators, domain_name, llm, llm_precondition_goal_ops, log_llm_path)
    if operator_learning_name == "TILDE":
        return FOLDTOperatorLearningModule(planning_operators)
    if operator_learning_name == "LLMIterative+ZPK":
        raise NotImplementedError("Learning and planning ops are not separate")
        return LLMZPKIterativeOperatorLearningModule(planning_operators, domain_name, llm, llm_precondition_goal_ops, log_llm_path)
    raise Exception("Unrecognized operator learning module '{}'".format(operator_learning_name))
