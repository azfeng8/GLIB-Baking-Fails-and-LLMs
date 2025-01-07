from .zpk import ZPKOperatorLearningModule

def create_operator_learning_module(operator_learning_name, learned_operators, domain_name, rand_state):
    if operator_learning_name == "LNDR":
        return ZPKOperatorLearningModule(learned_operators, domain_name, rand_state)
    raise Exception("Unrecognized operator learning module '{}'".format(operator_learning_name))
