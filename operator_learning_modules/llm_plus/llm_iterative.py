from operator_learning_modules import ZPKOperatorLearningModule
from settings import AgentConfig as ac
from openai_interface import OpenAI_Model
from pddlgym.parser import Operator
from abc import abstractmethod

def _score(operators):
    """Score a set of operators."""
    pass

def _operator_search(operators):
    """Search over operator sets for best score."""
    pass

class BaseLLMIterativeOperatorLearningModule:
    def __init__(self, learned_operators, domain_name, llm):
        self._llm:OpenAI_Model = llm
        self._llm_learn_interval = ac.LLM_learn_interval[domain_name]
        self._learn_iter = 0

        # List of list of episodes
        self._trajectories = []
        
    def observe(self, state, action, effects):
        self.learner.observe(state, action, effects)
        # exclude no-ops


    def learn(self):
        self.learner.learn()
        if self._learn_iter % self._llm_learn_interval != 0:
            return
        self._learn_iter += 1

        # sample random episode. Use all the actions in the episode.
        traj = self._sample_trajectory()
        
        # LLM proposes new operator for each of the actions in the trajectory.
        ops = self._propose_operators(traj)

        # score and filter the PDDL operators
        ops.extend(self.learner.learned_operators)
        ops = self._score_and_filter(ops)

        # update learner operators
        self._update_operator_rep(ops)
            
    def _sample_trajectory(self):
        return

    def _propose_operators(self, transitions):
        """
        Args:
            transitions (list): list of (s,a,s') transitions in the trajectory
        """
        # translate the start state into natural language.
        prompt_start = """"""
        # translate the end state into natural language.
        prompt_goal = """"""

        # create the task decomposition 
        task_decomp = []

        operators = []
        for action in task_decomp:
            prompt_operator = """
            Given a successful plan from initial state to goal state, and predicates, propose a PDDL operator called ''.
            """
            response = self._llm.sample_completions()
            operators.append(self._parse_operator(response))
        return operators

    def _parse_operator(self, llm_response) -> Operator:
        pass

    def _score_and_filter(self, ops:list[Operator]) -> list[Operator]:
        pass

    @abstractmethod
    def update_operator_rep(self):
        raise NotImplementedError("Override me!")
        
class LLMZPKIterativeLearningModule(BaseLLMIterativeOperatorLearningModule):
    def __init__(self, learned_operators, domain_name, llm):
        self.learner = ZPKOperatorLearningModule(learned_operators, domain_name)

        super().__init__(learned_operators, domain_name, llm)

    def update_operator_rep(self, ops:list[Operator]):
        # update NDRs and learned_operators
        pass