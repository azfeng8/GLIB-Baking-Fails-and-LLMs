from planning_modules.base_planner import Planner, PlannerTimeoutException, \
    NoPlanFoundException

from settings import AgentConfig as ac

from pddlgym.structs import ground_literal
import sys
import os
import re
import subprocess
import time


def parse_plan_step(plan_step, operators, action_predicates, objects):
    plan_step_split = plan_step.split()

    # Get the operator from its name
    operator = None
    for op in operators:
        if op.name.lower() == plan_step_split[0]:
            operator = op
            break
    assert operator is not None, "Unknown operator '{}'".format(plan_step_split[0])

    assert len(plan_step_split) == len(operator.params) + 1
    object_names = plan_step_split[1:]
    args = []
    for name in object_names:
        matches = [o for o in objects if o.name == name]
        assert len(matches) == 1
        args.append(matches[0])
    assignments = dict(zip(operator.params, args))

    for cond in operator.preconds.literals:
        if cond.predicate in action_predicates:
            ground_action = ground_literal(cond, assignments)
            return ground_action, operator.name

    import ipdb; ipdb.set_trace()
    raise Exception("Unrecognized plan step: `{}`".format(str(plan_step)))


class FastForwardPlanner(Planner):
    FF_PATH = os.environ['FF_PATH']

    def get_policy(self, raw_problem_fname, use_learned_ops=False):
        actions, _ = self.get_plan(raw_problem_fname, use_learned_ops)
        def policy(_):
            if len(actions) == 0:
                raise NoPlanFoundException() 
            return actions.pop(0)
        return policy

    def get_plan(self, raw_problem_fname, use_learned_ops=False, use_cache=True):
        # If there are no operators yet, we're not going to be able to find a plan
        if (not use_learned_ops and not self._planning_operators) or (use_learned_ops and not self._learned_operators):
            raise NoPlanFoundException()
        
        domain_fname = self._create_domain_file(use_learned_ops)
        problem_fname, objects = self._create_problem_file(raw_problem_fname, use_cache=use_cache)
        cmd_str = self._get_cmd_str(domain_fname, problem_fname)
        start_time = time.time()
        output = subprocess.getoutput(cmd_str)
        end_time = time.time()
        if end_time - start_time > 0.9*ac.planner_timeout:
            raise PlannerTimeoutException()
        plan = self._output_to_plan(output)
        print(domain_fname)
        os.remove(domain_fname)
        if not use_cache:
            os.remove(problem_fname)
 
        actions, operator_names = self._plan_to_actions(plan, objects)
        return actions, operator_names

    def _get_cmd_str(self, domain_fname, problem_fname):
        timeout = "gtimeout" if sys.platform == "darwin" else "timeout"
        return "{} {} {} -o {} -f {}".format(
            timeout, ac.planner_timeout, self.FF_PATH,
            domain_fname, problem_fname)

    @staticmethod
    def _output_to_plan(output):
        if not output.strip() or \
           "goal can be simplified to FALSE" in output or \
            "unsolvable" in output:
            raise NoPlanFoundException()
        plan = re.findall(r"\d+?: (.+)", output.lower())
        if not plan and "found legal" not in output and \
           "The empty plan solves it" not in output:
            raise Exception("Plan not found with FF! Error: {}".format(output))
        return plan

    def _plan_to_actions(self, plan, objects, use_learned_ops=False):
        if use_learned_ops:
            operators = self._learned_operators
        else:
            operators = self._planning_operators
        action_predicates = self._action_space.predicates

        actions = []
        operator_names = []
        for plan_step in plan:
            if plan_step == "reach-goal":
                continue
            action, op_name = parse_plan_step(plan_step, operators, action_predicates, objects)
            actions.append(action)
            operator_names.append(op_name)
        return actions, operator_names
