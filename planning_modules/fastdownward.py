from planning_modules.base_planner import Planner, PlannerTimeoutException, \
    NoPlanFoundException

from settings import AgentConfig as ac

from pddlgym.structs import ground_literal
import sys
import os
import re
import subprocess
import time

class FastDownwardPlanner(Planner):
    FD_PATH = os.environ['FD_PATH']
    def get_policy(self, raw_problem_fname, use_learned_ops=False):
        actions, _ = self.get_plan(raw_problem_fname, use_learned_ops)
        def policy(_):
            if len(actions) == 0:
                raise NoPlanFoundException() 
            return actions.pop(0)
        return policy
    
    def get_plan(self,  raw_problem_fname, use_learned_ops=False, use_cache=True):
        if (not use_learned_ops and not self._planning_operators) or (use_learned_ops and not self._learned_operators):
            raise NoPlanFoundException()
        domain_fname = self._create_domain_file(use_learned_ops)
        problem_fname, objects = self._create_problem_file(raw_problem_fname)
        cmd_str1, cmd_str2 = self._get_cmd_str(domain_fname, problem_fname)
        start_time = time.time()
        output = subprocess.getoutput(cmd_str1)
        if "exit code: 31" in output:
            print(output)
            raise Exception
        output = subprocess.getoutput(cmd_str2)
        end_time = time.time()
        if end_time - start_time > 0.9*ac.planner_timeout:
            self.delete_cached_plan_files(domain_fname, problem_fname, use_cache=True)
            raise PlannerTimeoutException()
        try:
            plan = self._output_to_plan(output)
        except Exception as e:
            self.delete_cached_plan_files(domain_fname, problem_fname, use_cache=True)
            raise e 
        actions, operator_names = self._plan_to_actions(plan, objects, domain_fname, use_learned_ops=use_learned_ops)
        self.delete_cached_plan_files(domain_fname, problem_fname, use_cache=True)
        return actions, operator_names

    def _get_cmd_str(self, domain_fname, problem_fname):
        timeout = "gtimeout" if sys.platform == "darwin" else "timeout"
        #FIXME: rename the output file so that it's unique, and so multiple seeds can run from the same folder
        return f"{timeout} {ac.planner_timeout} {self.FD_PATH} {domain_fname} {problem_fname}", f'{timeout} {ac.planner_timeout} {self.FD_PATH} --alias lama-first output.sas'

    @staticmethod
    def _output_to_plan(output):
        if not output.strip() or "Search stopped without finding a solution" in output:
            raise NoPlanFoundException()
        plan = [line[:-3].strip() for line in output.split('\n') if line.endswith('(1)')]
        return plan

    def _plan_to_actions(self, plan, objects, domain_fname, use_learned_ops=False):
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
            action, op_name = self.parse_plan_step(plan_step, operators, action_predicates, objects, domain_fname)
            actions.append(action)
            operator_names.append(op_name)
        return actions, operator_names

    def delete_cached_plan_files(self, domain_fname, problem_fname, use_cache=True):
        """Deletes domain and problem files so they don't persist."""
        os.remove(domain_fname)
        if not use_cache:
            os.remove(problem_fname)
        if os.path.exists('output.sas'):
            os.remove('output.sas')


    def parse_plan_step(self, plan_step, operators, action_predicates, objects, domain_file):
        plan_step_split = plan_step.split()

        # Get the operator from its name
        operator = None
        for op in operators:
            if op.name.lower() == plan_step_split[0]:
                operator = op
                break
        assert operator is not None, "Unknown operator '{}' in file {}".format(plan_step_split[0],domain_file)

        assert len(plan_step_split) == len(operator.params) + 1, f"{operator.pddl_str()}\nplan:{plan_step_split}"
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