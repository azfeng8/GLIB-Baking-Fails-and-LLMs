import pddlgym
import gym
from pddlgym.structs import Predicate, Exists, State
from pddlgym.parser import PDDLProblemParser
from settings import AgentConfig as ac

import random
import abc
import os


class Planner:
    def __init__(self, env):
        self.domain_name = 'bakingrealistic'
        self._action_space = env.action_space
        self._observation_space = env.observation_space
        self._predicates = {p.name : p \
            for p in env.observation_space.predicates + env.action_space.predicates}
        self._types = {str(t) : t for p in self._predicates.values() for t in p.var_types}
        self._problem_files = {}
        self.operators = set(env.domain.operators.values())

    @abc.abstractmethod
    def get_policy(self, raw_problem_fname, use_learned_ops=False):
        pass

    def _create_domain_file(self, use_learned_ops=False):
        dom_str = self._create_domain_file_header()
        dom_str += self._create_domain_file_types()
        dom_str += self._create_domain_file_predicates()

        for operator in sorted(self.operators, key=lambda o:o.name):
            dom_str += self._create_domain_file_operator(operator)
        dom_str += '\n)'

        return self._create_domain_file_from_str(dom_str)

    def _create_domain_file_header(self):
        return """(define (domain {})\n\t(:requirements :strips :typing)\n""".format(
            self.domain_name.lower())

    def _create_domain_file_types(self):
        return "\t(:types " + " ".join(self._types.values()) + ")\n"

    def _create_domain_file_predicates(self):
        preds_pddl = []
        for pred in self._predicates.values():
            if pred.name in ('name-less-than',):
                continue
            var_part = []
            for i, var_type in enumerate(pred.var_types):
                var_part.append("?arg{} - {}".format(i, var_type))
            preds_pddl.append("\t\t({} {})".format(pred.name, " ".join(var_part)))
        for t in self._types:
            diff_pred = f"\t\t(different ?arg0 - {t} ?arg1 - {t})"
            if diff_pred not in preds_pddl:
                preds_pddl.append(diff_pred)
        return """\t(:predicates\n{}\n\t)""".format("\n".join(preds_pddl))

    def _create_domain_file_operator(self, operator):
        param_strs = [str(param).replace(":", " - ") for param in operator.params]
        dom_str = "\n\n\t(:action {}".format(operator.name)
        dom_str += "\n\t\t:parameters ({})".format(" ".join(param_strs))
        preconds_pddl_str = self._create_preconds_pddl_str(operator.preconds, operator.name)
        dom_str += "\n\t\t:precondition (and {})".format(preconds_pddl_str)
        indented_effs = operator.effects.pddl_str().replace("\n", "\n\t\t")
        dom_str += "\n\t\t:effect {}".format(indented_effs)
        dom_str += "\n\t)"
        return dom_str

    def _create_preconds_pddl_str(self, preconds, name):
        all_params = set()
        precond_strs = []
        for term in preconds.literals:
            params = set(map(str, term.variables))
            if term.negated_as_failure:
                # Negative term. The variables to universally
                # quantify over are those which we have not
                # encountered yet in this clause.
                universally_quantified_vars = list(sorted(
                    params-all_params))
                precond = ""
                for var in universally_quantified_vars:
                    precond += "(forall ({}) ".format(
                        var.replace(":", " - "))
                precond += "(or "
                precond += "(not {}))".format(term.positive.pddl_str())
                for var in universally_quantified_vars:
                    precond += ")"
                if precond not in precond_strs:
                    precond_strs.append(precond)
            else:
                # Positive term.
                if 'name-less-than' in term.pddl_str():
                    continue
                all_params.update(params)
                if term.pddl_str() not in precond_strs:
                    precond_strs.append(term.pddl_str())
        return "\n\t\t\t".join(precond_strs)

    def _create_domain_file_from_str(self, dom_str):
        filename = "dom.pddl"
        with open(filename, 'w') as f:
            f.write(dom_str)
        return filename

    def _create_problem_file(self, raw_problem_fname, use_cache=True):
        if (not use_cache) or (raw_problem_fname not in self._problem_files):
            problem_fname = raw_problem_fname

            # Parse raw problem
            action_names = []  # purposely empty b/c we WANT action literals in there
            problem_parser = PDDLProblemParser(
                raw_problem_fname, self.domain_name.lower(), self._types,
                self._predicates, action_names)

            # Add action literals (in case they're not already present in the initial state)
            # which will be true when the original domain uses operators_as_actions
            init_state = State(problem_parser.initial_state, problem_parser.objects, None)
            act_lits = self._action_space.all_ground_literals(init_state, valid_only=False)

            initial_state = set()
            for lit in problem_parser.initial_state:
                if any(n in lit.predicate.name for n in ('name-less-than',)):
                    continue
                initial_state.add(lit)

            problem_parser.initial_state = frozenset(act_lits | initial_state)
            # Add 'Different' pairs for each pair of objects
           # Write out new temporary problem file
            problem_parser.write(os.path.basename(problem_fname), fast_downward_order=True)

            # Add to cache
            self._problem_files[raw_problem_fname] = (problem_fname, problem_parser.objects)

        return self._problem_files[raw_problem_fname]


env = pddlgym.make("PDDLEnvBakingrealistic-v0")
p = Planner(env)
p._create_domain_file()
for idx in range(len(env.problems)):
    env.fix_problem_index(idx)
    _, debug_info = env.reset()
    print(debug_info)
    p._create_problem_file(debug_info['problem_file'])

