import logging
from collections import defaultdict
import itertools
from pddlgym.structs import Anti, Type, LiteralConjunction, Literal,TypedEntity, Not, LiteralDisjunction, Predicate
from pddlgym.parser import Operator
from ndr.learn import iter_variable_names
from itertools import product
from typing import Optional
import re
from typing import Optional

PARSING_LOGGER = logging.getLogger('PARSER')

class LLM_PDDL_Parser:
    def __init__(self, action_preds:dict, observation_preds:dict, object_types:set):
        """Parser for PDDL from LLMs. Use PDDLGym environment-specific predicates to do syntax error detection / correction.

        Args:
            action_preds (dict[str, pddlgym.structs.Predicate]): dict from name to predicate
            observation_preds (dict[str, pddlgym.structs.Predicate]):  dict from name to predicate
            object_types (set[str]): types of objects
        """
        self._observation_predicates = observation_preds
        self._action_predicates = action_preds
        self._types = object_types
    
    def _purge_comments(self, pddl_str):
        # Purge comments from the given string.
        while True:
            match = re.search(r";(.*)\n", pddl_str)
            if match is None:
                return pddl_str
            start, end = match.start(), match.end()
            pddl_str = pddl_str[:start]+pddl_str[end-1:]

    def parse_operators(self, llm_response:str, parse_action_using_op_name=True) -> list[Operator]:
        matches = re.finditer(r"\(:action", llm_response)
        ops = []
        for match in matches:
            start_ind = match.start()
            op_str = find_closing_paran(llm_response[start_ind:])
            if op_str is None:
                continue
            op_str = op_str.strip()
            op_str = self._purge_comments(op_str)
            ops_to_add = self._parse_operators(op_str, parse_action_using_op_name=parse_action_using_op_name)
            if ops_to_add is not None:
                ops.extend(ops_to_add)
        return ops
 
    def _parse_operators(self, operator_str:str, parse_action_using_op_name=True) -> Optional[list[Operator]]:
        """Parse an Operator from the LLM response.

        Args:
            llm_response (str)

        Raises:
            Exception: Used in debugging only, will remove #TODO.

        Returns:
            list[Operator] or None: operators that were parsed, or None if not able to parse a non-null-effect operator.
        """
        # Extract operator name.
        match = re.search("\(\:action\s\w+", operator_str)
        patt = r"\(:action(.*):parameters(.*):precondition(.*):effect(.*)\)"
        op_match = re.match(patt, operator_str, re.DOTALL)
        if op_match is None:
            return None
        op_name, _, _, _ = op_match.groups()
        op_name = op_name.strip()

        # Extract parameters.
        match = re.search("\:parameters[^\)]*\)", operator_str)
        param_str = operator_str[match.start() + len(":parameters"): match.end()]
        param_names:list[str] = []
        param_types:list[str] = []
        for arg_match in re.finditer("[\w\?]+\s-\s[\w\?]+", param_str):
            arg = param_str[arg_match.start(): arg_match.end()] 
            name, var_type = arg.split(' - ')
            name = name.strip()
            var_type = var_type.strip()
            if var_type in self._types:
                param_names.append(name)
                param_types.append(var_type)

        # Extract preconditions.
        precond_match = re.search(":precondition[\(\s]*\w", operator_str)
        if precond_match is None:
            # No preconditions found.
            precond_list = [[]]
        else:
            # Get rid of space between ":effect (" and the first word such as "and" or a predicate name
            operator_str = operator_str[:precond_match.end() - 1].strip() + operator_str[precond_match.end() - 1:]

            precond_match = re.search(":precondition[\(\s]*\w", operator_str)
            precond_str = find_closing_paran(operator_str[precond_match.end() - 2:])
            #NOTE: Supports only LiteralConjunction and Literal for now.
            precond_list = self._parse_into_cnf(precond_str, param_names, param_types, False)
            PARSING_LOGGER.debug(f"PRECONDS: {precond_list}")

            if not all(precond_list):
                return None

            for i in range(len(precond_list)):
                lc = precond_list[i]
                if isinstance(lc, Literal):
                    precond_list[i] = LiteralConjunction([lc])
                elif isinstance(lc, LiteralConjunction):
                    precond_list[i] = LiteralConjunction(lc.literals)
                else:
                    raise Exception(f"Unsupported type: {type(lc)}")    

        # Extract effects.
        effect_str_match = re.search(":effect[\(\s]*\w", operator_str)
        if effect_str_match is None:
            # Don't consider operators with no effects.
            return None
        # Get rid of space between ":effect (" and the first word such as "and" or a predicate name
        operator_str = operator_str[:effect_str_match.end() - 1].strip() + operator_str[effect_str_match.end() - 1:]
        effect_str_match = re.search(":effect[\(\s]*\w", operator_str)
        effect_str = (operator_str[effect_str_match.end() - 2:].strip())
        effect_str = find_closing_paran(effect_str)

        effects_list = self._parse_into_cnf(effect_str, param_names, param_types, is_effect=True)
        PARSING_LOGGER.debug(f"EFFECTS:\n{effects_list}")

        if not all(effects_list):
            return None

        for j in range(len(effects_list)):
            effects = effects_list[j]
            if isinstance(effects, Literal):
                effects_list[j]  = LiteralConjunction([effects])
            elif isinstance(effects, LiteralConjunction):
                pass
            else:
                raise Exception(f"Unsupported type: {type(effects)}")    


        operators = []
        for effects in effects_list:
            for preconds in precond_list:
                # Rename the variables
                var_name_gen = iter_variable_names()
                variables = {}
                # PARSING_LOGGER.debug(f"effects: {effects}")
                # PARSING_LOGGER.debug(f"preconds: {preconds}")
                for l in effects.literals + preconds.literals:
                    for v in l.variables:
                        if v not in variables:
                            v_name = next(var_name_gen)
                            variables[v] = Type(v_name)

                literals = []
                for l in preconds.literals:
                    args = []
                    for v in l.variables:
                        args.append(variables[v])
                    literals.append(Literal(l.predicate, args))
                preconds_renamed = LiteralConjunction(literals)

                literals = []
                for l in effects.literals:
                    args = []
                    for v in l.variables:
                        args.append(variables[v])
                    literals.append(Literal(l.predicate, args))
                effects_renamed = LiteralConjunction(literals)       

                params = set()
                for l in effects_renamed.literals + preconds_renamed.literals:
                    for v in l.variables:
                        params.add(v)
                        
                operators.append(Operator(op_name, params, preconds_renamed, effects_renamed))

        if parse_action_using_op_name:
            if op_name not in self._action_predicates:
                return None
            # Get several operators with different action groundings
            operators = create_final_operators([(op, op_name) for op in operators], list(self._action_predicates.values()))

        return operators

    def _find_all_balanced_expressions(self, string):
        """Return a list of all balanced expressions in a string,
        starting from the beginning.

        Source: parser.py in pddlgym PDDLParser.
        """
        if not string:
            return []
        if string[0] != "(": return None
        if string[-1] != ")": return None
        exprs = []
        index = 0
        start_index = index
        balance = 1
        while index < len(string)-1:
            index += 1
            if balance == 0:
                exprs.append(string[start_index:index])
                # Jump to next "(".
                while index < len(string) - 1:
                    if string[index] == "(":
                        break
                    index += 1
                start_index = index
                balance = 1
                continue
            symbol = string[index]
            if symbol == "(":
                balance += 1
            elif symbol == ")":
                balance -= 1
        assert balance == 0
        exprs.append(string[start_index:index+1])
        return exprs

    def _parse_into_cnf(self, string:str, param_names:list, param_types:list, is_effect:bool) -> list[Optional[Literal]]:
        """Parses the string into a CNF or None if predicate name or argument types are invalid.

        Args:
            string: Effect or Precondition string. Starts with '(' and ends with its closing mirror ')'
            is_effect (bool): if the string is effect.
            param_names (list): variable names (such as '?x0') 
            param_types (list): types (such as 'ingredient')
        Returns:
            [ Literal or None ]: lists of the literals. Each literal in the list is an item in the Disjunction, like a CNF: [ [AND] OR [AND] ].
        """
        if string.startswith("(and") and string[4] in (" ", "\n", "(", ")"):
            clauses = self._find_all_balanced_expressions(string[4:-1].strip())
            if clauses is None:
               return [None]
            clauses = clauses
            lits_list = [self._parse_into_cnf(clause, param_names, param_types, 
                                        is_effect=is_effect) for clause in clauses]
            clauses_to_and = []
            # AND together the cnfs in `lits_list`
            for _cnf in lits_list:
                # Clear out empty clauses
                _cnf = [l for l in _cnf if l is not None]
                if len(_cnf) != 0:
                    # itertools.product needs iterables, so create `final_cnf` which is just making the clauses in cnf into iterables if they aren't already.
                    clauses_to_and.append(_cnf)

            if len(clauses_to_and) == 0:
                return [None]

            if len(clauses_to_and) == 1:
                # [ expression ]
                if isinstance(clauses_to_and[0], list):
                    return clauses_to_and[0]
                elif isinstance(clauses_to_and[0], Literal) or isinstance(cnf[0], LiteralConjunction):
                    return clauses_to_and
                else: 
                    raise Exception(f"Got type unexpected: {cnf}")
                    
            lcs = []
            
            for lits in product(*clauses_to_and):
                conj = []
                for l in lits:
                    if isinstance(l, LiteralConjunction):
                        conj.extend(l.literals)
                    elif isinstance(l, Literal):
                        conj.append(l)
                    else:
                        raise Exception(f"Got unexpected type: {l} in {lits}")
                lcs.append(LiteralConjunction(list(conj)))
            return lcs

        if string.startswith("(not") and string[4] in (" ", "\n", "("):
            clause = string[4:-1].strip()
            # the list contains a LiteralConjunction or literals (Disjunction)
            lits = self._parse_into_cnf(clause, param_names, param_types, is_effect=is_effect)

            lits = [l for l in lits if l is not None]
            if len(lits) == 0:
                return [None]
            
            # DeMorgan's: Push in the negation
            if is_effect:
                negated_lits = [rAnti(l) for l in lits]
            else:
                negated_lits = [Not(l) for l in lits]

            if len(lits) == 1:
                if isinstance(lits[0], LiteralConjunction):
                # Conjunction at the top turns into a Disjunction
                    return negated_lits[0].literals
                elif isinstance(lits[0], Literal):
                    return negated_lits
                else:
                    raise Exception(f"Got unexpected type {lits[0]}")
            else:
                # Disjunction at the top turns into a Conjunction
                # [ [OR] AND [OR] AND [OR] ]
                conjunction = []
                for nl in negated_lits:
                    if isinstance(nl, LiteralDisjunction):
                        conjunction.append(nl.literals)
                    elif isinstance(nl, LiteralConjunction):
                        for l in nl.literals:
                            conjunction.append([l])
                    elif isinstance(nl, Literal): # Literal
                        conjunction.append([nl])
                    else:
                        raise Exception(f"Got unexpected type {lits[0]}")
                if len(conjunction) == 1:
                    # A Conjunction of 1 literal
                    return conjunction[0]

                # Turn [ [OR] AND [OR] AND [OR] ] into CNF: [ [AND] OR [AND] ]
                cnf = []
                for lits in product(*conjunction):
                    cnf.append(list(lits))
                return [LiteralConjunction(clause) for clause in cnf]

        if string.startswith("(or") and string[3] in (" ", "\n", "(", ")"):
            clauses = self._find_all_balanced_expressions(string[3:-1].strip())
            lits_list = [self._parse_into_cnf(clause, param_names, param_types, is_effect=is_effect) for clause in clauses]
            # OR the AND clauses
            disjunctions = []
            for _cnf in lits_list:
                _cnf = [l for l in _cnf if l is not None]
                for clause in _cnf:
                    if isinstance(clause, list):
                        disjunctions.extend(clause)
                    else:
                        disjunctions.append(clause)

            if len(disjunctions) == 0:
                return [None]
            return disjunctions
 
        string = string[1:-1].split()
        pred, args = string[0], string[1:]
        typed_args = []

        # Validate types against the given param names.
        if pred not in self._observation_predicates:
            PARSING_LOGGER.debug(f"Parsed unknown predicate {pred}")
            return [None]
        if len(args) != self._observation_predicates[pred].arity:
            PARSING_LOGGER.debug(f"Parsed incongruent number of argument types for predicate {pred}")
            return [None]

        arg_types = []
        for i, arg in enumerate(args):
            if arg not in param_names:
                PARSING_LOGGER.debug("Argument {} not in params {}".format(arg, param_names))
                return [None]
            t = param_types[param_names.index(arg)]
            typed_arg = TypedEntity(arg, Type(t))
            arg_types.append(t)
            typed_args.append(typed_arg)

        if self._observation_predicates[pred].var_types != arg_types:
            PARSING_LOGGER.debug(f"Parsed incongruent argument types for predicate {pred}")
            return [None]

        return [self._observation_predicates[pred](*typed_args)]

