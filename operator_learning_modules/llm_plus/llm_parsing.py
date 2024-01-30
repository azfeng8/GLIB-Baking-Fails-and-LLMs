#TODO: support "or" and "forall"
import logging
from pddlgym.structs import Anti, Type, LiteralConjunction, Literal,TypedEntity, Not
from pddlgym.parser import Operator
from ndr.learn import iter_variable_names
import re

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

    def parse_operator(self, llm_response:str) -> Operator or None:
        """Parse an Operator from the LLM response.

        Args:
            llm_response (str)

        Raises:
            Exception: Used in debugging only, will remove #TODO.

        Returns:
            Operator or None: operator that was parsed, or None if not able to parse a non-null-effect operator.
        """
        # Find the PDDL operator in the response.
        match = re.search("\(\:action", llm_response)
        # Count parantheses: look for the closing to "(:action" to get the operator string.
        operator_str = find_closing_paran(llm_response[match.start():])

        if operator_str is None: raise Exception(f"Parsing error: {llm_response}")
        # Extract operator name.
        match = re.search("\(\:action\s\w+", operator_str)
        op_name = operator_str[match.start() + len("(:action "):match.end()]

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

        # NOTE: Propose a single operator for an action. Prompting the action multiple times will result in different operators.
        if op_name not in self._action_predicates:
            return None
        action_pred = self._action_predicates[op_name]
        args = []
        param_types_temp = param_types[:]
        for v_type in action_pred.var_types:
            if v_type not in param_types_temp:
                    # Can't form the action from the operator arguments
                    return None 
            i = param_types_temp.index(str(v_type))
            param_types_temp[i] = None
            v_name = param_names[i]
            args.append(Type(v_name))
        action = action_pred(*args)

        # Extract preconditions.
        precond_match = re.search(":precondition[\(\s]*\w", operator_str)
        if precond_match is None:
            # No preconditions found.
            return None
        # Get rid of space between ":effect (" and the first word such as "and" or a predicate name
        operator_str = operator_str[:precond_match.end() - 1].strip() + operator_str[precond_match.end() - 1:]

        precond_match = re.search(":precondition[\(\s]*\w", operator_str)
        precond_str = find_closing_paran(operator_str[precond_match.end() - 2:])
        # print(precond_str)
        #NOTE: Supports only LiteralConjunction and Literal for now.
        lc = self._parse_into_literal(precond_str, param_names, param_types, False)
        # print(lc)

        if lc is None:
            return None

        if isinstance(lc, Literal):
            preconds = LiteralConjunction([lc, action])
        elif isinstance(lc, LiteralConjunction):
            preconds = LiteralConjunction(lc.literals + [action])
        else:
            raise Exception(f"Unsupported type: {type(lc)}")    

        # Extract effects.
        effect_str_match = re.search(":effect[\(\s]*\w", operator_str)
        # Get rid of space between ":effect (" and the first word such as "and" or a predicate name
        operator_str = operator_str[:effect_str_match.end() - 1].strip() + operator_str[effect_str_match.end() - 1:]
        effect_str_match = re.search(":effect[\(\s]*\w", operator_str)
        effect_str = (operator_str[effect_str_match.end() - 2:].strip())
        effect_str = find_closing_paran(effect_str)

        effects = self._parse_into_literal(effect_str, param_names, param_types, is_effect=True)

        if effects is None:
            return None

        if isinstance(effects, Literal):
            effects  = LiteralConjunction([effects])
        elif isinstance(effects, LiteralConjunction):
            pass
        else:
            raise Exception(f"Unsupported type: {type(effects)}")    


        # Rename the variables
        var_name_gen = iter_variable_names()
        variables = {}
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
        preconds = LiteralConjunction(literals)

        literals = []
        for l in effects.literals:
            args = []
            for v in l.variables:
                args.append(variables[v])
            literals.append(Literal(l.predicate, args))
        effects = LiteralConjunction(literals)       

        params = set()
        for l in effects.literals + preconds.literals:
            for v in l.variables:
                params.add(v)
                
        return Operator(op_name, params, preconds, effects)

    def _find_all_balanced_expressions(self, string):
        """Return a list of all balanced expressions in a string,
        starting from the beginning.

        Source: parser.py in pddlgym PDDLParser.
        """
        if not string:
            return []
        assert string[0] == "("
        assert string[-1] == ")"
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

    def _parse_into_literal(self, string:str, param_names:list, param_types:list, is_effect:bool) -> Literal or None:
        """Parses the string into a literal or None if predicate name or argument types are invalid.

        Args:
            string: Effect or Precondition string. Starts with '(' and ends with its closing mirror ')'
            is_effect (bool): if the string is effect.
            param_names (list): variable names (such as '?x0') 
            param_types (list): types (such as 'ingredient')
        """
        if string.startswith("(and") and string[4] in (" ", "\n", "(", ")"):
            clauses = self._find_all_balanced_expressions(string[4:-1].strip())
            lits = [self._parse_into_literal(clause, param_names, param_types, 
                                        is_effect=is_effect) for clause in clauses]
            lits = [l for l in lits if l is not None]
            if len(lits) == 0:
                return None
            return LiteralConjunction(lits)

        if string.startswith("(not") and string[4] in (" ", "\n", "("):
            clause = string[4:-1].strip()
            lit = self._parse_into_literal(clause, param_names, param_types, is_effect=is_effect)
            if lit is None:
                return None
            if is_effect:
                    return Anti(lit)
            else:
                    return Not(lit)
        string = string[1:-1].split()
        pred, args = string[0], string[1:]
        typed_args = []

        # Validate types against the given param names.
        if pred not in self._observation_predicates:
            PARSING_LOGGER.debug(f"Parsed unknown predicate {pred}")
            return None
        if len(args) != self._observation_predicates[pred].arity:
            PARSING_LOGGER.debug(f"Parsed incongruent number of argument types for predicate {pred}")
            return None

        arg_types = []
        for i, arg in enumerate(args):
            if arg not in param_names:
                PARSING_LOGGER.debug("Argument {} not in params {}".format(arg, param_names))
                return None
            t = param_types[param_names.index(arg)]
            typed_arg = TypedEntity(arg, Type(t))
            arg_types.append(t)
            typed_args.append(typed_arg)

        if self._observation_predicates[pred].var_types != arg_types:
            PARSING_LOGGER.debug(f"Parsed incongruent argument types for predicate {pred}")
            return None

        return self._observation_predicates[pred](*typed_args)

def find_closing_paran(string:str) -> str:
    """_summary_

    Args:
        string: starts with "(" open paran.
    Returns:
        string: string truncated right after the mirrored closing paran
    """
    assert string[0] == "("
    balance = 0
    for i,c in enumerate(string):
        if c == ")":
            balance -= 1
        elif c == '(':
            balance += 1
        if balance == 0:
            return string[:i+1]
    raise Exception("Closing parantheses not found")


if __name__ == "__main__":
    import pddlgym
    import pickle
    import os

    PARSING_LOGGER.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    PARSING_LOGGER.addHandler(ch)


    env = pddlgym.make("PDDLEnvMinecraft-v0")
    observation_predicates = {p.name: p for p in env.observation_space.predicates}
    action_predicates = {p.name: p for p in env.action_space.predicates}
    types = set()
    for pred in ([p for p in env.observation_space.predicates] + [p for p in env.action_space.predicates]):
        for v_type in pred.var_types:
            types.add(v_type)

    parser = LLM_PDDL_Parser(action_predicates, observation_predicates, types)

    # PARSING_LOGGER.debug("TEST")
    problem = """(:action putpaninoven
    :parameters (?p - pan ?o - oven)
    :precondition (and 
        (not (ovenisfull ?o))
        (or (panhasegg ?p) (panhasflour ?p))
    )
    :effect (and 
        (inoven ?p ?o)
        (not (panisclean ?p))
        (ovenisfull ?o)
    )
)"""
    problem = """Based on the given predicates and object types, the PDDL operator for the "craftplank" action can be defined as follows:

```pddl
(:action craftplank
    :parameters (?agent - agent ?log - moveable ?planks - moveable)
    :precondition (and 
        (equipped ?log ?agent)
        (islog ?log)
        (hypothetical ?planks)
    )
    :effect (and
        (not (equipped ?log ?agent))
        (not (islog ?log))
        (not (hypothetical ?planks))
        (inventory ?planks)
        (isplanks ?planks)
        (handsfree ?agent)
    )
)
```

In this operator, the parameters are the agent, the log, and the hypothetical planks. The precondition for the action to take place is that the agent must be equipped with the log, the log must be identified as a log, and the planks must be hypothetical. The effect of the action is that the agent is no longer equipped with the log, the log is no longer identified as a log, the planks are no longer hypothetical but are now in the inventory and identified as planks, and the agent's hands are free."""
    print(parser.parse_operator(problem))
    # raise Exception
    # with open('/home/catalan/temp/later.pkl', 'rb') as f:
    #     later = pickle.load(f)
    # with open('/home/catalan/temp/done.pkl', 'rb') as f:
    #     done = pickle.load(f)
    # for file in os.listdir('/home/catalan/llm_cache'):
    #     if file == 'p.py': continue
    #     with open(os.path.join('/home/catalan/llm_cache', file), 'rb') as f:
    #         if file in done: continue
    #         # if file in later: continue
    #         s = pickle.load(f)[0]
    #         if ":action" in s:
    #             print(s)
    #             parsed =(parser.parse_operator(s))
    #             print(parsed)
    #             i = input()
    #             if i == 'y':
    #                 done.append(file)
    #             elif i == 'l':
    #                 later.append(file)
    #     with open('/home/catalan/temp/done.pkl', 'wb') as f:
    #         pickle.dump(done, f)
    #     with open('/home/catalan/temp/later.pkl', 'wb') as f:
    #         pickle.dump(later, f)