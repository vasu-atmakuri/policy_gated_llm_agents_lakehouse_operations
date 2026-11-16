"""
Fail-closed structural preconditions, evaluated before any governance rule.

structural_problem(operation, context, environment) returns None when the operation is well formed and every
resource and principal it names exists, and otherwise a short reason. A validator that sends every operation with a
structural problem to human review no longer depends on rule G20 for this, so removing or editing a governance rule
cannot let an unresolved operation through.

canonical_predicate(text) compares SQL row predicates field-specifically: keywords, column names and operators are
case-folded and whitespace-collapsed, while quoted literals are kept exactly (only the quote character is unified).
"""

import re

from .operations import OPERATIONS

PRINCIPAL_PARAMS = {"grantee", "user"}


def structural_problem(operation, context, environment):
    if not isinstance(operation, dict) or operation.get("intent") not in OPERATIONS:
        return "unknown_operation"
    spec = OPERATIONS[operation["intent"]]
    params = operation.get("params") or {}
    if environment.principals.get(context.get("requester")) is None:
        return "unknown_requester"
    for k, kind in spec.params.items():
        v = params.get(k)
        if kind != "predicate" and v in (None, ""):
            return "missing_argument"
        if kind == "int":
            try:
                int(v)
            except (TypeError, ValueError):
                return "malformed_argument"
    if spec.target_kind and environment.lookup(spec.target_kind, params.get(spec.target_param)) is None:
        return "unknown_resource"
    if operation["intent"] == "create_table" and not str(params.get("table", "")).startswith(("prod", "staging", "dev")):
        return "unknown_resource"
    for k in PRINCIPAL_PARAMS & set(spec.params):
        if environment.principals.get(params.get(k)) is None:
            return "unknown_principal"
    if operation["intent"] == "add_user_to_group" and environment.groups.get(params.get("group")) is None:
        return "unknown_resource"
    return None


_TOKEN = re.compile(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"|\S")


def canonical_predicate(text):
    if text is None:
        return ""
    out, word = [], ""
    s = str(text).strip().rstrip(";")
    i = 0
    while i < len(s):
        ch = s[i]
        if ch in "'\"":
            j = i + 1
            while j < len(s) and not (s[j] == ch and (j + 1 >= len(s) or s[j + 1] != ch)):
                j += 2 if s[j] == ch else 1
            lit = s[i + 1:j]
            if word:
                out.append(word.lower())
                word = ""
            out.append("'" + lit + "'")
            i = j + 1
        elif ch.isspace():
            if word:
                out.append(word.lower())
                word = ""
            i += 1
        elif ch in "=<>!(),":
            if word:
                out.append(word.lower())
                word = ""
            op = s[i:i + 2] if s[i:i + 2] in ("<=", ">=", "<>", "!=") else ch
            out.append(op)
            i += len(op)
        else:
            word += ch
            i += 1
    if word:
        out.append(word.lower())
    return " ".join(out)
