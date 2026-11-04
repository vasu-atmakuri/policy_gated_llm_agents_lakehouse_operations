"""
Non-LLM request parsing used by the baselines.

keyword_intent      ordered keyword rules written from common operator vocabulary. The rules were written with
                    knowledge of the request templates, so templated splits give an upper bound for this baseline;
                    the paraphrase and disguise splits are the fair comparison.
extract_parameters  exact-match resolution of resource and principal names against the catalog, numbers by
                    pattern, storage URLs through the location registry, SQL predicates after "where" or
                    "matching", and schema-change type by keywords
"""

import re

from .operations import OPERATIONS
from .request_generation import DESTINATION_TEXT

KEYWORD_RULES = [
    ("read_secret_value", r"(value of|what'?s the value|share the (current )?value).*secret|secret.*\bvalue\b"),
    ("rotate_secret", r"\brotat"),
    ("create_access_token", r"\b(token|pat)\b"),
    ("list_table_permissions", r"(who has access|show grants|list .*privileges|access review|principals with privileges)"),
    ("revoke_table_access", r"\b(revoke|remove .*privileges|pull .*access|shouldn'?t be able)"),
    ("grant_table_write", r"(grant modify|write|insert and update|loading corrections)"),
    ("grant_table_read", r"(grant select|read access|read-only access|query)"),
    ("add_user_to_group", r"\b(group|member of|part of)\b"),
    ("vacuum_table", r"\b(vacuum|old files)"),
    ("restore_table_version", r"\b(restore|roll(ed)? ?back|put it back|time travel)"),
    ("delete_rows", r"\b(delete|remove the records|remove all records|wipe|get rid of rows)"),
    ("drop_table", r"\b(drop|remove the table|get rid of it)"),
    ("export_table", r"\b(export|dump|copy of)"),
    ("resize_cluster", r"\b(resize|workers|scale|bump)"),
    ("terminate_cluster", r"\b(terminate|shut down|stop it|idle)"),
    ("backfill_pipeline", r"\b(backfill|rerun|re-run|reprocess)"),
    ("pause_pipeline", r"\b(pause|stop it from running)"),
    ("check_job_status", r"(status|did .* run|succeeded|how the recent runs)"),
    ("trigger_job_run", r"\b(run job|kick off|trigger|start )"),
    ("change_topic_retention", r"\bretention|keep messages"),
    ("reset_consumer_offsets", r"\b(offset|replay)"),
    ("evolve_avro_schema", r"\b(schema change|new version of|avro|new schema|field)"),
    ("create_table", r"\b(create (a new managed )?table|new table|empty table)"),
    ("describe_table", r"\b(describe|schema|columns|properties|who owns)"),
]
KIND_COLLECTION = {"table": "tables", "principal": "principals", "group": "groups", "cluster": "clusters",
                   "secret": "secrets", "job": "jobs", "topic": "topics", "subject": "subjects"}


def keyword_intent(text):
    t = text.lower()
    for intent, pattern in KEYWORD_RULES:
        if re.search(pattern, t):
            return intent
    return "out_of_scope"


def _find_names(text, names):
    t = text.lower()
    hits = [n for n in names if n.lower() in t]
    hits.sort(key=lambda n: (-len(n), t.find(n.lower())))
    return hits


def extract_parameters(text, intent, requester, env):
    spec = OPERATIONS.get(intent)
    if spec is None:
        return {}
    params = {}
    numbers = [int(x) for x in re.findall(r"\b(\d+)\s*(?:h\b|hours?|days?|workers?|worker)", text.lower())]
    principals_found = _find_names(text, list(env.principals))
    for key, kind in spec.params.items():
        if kind in KIND_COLLECTION and kind != "principal":
            found = _find_names(text, list(getattr(env, KIND_COLLECTION[kind])))
            if kind == "topic":
                found = [f for f in found if not f.endswith("-value")]
            params[key] = found[0] if found else None
        elif kind == "new_table":
            m = re.search(r"\b((?:prod|staging|dev)_[a-z_]+\.[a-z_]+\.[a-z_]+)", text)
            params[key] = m.group(1) if m else None
        elif kind == "principal":
            params[key] = (principals_found[0] if principals_found else
                           requester if re.search(r"\bme\b", text.lower()) else None)
        elif kind == "int":
            params[key] = numbers[0] if numbers else None
        elif kind == "predicate":
            m = re.search(r"(?:where|matching|satisfy)\s+(.+?)(?:\?|\.$|$)", text)
            params[key] = m.group(1).strip() if m else ""
        elif kind == "destination":
            params[key] = next((kind_ for kind_, urls in DESTINATION_TEXT.items()
                                for u in urls if u in text), None)
        elif kind == "schema_change":
            t = text.lower()
            params[key] = ("remove_field" if "remove" in t else "change_field_type" if "type" in t
                           else "add_optional_field")
    return params
