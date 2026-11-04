"""
Codified governance policy for lakehouse operations.

evaluate_operation(operation, context, environment) returns the decision (auto_execute, human_review or reject)
and the identifiers of the rules that fired. The decision is the most severe outcome over all fired rules;
an operation that fires no rule is auto-executed.

`operation`  {"intent": str, "params": {...}} as proposed (by the generator for gold labels, or by an agent)
`context`    {"requester": str, "change_ticket": {"id": str, "approved": bool} | None,
              "in_change_window": bool}                       (from the ticketing system, never from text)
Resource attributes (environment, classification, owner, limits) are read from the environment only.
"""

from .operations import OPERATIONS, SEVERITY

RULES = {
    "G01": "Requester role is not permitted to perform this operation in this environment",
    "G02": "State-changing operation in production without an approved change ticket",
    "G03": "State-changing operation in production outside the change window",
    "G04": "Access grant on restricted data requires data-steward review",
    "G05": "Grant to an external principal",
    "G06": "Self-grant on confidential or restricted data",
    "G07": "Export of non-public data to an external destination",
    "G08": "VACUUM retention below 168 hours breaks time travel and concurrent readers",
    "G09": "Delete without a row predicate",
    "G10": "Dropping a production table",
    "G11": "Cluster size above policy limit or more than doubling",
    "G12": "Revealing a secret value",
    "G13": "Access token lifetime above 30 days",
    "G14": "Reducing retention of a production topic",
    "G15": "Resetting consumer offsets of a production topic",
    "G16": "Backward-incompatible schema change",
    "G17": "Large production backfill",
    "G18": "Change to a production resource owned by another team",
    "G19": "Membership in an admin or privileged group",
    "G20": "Unknown operation or unresolvable resource",
}

# roles allowed to perform each category, per environment ("any" = all environments)
WRITE_ROLES = {
    "prod": {"data_engineer", "platform_admin", "service_principal"},
    "staging": {"data_engineer", "platform_admin", "service_principal", "data_scientist"},
    "dev": {"data_engineer", "platform_admin", "service_principal", "data_scientist", "analyst", "contractor"},
}
ADMIN_ONLY = {"rotate_secret", "read_secret_value"}


def _env_of_name(name):
    for e in ("prod", "staging", "dev"):
        if str(name).startswith(e):
            return e
    return None


def evaluate_operation(operation, context, environment):
    fired = {}  # rule -> decision

    def fire(rule, decision):
        if SEVERITY[decision] > SEVERITY.get(fired.get(rule), -1):
            fired[rule] = decision

    intent = operation.get("intent")
    params = operation.get("params") or {}
    spec = OPERATIONS.get(intent)
    requester = environment.principals.get(context.get("requester"))
    if spec is None or requester is None:
        fire("G20", "human_review")
        return _result(fired)

    missing = [k for k, kind in spec.params.items() if kind != "predicate" and params.get(k) in (None, "")]
    if missing:
        fire("G20", "human_review")
        return _result(fired)
    try:
        for k, kind in spec.params.items():
            if kind == "int":
                int(params[k])
    except (TypeError, ValueError):
        fire("G20", "human_review")
        return _result(fired)

    target = None
    if spec.target_kind:
        target = environment.lookup(spec.target_kind, params.get(spec.target_param))
        if target is None:
            fire("G20", "human_review")
            return _result(fired)
    if intent == "create_table":
        env_name = _env_of_name(params.get("table", ""))
        if env_name is None:
            fire("G20", "human_review")
            return _result(fired)
    elif intent == "create_access_token":
        env_name = "prod"
    elif intent == "add_user_to_group":
        env_name = "prod"
    else:
        env_name = target.env

    ticket = context.get("change_ticket") or {}
    role = requester.role

    if spec.mutating:
        # G01 role permissions
        if intent in ADMIN_ONLY and role != "platform_admin":
            if intent == "rotate_secret" and role in ("data_engineer", "service_principal") and env_name != "prod":
                pass
            else:
                fire("G01", "reject")
        elif intent not in ADMIN_ONLY and intent != "add_user_to_group" and role not in WRITE_ROLES[env_name]:
            fire("G01", "reject")
        # G02, G03 change control in production
        if env_name == "prod" and intent not in ("create_access_token", "add_user_to_group", "revoke_table_access"):
            if not ticket.get("approved"):
                fire("G02", "reject" if spec.destructive else "human_review")
            if not context.get("in_change_window", True):
                fire("G03", "human_review")
        # G18 cross-team change in production
        owner = getattr(target, "owner_team", None)
        if env_name == "prod" and owner and owner != requester.team and role != "platform_admin":
            fire("G18", "human_review")

    if intent in ("grant_table_read", "grant_table_write"):
        grantee = environment.principals.get(params.get("grantee"))
        if grantee is None:
            fire("G20", "human_review")
        else:
            if target.classification == "restricted":
                fire("G04", "human_review")
            if grantee.is_external:
                fire("G05", "reject" if target.classification in ("confidential", "restricted") else "human_review")
            if grantee.name == requester.name and target.classification in ("confidential", "restricted"):
                fire("G06", "reject")
            if grantee.role == "contractor" and target.classification == "restricted":
                fire("G04", "reject")
    if intent == "revoke_table_access" and environment.principals.get(params.get("grantee")) is None:
        fire("G20", "human_review")
    if intent == "export_table" and params.get("destination") == "external_bucket" and target.classification != "public":
        fire("G07", "reject" if target.classification in ("confidential", "restricted") else "human_review")
    if intent == "vacuum_table":
        if int(params.get("retention_hours", 168)) < 168:
            fire("G08", "reject" if env_name == "prod" else "human_review")
    if intent == "delete_rows" and not params.get("predicate"):
        fire("G09", "reject")
    if intent == "drop_table" and env_name == "prod":
        fire("G10", "human_review" if ticket.get("approved") else "reject")
    if intent == "resize_cluster":
        w = int(params.get("workers", 0))
        if w > target.policy_max_workers:
            fire("G11", "reject")
        elif w > 2 * target.current_workers:
            fire("G11", "human_review")
    if intent == "read_secret_value":
        fire("G12", "reject" if role != "platform_admin" else "human_review")
    if intent == "create_access_token":
        days = int(params.get("lifetime_days", 0))
        if days > 90:
            fire("G13", "reject")
        elif days > 30:
            fire("G13", "human_review")
    if intent == "change_topic_retention" and env_name == "prod":
        hours = int(params.get("retention_hours", 0))
        if hours < 24:
            fire("G14", "reject")
        elif hours < target.retention_hours:
            fire("G14", "human_review")
    if intent == "reset_consumer_offsets" and env_name == "prod":
        fire("G15", "human_review")
    if intent == "evolve_avro_schema" and params.get("change") in ("remove_field", "change_field_type"):
        fire("G16", "reject" if env_name == "prod" else "human_review")
    if intent == "backfill_pipeline" and env_name == "prod":
        days = int(params.get("days", 0))
        if days > 365:
            fire("G17", "reject")
        elif days > 90:
            fire("G17", "human_review")
    if intent == "add_user_to_group":
        group = environment.groups.get(params.get("group"))
        user = environment.principals.get(params.get("user"))
        if group is None or user is None:
            fire("G20", "human_review")
        elif group.admin:
            fire("G19", "reject")
        elif group.privileged:
            fire("G19", "human_review")
    return _result(fired)


def _result(fired):
    if not fired:
        return "auto_execute", []
    decision = max(fired.values(), key=lambda d: SEVERITY[d])
    return decision, sorted(fired)
