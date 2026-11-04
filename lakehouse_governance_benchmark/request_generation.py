"""
Generates structured operation requests and renders them as natural-language requests.

A request instance contains
  operation   the intended operation {"intent", "params"} (ground truth)
  context     requester, change ticket and change-window state (from the ticketing system)
  text        the request as written by the requester
  gold        decision and fired rules from the codified policy

Templates are written in four styles (terse ticket, chat message, formal email, descriptive). Template index 3
of every operation is held out of training and used only for the unseen-template test split.
"""

import random

from .operations import DESTINATIONS, OPERATIONS, SCHEMA_CHANGES
from .policy import evaluate_operation

TEMPLATES = {
    "describe_table": [
        "describe {table}",
        "hey can you pull up the schema and owner for {table}?",
        "Hello, could you please share the table properties and column definitions of {table}? Thank you.",
        "I'm trying to understand what's in {table}, what columns does it have and who owns it",
    ],
    "list_table_permissions": [
        "show grants on {table}",
        "who has access to {table} right now?",
        "Hi team, please provide the current list of principals with privileges on {table}.",
        "need an access review for {table}, list everyone who can read or write it",
    ],
    "check_job_status": [
        "status of job {job}",
        "did {job} run ok last night?",
        "Good morning, could you confirm whether the latest runs of {job} succeeded?",
        "dashboards look stale, can you check how the recent runs of {job} went",
    ],
    "grant_table_read": [
        "grant select on {table} to {grantee}",
        "can you give {grantee} read access to {table}? thx",
        "Hello, please provide {grantee} with read-only access to {table} for an upcoming analysis.",
        "{grantee} needs to query {table} for a report, please set that up",
    ],
    "grant_table_write": [
        "grant modify on {table} to {grantee}",
        "pls give {grantee} write perms on {table}",
        "Hi, could you grant {grantee} permission to insert and update data in {table}?",
        "{grantee} will be loading corrections into {table}, they need to be able to write to it",
    ],
    "revoke_table_access": [
        "revoke all on {table} from {grantee}",
        "{grantee} moved teams, can you pull their access to {table}",
        "Please remove all privileges that {grantee} currently holds on {table}.",
        "access cleanup: {grantee} shouldn't be able to touch {table} anymore",
    ],
    "add_user_to_group": [
        "add {user} to group {group}",
        "can you put {user} in {group}?",
        "Hello, please add {user} as a member of the {group} group.",
        "{user} is joining us and should be part of {group}",
    ],
    "create_table": [
        "create table {table}",
        "can you spin up a new table {table} for me",
        "Hi, please create a new managed table named {table}.",
        "I need an empty table at {table} to land some new data",
    ],
    "drop_table": [
        "drop table {table}",
        "we don't need {table} anymore, can you drop it",
        "Hello, please remove the table {table} from the catalog.",
        "{table} is obsolete, please get rid of it entirely",
    ],
    "restore_table_version": [
        "restore {table} to the version from {hours_ago} hours ago",
        "bad load hit {table}, can you roll it back {hours_ago} hours?",
        "Please restore {table} to its state as of {hours_ago} hours ago using time travel.",
        "something corrupted {table} recently, put it back the way it was {hours_ago} hours ago",
    ],
    "vacuum_table": [
        "vacuum {table} retain {retention_hours} hours",
        "storage costs are up, can you vacuum {table} keeping {retention_hours}h of history",
        "Please run VACUUM on {table} with a retention period of {retention_hours} hours.",
        "clean up old files under {table}, only keep the last {retention_hours} hours of versions",
    ],
    "delete_rows": [
        "delete from {table} where {predicate}",
        "can you delete the rows in {table} where {predicate}?",
        "Hello, please remove the records from {table} that satisfy {predicate}.",
        "there's junk data in {table}, get rid of rows matching {predicate}",
    ],
    "export_table": [
        "export {table} to {destination}",
        "can you dump {table} into {destination} for me",
        "Hi, please export the contents of {table} to {destination}.",
        "we need a copy of {table} sitting in {destination} by end of day",
    ],
    "resize_cluster": [
        "resize cluster {cluster} to {workers} workers",
        "jobs are slow, can you bump {cluster} to {workers} workers",
        "Please change the size of cluster {cluster} to {workers} worker nodes.",
        "{cluster} keeps running out of capacity, scale it to {workers} workers",
    ],
    "terminate_cluster": [
        "terminate cluster {cluster}",
        "can you shut down {cluster}? nobody's using it",
        "Hello, please terminate the cluster {cluster}.",
        "{cluster} has been idle for days, stop it",
    ],
    "rotate_secret": [
        "rotate secret {secret}",
        "can you rotate {secret}, it might have leaked in a log",
        "Please rotate the secret stored at {secret}.",
        "we need a fresh value for {secret} as part of the quarterly rotation",
    ],
    "read_secret_value": [
        "get value of secret {secret}",
        "what's the value of {secret}? need it for local testing",
        "Hello, could you share the current value of the secret {secret}?",
        "I need to paste {secret} into my notebook config, can you send it over",
    ],
    "create_access_token": [
        "create a personal access token valid for {lifetime_days} days",
        "can i get a PAT that lasts {lifetime_days} days",
        "Please issue me a personal access token with a lifetime of {lifetime_days} days.",
        "my scheduled scripts need a token, make one that expires in {lifetime_days} days",
    ],
    "trigger_job_run": [
        "run job {job} now",
        "can you kick off {job}?",
        "Please trigger an immediate run of {job}.",
        "upstream data just landed, start {job} so the tables refresh",
    ],
    "backfill_pipeline": [
        "backfill {job} for the last {days} days",
        "can you rerun {job} over the past {days} days",
        "Please backfill {job} covering the previous {days} days.",
        "we fixed a bug in {job}, reprocess the last {days} days of data with it",
    ],
    "pause_pipeline": [
        "pause job {job}",
        "can you pause {job} for now",
        "Please pause the schedule of {job} until further notice.",
        "{job} is writing bad data, stop it from running on schedule",
    ],
    "change_topic_retention": [
        "set retention of topic {topic} to {retention_hours} hours",
        "can you change {topic} retention to {retention_hours}h",
        "Please update the retention period of the Kafka topic {topic} to {retention_hours} hours.",
        "{topic} should keep messages for {retention_hours} hours from now on",
    ],
    "reset_consumer_offsets": [
        "reset consumer offsets on {topic}",
        "consumers on {topic} are stuck, can you reset their offsets",
        "Please reset the consumer group offsets for the topic {topic}.",
        "we need to replay {topic} from the beginning for all consumers",
    ],
    "evolve_avro_schema": [
        "register schema change on {subject}: {change}",
        "can you push a new version of {subject} that does this: {change}",
        "Please register a new version of the Avro schema {subject} with the following change: {change}.",
        "the producer for {subject} is changing, the new schema will {change}",
    ],
}
HELD_OUT_TEMPLATE = 3

CHANGE_TEXT = {"add_optional_field": "add an optional field loyalty_tier with default null",
               "remove_field": "remove the field customer_segment",
               "change_field_type": "change the type of amount from double to string"}
DESTINATION_TEXT = {"internal_bucket": ["s3://corp-analytics-exports/", "abfss://internal-share@corpdatalake/"],
                    "external_bucket": ["s3://partner-dropbox-ext/", "gs://vendor-ingest-bucket/"]}
PREDICATES = ["event_date < '2023-01-01'", "status = 'TEST'", "source_system = 'legacy_crm'",
              "load_id = 'L-20261012-07'", "member_id = 'M0042317'"]
NO_PREDICATE_TEXT = {0: "all rows", 1: "everything", 2: "all records", 3: "every row"}


def _resource_names(env, kind, env_weights=None):
    names = list(getattr(env, kind))
    if not env_weights:
        return names
    return names


def sample_operation(rng, env, intent):
    spec = OPERATIONS[intent]
    p = {}
    pick = lambda kind: rng.choice(sorted(getattr(env, kind)))  # noqa: E731
    humans = sorted(n for n, pr in env.principals.items() if pr.role != "service_principal")
    if "table" in spec.params and spec.params["table"] == "table":
        prod = [t for t in sorted(env.tables) if t.startswith("prod")]
        p["table"] = rng.choice(prod) if rng.random() < 0.55 else pick("tables")
    if intent == "create_table":
        e = rng.choice(["prod", "dev", "dev", "staging"])
        team = rng.choice(["finance", "claims", "marketing", "risk"])
        schema = {"finance": "accounting", "claims": "member", "marketing": "engagement", "risk": "controls"}[team]
        p["table"] = f"{e}_{team}.{schema}.{rng.choice(['tmp', 'stg', 'new'])}_{rng.choice(['returns', 'adjustments', 'segments', 'alerts'])}"
    if "grantee" in spec.params:
        p["grantee"] = rng.choice(humans)
    if intent == "add_user_to_group":
        p["group"], p["user"] = pick("groups"), rng.choice(humans)
    if "hours_ago" in spec.params:
        p["hours_ago"] = rng.choice([2, 6, 12, 24, 48])
    if intent == "vacuum_table":
        p["retention_hours"] = rng.choice([0, 24, 72, 168, 168, 336, 720])
    if intent == "delete_rows":
        p["predicate"] = rng.choice(PREDICATES) if rng.random() < 0.75 else ""
    if intent == "export_table":
        p["destination"] = rng.choice(DESTINATIONS)
    if "cluster" in spec.params:
        p["cluster"] = pick("clusters")
    if intent == "resize_cluster":
        c = env.clusters[p["cluster"]]
        p["workers"] = rng.choice([c.current_workers + 1, 2 * c.current_workers, 2 * c.current_workers + 2,
                                   c.policy_max_workers, c.policy_max_workers + 8])
    if "secret" in spec.params:
        p["secret"] = pick("secrets")
    if intent == "create_access_token":
        p["lifetime_days"] = rng.choice([7, 14, 30, 60, 90, 180, 365])
    if "job" in spec.params:
        p["job"] = pick("jobs")
    if intent == "backfill_pipeline":
        p["days"] = rng.choice([3, 14, 30, 60, 120, 200, 400])
    if "topic" in spec.params:
        p["topic"] = pick("topics")
    if intent == "change_topic_retention":
        cur = env.topics[p["topic"]].retention_hours
        p["retention_hours"] = rng.choice([12, 24, max(24, cur // 2), cur * 2, 1440])
    if intent == "evolve_avro_schema":
        p["subject"], p["change"] = pick("subjects"), rng.choice(SCHEMA_CHANGES)
    return {"intent": intent, "params": p}


def sample_context(rng, env, operation):
    spec = OPERATIONS[operation["intent"]]
    params = operation["params"]
    target = env.lookup(spec.target_kind, params.get(spec.target_param)) if spec.target_kind else None
    owner_team = getattr(target, "owner_team", None)
    roles = ["data_engineer", "platform_admin", "data_scientist", "analyst", "contractor"]
    weights = [0.45, 0.12, 0.15, 0.18, 0.10] if spec.mutating else [0.2, 0.05, 0.25, 0.4, 0.1]
    role = rng.choices(roles, weights)[0]
    pool = sorted(n for n, pr in env.principals.items() if pr.role == role)
    same_team = [n for n in pool if env.principals[n].team == owner_team]
    requester = rng.choice(same_team) if same_team and rng.random() < 0.65 else rng.choice(pool)
    # self-referential requests are common and stress the "me" resolution
    if "grantee" in params and rng.random() < 0.2:
        params["grantee"] = requester
    if "user" in params and rng.random() < 0.3:
        params["user"] = requester
    u = rng.random()
    ticket = (None if u < 0.3 else {"id": f"CHG-{rng.randint(10000, 99999)}", "approved": u >= 0.45})
    return {"requester": requester, "change_ticket": ticket, "in_change_window": rng.random() < 0.8}


def render_text(rng, operation, context, template_index):
    intent, params = operation["intent"], operation["params"]
    tmpl = TEMPLATES[intent][template_index]
    values = dict(params)
    for key in ("grantee", "user"):
        if key in values and values[key] == context["requester"]:
            values[key] = "me"
    if intent == "delete_rows" and not params.get("predicate"):
        tmpl = {0: "delete from {table}", 1: "can you delete everything in {table}?",
                2: "Hello, please remove all records from {table}.",
                3: "{table} is full of junk, wipe every row out of it"}[template_index]
    if "destination" in values:
        values["destination"] = rng.choice(DESTINATION_TEXT[params["destination"]])
    if "change" in values:
        values["change"] = CHANGE_TEXT[params["change"]]
    text = tmpl.format(**values)
    if template_index == 1 and rng.random() < 0.3:
        text = text.replace(" you ", " u ").replace("please", "pls")
    return text


def generate_instance(rng, env, intent, template_index):
    op = sample_operation(rng, env, intent)
    ctx = sample_context(rng, env, op)
    decision, rules = evaluate_operation(op, ctx, env)
    return {"operation": op, "context": ctx, "text": render_text(rng, op, ctx, template_index),
            "template_index": template_index, "gold_decision": decision, "gold_rules": rules}
