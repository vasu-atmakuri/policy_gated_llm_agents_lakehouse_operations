"""
Catalog of lakehouse operations an agent may be asked to perform.

Each operation has a category, typed parameters and flags used by the governance policy:
  mutating     changes state (anything except read-only inspection)
  destructive  can lose data or availability irreversibly
  target_kind  the environment collection that holds the operation's primary resource
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class OperationSpec:
    name: str
    category: str
    description: str
    params: dict = field(default_factory=dict)  # parameter -> kind
    target_param: str = ""
    target_kind: str = ""
    mutating: bool = True
    destructive: bool = False


OPERATIONS = {s.name: s for s in [
    # read-only inspection
    OperationSpec("describe_table", "inspection", "Show schema, owner and properties of a table",
                  {"table": "table"}, "table", "tables", mutating=False),
    OperationSpec("list_table_permissions", "inspection", "List who has access to a table",
                  {"table": "table"}, "table", "tables", mutating=False),
    OperationSpec("check_job_status", "inspection", "Show the status of recent runs of a job",
                  {"job": "job"}, "job", "jobs", mutating=False),
    # access control
    OperationSpec("grant_table_read", "access_control", "Grant SELECT on a table to a principal",
                  {"table": "table", "grantee": "principal"}, "table", "tables"),
    OperationSpec("grant_table_write", "access_control", "Grant MODIFY on a table to a principal",
                  {"table": "table", "grantee": "principal"}, "table", "tables"),
    OperationSpec("revoke_table_access", "access_control", "Revoke all privileges of a principal on a table",
                  {"table": "table", "grantee": "principal"}, "table", "tables"),
    OperationSpec("add_user_to_group", "access_control", "Add a principal to a workspace group",
                  {"group": "group", "user": "principal"}, "group", "groups"),
    # table lifecycle
    OperationSpec("create_table", "table_lifecycle", "Create a new managed table",
                  {"table": "new_table"}, "table", ""),
    OperationSpec("drop_table", "table_lifecycle", "Drop a table",
                  {"table": "table"}, "table", "tables", destructive=True),
    OperationSpec("restore_table_version", "table_lifecycle", "Restore a Delta table to an earlier version",
                  {"table": "table", "hours_ago": "int"}, "table", "tables", destructive=True),
    OperationSpec("vacuum_table", "table_lifecycle", "Remove old data files of a Delta table",
                  {"table": "table", "retention_hours": "int"}, "table", "tables", destructive=True),
    OperationSpec("delete_rows", "table_lifecycle", "Delete rows from a table matching a predicate",
                  {"table": "table", "predicate": "predicate"}, "table", "tables", destructive=True),
    OperationSpec("export_table", "table_lifecycle", "Export a table to a storage location",
                  {"table": "table", "destination": "destination"}, "table", "tables"),
    # compute
    OperationSpec("resize_cluster", "compute", "Change the number of workers of a cluster",
                  {"cluster": "cluster", "workers": "int"}, "cluster", "clusters"),
    OperationSpec("terminate_cluster", "compute", "Terminate a running cluster",
                  {"cluster": "cluster"}, "cluster", "clusters", destructive=True),
    # secrets and credentials
    OperationSpec("rotate_secret", "secrets", "Rotate a secret value",
                  {"secret": "secret"}, "secret", "secrets"),
    OperationSpec("read_secret_value", "secrets", "Reveal the plaintext value of a secret",
                  {"secret": "secret"}, "secret", "secrets"),
    OperationSpec("create_access_token", "secrets", "Create a personal access token",
                  {"lifetime_days": "int"}, "", ""),
    # pipelines
    OperationSpec("trigger_job_run", "pipelines", "Start a run of a job now",
                  {"job": "job"}, "job", "jobs"),
    OperationSpec("backfill_pipeline", "pipelines", "Re-run a pipeline over a past date range",
                  {"job": "job", "days": "int"}, "job", "jobs"),
    OperationSpec("pause_pipeline", "pipelines", "Pause the schedule of a job",
                  {"job": "job"}, "job", "jobs"),
    # streaming and schemas
    OperationSpec("change_topic_retention", "streaming", "Change the retention period of a Kafka topic",
                  {"topic": "topic", "retention_hours": "int"}, "topic", "topics"),
    OperationSpec("reset_consumer_offsets", "streaming", "Reset consumer-group offsets of a topic",
                  {"topic": "topic"}, "topic", "topics", destructive=True),
    OperationSpec("evolve_avro_schema", "streaming", "Register a new version of an Avro schema",
                  {"subject": "subject", "change": "schema_change"}, "subject", "subjects"),
]}

SCHEMA_CHANGES = ("add_optional_field", "remove_field", "change_field_type")
DESTINATIONS = ("internal_bucket", "external_bucket")
DECISIONS = ("auto_execute", "human_review", "reject")
SEVERITY = {d: i for i, d in enumerate(DECISIONS)}
