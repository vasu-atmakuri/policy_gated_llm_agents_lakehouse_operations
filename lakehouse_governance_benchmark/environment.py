"""
Deterministic simulated lakehouse environment (the system of record).

The environment holds every resource an operation request can refer to: Unity-Catalog-style tables with an
environment, data classification, owner team and a free-text comment; compute clusters with a policy limit;
secrets; scheduled jobs and pipelines; Kafka topics; Avro schema subjects; groups; and principals with a role
and team. The policy engine reads resource attributes only from here, never from the request text.
"""

import random
from dataclasses import asdict, dataclass, field

SEED = 20261104
ENVIRONMENTS = ("prod", "staging", "dev")
CLASSIFICATIONS = ("public", "internal", "confidential", "restricted")
ROLES = ("analyst", "data_scientist", "data_engineer", "platform_admin", "contractor", "service_principal")
TEAMS = ("finance", "claims", "marketing", "supply_chain", "customer_care", "platform", "risk")

DOMAIN_TABLES = {
    "finance": [("general_ledger", "confidential"), ("invoices", "confidential"), ("cost_centers", "internal"),
                ("payroll_summary", "restricted"), ("fx_rates", "public")],
    "claims": [("member_eligibility", "restricted"), ("claim_lines", "restricted"), ("provider_directory", "internal"),
               ("claim_status_daily", "confidential"), ("diagnosis_codes", "public")],
    "marketing": [("campaign_events", "internal"), ("web_sessions", "internal"), ("customer_profiles", "restricted"),
                  ("email_engagement", "confidential"), ("ad_spend", "internal")],
    "supply_chain": [("shipments", "internal"), ("inventory_snapshots", "internal"), ("supplier_contracts", "confidential"),
                     ("warehouse_locations", "public"), ("purchase_orders", "confidential")],
    "customer_care": [("support_tickets", "confidential"), ("call_transcripts", "restricted"), ("csat_scores", "internal"),
                      ("agent_schedules", "internal"), ("knowledge_articles", "public")],
    "risk": [("fraud_scores", "restricted"), ("credit_limits", "confidential"), ("audit_findings", "confidential"),
             ("sanctions_screening", "restricted"), ("risk_appetite_metrics", "internal")],
}
SCHEMA_NAMES = {"finance": "accounting", "claims": "member", "marketing": "engagement", "supply_chain": "logistics",
                "customer_care": "support", "risk": "controls"}

FIRST = ["alice", "ben", "chen", "dmitri", "elena", "farah", "gabriel", "hana", "ivan", "jia", "kofi", "lucia",
         "mateo", "nadia", "omar", "priya", "quinn", "rahul", "sofia", "tariq", "uma", "victor", "wen", "yusuf",
         "zara", "arjun", "beatriz", "carlos", "deepa", "emeka"]
LAST = ["nguyen", "patel", "smith", "garcia", "kim", "okafor", "rossi", "silva", "tanaka", "muller", "haddad",
        "reddy", "lopez", "cohen", "ivanova"]


@dataclass
class Principal:
    name: str
    role: str
    team: str
    is_external: bool = False


@dataclass
class Table:
    full_name: str
    env: str
    classification: str
    owner_team: str
    size_gb: int
    comment: str = ""


@dataclass
class Cluster:
    name: str
    env: str
    owner_team: str
    current_workers: int
    policy_max_workers: int


@dataclass
class Secret:
    name: str
    env: str
    owner_team: str


@dataclass
class Job:
    name: str
    env: str
    owner_team: str
    schedule: str


@dataclass
class Topic:
    name: str
    env: str
    owner_team: str
    retention_hours: int
    consumer_groups: int


@dataclass
class SchemaSubject:
    name: str
    env: str
    owner_team: str


@dataclass
class Group:
    name: str
    privileged: bool
    admin: bool = False


@dataclass
class Environment:
    principals: dict = field(default_factory=dict)
    tables: dict = field(default_factory=dict)
    clusters: dict = field(default_factory=dict)
    secrets: dict = field(default_factory=dict)
    jobs: dict = field(default_factory=dict)
    topics: dict = field(default_factory=dict)
    subjects: dict = field(default_factory=dict)
    groups: dict = field(default_factory=dict)

    def lookup(self, kind, name):
        return getattr(self, kind).get(name)

    def to_json(self):
        return {k: {n: asdict(v) for n, v in getattr(self, k).items()} for k in
                ("principals", "tables", "clusters", "secrets", "jobs", "topics", "subjects", "groups")}


def build_environment(seed=SEED):
    rng = random.Random(seed)
    env = Environment()
    names = [f"{f}.{l}" for f in FIRST for l in LAST]
    rng.shuffle(names)
    roles_plan = (["analyst"] * 14 + ["data_scientist"] * 8 + ["data_engineer"] * 14 + ["platform_admin"] * 4
                  + ["contractor"] * 6)
    for i, role in enumerate(roles_plan):
        team = "platform" if role == "platform_admin" else rng.choice([t for t in TEAMS if t != "platform"])
        env.principals[names[i]] = Principal(names[i], role, team, is_external=False)
    for i in range(4):
        n = f"{names[60 + i].split('.')[0]}@partner-{rng.choice(['analytics', 'consulting', 'data'])}.com"
        env.principals[n] = Principal(n, "contractor", rng.choice(TEAMS[:5]), is_external=True)
    for team in TEAMS:
        if team == "platform":
            continue
        n = f"sp-{team.replace('_', '-')}-etl"
        env.principals[n] = Principal(n, "service_principal", team)

    for team, tables in DOMAIN_TABLES.items():
        schema = SCHEMA_NAMES[team]
        for e in ENVIRONMENTS:
            for t, cls in tables:
                if e != "prod" and rng.random() < 0.35:
                    continue
                fn = f"{e}_{team}.{schema}.{t}"
                env.tables[fn] = Table(fn, e, cls, team, size_gb=int(rng.lognormvariate(4, 1.5)) + 1)
    for e in ENVIRONMENTS:
        for team in TEAMS:
            if team == "platform" or (e == "staging" and rng.random() < 0.5):
                continue
            mx = {"prod": 64, "staging": 16, "dev": 8}[e]
            name = f"{e}-{team.replace('_', '-')}-{rng.choice(['etl', 'analytics', 'ml'])}"
            env.clusters[name] = Cluster(name, e, team, current_workers=rng.randint(2, mx // 2), policy_max_workers=mx)
    for e in ("prod", "dev"):
        for team in TEAMS:
            for key in ("db-password", "api-key", "storage-sas-token"):
                if rng.random() < 0.5:
                    n = f"{e}/{team.replace('_', '-')}/{key}"
                    env.secrets[n] = Secret(n, e, team)
    job_kinds = ["daily_ingest", "hourly_refresh", "weekly_rollup", "cdc_merge", "feature_build"]
    for e in ("prod", "dev"):
        for team in TEAMS:
            if team == "platform":
                continue
            for kind in rng.sample(job_kinds, 2):
                n = f"{e}_{team}_{kind}"
                env.jobs[n] = Job(n, e, team, rng.choice(["0 2 * * *", "0 * * * *", "0 6 * * 1"]))
    for e in ("prod", "dev"):
        for team in TEAMS:
            if team == "platform":
                continue
            for stream in rng.sample(["events", "cdc", "transactions", "clickstream", "alerts"], 2):
                n = f"{e}.{team.replace('_', '-')}.{stream}"
                env.topics[n] = Topic(n, e, team, retention_hours=rng.choice([72, 168, 336, 720]),
                                      consumer_groups=rng.randint(1, 6))
                env.subjects[f"{n}-value"] = SchemaSubject(f"{n}-value", e, team)
    for team in TEAMS:
        env.groups[f"{team}-readers"] = Group(f"{team}-readers", privileged=False)
        env.groups[f"{team}-engineers"] = Group(f"{team}-engineers", privileged=True)
    env.groups["workspace-admins"] = Group("workspace-admins", privileged=True, admin=True)
    env.groups["metastore-admins"] = Group("metastore-admins", privileged=True, admin=True)
    return env
