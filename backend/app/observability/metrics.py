from prometheus_client import Counter, Gauge, Histogram

# Request Counters
REQUEST_COUNT = Counter(
    "routemind_requests_total",
    "Total LLM gateway requests processed",
    ["provider", "model", "strategy", "status"],
)

# Latency Histograms
REQUEST_LATENCY = Histogram(
    "routemind_request_duration_seconds",
    "End-to-end gateway request duration in seconds",
    ["provider", "model"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

# Token Counters
TOKEN_USAGE = Counter(
    "routemind_tokens_total",
    "Total token consumption across tenants and models",
    ["tenant", "model", "token_type"],
)

# Fallback & Resilience Counters
FALLBACK_COUNT = Counter(
    "routemind_fallbacks_total",
    "Total fallback escalations triggered",
    ["primary_model", "fallback_model"],
)

RATE_LIMIT_REJECTIONS = Counter(
    "routemind_rate_limit_rejections_total",
    "Total rate limit 429 rejections",
    ["tenant"],
)

# Bandit Telemetry Gauges
BANDIT_ARM_PULLS = Counter(
    "routemind_bandit_arm_pulls_total",
    "Total selections per bandit arm",
    ["model_id"],
)

BANDIT_AVG_REWARD = Gauge(
    "routemind_bandit_avg_reward",
    "Running average reward per candidate model arm",
    ["model_id"],
)
