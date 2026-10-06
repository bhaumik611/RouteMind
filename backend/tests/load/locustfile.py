import random
from locust import HttpUser, between, task


class RouteMindLoadTester(HttpUser):
    wait_time = between(0.1, 0.5)

    prompts = [
        "Explain the raft consensus protocol.",
        "Write a python quicksort implementation with comments.",
        "Summarize the differences between SQL and NoSQL in two sentences.",
        "Extract dates and currency values from this invoice text.",
        "In one sentence, what is exponential backoff?",
    ]

    strategies = ["bandit", "rule_based", "latency_first", "round_robin"]

    @task(5)
    def test_auto_completions(self):
        payload = {
            "model": "auto",
            "messages": [{"role": "user", "content": random.choice(self.prompts)}],
            "routing_strategy": random.choice(self.strategies),
        }
        headers = {"X-Tenant-ID": f"tenant_{random.randint(1, 5)}"}
        with self.client.post("/v1/chat/completions", json=payload, headers=headers, catch_response=True) as res:
            if res.status_code in (200, 429):
                res.success()
            else:
                res.failure(f"Unexpected status: {res.status_code}")

    @task(1)
    def test_health_check(self):
        self.client.get("/v1/health")

    @task(1)
    def test_usage_check(self):
        self.client.get("/v1/usage?tenant_id=tenant_1")
