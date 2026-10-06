import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List

# Ensure backend path is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend")))

from app.core.constants import RoutingStrategy
from app.providers.registry import registry
from app.resiliency.fallback import fallback_engine
from app.routing.analyzer import RequestAnalyzer
from app.routing.bandit.bandit_router import bandit_router
from app.routing.bandit.reward import reward_evaluator
from app.routing.engine import router_engine
from app.schemas.chat import ChatCompletionRequest, ChatMessage


async def run_benchmark_for_strategy(
    strategy: RoutingStrategy, test_cases: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Evaluates a specific routing strategy over the benchmark dataset."""
    rewards: List[float] = []
    latencies: List[float] = []
    total_tokens = 0
    model_selections: Dict[str, int] = {}
    success_count = 0

    for item in test_cases:
        prompt = item["prompt"]
        req = ChatCompletionRequest(
            model="auto",
            messages=[ChatMessage(role="user", content=prompt)],
            routing_strategy=strategy,
        )

        start_time = time.perf_counter()
        features = RequestAnalyzer.analyze(req)

        # 1. Route
        router = router_engine.get_strategy(strategy)
        candidate_models = registry.list_models(free_only=True)
        selected_model = router.select_model(req, features, candidate_models)

        # Track arm distribution
        model_selections[selected_model.id] = model_selections.get(selected_model.id, 0) + 1

        # 2. Execute
        try:
            result = await fallback_engine.execute_resiliently(
                request=req,
                primary_model=selected_model,
                candidate_pool=candidate_models,
            )
            duration_ms = (time.perf_counter() - start_time) * 1000
            latencies.append(duration_ms)
            total_tokens += result.response.usage.total_tokens
            success_count += 1

            # 3. Compute Reward
            reward = reward_evaluator.compute_reward(
                features=features,
                response=result.response,
                latency_ms=duration_ms,
                is_success=True,
            )
            rewards.append(reward)

            # Online update for bandit
            if strategy == RoutingStrategy.BANDIT:
                bandit_router.bandit.update_arm(
                    arm_id=result.model.id,
                    feature_vector=features.to_feature_vector(),
                    reward=reward,
                )

        except Exception:
            duration_ms = (time.perf_counter() - start_time) * 1000
            latencies.append(duration_ms)
            rewards.append(-2.0)  # Failure penalty

    avg_reward = round(sum(rewards) / max(len(rewards), 1), 4)
    avg_latency = round(sum(latencies) / max(len(latencies), 1), 2)
    p95_latency = round(sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0.0, 2)
    success_rate = round((success_count / len(test_cases)) * 100, 1)

    return {
        "strategy": strategy.value,
        "total_requests": len(test_cases),
        "average_reward": avg_reward,
        "mean_latency_ms": avg_latency,
        "p95_latency_ms": p95_latency,
        "success_rate_pct": success_rate,
        "total_tokens": total_tokens,
        "model_distribution": model_selections,
    }


async def main():
    dataset_path = os.path.join(os.path.dirname(__file__), "datasets/benchmark_prompts.json")
    with open(dataset_path, "r") as f:
        test_cases = json.load(f)

    # Multiply dataset for sufficient online learning rounds
    multi_round_dataset = test_cases * 10

    strategies = [
        RoutingStrategy.RANDOM,
        RoutingStrategy.ROUND_ROBIN,
        RoutingStrategy.CHEAPEST,
        RoutingStrategy.LATENCY_FIRST,
        RoutingStrategy.QUALITY_FIRST,
        RoutingStrategy.RULE_BASED,
        RoutingStrategy.BANDIT,
    ]

    results: Dict[str, Any] = {}
    print("=" * 70)
    print(" RouteMind Empirical Research & Routing Strategy Benchmark")
    print("=" * 70)

    for strat in strategies:
        print(f"Running benchmark for [{strat.value}] over {len(multi_round_dataset)} requests...")
        metrics = await run_benchmark_for_strategy(strat, multi_round_dataset)
        results[strat.value] = metrics
        print(
            f" -> Avg Reward: {metrics['average_reward']:<7} | P95 Latency: {metrics['p95_latency_ms']}ms | Success: {metrics['success_rate_pct']}%"
        )

    # Compute Oracle & Regret
    max_avg_reward = max(m["average_reward"] for m in results.values())
    for strat_key, data in results.items():
        data["empirical_regret"] = round(max(0.0, max_avg_reward - data["average_reward"]), 4)

    # Save Results
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    summary_file = os.path.join(results_dir, "experiment_summary.json")

    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\nBenchmark complete. Saved results to experiments/results/experiment_summary.json")


if __name__ == "__main__":
    asyncio.run(main())
