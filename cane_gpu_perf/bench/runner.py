"""Benchmark runner - sends requests to inference backends and collects metrics."""

import asyncio
import time
import statistics

import httpx
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from cane_gpu_perf.config import BenchmarkConfig, BenchmarkResult
from cane_gpu_perf.utils.tokens import count_tokens

console = Console()

PROMPT_SETS = {
    "default": [
        "Explain the concept of neural network quantization and its tradeoffs.",
        "Write a Python function that implements binary search on a sorted list.",
        "What are the key differences between transformer and RNN architectures?",
    ],
    "conversation": [
        "Hello! Can you help me understand how LLMs work?",
        "What's the difference between fine-tuning and prompt engineering?",
        "Can you give me a practical example of RAG?",
        "How do I choose the right model size for my use case?",
        "Thanks! One more question - what about model distillation?",
    ],
    "rag": [
        (
            "Based on the following retrieved documents, answer the user's question.\n\n"
            "Document 1: Machine learning models require significant computational resources...\n"
            "Document 2: GPU acceleration has become essential for training deep learning models...\n"
            "Document 3: Inference optimization techniques include quantization, pruning, and distillation...\n\n"
            "Question: What are the best practices for optimizing ML inference costs?"
        ),
    ],
    "short": [
        "Summarize: GPUs accelerate matrix operations used in deep learning.",
        "Translate to French: The model is running efficiently.",
        "Complete: The main advantage of quantization is",
    ],
    "code": [
        "Write a Python function that sorts a list of dictionaries by a given key.",
        "Implement a simple LRU cache class in Python with get and put methods.",
        "Write a async HTTP client that retries failed requests with exponential backoff.",
    ],
}


def _get_prompts(config: BenchmarkConfig) -> list[str]:
    """Get prompts for the benchmark, cycling through the prompt set."""
    base_prompts = PROMPT_SETS.get(config.prompt_set, PROMPT_SETS["default"])

    # Pad context if context_length is specified
    if config.context_length:
        padded = []
        for p in base_prompts:
            current_tokens = count_tokens(p)
            if current_tokens < config.context_length:
                padding = " context" * ((config.context_length - current_tokens) // 2)
                padded.append(p + padding)
            else:
                padded.append(p)
        base_prompts = padded

    # Cycle prompts to fill num_requests
    prompts = []
    for i in range(config.num_requests):
        prompts.append(base_prompts[i % len(base_prompts)])
    return prompts


def _build_url(config: BenchmarkConfig) -> str:
    """Build the API endpoint URL based on backend."""
    if config.base_url:
        return config.base_url

    urls = {
        "openrouter": "https://openrouter.ai/api/v1/chat/completions",
        "openai": "https://api.openai.com/v1/chat/completions",
        "ollama": "http://localhost:11434/v1/chat/completions",
        "vllm": "http://localhost:8000/v1/chat/completions",
        "sglang": "http://localhost:30000/v1/chat/completions",
        "modal": "https://modal.com/v1/chat/completions",
    }
    return urls.get(config.backend, urls["openai"])


class BenchmarkRunner:
    """Runs benchmark requests against an inference backend."""

    async def run(self, config: BenchmarkConfig) -> BenchmarkResult:
        prompts = _get_prompts(config)
        url = _build_url(config)

        headers = {"Content-Type": "application/json"}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"
        elif config.backend == "openrouter":
            import os
            key = os.getenv("OPENROUTER_API_KEY", "")
            if key:
                headers["Authorization"] = f"Bearer {key}"
        elif config.backend == "openai":
            import os
            key = os.getenv("OPENAI_API_KEY", "")
            if key:
                headers["Authorization"] = f"Bearer {key}"

        semaphore = asyncio.Semaphore(config.concurrency)
        individual_results = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"[cyan]Benchmarking {config.backend} (c={config.concurrency})",
                total=config.num_requests,
            )

            async def _run_single(prompt: str) -> dict:
                async with semaphore:
                    payload = {
                        "model": config.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": config.max_tokens,
                        "temperature": config.temperature,
                        "stream": True,
                    }

                    result = {
                        "prompt_tokens": count_tokens(prompt),
                        "output_tokens": 0,
                        "ttft_ms": 0.0,
                        "latency_ms": 0.0,
                        "tokens_per_second": 0.0,
                        "error": None,
                    }

                    start = time.perf_counter()
                    first_token_time = None
                    output_text = ""

                    try:
                        async with httpx.AsyncClient(timeout=config.timeout) as client:
                            async with client.stream("POST", url, json=payload, headers=headers) as resp:
                                resp.raise_for_status()
                                async for line in resp.aiter_lines():
                                    if not line.startswith("data: "):
                                        continue
                                    data = line[6:]
                                    if data.strip() == "[DONE]":
                                        break

                                    if first_token_time is None:
                                        first_token_time = time.perf_counter()
                                        result["ttft_ms"] = (first_token_time - start) * 1000

                                    import json
                                    try:
                                        chunk = json.loads(data)
                                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                                        content = delta.get("content", "")
                                        if content:
                                            output_text += content
                                    except (json.JSONDecodeError, IndexError, KeyError):
                                        pass

                        end = time.perf_counter()
                        result["latency_ms"] = (end - start) * 1000
                        result["output_tokens"] = count_tokens(output_text)
                        if result["latency_ms"] > 0:
                            result["tokens_per_second"] = result["output_tokens"] / (result["latency_ms"] / 1000)

                    except Exception as e:
                        result["error"] = str(e)
                        result["latency_ms"] = (time.perf_counter() - start) * 1000

                    progress.advance(task)
                    return result

            tasks = [_run_single(p) for p in prompts]
            individual_results = await asyncio.gather(*tasks)

        return self._aggregate(config, list(individual_results))

    def _aggregate(self, config: BenchmarkConfig, results: list[dict]) -> BenchmarkResult:
        successful = [r for r in results if r["error"] is None]
        failed = [r for r in results if r["error"] is not None]

        if not successful:
            return BenchmarkResult(
                config=config,
                total_requests=len(results),
                failed_requests=len(failed),
            )

        latencies = [r["latency_ms"] for r in successful]
        ttfts = [r["ttft_ms"] for r in successful if r["ttft_ms"] > 0]
        tps_values = [r["tokens_per_second"] for r in successful if r["tokens_per_second"] > 0]

        total_tokens = sum(r["output_tokens"] for r in successful)
        total_time_s = max(r["latency_ms"] for r in successful) / 1000 if successful else 1

        def percentile(data, p):
            if not data:
                return 0.0
            sorted_data = sorted(data)
            idx = int(len(sorted_data) * p / 100)
            idx = min(idx, len(sorted_data) - 1)
            return sorted_data[idx]

        return BenchmarkResult(
            config=config,
            latency_p50=percentile(latencies, 50),
            latency_p95=percentile(latencies, 95),
            latency_p99=percentile(latencies, 99),
            ttft_p50=percentile(ttfts, 50),
            ttft_p95=percentile(ttfts, 95),
            ttft_p99=percentile(ttfts, 99),
            tokens_per_second_mean=statistics.mean(tps_values) if tps_values else 0.0,
            aggregate_tps=total_tokens / total_time_s if total_time_s > 0 else 0.0,
            requests_per_second=len(successful) / total_time_s if total_time_s > 0 else 0.0,
            total_requests=len(results),
            failed_requests=len(failed),
            individual_results=results,
        )
