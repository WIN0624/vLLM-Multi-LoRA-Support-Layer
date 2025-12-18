# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Simple client-side benchmark for routing traffic across static LoRA adapters
with a configurable popularity ratio.

Example:
    python benchmarks/multi_lora/benchmark_lora_popularity.py \
        --base-model baffo32/decapoda-research-llama-7B-hf \
        --tokenizer huggyllama/llama-7b \
        --hot-adapter hot-lora \
        --cold-adapters cold-lora1 cold-lora2 \
        --hot-ratio 0.0 \
        --request-rate 10 \
        --num-requests 20 \
        --trust-remote-code \
        --ignore-eos \
        --save_results \
        --result-dir benchmarks/multi_lora/test_results
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import argparse
import asyncio
import random
import time
from tqdm import tqdm
from collections import Counter
from dataclasses import dataclass
from typing import Optional
from collections.abc import AsyncGenerator

import pandas as pd
import numpy as np
import warnings
from transformers import PreTrainedTokenizerBase

from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.transformers_utils.tokenizer import get_tokenizer

from backend_request_func import (
    ASYNC_REQUEST_FUNCS,
    RequestFuncInput,
    RequestFuncOutput,
)

MILLISECONDS_TO_SECONDS_CONVERSION = 1000

@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    request_throughput: float
    output_throughput: float
    total_token_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    percentiles_ttft_ms: list[tuple[float, float]]
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    percentiles_tpot_ms: list[tuple[float, float]]
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    percentiles_itl_ms: list[tuple[float, float]]
    # E2EL stands for end-to-end latency per request.
    # It is the time taken on the client side from sending
    # a request to receiving a complete response.
    mean_e2el_ms: float
    median_e2el_ms: float
    std_e2el_ms: float
    percentiles_e2el_ms: list[tuple[float, float]]

@dataclass
class SampleRequest:
    """A class representing a single inference request for benchmarking.

    Attributes:
        prompt: The input text prompt for the model.
        multi_modal_data: Optional dictionary containing multi-modal data (e.g.
            images).
        prompt_len: The length of the prompt in tokens.
        expected_output_len: The expected length of the output in tokens.
    """
    model_name: str
    prompt: str
    prompt_len: int
    expected_output_len: int
    completion: str = None

def sample_requests(
    tokenizer: PreTrainedTokenizerBase, args: argparse.Namespace
) -> list[SampleRequest]:
    requests: list[SampleRequest] = []
    rng = random.Random(args.seed)
    
    def choose_adapter():
        if rng.random() < args.hot_ratio:
            return args.hot_adapter
        return args.cold_adapters[rng.randrange(len(args.cold_adapters))]
    
    # Use seeded random for unique ID instead of uuid
    def generate_unique_id():
        return ''.join(rng.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=16))
    
    for i in range(args.num_requests):
        adapter_name = choose_adapter()
        unique_id = generate_unique_id()
        base_text = "Hello " * (args.prompt_len - 10)
        prompt = f"{unique_id} {base_text}"
        request = SampleRequest(
                model_name=adapter_name,
                prompt=prompt,
                prompt_len=len(tokenizer(prompt)['input_ids']),
                expected_output_len=args.output_len,
            )
        requests.append(request)

    return requests

def calculate_metrics(
    input_requests: list[tuple[str, int, int]],
    outputs: list[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
    selected_percentile_metrics: list[str],
    selected_percentiles: list[float],
) -> tuple[BenchmarkMetrics, list[int]]:
    actual_output_lens: list[int] = []
    total_input = 0
    completed = 0
    itls: list[float] = []
    tpots: list[float] = []
    all_tpots: list[float] = []
    ttfts: list[float] = []
    e2els: list[float] = []
    for i in range(len(outputs)):
        if outputs[i].success:
            # We use the tokenizer to count the number of output tokens for all
            # serving backends instead of looking at len(outputs[i].itl) since
            # multiple output tokens may be bundled together
            # Note : this may inflate the output token count slightly
            output_len = len(
                tokenizer(outputs[i].generated_text, add_special_tokens=False).input_ids
            )
            actual_output_lens.append(output_len)
            total_input += input_requests[i].prompt_len
            tpot = 0
            if output_len > 1:
                latency_minus_ttft = outputs[i].latency - outputs[i].ttft
                tpot = latency_minus_ttft / (output_len - 1)
                tpots.append(tpot)
            outputs[i].tpot = tpot
            # Note: if output_len <= 1, we regard tpot as 0 for goodput
            all_tpots.append(tpot)
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)
            e2els.append(outputs[i].latency)
            completed += 1
        else:
            actual_output_lens.append(0)

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2,
        )
    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=sum(actual_output_lens),
        request_throughput=completed / dur_s,
        output_throughput=sum(actual_output_lens) / dur_s,
        total_token_throughput=(total_input + sum(actual_output_lens)) / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0)
        * 1000,  # ttfts is empty if streaming is not supported by backend
        std_ttft_ms=np.std(ttfts or 0) * 1000,
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        percentiles_ttft_ms=[
            (p, np.percentile(ttfts or 0, p) * 1000) for p in selected_percentiles
        ],
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        std_tpot_ms=np.std(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        percentiles_tpot_ms=[
            (p, np.percentile(tpots or 0, p) * 1000) for p in selected_percentiles
        ],
        mean_itl_ms=np.mean(itls or 0) * 1000,
        std_itl_ms=np.std(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        percentiles_itl_ms=[
            (p, np.percentile(itls or 0, p) * 1000) for p in selected_percentiles
        ],
        mean_e2el_ms=np.mean(e2els or 0) * 1000,
        std_e2el_ms=np.std(e2els or 0) * 1000,
        median_e2el_ms=np.median(e2els or 0) * 1000,
        percentiles_e2el_ms=[
            (p, np.percentile(e2els or 0, p) * 1000) for p in selected_percentiles
        ],
    )

    return metrics, actual_output_lens


def calculate_metrics_per_lora(
    input_requests: list[SampleRequest],
    outputs: list[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
    selected_percentiles: list[float],
) -> dict[str, BenchmarkMetrics]:
    """Calculate metrics grouped by LoRA adapter name, returning BenchmarkMetrics objects."""
    # Group requests and outputs by LoRA adapter
    lora_groups: dict[str, dict] = {}
    
    for i, (request, output) in enumerate(zip(input_requests, outputs)):
        lora_name = request.model_name
        if lora_name not in lora_groups:
            lora_groups[lora_name] = {
                "requests": [],
                "outputs": [],
            }
        lora_groups[lora_name]["requests"].append(request)
        lora_groups[lora_name]["outputs"].append(output)
    
    # Calculate metrics for each LoRA adapter
    per_lora_metrics: dict[str, BenchmarkMetrics] = {}
    
    for lora_name, group in lora_groups.items():
        requests = group["requests"]
        group_outputs = group["outputs"]
        
        actual_output_lens: list[int] = []
        total_input = 0
        completed = 0
        itls: list[float] = []
        tpots: list[float] = []
        ttfts: list[float] = []
        e2els: list[float] = []
        
        for j, output in enumerate(group_outputs):
            if output.success:
                output_len = len(
                    tokenizer(output.generated_text, add_special_tokens=False).input_ids
                )
                actual_output_lens.append(output_len)
                total_input += requests[j].prompt_len
                tpot = 0
                if output_len > 1:
                    latency_minus_ttft = output.latency - output.ttft
                    tpot = latency_minus_ttft / (output_len - 1)
                    tpots.append(tpot)
                itls += output.itl
                ttfts.append(output.ttft)
                e2els.append(output.latency)
                completed += 1
            else:
                actual_output_lens.append(0)
        
        # Create BenchmarkMetrics object (same structure as main benchmark)
        per_lora_metrics[lora_name] = BenchmarkMetrics(
            completed=completed,
            total_input=total_input,
            total_output=sum(actual_output_lens),
            request_throughput=completed / dur_s if dur_s > 0 else 0,
            output_throughput=sum(actual_output_lens) / dur_s if dur_s > 0 else 0,
            total_token_throughput=(total_input + sum(actual_output_lens)) / dur_s if dur_s > 0 else 0,
            mean_ttft_ms=float(np.mean(ttfts or 0) * 1000),
            std_ttft_ms=float(np.std(ttfts or 0) * 1000),
            median_ttft_ms=float(np.median(ttfts or 0) * 1000),
            percentiles_ttft_ms=[
                (p, float(np.percentile(ttfts or 0, p) * 1000)) for p in selected_percentiles
            ],
            mean_tpot_ms=float(np.mean(tpots or 0) * 1000),
            std_tpot_ms=float(np.std(tpots or 0) * 1000),
            median_tpot_ms=float(np.median(tpots or 0) * 1000),
            percentiles_tpot_ms=[
                (p, float(np.percentile(tpots or 0, p) * 1000)) for p in selected_percentiles
            ],
            mean_itl_ms=float(np.mean(itls or 0) * 1000),
            std_itl_ms=float(np.std(itls or 0) * 1000),
            median_itl_ms=float(np.median(itls or 0) * 1000),
            percentiles_itl_ms=[
                (p, float(np.percentile(itls or 0, p) * 1000)) for p in selected_percentiles
            ],
            mean_e2el_ms=float(np.mean(e2els or 0) * 1000),
            std_e2el_ms=float(np.std(e2els or 0) * 1000),
            median_e2el_ms=float(np.median(e2els or 0) * 1000),
            percentiles_e2el_ms=[
                (p, float(np.percentile(e2els or 0, p) * 1000)) for p in selected_percentiles
            ],
        )
    
    return per_lora_metrics

async def get_request(
    input_requests: list[SampleRequest],
    request_rate: float,
    burstiness: float = 1.0,
) -> AsyncGenerator[tuple[int, SampleRequest], None]:
    """
    Asynchronously generates requests at a specified rate
    with OPTIONAL burstiness.

    Args:
        input_requests:
            A list of input requests, each represented as a tuple.
        request_rate:
            The rate at which requests are generated (requests/s).
        burstiness (optional):
            The burstiness factor of the request generation.
            Only takes effect when request_rate is not inf.
            Default value is 1, which follows a Poisson process.
            Otherwise, the request intervals follow a gamma distribution.
            A lower burstiness value (0 < burstiness < 1) results
            in more bursty requests, while a higher burstiness value
            (burstiness > 1) results in a more uniform arrival of requests.
    """
    input_requests = iter(input_requests)

    # Calculate scale parameter theta to maintain the desired request_rate.
    assert burstiness > 0, (
        f"A positive burstiness factor is expected, but given {burstiness}."
    )
    theta = 1.0 / (request_rate * burstiness)

    for i, request in enumerate(input_requests):
        yield i, request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue

        # Sample the request interval from the gamma distribution.
        # If burstiness is 1, it follows exponential distribution.
        interval = np.random.gamma(shape=burstiness, scale=theta)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)

async def benchmark(
    backend: str,
    api_url: str,
    base_url: str,
    tokenizer: PreTrainedTokenizerBase,
    input_requests: list[SampleRequest],
    request_rate: float,
    burstiness: float,
    disable_tqdm: bool,
    profile: bool,
    selected_percentile_metrics: list[str],
    selected_percentiles: list[str],
    ignore_eos: bool,
    max_concurrency: Optional[int],
    hot_adapter: str,
    cold_adapters: list[str],
    hot_ratio: float,
):
    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS[backend]
    else:
        raise ValueError(f"Unknown backend: {backend}")

    print(f"Using backend: {backend}")
    print(f"Using hot adapter: {hot_adapter}")
    print(f"Using cold adapters: {cold_adapters}")
    print(f"Using hot ratio: {hot_ratio}\n")

    # No warmup - directly start benchmark to measure cold-start latency
    print("Starting benchmark (no warmup)...")

    if profile:
        print("Starting profiler...")
        first_request = input_requests[0]
        profile_input = RequestFuncInput(
            model=first_request.model_name,
            prompt=first_request.prompt,
            api_url=base_url + "/start_profile",
            prompt_len=first_request.prompt_len,
            output_len=first_request.expected_output_len,
            ignore_eos=ignore_eos,
        )
        profile_output = await request_func(request_func_input=profile_input)
        if profile_output.success:
            print("Profiler started")

    distribution = "Poisson process" if burstiness == 1.0 else "Gamma distribution"

    print(f"Traffic request rate: {request_rate}")
    print(f"Burstiness factor: {burstiness} ({distribution})")
    print(f"Maximum request concurrency: {max_concurrency}")

    pbar = None if disable_tqdm else tqdm(total=len(input_requests))

    # This can be used once the minimum Python version is 3.10 or higher,
    # and it will simplify the code in limited_request_func.
    #    semaphore = (asyncio.Semaphore(max_concurrency)
    #                 if max_concurrency else contextlib.nullcontext())
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    async def limited_request_func(request_func_input, pbar):
        if semaphore is None:
            return await request_func(request_func_input=request_func_input, pbar=pbar)
        async with semaphore:
            return await request_func(request_func_input=request_func_input, pbar=pbar)

    benchmark_start_time = time.perf_counter()
    tasks: list[asyncio.Task] = []
    expected: list[str] = []
    async for i, request in get_request(input_requests, request_rate, burstiness):
        request_func_input = RequestFuncInput(
            model=request.model_name,
            prompt=request.prompt,
            api_url=api_url,
            prompt_len=request.prompt_len,
            output_len=request.expected_output_len,
            ignore_eos=ignore_eos,
        )
        expected.append(request.completion)
        tasks.append(
            asyncio.create_task(
                limited_request_func(request_func_input=request_func_input, pbar=pbar)
            )
        )
    outputs: list[RequestFuncOutput] = await asyncio.gather(*tasks)

    if pbar is not None:
        pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time

    metrics, actual_output_lens = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
        selected_percentile_metrics=selected_percentile_metrics,
        selected_percentiles=selected_percentiles,
    )

    print("{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total generated tokens:", metrics.total_output))
    print(
        "{:<40} {:<10.2f}".format(
            "Request throughput (req/s):", metrics.request_throughput
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Output token throughput (tok/s):", metrics.output_throughput
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Total Token throughput (tok/s):", metrics.total_token_throughput
        )
    )

    result = {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "output_throughput": metrics.output_throughput,
        "total_token_throughput": metrics.total_token_throughput,
        "ttft_description": pd.Series([output.ttft for output in outputs])
        .describe()
        .to_dict(),
        "tpot_description": pd.Series([output.tpot for output in outputs])
        .describe()
        .to_dict(),
        "input_lens": [output.prompt_len for output in outputs],
        "output_lens": actual_output_lens,
        "ttfts": [output.ttft for output in outputs],
        "itls": [output.itl for output in outputs],
        "errors": [output.error for output in outputs],
    }

    ret = [
        {"generated": output.generated_text, "expected": gt}
        for output, gt in zip(outputs, expected)
    ]

    def process_one_metric(
        # E.g., "ttft"
        metric_attribute_name: str,
        # E.g., "TTFT"
        metric_name: str,
        # E.g., "Time to First Token"
        metric_header: str,
    ):
        # This function prints and adds statistics of the specified
        # metric.
        if metric_attribute_name not in selected_percentile_metrics:
            return
        print("{s:{c}^{n}}".format(s=metric_header, n=50, c="-"))
        print(
            "{:<40} {:<10.2f}".format(
                f"Mean {metric_name} (ms):",
                getattr(metrics, f"mean_{metric_attribute_name}_ms"),
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                f"Median {metric_name} (ms):",
                getattr(metrics, f"median_{metric_attribute_name}_ms"),
            )
        )
        result[f"mean_{metric_attribute_name}_ms"] = getattr(
            metrics, f"mean_{metric_attribute_name}_ms"
        )
        result[f"median_{metric_attribute_name}_ms"] = getattr(
            metrics, f"median_{metric_attribute_name}_ms"
        )
        result[f"std_{metric_attribute_name}_ms"] = getattr(
            metrics, f"std_{metric_attribute_name}_ms"
        )
        for p, value in getattr(metrics, f"percentiles_{metric_attribute_name}_ms"):
            p_word = str(int(p)) if int(p) == p else str(p)
            print("{:<40} {:<10.2f}".format(f"P{p_word} {metric_name} (ms):", value))
            result[f"p{p_word}_{metric_attribute_name}_ms"] = value

    process_one_metric("ttft", "TTFT", "Time to First Token")
    process_one_metric("tpot", "TPOT", "Time per Output Token (excl. 1st token)")
    process_one_metric("itl", "ITL", "Inter-token Latency")
    process_one_metric("e2el", "E2EL", "End-to-end Latency")

    print("=" * 50)

    # Calculate and print per-LoRA metrics
    per_lora_metrics = calculate_metrics_per_lora(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
        selected_percentiles=selected_percentiles,
    )
    
    # Print per-LoRA metrics using the same process_one_metric pattern
    sorted_adapters = sorted(
        per_lora_metrics.keys(),
        key=lambda x: (0 if x == hot_adapter else 1, x)
    )
    
    for lora_name in sorted_adapters:
        lora_metrics = per_lora_metrics[lora_name]
        adapter_type = "[HOT]" if lora_name == hot_adapter else "[COLD]"
        
        print("{s:{c}^{n}}".format(s=f" {lora_name} {adapter_type} ", n=50, c="="))
        print("{:<40} {:<10}".format("Successful requests:", lora_metrics.completed))
        print("{:<40} {:<10}".format("Total input tokens:", lora_metrics.total_input))
        print("{:<40} {:<10}".format("Total generated tokens:", lora_metrics.total_output))
        print("{:<40} {:<10.2f}".format("Request throughput (req/s):", lora_metrics.request_throughput))
        print("{:<40} {:<10.2f}".format("Output token throughput (tok/s):", lora_metrics.output_throughput))
        
        # Reuse process_one_metric logic for each LoRA
        def process_lora_metric(metric_attribute_name, metric_name, metric_header):
            if metric_attribute_name not in selected_percentile_metrics:
                return
            print("{s:{c}^{n}}".format(s=metric_header, n=50, c="-"))
            print("{:<40} {:<10.2f}".format(
                f"Mean {metric_name} (ms):",
                getattr(lora_metrics, f"mean_{metric_attribute_name}_ms"),
            ))
            print("{:<40} {:<10.2f}".format(
                f"Median {metric_name} (ms):",
                getattr(lora_metrics, f"median_{metric_attribute_name}_ms"),
            ))
            for p, value in getattr(lora_metrics, f"percentiles_{metric_attribute_name}_ms"):
                p_word = str(int(p)) if int(p) == p else str(p)
                print("{:<40} {:<10.2f}".format(f"P{p_word} {metric_name} (ms):", value))
        
        process_lora_metric("ttft", "TTFT", "Time to First Token")
        process_lora_metric("tpot", "TPOT", "Time per Output Token (excl. 1st token)")
        process_lora_metric("itl", "ITL", "Inter-token Latency")
        process_lora_metric("e2el", "E2EL", "End-to-end Latency")
        print("=" * 50)
    
    # Add per-LoRA metrics to result
    result["per_lora_metrics"] = {
        lora_name: {
            "completed": m.completed,
            "total_input": m.total_input,
            "total_output": m.total_output,
            "request_throughput": m.request_throughput,
            "output_throughput": m.output_throughput,
            "total_token_throughput": m.total_token_throughput,
            "mean_ttft_ms": m.mean_ttft_ms,
            "median_ttft_ms": m.median_ttft_ms,
            "mean_tpot_ms": m.mean_tpot_ms,
            "median_tpot_ms": m.median_tpot_ms,
            "mean_itl_ms": m.mean_itl_ms,
            "median_itl_ms": m.median_itl_ms,
            "mean_e2el_ms": m.mean_e2el_ms,
            "median_e2el_ms": m.median_e2el_ms,
        }
        for lora_name, m in per_lora_metrics.items()
    }

    return result, ret

def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    backend = args.backend
    base_model = args.base_model
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.base_model
    
    api_url = f"http://{args.host}:{args.port}{args.endpoint}"
    base_url = f"http://{args.host}:{args.port}"

    tokenizer = get_tokenizer(
        tokenizer_id,
        trust_remote_code=args.trust_remote_code,
        tokenizer_mode=args.tokenizer_mode,
    )

    if args.save_results:
        result_file_name = f"{args.backend}"
        result_file_name += f"_{args.request_rate}qps"
        result_file_name += f"_{args.num_requests}"
        result_file_name += f"_hot_ratio{args.hot_ratio}"
        result_file_name += f"_out{args.output_len}"
        result_file_name += ".json"
    else:
        result_file_name = None

    input_requests = sample_requests(tokenizer, args)

    benchmark_result, ret = asyncio.run(
        benchmark(
            backend=backend,
            api_url=api_url,
            base_url=base_url,
            tokenizer=tokenizer,
            input_requests=input_requests,
            request_rate=args.request_rate,
            burstiness=args.burstiness,
            disable_tqdm=args.disable_tqdm,
            profile=args.profile,
            selected_percentile_metrics=args.percentile_metrics.split(","),
            selected_percentiles=[float(p) for p in args.metric_percentiles.split(",")],
            ignore_eos=args.ignore_eos,
            max_concurrency=args.max_concurrency,
            hot_adapter=args.hot_adapter,
            cold_adapters=args.cold_adapters,
            hot_ratio=args.hot_ratio,
        )
    )

    # Save config and results to json
    if args.save_results:
        results = {
            "backend": backend,
            "base_model": base_model,
            "tokenizer_id": tokenizer_id,
            "num_prompts": args.num_requests,
            "request_rate": args.request_rate
            if args.request_rate < float("inf")
            else "inf",
            "burstiness": args.burstiness,
            "max_concurrency": args.max_concurrency,
        }
        results = {"outputs": ret, **results, **benchmark_result}

        # Save to file
        if args.result_filename:
            result_file_name = args.result_filename
        if args.result_dir:
            result_file_name = os.path.join(args.result_dir, result_file_name)
        with open(result_file_name, "w", encoding="utf-8") as outfile:
            json.dump(results, outfile, indent=4)


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark the online serving throughput."
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server or API base url if not using http host and port.",
    )
    # Use 127.0.0.1 here instead of localhost to force the use of ipv4
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/completions",
        help="API endpoint.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Maximum number of concurrent requests. This can be used "
        "to help simulate an environment where a higher level component "
        "is enforcing a maximum number of concurrent requests. While the "
        "--request-rate argument controls the rate at which requests are "
        "initiated, this argument will control how many are actually allowed "
        "to execute at a time. This means that when used in combination, the "
        "actual request rate may be lower than specified with --request-rate, "
        "if the server is not processing requests fast enough to keep up.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Name of the base_model.",
    )
    parser.add_argument(
        "--hot-adapter", 
        type=str, 
        required=True,
        help="Name of the hot LoRA adapter.",
    )
    parser.add_argument(
        "--cold-adapters", 
        nargs="+", 
        required=True,
        help="Names of the cold LoRA adapters.",
    )
    parser.add_argument(
        "--hot-ratio", 
        type=float, 
        default=0.3,
        help="The ratio of requests that go to the hot adapter.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Name or path of the tokenizer, if not using the default tokenizer.",  # noqa: E501
    )
    parser.add_argument(
        "--tokenizer-mode",
        type=str,
        default="auto",
        help="Name or path of the tokenizer, if not using the default tokenizer.",  # noqa: E501
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=200,
        help="Number of requests to process.",
    )
    parser.add_argument(
        "--prompt-len",
        type=int,
        default=400,
        help="Number of prompt tokens.",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=128,
        help="Number of output tokens.",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process or gamma distribution "
        "to synthesize the request arrival times.",
    )
    parser.add_argument(
        "--burstiness",
        type=float,
        default=1.0,
        help="Burstiness factor of the request generation. "
        "Only take effect when request_rate is not inf. "
        "Default value is 1, which follows Poisson process. "
        "Otherwise, the request intervals follow a gamma distribution. "
        "A lower burstiness value (0 < burstiness < 1) results in more "
        "bursty requests. A higher burstiness value (burstiness > 1) "
        "results in a more uniform arrival of requests.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from huggingface",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Specify to disable tqdm progress bar.",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Specify to save benchmark results to a json file",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Use Torch Profiler. The endpoint must be launched with "
        "VLLM_TORCH_PROFILER_DIR to enable profiler.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help="Specify directory to save benchmark json results."
        "If not specified, results are saved in the current directory.",
    )
    parser.add_argument(
        "--result-filename",
        type=str,
        default=None,
        help="Specify the filename to save benchmark json results."
        "If not specified, results will be saved in "
        "{backend}-{args.request_rate}qps-{base_model_id}-{current_dt}.json"
        " format.",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Set ignore_eos flag when sending the benchmark request."
        "Warning: ignore_eos is not supported in deepspeed_mii and tgi.",
    )
    parser.add_argument(
        "--percentile-metrics",
        type=str,
        default="ttft,tpot,itl,e2el",
        help="Comma-separated list of selected metrics to report percentils. "
        "This argument specifies the metrics to report percentiles. "
        'Allowed metric names are "ttft", "tpot", "itl", "e2el". '
        'Default value is "ttft,tpot,itl".',
    )
    parser.add_argument(
        "--metric-percentiles",
        type=str,
        default="99",
        help="Comma-separated list of percentiles for selected metrics. "
        'To report 25-th, 50-th, and 75-th percentiles, use "25,50,75". '
        'Default value is "99". '
        'Use "--percentile-metrics" to select metrics.',
    )
    
    args = parser.parse_args()
    main(args)