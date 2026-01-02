#!/usr/bin/env python3
"""
Model Router Selection Test Script

Tests various parameters to understand how Azure AI Foundry's model-router
selects underlying models (GPT-4o-mini, GPT-5-mini, Claude Sonnet, etc.)

Goal: Find the parameter combination that triggers Claude Sonnet selection
for narrative and structure generation tasks.

Uses the official Azure AI Inference SDK as per Azure documentation.

Usage:
    pip install azure-ai-inference
    python scripts/test_model_router_selection.py
"""

import os
import sys
import json
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

# Load .env file
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# Azure AI Inference SDK (official)
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

# Configuration from .env
API_KEY = os.getenv("FOUNDRY_API_KEY", "")
API_VERSION = os.getenv("FOUNDRY_API_VERSION", "2024-12-01-preview")
BASE_ENDPOINT = os.getenv("FOUNDRY_ENDPOINT", "").rstrip("/")

# Full endpoint for model-router (as per Azure docs)
ENDPOINT = f"{BASE_ENDPOINT}/openai/deployments/model-router"
MODEL_NAME = "model-router"


# Test prompts of varying complexity
SIMPLE_PROMPT = "What is 2+2?"

MEDIUM_PROMPT = """Write a short paragraph about Vikings for a 7-year-old child.
Include one educational fact about Viking ships."""

COMPLEX_PROMPT = """You are a master storyteller creating a children's story.

Create a detailed chapter outline for a Viking story with the following requirements:
- Target age: 7 years old
- Theme: Courage and unity
- Include 3 characters with personality traits
- Include educational content about Viking culture
- Write in a style inspired by Guillermo del Toro's world-building approach

Output as JSON with: title, theme, characters[], educational_points[]"""

VERY_COMPLEX_PROMPT = """You are Story Architect & World-Builder, embodying Guillermo del Toro's visionary approach.

Create a COMPLETE 5-chapter story structure for a Norwegian Viking tale about Harald Fairhair.

Requirements:
1. Each chapter needs: title, 200-word synopsis, characters featured, educational points
2. Include character arc milestones for the protagonist across all chapters
3. Integrate historical facts about Viking culture, food, music, and social structure
4. World-building notes: mythology, visual aesthetic, emotional atmosphere
5. Series potential: plant seeds for future books

Target: 7-year-old Norwegian child who wants to learn about their cultural heritage.

Output comprehensive JSON following professional story structure standards."""


def test_model_selection(
    client: ChatCompletionsClient,
    test_name: str,
    prompt: str,
    max_tokens: int = 1000,
    temperature: float = 0.7,
    top_p: float = 0.95,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
) -> Dict[str, Any]:
    """
    Test model-router with specific parameters and return which model was selected.
    """
    try:
        start_time = datetime.now()

        response = client.complete(
            messages=[
                SystemMessage(content="You are a helpful assistant."),
                UserMessage(content=prompt)
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            model=MODEL_NAME
        )

        elapsed = (datetime.now() - start_time).total_seconds()

        # Extract model info
        model_used = response.model
        usage = response.usage

        result = {
            "test_name": test_name,
            "status": "SUCCESS",
            "model_selected": model_used,
            "is_claude": "claude" in model_used.lower() if model_used else False,
            "is_gpt5": "gpt-5" in model_used.lower() if model_used else False,
            "is_gpt4o": "gpt-4o" in model_used.lower() if model_used else False,
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
            "elapsed_seconds": elapsed,
            "params": {
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "presence_penalty": presence_penalty,
                "frequency_penalty": frequency_penalty,
                "prompt_length": len(prompt),
            }
        }

        return result

    except Exception as e:
        return {
            "test_name": test_name,
            "status": "ERROR",
            "error": str(e),
            "params": {
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
        }


def run_all_tests():
    """Run comprehensive tests to understand model-router selection logic."""

    if not API_KEY or not BASE_ENDPOINT:
        print("ERROR: Missing Foundry configuration in .env file")
        print("Required variables:")
        print("  FOUNDRY_ENDPOINT=https://your-endpoint.cognitiveservices.azure.com/")
        print("  FOUNDRY_API_KEY=your-api-key")
        print(f"\nLooked for .env at: {env_path}")
        print(f"FOUNDRY_ENDPOINT: {'✅ Set' if BASE_ENDPOINT else '❌ Missing'}")
        print(f"FOUNDRY_API_KEY: {'✅ Set' if API_KEY else '❌ Missing'}")
        return

    print("=" * 80)
    print(" MODEL ROUTER SELECTION TEST")
    print(" Using Azure AI Inference SDK (official)")
    print("=" * 80)
    print(f" Endpoint: {ENDPOINT}")
    print(f" API Version: {API_VERSION}")
    print(" Goal: Find parameters that trigger Claude Sonnet selection")
    print("=" * 80)
    print()

    # Create client using official Azure AI Inference SDK
    client = ChatCompletionsClient(
        endpoint=ENDPOINT,
        credential=AzureKeyCredential(API_KEY),
        api_version=API_VERSION,
    )

    results = []

    # Test matrix - focusing on parameters that might influence model selection
    tests = [
        # Group 1: Baseline - Simple prompt, varying parameters
        ("1a: Simple prompt, default params", SIMPLE_PROMPT, 500, 0.7, 0.95, 0.0, 0.0),
        ("1b: Simple prompt, high temp", SIMPLE_PROMPT, 500, 1.0, 0.95, 0.0, 0.0),
        ("1c: Simple prompt, more tokens", SIMPLE_PROMPT, 2000, 0.7, 0.95, 0.0, 0.0),

        # Group 2: Medium prompt complexity
        ("2a: Medium prompt, 1K tokens", MEDIUM_PROMPT, 1000, 0.7, 0.95, 0.0, 0.0),
        ("2b: Medium prompt, 4K tokens", MEDIUM_PROMPT, 4000, 0.7, 0.95, 0.0, 0.0),
        ("2c: Medium prompt, high creativity", MEDIUM_PROMPT, 2000, 0.9, 0.95, 0.2, 0.4),

        # Group 3: Complex prompt (story structure)
        ("3a: Complex prompt, 2K tokens", COMPLEX_PROMPT, 2000, 0.75, 0.95, 0.0, 0.0),
        ("3b: Complex prompt, 4K tokens", COMPLEX_PROMPT, 4000, 0.75, 0.95, 0.0, 0.0),
        ("3c: Complex prompt, 8K tokens", COMPLEX_PROMPT, 8000, 0.75, 0.95, 0.0, 0.0),
        ("3d: Complex prompt, creative params", COMPLEX_PROMPT, 4000, 0.85, 0.95, 0.25, 0.4),

        # Group 4: Very complex prompt (full story structure - like StructureAgent)
        ("4a: VeryComplex, 4K tokens", VERY_COMPLEX_PROMPT, 4000, 0.75, 0.95, 0.0, 0.0),
        ("4b: VeryComplex, 8K tokens", VERY_COMPLEX_PROMPT, 8000, 0.75, 0.95, 0.0, 0.0),
        ("4c: VeryComplex, 12K tokens", VERY_COMPLEX_PROMPT, 12000, 0.75, 0.95, 0.0, 0.0),
        ("4d: VeryComplex, 16K tokens", VERY_COMPLEX_PROMPT, 16000, 0.75, 0.95, 0.0, 0.0),

        # Group 5: High creativity settings (like NarrativeAgent)
        ("5a: VeryComplex, narrative params", VERY_COMPLEX_PROMPT, 8000, 0.85, 0.95, 0.2, 0.4),
        ("5b: VeryComplex, max creativity", VERY_COMPLEX_PROMPT, 8000, 1.0, 0.99, 0.3, 0.5),

        # Group 6: Extreme token requests
        ("6a: Complex, 16K tokens", COMPLEX_PROMPT, 16000, 0.7, 0.95, 0.0, 0.0),
        ("6b: VeryComplex, 16K + creative", VERY_COMPLEX_PROMPT, 16000, 0.85, 0.95, 0.25, 0.4),
    ]

    for i, (test_name, prompt, max_tokens, temp, top_p, pres_pen, freq_pen) in enumerate(tests):
        print(f"[{i+1}/{len(tests)}] Running: {test_name}")

        result = test_model_selection(
            client,
            test_name=test_name,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temp,
            top_p=top_p,
            presence_penalty=pres_pen,
            frequency_penalty=freq_pen,
        )

        results.append(result)

        if result["status"] == "SUCCESS":
            model = result["model_selected"]
            is_claude = "✅ CLAUDE" if result["is_claude"] else ""
            is_gpt5 = "🔷 GPT-5" if result["is_gpt5"] else ""
            is_gpt4o = "⚪ GPT-4o" if result["is_gpt4o"] else ""
            marker = is_claude or is_gpt5 or is_gpt4o or "⚫ other"
            print(f"    → Model: {model} {marker}")
            print(f"    → Tokens: {result['total_tokens']} | Time: {result['elapsed_seconds']:.1f}s")
        else:
            error_msg = result.get('error', 'Unknown error')[:100]
            print(f"    → ERROR: {error_msg}...")

        # Delay to avoid rate limits
        import time
        time.sleep(2)

    # Summary
    print()
    print("=" * 80)
    print(" RESULTS SUMMARY")
    print("=" * 80)
    print()

    # Group by model selected
    model_counts = {}
    claude_tests = []
    gpt5_tests = []
    gpt4o_tests = []

    for r in results:
        if r["status"] == "SUCCESS":
            model = r["model_selected"]
            model_counts[model] = model_counts.get(model, 0) + 1

            if r["is_claude"]:
                claude_tests.append(r)
            elif r["is_gpt5"]:
                gpt5_tests.append(r)
            elif r["is_gpt4o"]:
                gpt4o_tests.append(r)

    print("Model Selection Distribution:")
    for model, count in sorted(model_counts.items(), key=lambda x: -x[1]):
        marker = ""
        if "claude" in model.lower():
            marker = " ✅ TARGET"
        elif "gpt-5" in model.lower():
            marker = " 🔷 REASONING"
        elif "gpt-4o-mini" in model.lower():
            marker = " ⚪ CHEAP"
        print(f"  {model}: {count} times{marker}")

    print()

    if claude_tests:
        print("✅ CLAUDE SONNET was selected in these tests:")
        for r in claude_tests:
            p = r['params']
            print(f"  - {r['test_name']}")
            print(f"    max_tokens={p['max_tokens']}, temp={p['temperature']}, prompt_len={p['prompt_length']}")
    else:
        print("❌ Claude Sonnet was NOT selected in any test")

    print()

    if gpt5_tests:
        print("🔷 GPT-5 (reasoning) was selected in these tests:")
        for r in gpt5_tests:
            p = r['params']
            print(f"  - {r['test_name']}")
            print(f"    max_tokens={p['max_tokens']}, temp={p['temperature']}, prompt_len={p['prompt_length']}")

    # Save detailed results to outputs folder
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"model_router_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print()
    print(f"📄 Detailed results saved to: {output_file}")

    # Analysis and Recommendations
    print()
    print("=" * 80)
    print(" ANALYSIS & RECOMMENDATIONS")
    print("=" * 80)
    print()

    if claude_tests:
        print("To trigger Claude Sonnet, use these parameters:")
        # Find common patterns in Claude tests
        for r in claude_tests[:3]:
            p = r['params']
            print(f"  - max_tokens={p['max_tokens']}, temp={p['temperature']}, prompt complexity=high")
    elif gpt5_tests:
        print("Model-router prefers GPT-5 for complex tasks.")
        print("GPT-5 was selected when:")
        for r in gpt5_tests[:3]:
            p = r['params']
            print(f"  - max_tokens={p['max_tokens']}, temp={p['temperature']}, prompt_len={p['prompt_length']}")
        print()
        print("To force Claude Sonnet, consider:")
        print("  1. Using AZURE_CLAUDE_ENDPOINT directly (bypasses model-router)")
        print("  2. Check if model-router has Claude in its routing pool")
        print("  3. Contact Azure support about Claude routing priority")
    else:
        print("Model-router selected GPT-4o-mini for most tasks (cost optimization).")
        print()
        print("The router appears to prioritize cost over capability.")
        print("For guaranteed Claude Sonnet, use AZURE_CLAUDE_ENDPOINT directly.")

    client.close()
    print()
    print("Done!")


if __name__ == "__main__":
    run_all_tests()
