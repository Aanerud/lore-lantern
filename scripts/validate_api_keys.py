#!/usr/bin/env python3
"""
API Key Validation Script

Tests all configured API keys by making actual API calls with a simple prompt.
This catches issues like disabled APIs, expired keys, or quota problems
before they cause runtime failures.

Usage:
    python scripts/validate_api_keys.py

    # Or with verbose output:
    python scripts/validate_api_keys.py -v
"""

import asyncio
import os
import sys
import argparse
import time
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# Load environment variables
load_dotenv(project_root / ".env")


@dataclass
class APITestResult:
    """Result of an API key test"""
    provider: str
    model: str
    success: bool
    response_preview: str
    latency_ms: int
    error: Optional[str] = None


TEST_PROMPT = "Why is the sky blue? Answer in exactly one sentence."


async def test_google_genai(api_key: str, verbose: bool = False) -> APITestResult:
    """Test Google Gemini API key"""
    provider = "Google Gemini"
    model = "gemini-2.0-flash"  # Fast, cheap model for testing

    if not api_key:
        return APITestResult(
            provider=provider,
            model=model,
            success=False,
            response_preview="",
            latency_ms=0,
            error="GOOGLE_GENAI_API_KEY not set"
        )

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)

        model_instance = genai.GenerativeModel(model)

        start = time.time()
        response = await model_instance.generate_content_async(TEST_PROMPT)
        latency_ms = int((time.time() - start) * 1000)

        response_text = response.text.strip()
        preview = response_text[:100] + "..." if len(response_text) > 100 else response_text

        if verbose:
            print(f"   Full response: {response_text}")

        return APITestResult(
            provider=provider,
            model=model,
            success=True,
            response_preview=preview,
            latency_ms=latency_ms
        )

    except Exception as e:
        return APITestResult(
            provider=provider,
            model=model,
            success=False,
            response_preview="",
            latency_ms=0,
            error=str(e)
        )


async def test_anthropic(api_key: str, verbose: bool = False) -> APITestResult:
    """Test Anthropic Claude API key"""
    provider = "Anthropic Claude"
    model = "claude-3-haiku-20240307"  # Fastest, cheapest Claude model

    if not api_key:
        return APITestResult(
            provider=provider,
            model=model,
            success=False,
            response_preview="",
            latency_ms=0,
            error="CLAUDE_API_KEY not set"
        )

    try:
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=api_key)

        start = time.time()
        response = await client.messages.create(
            model=model,
            max_tokens=100,
            messages=[{"role": "user", "content": TEST_PROMPT}]
        )
        latency_ms = int((time.time() - start) * 1000)

        response_text = response.content[0].text.strip() if response.content else ""
        preview = response_text[:100] + "..." if len(response_text) > 100 else response_text

        if verbose:
            print(f"   Full response: {response_text}")

        return APITestResult(
            provider=provider,
            model=model,
            success=True,
            response_preview=preview,
            latency_ms=latency_ms
        )

    except Exception as e:
        return APITestResult(
            provider=provider,
            model=model,
            success=False,
            response_preview="",
            latency_ms=0,
            error=str(e)
        )


async def test_azure_foundry(api_key: str, endpoint: str, verbose: bool = False) -> APITestResult:
    """Test Azure AI Foundry Model Router"""
    provider = "Azure Foundry"
    model = "model-router"

    if not api_key or not endpoint:
        return APITestResult(
            provider=provider,
            model=model,
            success=False,
            response_preview="",
            latency_ms=0,
            error="FOUNDRY_API_KEY or FOUNDRY_ENDPOINT not set"
        )

    try:
        from azure.ai.inference import ChatCompletionsClient
        from azure.ai.inference.models import SystemMessage, UserMessage
        from azure.core.credentials import AzureKeyCredential

        full_endpoint = f"{endpoint.rstrip('/')}/openai/deployments/model-router"
        client = ChatCompletionsClient(
            endpoint=full_endpoint,
            credential=AzureKeyCredential(api_key),
        )

        start = time.time()
        response = client.complete(
            messages=[UserMessage(content=TEST_PROMPT)],
            max_tokens=100,
            model="model-router"
        )
        latency_ms = int((time.time() - start) * 1000)

        response_text = response.choices[0].message.content.strip() if response.choices else ""
        model_used = response.model or "unknown"
        preview = f"[{model_used}] {response_text[:80]}..." if len(response_text) > 80 else f"[{model_used}] {response_text}"

        if verbose:
            print(f"   Model selected: {model_used}")
            print(f"   Full response: {response_text}")

        client.close()

        return APITestResult(
            provider=provider,
            model=f"router→{model_used}",
            success=True,
            response_preview=preview,
            latency_ms=latency_ms
        )

    except Exception as e:
        return APITestResult(
            provider=provider,
            model=model,
            success=False,
            response_preview="",
            latency_ms=0,
            error=str(e)
        )


async def test_elevenlabs(api_key: str, verbose: bool = False) -> APITestResult:
    """Test ElevenLabs TTS API key"""
    provider = "ElevenLabs TTS"
    model = "eleven_v3"

    if not api_key:
        return APITestResult(
            provider=provider,
            model=model,
            success=False,
            response_preview="",
            latency_ms=0,
            error="ELEVENLABS_API_KEY not set"
        )

    try:
        import requests

        start = time.time()
        # Just verify the API key by listing voices (cheaper than generating audio)
        response = requests.get(
            "https://api.elevenlabs.io/v1/voices",
            headers={"xi-api-key": api_key},
            timeout=10
        )
        latency_ms = int((time.time() - start) * 1000)

        if response.status_code == 200:
            voices = response.json().get("voices", [])
            preview = f"{len(voices)} voices available"
            return APITestResult(
                provider=provider,
                model=model,
                success=True,
                response_preview=preview,
                latency_ms=latency_ms
            )
        else:
            return APITestResult(
                provider=provider,
                model=model,
                success=False,
                response_preview="",
                latency_ms=0,
                error=f"HTTP {response.status_code}: {response.text[:100]}"
            )

    except Exception as e:
        return APITestResult(
            provider=provider,
            model=model,
            success=False,
            response_preview="",
            latency_ms=0,
            error=str(e)
        )


async def test_speechify(api_key: str, verbose: bool = False) -> APITestResult:
    """Test Speechify TTS API key"""
    provider = "Speechify TTS"
    model = "simba-multilingual"

    if not api_key:
        return APITestResult(
            provider=provider,
            model=model,
            success=False,
            response_preview="",
            latency_ms=0,
            error="SPEECHIFY_API_KEY not set"
        )

    try:
        import requests

        start = time.time()
        # Verify API key by listing voices
        response = requests.get(
            "https://api.sws.speechify.com/v1/voices",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10
        )
        latency_ms = int((time.time() - start) * 1000)

        if response.status_code == 200:
            data = response.json()
            voices = data if isinstance(data, list) else data.get("voices", [])
            preview = f"{len(voices)} voices available"
            return APITestResult(
                provider=provider,
                model=model,
                success=True,
                response_preview=preview,
                latency_ms=latency_ms
            )
        else:
            return APITestResult(
                provider=provider,
                model=model,
                success=False,
                response_preview="",
                latency_ms=0,
                error=f"HTTP {response.status_code}: {response.text[:100]}"
            )

    except Exception as e:
        return APITestResult(
            provider=provider,
            model=model,
            success=False,
            response_preview="",
            latency_ms=0,
            error=str(e)
        )


async def test_openai(api_key: str, verbose: bool = False) -> APITestResult:
    """Test OpenAI API key"""
    provider = "OpenAI"
    model = "gpt-4o-mini"  # Cheapest GPT-4 class model

    if not api_key:
        return APITestResult(
            provider=provider,
            model=model,
            success=False,
            response_preview="",
            latency_ms=0,
            error="OPENAI_API_KEY not set"
        )

    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key)

        start = time.time()
        response = await client.chat.completions.create(
            model=model,
            max_tokens=100,
            messages=[{"role": "user", "content": TEST_PROMPT}]
        )
        latency_ms = int((time.time() - start) * 1000)

        response_text = response.choices[0].message.content.strip() if response.choices else ""
        preview = response_text[:100] + "..." if len(response_text) > 100 else response_text

        if verbose:
            print(f"   Full response: {response_text}")

        return APITestResult(
            provider=provider,
            model=model,
            success=True,
            response_preview=preview,
            latency_ms=latency_ms
        )

    except Exception as e:
        return APITestResult(
            provider=provider,
            model=model,
            success=False,
            response_preview="",
            latency_ms=0,
            error=str(e)
        )


def print_result(result: APITestResult):
    """Pretty-print a test result"""
    status = "✅" if result.success else "❌"
    print(f"\n{status} {result.provider} ({result.model})")

    if result.success:
        print(f"   Latency: {result.latency_ms}ms")
        print(f"   Response: \"{result.response_preview}\"")
    else:
        print(f"   Error: {result.error}")


async def main():
    parser = argparse.ArgumentParser(description="Validate API keys by testing with a simple prompt")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show full responses")
    parser.add_argument("--provider", choices=["google", "anthropic", "openai", "foundry", "elevenlabs", "speechify"],
                        help="Test only a specific provider")
    parser.add_argument("--llm-only", action="store_true", help="Only test LLM providers (skip TTS)")
    parser.add_argument("--tts-only", action="store_true", help="Only test TTS providers (skip LLM)")
    args = parser.parse_args()

    print("=" * 60)
    print("API Key Validation Test")
    print("=" * 60)
    print(f"Test prompt: \"{TEST_PROMPT}\"")
    print("-" * 60)

    # Get API keys from environment
    google_key = os.getenv("GOOGLE_GENAI_API_KEY")
    anthropic_key = os.getenv("CLAUDE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    foundry_key = os.getenv("FOUNDRY_API_KEY")
    foundry_endpoint = os.getenv("FOUNDRY_ENDPOINT")
    elevenlabs_key = os.getenv("ELEVENLABS_API_KEY")
    speechify_key = os.getenv("SPEECHIFY_API_KEY")

    results: List[APITestResult] = []

    # Determine which providers to test
    test_llm = not args.tts_only
    test_tts = not args.llm_only

    # Run LLM tests
    if test_llm:
        if args.provider is None or args.provider == "foundry":
            print("\nTesting Azure Foundry Model Router...")
            result = await test_azure_foundry(foundry_key, foundry_endpoint, args.verbose)
            results.append(result)
            print_result(result)

        if args.provider is None or args.provider == "google":
            print("\nTesting Google Gemini...")
            result = await test_google_genai(google_key, args.verbose)
            results.append(result)
            print_result(result)

        if args.provider is None or args.provider == "anthropic":
            print("\nTesting Anthropic Claude...")
            result = await test_anthropic(anthropic_key, args.verbose)
            results.append(result)
            print_result(result)

        if args.provider is None or args.provider == "openai":
            print("\nTesting OpenAI...")
            result = await test_openai(openai_key, args.verbose)
            results.append(result)
            print_result(result)

    # Run TTS tests
    if test_tts:
        if args.provider is None or args.provider == "elevenlabs":
            print("\nTesting ElevenLabs TTS...")
            result = await test_elevenlabs(elevenlabs_key, args.verbose)
            results.append(result)
            print_result(result)

        if args.provider is None or args.provider == "speechify":
            print("\nTesting Speechify TTS...")
            result = await test_speechify(speechify_key, args.verbose)
            results.append(result)
            print_result(result)

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    passed = sum(1 for r in results if r.success)
    failed = sum(1 for r in results if not r.success)

    print(f"Passed: {passed}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")

    if failed > 0:
        print("\nFailed providers:")
        for r in results:
            if not r.success:
                print(f"  - {r.provider}: {r.error}")

    # Recommendations based on results
    print("\n" + "-" * 60)
    print("Recommendations for Lore Lantern:")
    print("-" * 60)

    foundry_ok = any(r.provider == "Azure Foundry" and r.success for r in results)
    google_ok = any(r.provider == "Google Gemini" and r.success for r in results)
    anthropic_ok = any(r.provider == "Anthropic Claude" and r.success for r in results)
    openai_ok = any(r.provider == "OpenAI" and r.success for r in results)
    elevenlabs_ok = any(r.provider == "ElevenLabs TTS" and r.success for r in results)
    speechify_ok = any(r.provider == "Speechify TTS" and r.success for r in results)

    print("\n  LLM Providers:")
    if foundry_ok:
        print("  ✅ Azure Foundry: Model Router available (primary for agents)")
    else:
        print("  ❌ Azure Foundry: Not available - check FOUNDRY_ENDPOINT/API_KEY")

    if anthropic_ok:
        print("  ✅ Anthropic Claude: Available (fallback for agents)")
    else:
        print("  ⚠️  Anthropic Claude: Not available")

    if google_ok:
        print("  ✅ Google Gemini: Available")
    else:
        print("  ⚠️  Google Gemini: Not available")

    if openai_ok:
        print("  ✅ OpenAI: Available (for dialogue TTS)")
    else:
        print("  ❌ OpenAI: Not available - dialogue TTS will fail!")

    print("\n  TTS Providers:")
    if elevenlabs_ok:
        print("  ✅ ElevenLabs: Available (primary narration)")
    else:
        print("  ⚠️  ElevenLabs: Not available - will use Speechify fallback")

    if speechify_ok:
        print("  ✅ Speechify: Available (fallback narration)")
    else:
        print("  ⚠️  Speechify: Not available")

    if not elevenlabs_ok and not speechify_ok:
        print("  ❌ No TTS for narration! Stories will not have audio.")

    print()

    # Exit with error code if any tests failed
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
