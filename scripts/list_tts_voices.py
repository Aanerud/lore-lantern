#!/usr/bin/env python3
"""
List Available Text-to-Speech Voices and Languages

This script queries available TTS providers and lists their voices,
languages, and SSML support.

Lore Lantern TTS Architecture:
- ElevenLabs: Primary narration (eleven_v3 model, audio tags like [whispers])
- Speechify: Fallback narration (simba-multilingual model)
- OpenAI: Dialogue TTS (gpt-4o-mini-tts or direct audio)

Other providers (Google, Azure, AWS) are informational only.

Usage:
    python scripts/list_tts_voices.py
    python scripts/list_tts_voices.py --provider elevenlabs
    python scripts/list_tts_voices.py --provider speechify
    python scripts/list_tts_voices.py --language no  # Filter by language
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


@dataclass
class VoiceInfo:
    """Information about a TTS voice"""
    name: str
    language_code: str
    language_name: str
    gender: str
    provider: str
    ssml_support: bool
    quality_tier: str  # standard, neural, wavenet, hd
    description: str = ""


def check_openai_voices() -> List[VoiceInfo]:
    """List available OpenAI TTS voices"""
    voices = []

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("  ❌ OPENAI_API_KEY not set")
        return voices

    # OpenAI has 6 fixed voices - they don't have a list API
    # All voices support all languages (auto-detected)
    openai_voices = [
        ("alloy", "neutral", "Balanced, versatile voice"),
        ("echo", "male", "Warm, conversational male voice"),
        ("fable", "male", "Expressive, British-accented voice"),
        ("onyx", "male", "Deep, authoritative voice"),
        ("nova", "female", "Warm, friendly female voice - recommended for children"),
        ("shimmer", "female", "Soft, gentle female voice"),
    ]

    # OpenAI TTS supports ~57 languages automatically
    supported_languages = [
        ("en", "English"),
        ("no", "Norwegian"),
        ("es", "Spanish"),
        ("fr", "French"),
        ("de", "German"),
        ("it", "Italian"),
        ("pt", "Portuguese"),
        ("pl", "Polish"),
        ("ru", "Russian"),
        ("nl", "Dutch"),
        ("sv", "Swedish"),
        ("da", "Danish"),
        ("fi", "Finnish"),
        ("ja", "Japanese"),
        ("ko", "Korean"),
        ("zh", "Chinese"),
        ("ar", "Arabic"),
        ("hi", "Hindi"),
        ("tr", "Turkish"),
        ("uk", "Ukrainian"),
        ("cs", "Czech"),
        ("el", "Greek"),
        ("he", "Hebrew"),
        ("id", "Indonesian"),
        ("ms", "Malay"),
        ("ro", "Romanian"),
        ("th", "Thai"),
        ("vi", "Vietnamese"),
    ]

    for voice_name, gender, description in openai_voices:
        for lang_code, lang_name in supported_languages:
            voices.append(VoiceInfo(
                name=voice_name,
                language_code=lang_code,
                language_name=lang_name,
                gender=gender,
                provider="openai",
                ssml_support=False,  # OpenAI does NOT support SSML
                quality_tier="hd",  # tts-1-hd available
                description=description
            ))

    return voices


def check_google_voices() -> List[VoiceInfo]:
    """List available Google Cloud TTS voices"""
    voices = []

    # Can use either service account credentials or API key
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    api_key = os.getenv("GOOGLE_GENAI_API_KEY")

    if not creds_path and not api_key:
        print("  ❌ No Google credentials (GOOGLE_APPLICATION_CREDENTIALS or GOOGLE_GENAI_API_KEY)")
        return voices

    if creds_path and not os.path.exists(creds_path):
        print(f"  ⚠️  GOOGLE_APPLICATION_CREDENTIALS file not found: {creds_path}")
        if not api_key:
            return voices

    try:
        from google.cloud import texttospeech

        client = texttospeech.TextToSpeechClient()
        response = client.list_voices()

        for voice in response.voices:
            # Determine quality tier from voice name
            name = voice.name
            if "Neural2" in name:
                tier = "neural2"
            elif "Wavenet" in name:
                tier = "wavenet"
            elif "Studio" in name:
                tier = "studio"
            elif "Polyglot" in name:
                tier = "polyglot"
            else:
                tier = "standard"

            # Determine gender
            gender_map = {
                texttospeech.SsmlVoiceGender.MALE: "male",
                texttospeech.SsmlVoiceGender.FEMALE: "female",
                texttospeech.SsmlVoiceGender.NEUTRAL: "neutral",
            }
            gender = gender_map.get(voice.ssml_gender, "unknown")

            for lang_code in voice.language_codes:
                # Get language name from code
                lang_name = get_language_name(lang_code)

                voices.append(VoiceInfo(
                    name=voice.name,
                    language_code=lang_code,
                    language_name=lang_name,
                    gender=gender,
                    provider="google",
                    ssml_support=True,  # Google Cloud TTS fully supports SSML
                    quality_tier=tier,
                    description=f"Sample rate: {voice.natural_sample_rate_hertz}Hz"
                ))

        return voices

    except ImportError:
        print("  ❌ google-cloud-texttospeech not installed")
        print("     pip install google-cloud-texttospeech")
        return voices
    except Exception as e:
        print(f"  ❌ Google Cloud TTS error: {e}")
        return voices


def check_elevenlabs_voices() -> List[VoiceInfo]:
    """List available ElevenLabs voices (if configured)"""
    voices = []

    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        # Silently skip - not configured
        return voices

    try:
        import requests

        headers = {"xi-api-key": api_key}
        response = requests.get(
            "https://api.elevenlabs.io/v1/voices",
            headers=headers
        )

        if response.status_code == 200:
            data = response.json()
            for voice in data.get("voices", []):
                voices.append(VoiceInfo(
                    name=voice.get("name", "unknown"),
                    language_code="multi",  # ElevenLabs voices are multilingual
                    language_name="Multilingual",
                    gender=voice.get("labels", {}).get("gender", "unknown"),
                    provider="elevenlabs",
                    ssml_support=True,  # ElevenLabs supports SSML
                    quality_tier="neural",
                    description=voice.get("description", "")[:50]
                ))

        return voices

    except Exception as e:
        print(f"  ❌ ElevenLabs error: {e}")
        return voices


def check_speechify_voices() -> List[VoiceInfo]:
    """List available Speechify voices (if configured)"""
    voices = []

    api_key = os.getenv("SPEECHIFY_API_KEY")
    if not api_key:
        return voices

    try:
        import requests

        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get(
            "https://api.sws.speechify.com/v1/voices",
            headers=headers
        )

        if response.status_code == 200:
            data = response.json()
            for voice in data if isinstance(data, list) else data.get("voices", []):
                # Extract voice details
                voice_id = voice.get("id", voice.get("voice_id", "unknown"))
                name = voice.get("name", voice.get("display_name", voice_id))
                lang = voice.get("language_code", voice.get("language", "multi"))
                gender = voice.get("gender", "unknown")

                voices.append(VoiceInfo(
                    name=name,
                    language_code=lang,
                    language_name=get_language_name(lang) if "-" in lang else lang.title(),
                    gender=gender.lower() if isinstance(gender, str) else "unknown",
                    provider="speechify",
                    ssml_support=False,  # Speechify uses plain text
                    quality_tier="simba",
                    description=f"Model: simba-multilingual"
                ))

        return voices

    except Exception as e:
        print(f"  ❌ Speechify error: {e}")
        return voices


def check_azure_voices() -> List[VoiceInfo]:
    """List available Azure Speech voices via Azure AI Foundry"""
    voices = []

    # Azure Speech can be accessed via Foundry endpoint or dedicated Speech key
    foundry_key = os.getenv("FOUNDRY_API_KEY")
    foundry_endpoint = os.getenv("FOUNDRY_ENDPOINT")
    speech_key = os.getenv("AZURE_SPEECH_KEY")
    region = os.getenv("AZURE_SPEECH_REGION", "swedencentral")  # Default to Sweden Central (our Foundry region)

    if not foundry_key and not speech_key:
        return voices

    try:
        import requests

        # Try Foundry endpoint first, then dedicated Speech endpoint
        if foundry_endpoint and foundry_key:
            # Extract region from Foundry endpoint (e.g., swedencentral from aaner-mjaezuiz-swedencentral.cognitiveservices.azure.com)
            try:
                endpoint_parts = foundry_endpoint.replace("https://", "").split(".")
                if endpoint_parts:
                    region = endpoint_parts[0].split("-")[-1]  # Get last part before .cognitiveservices
            except:
                pass
            url = f"https://{region}.tts.speech.microsoft.com/cognitiveservices/voices/list"
            headers = {"Ocp-Apim-Subscription-Key": foundry_key}
        else:
            url = f"https://{region}.tts.speech.microsoft.com/cognitiveservices/voices/list"
            headers = {"Ocp-Apim-Subscription-Key": speech_key}

        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            for voice in response.json():
                # Determine quality tier
                name = voice.get("ShortName", "")
                if "Neural" in name:
                    tier = "neural"
                else:
                    tier = "standard"

                voices.append(VoiceInfo(
                    name=voice.get("ShortName", "unknown"),
                    language_code=voice.get("Locale", "unknown"),
                    language_name=voice.get("LocaleName", "unknown"),
                    gender=voice.get("Gender", "unknown").lower(),
                    provider="azure",
                    ssml_support=True,  # Azure has EXCELLENT SSML support
                    quality_tier=tier,
                    description=voice.get("VoiceType", "")
                ))

        return voices

    except Exception as e:
        print(f"  ❌ Azure Speech error: {e}")
        return voices


def get_language_name(code: str) -> str:
    """Get language name from code"""
    languages = {
        "en-US": "English (US)",
        "en-GB": "English (UK)",
        "en-AU": "English (Australia)",
        "nb-NO": "Norwegian (Bokmål)",
        "nn-NO": "Norwegian (Nynorsk)",
        "no-NO": "Norwegian",
        "es-ES": "Spanish (Spain)",
        "es-MX": "Spanish (Mexico)",
        "es-US": "Spanish (US)",
        "fr-FR": "French (France)",
        "fr-CA": "French (Canada)",
        "de-DE": "German",
        "it-IT": "Italian",
        "pt-BR": "Portuguese (Brazil)",
        "pt-PT": "Portuguese (Portugal)",
        "nl-NL": "Dutch",
        "sv-SE": "Swedish",
        "da-DK": "Danish",
        "fi-FI": "Finnish",
        "pl-PL": "Polish",
        "ru-RU": "Russian",
        "ja-JP": "Japanese",
        "ko-KR": "Korean",
        "zh-CN": "Chinese (Simplified)",
        "zh-TW": "Chinese (Traditional)",
        "ar-XA": "Arabic",
        "hi-IN": "Hindi",
        "tr-TR": "Turkish",
        "uk-UA": "Ukrainian",
        "cs-CZ": "Czech",
        "el-GR": "Greek",
        "he-IL": "Hebrew",
        "id-ID": "Indonesian",
        "ms-MY": "Malay",
        "ro-RO": "Romanian",
        "th-TH": "Thai",
        "vi-VN": "Vietnamese",
        "fil-PH": "Filipino",
        "hu-HU": "Hungarian",
        "sk-SK": "Slovak",
        "bg-BG": "Bulgarian",
        "ca-ES": "Catalan",
        "hr-HR": "Croatian",
        "lt-LT": "Lithuanian",
        "lv-LV": "Latvian",
        "sl-SI": "Slovenian",
        "sr-RS": "Serbian",
        "af-ZA": "Afrikaans",
        "bn-IN": "Bengali",
        "gu-IN": "Gujarati",
        "kn-IN": "Kannada",
        "ml-IN": "Malayalam",
        "mr-IN": "Marathi",
        "ta-IN": "Tamil",
        "te-IN": "Telugu",
        "cmn-CN": "Mandarin Chinese",
        "cmn-TW": "Mandarin (Taiwan)",
        "yue-HK": "Cantonese",
    }
    return languages.get(code, code)


def print_provider_summary(voices: List[VoiceInfo], provider: str):
    """Print summary for a provider"""
    provider_voices = [v for v in voices if v.provider == provider]

    if not provider_voices:
        return

    # Get unique voices and languages
    unique_voices = set(v.name for v in provider_voices)
    unique_languages = set(v.language_code for v in provider_voices)
    ssml_support = provider_voices[0].ssml_support

    # Provider display names
    display_names = {
        "openai": "OPENAI",
        "google": "GOOGLE CLOUD",
        "azure": "MICROSOFT AZURE",
        "aws-polly": "AWS POLLY",
        "elevenlabs": "ELEVENLABS",
        "speechify": "SPEECHIFY"
    }

    print(f"\n{'='*60}")
    print(f"  {display_names.get(provider, provider.upper())} TEXT-TO-SPEECH")
    print(f"{'='*60}")
    print(f"  Voices: {len(unique_voices)}")
    print(f"  Languages: {len(unique_languages)}")
    print(f"  SSML Support: {'✅ Full' if ssml_support else '❌ None (converted to plain text)'}")

    # Group by quality tier
    tiers = {}
    for v in provider_voices:
        tier = v.quality_tier
        if tier not in tiers:
            tiers[tier] = set()
        tiers[tier].add(v.name)

    print(f"  Quality Tiers:")
    for tier, names in sorted(tiers.items()):
        print(f"    - {tier}: {len(names)} voices")


def print_voices_table(voices: List[VoiceInfo], filter_lang: str = None, filter_provider: str = None):
    """Print voices in a table format"""
    filtered = voices

    if filter_lang:
        # Match language code prefix (e.g., "no" matches "nb-NO", "nn-NO")
        filtered = [v for v in filtered if v.language_code.startswith(filter_lang) or
                   filter_lang in v.language_code.lower()]

    if filter_provider:
        filtered = [v for v in filtered if v.provider == filter_provider]

    if not filtered:
        print("\n  No voices found matching criteria.")
        return

    # Group by provider and language for cleaner output
    by_provider = {}
    for v in filtered:
        if v.provider not in by_provider:
            by_provider[v.provider] = {}
        lang = v.language_code
        if lang not in by_provider[v.provider]:
            by_provider[v.provider][lang] = []
        by_provider[v.provider][lang].append(v)

    for provider, langs in by_provider.items():
        print(f"\n  {provider.upper()} Voices:")
        print(f"  {'-'*50}")

        for lang, lang_voices in sorted(langs.items()):
            # Deduplicate by voice name
            unique_voices = {}
            for v in lang_voices:
                if v.name not in unique_voices:
                    unique_voices[v.name] = v

            print(f"\n    {get_language_name(lang)} ({lang}):")
            for v in sorted(unique_voices.values(), key=lambda x: (x.quality_tier, x.name)):
                ssml_icon = "✅" if v.ssml_support else "❌"
                print(f"      {v.name:25} {v.gender:8} {v.quality_tier:10} SSML:{ssml_icon}")
                if v.description:
                    print(f"        └─ {v.description}")


def print_ssml_recommendation(voices: List[VoiceInfo]):
    """Print recommendation for SSML-based narration"""
    print(f"\n{'='*60}")
    print("  SSML NARRATION RECOMMENDATION")
    print(f"{'='*60}")

    google_voices = [v for v in voices if v.provider == "google"]
    openai_voices = [v for v in voices if v.provider == "openai"]
    elevenlabs_voices = [v for v in voices if v.provider == "elevenlabs"]

    print("\n  For full SSML support with prosody, breaks, and emphasis:")

    if google_voices:
        print("\n  🏆 RECOMMENDED: Google Cloud TTS")
        print("     ✅ Full SSML support (prosody, breaks, emphasis, say-as)")
        print("     ✅ Neural2 and Wavenet voices (high quality)")
        print("     ✅ 40+ languages with child-friendly voices")
        print("     💰 Cost: ~$4/million chars (Neural2), ~$16/million (Wavenet)")

        # Recommend specific voices for our target languages
        target_langs = ["en-US", "nb-NO", "es-ES"]
        print("\n     Recommended voices for Lore Lantern:")
        for lang in target_langs:
            lang_voices = [v for v in google_voices if v.language_code == lang
                         and v.quality_tier in ("neural2", "wavenet")
                         and v.gender == "female"]
            if lang_voices:
                best = sorted(lang_voices, key=lambda x: x.quality_tier)[0]
                print(f"       {get_language_name(lang):20} → {best.name}")

    if elevenlabs_voices:
        print("\n  🥈 ALTERNATIVE: ElevenLabs")
        print("     ✅ SSML support")
        print("     ✅ Very natural voices")
        print("     ⚠️  Higher cost")

    if openai_voices:
        print("\n  ⚠️  OpenAI TTS (Current)")
        print("     ❌ NO SSML support")
        print("     ✅ Easy to use, good quality")
        print("     ✅ Auto language detection")
        print("     ℹ️  SSML tags are stripped before synthesis")

    print("\n  📝 Note: Your SSMLProcessor creates rich SSML markup, but")
    print("     it only produces enhanced narration with Google Cloud TTS.")
    print("     With OpenAI, SSML is converted to plain text hints.")


def print_providers_comparison():
    """Print a comparison table of all TTS providers"""
    print(f"\n{'='*60}")
    print("  TTS PROVIDERS COMPARISON")
    print(f"{'='*60}")
    print("""
  LORE LANTERN TTS STACK:
  ┌─────────────────┬────────────────────────────────────────────┐
  │ Use Case        │ Provider                                   │
  ├─────────────────┼────────────────────────────────────────────┤
  │ Narration (1st) │ ElevenLabs (eleven_v3, audio tags)         │
  │ Narration (2nd) │ Speechify (simba-multilingual)             │
  │ Dialogue        │ OpenAI (gpt-4o-mini-tts / direct audio)    │
  └─────────────────┴────────────────────────────────────────────┘

  ALL AVAILABLE PROVIDERS:
  Provider        | SSML | Norwegian | Quality     | Cost/1M chars
  ----------------|------|-----------|-------------|---------------
  OpenAI          | ❌   | ✅ Auto   | HD          | ~$15
  ElevenLabs      | ⚠️   | ✅ Multi  | Ultra       | ~$30-99
  Speechify       | ❌   | ✅ Beta   | Simba       | ~$10-20
  Google Cloud    | ✅   | ✅ Wavenet| Neural2     | ~$4-16
  Microsoft Azure | ✅   | ✅ Neural | Best Neural | ~$4-16

  SSML/AUDIO TAGS BY PROVIDER:
  ┌─────────────────┬────────┬──────────┬──────────┬────────┬────────┐
  │ Feature         │ OpenAI │ ElevenLab│ Speechify│ Google │ Azure  │
  ├─────────────────┼────────┼──────────┼──────────┼────────┼────────┤
  │ [whispers]      │ ❌     │ ✅       │ ❌       │ ❌     │ ❌     │
  │ [laughing]      │ ❌     │ ✅       │ ❌       │ ❌     │ ❌     │
  │ <break>         │ ❌     │ ✅       │ ❌       │ ✅     │ ✅     │
  │ <prosody>       │ ❌     │ ⚠️       │ ❌       │ ✅     │ ✅     │
  │ <emphasis>      │ ❌     │ ⚠️       │ ❌       │ ✅     │ ✅     │
  │ <say-as>        │ ❌     │ ❌       │ ❌       │ ✅     │ ✅     │
  └─────────────────┴────────┴──────────┴──────────┴────────┴────────┘

  API KEY ENVIRONMENT VARIABLES:
  - OPENAI_API_KEY               → OpenAI TTS (dialogue)
  - ELEVENLABS_API_KEY           → ElevenLabs (narration, primary)
  - SPEECHIFY_API_KEY            → Speechify (narration, fallback)
  - GOOGLE_GENAI_API_KEY         → Google Cloud TTS
  - FOUNDRY_API_KEY              → Azure Speech (via Foundry)

  NOTE: VoiceDirectorAgent adds ElevenLabs audio tags like [whispers], [laughing]
        for expressive narration. Speechify receives plain text.
""")


def main():
    parser = argparse.ArgumentParser(description="List available TTS voices")
    parser.add_argument("--provider", "-p", choices=["openai", "elevenlabs", "speechify", "google", "azure"],
                       help="Filter by provider")
    parser.add_argument("--language", "-l", help="Filter by language code (e.g., 'no', 'en-US')")
    parser.add_argument("--ssml-only", action="store_true", help="Only show providers with SSML support")
    parser.add_argument("--quiet", "-q", action="store_true", help="Less verbose output")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  TEXT-TO-SPEECH VOICES & LANGUAGES")
    print("="*60)

    all_voices = []

    # Check each provider
    print("\n  Checking providers...")

    print("\n  OpenAI:")
    openai_voices = check_openai_voices()
    if openai_voices:
        print(f"    ✅ {len(set(v.name for v in openai_voices))} voices available")
        print(f"    ⚠️  NO SSML support")
        all_voices.extend(openai_voices)

    print("\n  Google Cloud:")
    google_voices = check_google_voices()
    if google_voices:
        print(f"    ✅ {len(set(v.name for v in google_voices))} voices available")
        print(f"    ✅ Full SSML support")
        all_voices.extend(google_voices)

    print("\n  Microsoft Azure (via Foundry):")
    azure_voices = check_azure_voices()
    if azure_voices:
        print(f"    ✅ {len(set(v.name for v in azure_voices))} voices available")
        print(f"    ✅ Full SSML support (best in class)")
        all_voices.extend(azure_voices)
    else:
        print("    ⏭️  Not accessible (FOUNDRY_API_KEY may not have Speech permissions)")

    print("\n  ElevenLabs:")
    elevenlabs_voices = check_elevenlabs_voices()
    if elevenlabs_voices:
        print(f"    ✅ {len(elevenlabs_voices)} voices available")
        print(f"    ✅ Audio tags support ([whispers], [laughing], etc.)")
        all_voices.extend(elevenlabs_voices)
    else:
        print("    ⏭️  Not configured (set ELEVENLABS_API_KEY)")

    print("\n  Speechify:")
    speechify_voices = check_speechify_voices()
    if speechify_voices:
        print(f"    ✅ {len(speechify_voices)} voices available")
        print(f"    ✅ Norwegian beta support (nb-NO)")
        all_voices.extend(speechify_voices)
    else:
        print("    ⏭️  Not configured (set SPEECHIFY_API_KEY)")

    if not all_voices:
        print("\n  ❌ No TTS providers available!")
        print("     Set OPENAI_API_KEY or GOOGLE_APPLICATION_CREDENTIALS")
        return 1

    # Filter for SSML if requested
    if args.ssml_only:
        all_voices = [v for v in all_voices if v.ssml_support]

    # Print summaries
    for provider in ["openai", "elevenlabs", "speechify", "google", "azure"]:
        print_provider_summary(all_voices, provider)

    # Print detailed voice list if filtering
    if args.provider or args.language:
        print_voices_table(all_voices, args.language, args.provider)

    # Print SSML recommendation
    if not args.quiet:
        print_ssml_recommendation(all_voices)

    # Print all providers comparison
    print_providers_comparison()

    # Print quick commands
    print(f"\n{'='*60}")
    print("  QUICK COMMANDS")
    print(f"{'='*60}")
    print("  List Norwegian voices:  python scripts/list_tts_voices.py -l no")
    print("  List Google voices:     python scripts/list_tts_voices.py -p google")
    print("  SSML-only providers:    python scripts/list_tts_voices.py --ssml-only")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
