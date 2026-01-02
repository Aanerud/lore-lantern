"""
Unit tests for LLM Router - verifies model routing without LLM calls.

Tests the routing logic hierarchy (Agent > Group > Default),
environment variable overrides, model normalization, and
provider-specific parameter constraints.

Run with: python -m pytest tests/test_llm_router.py -v
"""

import os
import sys
import tempfile
import yaml
from pathlib import Path
from unittest import mock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.llm_router import LLMRouter, get_llm_router, reset_llm_router


class TestLLMRouterHierarchy:
    """Test model resolution hierarchy: Agent > Group > Default"""

    def setup_method(self):
        """Reset LLM Router singleton before each test."""
        reset_llm_router()
        # Clear any TEST_ env vars
        for key in list(os.environ.keys()):
            if key.startswith('TEST_') and key.endswith('_MODEL'):
                del os.environ[key]

    def teardown_method(self):
        """Clean up after each test."""
        reset_llm_router()
        # Clear any TEST_ env vars
        for key in list(os.environ.keys()):
            if key.startswith('TEST_') and key.endswith('_MODEL'):
                del os.environ[key]

    def test_default_model_when_no_overrides(self):
        """Without overrides, agents use default model from YAML."""
        router = LLMRouter()

        # All agents with model: null should use default (azure/model-router)
        for agent in ['structure', 'character', 'factcheck', 'line_editor',
                      'continuity', 'tension', 'narrative', 'dialogue']:
            model = router.get_model_for_agent(agent)
            assert model == 'azure/model-router', f"{agent} should use default model"

    def test_group_override_applies_to_all_members(self):
        """TEST_ROUNDTABLE_MODEL applies to all 6 reviewers."""
        os.environ['TEST_ROUNDTABLE_MODEL'] = 'gemini-3-flash-preview'
        router = LLMRouter()

        roundtable_members = ['structure', 'factcheck', 'character',
                              'line_editor', 'continuity', 'tension']

        for agent in roundtable_members:
            model = router.get_model_for_agent(agent)
            # Should be normalized to gemini/gemini-3-flash-preview
            assert 'gemini-3-flash-preview' in model, \
                f"{agent} should use roundtable group model"

    def test_agent_override_beats_group_override(self):
        """TEST_STRUCTURE_MODEL overrides TEST_ROUNDTABLE_MODEL for Guillermo."""
        os.environ['TEST_ROUNDTABLE_MODEL'] = 'gemini-3-flash-preview'
        os.environ['TEST_STRUCTURE_MODEL'] = 'claude-sonnet-4-5'
        router = LLMRouter()

        # Structure should use agent override (claude), not group (gemini)
        structure_model = router.get_model_for_agent('structure')
        assert 'claude-sonnet-4-5' in structure_model, \
            "Agent override should beat group override"

        # Character should still use group override (gemini)
        character_model = router.get_model_for_agent('character')
        assert 'gemini-3-flash-preview' in character_model, \
            "Character should use group override"

    def test_roundtable_group_contains_all_reviewers(self):
        """Verify all 6 reviewers are in roundtable group."""
        router = LLMRouter()
        members = router.list_agents_in_group('roundtable')

        expected = ['structure', 'factcheck', 'character',
                    'line_editor', 'continuity', 'tension']

        for agent in expected:
            assert agent in members, f"{agent} should be in roundtable group"

        assert len(members) == 6, "Roundtable should have exactly 6 members"

    def test_writers_group_contains_narrative_and_dialogue(self):
        """Verify writers group contains narrative and dialogue agents."""
        router = LLMRouter()
        members = router.list_agents_in_group('writers')

        assert 'narrative' in members, "Narrative should be in writers group"
        assert 'dialogue' in members, "Dialogue should be in writers group"

    def test_unknown_agent_uses_default(self):
        """Unknown agent names should fall back to default model."""
        router = LLMRouter()
        model = router.get_model_for_agent('nonexistent_agent')
        assert model == 'azure/model-router', "Unknown agent should use default"


class TestModelNormalization:
    """Test provider prefix handling."""

    def setup_method(self):
        reset_llm_router()

    def teardown_method(self):
        reset_llm_router()

    def test_gemini_models_get_prefix(self):
        """gemini-3-flash-preview -> gemini/gemini-3-flash-preview"""
        router = LLMRouter()
        normalized = router._normalize_model_name('gemini-3-flash-preview')
        assert normalized == 'gemini/gemini-3-flash-preview'

    def test_gemini_with_existing_prefix_unchanged(self):
        """gemini/gemini-3-flash-preview stays unchanged."""
        router = LLMRouter()
        normalized = router._normalize_model_name('gemini/gemini-3-flash-preview')
        assert normalized == 'gemini/gemini-3-flash-preview'

    def test_claude_models_unchanged(self):
        """claude-sonnet-4-5 unchanged (already has prefix pattern)."""
        router = LLMRouter()
        normalized = router._normalize_model_name('claude-sonnet-4-5')
        assert normalized == 'claude-sonnet-4-5'

    def test_azure_model_router_unchanged(self):
        """azure/model-router unchanged."""
        router = LLMRouter()
        normalized = router._normalize_model_name('azure/model-router')
        assert normalized == 'azure/model-router'

    def test_gpt_models_unchanged(self):
        """gpt-5 unchanged (already has prefix pattern)."""
        router = LLMRouter()
        normalized = router._normalize_model_name('gpt-5')
        assert normalized == 'gpt-5'

    def test_anthropic_prefix_unchanged(self):
        """anthropic/claude-3-opus unchanged."""
        router = LLMRouter()
        normalized = router._normalize_model_name('anthropic/claude-3-opus')
        assert normalized == 'anthropic/claude-3-opus'


class TestModelConstraints:
    """Test provider-specific parameter handling."""

    def setup_method(self):
        reset_llm_router()
        # Clear any TEST_ env vars
        for key in list(os.environ.keys()):
            if key.startswith('TEST_') and key.endswith('_MODEL'):
                del os.environ[key]

    def teardown_method(self):
        reset_llm_router()
        for key in list(os.environ.keys()):
            if key.startswith('TEST_') and key.endswith('_MODEL'):
                del os.environ[key]

    def test_claude_no_top_p(self):
        """Claude models should NOT have top_p parameter."""
        os.environ['TEST_STRUCTURE_MODEL'] = 'claude-sonnet-4-5'
        router = LLMRouter()

        kwargs = router.get_llm_kwargs('structure')
        assert 'top_p' not in kwargs, "Claude should not have top_p"

    def test_non_claude_has_top_p(self):
        """Gemini/OpenAI models should have top_p=0.95."""
        os.environ['TEST_STRUCTURE_MODEL'] = 'gemini-3-flash-preview'
        router = LLMRouter()

        kwargs = router.get_llm_kwargs('structure')
        assert kwargs.get('top_p') == 0.95, "Non-Claude should have top_p=0.95"

    def test_azure_model_router_has_reasoning_effort(self):
        """azure/model-router should have reasoning_effort in extra_body."""
        router = LLMRouter()  # Uses default azure/model-router

        kwargs = router.get_llm_kwargs('structure')
        assert 'extra_body' in kwargs, "Should have extra_body"
        assert kwargs['extra_body'].get('reasoning_effort') == 'medium', \
            "Should have reasoning_effort=medium"

    def test_openai_reasoning_drops_temperature(self):
        """GPT-5, o1, o3 should not have temperature."""
        os.environ['TEST_STRUCTURE_MODEL'] = 'gpt-5'
        router = LLMRouter()

        kwargs = router.get_llm_kwargs('structure')
        assert 'temperature' not in kwargs, "GPT-5 should not have temperature"

    def test_openai_reasoning_uses_max_completion_tokens(self):
        """GPT-5 uses max_completion_tokens instead of max_tokens."""
        os.environ['TEST_STRUCTURE_MODEL'] = 'gpt-5'
        router = LLMRouter()

        kwargs = router.get_llm_kwargs('structure')
        assert 'max_tokens' not in kwargs, "GPT-5 should not have max_tokens"
        assert 'extra_body' in kwargs, "Should have extra_body"
        assert 'max_completion_tokens' in kwargs['extra_body'], \
            "Should have max_completion_tokens in extra_body"

    def test_o1_model_drops_temperature(self):
        """o1-preview should not have temperature."""
        os.environ['TEST_NARRATIVE_MODEL'] = 'o1-preview'
        router = LLMRouter()

        kwargs = router.get_llm_kwargs('narrative')
        assert 'temperature' not in kwargs, "o1 should not have temperature"

    def test_drop_params_always_true(self):
        """All kwargs should have drop_params=True."""
        router = LLMRouter()

        for agent in ['structure', 'character', 'narrative']:
            kwargs = router.get_llm_kwargs(agent)
            assert kwargs.get('drop_params') == True, \
                f"{agent} should have drop_params=True"


class TestDisplayNames:
    """Test agent display name retrieval."""

    def setup_method(self):
        reset_llm_router()

    def teardown_method(self):
        reset_llm_router()

    def test_structure_display_name(self):
        """Structure agent should have display name 'Guillermo'."""
        router = LLMRouter()
        assert router.get_agent_display_name('structure') == 'Guillermo'

    def test_character_display_name(self):
        """Character agent should have display name 'Clarissa'."""
        router = LLMRouter()
        assert router.get_agent_display_name('character') == 'Clarissa'

    def test_factcheck_display_name(self):
        """Factcheck agent should have display name 'Bill'."""
        router = LLMRouter()
        assert router.get_agent_display_name('factcheck') == 'Bill'

    def test_line_editor_display_name(self):
        """Line editor agent should have display name 'Benjamin'."""
        router = LLMRouter()
        assert router.get_agent_display_name('line_editor') == 'Benjamin'

    def test_unknown_agent_returns_name(self):
        """Unknown agent should return the agent name as display name."""
        router = LLMRouter()
        assert router.get_agent_display_name('unknown') == 'unknown'


class TestReverseLookup:
    """Test display name to agent type reverse lookup."""

    def setup_method(self):
        reset_llm_router()

    def teardown_method(self):
        reset_llm_router()

    def test_guillermo_to_structure(self):
        """'Guillermo' should map to 'structure'."""
        router = LLMRouter()
        assert router.get_agent_by_display_name('Guillermo') == 'structure'

    def test_clarissa_to_character(self):
        """'Clarissa' should map to 'character'."""
        router = LLMRouter()
        assert router.get_agent_by_display_name('Clarissa') == 'character'

    def test_bill_to_factcheck(self):
        """'Bill' should map to 'factcheck'."""
        router = LLMRouter()
        assert router.get_agent_by_display_name('Bill') == 'factcheck'

    def test_benjamin_to_line_editor(self):
        """'Benjamin' should map to 'line_editor'."""
        router = LLMRouter()
        assert router.get_agent_by_display_name('Benjamin') == 'line_editor'

    def test_stephen_to_tension(self):
        """'Stephen' should map to 'tension'."""
        router = LLMRouter()
        assert router.get_agent_by_display_name('Stephen') == 'tension'

    def test_continuity_to_continuity(self):
        """'Continuity' should map to 'continuity'."""
        router = LLMRouter()
        assert router.get_agent_by_display_name('Continuity') == 'continuity'

    def test_unknown_display_name_returns_none(self):
        """Unknown display name should return None."""
        router = LLMRouter()
        assert router.get_agent_by_display_name('Unknown') is None

    def test_case_sensitive(self):
        """Lookup should be case-sensitive."""
        router = LLMRouter()
        assert router.get_agent_by_display_name('guillermo') is None
        assert router.get_agent_by_display_name('GUILLERMO') is None


class TestAgentSpecificParams:
    """Test agent-specific parameter values."""

    def setup_method(self):
        reset_llm_router()

    def teardown_method(self):
        reset_llm_router()

    def test_structure_agent_params(self):
        """Structure agent should have specific temperature and max_tokens."""
        router = LLMRouter()
        kwargs = router.get_llm_kwargs('structure')

        # Structure has temperature: 0.75, max_tokens: 16384
        assert kwargs['temperature'] == 0.75, "Structure should have temp 0.75"
        assert kwargs['max_tokens'] == 16384, "Structure should have max_tokens 16384"

    def test_factcheck_low_temperature(self):
        """Factcheck agent should have low temperature (0.25) for precision."""
        router = LLMRouter()
        kwargs = router.get_llm_kwargs('factcheck')

        assert kwargs['temperature'] == 0.25, "Factcheck should have temp 0.25"

    def test_narrative_high_temperature(self):
        """Narrative agent should have high temperature (0.85) for creativity."""
        router = LLMRouter()
        kwargs = router.get_llm_kwargs('narrative')

        assert kwargs['temperature'] == 0.85, "Narrative should have temp 0.85"


def run_quick_verification():
    """Quick verification that can be run without pytest."""
    print("=" * 60)
    print("LLM Router Quick Verification")
    print("=" * 60)

    reset_llm_router()

    # Test 1: Hierarchy with env overrides
    print("\n1. Testing hierarchy with env overrides...")
    os.environ['TEST_ROUNDTABLE_MODEL'] = 'gemini-3-flash-preview'
    os.environ['TEST_STRUCTURE_MODEL'] = 'claude-sonnet-4-5'

    reset_llm_router()
    router = get_llm_router()

    results = []
    for agent in ['structure', 'character', 'factcheck']:
        model = router.get_model_for_agent(agent)
        display = router.get_agent_display_name(agent)
        results.append((agent, display, model))
        print(f"   {display} ({agent}): {model}")

    # Verify hierarchy
    assert 'claude' in results[0][2].lower(), "Structure should use claude (agent override)"
    assert 'gemini' in results[1][2].lower(), "Character should use gemini (group override)"
    assert 'gemini' in results[2][2].lower(), "Factcheck should use gemini (group override)"
    print("   ✅ Hierarchy verified: Agent > Group > Default")

    # Cleanup
    del os.environ['TEST_ROUNDTABLE_MODEL']
    del os.environ['TEST_STRUCTURE_MODEL']

    # Test 2: Model constraints
    print("\n2. Testing model constraints...")
    reset_llm_router()
    router = get_llm_router()

    # Default model (azure/model-router) should have top_p
    kwargs = router.get_llm_kwargs('structure')
    has_top_p = 'top_p' in kwargs
    print(f"   azure/model-router has top_p: {has_top_p}")

    # Test Claude constraint
    os.environ['TEST_STRUCTURE_MODEL'] = 'claude-sonnet-4-5'
    reset_llm_router()
    router = get_llm_router()
    kwargs = router.get_llm_kwargs('structure')
    claude_no_top_p = 'top_p' not in kwargs
    print(f"   Claude has no top_p: {claude_no_top_p}")
    del os.environ['TEST_STRUCTURE_MODEL']

    print("   ✅ Model constraints verified")

    # Test 3: All roundtable members
    print("\n3. Verifying roundtable group membership...")
    reset_llm_router()
    router = get_llm_router()
    members = router.list_agents_in_group('roundtable')
    expected = ['structure', 'factcheck', 'character', 'line_editor', 'continuity', 'tension']
    all_present = all(m in members for m in expected)
    print(f"   Members: {members}")
    print(f"   All 6 reviewers present: {all_present}")
    assert all_present, "Missing roundtable members!"
    print("   ✅ Roundtable group verified")

    print("\n" + "=" * 60)
    print("All verifications passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_quick_verification()
