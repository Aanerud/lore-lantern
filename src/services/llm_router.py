"""
Centralized LLM Router Service

Handles:
1. Model configuration loading from YAML
2. Hierarchy resolution (Agent > Group > Default)
3. Environment variable overrides for A/B testing
4. Provider-specific parameter constraints (Claude, OpenAI reasoning, Azure)

Usage:
    from src.services.llm_router import get_llm_router

    router = get_llm_router()
    llm_kwargs = router.get_llm_kwargs("structure")  # Returns ready-to-use kwargs
    model = router.get_model_for_agent("line_editor")  # Get model name only
"""

import os
import yaml
import logging
from typing import Dict, Optional, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)


class LLMRouter:
    """Centralized LLM routing with model-aware parameter handling"""

    def __init__(self, config_path: str = None):
        """
        Initialize LLM Router.

        Args:
            config_path: Path to models.yaml config file.
                         If None, uses src/config/models.yaml
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "models.yaml"
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self._apply_env_overrides()

    def _load_config(self, config_path: str) -> dict:
        """Load YAML config file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _apply_env_overrides(self):
        """
        Apply environment variable overrides for A/B testing.

        Supports:
        - TEST_<GROUP_NAME>_MODEL: Override all agents in a group
        - TEST_<AGENT_NAME>_MODEL: Override specific agent
        """
        # Group overrides: TEST_ROUNDTABLE_MODEL, TEST_WRITERS_MODEL, TEST_POST_MODEL
        for group_name in self.config.get("groups", {}):
            env_key = f"TEST_{group_name.upper()}_MODEL"
            env_value = os.getenv(env_key)
            if env_value:
                self.config["groups"][group_name]["model"] = env_value
                logger.info(f"🔬 A/B Override: {group_name} group → {env_value}")

        # Agent overrides: TEST_STRUCTURE_MODEL, TEST_NARRATIVE_MODEL, etc.
        for agent_name in self.config.get("agents", {}):
            env_key = f"TEST_{agent_name.upper()}_MODEL"
            env_value = os.getenv(env_key)
            if env_value:
                self.config["agents"][agent_name]["model"] = env_value
                logger.info(f"🔬 A/B Override: {agent_name} → {env_value}")

    def _normalize_model_name(self, model: str) -> str:
        """
        Add provider prefix if missing (for LiteLLM routing).

        LiteLLM requires provider prefixes for routing:
        - gemini/ for Google AI Studio
        - azure/ for Azure OpenAI
        - anthropic/ for Anthropic (optional, works without)

        Args:
            model: Model name, possibly without prefix

        Returns:
            Model name with appropriate provider prefix
        """
        if not model:
            return model

        # Already has a known prefix
        known_prefixes = ['gemini/', 'vertex_ai/', 'openai/', 'azure/', 'azure_ai/',
                          'anthropic/', 'claude-', 'gpt-', 'o1-', 'o3-']
        if any(model.startswith(prefix) for prefix in known_prefixes):
            return model

        # Auto-add gemini/ prefix for Gemini models
        if model.startswith('gemini-'):
            return f'gemini/{model}'

        return model

    def get_model_for_agent(self, agent_name: str) -> str:
        """
        Get model for agent using hierarchy:
        1. Agent-specific model (if set)
        2. Group model (if agent is in a group with model set)
        3. Default model

        Args:
            agent_name: Agent identifier (e.g., "structure", "line_editor")

        Returns:
            Normalized model name ready for LiteLLM
        """
        # 1. Check agent-specific model
        agent_cfg = self.config.get("agents", {}).get(agent_name, {})
        if agent_cfg.get("model"):
            return self._normalize_model_name(agent_cfg["model"])

        # 2. Check group model
        for group_name, group_cfg in self.config.get("groups", {}).items():
            if agent_name in group_cfg.get("members", []):
                if group_cfg.get("model"):
                    return self._normalize_model_name(group_cfg["model"])

        # 3. Default
        default_model = self.config.get("default", {}).get("model", "azure/model-router")
        return self._normalize_model_name(default_model)

    def get_llm_kwargs(self, agent_name: str, model_override: str = None) -> dict:
        """
        Build complete LLM kwargs for an agent, including:
        - Model name
        - Agent-specific parameters (temp, max_tokens, timeout)
        - Provider-specific constraints (Claude, OpenAI reasoning, etc.)

        Args:
            agent_name: Agent identifier
            model_override: Optional model to use instead of config

        Returns:
            Dict ready to pass to CrewAI LLM(**kwargs)
        """
        if model_override:
            model = self._normalize_model_name(model_override)
        else:
            model = self.get_model_for_agent(agent_name)

        agent_cfg = self.config.get("agents", {}).get(agent_name, {})
        default_cfg = self.config.get("default", {})

        # Base kwargs with agent-specific or default values
        kwargs = {
            "model": model,
            "temperature": agent_cfg.get("temperature", default_cfg.get("temperature", 0.7)),
            "max_tokens": agent_cfg.get("max_tokens", default_cfg.get("max_tokens", 8000)),
            "timeout": agent_cfg.get("timeout", default_cfg.get("timeout", 900)),
            "drop_params": True,  # Auto-drop unsupported params
        }

        # Add optional params if specified in config
        if agent_cfg.get("frequency_penalty") is not None:
            kwargs["frequency_penalty"] = agent_cfg["frequency_penalty"]
        if agent_cfg.get("presence_penalty") is not None:
            kwargs["presence_penalty"] = agent_cfg["presence_penalty"]

        # Apply provider-specific constraints
        kwargs = self._apply_model_constraints(model, kwargs, agent_cfg)

        return kwargs

    def _apply_model_constraints(self, model: str, kwargs: dict, agent_cfg: dict = None) -> dict:
        """
        Handle provider-specific parameter constraints.

        - Claude: No temperature + top_p together
        - Azure Model Router: Add reasoning_effort
        - OpenAI Reasoning (o1, o3, gpt-5): Very limited params
        - Azure AI Claude: No stop_sequences

        Args:
            model: Normalized model name
            kwargs: Current kwargs dict
            agent_cfg: Agent config for custom top_p

        Returns:
            Modified kwargs dict
        """
        model_lower = model.lower()
        agent_cfg = agent_cfg or {}

        # Claude: No temperature + top_p together
        is_claude = "claude" in model_lower or "anthropic" in model_lower
        if not is_claude:
            # Use agent-specific top_p if set, otherwise default 0.95
            kwargs["top_p"] = agent_cfg.get("top_p", 0.95)
        # (If Claude, don't add top_p at all)

        # Azure Model Router: Add reasoning_effort
        if "model-router" in model_lower:
            kwargs.setdefault("extra_body", {})
            kwargs["extra_body"]["reasoning_effort"] = "medium"

        # OpenAI Reasoning Models (o1, o3, gpt-5): Very limited params
        is_openai_reasoning = any(x in model_lower for x in ["gpt-5", "o3-", "o1-"])
        if is_openai_reasoning:
            kwargs["additional_drop_params"] = [
                "temperature", "top_p", "frequency_penalty",
                "presence_penalty", "max_tokens", "stop", "logit_bias",
                "stop_sequences"
            ]
            # Use max_completion_tokens instead of max_tokens
            max_tokens = kwargs.pop("max_tokens", 16384)
            kwargs.setdefault("extra_body", {})
            kwargs["extra_body"]["max_completion_tokens"] = max_tokens
            # Remove params that cause errors even before drop_params
            kwargs.pop("temperature", None)
            kwargs.pop("top_p", None)
            kwargs.pop("frequency_penalty", None)
            kwargs.pop("presence_penalty", None)

        # Azure AI Claude: No stop_sequences
        if model_lower.startswith("azure_ai/"):
            kwargs.setdefault("additional_drop_params", [])
            if isinstance(kwargs["additional_drop_params"], list):
                kwargs["additional_drop_params"].extend(["stop_sequences", "stop"])

        return kwargs

    def get_agent_display_name(self, agent_name: str) -> str:
        """
        Get human-readable display name for agent.

        Args:
            agent_name: Agent identifier

        Returns:
            Display name (e.g., "Guillermo" for structure agent)
        """
        return self.config.get("agents", {}).get(agent_name, {}).get("display_name", agent_name)

    def get_agent_description(self, agent_name: str) -> str:
        """Get agent description."""
        return self.config.get("agents", {}).get(agent_name, {}).get("description", "")

    def get_agent_by_display_name(self, display_name: str) -> Optional[str]:
        """
        Reverse lookup: get agent type from display name.

        Args:
            display_name: Display name like "Guillermo", "Clarissa", etc.

        Returns:
            Agent type like "structure", "character", or None if not found.
        """
        for agent_name, agent_cfg in self.config.get("agents", {}).items():
            if agent_cfg.get("display_name") == display_name:
                return agent_name
        return None

    def list_agents_in_group(self, group_name: str) -> List[str]:
        """
        Get list of agent names in a group.

        Args:
            group_name: Group identifier (e.g., "roundtable")

        Returns:
            List of agent names in the group
        """
        return self.config.get("groups", {}).get(group_name, {}).get("members", [])

    def list_all_agents(self) -> List[str]:
        """Get list of all configured agent names."""
        return list(self.config.get("agents", {}).keys())

    def get_active_overrides(self) -> Dict[str, str]:
        """
        Get currently active model overrides.

        Returns:
            Dict mapping agent/group name to override model
        """
        overrides = {}

        # Check group overrides
        for group_name, group_cfg in self.config.get("groups", {}).items():
            if group_cfg.get("model"):
                overrides[f"group:{group_name}"] = group_cfg["model"]

        # Check agent overrides
        for agent_name, agent_cfg in self.config.get("agents", {}).items():
            if agent_cfg.get("model"):
                overrides[f"agent:{agent_name}"] = agent_cfg["model"]

        return overrides

    def log_configuration(self, use_print: bool = True):
        """
        Log current model configuration for all agents.

        Args:
            use_print: If True, use print() for console visibility.
                       If False, use logger.info() for file logs only.
        """
        output = print if use_print else logger.info

        output("📋 LLM Router Configuration:")

        # Log active overrides
        overrides = self.get_active_overrides()
        if overrides:
            output("  🔄 Active overrides (from env vars):")
            for name, model in overrides.items():
                output(f"    • {name} → {model}")

        # Log each agent's effective model
        output("  🤖 Agent models:")
        for agent_name in self.list_all_agents():
            model = self.get_model_for_agent(agent_name)
            display = self.get_agent_display_name(agent_name)
            # Show source of model selection
            source = self._get_model_source(agent_name)
            output(f"    • {display} ({agent_name}): {model} [{source}]")

    def _get_model_source(self, agent_name: str) -> str:
        """Get the source of model selection for an agent (for logging)."""
        # Check agent-specific model
        agent_cfg = self.config.get("agents", {}).get(agent_name, {})
        if agent_cfg.get("model"):
            return "agent override"

        # Check group model
        for group_name, group_cfg in self.config.get("groups", {}).items():
            if agent_name in group_cfg.get("members", []):
                if group_cfg.get("model"):
                    return f"group:{group_name}"

        return "default"


# Singleton instance
_llm_router: Optional[LLMRouter] = None


def get_llm_router() -> LLMRouter:
    """
    Get singleton LLM Router instance.

    Returns:
        LLMRouter instance (creates one if not initialized)
    """
    global _llm_router
    if _llm_router is None:
        _llm_router = LLMRouter()
    return _llm_router


def init_llm_router(config_path: str = None) -> LLMRouter:
    """
    Initialize LLM Router (call at app startup).

    Args:
        config_path: Optional path to models.yaml

    Returns:
        Initialized LLMRouter instance
    """
    global _llm_router
    _llm_router = LLMRouter(config_path)
    return _llm_router


def reset_llm_router():
    """Reset the singleton (useful for testing)."""
    global _llm_router
    _llm_router = None
