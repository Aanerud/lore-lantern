"""
LLM service for Kids Storyteller V2

Provides configured LLM instances for agents.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Optional
import os


class LLMService:
    """Service for managing LLM instances"""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-1.5-flash",
        temperature: float = 0.7,
        max_tokens: int = 2048
    ):
        """
        Initialize LLM service.

        Args:
            api_key: Google Gemini API key
            model: Model name (default: gemini-1.5-flash)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum output tokens
        """
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Set environment variable for LangChain
        os.environ["GOOGLE_API_KEY"] = api_key

        self._llm: Optional[ChatGoogleGenerativeAI] = None

    def get_llm(self) -> ChatGoogleGenerativeAI:
        """
        Get configured LLM instance.

        Returns:
            ChatGoogleGenerativeAI instance
        """
        if self._llm is None:
            self._llm = ChatGoogleGenerativeAI(
                model=self.model,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                convert_system_message_to_human=True  # Gemini compatibility
            )

        return self._llm

    def get_creative_llm(self) -> ChatGoogleGenerativeAI:
        """
        Get LLM configured for creative tasks (higher temperature).

        Returns:
            ChatGoogleGenerativeAI with temperature=0.9
        """
        return ChatGoogleGenerativeAI(
            model=self.model,
            temperature=0.9,
            max_output_tokens=self.max_tokens,
            convert_system_message_to_human=True
        )

    def get_factual_llm(self) -> ChatGoogleGenerativeAI:
        """
        Get LLM configured for factual tasks (lower temperature).

        Returns:
            ChatGoogleGenerativeAI with temperature=0.3
        """
        return ChatGoogleGenerativeAI(
            model=self.model,
            temperature=0.3,
            max_output_tokens=self.max_tokens,
            convert_system_message_to_human=True
        )
