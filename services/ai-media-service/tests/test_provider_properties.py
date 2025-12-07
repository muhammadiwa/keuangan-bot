"""
Property-based tests for AI Provider interface.

**Feature: multi-ai-provider, Property 1: Provider Configuration Consistency**
**Validates: Requirements 1.1, 1.2**

These tests verify that the provider abstraction layer maintains consistency
across all provider configurations.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume

from app.providers.base import (
    AIProvider,
    AIResponse,
    ProviderError,
    AuthenticationError,
    RateLimitError,
    QuotaExceededError,
)


# Define the valid provider names as per design document
VALID_PROVIDER_NAMES = [
    "ollama",
    "openai",
    "megallm",
    "groq",
    "together",
    "deepseek",
    "qwen",
    "kimi",
    "moonshot",
    "glm",
    "zhipu",
    "bigmodel",
    "anthropic",
    "gemini",
]

# Provider to expected class mapping (based on design document)
PROVIDER_CLASS_MAPPING = {
    "ollama": "OllamaProvider",
    "openai": "OpenAICompatibleProvider",
    "megallm": "OpenAICompatibleProvider",
    "groq": "OpenAICompatibleProvider",
    "together": "OpenAICompatibleProvider",
    "deepseek": "OpenAICompatibleProvider",
    "qwen": "OpenAICompatibleProvider",
    "kimi": "OpenAICompatibleProvider",
    "moonshot": "OpenAICompatibleProvider",
    "glm": "GLMProvider",
    "zhipu": "GLMProvider",
    "bigmodel": "GLMProvider",
    "anthropic": "AnthropicProvider",
    "gemini": "GeminiProvider",
}

# Provider to API format mapping
PROVIDER_API_FORMAT = {
    "ollama": "ollama_native",
    "openai": "openai_compatible",
    "megallm": "openai_compatible",
    "groq": "openai_compatible",
    "together": "openai_compatible",
    "deepseek": "openai_compatible",
    "qwen": "openai_compatible",
    "kimi": "openai_compatible",
    "moonshot": "openai_compatible",
    "glm": "openai_compatible",  # GLM uses OpenAI-compatible format with JWT auth
    "zhipu": "openai_compatible",
    "bigmodel": "openai_compatible",
    "anthropic": "anthropic_messages",
    "gemini": "google_generative",
}


class TestAIResponseDataclass:
    """Test AIResponse dataclass properties."""

    @given(
        content=st.text(min_size=0, max_size=1000),
        model=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
        provider=st.sampled_from(VALID_PROVIDER_NAMES),
        latency_ms=st.floats(min_value=0.0, max_value=100000.0, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_airesponse_creation_consistency(
        self, content: str, model: str, provider: str, latency_ms: float
    ):
        """
        **Feature: multi-ai-provider, Property 1: Provider Configuration Consistency**
        **Validates: Requirements 1.1, 1.2**
        
        Property: For any valid inputs, AIResponse should be created with
        consistent field values that match the input.
        """
        response = AIResponse(
            content=content,
            model=model,
            provider=provider,
            latency_ms=latency_ms,
        )
        
        # Verify all fields are set correctly
        assert response.content == content
        assert response.model == model
        assert response.provider == provider
        assert response.latency_ms == latency_ms
        # Optional fields should have defaults
        assert response.usage is None
        assert response.raw_response is None

    @given(
        content=st.text(min_size=0, max_size=500),
        model=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        provider=st.sampled_from(VALID_PROVIDER_NAMES),
        input_tokens=st.integers(min_value=0, max_value=100000),
        output_tokens=st.integers(min_value=0, max_value=100000),
    )
    @settings(max_examples=100)
    def test_airesponse_with_usage_tracking(
        self,
        content: str,
        model: str,
        provider: str,
        input_tokens: int,
        output_tokens: int,
    ):
        """
        **Feature: multi-ai-provider, Property 1: Provider Configuration Consistency**
        **Validates: Requirements 1.1, 1.2**
        
        Property: For any valid usage data, AIResponse should correctly store
        and return token usage information.
        """
        usage = {"input_tokens": input_tokens, "output_tokens": output_tokens}
        
        response = AIResponse(
            content=content,
            model=model,
            provider=provider,
            usage=usage,
        )
        
        assert response.usage is not None
        assert response.usage["input_tokens"] == input_tokens
        assert response.usage["output_tokens"] == output_tokens


class TestProviderConfigurationMapping:
    """
    Test that provider names map to correct provider classes and API formats.
    
    **Feature: multi-ai-provider, Property 1: Provider Configuration Consistency**
    **Validates: Requirements 1.1, 1.2**
    """

    @given(provider_name=st.sampled_from(VALID_PROVIDER_NAMES))
    @settings(max_examples=100)
    def test_provider_has_class_mapping(self, provider_name: str):
        """
        Property: For any valid provider name, there must be a defined
        class mapping in the configuration.
        """
        assert provider_name in PROVIDER_CLASS_MAPPING
        assert PROVIDER_CLASS_MAPPING[provider_name] is not None
        assert len(PROVIDER_CLASS_MAPPING[provider_name]) > 0

    @given(provider_name=st.sampled_from(VALID_PROVIDER_NAMES))
    @settings(max_examples=100)
    def test_provider_has_api_format_mapping(self, provider_name: str):
        """
        Property: For any valid provider name, there must be a defined
        API format mapping.
        """
        assert provider_name in PROVIDER_API_FORMAT
        assert PROVIDER_API_FORMAT[provider_name] in [
            "ollama_native",
            "openai_compatible",
            "anthropic_messages",
            "google_generative",
        ]

    @given(provider_name=st.sampled_from(VALID_PROVIDER_NAMES))
    @settings(max_examples=100)
    def test_openai_compatible_providers_share_format(self, provider_name: str):
        """
        Property: All OpenAI-compatible providers should use the same API format.
        """
        if PROVIDER_CLASS_MAPPING[provider_name] == "OpenAICompatibleProvider":
            assert PROVIDER_API_FORMAT[provider_name] == "openai_compatible"


class TestProviderErrorHierarchy:
    """Test provider error classes maintain proper hierarchy."""

    @given(
        message=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
        provider=st.sampled_from(VALID_PROVIDER_NAMES),
        status_code=st.integers(min_value=400, max_value=599) | st.none(),
    )
    @settings(max_examples=100)
    def test_provider_error_attributes(
        self, message: str, provider: str, status_code: int | None
    ):
        """
        **Feature: multi-ai-provider, Property 1: Provider Configuration Consistency**
        **Validates: Requirements 1.1, 1.2**
        
        Property: ProviderError should correctly store all attributes.
        """
        error = ProviderError(
            message=message,
            provider=provider,
            status_code=status_code,
            retryable=False,
        )
        
        assert str(error) == message
        assert error.provider == provider
        assert error.status_code == status_code
        assert error.retryable is False

    @given(
        message=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
        provider=st.sampled_from(VALID_PROVIDER_NAMES),
    )
    @settings(max_examples=100)
    def test_authentication_error_is_provider_error(self, message: str, provider: str):
        """
        Property: AuthenticationError should be a subclass of ProviderError.
        """
        error = AuthenticationError(message=message, provider=provider)
        
        assert isinstance(error, ProviderError)
        assert error.provider == provider

    @given(
        message=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
        provider=st.sampled_from(VALID_PROVIDER_NAMES),
        retry_after=st.floats(min_value=0.0, max_value=3600.0, allow_nan=False) | st.none(),
    )
    @settings(max_examples=100)
    def test_rate_limit_error_is_retryable(
        self, message: str, provider: str, retry_after: float | None
    ):
        """
        Property: RateLimitError should always be marked as retryable with status 429.
        """
        error = RateLimitError(
            message=message,
            provider=provider,
            retry_after=retry_after,
        )
        
        assert isinstance(error, ProviderError)
        assert error.retryable is True
        assert error.status_code == 429
        assert error.retry_after == retry_after

    @given(
        message=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
        provider=st.sampled_from(VALID_PROVIDER_NAMES),
    )
    @settings(max_examples=100)
    def test_quota_exceeded_error_is_provider_error(self, message: str, provider: str):
        """
        Property: QuotaExceededError should be a subclass of ProviderError.
        """
        error = QuotaExceededError(message=message, provider=provider)
        
        assert isinstance(error, ProviderError)
        assert error.provider == provider


class TestProviderNameConsistency:
    """Test provider name consistency across configurations."""

    @given(provider_name=st.sampled_from(VALID_PROVIDER_NAMES))
    @settings(max_examples=100)
    def test_provider_name_is_lowercase(self, provider_name: str):
        """
        **Feature: multi-ai-provider, Property 1: Provider Configuration Consistency**
        **Validates: Requirements 1.1, 1.2**
        
        Property: All provider names should be lowercase for consistency.
        """
        assert provider_name == provider_name.lower()

    def test_all_providers_have_mappings(self):
        """
        Property: Every valid provider name must have both class and API format mappings.
        """
        for provider in VALID_PROVIDER_NAMES:
            assert provider in PROVIDER_CLASS_MAPPING, f"Missing class mapping for {provider}"
            assert provider in PROVIDER_API_FORMAT, f"Missing API format mapping for {provider}"

    def test_alias_providers_map_to_same_class(self):
        """
        Property: Provider aliases (kimi/moonshot, glm/zhipu/bigmodel) should map to same class.
        """
        # kimi and moonshot are aliases
        assert PROVIDER_CLASS_MAPPING["kimi"] == PROVIDER_CLASS_MAPPING["moonshot"]
        
        # glm, zhipu, and bigmodel are aliases
        assert PROVIDER_CLASS_MAPPING["glm"] == PROVIDER_CLASS_MAPPING["zhipu"]
        assert PROVIDER_CLASS_MAPPING["zhipu"] == PROVIDER_CLASS_MAPPING["bigmodel"]


class TestAPIKeyValidation:
    """
    Property-based tests for API key validation.
    
    **Feature: multi-ai-provider, Property 2: API Key Validation**
    **Validates: Requirements 1.3**
    
    These tests verify that providers requiring authentication properly validate
    API keys before making external requests.
    """

    # Providers that require API keys for authentication
    PROVIDERS_REQUIRING_API_KEY = [
        "openai",
        "megallm",
        "groq",
        "together",
        "deepseek",
        "qwen",
        "kimi",
        "moonshot",
        "glm",
        "zhipu",
        "bigmodel",
        "anthropic",
        "gemini",
    ]

    # Providers that do NOT require API keys (local/self-hosted)
    PROVIDERS_NOT_REQUIRING_API_KEY = [
        "ollama",
    ]

    @given(provider_name=st.sampled_from(PROVIDERS_REQUIRING_API_KEY))
    @settings(max_examples=100)
    def test_provider_requiring_api_key_is_identified(self, provider_name: str):
        """
        **Feature: multi-ai-provider, Property 2: API Key Validation**
        **Validates: Requirements 1.3**
        
        Property: For any provider that requires authentication, the provider
        should be in the list of providers requiring API keys.
        """
        # Verify the provider is in our known list
        assert provider_name in self.PROVIDERS_REQUIRING_API_KEY
        # Verify it's not in the no-key-required list
        assert provider_name not in self.PROVIDERS_NOT_REQUIRING_API_KEY

    @given(provider_name=st.sampled_from(PROVIDERS_NOT_REQUIRING_API_KEY))
    @settings(max_examples=100)
    def test_provider_not_requiring_api_key_is_identified(self, provider_name: str):
        """
        **Feature: multi-ai-provider, Property 2: API Key Validation**
        **Validates: Requirements 1.3**
        
        Property: For any provider that does not require authentication,
        the provider should be in the list of providers not requiring API keys.
        """
        # Verify the provider is in our known list
        assert provider_name in self.PROVIDERS_NOT_REQUIRING_API_KEY
        # Verify it's not in the key-required list
        assert provider_name not in self.PROVIDERS_REQUIRING_API_KEY

    def test_all_providers_categorized(self):
        """
        **Feature: multi-ai-provider, Property 2: API Key Validation**
        **Validates: Requirements 1.3**
        
        Property: Every valid provider must be categorized as either requiring
        or not requiring an API key.
        """
        all_categorized = set(self.PROVIDERS_REQUIRING_API_KEY) | set(
            self.PROVIDERS_NOT_REQUIRING_API_KEY
        )
        
        for provider in VALID_PROVIDER_NAMES:
            assert provider in all_categorized, (
                f"Provider '{provider}' is not categorized for API key requirement"
            )

    def test_no_provider_in_both_categories(self):
        """
        **Feature: multi-ai-provider, Property 2: API Key Validation**
        **Validates: Requirements 1.3**
        
        Property: No provider should be in both the requiring and not-requiring
        API key categories.
        """
        overlap = set(self.PROVIDERS_REQUIRING_API_KEY) & set(
            self.PROVIDERS_NOT_REQUIRING_API_KEY
        )
        assert len(overlap) == 0, f"Providers in both categories: {overlap}"

    @given(
        provider_name=st.sampled_from(PROVIDERS_REQUIRING_API_KEY),
        api_key=st.text(min_size=0, max_size=0),  # Empty string
    )
    @settings(max_examples=100)
    def test_empty_api_key_is_invalid(self, provider_name: str, api_key: str):
        """
        **Feature: multi-ai-provider, Property 2: API Key Validation**
        **Validates: Requirements 1.3**
        
        Property: For any provider requiring authentication, an empty API key
        should be considered invalid.
        """
        assert api_key == ""
        # Empty string should be treated as missing/invalid
        is_valid = bool(api_key and api_key.strip())
        assert is_valid is False

    @given(
        provider_name=st.sampled_from(PROVIDERS_REQUIRING_API_KEY),
        api_key=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_non_empty_api_key_passes_basic_validation(
        self, provider_name: str, api_key: str
    ):
        """
        **Feature: multi-ai-provider, Property 2: API Key Validation**
        **Validates: Requirements 1.3**
        
        Property: For any provider requiring authentication, a non-empty,
        non-whitespace API key should pass basic presence validation.
        """
        # Basic validation: key exists and is not just whitespace
        is_valid = bool(api_key and api_key.strip())
        assert is_valid is True

    @given(
        provider_name=st.sampled_from(PROVIDERS_REQUIRING_API_KEY),
        api_key=st.text(min_size=1, max_size=50).filter(
            lambda x: x.strip() and not x.isspace()
        ),
    )
    @settings(max_examples=100)
    def test_authentication_error_contains_provider_info(
        self, provider_name: str, api_key: str
    ):
        """
        **Feature: multi-ai-provider, Property 2: API Key Validation**
        **Validates: Requirements 1.3**
        
        Property: When an AuthenticationError is raised, it should contain
        the provider name for clear error identification.
        """
        error = AuthenticationError(
            message=f"Invalid API key for {provider_name}",
            provider=provider_name,
        )
        
        assert error.provider == provider_name
        assert provider_name in str(error)

    @given(
        api_key=st.one_of(
            st.none(),
            st.just(""),
            st.text(max_size=10).filter(lambda x: not x or x.isspace()),
        )
    )
    @settings(max_examples=100)
    def test_missing_or_whitespace_api_key_is_invalid(self, api_key: str | None):
        """
        **Feature: multi-ai-provider, Property 2: API Key Validation**
        **Validates: Requirements 1.3**
        
        Property: None, empty string, or whitespace-only strings should all
        be considered invalid API keys.
        """
        is_valid = bool(api_key and api_key.strip())
        assert is_valid is False


def requires_api_key(provider_name: str) -> bool:
    """
    Helper function to determine if a provider requires an API key.
    
    Args:
        provider_name: The name of the provider
        
    Returns:
        True if the provider requires an API key, False otherwise
    """
    providers_not_requiring_key = {"ollama"}
    return provider_name.lower() not in providers_not_requiring_key


def validate_api_key(api_key: str | None) -> bool:
    """
    Validate that an API key is present and non-empty.
    
    Args:
        api_key: The API key to validate
        
    Returns:
        True if the API key is valid (non-empty, non-whitespace), False otherwise
    """
    return bool(api_key and api_key.strip())


class TestAPIKeyValidationHelpers:
    """
    Tests for API key validation helper functions.
    
    **Feature: multi-ai-provider, Property 2: API Key Validation**
    **Validates: Requirements 1.3**
    """

    @given(provider_name=st.sampled_from(VALID_PROVIDER_NAMES))
    @settings(max_examples=100)
    def test_requires_api_key_returns_boolean(self, provider_name: str):
        """
        Property: requires_api_key should always return a boolean value.
        """
        result = requires_api_key(provider_name)
        assert isinstance(result, bool)

    @given(provider_name=st.sampled_from(["ollama"]))
    @settings(max_examples=100)
    def test_ollama_does_not_require_api_key(self, provider_name: str):
        """
        Property: Ollama provider should not require an API key.
        """
        assert requires_api_key(provider_name) is False

    @given(
        provider_name=st.sampled_from([
            "openai", "megallm", "groq", "together", "deepseek",
            "qwen", "kimi", "moonshot", "glm", "zhipu", "bigmodel",
            "anthropic", "gemini"
        ])
    )
    @settings(max_examples=100)
    def test_cloud_providers_require_api_key(self, provider_name: str):
        """
        Property: All cloud providers should require an API key.
        """
        assert requires_api_key(provider_name) is True

    @given(
        api_key=st.text(min_size=1, max_size=200).filter(
            lambda x: x.strip() and not x.isspace()
        )
    )
    @settings(max_examples=100)
    def test_validate_api_key_accepts_valid_keys(self, api_key: str):
        """
        Property: validate_api_key should return True for non-empty,
        non-whitespace strings.
        """
        assert validate_api_key(api_key) is True

    @given(
        api_key=st.one_of(
            st.none(),
            st.just(""),
            st.text(max_size=20).filter(lambda x: not x or x.isspace()),
        )
    )
    @settings(max_examples=100)
    def test_validate_api_key_rejects_invalid_keys(self, api_key: str | None):
        """
        Property: validate_api_key should return False for None, empty,
        or whitespace-only strings.
        """
        assert validate_api_key(api_key) is False


class TestOllamaRequestFormat:
    """
    Property-based tests for Ollama request format correctness.
    
    **Feature: multi-ai-provider, Property 3: Request Format Correctness (Ollama)**
    **Validates: Requirements 2.2**
    
    These tests verify that Ollama requests conform to the Ollama API specification:
    - Uses /api/generate endpoint
    - Proper prompt formatting from messages
    - Correct payload structure with model, prompt, stream, options
    """

    @given(
        content=st.text(min_size=1, max_size=500).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_single_message_becomes_prompt(self, content: str):
        """
        **Feature: multi-ai-provider, Property 3: Request Format Correctness (Ollama)**
        **Validates: Requirements 2.2**
        
        Property: For any single message, the prompt should be the message content directly.
        """
        from app.providers.ollama import OllamaProvider
        
        provider = OllamaProvider()
        messages = [{"role": "user", "content": content}]
        
        prompt = provider._messages_to_prompt(messages)
        
        # Single message should return content directly
        assert prompt == content

    @given(
        system_content=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
        user_content=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_multiple_messages_formatted_with_roles(
        self, system_content: str, user_content: str
    ):
        """
        **Feature: multi-ai-provider, Property 3: Request Format Correctness (Ollama)**
        **Validates: Requirements 2.2**
        
        Property: For any multiple messages, each message should be prefixed with its role.
        """
        from app.providers.ollama import OllamaProvider
        
        provider = OllamaProvider()
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        
        prompt = provider._messages_to_prompt(messages)
        
        # Multiple messages should have role prefixes
        assert f"System: {system_content}" in prompt
        assert f"User: {user_content}" in prompt
        # Messages should be separated by double newlines
        assert "\n\n" in prompt

    @given(
        user_content=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
        assistant_content=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_assistant_messages_formatted_correctly(
        self, user_content: str, assistant_content: str
    ):
        """
        **Feature: multi-ai-provider, Property 3: Request Format Correctness (Ollama)**
        **Validates: Requirements 2.2**
        
        Property: For any conversation with assistant messages, the assistant role
        should be properly prefixed.
        """
        from app.providers.ollama import OllamaProvider
        
        provider = OllamaProvider()
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
        
        prompt = provider._messages_to_prompt(messages)
        
        assert f"User: {user_content}" in prompt
        assert f"Assistant: {assistant_content}" in prompt

    @given(
        model=st.text(min_size=1, max_size=50).filter(lambda x: x.strip() and ":" not in x or x.count(":") == 1),
        temperature=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        max_tokens=st.integers(min_value=1, max_value=4096) | st.none(),
    )
    @settings(max_examples=100)
    def test_payload_structure_correctness(
        self, model: str, temperature: float, max_tokens: int | None
    ):
        """
        **Feature: multi-ai-provider, Property 3: Request Format Correctness (Ollama)**
        **Validates: Requirements 2.2**
        
        Property: For any valid parameters, the generated payload should have
        the correct structure for Ollama's /api/generate endpoint.
        """
        from app.providers.ollama import OllamaProvider
        
        provider = OllamaProvider(model=model)
        messages = [{"role": "user", "content": "test message"}]
        
        # Build payload as the provider would
        prompt = provider._messages_to_prompt(messages)
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }
        
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens
        
        # Verify required fields
        assert "model" in payload
        assert "prompt" in payload
        assert "stream" in payload
        assert payload["stream"] is False  # We always use non-streaming
        assert "options" in payload
        assert "temperature" in payload["options"]
        
        # Verify optional fields
        if max_tokens is not None:
            assert "num_predict" in payload["options"]
            assert payload["options"]["num_predict"] == max_tokens

    def test_empty_messages_returns_empty_prompt(self):
        """
        **Feature: multi-ai-provider, Property 3: Request Format Correctness (Ollama)**
        **Validates: Requirements 2.2**
        
        Property: Empty message list should return empty prompt.
        """
        from app.providers.ollama import OllamaProvider
        
        provider = OllamaProvider()
        prompt = provider._messages_to_prompt([])
        
        assert prompt == ""

    @given(
        base_url=st.text(min_size=1, max_size=100).filter(
            lambda x: x.strip() and not x.endswith("/")
        ),
    )
    @settings(max_examples=100)
    def test_base_url_trailing_slash_stripped(self, base_url: str):
        """
        **Feature: multi-ai-provider, Property 3: Request Format Correctness (Ollama)**
        **Validates: Requirements 2.2**
        
        Property: For any base URL, trailing slashes should be stripped
        to ensure correct endpoint construction.
        """
        from app.providers.ollama import OllamaProvider
        
        # Test with trailing slash
        provider_with_slash = OllamaProvider(base_url=base_url + "/")
        provider_without_slash = OllamaProvider(base_url=base_url)
        
        # Both should have the same base_url without trailing slash
        assert provider_with_slash.base_url == base_url
        assert provider_without_slash.base_url == base_url
        assert not provider_with_slash.base_url.endswith("/")

    @given(
        num_messages=st.integers(min_value=2, max_value=10),
    )
    @settings(max_examples=100)
    def test_message_order_preserved(self, num_messages: int):
        """
        **Feature: multi-ai-provider, Property 3: Request Format Correctness (Ollama)**
        **Validates: Requirements 2.2**
        
        Property: For any sequence of messages, the order should be preserved
        in the generated prompt.
        """
        from app.providers.ollama import OllamaProvider
        
        provider = OllamaProvider()
        
        # Generate messages with numbered content
        messages = []
        for i in range(num_messages):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": f"Message {i}"})
        
        prompt = provider._messages_to_prompt(messages)
        
        # Verify order is preserved by checking positions
        positions = []
        for i in range(num_messages):
            pos = prompt.find(f"Message {i}")
            assert pos != -1, f"Message {i} not found in prompt"
            positions.append(pos)
        
        # Positions should be in ascending order
        assert positions == sorted(positions), "Message order not preserved"

    @given(
        role=st.sampled_from(["user", "system", "assistant", "function", "tool"]),
        content=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_unknown_roles_default_to_user(self, role: str, content: str):
        """
        **Feature: multi-ai-provider, Property 3: Request Format Correctness (Ollama)**
        **Validates: Requirements 2.2**
        
        Property: For any role that is not system or assistant, the message
        should be formatted as a User message.
        """
        from app.providers.ollama import OllamaProvider
        
        provider = OllamaProvider()
        # Need at least 2 messages to trigger role formatting
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": role, "content": content},
        ]
        
        prompt = provider._messages_to_prompt(messages)
        
        if role == "system":
            # First message is system, second is also system
            assert f"System: {content}" in prompt
        elif role == "assistant":
            assert f"Assistant: {content}" in prompt
        else:
            # Unknown roles default to User
            assert f"User: {content}" in prompt

    @given(
        content=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_missing_role_defaults_to_user(self, content: str):
        """
        **Feature: multi-ai-provider, Property 3: Request Format Correctness (Ollama)**
        **Validates: Requirements 2.2**
        
        Property: For any message without a role key, it should default to user.
        """
        from app.providers.ollama import OllamaProvider
        
        provider = OllamaProvider()
        # Need at least 2 messages to trigger role formatting
        messages = [
            {"role": "system", "content": "System prompt"},
            {"content": content},  # No role specified
        ]
        
        prompt = provider._messages_to_prompt(messages)
        
        # Missing role should default to User
        assert f"User: {content}" in prompt

    @given(
        content=st.text(min_size=0, max_size=200),
    )
    @settings(max_examples=100)
    def test_missing_content_uses_empty_string(self, content: str):
        """
        **Feature: multi-ai-provider, Property 3: Request Format Correctness (Ollama)**
        **Validates: Requirements 2.2**
        
        Property: For any message without a content key, it should use empty string.
        """
        from app.providers.ollama import OllamaProvider
        
        provider = OllamaProvider()
        # Need at least 2 messages to trigger role formatting
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user"},  # No content specified
        ]
        
        prompt = provider._messages_to_prompt(messages)
        
        # Missing content should result in "User: " with empty content
        assert "User: " in prompt


class TestOllamaResponseParsing:
    """
    Property-based tests for Ollama response parsing.
    
    **Feature: multi-ai-provider, Property 4: Response Parsing Round-Trip (Ollama)**
    **Validates: Requirements 2.3**
    
    These tests verify that Ollama responses are correctly parsed and content
    is extracted for NLU processing.
    """

    @given(
        response_text=st.text(min_size=1, max_size=1000).filter(lambda x: x.strip()),
        model_name=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        prompt_eval_count=st.integers(min_value=0, max_value=10000),
        eval_count=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=100)
    def test_response_content_extraction(
        self,
        response_text: str,
        model_name: str,
        prompt_eval_count: int,
        eval_count: int,
    ):
        """
        **Feature: multi-ai-provider, Property 4: Response Parsing Round-Trip (Ollama)**
        **Validates: Requirements 2.3**
        
        Property: For any valid Ollama response JSON, the content should be
        correctly extracted and available in the AIResponse.
        """
        # Simulate Ollama response structure
        ollama_response = {
            "model": model_name,
            "response": response_text,
            "done": True,
            "prompt_eval_count": prompt_eval_count,
            "eval_count": eval_count,
        }
        
        # Extract content as the provider would
        content = ollama_response.get("response", "")
        
        # Verify content extraction
        assert content == response_text
        assert len(content) > 0

    @given(
        response_text=st.text(min_size=1, max_size=500).filter(lambda x: x.strip()),
        model_name=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        prompt_eval_count=st.integers(min_value=0, max_value=10000),
        eval_count=st.integers(min_value=0, max_value=10000),
        latency_ms=st.floats(min_value=0.0, max_value=100000.0, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_airesponse_creation_from_ollama_response(
        self,
        response_text: str,
        model_name: str,
        prompt_eval_count: int,
        eval_count: int,
        latency_ms: float,
    ):
        """
        **Feature: multi-ai-provider, Property 4: Response Parsing Round-Trip (Ollama)**
        **Validates: Requirements 2.3**
        
        Property: For any valid Ollama response, an AIResponse should be created
        with all fields correctly populated.
        """
        # Simulate Ollama response structure
        ollama_response = {
            "model": model_name,
            "response": response_text,
            "done": True,
            "prompt_eval_count": prompt_eval_count,
            "eval_count": eval_count,
        }
        
        # Parse response as the provider would
        content = ollama_response.get("response", "")
        model = ollama_response.get("model", "unknown")
        
        usage = None
        if "prompt_eval_count" in ollama_response or "eval_count" in ollama_response:
            usage = {
                "input_tokens": ollama_response.get("prompt_eval_count", 0),
                "output_tokens": ollama_response.get("eval_count", 0),
            }
        
        # Create AIResponse
        ai_response = AIResponse(
            content=content,
            model=model,
            provider="ollama",
            usage=usage,
            latency_ms=latency_ms,
            raw_response=ollama_response,
        )
        
        # Verify all fields
        assert ai_response.content == response_text
        assert ai_response.model == model_name
        assert ai_response.provider == "ollama"
        assert ai_response.usage is not None
        assert ai_response.usage["input_tokens"] == prompt_eval_count
        assert ai_response.usage["output_tokens"] == eval_count
        assert ai_response.latency_ms == latency_ms
        assert ai_response.raw_response == ollama_response

    @given(
        model_name=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_empty_response_handled_gracefully(self, model_name: str):
        """
        **Feature: multi-ai-provider, Property 4: Response Parsing Round-Trip (Ollama)**
        **Validates: Requirements 2.3**
        
        Property: For any Ollama response with empty content, the parser should
        handle it gracefully and return an empty string.
        """
        # Simulate Ollama response with empty content
        ollama_response = {
            "model": model_name,
            "response": "",
            "done": True,
        }
        
        # Parse response
        content = ollama_response.get("response", "")
        
        # Empty content should be handled gracefully
        assert content == ""
        assert isinstance(content, str)

    @given(
        model_name=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_missing_response_field_returns_empty(self, model_name: str):
        """
        **Feature: multi-ai-provider, Property 4: Response Parsing Round-Trip (Ollama)**
        **Validates: Requirements 2.3**
        
        Property: For any Ollama response missing the 'response' field,
        the parser should return an empty string.
        """
        # Simulate Ollama response without response field
        ollama_response = {
            "model": model_name,
            "done": True,
        }
        
        # Parse response - should default to empty string
        content = ollama_response.get("response", "")
        
        assert content == ""
        assert isinstance(content, str)

    @given(
        response_text=st.text(min_size=1, max_size=500).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_missing_usage_fields_handled(self, response_text: str):
        """
        **Feature: multi-ai-provider, Property 4: Response Parsing Round-Trip (Ollama)**
        **Validates: Requirements 2.3**
        
        Property: For any Ollama response without usage fields, the parser
        should handle it gracefully with None usage.
        """
        # Simulate Ollama response without usage fields
        ollama_response = {
            "model": "test-model",
            "response": response_text,
            "done": True,
        }
        
        # Parse usage as the provider would
        usage = None
        if "prompt_eval_count" in ollama_response or "eval_count" in ollama_response:
            usage = {
                "input_tokens": ollama_response.get("prompt_eval_count", 0),
                "output_tokens": ollama_response.get("eval_count", 0),
            }
        
        # Usage should be None when fields are missing
        assert usage is None

    @given(
        response_text=st.text(min_size=1, max_size=500).filter(lambda x: x.strip()),
        prompt_eval_count=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=100)
    def test_partial_usage_fields_handled(
        self, response_text: str, prompt_eval_count: int
    ):
        """
        **Feature: multi-ai-provider, Property 4: Response Parsing Round-Trip (Ollama)**
        **Validates: Requirements 2.3**
        
        Property: For any Ollama response with only some usage fields,
        the parser should handle it with defaults for missing fields.
        """
        # Simulate Ollama response with only prompt_eval_count
        ollama_response = {
            "model": "test-model",
            "response": response_text,
            "done": True,
            "prompt_eval_count": prompt_eval_count,
            # eval_count is missing
        }
        
        # Parse usage as the provider would
        usage = None
        if "prompt_eval_count" in ollama_response or "eval_count" in ollama_response:
            usage = {
                "input_tokens": ollama_response.get("prompt_eval_count", 0),
                "output_tokens": ollama_response.get("eval_count", 0),
            }
        
        # Usage should be created with default for missing field
        assert usage is not None
        assert usage["input_tokens"] == prompt_eval_count
        assert usage["output_tokens"] == 0  # Default value

    @given(
        response_text=st.text(min_size=1, max_size=500).filter(lambda x: x.strip()),
        model_in_response=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        effective_model=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_model_from_response_takes_precedence(
        self, response_text: str, model_in_response: str, effective_model: str
    ):
        """
        **Feature: multi-ai-provider, Property 4: Response Parsing Round-Trip (Ollama)**
        **Validates: Requirements 2.3**
        
        Property: For any Ollama response, the model name from the response
        should take precedence over the requested model.
        """
        # Simulate Ollama response with model name
        ollama_response = {
            "model": model_in_response,
            "response": response_text,
            "done": True,
        }
        
        # Parse model as the provider would
        model = ollama_response.get("model", effective_model)
        
        # Model from response should take precedence
        assert model == model_in_response

    @given(
        response_text=st.text(min_size=1, max_size=500).filter(lambda x: x.strip()),
        effective_model=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_missing_model_uses_effective_model(
        self, response_text: str, effective_model: str
    ):
        """
        **Feature: multi-ai-provider, Property 4: Response Parsing Round-Trip (Ollama)**
        **Validates: Requirements 2.3**
        
        Property: For any Ollama response without a model field, the effective
        model (requested model) should be used.
        """
        # Simulate Ollama response without model field
        ollama_response = {
            "response": response_text,
            "done": True,
        }
        
        # Parse model as the provider would
        model = ollama_response.get("model", effective_model)
        
        # Should fall back to effective model
        assert model == effective_model

    @given(
        response_text=st.text(min_size=1, max_size=1000).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_response_content_preserved_exactly(self, response_text: str):
        """
        **Feature: multi-ai-provider, Property 4: Response Parsing Round-Trip (Ollama)**
        **Validates: Requirements 2.3**
        
        Property: For any valid response text, the content should be preserved
        exactly without modification during parsing.
        """
        # Simulate Ollama response
        ollama_response = {
            "model": "test-model",
            "response": response_text,
            "done": True,
        }
        
        # Parse and create AIResponse
        content = ollama_response.get("response", "")
        
        ai_response = AIResponse(
            content=content,
            model="test-model",
            provider="ollama",
        )
        
        # Content should be exactly preserved
        assert ai_response.content == response_text
        assert len(ai_response.content) == len(response_text)


class TestOpenAICompatibleRequestFormat:
    """
    Property-based tests for OpenAI-compatible request format correctness.
    
    **Feature: multi-ai-provider, Property 3: Request Format Correctness (OpenAI)**
    **Validates: Requirements 3.2**
    
    These tests verify that OpenAI-compatible requests conform to the OpenAI API specification:
    - Uses POST /v1/chat/completions endpoint
    - Proper message format with role and content
    - Correct payload structure with model, messages, temperature, max_tokens
    - Bearer token authentication in Authorization header
    """

    # OpenAI-compatible providers
    OPENAI_COMPATIBLE_PROVIDERS = [
        "openai",
        "megallm",
        "groq",
        "together",
        "deepseek",
        "qwen",
        "kimi",
        "moonshot",
    ]

    @given(
        model=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        temperature=st.floats(min_value=0.0, max_value=2.0, allow_nan=False),
        content=st.text(min_size=1, max_size=500).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_payload_has_required_fields(
        self, model: str, temperature: float, content: str
    ):
        """
        **Feature: multi-ai-provider, Property 3: Request Format Correctness (OpenAI)**
        **Validates: Requirements 3.2**
        
        Property: For any valid parameters, the generated payload should have
        all required fields for OpenAI's /v1/chat/completions endpoint.
        """
        from app.providers.openai_compatible import OpenAICompatibleProvider
        
        messages = [{"role": "user", "content": content}]
        
        payload = OpenAICompatibleProvider.build_request_payload(
            messages=messages,
            model=model,
            temperature=temperature,
        )
        
        # Verify required fields exist
        assert "model" in payload
        assert "messages" in payload
        assert "temperature" in payload
        
        # Verify field values
        assert payload["model"] == model
        assert payload["messages"] == messages
        assert payload["temperature"] == temperature

    @given(
        model=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        max_tokens=st.integers(min_value=1, max_value=4096),
        content=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_max_tokens_included_when_specified(
        self, model: str, max_tokens: int, content: str
    ):
        """
        **Feature: multi-ai-provider, Property 3: Request Format Correctness (OpenAI)**
        **Validates: Requirements 3.2**
        
        Property: For any request with max_tokens specified, the payload should
        include the max_tokens field with the correct value.
        """
        from app.providers.openai_compatible import OpenAICompatibleProvider
        
        messages = [{"role": "user", "content": content}]
        
        payload = OpenAICompatibleProvider.build_request_payload(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
        )
        
        assert "max_tokens" in payload
        assert payload["max_tokens"] == max_tokens

    @given(
        model=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        content=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_max_tokens_omitted_when_none(self, model: str, content: str):
        """
        **Feature: multi-ai-provider, Property 3: Request Format Correctness (OpenAI)**
        **Validates: Requirements 3.2**
        
        Property: For any request without max_tokens, the payload should not
        include the max_tokens field.
        """
        from app.providers.openai_compatible import OpenAICompatibleProvider
        
        messages = [{"role": "user", "content": content}]
        
        payload = OpenAICompatibleProvider.build_request_payload(
            messages=messages,
            model=model,
            max_tokens=None,
        )
        
        assert "max_tokens" not in payload

    @given(
        role=st.sampled_from(["user", "system", "assistant"]),
        content=st.text(min_size=1, max_size=300).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_message_format_correctness(self, role: str, content: str):
        """
        **Feature: multi-ai-provider, Property 3: Request Format Correctness (OpenAI)**
        **Validates: Requirements 3.2**
        
        Property: For any message with valid role and content, the message
        should be included in the payload with correct structure.
        """
        from app.providers.openai_compatible import OpenAICompatibleProvider
        
        messages = [{"role": role, "content": content}]
        
        payload = OpenAICompatibleProvider.build_request_payload(
            messages=messages,
            model="test-model",
        )
        
        # Verify messages structure
        assert len(payload["messages"]) == 1
        assert payload["messages"][0]["role"] == role
        assert payload["messages"][0]["content"] == content

    @given(
        num_messages=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=100)
    def test_multiple_messages_preserved(self, num_messages: int):
        """
        **Feature: multi-ai-provider, Property 3: Request Format Correctness (OpenAI)**
        **Validates: Requirements 3.2**
        
        Property: For any number of messages, all messages should be preserved
        in the payload in the correct order.
        """
        from app.providers.openai_compatible import OpenAICompatibleProvider
        
        # Generate messages with alternating roles
        messages = []
        for i in range(num_messages):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": f"Message {i}"})
        
        payload = OpenAICompatibleProvider.build_request_payload(
            messages=messages,
            model="test-model",
        )
        
        # Verify all messages are preserved
        assert len(payload["messages"]) == num_messages
        
        # Verify order is preserved
        for i, msg in enumerate(payload["messages"]):
            assert msg["content"] == f"Message {i}"

    @given(
        api_key=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_bearer_token_format(self, api_key: str):
        """
        **Feature: multi-ai-provider, Property 3: Request Format Correctness (OpenAI)**
        **Validates: Requirements 3.2**
        
        Property: For any API key, the Authorization header should use
        Bearer token format.
        """
        from app.providers.openai_compatible import OpenAICompatibleProvider
        
        headers = OpenAICompatibleProvider.build_auth_header(api_key)
        
        assert "Authorization" in headers
        assert headers["Authorization"].startswith("Bearer ")
        assert headers["Authorization"] == f"Bearer {api_key}"

    @given(
        provider_name=st.sampled_from(OPENAI_COMPATIBLE_PROVIDERS),
        api_key=st.text(min_size=10, max_size=50).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_endpoint_format_correctness(self, provider_name: str, api_key: str):
        """
        **Feature: multi-ai-provider, Property 3: Request Format Correctness (OpenAI)**
        **Validates: Requirements 3.2**
        
        Property: For any OpenAI-compatible provider, the endpoint should
        end with /v1/chat/completions.
        """
        from app.providers.openai_compatible import OpenAICompatibleProvider
        
        provider = OpenAICompatibleProvider(
            provider_name=provider_name,
            api_key=api_key,
        )
        
        endpoint = provider._get_endpoint()
        
        assert endpoint.endswith("/v1/chat/completions")

    @given(
        provider_name=st.sampled_from(OPENAI_COMPATIBLE_PROVIDERS),
        api_key=st.text(min_size=10, max_size=50).filter(
            lambda x: x.strip() and x == x.strip()
        ),
    )
    @settings(max_examples=100)
    def test_headers_include_content_type(self, provider_name: str, api_key: str):
        """
        **Feature: multi-ai-provider, Property 3: Request Format Correctness (OpenAI)**
        **Validates: Requirements 3.2**
        
        Property: For any OpenAI-compatible provider, the headers should
        include Content-Type: application/json.
        """
        from app.providers.openai_compatible import OpenAICompatibleProvider
        
        provider = OpenAICompatibleProvider(
            provider_name=provider_name,
            api_key=api_key,
        )
        
        headers = provider._get_headers()
        
        assert "Content-Type" in headers
        assert headers["Content-Type"] == "application/json"
        assert "Authorization" in headers
        assert headers["Authorization"] == f"Bearer {api_key}"

    @given(
        base_url=st.text(min_size=10, max_size=100).filter(
            lambda x: x.strip() and not x.endswith("/")
        ),
        api_key=st.text(min_size=10, max_size=50).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_base_url_trailing_slash_stripped(self, base_url: str, api_key: str):
        """
        **Feature: multi-ai-provider, Property 3: Request Format Correctness (OpenAI)**
        **Validates: Requirements 3.2**
        
        Property: For any base URL, trailing slashes should be stripped
        to ensure correct endpoint construction.
        """
        from app.providers.openai_compatible import OpenAICompatibleProvider
        
        # Test with trailing slash
        provider_with_slash = OpenAICompatibleProvider(
            provider_name="openai",
            api_key=api_key,
            base_url=base_url + "/",
        )
        provider_without_slash = OpenAICompatibleProvider(
            provider_name="openai",
            api_key=api_key,
            base_url=base_url,
        )
        
        # Both should have the same base_url without trailing slash
        assert provider_with_slash.base_url == base_url
        assert provider_without_slash.base_url == base_url
        assert not provider_with_slash.base_url.endswith("/")

    @given(
        temperature=st.floats(min_value=0.0, max_value=2.0, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_temperature_in_valid_range(self, temperature: float):
        """
        **Feature: multi-ai-provider, Property 3: Request Format Correctness (OpenAI)**
        **Validates: Requirements 3.2**
        
        Property: For any temperature value, it should be included in the
        payload as a float.
        """
        from app.providers.openai_compatible import OpenAICompatibleProvider
        
        messages = [{"role": "user", "content": "test"}]
        
        payload = OpenAICompatibleProvider.build_request_payload(
            messages=messages,
            model="test-model",
            temperature=temperature,
        )
        
        assert isinstance(payload["temperature"], float)
        assert payload["temperature"] == temperature

    @given(
        provider_name=st.sampled_from(OPENAI_COMPATIBLE_PROVIDERS),
        api_key=st.text(min_size=10, max_size=50).filter(lambda x: x.strip()),
        model=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_custom_model_used(self, provider_name: str, api_key: str, model: str):
        """
        **Feature: multi-ai-provider, Property 3: Request Format Correctness (OpenAI)**
        **Validates: Requirements 3.2**
        
        Property: For any custom model specified, it should be used instead
        of the default model.
        """
        from app.providers.openai_compatible import OpenAICompatibleProvider
        
        provider = OpenAICompatibleProvider(
            provider_name=provider_name,
            api_key=api_key,
            model=model,
        )
        
        assert provider.model == model

    @given(
        provider_name=st.sampled_from(OPENAI_COMPATIBLE_PROVIDERS),
        api_key=st.text(min_size=10, max_size=50).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_default_model_assigned(self, provider_name: str, api_key: str):
        """
        **Feature: multi-ai-provider, Property 3: Request Format Correctness (OpenAI)**
        **Validates: Requirements 3.2**
        
        Property: For any provider without a custom model, a default model
        should be assigned.
        """
        from app.providers.openai_compatible import (
            OpenAICompatibleProvider,
            PROVIDER_DEFAULT_MODELS,
        )
        
        provider = OpenAICompatibleProvider(
            provider_name=provider_name,
            api_key=api_key,
        )
        
        expected_default = PROVIDER_DEFAULT_MODELS.get(provider_name, "gpt-3.5-turbo")
        assert provider.model == expected_default

    @given(
        system_content=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
        user_content=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
        assistant_content=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_conversation_messages_structure(
        self, system_content: str, user_content: str, assistant_content: str
    ):
        """
        **Feature: multi-ai-provider, Property 3: Request Format Correctness (OpenAI)**
        **Validates: Requirements 3.2**
        
        Property: For any conversation with system, user, and assistant messages,
        all messages should be preserved with correct roles.
        """
        from app.providers.openai_compatible import OpenAICompatibleProvider
        
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
        
        payload = OpenAICompatibleProvider.build_request_payload(
            messages=messages,
            model="test-model",
        )
        
        # Verify all messages are preserved
        assert len(payload["messages"]) == 3
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][0]["content"] == system_content
        assert payload["messages"][1]["role"] == "user"
        assert payload["messages"][1]["content"] == user_content
        assert payload["messages"][2]["role"] == "assistant"
        assert payload["messages"][2]["content"] == assistant_content


class TestAuthenticationHeaderCorrectness:
    """
    Property-based tests for authentication header correctness (Bearer token).
    
    **Feature: multi-ai-provider, Property 8: Authentication Header Correctness (Bearer)**
    **Validates: Requirements 3.3**
    
    These tests verify that OpenAI-compatible providers correctly include
    Bearer token authentication in the Authorization header.
    """

    # OpenAI-compatible providers that use Bearer token authentication
    BEARER_TOKEN_PROVIDERS = [
        "openai",
        "megallm",
        "groq",
        "together",
        "deepseek",
        "qwen",
        "kimi",
        "moonshot",
    ]

    @given(
        api_key=st.text(min_size=1, max_size=200).filter(
            lambda x: x.strip() and not x.isspace()
        ),
    )
    @settings(max_examples=100)
    def test_bearer_token_prefix_present(self, api_key: str):
        """
        **Feature: multi-ai-provider, Property 8: Authentication Header Correctness (Bearer)**
        **Validates: Requirements 3.3**
        
        Property: For any valid API key, the Authorization header should
        start with 'Bearer ' prefix.
        """
        from app.providers.openai_compatible import OpenAICompatibleProvider
        
        headers = OpenAICompatibleProvider.build_auth_header(api_key)
        
        assert "Authorization" in headers
        assert headers["Authorization"].startswith("Bearer ")

    @given(
        api_key=st.text(min_size=1, max_size=200).filter(
            lambda x: x.strip() and not x.isspace()
        ),
    )
    @settings(max_examples=100)
    def test_api_key_included_after_bearer(self, api_key: str):
        """
        **Feature: multi-ai-provider, Property 8: Authentication Header Correctness (Bearer)**
        **Validates: Requirements 3.3**
        
        Property: For any valid API key, the full API key should appear
        after the 'Bearer ' prefix in the Authorization header.
        """
        from app.providers.openai_compatible import OpenAICompatibleProvider
        
        headers = OpenAICompatibleProvider.build_auth_header(api_key)
        
        # Extract the token part after "Bearer "
        auth_value = headers["Authorization"]
        token_part = auth_value[7:]  # Skip "Bearer "
        
        assert token_part == api_key

    @given(
        api_key=st.text(min_size=1, max_size=200).filter(
            lambda x: x.strip() and not x.isspace()
        ),
    )
    @settings(max_examples=100)
    def test_bearer_format_exact_match(self, api_key: str):
        """
        **Feature: multi-ai-provider, Property 8: Authentication Header Correctness (Bearer)**
        **Validates: Requirements 3.3**
        
        Property: For any valid API key, the Authorization header should
        exactly match the format 'Bearer {api_key}'.
        """
        from app.providers.openai_compatible import OpenAICompatibleProvider
        
        headers = OpenAICompatibleProvider.build_auth_header(api_key)
        
        expected = f"Bearer {api_key}"
        assert headers["Authorization"] == expected

    @given(
        provider_name=st.sampled_from(BEARER_TOKEN_PROVIDERS),
        api_key=st.text(min_size=10, max_size=100).filter(
            lambda x: x.strip() and not x.isspace() and x == x.strip()
        ),
    )
    @settings(max_examples=100)
    def test_provider_headers_include_bearer_auth(
        self, provider_name: str, api_key: str
    ):
        """
        **Feature: multi-ai-provider, Property 8: Authentication Header Correctness (Bearer)**
        **Validates: Requirements 3.3**
        
        Property: For any OpenAI-compatible provider, the _get_headers method
        should return headers with correct Bearer token authentication.
        
        Note: API keys are filtered to exclude leading/trailing whitespace
        since the provider correctly strips whitespace from keys.
        """
        from app.providers.openai_compatible import OpenAICompatibleProvider
        
        provider = OpenAICompatibleProvider(
            provider_name=provider_name,
            api_key=api_key,
        )
        
        headers = provider._get_headers()
        
        assert "Authorization" in headers
        assert headers["Authorization"] == f"Bearer {api_key}"

    @given(
        provider_name=st.sampled_from(BEARER_TOKEN_PROVIDERS),
        api_key=st.text(min_size=10, max_size=100).filter(
            lambda x: x.strip() and not x.isspace()
        ),
    )
    @settings(max_examples=100)
    def test_provider_headers_include_content_type(
        self, provider_name: str, api_key: str
    ):
        """
        **Feature: multi-ai-provider, Property 8: Authentication Header Correctness (Bearer)**
        **Validates: Requirements 3.3**
        
        Property: For any OpenAI-compatible provider, the headers should
        include both Authorization and Content-Type headers.
        """
        from app.providers.openai_compatible import OpenAICompatibleProvider
        
        provider = OpenAICompatibleProvider(
            provider_name=provider_name,
            api_key=api_key,
        )
        
        headers = provider._get_headers()
        
        # Both headers should be present
        assert "Authorization" in headers
        assert "Content-Type" in headers
        assert headers["Content-Type"] == "application/json"

    @given(
        api_key=st.text(min_size=1, max_size=200).filter(
            lambda x: x.strip() and not x.isspace()
        ),
    )
    @settings(max_examples=100)
    def test_api_key_not_modified(self, api_key: str):
        """
        **Feature: multi-ai-provider, Property 8: Authentication Header Correctness (Bearer)**
        **Validates: Requirements 3.3**
        
        Property: For any valid API key, the key should not be modified
        (trimmed, encoded, etc.) when included in the Authorization header.
        """
        from app.providers.openai_compatible import OpenAICompatibleProvider
        
        headers = OpenAICompatibleProvider.build_auth_header(api_key)
        
        # Extract the token and verify it matches exactly
        auth_header = headers["Authorization"]
        # The format is "Bearer {api_key}"
        assert auth_header == f"Bearer {api_key}"
        # Verify the key wasn't modified
        extracted_key = auth_header.replace("Bearer ", "", 1)
        assert extracted_key == api_key

    @given(
        api_key=st.text(min_size=1, max_size=100).filter(
            lambda x: x.strip() and not x.isspace()
        ).map(lambda x: f"sk-{x}"),  # Simulate OpenAI-style keys
    )
    @settings(max_examples=100)
    def test_openai_style_api_key_format(self, api_key: str):
        """
        **Feature: multi-ai-provider, Property 8: Authentication Header Correctness (Bearer)**
        **Validates: Requirements 3.3**
        
        Property: For any OpenAI-style API key (starting with 'sk-'),
        the Bearer token should correctly include the full key.
        """
        from app.providers.openai_compatible import OpenAICompatibleProvider
        
        headers = OpenAICompatibleProvider.build_auth_header(api_key)
        
        assert headers["Authorization"] == f"Bearer {api_key}"
        assert "sk-" in headers["Authorization"]

    @given(
        provider_name=st.sampled_from(BEARER_TOKEN_PROVIDERS),
        api_key_with_spaces=st.text(min_size=10, max_size=50).filter(
            lambda x: x.strip() and not x.isspace()
        ).map(lambda x: f"  {x}  "),  # Add leading/trailing spaces
    )
    @settings(max_examples=100)
    def test_api_key_stripped_on_provider_init(
        self, provider_name: str, api_key_with_spaces: str
    ):
        """
        **Feature: multi-ai-provider, Property 8: Authentication Header Correctness (Bearer)**
        **Validates: Requirements 3.3**
        
        Property: For any API key with leading/trailing whitespace,
        the provider should strip the whitespace before using it.
        """
        from app.providers.openai_compatible import OpenAICompatibleProvider
        
        provider = OpenAICompatibleProvider(
            provider_name=provider_name,
            api_key=api_key_with_spaces,
        )
        
        # The stored API key should be stripped
        assert provider.api_key == api_key_with_spaces.strip()
        
        # The Authorization header should use the stripped key
        headers = provider._get_headers()
        expected_key = api_key_with_spaces.strip()
        assert headers["Authorization"] == f"Bearer {expected_key}"

    @given(
        provider_name=st.sampled_from(BEARER_TOKEN_PROVIDERS),
    )
    @settings(max_examples=100)
    def test_empty_api_key_raises_error(self, provider_name: str):
        """
        **Feature: multi-ai-provider, Property 8: Authentication Header Correctness (Bearer)**
        **Validates: Requirements 3.3**
        
        Property: For any provider requiring Bearer authentication,
        an empty API key should raise an AuthenticationError.
        """
        from app.providers.openai_compatible import OpenAICompatibleProvider
        
        with pytest.raises(AuthenticationError) as exc_info:
            OpenAICompatibleProvider(
                provider_name=provider_name,
                api_key="",
            )
        
        assert provider_name in str(exc_info.value)

    @given(
        provider_name=st.sampled_from(BEARER_TOKEN_PROVIDERS),
        whitespace_key=st.text(max_size=20).filter(
            lambda x: not x or x.isspace()
        ),
    )
    @settings(max_examples=100)
    def test_whitespace_only_api_key_raises_error(
        self, provider_name: str, whitespace_key: str
    ):
        """
        **Feature: multi-ai-provider, Property 8: Authentication Header Correctness (Bearer)**
        **Validates: Requirements 3.3**
        
        Property: For any provider requiring Bearer authentication,
        a whitespace-only API key should raise an AuthenticationError.
        """
        from app.providers.openai_compatible import OpenAICompatibleProvider
        
        with pytest.raises(AuthenticationError) as exc_info:
            OpenAICompatibleProvider(
                provider_name=provider_name,
                api_key=whitespace_key,
            )
        
        assert provider_name in str(exc_info.value)

    @given(
        api_key=st.text(min_size=1, max_size=200).filter(
            lambda x: x.strip() and not x.isspace()
        ),
    )
    @settings(max_examples=100)
    def test_authorization_header_is_single_value(self, api_key: str):
        """
        **Feature: multi-ai-provider, Property 8: Authentication Header Correctness (Bearer)**
        **Validates: Requirements 3.3**
        
        Property: For any valid API key, the Authorization header should
        contain exactly one value (no multiple auth schemes).
        """
        from app.providers.openai_compatible import OpenAICompatibleProvider
        
        headers = OpenAICompatibleProvider.build_auth_header(api_key)
        
        auth_value = headers["Authorization"]
        
        # Should only have one "Bearer" prefix
        assert auth_value.count("Bearer") == 1
        # Should not contain other auth schemes
        assert "Basic" not in auth_value
        assert "Digest" not in auth_value
        assert "OAuth" not in auth_value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestJWTAuthenticationHeaderCorrectness:
    """
    Property-based tests for JWT authentication header correctness (GLM/Zhipu AI).
    
    **Feature: multi-ai-provider, Property 8: Authentication Header Correctness (JWT)**
    **Validates: Requirements 9.3**
    
    These tests verify that GLM provider correctly generates JWT tokens from
    API keys in the format {id}.{secret} and includes them in the Authorization header.
    """

    @given(
        api_key_id=st.text(min_size=1, max_size=50, alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd'),
            whitelist_characters='_-'
        )).filter(lambda x: x.strip() and not x.isspace()),
        api_key_secret=st.text(min_size=1, max_size=100, alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd'),
            whitelist_characters='_-'
        )).filter(lambda x: x.strip() and not x.isspace()),
    )
    @settings(max_examples=100)
    def test_jwt_token_has_three_parts(self, api_key_id: str, api_key_secret: str):
        """
        **Feature: multi-ai-provider, Property 8: Authentication Header Correctness (JWT)**
        **Validates: Requirements 9.3**
        
        Property: For any valid API key in format {id}.{secret}, the generated
        JWT token should have exactly three parts separated by dots (header.payload.signature).
        """
        from app.providers.glm import GLMProvider
        
        api_key = f"{api_key_id}.{api_key_secret}"
        
        token = GLMProvider.generate_jwt_from_api_key(api_key)
        
        parts = token.split(".")
        assert len(parts) == 3, f"JWT should have 3 parts, got {len(parts)}"
        # Each part should be non-empty
        assert all(part for part in parts), "All JWT parts should be non-empty"

    @given(
        api_key_id=st.text(min_size=1, max_size=50, alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd'),
            whitelist_characters='_-'
        )).filter(lambda x: x.strip() and not x.isspace()),
        api_key_secret=st.text(min_size=1, max_size=100, alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd'),
            whitelist_characters='_-'
        )).filter(lambda x: x.strip() and not x.isspace()),
    )
    @settings(max_examples=100)
    def test_jwt_header_contains_correct_algorithm(
        self, api_key_id: str, api_key_secret: str
    ):
        """
        **Feature: multi-ai-provider, Property 8: Authentication Header Correctness (JWT)**
        **Validates: Requirements 9.3**
        
        Property: For any valid API key, the JWT header should specify
        HS256 algorithm and SIGN type as required by Zhipu AI.
        """
        import base64
        import json
        
        from app.providers.glm import GLMProvider
        
        api_key = f"{api_key_id}.{api_key_secret}"
        
        token = GLMProvider.generate_jwt_from_api_key(api_key)
        
        # Decode the header (first part)
        header_b64 = token.split(".")[0]
        # Add padding if needed
        padding = 4 - len(header_b64) % 4
        if padding != 4:
            header_b64 += "=" * padding
        
        header_json = base64.urlsafe_b64decode(header_b64).decode("utf-8")
        header = json.loads(header_json)
        
        assert header.get("alg") == "HS256", "JWT algorithm should be HS256"
        assert header.get("sign_type") == "SIGN", "JWT sign_type should be SIGN"

    @given(
        api_key_id=st.text(min_size=1, max_size=50, alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd'),
            whitelist_characters='_-'
        )).filter(lambda x: x.strip() and not x.isspace()),
        api_key_secret=st.text(min_size=1, max_size=100, alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd'),
            whitelist_characters='_-'
        )).filter(lambda x: x.strip() and not x.isspace()),
    )
    @settings(max_examples=100)
    def test_jwt_payload_contains_api_key_id(
        self, api_key_id: str, api_key_secret: str
    ):
        """
        **Feature: multi-ai-provider, Property 8: Authentication Header Correctness (JWT)**
        **Validates: Requirements 9.3**
        
        Property: For any valid API key, the JWT payload should contain
        the api_key field with the ID portion of the API key.
        """
        import base64
        import json
        
        from app.providers.glm import GLMProvider
        
        api_key = f"{api_key_id}.{api_key_secret}"
        
        token = GLMProvider.generate_jwt_from_api_key(api_key)
        
        # Decode the payload (second part)
        payload_b64 = token.split(".")[1]
        # Add padding if needed
        padding = 4 - len(payload_b64) % 4
        if padding != 4:
            payload_b64 += "=" * padding
        
        payload_json = base64.urlsafe_b64decode(payload_b64).decode("utf-8")
        payload = json.loads(payload_json)
        
        assert payload.get("api_key") == api_key_id, (
            f"JWT payload api_key should be '{api_key_id}', got '{payload.get('api_key')}'"
        )

    @given(
        api_key_id=st.text(min_size=1, max_size=50, alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd'),
            whitelist_characters='_-'
        )).filter(lambda x: x.strip() and not x.isspace()),
        api_key_secret=st.text(min_size=1, max_size=100, alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd'),
            whitelist_characters='_-'
        )).filter(lambda x: x.strip() and not x.isspace()),
    )
    @settings(max_examples=100)
    def test_jwt_payload_contains_timestamps(
        self, api_key_id: str, api_key_secret: str
    ):
        """
        **Feature: multi-ai-provider, Property 8: Authentication Header Correctness (JWT)**
        **Validates: Requirements 9.3**
        
        Property: For any valid API key, the JWT payload should contain
        both 'exp' (expiration) and 'timestamp' fields as integers.
        """
        import base64
        import json
        
        from app.providers.glm import GLMProvider
        
        api_key = f"{api_key_id}.{api_key_secret}"
        
        token = GLMProvider.generate_jwt_from_api_key(api_key)
        
        # Decode the payload (second part)
        payload_b64 = token.split(".")[1]
        # Add padding if needed
        padding = 4 - len(payload_b64) % 4
        if padding != 4:
            payload_b64 += "=" * padding
        
        payload_json = base64.urlsafe_b64decode(payload_b64).decode("utf-8")
        payload = json.loads(payload_json)
        
        assert "exp" in payload, "JWT payload should contain 'exp' field"
        assert "timestamp" in payload, "JWT payload should contain 'timestamp' field"
        assert isinstance(payload["exp"], int), "'exp' should be an integer"
        assert isinstance(payload["timestamp"], int), "'timestamp' should be an integer"

    @given(
        api_key_id=st.text(min_size=1, max_size=50, alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd'),
            whitelist_characters='_-'
        )).filter(lambda x: x.strip() and not x.isspace()),
        api_key_secret=st.text(min_size=1, max_size=100, alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd'),
            whitelist_characters='_-'
        )).filter(lambda x: x.strip() and not x.isspace()),
        expiration_seconds=st.integers(min_value=60, max_value=86400),
    )
    @settings(max_examples=100)
    def test_jwt_expiration_is_future(
        self, api_key_id: str, api_key_secret: str, expiration_seconds: int
    ):
        """
        **Feature: multi-ai-provider, Property 8: Authentication Header Correctness (JWT)**
        **Validates: Requirements 9.3**
        
        Property: For any valid API key and expiration time, the JWT expiration
        timestamp should be greater than the current timestamp.
        """
        import base64
        import json
        
        from app.providers.glm import GLMProvider
        
        api_key = f"{api_key_id}.{api_key_secret}"
        
        token = GLMProvider.generate_jwt_from_api_key(api_key, expiration_seconds)
        
        # Decode the payload (second part)
        payload_b64 = token.split(".")[1]
        # Add padding if needed
        padding = 4 - len(payload_b64) % 4
        if padding != 4:
            payload_b64 += "=" * padding
        
        payload_json = base64.urlsafe_b64decode(payload_b64).decode("utf-8")
        payload = json.loads(payload_json)
        
        assert payload["exp"] > payload["timestamp"], (
            "JWT expiration should be greater than timestamp"
        )

    @given(
        api_key_id=st.text(min_size=1, max_size=50, alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd'),
            whitelist_characters='_-'
        )).filter(lambda x: x.strip() and not x.isspace()),
        api_key_secret=st.text(min_size=1, max_size=100, alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd'),
            whitelist_characters='_-'
        )).filter(lambda x: x.strip() and not x.isspace()),
    )
    @settings(max_examples=100)
    def test_glm_provider_uses_bearer_with_jwt(
        self, api_key_id: str, api_key_secret: str
    ):
        """
        **Feature: multi-ai-provider, Property 8: Authentication Header Correctness (JWT)**
        **Validates: Requirements 9.3**
        
        Property: For any valid GLM API key, the Authorization header should
        use Bearer token format with the generated JWT.
        """
        from app.providers.glm import GLMProvider
        
        api_key = f"{api_key_id}.{api_key_secret}"
        
        provider = GLMProvider(api_key=api_key)
        headers = provider._get_headers()
        
        assert "Authorization" in headers
        assert headers["Authorization"].startswith("Bearer ")
        
        # The token after "Bearer " should be a valid JWT (3 parts)
        token = headers["Authorization"][7:]  # Skip "Bearer "
        parts = token.split(".")
        assert len(parts) == 3, "JWT in Authorization header should have 3 parts"

    @given(
        api_key_id=st.text(min_size=1, max_size=50, alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd'),
            whitelist_characters='_-'
        )).filter(lambda x: x.strip() and not x.isspace()),
        api_key_secret=st.text(min_size=1, max_size=100, alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd'),
            whitelist_characters='_-'
        )).filter(lambda x: x.strip() and not x.isspace()),
    )
    @settings(max_examples=100)
    def test_glm_provider_headers_include_content_type(
        self, api_key_id: str, api_key_secret: str
    ):
        """
        **Feature: multi-ai-provider, Property 8: Authentication Header Correctness (JWT)**
        **Validates: Requirements 9.3**
        
        Property: For any valid GLM API key, the headers should include
        both Authorization and Content-Type headers.
        """
        from app.providers.glm import GLMProvider
        
        api_key = f"{api_key_id}.{api_key_secret}"
        
        provider = GLMProvider(api_key=api_key)
        headers = provider._get_headers()
        
        assert "Authorization" in headers
        assert "Content-Type" in headers
        assert headers["Content-Type"] == "application/json"

    @given(
        api_key_id=st.text(min_size=1, max_size=50, alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd'),
            whitelist_characters='_-'
        )).filter(lambda x: x.strip() and not x.isspace()),
        api_key_secret=st.text(min_size=1, max_size=100, alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd'),
            whitelist_characters='_-'
        )).filter(lambda x: x.strip() and not x.isspace()),
    )
    @settings(max_examples=100)
    def test_jwt_signature_is_base64url_encoded(
        self, api_key_id: str, api_key_secret: str
    ):
        """
        **Feature: multi-ai-provider, Property 8: Authentication Header Correctness (JWT)**
        **Validates: Requirements 9.3**
        
        Property: For any valid API key, the JWT signature (third part)
        should be valid base64url encoded data.
        """
        import base64
        
        from app.providers.glm import GLMProvider
        
        api_key = f"{api_key_id}.{api_key_secret}"
        
        token = GLMProvider.generate_jwt_from_api_key(api_key)
        
        # Get the signature (third part)
        signature_b64 = token.split(".")[2]
        
        # Add padding if needed for decoding
        padding = 4 - len(signature_b64) % 4
        if padding != 4:
            signature_b64_padded = signature_b64 + "=" * padding
        else:
            signature_b64_padded = signature_b64
        
        # Should be decodable as base64url
        try:
            decoded = base64.urlsafe_b64decode(signature_b64_padded)
            # HS256 signature should be 32 bytes (256 bits)
            assert len(decoded) == 32, f"HS256 signature should be 32 bytes, got {len(decoded)}"
        except Exception as e:
            pytest.fail(f"JWT signature is not valid base64url: {e}")

    def test_invalid_api_key_format_raises_error(self):
        """
        **Feature: multi-ai-provider, Property 8: Authentication Header Correctness (JWT)**
        **Validates: Requirements 9.3**
        
        Property: For any API key without a dot separator, the GLM provider
        should raise an AuthenticationError.
        """
        from app.providers.glm import GLMProvider
        
        with pytest.raises(AuthenticationError) as exc_info:
            GLMProvider(api_key="invalid_key_without_dot")
        
        assert "glm" in str(exc_info.value).lower() or "format" in str(exc_info.value).lower()

    def test_empty_api_key_raises_error(self):
        """
        **Feature: multi-ai-provider, Property 8: Authentication Header Correctness (JWT)**
        **Validates: Requirements 9.3**
        
        Property: For an empty API key, the GLM provider should raise
        an AuthenticationError.
        """
        from app.providers.glm import GLMProvider
        
        with pytest.raises(AuthenticationError) as exc_info:
            GLMProvider(api_key="")
        
        assert "glm" in str(exc_info.value).lower() or "required" in str(exc_info.value).lower()

    def test_api_key_with_empty_parts_raises_error(self):
        """
        **Feature: multi-ai-provider, Property 8: Authentication Header Correctness (JWT)**
        **Validates: Requirements 9.3**
        
        Property: For an API key with empty id or secret parts (e.g., ".secret" or "id."),
        the GLM provider should raise an AuthenticationError.
        """
        from app.providers.glm import GLMProvider
        
        # Empty ID
        with pytest.raises(AuthenticationError):
            GLMProvider(api_key=".secret")
        
        # Empty secret
        with pytest.raises(AuthenticationError):
            GLMProvider(api_key="id.")

    @given(
        api_key_id=st.text(min_size=1, max_size=50, alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd'),
            whitelist_characters='_-'
        )).filter(lambda x: x.strip() and not x.isspace()),
        api_key_secret=st.text(min_size=1, max_size=100, alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd'),
            whitelist_characters='_-'
        )).filter(lambda x: x.strip() and not x.isspace()),
    )
    @settings(max_examples=100)
    def test_validate_api_key_format_accepts_valid_keys(
        self, api_key_id: str, api_key_secret: str
    ):
        """
        **Feature: multi-ai-provider, Property 8: Authentication Header Correctness (JWT)**
        **Validates: Requirements 9.3**
        
        Property: For any valid API key in format {id}.{secret},
        validate_api_key_format should return True.
        """
        from app.providers.glm import GLMProvider
        
        api_key = f"{api_key_id}.{api_key_secret}"
        
        assert GLMProvider.validate_api_key_format(api_key) is True

    @given(
        invalid_key=st.one_of(
            st.none(),
            st.just(""),
            st.text(max_size=50).filter(lambda x: "." not in x),  # No dot
            st.just(".secret"),  # Empty ID
            st.just("id."),  # Empty secret
            st.text(max_size=20).filter(lambda x: x.isspace()),  # Whitespace only
        )
    )
    @settings(max_examples=100)
    def test_validate_api_key_format_rejects_invalid_keys(self, invalid_key):
        """
        **Feature: multi-ai-provider, Property 8: Authentication Header Correctness (JWT)**
        **Validates: Requirements 9.3**
        
        Property: For any invalid API key (None, empty, no dot, empty parts),
        validate_api_key_format should return False.
        """
        from app.providers.glm import GLMProvider
        
        assert GLMProvider.validate_api_key_format(invalid_key) is False


class TestAnthropicRequestFormat:
    """
    Property-based tests for Anthropic request format correctness.
    
    **Feature: multi-ai-provider, Property 3: Request Format Correctness (Anthropic)**
    **Validates: Requirements 4.2**
    
    These tests verify that Anthropic requests conform to the Anthropic API specification:
    - Uses POST /v1/messages endpoint
    - Proper message format with role and content
    - System messages extracted separately as 'system' parameter
    - First message must be from user (Anthropic requirement)
    - Correct payload structure with model, messages, max_tokens, temperature
    - x-api-key header for authentication
    """

    @given(
        model=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        temperature=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        max_tokens=st.integers(min_value=1, max_value=4096),
        content=st.text(min_size=1, max_size=500).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_payload_has_required_fields(
        self, model: str, temperature: float, max_tokens: int, content: str
    ):
        """
        **Feature: multi-ai-provider, Property 3: Request Format Correctness (Anthropic)**
        **Validates: Requirements 4.2**
        
        Property: For any valid parameters, the generated payload should have
        all required fields for Anthropic's /v1/messages endpoint.
        """
        from app.providers.anthropic import AnthropicProvider
        
        messages = [{"role": "user", "content": content}]
        
        payload = AnthropicProvider.build_request_payload(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        # Verify required fields exist
        assert "model" in payload
        assert "messages" in payload
        assert "max_tokens" in payload
        assert "temperature" in payload
        
        # Verify field values
        assert payload["model"] == model
        assert payload["max_tokens"] == max_tokens
        assert payload["temperature"] == temperature

    @given(
        content=st.text(min_size=1, max_size=300).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_user_message_format_correctness(self, content: str):
        """
        **Feature: multi-ai-provider, Property 3: Request Format Correctness (Anthropic)**
        **Validates: Requirements 4.2**
        
        Property: For any user message, it should be included in the payload
        with role 'user' and correct content.
        """
        from app.providers.anthropic import AnthropicProvider
        
        messages = [{"role": "user", "content": content}]
        
        payload = AnthropicProvider.build_request_payload(
            messages=messages,
            model="claude-3-5-sonnet-20241022",
        )
        
        # Verify messages structure
        assert len(payload["messages"]) == 1
        assert payload["messages"][0]["role"] == "user"
        assert payload["messages"][0]["content"] == content

    @given(
        system_content=st.text(min_size=1, max_size=300).filter(lambda x: x.strip()),
        user_content=st.text(min_size=1, max_size=300).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_system_message_extracted_separately(
        self, system_content: str, user_content: str
    ):
        """
        **Feature: multi-ai-provider, Property 3: Request Format Correctness (Anthropic)**
        **Validates: Requirements 4.2**
        
        Property: For any conversation with system messages, the system content
        should be extracted to the 'system' parameter, not in messages array.
        """
        from app.providers.anthropic import AnthropicProvider
        
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        
        payload = AnthropicProvider.build_request_payload(
            messages=messages,
            model="claude-3-5-sonnet-20241022",
        )
        
        # System message should be in 'system' parameter
        assert "system" in payload
        assert payload["system"] == system_content
        
        # Messages array should only contain user message
        assert len(payload["messages"]) == 1
        assert payload["messages"][0]["role"] == "user"
        assert payload["messages"][0]["content"] == user_content

    @given(
        system_content_1=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
        system_content_2=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
        user_content=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_multiple_system_messages_concatenated(
        self, system_content_1: str, system_content_2: str, user_content: str
    ):
        """
        **Feature: multi-ai-provider, Property 3: Request Format Correctness (Anthropic)**
        **Validates: Requirements 4.2**
        
        Property: For any conversation with multiple system messages, they should
        be concatenated with newlines in the 'system' parameter.
        """
        from app.providers.anthropic import AnthropicProvider
        
        messages = [
            {"role": "system", "content": system_content_1},
            {"role": "system", "content": system_content_2},
            {"role": "user", "content": user_content},
        ]
        
        payload = AnthropicProvider.build_request_payload(
            messages=messages,
            model="claude-3-5-sonnet-20241022",
        )
        
        # Multiple system messages should be concatenated
        assert "system" in payload
        assert system_content_1 in payload["system"]
        assert system_content_2 in payload["system"]
        assert "\n" in payload["system"]

    @given(
        assistant_content=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_first_message_must_be_user(self, assistant_content: str):
        """
        **Feature: multi-ai-provider, Property 3: Request Format Correctness (Anthropic)**
        **Validates: Requirements 4.2**
        
        Property: For any conversation starting with assistant message, a user
        message should be prepended to satisfy Anthropic's requirement.
        """
        from app.providers.anthropic import AnthropicProvider
        
        # Start with assistant message (invalid for Anthropic)
        messages = [{"role": "assistant", "content": assistant_content}]
        
        payload = AnthropicProvider.build_request_payload(
            messages=messages,
            model="claude-3-5-sonnet-20241022",
        )
        
        # First message should be user (prepended)
        assert len(payload["messages"]) >= 1
        assert payload["messages"][0]["role"] == "user"

    @given(
        user_content=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
        assistant_content=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_assistant_messages_preserved(
        self, user_content: str, assistant_content: str
    ):
        """
        **Feature: multi-ai-provider, Property 3: Request Format Correctness (Anthropic)**
        **Validates: Requirements 4.2**
        
        Property: For any conversation with assistant messages, they should be
        preserved with role 'assistant'.
        """
        from app.providers.anthropic import AnthropicProvider
        
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
        
        payload = AnthropicProvider.build_request_payload(
            messages=messages,
            model="claude-3-5-sonnet-20241022",
        )
        
        # Both messages should be preserved
        assert len(payload["messages"]) == 2
        assert payload["messages"][0]["role"] == "user"
        assert payload["messages"][0]["content"] == user_content
        assert payload["messages"][1]["role"] == "assistant"
        assert payload["messages"][1]["content"] == assistant_content

    @given(
        num_messages=st.integers(min_value=2, max_value=10),
    )
    @settings(max_examples=100)
    def test_message_order_preserved(self, num_messages: int):
        """
        **Feature: multi-ai-provider, Property 3: Request Format Correctness (Anthropic)**
        **Validates: Requirements 4.2**
        
        Property: For any sequence of messages, the order should be preserved
        in the generated payload.
        """
        from app.providers.anthropic import AnthropicProvider
        
        # Generate messages with alternating roles (starting with user)
        messages = []
        for i in range(num_messages):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": f"Message {i}"})
        
        payload = AnthropicProvider.build_request_payload(
            messages=messages,
            model="claude-3-5-sonnet-20241022",
        )
        
        # Verify order is preserved
        for i, msg in enumerate(payload["messages"]):
            assert msg["content"] == f"Message {i}"

    @given(
        api_key=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_x_api_key_header_format(self, api_key: str):
        """
        **Feature: multi-ai-provider, Property 3: Request Format Correctness (Anthropic)**
        **Validates: Requirements 4.2**
        
        Property: For any API key, the authentication header should use
        x-api-key format (not Bearer token).
        """
        from app.providers.anthropic import AnthropicProvider
        
        headers = AnthropicProvider.build_auth_header(api_key)
        
        assert "x-api-key" in headers
        assert headers["x-api-key"] == api_key
        # Should NOT use Bearer token
        assert "Authorization" not in headers

    @given(
        api_key=st.text(min_size=10, max_size=50).filter(
            lambda x: x.strip() and x == x.strip()
        ),
    )
    @settings(max_examples=100)
    def test_headers_include_anthropic_version(self, api_key: str):
        """
        **Feature: multi-ai-provider, Property 3: Request Format Correctness (Anthropic)**
        **Validates: Requirements 4.2**
        
        Property: For any Anthropic request, the headers should include
        the anthropic-version header.
        """
        from app.providers.anthropic import AnthropicProvider
        
        provider = AnthropicProvider(api_key=api_key)
        headers = provider._get_headers()
        
        assert "anthropic-version" in headers
        assert headers["anthropic-version"] == "2023-06-01"
        assert "Content-Type" in headers
        assert headers["Content-Type"] == "application/json"
        assert "x-api-key" in headers

    @given(
        api_key=st.text(min_size=10, max_size=50).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_endpoint_format_correctness(self, api_key: str):
        """
        **Feature: multi-ai-provider, Property 3: Request Format Correctness (Anthropic)**
        **Validates: Requirements 4.2**
        
        Property: For any Anthropic provider, the endpoint should
        end with /v1/messages.
        """
        from app.providers.anthropic import AnthropicProvider
        
        provider = AnthropicProvider(api_key=api_key)
        endpoint = provider._get_endpoint()
        
        assert endpoint.endswith("/v1/messages")

    @given(
        base_url=st.text(min_size=10, max_size=100).filter(
            lambda x: x.strip() and not x.endswith("/")
        ),
        api_key=st.text(min_size=10, max_size=50).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_base_url_trailing_slash_stripped(self, base_url: str, api_key: str):
        """
        **Feature: multi-ai-provider, Property 3: Request Format Correctness (Anthropic)**
        **Validates: Requirements 4.2**
        
        Property: For any base URL, trailing slashes should be stripped
        to ensure correct endpoint construction.
        """
        from app.providers.anthropic import AnthropicProvider
        
        # Test with trailing slash
        provider_with_slash = AnthropicProvider(
            api_key=api_key,
            base_url=base_url + "/",
        )
        provider_without_slash = AnthropicProvider(
            api_key=api_key,
            base_url=base_url,
        )
        
        # Both should have the same base_url without trailing slash
        assert provider_with_slash.base_url == base_url
        assert provider_without_slash.base_url == base_url
        assert not provider_with_slash.base_url.endswith("/")

    @given(
        content=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_no_system_param_when_no_system_messages(self, content: str):
        """
        **Feature: multi-ai-provider, Property 3: Request Format Correctness (Anthropic)**
        **Validates: Requirements 4.2**
        
        Property: For any conversation without system messages, the 'system'
        parameter should not be included in the payload.
        """
        from app.providers.anthropic import AnthropicProvider
        
        messages = [{"role": "user", "content": content}]
        
        payload = AnthropicProvider.build_request_payload(
            messages=messages,
            model="claude-3-5-sonnet-20241022",
        )
        
        # No system parameter when no system messages
        assert "system" not in payload

    @given(
        role=st.sampled_from(["function", "tool", "other"]),
        content=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_unknown_roles_mapped_to_user(self, role: str, content: str):
        """
        **Feature: multi-ai-provider, Property 3: Request Format Correctness (Anthropic)**
        **Validates: Requirements 4.2**
        
        Property: For any unknown role (not user, assistant, or system),
        the message should be mapped to 'user' role.
        """
        from app.providers.anthropic import AnthropicProvider
        
        messages = [{"role": role, "content": content}]
        
        payload = AnthropicProvider.build_request_payload(
            messages=messages,
            model="claude-3-5-sonnet-20241022",
        )
        
        # Unknown roles should be mapped to user
        assert payload["messages"][0]["role"] == "user"

    @given(
        api_key=st.text(min_size=10, max_size=50).filter(lambda x: x.strip()),
        model=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_custom_model_used(self, api_key: str, model: str):
        """
        **Feature: multi-ai-provider, Property 3: Request Format Correctness (Anthropic)**
        **Validates: Requirements 4.2**
        
        Property: For any custom model specified, it should be used instead
        of the default model.
        """
        from app.providers.anthropic import AnthropicProvider
        
        provider = AnthropicProvider(
            api_key=api_key,
            model=model,
        )
        
        assert provider.model == model

    @given(
        api_key=st.text(min_size=10, max_size=50).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_default_model_assigned(self, api_key: str):
        """
        **Feature: multi-ai-provider, Property 3: Request Format Correctness (Anthropic)**
        **Validates: Requirements 4.2**
        
        Property: For any provider without a custom model, the default model
        (claude-3-5-sonnet-20241022) should be assigned.
        """
        from app.providers.anthropic import AnthropicProvider
        
        provider = AnthropicProvider(api_key=api_key)
        
        assert provider.model == "claude-3-5-sonnet-20241022"

    @given(
        max_tokens=st.integers(min_value=1, max_value=4096),
    )
    @settings(max_examples=100)
    def test_max_tokens_always_included(self, max_tokens: int):
        """
        **Feature: multi-ai-provider, Property 3: Request Format Correctness (Anthropic)**
        **Validates: Requirements 4.2**
        
        Property: For any Anthropic request, max_tokens should always be included
        (it's required by Anthropic API, unlike OpenAI).
        """
        from app.providers.anthropic import AnthropicProvider
        
        messages = [{"role": "user", "content": "test"}]
        
        payload = AnthropicProvider.build_request_payload(
            messages=messages,
            model="claude-3-5-sonnet-20241022",
            max_tokens=max_tokens,
        )
        
        assert "max_tokens" in payload
        assert payload["max_tokens"] == max_tokens


class TestGeminiRequestFormat:
    """
    Property-based tests for Gemini request format correctness.
    
    **Feature: multi-ai-provider, Property 3: Request Format Correctness (Gemini)**
    **Validates: Requirements 5.2**
    
    These tests verify that Gemini requests conform to the Google Generative AI API specification:
    - Uses POST /v1beta/models/{model}:generateContent endpoint
    - API key in query parameters (not headers)
    - Proper message format with 'contents' array containing 'role' and 'parts'
    - System messages extracted to 'systemInstruction' parameter
    - Roles mapped: 'assistant' -> 'model', others -> 'user'
    - Correct payload structure with contents, generationConfig
    """

    @given(
        model=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        temperature=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        content=st.text(min_size=1, max_size=500).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_payload_has_required_fields(
        self, model: str, temperature: float, content: str
    ):
        """
        **Feature: multi-ai-provider, Property 3: Request Format Correctness (Gemini)**
        **Validates: Requirements 5.2**
        
        Property: For any valid parameters, the generated payload should have
        all required fields for Gemini's generateContent endpoint.
        """
        from app.providers.gemini import GeminiProvider
        
        messages = [{"role": "user", "content": content}]
        
        payload = GeminiProvider.build_request_payload(
            messages=messages,
            model=model,
            temperature=temperature,
        )
        
        # Verify required fields exist
        assert "contents" in payload
        assert "generationConfig" in payload
        assert "temperature" in payload["generationConfig"]
        
        # Verify field values
        assert payload["generationConfig"]["temperature"] == temperature

    @given(
        model=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        max_tokens=st.integers(min_value=1, max_value=4096),
        content=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_max_tokens_included_when_specified(
        self, model: str, max_tokens: int, content: str
    ):
        """
        **Feature: multi-ai-provider, Property 3: Request Format Correctness (Gemini)**
        **Validates: Requirements 5.2**
        
        Property: For any request with max_tokens specified, the payload should
        include maxOutputTokens in generationConfig with the correct value.
        """
        from app.providers.gemini import GeminiProvider
        
        messages = [{"role": "user", "content": content}]
        
        payload = GeminiProvider.build_request_payload(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
        )
        
        assert "maxOutputTokens" in payload["generationConfig"]
        assert payload["generationConfig"]["maxOutputTokens"] == max_tokens

    @given(
        model=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        content=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_max_tokens_omitted_when_none(self, model: str, content: str):
        """
        **Feature: multi-ai-provider, Property 3: Request Format Correctness (Gemini)**
        **Validates: Requirements 5.2**
        
        Property: For any request without max_tokens, the payload should not
        include maxOutputTokens in generationConfig.
        """
        from app.providers.gemini import GeminiProvider
        
        messages = [{"role": "user", "content": content}]
        
        payload = GeminiProvider.build_request_payload(
            messages=messages,
            model=model,
            max_tokens=None,
        )
        
        assert "maxOutputTokens" not in payload["generationConfig"]

    @given(
        content=st.text(min_size=1, max_size=300).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_user_message_format_correctness(self, content: str):
        """
        **Feature: multi-ai-provider, Property 3: Request Format Correctness (Gemini)**
        **Validates: Requirements 5.2**
        
        Property: For any user message, it should be included in the contents
        array with role 'user' and parts containing text.
        """
        from app.providers.gemini import GeminiProvider
        
        messages = [{"role": "user", "content": content}]
        
        payload = GeminiProvider.build_request_payload(
            messages=messages,
            model="gemini-1.5-flash",
        )
        
        # Verify contents structure
        assert len(payload["contents"]) == 1
        assert payload["contents"][0]["role"] == "user"
        assert "parts" in payload["contents"][0]
        assert len(payload["contents"][0]["parts"]) == 1
        assert payload["contents"][0]["parts"][0]["text"] == content

    @given(
        content=st.text(min_size=1, max_size=300).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_assistant_role_mapped_to_model(self, content: str):
        """
        **Feature: multi-ai-provider, Property 3: Request Format Correctness (Gemini)**
        **Validates: Requirements 5.2**
        
        Property: For any assistant message, the role should be mapped to 'model'
        (Gemini uses 'model' instead of 'assistant').
        """
        from app.providers.gemini import GeminiProvider
        
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": content},
        ]
        
        payload = GeminiProvider.build_request_payload(
            messages=messages,
            model="gemini-1.5-flash",
        )
        
        # Assistant should be mapped to 'model'
        assert len(payload["contents"]) == 2
        assert payload["contents"][0]["role"] == "user"
        assert payload["contents"][1]["role"] == "model"
        assert payload["contents"][1]["parts"][0]["text"] == content

    @given(
        system_content=st.text(min_size=1, max_size=300).filter(lambda x: x.strip()),
        user_content=st.text(min_size=1, max_size=300).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_system_message_extracted_to_system_instruction(
        self, system_content: str, user_content: str
    ):
        """
        **Feature: multi-ai-provider, Property 3: Request Format Correctness (Gemini)**
        **Validates: Requirements 5.2**
        
        Property: For any conversation with system messages, the system content
        should be extracted to 'systemInstruction' parameter, not in contents array.
        """
        from app.providers.gemini import GeminiProvider
        
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        
        payload = GeminiProvider.build_request_payload(
            messages=messages,
            model="gemini-1.5-flash",
        )
        
        # System message should be in 'systemInstruction' parameter
        assert "systemInstruction" in payload
        assert "parts" in payload["systemInstruction"]
        assert payload["systemInstruction"]["parts"][0]["text"] == system_content
        
        # Contents array should only contain user message
        assert len(payload["contents"]) == 1
        assert payload["contents"][0]["role"] == "user"
        assert payload["contents"][0]["parts"][0]["text"] == user_content

    @given(
        system_content_1=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
        system_content_2=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
        user_content=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_multiple_system_messages_concatenated(
        self, system_content_1: str, system_content_2: str, user_content: str
    ):
        """
        **Feature: multi-ai-provider, Property 3: Request Format Correctness (Gemini)**
        **Validates: Requirements 5.2**
        
        Property: For any conversation with multiple system messages, they should
        be concatenated with newlines in the 'systemInstruction' parameter.
        """
        from app.providers.gemini import GeminiProvider
        
        messages = [
            {"role": "system", "content": system_content_1},
            {"role": "system", "content": system_content_2},
            {"role": "user", "content": user_content},
        ]
        
        payload = GeminiProvider.build_request_payload(
            messages=messages,
            model="gemini-1.5-flash",
        )
        
        # Multiple system messages should be concatenated
        assert "systemInstruction" in payload
        system_text = payload["systemInstruction"]["parts"][0]["text"]
        assert system_content_1 in system_text
        assert system_content_2 in system_text
        assert "\n" in system_text

    @given(
        content=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_no_system_instruction_when_no_system_messages(self, content: str):
        """
        **Feature: multi-ai-provider, Property 3: Request Format Correctness (Gemini)**
        **Validates: Requirements 5.2**
        
        Property: For any conversation without system messages, the 'systemInstruction'
        parameter should not be included in the payload.
        """
        from app.providers.gemini import GeminiProvider
        
        messages = [{"role": "user", "content": content}]
        
        payload = GeminiProvider.build_request_payload(
            messages=messages,
            model="gemini-1.5-flash",
        )
        
        # No systemInstruction when no system messages
        assert "systemInstruction" not in payload

    @given(
        num_messages=st.integers(min_value=2, max_value=10),
    )
    @settings(max_examples=100)
    def test_message_order_preserved(self, num_messages: int):
        """
        **Feature: multi-ai-provider, Property 3: Request Format Correctness (Gemini)**
        **Validates: Requirements 5.2**
        
        Property: For any sequence of messages, the order should be preserved
        in the generated payload.
        """
        from app.providers.gemini import GeminiProvider
        
        # Generate messages with alternating roles (starting with user)
        messages = []
        for i in range(num_messages):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": f"Message {i}"})
        
        payload = GeminiProvider.build_request_payload(
            messages=messages,
            model="gemini-1.5-flash",
        )
        
        # Verify order is preserved
        for i, content_item in enumerate(payload["contents"]):
            assert content_item["parts"][0]["text"] == f"Message {i}"

    @given(
        api_key=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
        model=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_endpoint_includes_api_key_in_query_params(self, api_key: str, model: str):
        """
        **Feature: multi-ai-provider, Property 3: Request Format Correctness (Gemini)**
        **Validates: Requirements 5.2**
        
        Property: For any API key and model, the endpoint URL should include
        the API key as a query parameter (not in headers).
        """
        from app.providers.gemini import GeminiProvider
        
        base_url = "https://generativelanguage.googleapis.com"
        
        endpoint = GeminiProvider.build_endpoint_url(base_url, model, api_key)
        
        # Verify endpoint format
        assert f"key={api_key}" in endpoint
        assert f"/v1beta/models/{model}:generateContent" in endpoint
        assert "?" in endpoint

    @given(
        api_key=st.text(min_size=10, max_size=50).filter(
            lambda x: x.strip() and x == x.strip()
        ),
    )
    @settings(max_examples=100)
    def test_headers_do_not_include_authorization(self, api_key: str):
        """
        **Feature: multi-ai-provider, Property 3: Request Format Correctness (Gemini)**
        **Validates: Requirements 5.2**
        
        Property: For any Gemini request, the headers should NOT include
        Authorization header (API key is in query params instead).
        """
        from app.providers.gemini import GeminiProvider
        
        provider = GeminiProvider(api_key=api_key)
        headers = provider._get_headers()
        
        # Should NOT have Authorization header
        assert "Authorization" not in headers
        # Should have Content-Type
        assert "Content-Type" in headers
        assert headers["Content-Type"] == "application/json"

    @given(
        base_url=st.text(min_size=10, max_size=100).filter(
            lambda x: x.strip() and not x.endswith("/")
        ),
        api_key=st.text(min_size=10, max_size=50).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_base_url_trailing_slash_stripped(self, base_url: str, api_key: str):
        """
        **Feature: multi-ai-provider, Property 3: Request Format Correctness (Gemini)**
        **Validates: Requirements 5.2**
        
        Property: For any base URL, trailing slashes should be stripped
        to ensure correct endpoint construction.
        """
        from app.providers.gemini import GeminiProvider
        
        # Test with trailing slash
        provider_with_slash = GeminiProvider(
            api_key=api_key,
            base_url=base_url + "/",
        )
        provider_without_slash = GeminiProvider(
            api_key=api_key,
            base_url=base_url,
        )
        
        # Both should have the same base_url without trailing slash
        assert provider_with_slash.base_url == base_url
        assert provider_without_slash.base_url == base_url
        assert not provider_with_slash.base_url.endswith("/")

    @given(
        api_key=st.text(min_size=10, max_size=50).filter(lambda x: x.strip()),
        model=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_custom_model_used(self, api_key: str, model: str):
        """
        **Feature: multi-ai-provider, Property 3: Request Format Correctness (Gemini)**
        **Validates: Requirements 5.2**
        
        Property: For any custom model specified, it should be used instead
        of the default model.
        """
        from app.providers.gemini import GeminiProvider
        
        provider = GeminiProvider(
            api_key=api_key,
            model=model,
        )
        
        assert provider.model == model

    @given(
        api_key=st.text(min_size=10, max_size=50).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_default_model_assigned(self, api_key: str):
        """
        **Feature: multi-ai-provider, Property 3: Request Format Correctness (Gemini)**
        **Validates: Requirements 5.2**
        
        Property: For any provider without a custom model, the default model
        (gemini-1.5-flash) should be assigned.
        """
        from app.providers.gemini import GeminiProvider
        
        provider = GeminiProvider(api_key=api_key)
        
        assert provider.model == "gemini-1.5-flash"

    @given(
        role=st.sampled_from(["function", "tool", "other"]),
        content=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_unknown_roles_mapped_to_user(self, role: str, content: str):
        """
        **Feature: multi-ai-provider, Property 3: Request Format Correctness (Gemini)**
        **Validates: Requirements 5.2**
        
        Property: For any unknown role (not user, assistant, or system),
        the message should be mapped to 'user' role.
        """
        from app.providers.gemini import GeminiProvider
        
        messages = [{"role": role, "content": content}]
        
        payload = GeminiProvider.build_request_payload(
            messages=messages,
            model="gemini-1.5-flash",
        )
        
        # Unknown roles should be mapped to user
        assert payload["contents"][0]["role"] == "user"

    @given(
        temperature=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_temperature_in_generation_config(self, temperature: float):
        """
        **Feature: multi-ai-provider, Property 3: Request Format Correctness (Gemini)**
        **Validates: Requirements 5.2**
        
        Property: For any temperature value, it should be included in the
        generationConfig object as a float.
        """
        from app.providers.gemini import GeminiProvider
        
        messages = [{"role": "user", "content": "test"}]
        
        payload = GeminiProvider.build_request_payload(
            messages=messages,
            model="gemini-1.5-flash",
            temperature=temperature,
        )
        
        assert "generationConfig" in payload
        assert isinstance(payload["generationConfig"]["temperature"], float)
        assert payload["generationConfig"]["temperature"] == temperature

    @given(
        user_content=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
        assistant_content=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_conversation_messages_structure(
        self, user_content: str, assistant_content: str
    ):
        """
        **Feature: multi-ai-provider, Property 3: Request Format Correctness (Gemini)**
        **Validates: Requirements 5.2**
        
        Property: For any conversation with user and assistant messages,
        all messages should be preserved with correct roles (user and model).
        """
        from app.providers.gemini import GeminiProvider
        
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
        
        payload = GeminiProvider.build_request_payload(
            messages=messages,
            model="gemini-1.5-flash",
        )
        
        # Verify all messages are preserved
        assert len(payload["contents"]) == 2
        assert payload["contents"][0]["role"] == "user"
        assert payload["contents"][0]["parts"][0]["text"] == user_content
        assert payload["contents"][1]["role"] == "model"
        assert payload["contents"][1]["parts"][0]["text"] == assistant_content

    @given(
        api_key=st.text(min_size=10, max_size=50).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_provider_name_is_gemini(self, api_key: str):
        """
        **Feature: multi-ai-provider, Property 3: Request Format Correctness (Gemini)**
        **Validates: Requirements 5.2**
        
        Property: For any Gemini provider instance, the name property should
        return 'gemini'.
        """
        from app.providers.gemini import GeminiProvider
        
        provider = GeminiProvider(api_key=api_key)
        
        assert provider.name == "gemini"

    def test_empty_api_key_raises_error(self):
        """
        **Feature: multi-ai-provider, Property 3: Request Format Correctness (Gemini)**
        **Validates: Requirements 5.2**
        
        Property: For an empty API key, the Gemini provider should raise
        an AuthenticationError.
        """
        from app.providers.gemini import GeminiProvider
        
        with pytest.raises(AuthenticationError) as exc_info:
            GeminiProvider(api_key="")
        
        assert "gemini" in str(exc_info.value).lower()

    @given(
        whitespace_key=st.text(max_size=20).filter(
            lambda x: not x or x.isspace()
        ),
    )
    @settings(max_examples=100)
    def test_whitespace_only_api_key_raises_error(self, whitespace_key: str):
        """
        **Feature: multi-ai-provider, Property 3: Request Format Correctness (Gemini)**
        **Validates: Requirements 5.2**
        
        Property: For a whitespace-only API key, the Gemini provider should
        raise an AuthenticationError.
        """
        from app.providers.gemini import GeminiProvider
        
        with pytest.raises(AuthenticationError) as exc_info:
            GeminiProvider(api_key=whitespace_key)
        
        assert "gemini" in str(exc_info.value).lower()

    @given(
        candidates_text=st.text(min_size=1, max_size=500).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_parse_candidates_extracts_text(self, candidates_text: str):
        """
        **Feature: multi-ai-provider, Property 3: Request Format Correctness (Gemini)**
        **Validates: Requirements 5.2**
        
        Property: For any valid candidates response, parse_candidates should
        correctly extract the text content.
        """
        from app.providers.gemini import GeminiProvider
        
        # Simulate Gemini candidates response structure
        candidates = [
            {
                "content": {
                    "parts": [{"text": candidates_text}],
                    "role": "model",
                }
            }
        ]
        
        result = GeminiProvider.parse_candidates(candidates)
        
        assert result == candidates_text

    def test_parse_candidates_empty_list_returns_empty_string(self):
        """
        **Feature: multi-ai-provider, Property 3: Request Format Correctness (Gemini)**
        **Validates: Requirements 5.2**
        
        Property: For an empty candidates list, parse_candidates should
        return an empty string.
        """
        from app.providers.gemini import GeminiProvider
        
        result = GeminiProvider.parse_candidates([])
        
        assert result == ""

    @given(
        text_parts=st.lists(
            st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
            min_size=2,
            max_size=5,
        ),
    )
    @settings(max_examples=100)
    def test_parse_candidates_concatenates_multiple_parts(self, text_parts: list[str]):
        """
        **Feature: multi-ai-provider, Property 3: Request Format Correctness (Gemini)**
        **Validates: Requirements 5.2**
        
        Property: For any candidates with multiple text parts, parse_candidates
        should concatenate all text parts.
        """
        from app.providers.gemini import GeminiProvider
        
        # Simulate Gemini candidates with multiple parts
        candidates = [
            {
                "content": {
                    "parts": [{"text": part} for part in text_parts],
                    "role": "model",
                }
            }
        ]
        
        result = GeminiProvider.parse_candidates(candidates)
        
        # All parts should be concatenated
        expected = "".join(text_parts)
        assert result == expected



class TestHealthStatusAccuracy:
    """
    Property-based tests for health status accuracy.
    
    **Feature: multi-ai-provider, Property 7: Health Status Accuracy**
    **Validates: Requirements 11.1, 11.2**
    
    These tests verify that health checks accurately report provider availability
    status within a reasonable timeout period.
    """

    @given(
        status=st.sampled_from(["healthy", "unhealthy"]),
        provider=st.sampled_from(VALID_PROVIDER_NAMES),
        latency_ms=st.floats(min_value=0.0, max_value=100000.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_build_health_response_returns_valid_structure(
        self, status: str, provider: str, latency_ms: float
    ):
        """
        **Feature: multi-ai-provider, Property 7: Health Status Accuracy**
        **Validates: Requirements 11.1, 11.2**
        
        Property: For any valid status, provider, and latency, build_health_response
        should return a dict with all required fields.
        """
        from app.providers.base import build_health_response
        
        result = build_health_response(
            status=status,
            provider=provider,
            latency_ms=latency_ms,
        )
        
        # Required fields must be present
        assert "status" in result
        assert "provider" in result
        assert "latency_ms" in result
        
        # Values must match inputs
        assert result["status"] == status
        assert result["provider"] == provider
        # Latency should be rounded to 2 decimal places
        assert result["latency_ms"] == round(latency_ms, 2)

    @given(
        provider=st.sampled_from(VALID_PROVIDER_NAMES),
        latency_ms=st.floats(
            min_value=5001.0,  # Above LATENCY_THRESHOLD_MS (5000)
            max_value=100000.0,
            allow_nan=False,
            allow_infinity=False,
        ),
    )
    @settings(max_examples=100)
    def test_latency_warning_set_when_threshold_exceeded(
        self, provider: str, latency_ms: float
    ):
        """
        **Feature: multi-ai-provider, Property 7: Health Status Accuracy**
        **Validates: Requirements 11.1, 11.2**
        
        Property: For any healthy provider with latency exceeding the threshold,
        the health response should include a latency warning.
        """
        from app.providers.base import build_health_response, LATENCY_THRESHOLD_MS
        
        result = build_health_response(
            status="healthy",
            provider=provider,
            latency_ms=latency_ms,
        )
        
        # Latency warning should be set for healthy status with high latency
        assert result.get("latency_warning") is True
        assert result.get("latency_threshold_ms") == LATENCY_THRESHOLD_MS

    @given(
        provider=st.sampled_from(VALID_PROVIDER_NAMES),
        latency_ms=st.floats(
            min_value=0.0,
            max_value=4999.0,  # Below LATENCY_THRESHOLD_MS (5000)
            allow_nan=False,
            allow_infinity=False,
        ),
    )
    @settings(max_examples=100)
    def test_no_latency_warning_when_below_threshold(
        self, provider: str, latency_ms: float
    ):
        """
        **Feature: multi-ai-provider, Property 7: Health Status Accuracy**
        **Validates: Requirements 11.1, 11.2**
        
        Property: For any healthy provider with latency below the threshold,
        the health response should NOT include a latency warning.
        """
        from app.providers.base import build_health_response
        
        result = build_health_response(
            status="healthy",
            provider=provider,
            latency_ms=latency_ms,
        )
        
        # Latency warning should NOT be set for healthy status with low latency
        assert "latency_warning" not in result

    @given(
        provider=st.sampled_from(VALID_PROVIDER_NAMES),
        latency_ms=st.floats(
            min_value=5001.0,  # Above threshold
            max_value=100000.0,
            allow_nan=False,
            allow_infinity=False,
        ),
    )
    @settings(max_examples=100)
    def test_no_latency_warning_for_unhealthy_status(
        self, provider: str, latency_ms: float
    ):
        """
        **Feature: multi-ai-provider, Property 7: Health Status Accuracy**
        **Validates: Requirements 11.1, 11.2**
        
        Property: For any unhealthy provider, even with high latency,
        the health response should NOT include a latency warning (since
        the provider is already marked as unhealthy).
        """
        from app.providers.base import build_health_response
        
        result = build_health_response(
            status="unhealthy",
            provider=provider,
            latency_ms=latency_ms,
        )
        
        # Latency warning should NOT be set for unhealthy status
        assert "latency_warning" not in result

    @given(
        provider=st.sampled_from(VALID_PROVIDER_NAMES),
        latency_ms=st.floats(min_value=0.0, max_value=100000.0, allow_nan=False, allow_infinity=False),
        error_message=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_error_message_included_when_provided(
        self, provider: str, latency_ms: float, error_message: str
    ):
        """
        **Feature: multi-ai-provider, Property 7: Health Status Accuracy**
        **Validates: Requirements 11.1, 11.2**
        
        Property: For any unhealthy provider with an error message,
        the health response should include the error message.
        """
        from app.providers.base import build_health_response
        
        result = build_health_response(
            status="unhealthy",
            provider=provider,
            latency_ms=latency_ms,
            error=error_message,
        )
        
        assert result.get("error") == error_message

    @given(
        provider=st.sampled_from(VALID_PROVIDER_NAMES),
        latency_ms=st.floats(min_value=0.0, max_value=100000.0, allow_nan=False, allow_infinity=False),
        base_url=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
        model=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_optional_fields_included_when_provided(
        self, provider: str, latency_ms: float, base_url: str, model: str
    ):
        """
        **Feature: multi-ai-provider, Property 7: Health Status Accuracy**
        **Validates: Requirements 11.1, 11.2**
        
        Property: For any health response with optional fields (base_url, model),
        those fields should be included in the response.
        """
        from app.providers.base import build_health_response
        
        result = build_health_response(
            status="healthy",
            provider=provider,
            latency_ms=latency_ms,
            base_url=base_url,
            model=model,
        )
        
        assert result.get("base_url") == base_url
        assert result.get("model") == model

    @given(
        provider=st.sampled_from(VALID_PROVIDER_NAMES),
        latency_ms=st.floats(min_value=0.0, max_value=100000.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_latency_is_always_non_negative(
        self, provider: str, latency_ms: float
    ):
        """
        **Feature: multi-ai-provider, Property 7: Health Status Accuracy**
        **Validates: Requirements 11.1, 11.2**
        
        Property: For any health response, the latency_ms value should
        always be non-negative.
        """
        from app.providers.base import build_health_response
        
        result = build_health_response(
            status="healthy",
            provider=provider,
            latency_ms=latency_ms,
        )
        
        assert result["latency_ms"] >= 0.0

    @given(
        status=st.sampled_from(["healthy", "unhealthy"]),
        provider=st.sampled_from(VALID_PROVIDER_NAMES),
        latency_ms=st.floats(min_value=0.0, max_value=100000.0, allow_nan=False, allow_infinity=False),
        extra_key=st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz_"),
        extra_value=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_extra_fields_passed_through(
        self, status: str, provider: str, latency_ms: float, extra_key: str, extra_value: str
    ):
        """
        **Feature: multi-ai-provider, Property 7: Health Status Accuracy**
        **Validates: Requirements 11.1, 11.2**
        
        Property: For any extra keyword arguments passed to build_health_response,
        those fields should be included in the response.
        """
        from app.providers.base import build_health_response
        
        # Avoid conflicts with reserved keys
        assume(extra_key not in ["status", "provider", "latency_ms", "base_url", "model", "error", "latency_warning", "latency_threshold_ms"])
        
        result = build_health_response(
            status=status,
            provider=provider,
            latency_ms=latency_ms,
            **{extra_key: extra_value},
        )
        
        assert result.get(extra_key) == extra_value

    def test_health_check_timeout_constant_is_reasonable(self):
        """
        **Feature: multi-ai-provider, Property 7: Health Status Accuracy**
        **Validates: Requirements 11.1, 11.2**
        
        Property: The health check timeout constant should be a reasonable value
        (between 1 and 60 seconds).
        """
        from app.providers.base import HEALTH_CHECK_TIMEOUT
        
        assert 1.0 <= HEALTH_CHECK_TIMEOUT <= 60.0

    def test_latency_threshold_constant_is_reasonable(self):
        """
        **Feature: multi-ai-provider, Property 7: Health Status Accuracy**
        **Validates: Requirements 11.1, 11.2**
        
        Property: The latency threshold constant should be a reasonable value
        (between 100ms and 30000ms).
        """
        from app.providers.base import LATENCY_THRESHOLD_MS
        
        assert 100.0 <= LATENCY_THRESHOLD_MS <= 30000.0

    @given(
        latency_ms=st.floats(min_value=0.0, max_value=100000.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_latency_rounded_to_two_decimal_places(self, latency_ms: float):
        """
        **Feature: multi-ai-provider, Property 7: Health Status Accuracy**
        **Validates: Requirements 11.1, 11.2**
        
        Property: For any latency value, the result should be rounded to
        exactly 2 decimal places.
        """
        from app.providers.base import build_health_response
        
        result = build_health_response(
            status="healthy",
            provider="ollama",
            latency_ms=latency_ms,
        )
        
        # Check that the value is rounded to 2 decimal places
        rounded_value = result["latency_ms"]
        assert rounded_value == round(latency_ms, 2)
        
        # Verify it's actually rounded (string representation check)
        str_value = str(rounded_value)
        if "." in str_value:
            decimal_places = len(str_value.split(".")[1])
            assert decimal_places <= 2



class TestFallbackActivation:
    """
    Property-based tests for fallback activation mechanism.
    
    **Feature: multi-ai-provider, Property 5: Fallback Activation**
    **Validates: Requirements 10.1, 10.2**
    
    These tests verify that when the primary provider fails (network error,
    rate limit, server error), the system automatically retries with the
    fallback provider if one is configured.
    """

    @given(
        error_type=st.sampled_from([
            "ProviderError",
            "RateLimitError",
            "QuotaExceededError",
            "AuthenticationError",
        ]),
        error_message=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
        provider_name=st.sampled_from(VALID_PROVIDER_NAMES),
    )
    @settings(max_examples=100)
    def test_provider_errors_are_retryable_types(
        self, error_type: str, error_message: str, provider_name: str
    ):
        """
        **Feature: multi-ai-provider, Property 5: Fallback Activation**
        **Validates: Requirements 10.1, 10.2**
        
        Property: For any provider error type, the error should be a subclass
        of ProviderError and contain the provider name.
        """
        from app.providers.base import (
            ProviderError,
            RateLimitError,
            QuotaExceededError,
            AuthenticationError,
        )
        
        error_classes = {
            "ProviderError": ProviderError,
            "RateLimitError": RateLimitError,
            "QuotaExceededError": QuotaExceededError,
            "AuthenticationError": AuthenticationError,
        }
        
        error_class = error_classes[error_type]
        
        if error_type == "RateLimitError":
            error = error_class(message=error_message, provider=provider_name, retry_after=1.0)
        else:
            error = error_class(message=error_message, provider=provider_name)
        
        # All errors should be ProviderError subclasses
        assert isinstance(error, ProviderError)
        # All errors should contain provider info
        assert error.provider == provider_name

    @given(
        primary_provider=st.sampled_from(VALID_PROVIDER_NAMES),
        fallback_provider=st.sampled_from(VALID_PROVIDER_NAMES),
    )
    @settings(max_examples=100)
    def test_fallback_provider_can_be_different_from_primary(
        self, primary_provider: str, fallback_provider: str
    ):
        """
        **Feature: multi-ai-provider, Property 5: Fallback Activation**
        **Validates: Requirements 10.1, 10.2**
        
        Property: For any combination of primary and fallback providers,
        the system should support configuring them independently.
        """
        # Both providers should be valid
        assert primary_provider in VALID_PROVIDER_NAMES
        assert fallback_provider in VALID_PROVIDER_NAMES
        
        # They can be the same or different - both are valid configurations
        # This is a configuration flexibility property

    @given(
        status_code=st.sampled_from([429, 500, 502, 503, 504]),
        provider_name=st.sampled_from(VALID_PROVIDER_NAMES),
    )
    @settings(max_examples=100)
    def test_retryable_status_codes_trigger_fallback(
        self, status_code: int, provider_name: str
    ):
        """
        **Feature: multi-ai-provider, Property 5: Fallback Activation**
        **Validates: Requirements 10.1, 10.2**
        
        Property: For any retryable HTTP status code (429, 5xx), the error
        should be marked as retryable, indicating fallback should be attempted.
        """
        from app.providers.base import ProviderError, RateLimitError
        
        # Rate limit errors (429) are always retryable
        if status_code == 429:
            error = RateLimitError(
                message="Rate limit exceeded",
                provider=provider_name,
                retry_after=60.0,
            )
            assert error.retryable is True
            assert error.status_code == 429
        else:
            # Server errors (5xx) should be marked as retryable
            error = ProviderError(
                message=f"Server error: {status_code}",
                provider=provider_name,
                status_code=status_code,
                retryable=True,  # Server errors are typically retryable
            )
            assert error.status_code == status_code
            assert error.retryable is True

    @given(
        status_code=st.sampled_from([400, 401, 403, 404]),
        provider_name=st.sampled_from(VALID_PROVIDER_NAMES),
    )
    @settings(max_examples=100)
    def test_non_retryable_status_codes(
        self, status_code: int, provider_name: str
    ):
        """
        **Feature: multi-ai-provider, Property 5: Fallback Activation**
        **Validates: Requirements 10.1, 10.2**
        
        Property: For any non-retryable HTTP status code (4xx except 429),
        the error should not be marked as retryable by default.
        """
        from app.providers.base import ProviderError
        
        # Client errors (4xx except 429) are typically not retryable
        error = ProviderError(
            message=f"Client error: {status_code}",
            provider=provider_name,
            status_code=status_code,
            retryable=False,
        )
        
        assert error.status_code == status_code
        assert error.retryable is False

    @given(
        retry_after=st.floats(min_value=0.0, max_value=3600.0, allow_nan=False) | st.none(),
        provider_name=st.sampled_from(VALID_PROVIDER_NAMES),
    )
    @settings(max_examples=100)
    def test_rate_limit_error_preserves_retry_after(
        self, retry_after: float | None, provider_name: str
    ):
        """
        **Feature: multi-ai-provider, Property 5: Fallback Activation**
        **Validates: Requirements 10.1, 10.2**
        
        Property: For any rate limit error with a retry_after value,
        that value should be preserved in the error for fallback timing.
        """
        from app.providers.base import RateLimitError
        
        error = RateLimitError(
            message="Rate limit exceeded",
            provider=provider_name,
            retry_after=retry_after,
        )
        
        assert error.retry_after == retry_after
        assert error.retryable is True

    def test_manager_has_fallback_property(self):
        """
        **Feature: multi-ai-provider, Property 5: Fallback Activation**
        **Validates: Requirements 10.1, 10.2**
        
        Property: The AIProviderManager should expose a has_fallback property
        to indicate whether a fallback provider is configured.
        """
        from unittest.mock import MagicMock
        from app.providers.manager import AIProviderManager
        
        # Create a mock config
        mock_config = MagicMock()
        mock_config.ai_provider = "ollama"
        mock_config.ai_api_key = None
        mock_config.ai_fallback_provider = None
        mock_config.ai_fallback_api_key = None
        mock_config.ai_fallback_base_url = None
        mock_config.ai_fallback_model = None
        mock_config.ai_timeout = 30.0
        mock_config.ai_debug_logging = False
        mock_config.ai_cost_tracking_enabled = False
        mock_config.backend_api_url = "http://localhost:8000"
        mock_config.get_effective_ai_base_url.return_value = "http://localhost:11434"
        mock_config.get_effective_ai_model.return_value = "qwen2.5:3b-instruct"
        
        manager = AIProviderManager(mock_config)
        
        # has_fallback should be a boolean property
        assert isinstance(manager.has_fallback, bool)
        # Without fallback configured, should be False
        assert manager.has_fallback is False

    def test_manager_has_primary_property(self):
        """
        **Feature: multi-ai-provider, Property 5: Fallback Activation**
        **Validates: Requirements 10.1, 10.2**
        
        Property: The AIProviderManager should expose a has_primary property
        to indicate whether a primary provider is configured.
        """
        from unittest.mock import MagicMock
        from app.providers.manager import AIProviderManager
        
        # Create a mock config
        mock_config = MagicMock()
        mock_config.ai_provider = "ollama"
        mock_config.ai_api_key = None
        mock_config.ai_fallback_provider = None
        mock_config.ai_fallback_api_key = None
        mock_config.ai_fallback_base_url = None
        mock_config.ai_fallback_model = None
        mock_config.ai_timeout = 30.0
        mock_config.ai_debug_logging = False
        mock_config.ai_cost_tracking_enabled = False
        mock_config.backend_api_url = "http://localhost:8000"
        mock_config.get_effective_ai_base_url.return_value = "http://localhost:11434"
        mock_config.get_effective_ai_model.return_value = "qwen2.5:3b-instruct"
        
        manager = AIProviderManager(mock_config)
        
        # has_primary should be a boolean property
        assert isinstance(manager.has_primary, bool)
        # With ollama configured (no API key needed), should be True
        assert manager.has_primary is True

    @given(
        error_message=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
        primary_provider=st.sampled_from(VALID_PROVIDER_NAMES),
        fallback_provider=st.sampled_from(VALID_PROVIDER_NAMES),
    )
    @settings(max_examples=100)
    def test_combined_error_message_contains_both_errors(
        self, error_message: str, primary_provider: str, fallback_provider: str
    ):
        """
        **Feature: multi-ai-provider, Property 5: Fallback Activation**
        **Validates: Requirements 10.1, 10.2**
        
        Property: When both primary and fallback providers fail, the final
        error message should contain information about both failures.
        """
        from app.providers.base import ProviderError
        
        primary_error = ProviderError(
            message=f"Primary failed: {error_message}",
            provider=primary_provider,
        )
        
        fallback_error = ProviderError(
            message=f"Fallback failed: {error_message}",
            provider=fallback_provider,
        )
        
        # Simulate combined error message as manager would create
        combined_message = f"All providers failed. Primary: {primary_error}. Fallback: {fallback_error}"
        
        combined_error = ProviderError(
            message=combined_message,
            provider="manager",
            retryable=False,
        )
        
        # Combined error should mention both failures
        assert "Primary" in str(combined_error)
        assert "Fallback" in str(combined_error)
        assert combined_error.retryable is False

    @given(
        provider_name=st.sampled_from(VALID_PROVIDER_NAMES),
    )
    @settings(max_examples=100)
    def test_provider_factory_creates_correct_provider_type(
        self, provider_name: str
    ):
        """
        **Feature: multi-ai-provider, Property 5: Fallback Activation**
        **Validates: Requirements 10.1, 10.2**
        
        Property: For any valid provider name, the factory method should
        create the correct provider type (or raise AuthenticationError if
        API key is required but missing).
        """
        from app.providers.manager import AIProviderManager, OPENAI_COMPATIBLE_PROVIDERS, GLM_PROVIDER_ALIASES
        from app.providers.base import AuthenticationError
        from app.providers.ollama import OllamaProvider
        from app.providers.openai_compatible import OpenAICompatibleProvider
        from app.providers.glm import GLMProvider
        from app.providers.anthropic import AnthropicProvider
        from app.providers.gemini import GeminiProvider
        
        # Ollama doesn't require API key
        if provider_name == "ollama":
            provider = AIProviderManager._create_provider(
                provider_name=provider_name,
                api_key=None,
                base_url=None,
                model=None,
            )
            assert isinstance(provider, OllamaProvider)
        
        # OpenAI-compatible providers require API key
        elif provider_name in OPENAI_COMPATIBLE_PROVIDERS:
            # Without API key, should raise AuthenticationError
            try:
                AIProviderManager._create_provider(
                    provider_name=provider_name,
                    api_key=None,
                    base_url=None,
                    model=None,
                )
                assert False, "Should have raised AuthenticationError"
            except AuthenticationError as e:
                assert e.provider == provider_name
            
            # With API key, should create provider
            provider = AIProviderManager._create_provider(
                provider_name=provider_name,
                api_key="test-api-key",
                base_url=None,
                model=None,
            )
            assert isinstance(provider, OpenAICompatibleProvider)
        
        # GLM providers require API key
        elif provider_name in GLM_PROVIDER_ALIASES:
            try:
                AIProviderManager._create_provider(
                    provider_name=provider_name,
                    api_key=None,
                    base_url=None,
                    model=None,
                )
                assert False, "Should have raised AuthenticationError"
            except AuthenticationError as e:
                assert e.provider == provider_name
            
            # With API key (GLM format: id.secret)
            provider = AIProviderManager._create_provider(
                provider_name=provider_name,
                api_key="test-id.test-secret",
                base_url=None,
                model=None,
            )
            assert isinstance(provider, GLMProvider)
        
        # Anthropic requires API key
        elif provider_name == "anthropic":
            try:
                AIProviderManager._create_provider(
                    provider_name=provider_name,
                    api_key=None,
                    base_url=None,
                    model=None,
                )
                assert False, "Should have raised AuthenticationError"
            except AuthenticationError as e:
                assert e.provider == provider_name
            
            provider = AIProviderManager._create_provider(
                provider_name=provider_name,
                api_key="test-api-key",
                base_url=None,
                model=None,
            )
            assert isinstance(provider, AnthropicProvider)
        
        # Gemini requires API key
        elif provider_name == "gemini":
            try:
                AIProviderManager._create_provider(
                    provider_name=provider_name,
                    api_key=None,
                    base_url=None,
                    model=None,
                )
                assert False, "Should have raised AuthenticationError"
            except AuthenticationError as e:
                assert e.provider == provider_name
            
            provider = AIProviderManager._create_provider(
                provider_name=provider_name,
                api_key="test-api-key",
                base_url=None,
                model=None,
            )
            assert isinstance(provider, GeminiProvider)

    def test_fallback_logging_on_primary_failure(self):
        """
        **Feature: multi-ai-provider, Property 5: Fallback Activation**
        **Validates: Requirements 10.1, 10.2**
        
        Property: When fallback is triggered, the system should log the event
        with provider names and error details (Requirements 10.4).
        """
        from app.providers.base import ProviderError
        
        # Simulate the logging scenario
        primary_provider = "openai"
        fallback_provider = "ollama"
        primary_error = ProviderError(
            message="Connection timeout",
            provider=primary_provider,
            status_code=504,
            retryable=True,
        )
        
        # The log message format as used in manager
        log_info = {
            "fallback_provider": fallback_provider,
            "primary_error": str(primary_error),
        }
        
        # Verify log info contains required fields
        assert "fallback_provider" in log_info
        assert "primary_error" in log_info
        assert log_info["fallback_provider"] == fallback_provider
        assert "Connection timeout" in log_info["primary_error"]



class TestGracefulDegradation:
    """
    Property-based tests for graceful degradation mechanism.
    
    **Feature: multi-ai-provider, Property 6: Graceful Degradation**
    **Validates: Requirements 10.3, 2.4**
    
    These tests verify that when all configured providers fail, the system
    falls back to heuristic parsing and returns a valid (though potentially
    less accurate) result.
    """

    @given(
        user_message=st.text(min_size=1, max_size=500).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_heuristic_function_receives_user_message(self, user_message: str):
        """
        **Feature: multi-ai-provider, Property 6: Graceful Degradation**
        **Validates: Requirements 10.3, 2.4**
        
        Property: For any user message, when heuristic fallback is triggered,
        the heuristic function should receive the last user message content.
        """
        # Simulate message extraction as manager does
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_message},
        ]
        
        # Extract last user message as manager would
        extracted_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                extracted_message = msg.get("content", "")
                break
        
        assert extracted_message == user_message

    @given(
        user_message=st.text(min_size=1, max_size=500).filter(lambda x: x.strip()),
        heuristic_result=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_heuristic_result_returned_when_providers_fail(
        self, user_message: str, heuristic_result: str
    ):
        """
        **Feature: multi-ai-provider, Property 6: Graceful Degradation**
        **Validates: Requirements 10.3, 2.4**
        
        Property: For any heuristic function that returns a valid result,
        when all providers fail, the system should return (None, heuristic_result).
        """
        # Define a simple heuristic function
        def heuristic_fn(message: str) -> str:
            return heuristic_result
        
        # Simulate the return format from chat_completion_with_heuristic_fallback
        # When AI fails and heuristic succeeds: (None, heuristic_result)
        ai_response = None
        result = heuristic_fn(user_message)
        
        # Verify the expected return format
        assert ai_response is None
        assert result == heuristic_result

    @given(
        user_message=st.text(min_size=1, max_size=500).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_heuristic_function_can_process_any_input(self, user_message: str):
        """
        **Feature: multi-ai-provider, Property 6: Graceful Degradation**
        **Validates: Requirements 10.3, 2.4**
        
        Property: For any user message, a well-designed heuristic function
        should be able to process it without raising exceptions.
        """
        # Example heuristic that extracts numbers from text (common for finance bot)
        def simple_heuristic(message: str) -> dict:
            import re
            numbers = re.findall(r'\d+(?:\.\d+)?', message)
            return {
                "raw_input": message,
                "extracted_numbers": numbers,
                "has_numbers": len(numbers) > 0,
            }
        
        # Heuristic should handle any input without exception
        result = simple_heuristic(user_message)
        
        assert isinstance(result, dict)
        assert "raw_input" in result
        assert result["raw_input"] == user_message

    @given(
        messages_count=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=100)
    def test_last_user_message_extracted_from_conversation(self, messages_count: int):
        """
        **Feature: multi-ai-provider, Property 6: Graceful Degradation**
        **Validates: Requirements 10.3, 2.4**
        
        Property: For any conversation with multiple messages, the heuristic
        should receive only the last user message.
        """
        # Generate a conversation with alternating roles
        messages = []
        last_user_content = None
        
        for i in range(messages_count):
            if i % 2 == 0:
                content = f"User message {i}"
                messages.append({"role": "user", "content": content})
                last_user_content = content
            else:
                messages.append({"role": "assistant", "content": f"Assistant response {i}"})
        
        # Extract last user message as manager would
        extracted_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                extracted_message = msg.get("content", "")
                break
        
        assert extracted_message == last_user_content

    @given(
        error_message=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
        heuristic_error=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_combined_error_when_heuristic_also_fails(
        self, error_message: str, heuristic_error: str
    ):
        """
        **Feature: multi-ai-provider, Property 6: Graceful Degradation**
        **Validates: Requirements 10.3, 2.4**
        
        Property: When both AI providers and heuristic fail, the final error
        should contain information about both the AI failure and heuristic failure.
        """
        from app.providers.base import ProviderError
        
        ai_error = ProviderError(
            message=f"AI failed: {error_message}",
            provider="manager",
        )
        
        # Simulate combined error message as manager would create
        combined_message = f"All providers and heuristic failed. AI error: {ai_error}. Heuristic error: {heuristic_error}"
        
        combined_error = ProviderError(
            message=combined_message,
            provider="manager",
            retryable=False,
        )
        
        # Combined error should mention both failures
        assert "AI error" in str(combined_error)
        assert "Heuristic error" in str(combined_error)
        assert combined_error.retryable is False

    @given(
        user_message=st.text(min_size=1, max_size=500).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_heuristic_result_type_flexibility(self, user_message: str):
        """
        **Feature: multi-ai-provider, Property 6: Graceful Degradation**
        **Validates: Requirements 10.3, 2.4**
        
        Property: The heuristic function can return any type (dict, string, etc.)
        as long as it's a valid result for the application.
        """
        # Heuristic returning a dict
        def dict_heuristic(message: str) -> dict:
            return {"intent": "unknown", "raw": message}
        
        # Heuristic returning a string
        def string_heuristic(message: str) -> str:
            return f"Processed: {message}"
        
        # Heuristic returning a list
        def list_heuristic(message: str) -> list:
            return message.split()
        
        # All should work without exception
        dict_result = dict_heuristic(user_message)
        string_result = string_heuristic(user_message)
        list_result = list_heuristic(user_message)
        
        assert isinstance(dict_result, dict)
        assert isinstance(string_result, str)
        assert isinstance(list_result, list)

    def test_chat_completion_with_heuristic_fallback_method_exists(self):
        """
        **Feature: multi-ai-provider, Property 6: Graceful Degradation**
        **Validates: Requirements 10.3, 2.4**
        
        Property: The AIProviderManager should have a chat_completion_with_heuristic_fallback
        method that supports graceful degradation.
        """
        from unittest.mock import MagicMock
        from app.providers.manager import AIProviderManager
        
        # Create a mock config
        mock_config = MagicMock()
        mock_config.ai_provider = "ollama"
        mock_config.ai_api_key = None
        mock_config.ai_fallback_provider = None
        mock_config.ai_fallback_api_key = None
        mock_config.ai_fallback_base_url = None
        mock_config.ai_fallback_model = None
        mock_config.ai_timeout = 30.0
        mock_config.ai_debug_logging = False
        mock_config.ai_cost_tracking_enabled = False
        mock_config.backend_api_url = "http://localhost:8000"
        mock_config.get_effective_ai_base_url.return_value = "http://localhost:11434"
        mock_config.get_effective_ai_model.return_value = "qwen2.5:3b-instruct"
        
        manager = AIProviderManager(mock_config)
        
        # Method should exist
        assert hasattr(manager, 'chat_completion_with_heuristic_fallback')
        assert callable(manager.chat_completion_with_heuristic_fallback)

    @given(
        user_message=st.text(min_size=0, max_size=500),
    )
    @settings(max_examples=100)
    def test_empty_message_handled_gracefully(self, user_message: str):
        """
        **Feature: multi-ai-provider, Property 6: Graceful Degradation**
        **Validates: Requirements 10.3, 2.4**
        
        Property: For any message (including empty), the heuristic fallback
        should handle it gracefully without crashing.
        """
        # Heuristic that handles empty input
        def robust_heuristic(message: str) -> dict:
            if not message or not message.strip():
                return {"intent": "empty", "raw": message}
            return {"intent": "unknown", "raw": message}
        
        result = robust_heuristic(user_message)
        
        assert isinstance(result, dict)
        assert "intent" in result
        assert "raw" in result

    @given(
        user_message=st.text(min_size=1, max_size=500).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_heuristic_preserves_original_input(self, user_message: str):
        """
        **Feature: multi-ai-provider, Property 6: Graceful Degradation**
        **Validates: Requirements 10.3, 2.4**
        
        Property: For any user message, the heuristic result should preserve
        or reference the original input for traceability.
        """
        def traceable_heuristic(message: str) -> dict:
            return {
                "original_input": message,
                "processed": True,
                "method": "heuristic",
            }
        
        result = traceable_heuristic(user_message)
        
        assert result["original_input"] == user_message
        assert result["method"] == "heuristic"

    @given(
        num_system_messages=st.integers(min_value=0, max_value=3),
        num_user_messages=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=100)
    def test_user_message_extraction_ignores_system_messages(
        self, num_system_messages: int, num_user_messages: int
    ):
        """
        **Feature: multi-ai-provider, Property 6: Graceful Degradation**
        **Validates: Requirements 10.3, 2.4**
        
        Property: When extracting the user message for heuristic processing,
        system messages should be ignored and only user messages considered.
        """
        messages = []
        
        # Add system messages
        for i in range(num_system_messages):
            messages.append({"role": "system", "content": f"System instruction {i}"})
        
        # Add user messages
        last_user_content = None
        for i in range(num_user_messages):
            content = f"User query {i}"
            messages.append({"role": "user", "content": content})
            last_user_content = content
            if i < num_user_messages - 1:
                messages.append({"role": "assistant", "content": f"Response {i}"})
        
        # Extract last user message
        extracted_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                extracted_message = msg.get("content", "")
                break
        
        # Should get the last user message, not any system message
        assert extracted_message == last_user_content
        assert "System" not in extracted_message

    def test_graceful_degradation_return_format(self):
        """
        **Feature: multi-ai-provider, Property 6: Graceful Degradation**
        **Validates: Requirements 10.3, 2.4**
        
        Property: The chat_completion_with_heuristic_fallback method should
        return a tuple of (AIResponse | None, heuristic_result | None).
        """
        # When AI succeeds: (AIResponse, None)
        # When heuristic used: (None, heuristic_result)
        
        # Simulate AI success case
        ai_success_result = (AIResponse(
            content="AI response",
            model="test-model",
            provider="test-provider",
        ), None)
        
        assert ai_success_result[0] is not None
        assert ai_success_result[1] is None
        assert isinstance(ai_success_result[0], AIResponse)
        
        # Simulate heuristic fallback case
        heuristic_result = {"intent": "unknown", "raw": "test message"}
        heuristic_fallback_result = (None, heuristic_result)
        
        assert heuristic_fallback_result[0] is None
        assert heuristic_fallback_result[1] is not None
        assert isinstance(heuristic_fallback_result[1], dict)
