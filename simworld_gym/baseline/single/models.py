from typing import List, Optional, Dict, Any
from openai import AzureOpenAI, OpenAI
import os
import time
import traceback


class BaseModel:
    """Base model class for LLM generation with reasoning and non-reasoning modes."""
    
    def __init__(
        self,
        backend: str = "openai",
        api_key: str = "",
        azure_endpoint: str = "",
        api_version: str = "2025-04-01-preview",
        model: str = "gpt-4o",
        temperature: float = 0.2,
        reasoning_effort: str = "medium",
        use_reasoning: bool = False,
        max_output_tokens: int = 2048,
        top_p: float = 1.0,
        max_retries: int = 3,
        retry_backoff: float = 1.0,
    ):
        self.backend = backend
        self.api_key = api_key
        self.azure_endpoint = azure_endpoint
        self.api_version = api_version
        self.model = model
        print(f"Using model: {self.model}")
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.use_reasoning = use_reasoning
        self.max_output_tokens = max_output_tokens
        self.top_p = top_p
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.client = self._client_init()

    def _format_error(self, err: Exception) -> str:
        """Return a concise, debuggable error string with HTTP/status hints when available."""
        parts = [f"{err.__class__.__name__}: {err}"]
        status = getattr(err, "status_code", None) or getattr(err, "status", None)
        if status:
            parts.append(f"status={status}")
        code = getattr(err, "code", None)
        if code:
            parts.append(f"code={code}")
        # OpenAI HTTPResponseError exposes response / body
        resp = getattr(err, "response", None)
        if resp is not None:
            body = getattr(resp, "body", None) or getattr(resp, "text", None)
            if body:
                parts.append(f"body={body}")
        return " | ".join(parts)

    def _client_init(self):
        """Initialize the appropriate client based on backend."""
        if self.backend == "openai":
            return OpenAI(api_key=self.api_key if self.api_key else os.environ.get("OPENAI_API_KEY"))
        elif self.backend == "azure":
            return AzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.azure_endpoint,
                api_version=self.api_version,
            )
        elif self.backend == "gemini":
            return OpenAI(
                api_key=self.api_key if self.api_key else os.environ.get("GEMINI_API_KEY"),
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
        else:
            return OpenAI(api_key="EMPTY", base_url=self.backend)

    def _to_chat_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert Response-API style messages to chat.completions compatible format."""
        converted: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            if isinstance(content, list):
                new_content = []
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    t = item.get("type")
                    if t == "input_text":
                        new_content.append({"type": "text", "text": item.get("text", "")})
                    elif t == "input_image":
                        new_content.append({"type": "image_url", "image_url": {"url": item.get("image_url")}})
                    else:
                        # Already chat-style or unknown; pass through
                        new_content.append(item)
                converted.append({"role": role, "content": new_content})
            else:
                converted.append(msg)
        return converted

    def generation(
        self,
        messages: List[Dict[str, Any]],
        max_output_tokens: Optional[int] = None,
        enforce_json: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate response from the model.
        
        Args:
            messages: List of message dictionaries with role and content
            max_output_tokens: Optional override for max tokens
            
        Returns:
            Dictionary with:
                - output: The generated text
                - reason: Reasoning summary (if use_reasoning=True)
                - usage: Dict with input and output token counts
        """
        if self.use_reasoning:
            return self._generation_reasoning(messages, max_output_tokens, enforce_json=enforce_json)
        else:
            return self._generation_standard(messages, max_output_tokens, enforce_json=enforce_json)

    def _generation_standard(
        self,
        messages: List[Dict[str, Any]],
        max_output_tokens: Optional[int] = None,
        enforce_json: bool = False,
    ) -> Dict[str, Any]:
        """Standard generation with temperature control."""
        last_err = None
        chat_messages = self._to_chat_messages(messages)
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=chat_messages,
                    temperature=self.temperature,
                    max_tokens=max_output_tokens or self.max_output_tokens,
                    top_p=self.top_p,
                    response_format={"type": "json_object"} if enforce_json else None,
                )
                
                choice = response.choices[0]
                output_text = getattr(choice.message, "content", "") if choice else ""
                usage = {
                    "input": response.usage.prompt_tokens if response.usage else 0,
                    "output": response.usage.completion_tokens if response.usage else 0,
                }
                
                return {
                    "output": output_text,
                    "reason": "",  # No reasoning summary in standard mode
                    "usage": usage,
                }
            except Exception as e:
                last_err = e
                if attempt == self.max_retries - 1:
                    raise RuntimeError(
                        f"[generation] backend={self.backend} model={self.model} failed after {self.max_retries} attempts: "
                        f"{self._format_error(e)}"
                    ) from e
                time.sleep(self.retry_backoff * (2 ** attempt))

    def _generation_reasoning(
        self,
        messages: List[Dict[str, Any]],
        max_output_tokens: Optional[int] = None,
        enforce_json: bool = False,
    ) -> Dict[str, Any]:
        """Reasoning generation with effort parameter (for o1/o3/gpt-5 models)."""
        # chat.completions does not support the new "reasoning" API on all backends;
        # we still send temperature/top_p and rely on the model's defaults.
        for msg in messages:
            if msg["role"] == "system":
                msg["role"] = "user"
        last_err = None
        chat_messages = self._to_chat_messages(messages)
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=chat_messages,
                    temperature=self.temperature,
                    max_tokens=max_output_tokens or self.max_output_tokens,
                    top_p=self.top_p,
                    response_format={"type": "json_object"} if enforce_json else None,
                )

                choice = response.choices[0]
                output_text = getattr(choice.message, "content", "") if choice else ""
                reason_summary = ""  # chat.completions does not return reasoning summaries
                usage = {
                    "input": response.usage.prompt_tokens if response.usage else 0,
                    "output": response.usage.completion_tokens if response.usage else 0,
                }
                
                return {
                    "output": output_text,
                    "reason": reason_summary,
                    "usage": usage,
                }
            except Exception as e:
                last_err = e
                if attempt == self.max_retries - 1:
                    raise RuntimeError(
                        f"[generation(reasoning)] backend={self.backend} model={self.model} failed after {self.max_retries} attempts: "
                        f"{self._format_error(e)}"
                    ) from e
                time.sleep(self.retry_backoff * (2 ** attempt))
