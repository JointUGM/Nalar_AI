from typing import Any, ClassVar


class NalarAIError(Exception):
    """Base error. The HTTP layer renders code, message and details in the error envelope."""

    code: ClassVar[str] = "internal_error"
    status_code: ClassVar[int] = 500

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details or {}


class UnauthorizedError(NalarAIError):
    code = "unauthorized"
    status_code = 401


class InvalidInputError(NalarAIError):
    code = "invalid_input"
    status_code = 422


class DocumentUnreadableError(NalarAIError):
    code = "document_unreadable"
    status_code = 422


class SectionTooLargeError(NalarAIError):
    code = "section_too_large"
    status_code = 422


class PayloadTooLargeError(NalarAIError):
    code = "payload_too_large"
    status_code = 413


class BudgetExceededError(NalarAIError):
    code = "budget_exceeded"
    status_code = 422


class OutputValidationError(NalarAIError):
    code = "ai_output_invalid"
    status_code = 502


class ModelCallRejectedError(NalarAIError):
    code = "model_call_rejected"
    status_code = 502


class UpstreamUnavailableError(NalarAIError):
    code = "upstream_unavailable"
    status_code = 503


class ConfigurationError(NalarAIError):
    code = "configuration_error"
    status_code = 500
