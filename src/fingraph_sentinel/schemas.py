from datetime import datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PaymentEvent(BaseModel):
    """Canonical, pseudonymous payment event used by every downstream model."""

    model_config = ConfigDict(str_strip_whitespace=True)

    transaction_id: str = Field(min_length=1, max_length=128)
    event_time: datetime
    customer_id: str = Field(min_length=1, max_length=128)
    card_id: str = Field(min_length=1, max_length=128)
    merchant_id: str = Field(min_length=1, max_length=256)
    merchant_category_code: str | None = Field(default=None, max_length=16)
    amount: Decimal = Field(gt=0, max_digits=12, decimal_places=2)
    currency: str = Field(default="USD", min_length=3, max_length=3)
    merchant_city: str | None = Field(default=None, max_length=128)
    merchant_state: str | None = Field(default=None, max_length=128)
    merchant_country: str | None = Field(default=None, min_length=2, max_length=2)
    device_id: str | None = Field(default=None, max_length=256)
    ip_hash: str | None = Field(default=None, max_length=128)
    payment_channel: str | None = Field(default=None, max_length=64)

    @field_validator("currency")
    @classmethod
    def uppercase_currency(cls, value: str) -> str:
        return value.upper()


class RiskReason(BaseModel):
    feature: str
    direction: Literal["increases_risk", "reduces_risk", "context"]
    detail: str


class RiskDecision(BaseModel):
    transaction_id: str
    model_version: str
    fraud_probability: float = Field(ge=0, le=1)
    action: Literal["allow", "review", "hold"]
    reasons: list[RiskReason]
    is_model_ready: bool
