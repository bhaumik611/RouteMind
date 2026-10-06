from datetime import datetime, timezone
from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, Numeric, String
from sqlalchemy.orm import relationship

from app.core.database import Base


class ProviderModel(Base):
    __tablename__ = "providers"

    id = Column(String(32), primary_key=True)
    name = Column(String(64), nullable=False)
    is_enabled = Column(Boolean, default=True, nullable=False)
    is_free_only_compatible = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

    models = relationship("ModelCatalog", back_populates="provider", cascade="all, delete-orphan")


class ModelCatalog(Base):
    __tablename__ = "models"

    id = Column(String(128), primary_key=True)
    provider_id = Column(String(32), ForeignKey("providers.id", ondelete="CASCADE"), nullable=False, index=True)
    model_name = Column(String(128), nullable=False)
    is_free = Column(Boolean, default=True, nullable=False)
    context_window = Column(Integer, default=8192, nullable=False)
    supports_streaming = Column(Boolean, default=True, nullable=False)
    supports_structured_output = Column(Boolean, default=False, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    cost_per_1k_input_usd = Column(Float, default=0.0, nullable=False)
    cost_per_1k_output_usd = Column(Float, default=0.0, nullable=False)

    provider = relationship("ProviderModel", back_populates="models")
