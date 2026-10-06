from datetime import datetime, timezone
from sqlalchemy import BigInteger, Column, DateTime, Float, Integer, String
from sqlalchemy.types import JSON

from app.core.database import Base


class BanditModelState(Base):
    __tablename__ = "bandit_model_states"

    model_id = Column(String(128), primary_key=True, index=True)
    strategy = Column(String(32), default="linucb", nullable=False)
    feature_dimension = Column(Integer, default=10, nullable=False)
    matrix_a_json = Column(JSON, nullable=False)  # Covariance matrix (A = X^T X + I)
    vector_b_json = Column(JSON, nullable=False)  # Reward vector (b = X^T r)
    total_pulls = Column(BigInteger, default=0, nullable=False)
    cumulative_reward = Column(Float, default=0.0, nullable=False)
    last_updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
