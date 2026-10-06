from fastapi import APIRouter
from app.api.v1 import completions, health, metrics_api, models, providers, routing, tenants, usage

api_router = APIRouter()

# Diagnostics & Observability
api_router.include_router(health.router, tags=["Health & Diagnostics"])
api_router.include_router(metrics_api.router, tags=["Observability & Metrics"])

# Catalog & Management
api_router.include_router(models.router, tags=["Models"])
api_router.include_router(providers.router, tags=["Providers"])
api_router.include_router(tenants.router, tags=["Multi-Tenancy & RBAC"])

# Core Routing & Completions
api_router.include_router(completions.router, tags=["Chat Completions"])

# Introspection & Telemetry
api_router.include_router(usage.router, tags=["Usage & Quotas"])
api_router.include_router(routing.router, tags=["Routing Introspection"])
