import time
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.router import api_router
from app.core.config import settings
from app.core.constants import HEADER_LATENCY_MS, HEADER_REQUEST_ID
from app.core.exceptions import RouteMindException
from app.core.logging import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup Lifespan
    logger.info(
        f"Starting {settings.APP_NAME} v{settings.APP_VERSION} [Env: {settings.ENVIRONMENT}]",
        extra={"extra_data": {"free_only": settings.FREE_ONLY, "log_level": settings.LOG_LEVEL}},
    )

    # Free-tier runtime invariant assertion
    if settings.FREE_ONLY:
        logger.info("CRITICAL GUARD ACTIVE: FREE_ONLY=True. All paid routes and models strictly disabled.")

    yield

    # Shutdown Lifespan
    logger.info(f"Shutting down {settings.APP_NAME} gateway gracefully.")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Adaptive LLM Gateway with Intelligent Cost-, Quality-, Latency-, and Reliability-Aware Model Routing",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def correlation_and_latency_middleware(request: Request, call_next):
    """Injects correlation Request IDs and measures overall gateway processing latency."""
    request_id = request.headers.get(HEADER_REQUEST_ID, str(uuid.uuid4()))
    request.state.request_id = request_id

    start_time = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

    response.headers[HEADER_REQUEST_ID] = request_id
    response.headers[HEADER_LATENCY_MS] = str(duration_ms)
    return response


# Global Exception Handlers
@app.exception_handler(RouteMindException)
async def routemind_exception_handler(request: Request, exc: RouteMindException):
    req_id = getattr(request.state, "request_id", "unknown")
    logger.warning(
        f"RouteMind exception [{exc.error_code}]: {exc.message}",
        extra={"extra_data": {"request_id": req_id, "status_code": exc.status_code, "details": exc.details}},
    )
    return JSONResponse(status_code=exc.status_code, content=exc.to_dict())


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    req_id = getattr(request.state, "request_id", "unknown")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "message": "Request validation failed",
                "type": "VALIDATION_ERROR",
                "status_code": 422,
                "details": {"errors": exc.errors()},
            }
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    req_id = getattr(request.state, "request_id", "unknown")
    logger.error(
        f"Unhandled Gateway Error: {str(exc)}",
        exc_info=True,
        extra={"extra_data": {"request_id": req_id}},
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "message": "An unexpected internal gateway error occurred.",
                "type": "INTERNAL_SERVER_ERROR",
                "status_code": 500,
                "details": {},
            }
        },
    )


# Attach API router under configured prefix
app.include_router(api_router, prefix=settings.API_V1_STR)
