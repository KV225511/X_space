"""FastAPI Application Entrypoint for AstroFoundation."""

import logging
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure .env configuration is loaded before routing/middleware initialization
load_dotenv()

from .routes import router
from .rate_limiter import RateLimitMiddleware, rate_limiter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("astrofoundation.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for application startup and shutdown events."""
    logger.info("Initializing AstroFoundation Multi-Modal API server...")
    logger.info(
        "Turn Rate Limiter active: %d requests per turn (window: %.1fs)",
        rate_limiter.max_requests,
        rate_limiter.window_seconds,
    )
    # Pre-warming or resource initializations can occur here
    yield
    logger.info("Shutting down AstroFoundation API server.")


app = FastAPI(
    title="AstroFoundation Multi-Modal AI API",
    description=(
        "Production-grade API for multi-modal astronomical event classification, "
        "autonomous LangGraph agent database verification, and LLM scientific report generation."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Enable CORS for frontend web client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enforce turn-based rate limiting (configured from env: RATE_LIMIT_PER_TURN)
app.add_middleware(RateLimitMiddleware)

# Mount core routes
app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
