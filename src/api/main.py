"""FastAPI Application Entrypoint for AstroFoundation."""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("astrofoundation.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for application startup and shutdown events."""
    logger.info("Initializing AstroFoundation Multi-Modal API server...")
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

# Mount core routes
app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
