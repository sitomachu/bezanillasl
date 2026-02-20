from __future__ import annotations

# Backward-compatible shim. New code lives in src.ingestion.services.request_service.
from src.idealistaAPI.ingestion.services.request_service import add_common_args, run_new, run_resume_latest_rent

__all__ = ["add_common_args", "run_new", "run_resume_latest_rent"]
