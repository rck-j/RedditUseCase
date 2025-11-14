"""Web dashboard for viewing Reddit automation reports."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, ValidationError

try:
    from red import PostReport as BasePostReport
except ImportError:  # pragma: no cover - fallback for standalone execution
    class InitialAssessment(BaseModel):
        """Structured output for the initial title-only screen."""

        is_automation: bool = Field(...)
        rationale: str = Field(...)

    class AutomationInsight(BaseModel):
        """Structured output for the full deep-dive analysis."""

        automation_summary: str
        deep_analysis: str
        automation_complexity: str
        required_tools: List[str] = Field(default_factory=list)

    class BasePostReport(BaseModel):
        subreddit: str
        title: str
        url: str
        created: str
        score: int
        num_comments: int
        initial_assessment: InitialAssessment
        automation_insight: AutomationInsight


class PostReport(BasePostReport):
    """Local alias to ensure consistent validation."""


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data" / "report.json"
TEMPLATES = Jinja2Templates(directory=str(ROOT_DIR / "templates"))

app = FastAPI(title="Reddit Automation Report Dashboard")


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:  # pragma: no cover - runtime protection
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except json.JSONDecodeError as exc:  # pragma: no cover - runtime protection
        raise HTTPException(status_code=500, detail="Invalid report JSON") from exc


def _parse_reports(items: Iterable[Dict[str, Any]]) -> List[PostReport]:
    reports: List[PostReport] = []
    for item in items:
        try:
            reports.append(PostReport.model_validate(item))
        except ValidationError as exc:  # pragma: no cover - runtime protection
            raise HTTPException(status_code=500, detail=str(exc)) from exc
    return reports


@lru_cache(maxsize=1)
def get_reports() -> List[PostReport]:
    """Load and parse report entries from disk."""

    payload = _load_json(DATA_PATH)
    posts = payload.get("posts")
    if posts is None:
        if isinstance(payload, list):
            posts = payload
        else:
            raise HTTPException(status_code=500, detail="Report payload missing 'posts'.")
    if not isinstance(posts, list):
        raise HTTPException(status_code=500, detail="Report posts should be a list.")
    return _parse_reports(posts)


@app.get("/api/reports", response_model=List[PostReport])
def read_reports() -> List[PostReport]:
    """Return the parsed report entries as JSON."""

    return get_reports()


@app.get("/", response_class=HTMLResponse)
def render_dashboard(request: Request) -> HTMLResponse:
    """Render the dashboard shell; client fetches data via HTMX."""

    return TEMPLATES.TemplateResponse(
        "report_dashboard.html",
        {"request": request},
    )


__all__ = ["app", "get_reports", "PostReport"]
