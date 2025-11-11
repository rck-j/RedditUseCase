import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import praw
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError


def _require_env(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise RuntimeError(f"Missing {key}; define it in .env or your shell.")
    return value


load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
PROMPTS_PATH = Path("config/prompts.json")
DEFAULT_PROMPTS = {
    "initial_assessment": (
        "You are an automation strategist. Review the Reddit post details and "
        "decide if it hints at a potential automation use case for a small "
        "business or entrepreneur. Respond with 'YES - <short rationale>' or "
        "'NO - <short rationale>'."
    ),
    "deep_assessment": (
        "You now have the full Reddit submission (post body plus sampled comments). "
        "Assess whether a meaningful automation could help the author or community. "
        "Focus on friction, repetition, or coordination gaps and evaluate with the "
        "Move, Minimally ethos so computers do the work. Populate the structured "
        "fields as follows: automation_summary (one sentence verdict), deep_analysis "
        "(2-3 sentences with details), automation_complexity (one of: low, medium, "
        "high), and required_tools (list of concrete tools, APIs, or agent skills "
        "needed)."
    ),
}

reddit = praw.Reddit(
    client_id=_require_env("PRAW_CLIENT_ID"),
    client_secret=_require_env("PRAW_CLIENT_SECRET"),
    user_agent=_require_env("PRAW_USER_AGENT"),
)
openai_client = OpenAI(api_key=_require_env("OPENAI_API_KEY"))


def _load_prompts(path: Path) -> Dict[str, str]:
    """Load prompts from JSON, layering on top of defaults."""
    prompts = DEFAULT_PROMPTS.copy()
    try:
        with path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
        for key, value in data.items():
            if isinstance(value, str) and key in prompts:
                prompts[key] = value
    except FileNotFoundError:
        pass
    except json.JSONDecodeError as exc:
        print(f"Warning: could not parse prompts file ({exc}); using defaults.")
    return prompts


PROMPTS = _load_prompts(PROMPTS_PATH)


class AutomationInsight(BaseModel):
    automation_summary: str
    deep_analysis: str
    automation_complexity: str
    required_tools: List[str] = Field(default_factory=list)


STRUCTURED_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "automation_insight",
        "schema": AutomationInsight.model_json_schema(),
        "strict": True,
    },
}


def _responses_parse(**kwargs):
    """Wrapper that ensures the OpenAI client exposes responses.parse."""
    parse_method = getattr(openai_client.responses, "parse", None)
    if parse_method is None:
        raise RuntimeError(
            "openai client does not support responses.parse; upgrade the openai package."
        )
    return parse_method(**kwargs)


def _truncate(text: str, limit: int = 800) -> str:
    """Trim long strings so prompts stay small."""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def assess_automation_use_case(post: Dict[str, Any]) -> str:
    """Use OpenAI to judge whether a post implies an automation opportunity."""
    prompt = PROMPTS["initial_assessment"]
    post_details = (
        f"Subreddit: {post['subreddit']}\n"
        f"Title: {post['title']}\n"
        f"Score: {post['score']} | Comments: {post['num_comments']}\n"
        f"Permalink: {post['url']}"
    )
    combined_input = f"{prompt}\n\n{post_details}"
    try:
        response = openai_client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {
                    "role": "user",
                    "content": combined_input,
                }
            ],
            max_output_tokens=150,
        )
        content = response.output[0].content[0].text.strip()
    except Exception as exc:
        content = f"Assessment unavailable ({exc})"
    return content


def analyze_full_post_use_case(full_post: Dict[str, Any]) -> AutomationInsight:
    """Take full submission + comments and look for automation potential."""
    prompt = PROMPTS["deep_assessment"]
    comment_snippets: List[str] = []
    for comment in full_post.get("comments", [])[:5]:
        snippet = _truncate(comment.get("body", ""), 240)
        comment_snippets.append(f"- {snippet}")
    comments_section = "\n".join(comment_snippets) or "No comments captured."
    body = _truncate(full_post.get("selftext") or "", 1200)
    combined_input = (
        f"{prompt}\n\n"
        f"Title: {full_post.get('title')}\n"
        f"Body:\n{body or '[no body]'}\n\n"
        f"Top Comments:\n{comments_section}"
    )
    try:
        response = _responses_parse(
            model=OPENAI_MODEL,
            input=[{"role": "user", "content": combined_input}],
            max_output_tokens=350,
            response_format=STRUCTURED_RESPONSE_FORMAT,
        )
        raw = response.output[0].content[0].text.strip()
        payload = json.loads(raw)
        return AutomationInsight.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        return AutomationInsight(
            automation_summary="Deep assessment failed to parse response.",
            deep_analysis=f"Parsing error: {exc}",
            automation_complexity="unknown",
            required_tools=[],
        )
    except Exception as exc:  # OpenAI or network issue
        return AutomationInsight(
            automation_summary="Deep assessment unavailable.",
            deep_analysis=f"OpenAI error: {exc}",
            automation_complexity="unknown",
            required_tools=[],
        )


def fetch_full_post_with_praw(reddit_client, url_or_id, max_comments=5):
    """
    Returns a dict with the submission fields and a list of top comments.
    - reddit_client: your praw.Reddit() instance
    - url_or_id: either a full reddit URL or reddit id like 'abc123'
    - max_comments: max number of comments to return (None for all)
    """
    # create submission either by id or url
    if url_or_id.startswith("http"):
        submission = reddit_client.submission(url=url_or_id)
    else:
        submission = reddit_client.submission(id=url_or_id)

    # fetch main fields
    post = {
        "id": submission.id,
        "subreddit": str(submission.subreddit),
        "title": submission.title,
        "selftext": submission.selftext,        # full post body for text posts
        "author": str(submission.author) if submission.author else None,
        "created_utc": submission.created_utc,
        "score": submission.score,
        "url": submission.url,
        "is_self": submission.is_self,
        "num_comments": submission.num_comments,
        "media": submission.media,              # may be None or a dict for media posts
    }

    # load comments: be careful: replace_more(limit=None) returns ALL comments (can be heavy)
    submission.comments.replace_more(limit=0)  # expand top-level comments only (faster)
    comments = []
    for i, c in enumerate(submission.comments.list()):
        if max_comments is not None and i >= max_comments:
            break
        comments.append({
            "id": c.id,
            "author": str(c.author) if c.author else None,
            "body": c.body,
            "created_utc": c.created_utc,
            "score": c.score,
            "parent_id": c.parent_id,
        })

    post["comments"] = comments
    return post


def write_report(report_rows: List[Dict[str, Any]]) -> None:
    """Persist report incrementally so each assessment is saved as it completes."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report_payload = {
        "generated_at": RUN_STARTED_AT,
        "query": query,
        "time_filter": time_filter,
        "total_posts": len(report_rows),
        "posts": report_rows,
    }
    with REPORT_PATH.open("w", encoding="utf-8") as fp:
        json.dump(report_payload, fp, indent=2)

# subs = ["smallbusiness","Entrepreneur","AI_Agents","marketing","sales","automation"]
subs = ["smallbusiness","Entrepreneur"]
query = '(agent OR "ai agent" OR agentic OR automation) (small business OR smb OR entrepreneur)'
time_filter = "month"  # day|week|month|year|all
REPORT_PATH = Path("data/report.json")
RUN_STARTED_AT = datetime.now(timezone.utc).isoformat()

results = []
for sub in subs:
    subreddit = reddit.subreddit(sub)
    for post in subreddit.search(query, sort="new", time_filter=time_filter, limit=50):
        results.append({
            "subreddit": sub,
            "title": post.title,
            "url": f"https://www.reddit.com{post.permalink}",
            "created_utc": post.created_utc,
            "score": post.score,
            "num_comments": post.num_comments
        })


print(f"Found {len(results)} posts")
report_rows: List[Dict[str, Any]] = []
for r in results:
    created = datetime.fromtimestamp(
        r["created_utc"], tz=timezone.utc
    ).strftime('%Y-%m-%d %H:%M:%S')
    automation_callout = assess_automation_use_case(r)
    insight = AutomationInsight(
        automation_summary=automation_callout,
        deep_analysis="Not assessed (initial review not affirmative)",
        automation_complexity="unknown",
        required_tools=[],
    )
    if automation_callout.upper().startswith("YES"):
        try:
            full_post = fetch_full_post_with_praw(reddit, r["url"])
            insight = analyze_full_post_use_case(full_post)
        except Exception as exc:
            insight = AutomationInsight(
                automation_summary=automation_callout,
                deep_analysis=f"Deep assessment unavailable ({exc})",
                automation_complexity="unknown",
                required_tools=[],
            )

    print(
        f"[{r['subreddit']}] {created} | {r['score']} | {r['num_comments']} | "
        f"{r['title']} | {r['url']} | Automation: {insight.automation_summary}"
    )
    tools_display = ", ".join(insight.required_tools) if insight.required_tools else "n/a"
    print(
        f"    Deep dive: {insight.deep_analysis} "
        f"(Complexity: {insight.automation_complexity}, Tools: {tools_display})"
    )
    report_rows.append({
        **r,
        "created": created,
        "automation_summary": insight.automation_summary,
        "deep_analysis": insight.deep_analysis,
        "automation_complexity": insight.automation_complexity,
        "required_tools": insight.required_tools,
    })
    write_report(report_rows)
    print(f"Report updated ({len(report_rows)} entries) -> {REPORT_PATH}")
