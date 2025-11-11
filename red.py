"""Reddit automation opportunity analyzer."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Sequence,
    Tuple,
    Type,
    TypeVar,
)

import praw
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError


load_dotenv()


DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
PROMPTS_PATH = Path("config/prompts.json")


def _require_env(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise RuntimeError(f"Missing {key}; define it in .env or your shell.")
    return value


def _load_prompts(path: Path) -> Dict[str, str]:
    defaults = {
        "initial_assessment": (
            "You are an automation strategist supporting small businesses and "
            "entrepreneurs. Review the Reddit post details and decide whether it "
            "indicates a meaningful automation or agent opportunity. Return a "
            "binary decision and a short explanation."
        ),
        "deep_assessment": (
            "You now have the full Reddit submission (post body plus sampled "
            "comments). Assess whether an automation or agent solution could "
            "meaningfully help the author or community. Focus on friction, "
            "repetition, coordination gaps, and the Move, Minimally ethos so "
            "computers do the heavy lifting."
        ),
    }

    try:
        with path.open("r", encoding="utf-8") as handle:
            overrides = json.load(handle)
    except FileNotFoundError:
        return defaults
    except json.JSONDecodeError as exc:
        print(f"Warning: could not parse prompts file ({exc}); using defaults.")
        return defaults

    for key, value in overrides.items():
        if isinstance(value, str) and key in defaults:
            defaults[key] = value
    return defaults


PROMPTS = _load_prompts(PROMPTS_PATH)


class InitialAssessment(BaseModel):
    """Structured output for the initial title-only screen."""

    is_automation: bool = Field(
        ...,
        description=(
            "True when the post likely relates to an automation or agent use "
            "case that merits deeper analysis."
        ),
    )
    rationale: str = Field(..., description="Brief justification for the decision.")


class AutomationInsight(BaseModel):
    """Structured output for the full deep-dive analysis."""

    automation_summary: str
    deep_analysis: str
    automation_complexity: str
    required_tools: List[str] = Field(default_factory=list)


class PostReport(BaseModel):
    """Final report entry persisted to disk."""

    subreddit: str
    title: str
    url: str
    created: str
    score: int
    num_comments: int
    initial_assessment: InitialAssessment
    automation_insight: AutomationInsight


T = TypeVar("T", bound=BaseModel)


def _parse_as_model(
    client: OpenAI, *, model: str, prompt: str, text_format: Type[T]
) -> T:
    response = client.responses.parse(
        model=model,
        input=[{"role": "user", "content": prompt}],
        max_output_tokens=400,
        text_format=text_format,
    )
    if isinstance(response, text_format):
        return response

    parsed_payload = getattr(response, "parsed", None)
    if isinstance(parsed_payload, text_format):
        return parsed_payload
    if parsed_payload is not None:
        return text_format.model_validate(parsed_payload)

    output_parsed = getattr(response, "output_parsed", None)
    if isinstance(output_parsed, text_format):
        return output_parsed
    if output_parsed is not None:
        return text_format.model_validate(output_parsed)

    output_items = getattr(response, "output", None)
    if output_items is not None:
        for item in output_items:
            content_list = getattr(item, "content", None)
            if content_list is None:
                continue
            for content in content_list:
                parsed_value = getattr(content, "parsed", None)
                if isinstance(parsed_value, text_format):
                    return parsed_value
                if parsed_value is not None:
                    return text_format.model_validate(parsed_value)
    if hasattr(response, "model_dump"):
        dumped = response.model_dump()
        if isinstance(dumped, dict):
            try:
                return text_format.model_validate(dumped)
            except ValidationError:
                pass

    raise ValidationError(
        [
            {
                "type": "model_parsing",
                "loc": (text_format.__name__,),
                "msg": "Unable to locate parsed content in OpenAI response.",
                "input": response,
            }
        ],
        model=text_format,
    )


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def build_reddit_client() -> praw.Reddit:
    return praw.Reddit(
        client_id=_require_env("PRAW_CLIENT_ID"),
        client_secret=_require_env("PRAW_CLIENT_SECRET"),
        user_agent=_require_env("PRAW_USER_AGENT"),
    )


def build_openai_client() -> Tuple[OpenAI, str]:
    client = OpenAI(api_key=_require_env("OPENAI_API_KEY"))
    if not hasattr(client.responses, "parse"):
        raise RuntimeError(
            "openai client does not support responses.parse; upgrade the package."
        )
    return client, DEFAULT_OPENAI_MODEL


def summarize_post(post: praw.models.Submission) -> Dict[str, Any]:
    return {
        "subreddit": str(post.subreddit),
        "title": post.title,
        "url": f"https://www.reddit.com{post.permalink}",
        "created_utc": post.created_utc,
        "score": post.score,
        "num_comments": post.num_comments,
    }


def search_posts(
    reddit_client: praw.Reddit,
    *,
    subreddits: Sequence[str],
    query: str,
    time_filter: str,
    limit: int,
) -> Iterator[Dict[str, Any]]:
    for subreddit_name in subreddits:
        subreddit = reddit_client.subreddit(subreddit_name)
        for submission in subreddit.search(
            query, sort="new", time_filter=time_filter, limit=limit
        ):
            yield summarize_post(submission)


def fetch_full_post(
    reddit_client: praw.Reddit, url_or_id: str, *, max_comments: int
) -> Dict[str, Any]:
    if url_or_id.startswith("http"):
        submission = reddit_client.submission(url=url_or_id)
    else:
        submission = reddit_client.submission(id=url_or_id)

    post = {
        "id": submission.id,
        "subreddit": str(submission.subreddit),
        "title": submission.title,
        "selftext": submission.selftext,
        "author": str(submission.author) if submission.author else None,
        "created_utc": submission.created_utc,
        "score": submission.score,
        "url": submission.url,
        "is_self": submission.is_self,
        "num_comments": submission.num_comments,
        "media": submission.media,
    }

    submission.comments.replace_more(limit=0)
    comments = []
    for index, comment in enumerate(submission.comments.list()):
        if max_comments >= 0 and index >= max_comments:
            break
        comments.append(
            {
                "id": comment.id,
                "author": str(comment.author) if comment.author else None,
                "body": comment.body,
                "created_utc": comment.created_utc,
                "score": comment.score,
                "parent_id": comment.parent_id,
            }
        )

    post["comments"] = comments
    return post


@dataclass
class AnalyzerDependencies:
    reddit: praw.Reddit
    openai: OpenAI
    openai_model: str
    prompts: Dict[str, str]


class AutomationAnalyzer:
    def __init__(self, deps: AnalyzerDependencies) -> None:
        self._reddit = deps.reddit
        self._openai = deps.openai
        self._model = deps.openai_model
        self._prompts = deps.prompts

    def initial_assessment(self, post_summary: Dict[str, Any]) -> InitialAssessment:
        post_details = (
            f"Subreddit: {post_summary['subreddit']}\n"
            f"Title: {post_summary['title']}\n"
            f"Score: {post_summary['score']}\n"
            f"Comments: {post_summary['num_comments']}\n"
            f"Permalink: {post_summary['url']}"
        )
        prompt = (
            f"{self._prompts['initial_assessment']}\n\n"
            "Return structured data with fields is_automation (boolean) and "
            "rationale (short sentence)."
            f"\n\nPost Details:\n{post_details}"
        )
        try:
            return _parse_as_model(
                self._openai,
                model=self._model,
                prompt=prompt,
                text_format=InitialAssessment,
            )
        except ValidationError as exc:
            return InitialAssessment(
                is_automation=False,
                rationale=f"Parsing error: {exc}",
            )
        except Exception as exc:  # network or OpenAI error
            return InitialAssessment(
                is_automation=False,
                rationale=f"Assessment unavailable ({exc})",
            )

    def deep_assessment(self, full_post: Dict[str, Any]) -> AutomationInsight:
        comments = full_post.get("comments", [])[:5]
        comment_snippets = [
            f"- {_truncate(comment.get('body', ''), 240)}" for comment in comments
        ]
        comments_block = "\n".join(comment_snippets) or "No comments captured."
        body = _truncate(full_post.get("selftext") or "", 1200)
        prompt = (
            f"{self._prompts['deep_assessment']}\n\n"
            "Return structured data with fields automation_summary (one "
            "sentence), deep_analysis (two or three sentences), "
            "automation_complexity (one of: low, medium, high), and "
            "required_tools (list of tools or APIs)."
            f"\n\nTitle: {full_post.get('title')}\n"
            f"Body:\n{body or '[no body]'}\n\n"
            f"Top Comments:\n{comments_block}"
        )
        try:
            return _parse_as_model(
                self._openai,
                model=self._model,
                prompt=prompt,
                text_format=AutomationInsight,
            )
        except ValidationError as exc:
            return AutomationInsight(
                automation_summary="Deep assessment failed to parse response.",
                deep_analysis=f"Parsing error: {exc}",
                automation_complexity="unknown",
                required_tools=[],
            )
        except Exception as exc:
            return AutomationInsight(
                automation_summary="Deep assessment unavailable.",
                deep_analysis=f"OpenAI error: {exc}",
                automation_complexity="unknown",
                required_tools=[],
            )

    def analyze_post(
        self, post_summary: Dict[str, Any], *, comment_limit: int
    ) -> PostReport:
        initial = self.initial_assessment(post_summary)
        if initial.is_automation:
            try:
                full_post = fetch_full_post(
                    self._reddit,
                    post_summary["url"],
                    max_comments=comment_limit,
                )
                deep = self.deep_assessment(full_post)
            except Exception as exc:
                deep = AutomationInsight(
                    automation_summary=initial.rationale,
                    deep_analysis=f"Deep assessment unavailable ({exc})",
                    automation_complexity="unknown",
                    required_tools=[],
                )
        else:
            deep = AutomationInsight(
                automation_summary=initial.rationale,
                deep_analysis="Not assessed (initial review not affirmative).",
                automation_complexity="unknown",
                required_tools=[],
            )

        created = datetime.fromtimestamp(
            post_summary["created_utc"], tz=timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S")

        return PostReport(
            subreddit=post_summary["subreddit"],
            title=post_summary["title"],
            url=post_summary["url"],
            created=created,
            score=post_summary["score"],
            num_comments=post_summary["num_comments"],
            initial_assessment=initial,
            automation_insight=deep,
        )


def write_report(
    path: Path,
    *,
    generated_at: str,
    query: str,
    time_filter: str,
    posts: Sequence[PostReport],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": generated_at,
        "query": query,
        "time_filter": time_filter,
        "total_posts": len(posts),
    }
    payload["posts"] = [post.model_dump() for post in posts]
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _print_report_entry(report: PostReport) -> None:
    insight = report.automation_insight
    initial = report.initial_assessment
    tools = ", ".join(insight.required_tools) or "n/a"
    print(
        f"[{report.subreddit}] {report.created} | {report.score} | "
        f"{report.num_comments} | {report.title} | {report.url}"
    )
    print(
        f"    Initial Assessment: {'YES' if initial.is_automation else 'NO'} - "
        f"{initial.rationale}"
    )
    print(
        f"    Deep Dive: {insight.deep_analysis} "
        f"(Summary: {insight.automation_summary}; "
        f"Complexity: {insight.automation_complexity}; Tools: {tools})"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subs",
        nargs="+",
        default=["smallbusiness", "Entrepreneur"],
        help="Subreddits to search.",
    )
    parser.add_argument(
        "--query",
        default='(agent OR "ai agent" OR agentic OR automation) '
        "(small business OR smb OR entrepreneur)",
        help="Search query to submit to Reddit.",
    )
    parser.add_argument(
        "--time-filter",
        default="month",
        choices=["day", "week", "month", "year", "all"],
        help="Reddit time filter for the search.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum posts to retrieve per subreddit.",
    )
    parser.add_argument(
        "--comments-limit",
        type=int,
        default=5,
        help="Maximum comments to retrieve during deep analysis (-1 for all).",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("data/report.json"),
        help="Destination for the JSON report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reddit_client = build_reddit_client()
    openai_client, model_name = build_openai_client()
    deps = AnalyzerDependencies(
        reddit=reddit_client,
        openai=openai_client,
        openai_model=model_name,
        prompts=PROMPTS,
    )
    analyzer = AutomationAnalyzer(deps)

    generated_at = datetime.now(timezone.utc).isoformat()
    results: List[PostReport] = []
    for post_summary in search_posts(
        reddit_client,
        subreddits=args.subs,
        query=args.query,
        time_filter=args.time_filter,
        limit=args.limit,
    ):
        report = analyzer.analyze_post(
            post_summary, comment_limit=args.comments_limit
        )
        results.append(report)
        _print_report_entry(report)

    print(f"Processed {len(results)} posts.")
    write_report(
        args.report_path,
        generated_at=generated_at,
        query=args.query,
        time_filter=args.time_filter,
        posts=results,
    )
    print(f"Report saved to {args.report_path}")


if __name__ == "__main__":
    main()

