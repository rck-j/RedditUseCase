# Repository Guidelines

## Project Structure & Module Organization
The repository is intentionally lean: `red.py` at the root contains the Reddit scraping logic, while `bparser/` is a throwaway Python 3.13 virtual environment (keep it out of commits). Put additional scrapers under `src/` if you expand the project, mirroring their data helpers under `src/utils/` and writing fixtures in `tests/`. Temporary exports (CSV, JSON) belong in `data/` so they can be git-ignored together.

## Build, Test, and Development Commands
Create or refresh the local environment before running anything:
```bash
python3 -m venv bparser
source bparser/bin/activate
pip install praw python-dotenv
```
Run the collector with explicit config so results are reproducible:
```bash
PRAW_CLIENT_ID=xxx PRAW_CLIENT_SECRET=yyy PRAW_USER_AGENT="smb-agentic/0.1" python red.py
```
Keep secrets in `.env` or your shell profile; never hard-code them in `red.py`.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indents, 88-character lines, and snake_case for functions (`fetch_posts`). Reserve UPPER_SNAKE for module-level constants (e.g., `SUBS`). Prefer small pure functions that accept typed parameters and return plain dicts/lists so they can be reused in upcoming agents. Run `ruff check .` or `black red.py` before pushing if those tools are installed; keep formatting changes separate from feature commits.

## Testing Guidelines
There is no automated suite yet, so treat every change as an opportunity to add `pytest` cases under `tests/`. Start with fixture-driven tests that load canned subreddit responses serialized to JSON; assert that de-duplication, scoring, and query composition behave as expected. For manual checks, run `python red.py` with `time_filter="day"` so failures surface quickly, and capture at least one sample output in the PR description.

## Commit & Pull Request Guidelines
The repository has no shared Git history yet; adopt a concise imperative style such as `feat: add subreddit filter toggle` so logs stay scannable. Reference tracking tickets in the body when relevant. Pull requests should include: a one-paragraph summary, a checklist of verification steps (e.g., commands executed, new tests), and screenshots or sample console output when behavior changes. Mention any new environment variables or secrets so deployment agents can update their stores promptly.

## Security & Configuration Tips
Rotate Reddit credentials regularly and load them via environment variables accessed through `os.environ`. Before pushing, run `rg -n "client_secret"` to ensure no secrets slipped into commits. If you need to share results, scrub URLs that reveal private subreddits or user handles, and prefer storing sanitized exports inside `data/public/`.
