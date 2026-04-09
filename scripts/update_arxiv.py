#!/usr/bin/env python3

from __future__ import annotations

import html
import json
import re
import ssl
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
ARXIV_DIR = ROOT / "arxiv"
CONFIG_PATH = ARXIV_DIR / "config.json"
FEED_PATH = ARXIV_DIR / "feed.json"
META_PATH = ARXIV_DIR / "meta.json"
HTML_PATH = ARXIV_DIR / "index.html"
STYLE_PATH = ARXIV_DIR / "styles.css"

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_iso8601(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def squash_whitespace(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def shorten(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    trimmed = text[: limit - 1].rsplit(" ", 1)[0].strip()
    return f"{trimmed}..."


def escape(text: str) -> str:
    return html.escape(text, quote=True)


def phrase_present(text_blob: str, phrase: str) -> bool:
    pattern = r"\b" + re.escape(phrase.lower()) + r"\b"
    return re.search(pattern, text_blob) is not None


def fetch_entries(category: str, max_results: int, repo_url: str) -> list[dict[str, Any]]:
    params = urllib.parse.urlencode(
        {
            "search_query": f"cat:{category}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "lastUpdatedDate",
            "sortOrder": "descending",
        }
    )
    request = urllib.request.Request(
        f"{ARXIV_API_URL}?{params}",
        headers={
            "User-Agent": (
                "sirichen2-arxiv-radar/1.0 "
                f"(GitHub Pages: {repo_url}; contact: GitHub issue)"
            )
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return parse_feed(response.read())
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, ssl.SSLCertVerificationError):
            # Some local Python installations ship without an up-to-date CA bundle.
            # The arXiv API is public and read-only, so we retry once without
            # certificate verification instead of failing the whole daily refresh.
            insecure_context = ssl._create_unverified_context()
            with urllib.request.urlopen(request, timeout=60, context=insecure_context) as response:
                return parse_feed(response.read())
        raise


def parse_feed(payload: bytes) -> list[dict[str, Any]]:
    root = ET.fromstring(payload)
    papers: list[dict[str, Any]] = []

    for entry in root.findall("atom:entry", ATOM_NS):
        authors = [
            squash_whitespace(author.findtext("atom:name", default="", namespaces=ATOM_NS))
            for author in entry.findall("atom:author", ATOM_NS)
        ]
        categories = sorted(
            {
                item.attrib["term"]
                for item in entry.findall("atom:category", ATOM_NS)
                if item.attrib.get("term")
            }
        )
        abs_url = squash_whitespace(entry.findtext("atom:id", default="", namespaces=ATOM_NS)).replace(
            "http://", "https://", 1
        )
        pdf_url = ""
        for link in entry.findall("atom:link", ATOM_NS):
            title = link.attrib.get("title", "")
            href = link.attrib.get("href", "")
            if title == "pdf" and href:
                pdf_url = href
                break
        if not pdf_url and abs_url:
            pdf_url = abs_url.replace("/abs/", "/pdf/")

        primary_category = entry.find("arxiv:primary_category", ATOM_NS)
        papers.append(
            {
                "id": abs_url.rsplit("/", 1)[-1],
                "title": squash_whitespace(entry.findtext("atom:title", default="", namespaces=ATOM_NS)),
                "summary": squash_whitespace(entry.findtext("atom:summary", default="", namespaces=ATOM_NS)),
                "authors": [author for author in authors if author],
                "abs_url": abs_url,
                "pdf_url": pdf_url,
                "published": parse_iso8601(
                    entry.findtext("atom:published", default="", namespaces=ATOM_NS)
                ).astimezone(timezone.utc),
                "updated": parse_iso8601(
                    entry.findtext("atom:updated", default="", namespaces=ATOM_NS)
                ).astimezone(timezone.utc),
                "comment": squash_whitespace(
                    entry.findtext("arxiv:comment", default="", namespaces=ATOM_NS)
                ),
                "journal_ref": squash_whitespace(
                    entry.findtext("arxiv:journal_ref", default="", namespaces=ATOM_NS)
                ),
                "categories": categories,
                "primary_category": primary_category.attrib.get("term", "") if primary_category is not None else "",
            }
        )

    return papers


def score_paper(
    paper: dict[str, Any], keywords: list[str], watched_authors: list[str]
) -> tuple[list[str], list[str], int]:
    text_blob = " ".join(
        [paper["title"], paper["summary"], " ".join(paper["authors"])]
    ).lower()
    author_blob = " ".join(paper["authors"]).lower()

    matched_keywords = [
        keyword
        for keyword in keywords
        if phrase_present(text_blob, keyword)
    ]
    matched_authors = [
        author
        for author in watched_authors
        if author.lower() in author_blob
    ]
    score = len(matched_keywords) + 2 * len(matched_authors)
    return matched_keywords, matched_authors, score


def serialize_paper(
    paper: dict[str, Any],
    timezone_name: str,
    summary_char_limit: int,
) -> dict[str, Any]:
    local_zone = ZoneInfo(timezone_name)
    updated_local = paper["updated"].astimezone(local_zone)
    published_local = paper["published"].astimezone(local_zone)
    return {
        **paper,
        "published": paper["published"].isoformat(),
        "updated": paper["updated"].isoformat(),
        "published_local": published_local.strftime("%Y-%m-%d %H:%M"),
        "updated_local": updated_local.strftime("%Y-%m-%d %H:%M"),
        "summary_short": shorten(paper["summary"], summary_char_limit),
    }


def render_paper_card(paper: dict[str, Any]) -> str:
    badges = [
        f'<span class="badge">{escape(source["category"])}</span>'
        for source in paper["sources"]
    ]
    if paper["matched_keywords"]:
        badges.extend(
            f'<span class="badge accent">{escape(keyword)}</span>'
            for keyword in paper["matched_keywords"][:4]
        )
    if paper["matched_authors"]:
        badges.extend(
            f'<span class="badge warm">{escape(author)}</span>'
            for author in paper["matched_authors"][:3]
        )

    comment_block = (
        f'<p class="paper-note">{escape(paper["comment"])}</p>'
        if paper["comment"]
        else ""
    )
    journal_block = (
        f'<p class="paper-journal">Journal ref: {escape(paper["journal_ref"])}</p>'
        if paper["journal_ref"]
        else ""
    )
    hero_mark = '<span class="paper-flag">Matched to your radar</span>' if paper["score"] > 0 else ""

    return f"""
        <article class="paper-card">
            <div class="paper-head">
                <div>
                    {hero_mark}
                    <h2><a href="{escape(paper["abs_url"])}" target="_blank" rel="noopener noreferrer">{escape(paper["title"])}</a></h2>
                </div>
                <a class="pdf-link" href="{escape(paper["pdf_url"])}" target="_blank" rel="noopener noreferrer">PDF</a>
            </div>
            <div class="paper-meta">
                <span>Updated {escape(paper["updated_local"])} SGT</span>
                <span>{escape(", ".join(paper["authors"][:6]))}</span>
            </div>
            <div class="paper-badges">{"".join(badges)}</div>
            <p class="paper-summary">{escape(paper["summary_short"])}</p>
            {comment_block}
            {journal_block}
        </article>
    """


def render_html(payload: dict[str, Any]) -> str:
    meta = payload["meta"]
    papers = payload["papers"]
    site = payload["site"]
    featured = [paper for paper in papers if paper["score"] > 0][:8]
    latest = papers[: payload["display_limit"]]
    keywords = payload["keywords"]
    authors = payload["authors"]
    errors = meta["errors"]

    featured_markup = "".join(render_paper_card(paper) for paper in featured) or """
        <div class="empty-state">
            <p>No keyword or author matches landed in the latest batch yet.</p>
            <p>Edit <code>arxiv/config.json</code> to tune your radar.</p>
        </div>
    """

    latest_markup = "".join(render_paper_card(paper) for paper in latest)
    keyword_markup = "".join(f'<span class="chip">{escape(keyword)}</span>' for keyword in keywords)
    author_markup = "".join(f'<span class="chip warm">{escape(author)}</span>' for author in authors) or '<span class="chip muted">No watched authors yet</span>'
    source_markup = "".join(
        f"""
            <li>
                <span>{escape(source["title"])}</span>
                <code>{escape(source["category"])}</code>
                <strong>{source["max_results"]}</strong>
            </li>
        """
        for source in meta["sources"]
    )
    error_markup = ""
    if errors:
        error_markup = f"""
            <section class="notice">
                <h2>Sync note</h2>
                <p>The latest run completed with fallback data for at least one source.</p>
                <ul>{"".join(f"<li>{escape(error)}</li>" for error in errors)}</ul>
            </section>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{escape(site["title"])}</title>
    <meta name="description" content="{escape(site["subtitle"])}">
    <meta name="robots" content="index,follow,max-image-preview:large,max-snippet:-1,max-video-preview:-1">
    <meta name="author" content="Yitong Zhang">
    <link rel="canonical" href="{escape(site["base_url"])}">
    <meta property="og:type" content="website">
    <meta property="og:title" content="{escape(site["title"])}">
    <meta property="og:description" content="{escape(site["subtitle"])}">
    <meta property="og:url" content="{escape(site["base_url"])}">
    <meta property="og:image" content="https://sirichen2.github.io/assets/profile.png">
    <meta name="twitter:card" content="summary_large_image">
    <meta name="twitter:title" content="{escape(site["title"])}">
    <meta name="twitter:description" content="{escape(site["subtitle"])}">
    <link rel="icon" type="image/svg+xml" href="/favicon.svg?v=2">
    <link rel="stylesheet" href="styles.css">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=Newsreader:opsz,wght@6..72,500;6..72,700&display=swap" rel="stylesheet">
</head>
<body>
    <main class="page-shell">
        <section class="hero">
            <div class="hero-copy">
                <p class="eyebrow">GitHub Pages x arXiv x daily commit</p>
                <h1>{escape(site["title"])}</h1>
                <p class="hero-text">{escape(site["subtitle"])}</p>
                <div class="hero-actions">
                    <a class="primary-button" href="{escape(site["profile_url"])}" target="_blank" rel="noopener noreferrer">GitHub profile</a>
                    <a class="ghost-button" href="/" rel="noopener noreferrer">Back to homepage</a>
                </div>
            </div>
            <div class="hero-panel">
                <div class="stat-card">
                    <span class="stat-label">Status</span>
                    <strong>{escape(meta["status"].capitalize())}</strong>
                </div>
                <div class="stat-card">
                    <span class="stat-label">Last build</span>
                    <strong>{escape(meta["generated_at_local"])}</strong>
                    <small>Singapore time</small>
                </div>
                <div class="stat-card">
                    <span class="stat-label">Total papers</span>
                    <strong>{meta["paper_count"]}</strong>
                    <small>{meta["featured_count"]} matched your radar</small>
                </div>
            </div>
        </section>

        {error_markup}

        <section class="snapshot-grid">
            <article class="snapshot-card">
                <h2>Tracked sources</h2>
                <ul class="source-list">{source_markup}</ul>
            </article>
            <article class="snapshot-card">
                <h2>Keyword radar</h2>
                <div class="chips">{keyword_markup}</div>
            </article>
            <article class="snapshot-card">
                <h2>Watched authors</h2>
                <div class="chips">{author_markup}</div>
            </article>
        </section>

        <section class="section-block">
            <div class="section-head">
                <p class="section-kicker">Priority queue</p>
                <h2>Matched to your keywords</h2>
            </div>
            <div class="paper-grid">{featured_markup}</div>
        </section>

        <section class="section-block">
            <div class="section-head">
                <p class="section-kicker">Fresh pull</p>
                <h2>Latest papers across your tracked fields</h2>
            </div>
            <div class="paper-grid">{latest_markup}</div>
        </section>

        <footer class="page-footer">
            <p>Generated from the public arXiv API. Source repo: <a href="{escape(site["repo_url"])}" target="_blank" rel="noopener noreferrer">{escape(site["repo_url"])}</a></p>
            <p>Last refreshed at {escape(meta["generated_at_local"])} SGT ({escape(meta["generated_at_utc"])} UTC).</p>
        </footer>
    </main>
</body>
</html>
"""


def build_payload() -> dict[str, Any]:
    config = read_json(CONFIG_PATH, {})
    site = config["site"]
    timezone_name = site.get("timezone", "Asia/Singapore")
    now_utc = datetime.now(timezone.utc)
    now_local = now_utc.astimezone(ZoneInfo(timezone_name))
    previous_feed = read_json(FEED_PATH, {"papers": []})
    previous_papers = previous_feed.get("papers", [])
    previous_by_id = {paper["id"]: paper for paper in previous_papers}

    keywords = config.get("keywords", [])
    watched_authors = config.get("authors", [])
    summary_char_limit = int(config.get("summary_char_limit", 420))
    merged: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    succeeded_sources = 0

    for source in config.get("sources", []):
        try:
            entries = fetch_entries(
                source["category"],
                int(source["max_results"]),
                site["repo_url"],
            )
            succeeded_sources += 1
        except Exception as exc:  # noqa: BLE001
            errors.append(f'{source["category"]}: {exc}')
            entries = []

        for entry in entries:
            matched_keywords, matched_authors, score = score_paper(
                entry,
                keywords,
                watched_authors,
            )
            paper = merged.setdefault(
                entry["id"],
                {
                    **entry,
                    "sources": [],
                    "matched_keywords": [],
                    "matched_authors": [],
                    "score": 0,
                },
            )
            paper["sources"].append(
                {
                    "category": source["category"],
                    "title": source["title"],
                }
            )
            paper["matched_keywords"] = sorted(
                set(paper["matched_keywords"]) | set(matched_keywords)
            )
            paper["matched_authors"] = sorted(
                set(paper["matched_authors"]) | set(matched_authors)
            )
            paper["score"] = max(paper["score"], score)

    if not merged and previous_by_id:
        for paper in previous_papers:
            merged[paper["id"]] = {
                **paper,
                "published": parse_iso8601(paper["published"]),
                "updated": parse_iso8601(paper["updated"]),
            }

    papers = list(merged.values())
    papers.sort(key=lambda item: (item["score"], item["updated"]), reverse=True)
    serialized_papers = [
        serialize_paper(paper, timezone_name, summary_char_limit)
        for paper in papers
    ]

    if succeeded_sources == 0 and previous_papers:
        status = "stale"
    elif errors:
        status = "partial"
    else:
        status = "fresh"

    sources_counter: dict[str, int] = defaultdict(int)
    for paper in serialized_papers:
        for source in paper["sources"]:
            sources_counter[source["category"]] += 1

    meta = {
        "status": status,
        "generated_at_utc": now_utc.strftime("%Y-%m-%d %H:%M"),
        "generated_at_local": now_local.strftime("%Y-%m-%d %H:%M"),
        "paper_count": len(serialized_papers),
        "featured_count": sum(1 for paper in serialized_papers if paper["score"] > 0),
        "errors": errors,
        "sources": [
            {
                **source,
                "visible_papers": sources_counter[source["category"]],
            }
            for source in config.get("sources", [])
        ],
    }

    return {
        "site": site,
        "display_limit": int(config.get("display_limit", 28)),
        "summary_char_limit": summary_char_limit,
        "keywords": keywords,
        "authors": watched_authors,
        "meta": meta,
        "papers": serialized_papers,
    }


def main() -> None:
    payload = build_payload()
    write_json(FEED_PATH, payload)
    write_json(META_PATH, payload["meta"])
    HTML_PATH.write_text(render_html(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
