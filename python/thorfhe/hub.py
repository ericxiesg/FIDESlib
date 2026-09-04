"""Fetching model weights and datasets from Hugging Face, through a proxy, into a local cache.

The target machine reaches the internet through a proxy and should not re-download 400 MB of weights
on every run, so everything goes through :class:`HubClient`: it resolves a proxy from the command line
or the environment, keeps files under a cache directory keyed by repo and revision, and can be told to
work entirely offline once the cache is warm.

Deliberately depends on nothing but ``requests`` (with a ``urllib`` fallback). ``transformers`` and
``datasets`` are not installed on the benchmark machine and pulling them in would drag ``torch`` along
with them; the two things they would be used for here - reading a checkpoint and reading MRPC - are a
file format and an HTTP endpoint, and are handled in :mod:`thorfhe.checkpoint` and below.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
import urllib.parse
from pathlib import Path

DEFAULT_ENDPOINT = "https://huggingface.co"
#: The datasets-server returns rows as JSON, which saves a parquet reader and a pyarrow dependency.
DEFAULT_ROWS_ENDPOINT = "https://datasets-server.huggingface.co"
DEFAULT_CACHE = Path.home() / ".cache" / "thorfhe"
#: The rows endpoint caps a single request; larger splits are paged.
ROWS_PAGE = 100


def resolve_proxy(explicit: str | None = None) -> dict | None:
    """A ``requests``-shaped proxy mapping, from ``--proxy`` or the usual environment variables.

    ``--proxy`` wins; otherwise ``HTTPS_PROXY`` / ``HTTP_PROXY`` / ``ALL_PROXY`` are read in that
    order, in either case. Returns ``None`` when there is no proxy to use, which is not the same as an
    empty mapping - an empty mapping tells ``requests`` to ignore the environment.
    """
    if explicit:
        return {"http": explicit, "https": explicit}
    for name in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        value = os.environ.get(name)
        if value:
            return {"http": value, "https": value}
    return None


def _human(size: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{size:.0f} B"
        size /= 1024
    return f"{size:.1f} GiB"


class HubError(RuntimeError):
    pass


class HubClient:
    """Downloads with a proxy, a cache, and retries. One instance per benchmark run."""

    def __init__(self, *, cache_dir=None, proxy: str | None = None, token: str | None = None,
                 endpoint: str | None = None, rows_endpoint: str | None = None,
                 offline: bool = False, timeout: float = 30.0, retries: int = 3, quiet: bool = False):
        self.cache = Path(cache_dir or os.environ.get("THORFHE_CACHE") or DEFAULT_CACHE)
        self.proxies = resolve_proxy(proxy)
        self.token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        self.endpoint = (endpoint or os.environ.get("HF_ENDPOINT") or DEFAULT_ENDPOINT).rstrip("/")
        self.rows_endpoint = (rows_endpoint or os.environ.get("HF_ROWS_ENDPOINT")
                              or DEFAULT_ROWS_ENDPOINT).rstrip("/")
        self.offline = offline or os.environ.get("HF_HUB_OFFLINE") == "1"
        self.timeout = timeout
        self.retries = retries
        self.quiet = quiet

    # ------------------------------------------------------------------ plumbing
    def _log(self, message):
        if not self.quiet:
            print(message, file=sys.stderr, flush=True)

    def _headers(self):
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def _open(self, url, *, stream=False):
        """One GET, returning ``(status, headers, body-or-stream)``. requests if present, else urllib."""
        try:
            import requests
        except ImportError:
            requests = None

        if requests is not None:
            response = requests.get(url, headers=self._headers(), proxies=self.proxies,
                                    timeout=self.timeout, stream=stream, allow_redirects=True)
            return response.status_code, response.headers, response

        import urllib.request
        handlers = []
        if self.proxies:
            handlers.append(urllib.request.ProxyHandler(self.proxies))
        opener = urllib.request.build_opener(*handlers)
        request = urllib.request.Request(url, headers=self._headers())
        try:
            response = opener.open(request, timeout=self.timeout)
        except Exception as error:  # urllib raises on 4xx/5xx; normalise to a status
            status = getattr(error, "code", 0)
            return status, {}, error
        return response.status, dict(response.headers), response

    def _download(self, url, destination: Path, *, label: str):
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".part")

        last_error = None
        for attempt in range(1, self.retries + 1):
            try:
                status, headers, body = self._open(url, stream=True)
                if 400 <= status < 500:
                    # a missing or forbidden file will not appear on a retry
                    raise HubError(f"HTTP {status} for {url}", )
                if status != 200:
                    raise HubError(f"HTTP {status} for {url}")
                total = int(headers.get("Content-Length") or headers.get("content-length") or 0)
                self._log(f"  downloading {label} ({_human(total) if total else 'unknown size'})"
                          f"{' via proxy' if self.proxies else ''}")

                written = 0
                with open(partial, "wb") as handle:
                    if hasattr(body, "iter_content"):
                        chunks = body.iter_content(chunk_size=1 << 20)
                    else:
                        chunks = iter(lambda: body.read(1 << 20), b"")
                    for chunk in chunks:
                        if not chunk:
                            continue
                        handle.write(chunk)
                        written += len(chunk)
                if total and written != total:
                    raise HubError(f"truncated: {written} of {total} bytes")
                partial.replace(destination)
                return destination
            except Exception as error:                       # noqa: BLE001 - retried below
                last_error = error
                partial.unlink(missing_ok=True)
                if isinstance(error, HubError) and " 4" in str(error).split("HTTP")[-1][:3]:
                    break                                    # 4xx is permanent
                if attempt < self.retries:
                    delay = 2 ** attempt
                    self._log(f"  {label}: {error} - retrying in {delay}s "
                              f"({attempt}/{self.retries})")
                    time.sleep(delay)
        raise HubError(f"could not fetch {url}: {last_error}")

    # ------------------------------------------------------------------ models
    def file(self, repo: str, filename: str, *, revision: str = "main",
             repo_type: str = "model") -> Path:
        """One file from a repo, cached at ``<cache>/<repo_type>s/<repo>/<revision>/<filename>``."""
        destination = self.cache / f"{repo_type}s" / repo.replace("/", "--") / revision / filename
        if destination.exists() and destination.stat().st_size > 0:
            return destination
        if self.offline:
            raise HubError(f"offline and {destination} is not cached")

        prefix = "" if repo_type == "model" else f"{repo_type}s/"
        url = f"{self.endpoint}/{prefix}{repo}/resolve/{revision}/{urllib.parse.quote(filename)}"
        return self._download(url, destination, label=f"{repo}/{filename}")

    def first_file(self, repo: str, candidates, *, revision: str = "main",
                   repo_type: str = "model") -> Path:
        """The first of ``candidates`` that exists. Checkpoints come as safetensors *or* as a pickle."""
        errors = []
        for name in candidates:
            try:
                return self.file(repo, name, revision=revision, repo_type=repo_type)
            except HubError as error:
                errors.append(f"{name}: {error}")
        raise HubError(f"none of {list(candidates)} found in {repo}\n  " + "\n  ".join(errors))

    # ------------------------------------------------------------------ datasets
    def rows(self, dataset: str, config: str, split: str, *, limit: int | None = None) -> list:
        """Rows from the datasets-server, cached as one JSON file per (dataset, config, split).

        The cache holds whatever was fetched first; asking for more rows than are cached re-fetches.
        """
        key = f"{dataset.replace('/', '--')}--{config}--{split}.json"
        destination = self.cache / "datasets" / key
        if destination.exists():
            cached = json.loads(destination.read_text(encoding="utf-8"))
            if limit is None or len(cached["rows"]) >= limit or cached.get("complete"):
                rows = cached["rows"]
                return rows[:limit] if limit else rows
        if self.offline:
            raise HubError(f"offline and {destination} is not cached")

        collected, offset, complete = [], 0, False
        while limit is None or len(collected) < limit:
            want = ROWS_PAGE if limit is None else min(ROWS_PAGE, limit - len(collected))
            url = (f"{self.rows_endpoint}/rows?dataset={urllib.parse.quote(dataset)}"
                   f"&config={urllib.parse.quote(config)}&split={urllib.parse.quote(split)}"
                   f"&offset={offset}&length={want}")
            status, _, body = self._open(url)
            if status != 200:
                raise HubError(f"HTTP {status} for {url}")
            payload = json.loads(body.text if hasattr(body, "text") else body.read().decode("utf-8"))
            page = [entry["row"] for entry in payload.get("rows", [])]
            collected.extend(page)
            self._log(f"  {dataset}/{config}/{split}: {len(collected)} rows")
            offset += len(page)
            if len(page) < want:
                complete = True
                break

        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps({"rows": collected, "complete": complete}),
                               encoding="utf-8")
        return collected

    # ------------------------------------------------------------------ diagnostics
    def describe(self) -> str:
        proxy = next(iter(self.proxies.values())) if self.proxies else "none"
        return (f"cache {self.cache}\n  endpoint {self.endpoint}\n  proxy {proxy}"
                f"\n  offline {self.offline}")

    def clear(self):
        """Drop the whole cache. Only used by ``--clear-cache``."""
        if self.cache.exists():
            shutil.rmtree(self.cache)
