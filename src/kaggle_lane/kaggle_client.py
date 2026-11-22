"""Minimal Kaggle API client using a new-style KGAT bearer token.

No username is required for the calls implemented here: the token identifies the account and
the server returns the owner reference in push/view responses.

Endpoints are the public Kaggle API v1 paths also used by the official CLI. They can change;
`push_smoke.py` is the integration test that proves the lane works before any real run.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

import requests

BASE_URL = "https://www.kaggle.com/api/v1"


class KaggleApiError(RuntimeError):
    def __init__(self, status: int, body: str, url: str):
        self.status = status
        self.body = body
        self.url = url
        super().__init__(f"Kaggle API {status} on {url}: {body[:800]}")


def load_token() -> str:
    """Read the token from the environment or the per-user token file (never from the repo)."""
    for name in ("KAGGLE_API_TOKEN", "KAGGLE_KEY"):
        value = os.environ.get(name)
        if value:
            return value.strip()
    candidates = [
        Path.home() / ".kaggle" / "kgat.txt",
        Path.home() / ".kaggle" / "access_token",
        Path.home() / ".kaggle" / "kaggle.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8").strip()
        if text.startswith("{"):
            data = json.loads(text)
            return str(data.get("key") or data.get("token") or "").strip()
        if text:
            return text.splitlines()[0].strip()
    raise RuntimeError(
        "No Kaggle token found. Set KAGGLE_API_TOKEN or write the token to ~/.kaggle/kgat.txt"
    )


def parse_ref(ref: str) -> tuple[str, str]:
    """Normalize a kernel ref returned by the API into ``(owner, slug)``.

    Observed formats: ``"/code/<owner>/<slug>"`` (new kernel push) and ``"<owner>/<slug>"``.
    """
    cleaned = (ref or "").strip().strip("/")
    if cleaned.startswith("code/"):
        cleaned = cleaned[len("code/"):]
    parts = [part for part in cleaned.split("/") if part]
    if len(parts) < 2:
        raise ValueError(f"cannot parse kernel ref: {ref!r}")
    return parts[-2], parts[-1]


class KaggleClient:
    def __init__(self, token: str | None = None):
        self.token = token or load_token()
        self.session = requests.Session()
        self.session.headers.update(
            {"Authorization": f"Bearer {self.token}", "Accept": "application/json"}
        )

    def _request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        url = path if path.startswith("http") else BASE_URL + path
        timeout = kwargs.pop("timeout", 120)
        response = self.session.request(method, url, timeout=timeout, **kwargs)
        if response.status_code >= 400:
            raise KaggleApiError(response.status_code, response.text, url)
        return response

    # --- account / sanity -------------------------------------------------
    def list_kernels(self, page_size: int = 5, mine: bool | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {"page": 1, "pageSize": page_size}
        if mine:
            params["mine"] = "true"
        return self._request("GET", "/kernels/list", params=params).json()

    # --- kernels ----------------------------------------------------------
    def push_kernel(
        self,
        code: str,
        slug: str,
        title: str | None = None,
        private: bool = True,
        enable_gpu: bool = True,
        enable_internet: bool = True,
        dataset_sources: Iterable[str] | None = None,
        kernel_sources: Iterable[str] | None = None,
        language: str = "python",
        kernel_type: str = "script",
    ) -> dict[str, Any]:
        payload = {
            # New API tokens push new kernels under "me/": a bare slug is rejected with
            # "Invalid slug". Discovered 2026-11-01 (owner: mysteriousavailable).
            "slug": slug if "/" in slug else f"me/{slug}",
            "newTitle": title or slug,
            "text": code,
            "language": language,
            "kernelType": kernel_type,
            "isPrivate": bool(private),
            "enableGpu": bool(enable_gpu),
            "enableInternet": bool(enable_internet),
            "datasetDataSources": list(dataset_sources or []),
            "kernelDataSources": list(kernel_sources or []),
            "competitionDataSources": [],
            "categoryIds": [],
        }
        return self._request("POST", "/kernels/push", json=payload, timeout=600).json()

    def view_kernel(self, owner: str, slug: str) -> dict[str, Any]:
        return self._request("GET", f"/kernels/view/{owner}/{slug}").json()

    def wait_for_kernel(
        self,
        owner: str,
        slug: str,
        timeout_s: int = 900,
        poll_s: int = 20,
    ) -> dict[str, Any]:
        deadline = time.time() + timeout_s
        last: dict[str, Any] = {}
        while time.time() < deadline:
            last = self.view_kernel(owner, slug)
            status = str(last.get("status", "")).lower()
            print(f"[kaggle] {owner}/{slug} status={status}")
            if status in {"complete", "error", "failed", "cancelled", "canceled"}:
                return last
            time.sleep(poll_s)
        return last

    def download_output(self, owner: str, slug: str, destination: Path) -> Path | None:
        """Download the kernel output archive. Returns the written path, if any."""
        destination.mkdir(parents=True, exist_ok=True)
        response = self._request(
            "GET", f"/kernels/output/{owner}/{slug}", stream=True, timeout=900
        )
        content_type = response.headers.get("Content-Type", "")
        if "application/json" in content_type:
            listing = response.json()
            (destination / "listing.json").write_text(
                json.dumps(listing, indent=2), encoding="utf-8"
            )
            return destination / "listing.json"
        out_path = destination / "output.zip"
        with out_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if chunk:
                    handle.write(chunk)
        return out_path


def main() -> None:
    client = KaggleClient()
    print("[kaggle] token accepted; first 5 kernels from the public list:")
    print(json.dumps(client.list_kernels(page_size=5), indent=2)[:1500])


if __name__ == "__main__":
    main()
