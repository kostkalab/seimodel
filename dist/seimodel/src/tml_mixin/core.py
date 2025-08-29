"""Core, dependency-light primitives for model weight downloading, caching and verification.

All higher-level APIs (mixin + functional loader + single-file export) should call ONLY
functions in this module to avoid logic duplication.
"""
from __future__ import annotations
from pathlib import Path
import os
import hashlib
from typing import Optional

import torch
import requests

from .utils import get_app_cache_dir, calculate_file_sha256, LocalFileAdapter

# --------------------------------------------------------------------------------------
# Session / cache helpers
# --------------------------------------------------------------------------------------

def get_session() -> requests.Session:
    """Return a shared session with file:// support mounted.

    A new session is created per call (cheap) – callers can cache if desired.
    """
    session = requests.Session()
    session.mount("file://", LocalFileAdapter())
    return session


def resolve_cache_dir(app_name: str, version: str) -> Path:
    """Return (and create) the versioned cache directory."""
    return get_app_cache_dir(app_name) / version

# --------------------------------------------------------------------------------------
# Checksum handling
# --------------------------------------------------------------------------------------

def read_or_fetch_checksum(*, base_url: str, filename: str, cache_dir: Path,
                           session: requests.Session, expected: Optional[str],
                           verify: bool, timeout: int = 30) -> Optional[str]:
    """Return expected SHA256.

    Precedence:
    1. If verify is False -> returns None.
    2. If expected provided -> use it directly (offline / pinned case).
    3. Else read cached <filename>.sha256 if present.
    4. Else download <base_url>/<filename>.sha256, cache and return.
    """
    if not verify:
        return None
    if expected:
        return expected

    checksum_path = cache_dir / f"{filename}.sha256"
    if checksum_path.exists():
        return checksum_path.read_text().split()[0].strip()

    checksum_url = f"{base_url}/{filename}.sha256"
    resp = session.get(checksum_url, timeout=timeout)
    resp.raise_for_status()
    checksum_path.write_text(resp.text)
    return resp.text.split()[0].strip()

# --------------------------------------------------------------------------------------
# Download + verification
# --------------------------------------------------------------------------------------

from tqdm import tqdm

def download_file_atomic(url: str, dest: Path, *, session: requests.Session,
                          timeout: int = 30, chunk_size: int = 8192, verbose: bool = True, show_progress: bool = True) -> None:
    """Download to a temporary file then atomically rename into place."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with session.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0))
            if show_progress and total:
                if verbose:
                    print(f"Downloading {url} to {dest} ({total} bytes)...")
                with tmp.open('wb') as f, tqdm(total=total, unit='B', unit_scale=True, desc=str(dest.name)) as pbar:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            else:
                if verbose:
                    print(f"Downloading {url} to {dest}...")
                with tmp.open('wb') as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
        tmp.replace(dest)
        if verbose:
            print(f"Download complete: {dest}")
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def ensure_weight_file(*, base_url: str, filename: str, cache_dir: Path,
                       session: requests.Session, expected_sha256: Optional[str],
                       verify: bool, timeout: int = 30) -> Path:
    """Ensure the weight file exists, downloading + verifying if necessary.

    Returns Path to the (verified) weight file.
    """
    weight_path = cache_dir / filename
    if weight_path.exists() and verify and expected_sha256:
        if calculate_file_sha256(weight_path) == expected_sha256:
            return weight_path
        # stale / mismatched
        weight_path.unlink(missing_ok=True)
    elif weight_path.exists() and not verify:
        return weight_path

    # Download fresh
    cache_dir.mkdir(parents=True, exist_ok=True)
    download_file_atomic(f"{base_url}/{filename}", weight_path, session=session, timeout=timeout)

    if verify and expected_sha256:
        actual = calculate_file_sha256(weight_path)
        if actual != expected_sha256:
            weight_path.unlink(missing_ok=True)
            raise ValueError(f"Checksum mismatch for {filename}: expected {expected_sha256} got {actual}")
    return weight_path

# --------------------------------------------------------------------------------------
# High level load
# --------------------------------------------------------------------------------------

def load_state_dict_from_path(model: torch.nn.Module, weight_path: Path, *, strict: bool = True) -> torch.nn.Module:
    state_dict = torch.load(weight_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict, strict=strict)
    model.eval()
    return model


def functional_load_model(model_cls, *, base_url: str, filename: str, app_name: str,
                          version: str = "latest", strict: bool = True,
                          expected_sha256: Optional[str] = None, verify: bool = True,
                          timeout: int = 30) -> torch.nn.Module:
    """Instantiate model_cls, fetch weights, load and return the model.

    This is the simplest inference-oriented API.
    """
    base_url = base_url.rstrip('/')
    cache_dir = resolve_cache_dir(app_name, version)
    session = get_session()
    cache_dir.mkdir(parents=True, exist_ok=True)
    expected = read_or_fetch_checksum(base_url=base_url, filename=filename, cache_dir=cache_dir,
                                      session=session, expected=expected_sha256, verify=verify,
                                      timeout=timeout)
    weight_path = ensure_weight_file(base_url=base_url, filename=filename, cache_dir=cache_dir,
                                     session=session, expected_sha256=expected, verify=verify,
                                     timeout=timeout)
    model = model_cls()
    return load_state_dict_from_path(model, weight_path, strict=strict)
