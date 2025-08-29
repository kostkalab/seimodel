from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional

from .core import (
    resolve_cache_dir,
    get_session,
    read_or_fetch_checksum,
    ensure_weight_file,
    load_state_dict_from_path,
)


class TorchModelLoaderMixin(nn.Module):
    """Mixin adding lazy remote/local weight loading with checksum verification.

    After instantiation call ``load_weights()`` to download (if needed), verify and
    load the state dict.
    """

    def __init__(self, *args,
                 base_url: str,
                 filename: str,
                 app_name: str,
                 version: str = "latest",
                 verify: bool = True,
                 expected_sha256: Optional[str] = None,
                 timeout: int = 30,
                 **kwargs):
        super().__init__(*args, **kwargs)
        # Basic validation early
        if not base_url:
            raise ValueError("base_url must be provided")
        if not filename:
            raise ValueError("filename must be provided")
        if not app_name:
            raise ValueError("app_name must be provided")

        self.loader_base_url = base_url.rstrip('/')
        self.loader_filename = filename
        self.loader_app_name = app_name
        self.loader_version = version
        self.loader_verify = verify
        self.loader_expected_sha256 = expected_sha256
        self.loader_timeout = timeout
        self._session = None  # lazy
        # Backwards compatibility: expose cache dir path early like previous versions
        self.loader_cache_dir = resolve_cache_dir(self.loader_app_name, self.loader_version)

    def load_weights(self, *, strict: bool = True):
        """Download (if needed), verify and load weights; returns self."""
        if self._session is None:
            self._session = get_session()

        cache_dir = self.loader_cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)  # ensure exists
        expected = read_or_fetch_checksum(
            base_url=self.loader_base_url,
            filename=self.loader_filename,
            cache_dir=cache_dir,
            session=self._session,
            expected=self.loader_expected_sha256,
            verify=self.loader_verify,
            timeout=self.loader_timeout,
        )
        weight_path = ensure_weight_file(
            base_url=self.loader_base_url,
            filename=self.loader_filename,
            cache_dir=cache_dir,
            session=self._session,
            expected_sha256=expected,
            verify=self.loader_verify,
            timeout=self.loader_timeout,
        )
        load_state_dict_from_path(self, weight_path, strict=strict)
        return self