from pathlib import Path
import platformdirs
import shutil
import requests
from requests.adapters import BaseAdapter
from requests.exceptions import ConnectionError
from requests.models import Response
from urllib.parse import urlparse
from typing import Optional
import hashlib

# --- Cache Management ---

def get_app_cache_dir(app_name: str) -> Path:
    """Returns the cache directory for a given application name."""
    return Path(platformdirs.user_cache_dir(appname=app_name))

def clear_cache(app_name: str, version: Optional[str] = None):
    """
    Removes the entire cache directory for a given application,
    or a specific version's cache.
    """
    cache_dir = get_app_cache_dir(app_name)
    if version:
        cache_dir = cache_dir / version
        
    if cache_dir.exists():
        print(f"Clearing cache for '{app_name}' version '{version}' at {cache_dir}...")
        shutil.rmtree(cache_dir)
        print("Cache cleared.")
    else:
        print(f"No cache found for '{app_name}' version '{version}' to clear.")

# --- Local File Transport Adapter ---

class LocalFileAdapter(BaseAdapter):
    """
    A requests TransportAdapter for handling file:// URLs, with proper streaming support.
    """
    def send(self, request, stream=False, timeout=None, verify=True, cert=None, proxies=None):
        """
        Sends the request by opening a local file, supporting both streaming and non-streaming.
        """
        parsed_url = urlparse(request.url)
        if parsed_url.scheme != 'file':
            raise ConnectionError(f"LocalFileAdapter received a non-file scheme: {parsed_url.scheme}")

        path_str = parsed_url.path
        if parsed_url.netloc:
            path_str = parsed_url.netloc + path_str
        
        path = Path(path_str)

        if not path.is_file():
            raise ConnectionError(f"File not found at local path: {path}")

        response = Response()
        response.url = request.url
        response.request = request
        response.status_code = 200
        response.reason = 'OK'

        try:
            if stream:
                # For streaming requests, open the file and attach the handle for later use
                file_handle = path.open('rb')
                response.raw = file_handle
                response.close = file_handle.close
            else:
                # For non-streaming requests, read the content immediately and close the file
                response._content = path.read_bytes()
                response.raw = None  # No raw stream needed
        except Exception as e:
            response.status_code = 500
            response.reason = str(e)
            response.raw = None
            response._content = None

        return response

    def close(self):
        pass

# --- Hashing ---
def calculate_file_sha256(filepath: Path, chunk_size: int = 4096) -> str:
    """Calculates the SHA256 hash of a file."""
    
    #- assert filepath is a Path
    assert isinstance(filepath, Path), "Expected filepath to be a Path object"

    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()