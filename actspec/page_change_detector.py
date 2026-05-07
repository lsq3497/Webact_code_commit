"""
Lightweight page-change detection: whether anything observable changed since a snapshot.
Not full post-condition semantics; priority: URL → target presence → text hash → node count.
"""

import hashlib
from typing import Any, Dict, Optional


def _get_url_from_env(env: Any) -> str:
    """Current URL from env (same idea as PreConditionChecker._get_page_url)."""
    if env is None:
        return ""
    if hasattr(env, "get_url"):
        try:
            return env.get_url() or ""
        except Exception:
            pass
    if hasattr(env, "url"):
        return getattr(env, "url", "") or ""
    if hasattr(env, "webarena_env") and hasattr(env.webarena_env, "info"):
        info = getattr(env.webarena_env, "info", {})
        if isinstance(info, dict) and "page" in info:
            p = info["page"]
            if hasattr(p, "url"):
                try:
                    return p.url or ""
                except Exception:
                    pass
    return ""


def _get_observation_processor_from_env(env: Any) -> Any:
    """Observation processor from env (same as actspec_executor)."""
    if not env:
        return None
    try:
        if hasattr(env, "webarena_env"):
            webarena_env = env.webarena_env
            if hasattr(webarena_env, "observation_handler"):
                handler = webarena_env.observation_handler
                if hasattr(handler, "text_processor"):
                    return handler.text_processor
            if hasattr(webarena_env, "info") and isinstance(webarena_env.info, dict):
                if "observation_processor" in webarena_env.info:
                    return webarena_env.info["observation_processor"]
    except Exception:
        pass
    return None


def _get_observation_text_from_env(env: Any) -> str:
    """Observation text for hashing; supports str obs or dict with text str/list."""
    if env is None:
        return ""
    if hasattr(env, "observation"):
        try:
            obs = env.observation()
            if isinstance(obs, str):
                return obs
            if isinstance(obs, dict) and "text" in obs:
                t = obs["text"]
                if isinstance(t, str):
                    return t
                if isinstance(t, (list, tuple)) and len(t) > 0:
                    return str(t[0])[:50000]
                return str(t)[:50000]
        except Exception:
            pass
    return ""


def take_snapshot(env: Any) -> Dict[str, Any]:
    """
    Take a lightweight snapshot for has_change: url, full-page text sha256, node_count.
    """
    url = _get_url_from_env(env)
    text = _get_observation_text_from_env(env)
    key_region_text_hash = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    node_count = 0
    proc = _get_observation_processor_from_env(env)
    if proc and getattr(proc, "obs_nodes_info", None):
        node_count = len(proc.obs_nodes_info)
    return {
        "url": url,
        "key_region_text_hash": key_region_text_hash,
        "node_count": node_count,
    }


def has_change(
    env: Any,
    snapshot_before: Dict[str, Any],
    current_step_target_id: Optional[str] = None,
) -> bool:
    """
    True if any of (in order): URL changed; target id presence/shift; text hash changed; node count changed.
    """
    if not snapshot_before:
        return True
    url_now = _get_url_from_env(env)
    if url_now != snapshot_before.get("url", ""):
        return True
    
    proc = _get_observation_processor_from_env(env)
    if current_step_target_id and proc and getattr(proc, "obs_nodes_info", None):
        
        pass
    text_now = _get_observation_text_from_env(env)
    hash_now = hashlib.sha256(text_now.encode("utf-8", errors="replace")).hexdigest()
    if hash_now != snapshot_before.get("key_region_text_hash", ""):
        return True
    node_count_now = 0
    if proc and getattr(proc, "obs_nodes_info", None):
        node_count_now = len(proc.obs_nodes_info)
    if node_count_now != snapshot_before.get("node_count", 0):
        return True
    return False
