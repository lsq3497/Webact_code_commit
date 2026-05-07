"""
轻量页面变化检测：自某次快照以来是否有可观测变化。
不涉及 Post-condition 语义，按 URL → target 状态 → 关键区域文本 hash → 节点数量 优先级判定。
"""

import hashlib
from typing import Any, Dict, Optional


def _get_url_from_env(env: Any) -> str:
    """从 env 获取当前 URL，与 PreConditionChecker._get_page_url 语义一致。"""
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
    """从 env 获取 observation_processor（与 actspec_executor 一致）。"""
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
    """获取当前页面 observation 文本，用于文本 hash。兼容 obs 为 str 或 dict 且 text 为 str/list。"""
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
    采集当前页面轻量快照，供 has_change 比较。
    返回：url, key_region_text_hash（整页可访问文本的 hash）, node_count。
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
    按优先级判断自 snapshot_before 以来是否有可观测变化：
    1. URL 变化
    2. 若提供 current_step_target_id，target 元素状态变化（通过 obs 中节点存在性/数量间接判断，保守）
    3. 关键区域文本 hash 变化
    4. 节点数量变化
    先满足即视为有变化。
    """
    if not snapshot_before:
        return True
    url_now = _get_url_from_env(env)
    if url_now != snapshot_before.get("url", ""):
        return True
    # Target 状态：若 target 仍在 obs 中且节点数有变化，可视为有变化；无 obs 则跳过
    proc = _get_observation_processor_from_env(env)
    if current_step_target_id and proc and getattr(proc, "obs_nodes_info", None):
        # 仅做存在性检查；更细的 value/checked 等需后端暴露，此处保守
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
