"""
Readiness: whether the step target (element_id) is interactable.
No scrolling, clicks, or typing—judgment uses current observation only.
"""

from typing import Any, Tuple


def _get_observation_processor_from_env(env: Any) -> Any:
    """Get observation_processor from env (same path as actspec_executor)."""
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


def check_readiness(env: Any, element_id: str) -> Tuple[bool, str]:
    """
    Check whether target element_id is interactable (present, not disabled, etc.).
    Returns (ready, reason).
    """
    if not element_id:
        return False, "element_id is empty"
    proc = _get_observation_processor_from_env(env)
    if not proc or not getattr(proc, "obs_nodes_info", None):
        return False, "obs_nodes_info unavailable"
    obs_nodes_info = proc.obs_nodes_info
    
    key = str(element_id) if str(element_id) in obs_nodes_info else None
    if key is None:
        try:
            n = int(element_id)
            if str(n) in obs_nodes_info:
                key = str(n)
            elif n in obs_nodes_info:
                key = n
        except (ValueError, TypeError):
            pass
    if key is None:
        return False, "element_id not in current observation"
    
    if hasattr(proc, "get_node_info_by_element_id"):
        try:
            node = proc.get_node_info_by_element_id(int(element_id))
        except (ValueError, TypeError):
            node = None
        if node is not None and hasattr(node, "properties") and node.properties:
            props = node.properties
            if props.get("disabled") is True:
                return False, "element is disabled"
            if props.get("readonly") is True:
                
                pass
    return True, ""
