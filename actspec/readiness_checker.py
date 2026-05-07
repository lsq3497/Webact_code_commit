"""
Readiness 检查：step 的 target（element_id）是否可交互。
不引入新页面变化、不做 scroll，仅基于当前观测状态判断。
"""

from typing import Any, Tuple


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


def check_readiness(env: Any, element_id: str) -> Tuple[bool, str]:
    """
    检查 target（element_id）是否可交互：存在、非 disabled、非 readonly 等。
    不调用 scroll、不点击、不输入。
    返回 (ready: bool, reason: str)。
    """
    if not element_id:
        return False, "element_id 为空"
    proc = _get_observation_processor_from_env(env)
    if not proc or not getattr(proc, "obs_nodes_info", None):
        return False, "无法获取 obs_nodes_info"
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
        return False, "element_id 不在当前页面观测中"
    
    if hasattr(proc, "get_node_info_by_element_id"):
        try:
            node = proc.get_node_info_by_element_id(int(element_id))
        except (ValueError, TypeError):
            node = None
        if node is not None and hasattr(node, "properties") and node.properties:
            props = node.properties
            if props.get("disabled") is True:
                return False, "元素为 disabled"
            if props.get("readonly") is True:
                
                pass
    return True, ""
