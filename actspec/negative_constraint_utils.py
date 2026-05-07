"""
负约束工具：论文定义的两类负约束常量与从 trace 推断子类型的逻辑，
以及用于生成/匹配动作历史前缀指纹（action_history_prefix）的辅助函数。

两类负约束（保守抽取）：
1. Readiness Constraint：页面尚未 ready 时就执行了动作 → 禁止在该未稳定状态下执行 π(a, tgt)
2. Disambiguation Constraint：多个相似 target 反复选错 → 禁止在该上下文中使用某类 target descriptor
"""

from typing import Any, Dict, List, Optional

MAX_ACTION_HISTORY_PREFIX_LEN = 3


def _normalize_action_type_from_str(action: Any) -> Optional[str]:
    """
    从轨迹中的 action 字符串解析出标准化的动作类型（小写）。
    例如 "click [79]" -> "click", "goto url=..." -> "goto"。
    """
    if not action:
        return None
    s = str(action).strip().lower()
    if not s:
        return None
    
    i = 0
    while i < len(s) and not s[i].isalpha():
        i += 1
    if i >= len(s):
        return None
    j = i
    while j < len(s) and s[j].isalpha():
        j += 1
    t = s[i:j]
    if not t:
        return None
    return t


def build_action_history_prefix(
    trajectory: List[Dict[str, Any]],
    segment_start: int,
    max_prefix_len: int = MAX_ACTION_HISTORY_PREFIX_LEN,
) -> List[str]:
    """
    从完整轨迹中提取片段前最多 N 步的动作类型序列，作为 action_history_prefix。

    - 仅使用动作类型（click/type/scroll/goto/go_back/go_forward/note/stop 等）的顺序信息；
    - 不记录 element_id/text/url 等参数；
    - 若 segment_start <= 1，则视为轨迹开头，不记录历史（返回空列表）。
    """
    if not trajectory or segment_start is None:
        return []
    
    if segment_start <= 1:
        return []

    types: List[str] = []
    
    for idx in range(segment_start - 1, -1, -1):
        step = trajectory[idx]
        if not isinstance(step, dict):
            continue
        action = step.get("action")
        t = _normalize_action_type_from_str(action)
        if not t:
            continue
        types.append(t)
        if len(types) >= max_prefix_len:
            break

    if not types:
        return []
    
    return list(reversed(types))



CONSTRAINT_SUBTYPE_READINESS = "readiness"
CONSTRAINT_SUBTYPE_DISAMBIGUATION = "disambiguation"
CONSTRAINT_SUBTYPE_UNSPECIFIED = "unspecified"

VALID_CONSTRAINT_SUBTYPES = frozenset({
    CONSTRAINT_SUBTYPE_READINESS,
    CONSTRAINT_SUBTYPE_DISAMBIGUATION,
    CONSTRAINT_SUBTYPE_UNSPECIFIED,
})


def normalize_constraint_subtype(value: Any) -> str:
    """将任意值规范为有效的 constraint_subtype。"""
    if value in VALID_CONSTRAINT_SUBTYPES:
        return value
    s = (value or "").strip().lower()
    if s in ("readiness", "ready"):
        return CONSTRAINT_SUBTYPE_READINESS
    if s in ("disambiguation", "disambiguate"):
        return CONSTRAINT_SUBTYPE_DISAMBIGUATION
    return CONSTRAINT_SUBTYPE_UNSPECIFIED


def infer_constraint_subtype_from_trajectory(
    trajectory: List[Dict[str, Any]],
    segment_start: int,
    segment_end: int,
    failure_reason: str,
    actions_in_segment: List[str],
) -> str:
    """
    从轨迹片段推断负约束子类型（保守：仅当信号明确时返回 readiness/disambiguation）。

    用于在离线从 trace 生成负约束时，在 LLM 返回 unspecified 或未区分时做启发式补充。

    Args:
        trajectory: 完整轨迹，每步含 url, action, observation 等
        segment_start: 片段起始步（含）
        segment_end: 片段结束步（含）
        failure_reason: 已判定的失败原因文本
        actions_in_segment: 片段内标准化后的 action 字符串列表

    Returns:
        CONSTRAINT_SUBTYPE_READINESS | CONSTRAINT_SUBTYPE_DISAMBIGUATION | CONSTRAINT_SUBTYPE_UNSPECIFIED
    """
    if not trajectory or segment_start < 0 or segment_end >= len(trajectory):
        return CONSTRAINT_SUBTYPE_UNSPECIFIED

    reason_lower = (failure_reason or "").lower()
    actions_lower = [a.lower() for a in actions_in_segment if a]

    
    
    url_changed_in_or_before_segment = False
    prev_url = None
    for i in range(max(0, segment_start - 1), min(segment_end + 1, len(trajectory))):
        step = trajectory[i] if isinstance(trajectory[i], dict) else {}
        url = (step.get("url") or "").strip()
        if prev_url is not None and url and url != prev_url:
            url_changed_in_or_before_segment = True
            break
        if url and url != "about:blank":
            prev_url = url
    if segment_start > 0:
        prev_step = trajectory[segment_start - 1] if isinstance(trajectory[segment_start - 1], dict) else {}
        action_str = str(prev_step.get("action") or "").lower()
        if "goto" in action_str or "go_back" in action_str or "go_forward" in action_str:
            url_changed_in_or_before_segment = True

    has_click_or_type = any(
        "click" in a or "type" in a or "scroll" in a
        for a in actions_lower
    )
    readiness_keywords = (
        "未就绪", "未准备好", "还没加载", "页面未", "transition", "refresh",
        "no_page_change", "target_not_interactable", "无效果", "没有变化"
    )
    reason_suggests_readiness = any(k in reason_lower for k in readiness_keywords)

    if url_changed_in_or_before_segment and has_click_or_type and reason_suggests_readiness:
        return CONSTRAINT_SUBTYPE_READINESS

    
    
    click_count = sum(1 for a in actions_lower if "click" in a)
    type_count = sum(1 for a in actions_lower if "type" in a)
    disambiguation_keywords = (
        "歧义", "选错", "多个相似", "相似 target", "错误目标", "locate_multiple",
        "multiple candidates", "consistent incorrect", "反复"
    )
    reason_suggests_disambiguation = any(k in reason_lower for k in disambiguation_keywords)

    if (click_count >= 2 or type_count >= 2) and reason_suggests_disambiguation:
        return CONSTRAINT_SUBTYPE_DISAMBIGUATION

    return CONSTRAINT_SUBTYPE_UNSPECIFIED


def build_unstable_state_for_readiness(
    trajectory: List[Dict[str, Any]],
    segment_start: int,
    segment_end: int,
) -> Dict[str, Any]:
    """
    为 Readiness 约束构建 context.unstable_state 描述（ϕ(Δ)）。
    保守：仅当能推断出时返回非空。

    Returns:
        例如 {"type": "url_transition"} 或 {"type": "modal_or_region_refresh"} 或 {}
    """
    if not trajectory or segment_start < 0:
        return {}
    
    prev_idx = segment_start - 1
    if prev_idx < 0:
        return {}
    prev = trajectory[prev_idx] if isinstance(trajectory[prev_idx], dict) else {}
    action_str = str(prev.get("action") or "").lower()
    if "goto" in action_str or "go_back" in action_str or "go_forward" in action_str:
        return {"type": "url_transition"}
    return {"type": "unstable"}  
