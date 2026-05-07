"""
单步执行器：对单个 step 完成 Locate（由 executor 事前填入）→ Readiness → Action → 页面变化验证。
返回结构化 StepResult 供上层做失败判定与自动修复。
"""

from typing import Any, Callable, Dict, List, Optional, TypedDict

from . import page_change_detector
from . import readiness_checker


class StepResult(TypedDict, total=False):
    success: bool
    failure_reason: Optional[str]
    element_id: Optional[str]
    action_str: Optional[str]
    page_changed: bool


# 无页面变化时仍视为 step 成功的 primitive 白名单（readiness 通过且 action 未异常时）
NO_PAGE_CHANGE_ALLOWLIST_PRIMITIVES = frozenset({
    "FOCUS",      # 激活输入框类 click，通常无 DOM/URL 变化
    "CLICK",      # click on checkbox/radio 已是目标状态（幂等）；focus 类 click
    "TYPE",       # input where value == expected value
    "HOVER",      # hover / tooltip 未计入 DOM diff
})


def execute_step(
    step_idx: int,
    step: Dict[str, Any],
    plan: List[Dict[str, Any]],
    env: Any,
    plan_step_to_action: Callable[[Dict[str, Any]], Optional[str]],
    readiness_check_fn: Callable[[Any, str], tuple],
    page_change_detector_module: Any,
    locate_multiple_candidates_marker: str,
) -> StepResult:
    """
    执行单个 step：Readiness（若需要 element_id）→ Action → 页面变化验证。
    Locate 由上层在调用前完成，step["target"]["value"] 已为 element_id 或
    locate_multiple_candidates_marker 或空（表示 locate_empty）。
    """
    primitive = (step.get("primitive") or "").upper()
    target = step.get("target", {})
    strategy = target.get("strategy", "")
    value_raw = target.get("value")
    # 统一按字符串比较；value 可能来自绑定为 int（如 element_id）
    value_str = (str(value_raw).strip() if value_raw is not None else "")

    # 1. Locate 结果已由上层写入 step；此处仅做失败判定
    if strategy == "element_id":
        if value_str == locate_multiple_candidates_marker:
            return {
                "success": False,
                "failure_reason": "locate_multiple_candidates",
                "element_id": None,
                "action_str": None,
                "page_changed": False,
            }
        if value_raw is None or value_raw == "" or (isinstance(value_raw, str) and value_raw.strip() == ""):
            return {
                "success": False,
                "failure_reason": "locate_empty",
                "element_id": None,
                "action_str": None,
                "page_changed": False,
            }
        element_id = value_str
    else:
        element_id = None

    # 2. Readiness（仅当有 element_id 时）
    if element_id:
        ready, reason = readiness_check_fn(env, element_id)
        if not ready:
            return {
                "success": False,
                "failure_reason": "target_not_interactable",
                "element_id": element_id,
                "action_str": None,
                "page_changed": False,
            }

    # 3. Action
    action_str = plan_step_to_action(step)
    if not action_str:
        # 无 action 的 step（如 NOTE）视为成功、无页面变化
        return {
            "success": True,
            "failure_reason": None,
            "element_id": element_id,
            "action_str": None,
            "page_changed": False,
        }
    snapshot_before = page_change_detector_module.take_snapshot(env)
    try:
        env.step(action_str, is_actspec_internal=True)
    except Exception as e:
        return {
            "success": False,
            "failure_reason": "action_exception",
            "element_id": element_id,
            "action_str": action_str,
            "page_changed": False,
        }

    # 4. 页面变化验证
    current_step_target_id = element_id if strategy == "element_id" else None
    page_changed = page_change_detector_module.has_change(
        env, snapshot_before, current_step_target_id
    )
    if page_changed:
        return {
            "success": True,
            "failure_reason": None,
            "element_id": element_id,
            "action_str": action_str,
            "page_changed": True,
        }
    # 无页面变化：检查白名单
    if primitive in NO_PAGE_CHANGE_ALLOWLIST_PRIMITIVES:
        return {
            "success": True,
            "failure_reason": None,
            "element_id": element_id,
            "action_str": action_str,
            "page_changed": False,
        }
    return {
        "success": False,
        "failure_reason": "no_page_change",
        "element_id": element_id,
        "action_str": action_str,
        "page_changed": False,
    }
