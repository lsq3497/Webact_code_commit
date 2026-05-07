"""
ActSpec 离线复用评估器：
- 从单个任务的日志中提取 ActSpec 调用记录
- 结合库中 ActSpec 定义和执行前后状态，判断本次调用是否“成功复用”
- 聚合为 per-action_id 的 usage/success/fail 统计
- 通过 ActSpecLibrary.update_stats_batch 写回库，并按规则禁用不可靠的 ActSpec

注意：
- 目前优先使用在线执行阶段记录的 pre/post-condition 结果；
- 如需更强的语义判断，可在 _llm_judge_call 中扩展更复杂的 prompt。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

from llms import lm_config
from llms import utils as llm_utils

from .actspec_library import ActSpecLibrary


def _load_actspec_map(library: ActSpecLibrary, library_path: str) -> Dict[str, Dict[str, Any]]:
    """将指定库路径下的 ActSpec 加载为 {action_id: actspec} 映射。"""
    actspecs = library.load_library(library_path)
    mapping: Dict[str, Dict[str, Any]] = {}
    for spec in actspecs:
        aid = spec.get("action_id")
        if aid:
            mapping[aid] = spec
    return mapping


def _llm_judge_call(
    lm_cfg: lm_config.LMConfig,
    actspec: Dict[str, Any],
    call_record: Dict[str, Any],
) -> bool:
    """
    使用 LLM 对单次 ActSpec 调用是否“成功复用”做一个语义判断。
    当前实现尽量简单：给出 ActSpec 的 effect / post_condition 和执行前后文本。
    返回 True 表示成功，False 表示失败。
    """
    # 如果没有前后文本，退化为使用 executor 的结果
    pre_text = call_record.get("pre_text") or ""
    post_text = call_record.get("post_text") or ""
    if not pre_text and not post_text:
        return bool(call_record.get("executor_success"))

    description = actspec.get("description", {})
    effect = description.get("effect", "")
    post_condition = actspec.get("post_condition", {})

    prompt_content = (
        "You are an expert web agent evaluator.\n"
        "Your task is to judge whether a high-level ActSpec action was successfully reused "
        "in a concrete browser trace.\n\n"
        "ActSpec summary:\n"
        f"- action_id: {actspec.get('action_id', '')}\n"
        f"- effect: {effect}\n"
        f"- post_condition (JSON): {json.dumps(post_condition, ensure_ascii=False)}\n\n"
        "Execution trace (text content only):\n"
        f"- BEFORE executing ActSpec:\n{pre_text[:2000]}\n\n"
        f"- AFTER executing ActSpec:\n{post_text[:2000]}\n\n"
        "Question: According to the ActSpec effect and post_condition, did this execution "
        "achieve the intended successful reuse of the ActSpec?\n"
        "Answer STRICTLY with a single token: 'success' or 'fail'."
    )

    messages = [
        {"role": "system", "content": "You are a precise judge that only answers 'success' or 'fail'."},
        {"role": "user", "content": prompt_content},
    ]

    try:
        response = llm_utils.call_llm(lm_cfg, messages)
    except Exception:
        # 如果 LLM 调用失败，回退到 executor 的结果
        return bool(call_record.get("executor_success"))

    text = str(response).strip().lower()
    if "success" in text and "fail" not in text:
        return True
    if "fail" in text and "success" not in text:
        return False
    # 模糊情况时，回退到 executor 结果
    return bool(call_record.get("executor_success"))


def evaluate_actspec_reuse_for_log(
    log_file: str,
    library_path: str,
    llm_cfg: lm_config.LMConfig | None = None,
) -> Tuple[Dict[str, Dict[str, int]], List[Tuple[str, bool]]]:
    """
    对单个任务日志文件进行 ActSpec 复用评估，返回 per-action_id 的统计增量，
    以及每次调用的 (action_id, success_flag) 用于更新置信度。

    返回值形如：
        (
          { "action_id_1": {"usage_count": 3, "success_count": 2, "fail_count": 1}, ... },
          [("action_id_1", True), ("action_id_1", False), ...]  # 按调用顺序
        )
    """
    path = Path(log_file)
    if not path.exists():
        return {}, []

    try:
        with path.open("r", encoding="utf-8") as f:
            log_data = json.load(f)
    except Exception:
        return {}, []

    actspec_calls = log_data.get("actspec_calls", [])
    if not actspec_calls:
        return {}, []

    library = ActSpecLibrary(library_path)
    actspec_map = _load_actspec_map(library, library_path)

    stats: Dict[str, Dict[str, int]] = {}
    confidence_updates: List[Tuple[str, bool]] = []

    for call in actspec_calls:
        action_id = call.get("action_id")
        if not action_id:
            continue
        spec = actspec_map.get(action_id)
        if not spec:
            continue

        # 成功/失败判定（用于统计与淘汰）：
        # 只要未达到步数上限且 executor 成功完成复用过程（即打印了 [ActSpec] ActSpec执行成功），即计入成功次数。
        # - 达到 LLM 调整次数上限（plan step 数 - 1）→ 每一步都需调整，复用无效 → 失败
        # - 未达到上限且 executor_success 为 True → 成功（不再要求 post_condition_satisfied）
        success_flag = None
        if call.get("reached_adjustment_limit") is True:
            success_flag = False
        elif call.get("post_condition_satisfied") is not None:
            success_flag = bool(call.get("post_condition_satisfied"))
        elif call.get("executor_success") is True:
            success_flag = True
        else:
            success_flag = False

        # 若上述规则无法判定，再使用 LLM 语义判断（可选）
        if success_flag is None and llm_cfg is not None:
            try:
                success_flag = _llm_judge_call(llm_cfg, spec, call)
            except Exception:
                # 失败时保留原有 success_flag
                pass

        if success_flag is None:
            # 无法判断则跳过
            continue

        confidence_updates.append((action_id, success_flag))

        if action_id not in stats:
            stats[action_id] = {
                "usage_count": 0,
                "success_count": 0,
                "fail_count": 0,
            }

        stats[action_id]["usage_count"] += 1
        if success_flag:
            stats[action_id]["success_count"] += 1
        else:
            stats[action_id]["fail_count"] += 1

    return stats, confidence_updates


def evaluate_and_update_library_for_log(
    log_file: str,
    library_path: str,
    llm_cfg: lm_config.LMConfig | None = None,
    convert_to_negative_constraints: bool = True,
) -> None:
    """
    对单个任务日志做 ActSpec 复用评估，并将结果写回库（更新 usage/fail/disabled 与置信度）。
    """
    stats, confidence_updates = evaluate_actspec_reuse_for_log(log_file, library_path, llm_cfg)
    if not stats and not confidence_updates:
        return

    library = ActSpecLibrary(library_path)
    if stats:
        library.update_stats_batch(
            stats,
            library_path=library_path,
            convert_to_negative_constraints=convert_to_negative_constraints,
        )
    for action_id, post_success in confidence_updates:
        library.update_confidence(action_id, post_success, library_path=library_path)

