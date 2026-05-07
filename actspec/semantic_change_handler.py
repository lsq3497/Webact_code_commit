"""
语义变化处理：step 失败且自动修复穷尽（或多候选）时，判定「不可视为合理中间态」并可选调用 LLM 重写后续 plan。
"""

import json
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse


def _get_url_from_env(env: Any) -> str:
    if env is None:
        return ""
    if hasattr(env, "get_url"):
        try:
            return env.get_url() or ""
        except Exception:
            pass
    if hasattr(env, "url"):
        return getattr(env, "url", "") or ""
    return ""


def _get_observation_processor_from_env(env: Any) -> Any:
    if not env:
        return None
    try:
        if hasattr(env, "webarena_env"):
            we = env.webarena_env
            if hasattr(we, "observation_handler") and hasattr(we.observation_handler, "text_processor"):
                return we.observation_handler.text_processor
            if hasattr(we, "info") and isinstance(we.info, dict) and "observation_processor" in we.info:
                return we.info["observation_processor"]
    except Exception:
        pass
    return None


def _get_observation_text_from_env(env: Any) -> str:
    """获取当前页面 observation 文本。兼容 obs 为 str 或 dict、text 为 str/list。"""
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



ERROR_LOGIN_CAPTCHA_EMPTY_PATTERNS = [
    "/error", "/login", "/captcha", "/signin",
    "error", "login", "captcha", "暂无内容", "not found", "404",
]


def cannot_consider_valid_intermediate_state(
    actspec: Dict[str, Any],
    current_step_idx: int,
    env: Any,
) -> bool:
    """
    可执行定义（v1）：仅当以下至少一条成立时返回 True，才允许进入 semantic change handler。
    否则视为可能 transient execution error，不触发 LLM。
    """
    plan = actspec.get("plan", [])
    if current_step_idx >= len(plan):
        return False
    current_url = _get_url_from_env(env).lower()
    
    for part in ERROR_LOGIN_CAPTCHA_EMPTY_PATTERNS:
        if part in current_url:
            return True
    pre = actspec.get("pre_condition", {})
    url_pattern = pre.get("url_pattern", "")
    if url_pattern:
        try:
            from .pre_condition_checker import PreConditionChecker
            checker = PreConditionChecker()
            if not checker._match_url_pattern(current_url, url_pattern):
                return True
        except Exception:
            pass
    
    locate = actspec.get("locate", {})
    target_elements = locate.get("target_elements", [])
    proc = _get_observation_processor_from_env(env)
    obs_nodes_info = getattr(proc, "obs_nodes_info", None) if proc else None
    if not obs_nodes_info:
        return False
    for idx in range(current_step_idx, len(plan)):
        te = None
        for t in target_elements:
            if t.get("step") == idx:
                te = t
                break
        if te is None and idx < len(target_elements):
            te = target_elements[idx]
        if not te:
            continue
        for s in te.get("strategies", []):
            if s.get("strategy") != "semantic":
                continue
            cond = s.get("conditions", {})
            role = cond.get("role")
            label = cond.get("label") or cond.get("text")
            for eid, info in obs_nodes_info.items():
                try:
                    node = proc.get_node_info_by_element_id(int(eid)) if hasattr(proc, "get_node_info_by_element_id") else None
                    if not node:
                        continue
                    nr = getattr(node, "role", "") or ""
                    nn = getattr(node, "name", "") or ""
                    nt = getattr(node, "text", "") or ""
                    if role and nr != role:
                        continue
                    if not label:
                        return False
                    lb = label.lower().strip()
                    if lb in (nn or "").lower() or lb in (nt or "").lower():
                        return False
                except Exception:
                    continue
    return True


def _infer_provider_from_model(model: str, default: str = "google") -> str:
    """从模型名推断 provider：google/xxx -> google，openai/xxx -> openai，anthropic/ 或 claude/ -> anthropic，否则返回 default。"""
    if not model or not isinstance(model, str):
        return default
    m = model.strip().lower()
    if m.startswith("google/"):
        return "google"
    if m.startswith("openai/"):
        return "openai"
    if m.startswith("anthropic/") or m.startswith("claude/"):
        return "anthropic"
    if m.startswith("huggingface/") or m.startswith("meta/") or m.startswith("mistral/"):
        return "openai"  
    return default


def _call_llm_rewrite_plan(
    observation_text: str,
    page_description: str,
    actspec: Dict[str, Any],
    current_step_idx: int,
    failure_reason: str,
    env: Any,
) -> Optional[List[Dict[str, Any]]]:
    """
    调用 LLM 从当前 step 重写后续 plan。使用 config.yaml 中的 llm_provider 与
    models.agent_actor（与 Actor 一致）；provider 根据模型名自动识别（如 google/xxx -> google），
    默认为 google，通过 llms.call_llm 调用；若无可用配置或调用失败则返回 None。
    """
    lm_cfg = None
    try:
        from pathlib import Path
        from llms import lm_config
        
        config_dir = Path(__file__).resolve().parent.parent / "config"
        try:
            from config.config_loader import get_config
            unified = get_config(str(config_dir))
            model = unified.get_model("agent_actor", "default")
        except Exception as e:
            print(f"[ActSpec] LLM 重写跳过: 加载统一配置失败 — {type(e).__name__}: {e}")
            return None
        if not model:
            model = "google/gemini-2.5-flash"
        
        provider = _infer_provider_from_model(model, default="google")
        
        config = getattr(env, "global_config", None)
        actspec_cfg = getattr(config, "actspec", None) if config else None
        if actspec_cfg is not None and hasattr(actspec_cfg, "get"):
            llm_dict = actspec_cfg.get("llm_config", {}) or {}
        elif isinstance(actspec_cfg, dict):
            llm_dict = actspec_cfg.get("llm_config", {}) or {}
        else:
            llm_dict = {}
        gen_config = {
            "temperature": llm_dict.get("temperature", 0.1),
            "max_tokens": llm_dict.get("max_tokens", 2048),
            "top_p": llm_dict.get("top_p", 1.0),
            "context_length": llm_dict.get("context_length", 0),
        }
        
        
        lm_cfg = lm_config.LMConfig(
            provider=provider,
            model=model,
            mode="chat",
            gen_config=gen_config,
        )
    except Exception as e:
        print(f"[ActSpec] LLM 重写跳过: 构建 LMConfig 失败 — {type(e).__name__}: {e}")
        return None

    plan = actspec.get("plan", [])
    remaining = plan[current_step_idx:] if current_step_idx < len(plan) else []
    prompt_content = (
        "Current page observation (accessibility tree or text):\n" + (observation_text[:8000] or "N/A") + "\n\n"
        "Page description (URL / key info): " + (page_description or "N/A") + "\n\n"
        "ActSpec action_id: " + actspec.get("action_id", "") + "\n"
        "Failure reason: " + failure_reason + "\n"
        "Remaining plan steps from current step (JSON): " + json.dumps(remaining, ensure_ascii=False) + "\n\n"
        "Rewrite only the remaining action sequence from the current step to complete the task. "
        "Output a JSON array of plan steps with primitive, target (strategy, value), text/url/raw as needed. "
        "Keep the same schema as the existing plan."
    )
    messages = [
        {"role": "system", "content": "You output only a JSON array of action steps, no other text."},
        {"role": "user", "content": prompt_content},
    ]
    try:
        from llms import utils as llm_utils
        response = llm_utils.call_llm(lm_cfg, messages)
        if not response:
            return None
        text = response if isinstance(response, str) else getattr(response, "text", None) or str(response)
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            new_remaining = json.loads(text[start:end])
            if isinstance(new_remaining, list):
                return new_remaining
    except Exception as e:
        print(f"[ActSpec] LLM 重写失败（call_llm 或解析）: {type(e).__name__}: {e}")
    return None


def handle_semantic_change(
    actspec: Dict[str, Any],
    current_step_idx: int,
    env: Any,
    observation_text: str,
    page_description: str,
    failure_reason: str,
    llm_adjustment_count: int,
    max_llm_adjustments: int,
) -> Tuple[Optional[Dict[str, Any]], Optional[int], int]:
    """
    语义变化时：多候选直接尝试 LLM；其他失败仅当不可视为合理中间态时才调用 LLM。
    返回 (new_actspec_or_none, start_step_idx_or_none, new_llm_adjustment_count)。
    """
    plan = actspec.get("plan", [])
    if llm_adjustment_count >= max_llm_adjustments:
        return None, None, llm_adjustment_count
    if failure_reason != "locate_multiple_candidates" and not cannot_consider_valid_intermediate_state(
        actspec, current_step_idx, env
    ):
        return None, None, llm_adjustment_count
    new_remaining = _call_llm_rewrite_plan(
        observation_text, page_description, actspec, current_step_idx, failure_reason, env
    )
    if not new_remaining:
        return None, None, llm_adjustment_count
    new_plan = plan[:current_step_idx] + new_remaining
    new_actspec = {**actspec, "plan": new_plan}
    temp_subdir = tempfile.mkdtemp(prefix="actspec_rewrite_")
    try:
        from .actspec_library import ActSpecLibrary
        lib = ActSpecLibrary(base_path=os.path.dirname(temp_subdir))
        lib.save_actspec(new_actspec, library_path=temp_subdir)
    except Exception:
        pass
    return new_actspec, current_step_idx, llm_adjustment_count + 1
