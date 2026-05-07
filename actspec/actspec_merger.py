"""
ActSpec合并器：将ActSpec与primitive action合并
"""

import copy
import json
import re
from typing import Dict, List, Any, Optional


class ActSpecMerger:
    """ActSpec合并器，负责将ActSpec与primitive actions合并"""
    
    def __init__(self):
        """初始化ActSpec合并器"""
        pass
    
    def merge_actions(
        self,
        primitive_actions: List[str],
        actspecs: List[Dict[str, Any]],
        context: Dict[str, Any],
        negative_constraint_filter: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        合并actions
        
        Args:
            primitive_actions: primitive action列表
            actspecs: ActSpec列表
            context: 上下文信息
            negative_constraint_filter: 负约束过滤器（可选）
        
        Returns:
            合并后的actions字典，包含：
                - primitive_actions: List[str]
                - actspec_actions: List[str]
                - all_actions: List[str]
        """
        
        if negative_constraint_filter:
            actspecs = negative_constraint_filter.filter_actspecs(actspecs, context)
        
        
        actspec_actions = self.generate_action_candidates(actspecs, context)
        
        
        all_actions = primitive_actions + actspec_actions
        
        return {
            "primitive_actions": primitive_actions,
            "actspec_actions": actspec_actions,
            "all_actions": all_actions
        }
    
    def generate_action_candidates(
        self,
        actspecs: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> List[str]:
        """
        生成action候选
        
        Args:
            actspecs: ActSpec列表
            context: 上下文信息
        
        Returns:
            action候选字符串列表
        """
        candidates = []
        
        for actspec in actspecs:
            action_id = actspec.get("action_id", "")
            
            action_str = f"actspec [{action_id}]"
            candidates.append(action_str)
        
        return candidates
    
    def _build_plan_preview(self, actspec: Dict[str, Any], max_steps: int = 5) -> str:
        """
        根据单个 actspec 生成简短的「已填参数的多步动作序列预览」字符串，仅用于 prompt 展示。
        不修改 actspec 本身；对 plan 的副本做轻量参数填充后输出 CLICK/TYPE/GOTO/SCROLL 的最简行。
        """
        plan = actspec.get("plan", [])
        if not plan:
            return ""
        bindings = actspec.get("bindings", {})
        params_def = actspec.get("parameters", {})
        
        
        preview_params: Dict[str, Any] = {}
        all_candidates: Dict[str, List[Any]] = {}
        for pname, pdef in params_def.items():
            cands = list(pdef.get("candidates", []) or [])
            all_candidates[pname] = cands
            if cands:
                preview_params[pname] = cands[0]
            else:
                if pdef.get("type") == "number":
                    preview_params[pname] = 0
                elif pdef.get("type") == "string":
                    preview_params[pname] = ""
        preview_plan = copy.deepcopy(plan)
        
        for param_name, binding_info in bindings.items():
            bind_to = binding_info.get("bind_to", []) or []
            if not bind_to:
                continue
            param_cands = all_candidates.get(param_name) or []
            
            
            if param_cands and len(param_cands) >= len(bind_to):
                for idx, bind_rule in enumerate(bind_to):
                    step_idx = bind_rule.get("step")
                    field = bind_rule.get("field")
                    if step_idx is None or step_idx >= len(preview_plan):
                        continue
                    val = param_cands[idx]
                    step = preview_plan[step_idx]
                    if field == "text":
                        step["text"] = str(val)
                    elif field == "target.value":
                        if "target" in step:
                            step["target"]["value"] = str(val)
                    elif field == "url":
                        step["url"] = str(val)
            else:
                val = preview_params.get(param_name)
                if val is None:
                    continue
                for bind_rule in bind_to:
                    step_idx = bind_rule.get("step")
                    field = bind_rule.get("field")
                    if step_idx is None or step_idx >= len(preview_plan):
                        continue
                    step = preview_plan[step_idx]
                    if field == "text":
                        step["text"] = str(val)
                    elif field == "target.value":
                        if "target" in step:
                            step["target"]["value"] = str(val)
                    elif field == "url":
                        step["url"] = str(val)
        
        plan_str = json.dumps(preview_plan)
        for pname, pval in preview_params.items():
            plan_str = plan_str.replace(f"{ { {pname}} } ", str(pval))
        preview_plan = json.loads(plan_str)
        
        lines: List[str] = []
        for idx, step in enumerate(preview_plan):
            if len(lines) >= max_steps:
                lines.append("...")
                break
            prim = (step.get("primitive") or "").upper()
            if prim == "CLICK":
                t = step.get("target", {})
                v = t.get("value", "")
                lines.append(f"{idx+1}) CLICK element_id={v}")
            elif prim == "TYPE":
                t = step.get("target", {})
                v = t.get("value", "")
                text = step.get("text", "")
                text_repr = repr(text)[:60] + "..." if len(text) > 60 else repr(text)
                lines.append(f"{idx+1}) TYPE element_id={v} text={text_repr}")
            elif prim == "GOTO":
                url = step.get("url", "") or step.get("raw", "")
                url_repr = repr(url)[:80] + "..." if len(str(url)) > 80 else repr(url)
                lines.append(f"{idx+1}) GOTO url={url_repr}")
            elif prim == "SCROLL":
                direction = step.get("direction", "down")
                lines.append(f"{idx+1}) SCROLL direction={direction}")
        return "\n".join(lines) if lines else ""
    
    def update_prompt_with_actspecs(
        self,
        prompt: str,
        actspecs: List[Dict[str, Any]]
    ) -> str:
        """
        更新prompt以包含ActSpec描述
        
        Args:
            prompt: 原始prompt
            actspecs: ActSpec列表
        
        Returns:
            更新后的prompt
        """
        if not actspecs:
            return prompt
        
        
        actspec_section = "\n\n"
        actspec_section += (
            "ActSpec bundles multiple primitive actions into one high-level step. "
            "Whenever a suitable ActSpec is available, you SHOULD prefer reusing it "
            "to reduce total steps and LLM calls.\n\n"
        )
        actspec_section += "## Available ActSpec Actions:\n\n"
        actspec_section += (
            "ActSpec IDs: You MUST use the exact `action_id` shown below "
            "(for example `xxx.xxx.xxx`), never DOM element ids like `389` or `4345`, "
            "and never shorten or partially match the id.\n"
        )
        actspec_section += (
            "Parameters & defaults: Each ActSpec defines its own parameters and default values. "
            "When calling an ActSpec during planning, you ONLY need to specify parameters whose "
            "values you explicitly want to change; any parameter you do NOT mention will "
            "automatically use its default value from the ActSpec definition.\n"
        )
        actspec_section += (
            "Parameter types: Use only the listed parameter names for each ActSpec. "
            "You may override text/string parameters (for example `search_keyword`, `input_text`) "
            "if that helps the current task, but you should NOT override structural parameters "
            "such as `element_id`, `button_id`, `combobox_id`, or plain `id` — those are filled "
            "by the underlying plan/locate logic.\n"
        )
        actspec_section += (
            "Call format (single line only): "
            "`actspec [action_id]` or "
            "`actspec [action_id],param1=\"value1\",param2=\"value with spaces\"`. "
            "Parameters (if any) MUST be comma-separated, start with a comma right after the closing `]`, "
            "and always use double quotes for values. Do NOT insert any extra natural language "
            "before or after the command.\n\n"
        )
        
        for actspec in actspecs:
            action_id = actspec.get("action_id", "")
            description = actspec.get("description", {})
            meta = actspec.get("metadata", {})
            usage = int(meta.get("usage_count", 0) or 0)
            conf = float(meta.get("confidence", 1.0))
            if usage == 0:
                confidence_note = " (not yet used — highest priority)"
            else:
                confidence_note = f" (confidence: {conf})"
            
            
            actspec_section += f"### {action_id}{confidence_note}\n"
            actspec_section += f"- **Summary**: {description.get('summary', 'N/A')}\n"
            actspec_section += f"- **When to use**: {description.get('when_to_use', 'N/A')}\n"
            actspec_section += f"- **Effect**: {description.get('effect', 'N/A')}\n"
            plan_preview_str = self._build_plan_preview(actspec)
            if plan_preview_str:
                actspec_section += f"- **Plan preview (with default parameters)**:\n"
                for line in plan_preview_str.split("\n"):
                    actspec_section += f"  {line}\n"
            
            params_def = actspec.get("parameters", {}) or {}
            text_param_names = [k for k, p in params_def.items() if (p or {}).get("type") == "string"]
            if text_param_names:
                actspec_section += f"- **Overrideable text params**: {', '.join(text_param_names)}\n"
            actspec_section += "\n"
        
        
        updated_prompt = prompt + actspec_section
        
        return updated_prompt
    
    def parse_actspec_action(
        self,
        action_str: str
    ) -> Optional[Dict[str, Any]]:
        """
        解析ActSpec action字符串
        
        支持格式：actspec [action_id] 或 actspec [action_id] param1=value1 param2="value with spaces"
        约定：结构性参数（element_id 等）不建议由 LLM 自行填入，优先由系统默认值/locate 决定；
        文本参数（input_text、search_keyword 等）允许 LLM 在 action 中覆盖。
        
        Args:
            action_str: action字符串
        
        Returns:
            解析后的字典：{"action_id": str, "parameters": Dict} 或 None
        """
        if not action_str.startswith("actspec"):
            return None
        
        
        match = re.search(r"^actspec\s*\[(?P<id>[^\]]+)\]", action_str.strip())
        if not match:
            return None
        
        action_id = match.group("id")
        
        
        close_bracket_idx = action_str.find("]")
        params_part = action_str[close_bracket_idx + 1 :].strip()
        parameters: Dict[str, Any] = {}
        if not params_part:
            return {
                "action_id": action_id,
                "parameters": parameters
            }
        
        
        
        if params_part.startswith(","):
            params_part = params_part[1:].strip()
        if not params_part:
            return {
                "action_id": action_id,
                "parameters": parameters
            }
        
        
        raw_items = [item.strip() for item in params_part.split(",") if item.strip()]
        for item in raw_items:
            
            m_q = re.match(r'^(\w+)\s*=\s*"([^"]*)"$', item)
            if m_q:
                param_name = m_q.group(1)
                param_value = m_q.group(2)
                parameters[param_name] = param_value
                continue
            
            m_u = re.match(r'^(\w+)\s*=\s*([^\s,]+)$', item)
            if m_u:
                param_name = m_u.group(1)
                param_value = m_u.group(2)
                parameters[param_name] = param_value
                continue
            
        
        return {
            "action_id": action_id,
            "parameters": parameters
        }
