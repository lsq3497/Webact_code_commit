"""
ActSpec generator: turn segmented action traces into valid ActSpec JSON.
"""

import copy
import json
import re
from typing import Dict, List, Any, Optional
from llms import lm_config, utils as llm_utils

from .trace_segmenter import is_page_change_action
from .negative_constraint_utils import build_action_history_prefix


class ActSpecGenerator:
    """Build ActSpec JSON objects from trace segments."""

    def __init__(self, llm_config: Optional[lm_config.LMConfig] = None):
        """Optional LMConfig; otherwise default OpenAI chat settings."""
        self.llm_config = llm_config
        if self.llm_config is None:
            
            self.llm_config = lm_config.LMConfig(
                provider="openai",
                model="gpt-4-turbo",
                mode="chat",
                gen_config={
                    "temperature": 0.1,
                    "max_tokens": 3000,
                    "top_p": 1.0,
                    "context_length": 0,
                }
            )
    
    def generate_actspec(
        self,
        action_sequence: List[Any],
        context: Dict[str, Any],
        task_info: Dict[str, Any],
        trajectory: Optional[List[Dict[str, Any]]] = None,
        segment_start_idx: Optional[int] = None,
        segment_end_idx: Optional[int] = None
    ) -> Dict[str, Any]:
        """Assemble one ActSpec from segment actions and optional full trajectory indices."""
        
        parameters = self._extract_parameters(action_sequence)
        
        
        context_info = self._extract_context(action_sequence, context, task_info)
        
        
        description = self._generate_description(action_sequence, context)
        
        
        plan = self._create_executable_plan(action_sequence)
        
        
        plan = self._parameterize_plan(plan, parameters, action_sequence)
        
        
        bindings = self._create_bindings(parameters, plan)
        
        
        action_id, action_name = self._generate_action_identity(
            context_info, description, action_sequence
        )
        
        
        if trajectory is not None and segment_start_idx is not None:
            locate = self._extract_locate_strategies(
                action_sequence, trajectory, segment_start_idx, parameters
            )
        else:
            locate = {}
        
        
        if trajectory is not None and segment_start_idx is not None:
            pre_condition = self._extract_pre_condition(
                action_sequence, trajectory, segment_start_idx, context_info
            )
        else:
            pre_condition = {}
        
        
        if trajectory is not None and segment_start_idx is not None and segment_end_idx is not None:
            post_condition = self._extract_post_condition(
                action_sequence, trajectory, segment_start_idx, segment_end_idx, context_info
            )
        else:
            post_condition = {}
        
        
        
        action_history_prefix: List[str] = []
        if trajectory is not None and segment_start_idx is not None:
            try:
                action_history_prefix = build_action_history_prefix(
                    trajectory=trajectory,
                    segment_start=segment_start_idx,
                )
            except Exception as e:
                print(f"[ActSpec] Failed to build action_history_prefix: {e}")
                action_history_prefix = []

        actspec = {
            "action_id": action_id,
            "action_name": action_name,
            "parameters": parameters,
            "context": context_info,
            "description": description,
            "locate": locate,
            "pre_condition": pre_condition,
            "post_condition": post_condition,
            "action_history_prefix": action_history_prefix,
            "plan": plan,
            "bindings": bindings,
            "metadata": {
                "source": "offline_training",
                "generated_by": "llm_trace_segmentation",
                "avg_steps": len(plan),
                "usage_count": 0,
                "confidence": 1.0,  
            }
        }
        
        
        is_failed, failure_reason, constraint_subtype = self._is_failed_actspec(
            actspec, action_sequence, context_info
        )
        
        
        if is_failed:
            actspec["is_failed"] = True
            actspec["failure_reason"] = failure_reason
            actspec["constraint_subtype"] = constraint_subtype
            actspec["type"] = "negative_constraint_candidate"  
            
            plan_before_param = copy.deepcopy(actspec.get("plan", []))
        else:
            actspec["is_failed"] = False
            actspec["type"] = "executable_actspec"  
            plan_before_param = None
        
        
        actspec = self._post_process_parameterize(actspec, action_sequence)
        
        
        actspec = self._fill_parameter_candidates_from_trajectory(actspec, action_sequence)
        
        
        validation_errors = self._validate_actspec(actspec)
        if validation_errors:
            print(f"[Warning] ActSpec validation issues ({actspec.get('action_id', 'unknown')}):")
            for error in validation_errors:
                print(f"  - {error}")
            
            actspec = self._auto_fix_actspec(actspec, validation_errors)
        
        
        if is_failed and plan_before_param:
            actspec["plan"] = plan_before_param
        
        return actspec
    
    def _extract_parameters(self, action_sequence: List[Any]) -> Dict[str, Any]:
        """LLM-assisted parameter schema extraction from primitive strings."""
        parameters = {}
        
        
        
        action_strs = [str(action) for action in action_sequence]
        action_summary = "\n".join([f"{i}: {a}" for i, a in enumerate(action_strs)])
        
        
        text_values = []
        element_ids = []
        for i, action_str in enumerate(action_strs):
            
            type_match = re.search(r"type\s*\[.*?\]\s*\[(.*?)\]\s*\[.*?\]", action_str)
            if type_match:
                text = type_match.group(1).strip()
                if text and len(text) > 0:
                    text_values.append(text)
                
                type_id_match = re.search(r"type\s*\[(\d+)\]", action_str)
                if type_id_match:
                    element_ids.append(f"type[{i}]: element_id={type_id_match.group(1)}")
            
            click_id_match = re.search(r"click\s*\[(\d+)\]", action_str, re.I)
            if click_id_match:
                element_ids.append(f"click[{i}]: element_id={click_id_match.group(1)}")
        
        text_values_str = ", ".join(text_values[:5]) if text_values else "(none)"
        element_ids_str = ", ".join(element_ids[:5]) if element_ids else "(none)"
        
        system_prompt = """You extract parameter definitions from primitive action traces.

Types:
- enum: discrete values; include candidates[]
- string: free text (keywords, inputs)
- number: numeric ids or counters
- boolean: flags

Rules:
1. Parameterize values likely to change across tasks (filters, keywords).
2. Prefer business-meaningful fields (status, role).
3. Skip obviously fixed chrome (static button labels, hard-coded asset URLs) unless clearly reusable.
4. For enums, infer realistic candidate sets from context.

Return JSON only."""

        user_prompt = f"""Parameterize this trace:

Actions:
{action_summary}

Sampled text values: {text_values_str}
Sampled element ids: {element_ids_str}

Hints:
- TYPE steps: varying element ids -> dedicated id params (input_id, etc.).
- CLICK steps: varying targets -> button_id / click_id style params.
- TYPE literals -> search_keyword / input_text style params.

Example shape:
{{
  "status": {{
    "type": "enum",
    "candidates": ["pending", "paid", "shipped"],
    "description": "order status to filter",
    "optional": false
  }},
  "search_keyword": {{
    "type": "string",
    "optional": false,
    "description": "search keyword to query"
  }},
  "page_number": {{
    "type": "number",
    "optional": true,
    "description": "page number for pagination"
  }}
}}

JSON only, no prose."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = llm_utils.call_llm(self.llm_config, messages)
            response = response.strip()
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
            
            parameters = json.loads(response)
            
            
            validated_parameters = {}
            for param_name, param_def in parameters.items():
                if isinstance(param_def, dict):
                    validated_param = {
                        "type": param_def.get("type", "string"),
                        "description": param_def.get("description", ""),
                        "optional": param_def.get("optional", False)
                    }
                    if validated_param["type"] == "enum" and "candidates" in param_def:
                        validated_param["candidates"] = param_def["candidates"]
                    validated_parameters[param_name] = validated_param
                else:
                    
                    continue
            
            parameters = validated_parameters
        except Exception as e:
            print(f"[Warning] Parameter extraction failed: {e}, using empty parameters")
            parameters = {}
        
        return parameters
    
    def _extract_context(
        self,
        action_sequence: List[Any],
        context: Dict[str, Any],
        task_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Augment site/page/url_pattern/required_elements from actions + task metadata."""
        site = context.get("site", "unknown")
        page = context.get("page", "unknown")
        url = context.get("url", "")
        
        
        url_pattern = "/"
        if url:
            try:
                from urllib.parse import urlparse, parse_qs
                parsed = urlparse(url)
                path = parsed.path
                
                
                if not path or path == "/":
                    
                    if page == "home":
                        url_pattern = "/"
                    elif page == "forum":
                        url_pattern = "/forums"
                    elif page == "comment":
                        url_pattern = "/comments"
                    elif page == "search":
                        url_pattern = "/search"
                    elif page == "admin":
                        url_pattern = "/admin"
                    elif page == "order_list":
                        url_pattern = "/order"
                    elif page == "product_list":
                        url_pattern = "/product"
                    else:
                        url_pattern = "/"
                else:
                    
                    
                    
                    path_parts = path.split("/")
                    pattern_parts = []
                    for part in path_parts:
                        if not part:
                            continue
                        
                        if part.isdigit():
                            pattern_parts.append("{id}")
                        
                        elif re.match(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", part, re.I):
                            pattern_parts.append("{uuid}")
                        
                        elif re.match(r"^[a-zA-Z0-9_-]+$", part) and len(part) > 10:
                            pattern_parts.append("{id}")
                        else:
                            pattern_parts.append(part)
                    
                    if pattern_parts:
                        url_pattern = "/" + "/".join(pattern_parts)
                    else:
                        url_pattern = "/"
                    
                    
                    query_params = parse_qs(parsed.query)
                    if query_params:
                        
                        
                        param_patterns = []
                        for key, values in query_params.items():
                            if key in ["page", "limit", "offset", "id", "status", "type"]:
                                param_patterns.append(f"{key}={ {key}} ")
                            else:
                                param_patterns.append(f"{key}={ {key}} ")
                        if param_patterns:
                            url_pattern += "?" + "&".join(param_patterns)
            except Exception as e:
                
                if page != "unknown":
                    if page == "home":
                        url_pattern = "/"
                    elif page == "forum":
                        url_pattern = "/forums"
                    elif page == "comment":
                        url_pattern = "/comments"
                    elif page == "search":
                        url_pattern = "/search"
                    else:
                        url_pattern = "/"
                else:
                    url_pattern = "/"
        
        
        required_elements = []
        element_ids = set()
        
        for action in action_sequence:
            action_str = str(action).lower()
            
            
            id_match = re.search(r"\[(\d+)\]", action_str)
            if id_match:
                element_ids.add(id_match.group(1))
            
            
            if "click" in action_str:
                required_elements.append("ClickableElement")
            elif "type" in action_str:
                required_elements.append("InputElement")
            elif "hover" in action_str:
                required_elements.append("HoverableElement")
            elif "scroll" in action_str:
                required_elements.append("ScrollableContainer")
            elif "goto" in action_str:
                
                pass
        
        
        if len(action_sequence) > 3 and len(element_ids) > 0:
            
            
            if len(element_ids) > 3:
                required_elements.append("ListOrTableElement")
        
        context_info = {
            "site": site,
            "page": page,
            "url_pattern": url_pattern,
            "required_elements": sorted(list(set(required_elements)))  
        }
        
        return context_info
    
    def _generate_description(
        self,
        action_sequence: List[Any],
        context: Dict[str, Any]
    ) -> Dict[str, str]:
        """LLM-written summary / when_to_use / effect (intent-level, no literal DOM ids)."""
        action_strs = [str(action) for action in action_sequence]
        action_summary = "\n".join([f"{i}: {a}" for i, a in enumerate(action_strs)])
        
        system_prompt = """You describe high-level UI routines.

Rules:
1. Focus on user intent, not click-by-click mechanics.
2. State input → observable outcome.
3. Use product language, not DOM trivia.
4. Never bake concrete literals (city names, numeric ids, one-off keywords)—stay abstract.
5. Say "enter the search keywords" not "type Worcester".
6. Say "activate the target control" not "click element 79".
JSON only for outputs."""

        user_prompt = f"""Describe this routine:

Context:
- Site: {context.get('site', 'unknown')}
- Page: {context.get('page', 'unknown')}

Actions:
{action_summary}

Return JSON:
{{
  "summary": "one sentence overview",
  "when_to_use": "when to invoke this bundle",
  "effect": "expected state after success"
}}

JSON only, no extra text."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = llm_utils.call_llm(self.llm_config, messages)
            response = response.strip()
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
            
            description = json.loads(response)
        except Exception as e:
            print(f"[Warning] Description generation failed: {e}, using fallback")
            description = {
                "summary": "Execute the recorded UI steps",
                "when_to_use": "When this workflow is required",
                "effect": "Task completes after the listed interactions",
            }
        
        return description
    
    def _create_executable_plan(self, action_sequence: List[Any]) -> List[Dict[str, Any]]:
        """Normalize primitives into internal plan dicts."""
        plan = []
        
        for action in action_sequence:
            action_str = str(action).strip()
            
            
            
            
            
            
            action_str_clean = action_str.split(" where ")[0].strip()
            
            
            if is_page_change_action(action_str_clean):
                continue
            
            if action_str_clean.startswith("click") or "#Click#" in action_str:
                
                match = re.search(r"click\s*\[(\d+)\]", action_str_clean)
                if match:
                    element_id = match.group(1)
                    plan.append({
                        "primitive": "CLICK",
                        "target": {
                            "strategy": "element_id",
                            "value": element_id
                        }
                    })
                else:
                    
                    match = re.search(r"#Click#\s*(.+)", action_str)
                    if match:
                        label = match.group(1).strip()
                        plan.append({
                            "primitive": "CLICK",
                            "target": {
                                "strategy": "text",
                                "value": label
                            }
                        })
                    else:
                        
                        plan.append({
                            "primitive": "CLICK",
                            "raw": action_str
                        })
                        
            elif action_str_clean.startswith("type") or "#Type#" in action_str:
                
                match = re.search(r"type\s*\[(\d+)\]\s*\[(.*?)\]\s*\[(\d+)\]", action_str_clean)
                if match:
                    element_id = match.group(1)
                    text = match.group(2)
                    enter_flag = match.group(3)
                    
                    plan.append({
                        "primitive": "TYPE",
                        "target": {
                            "strategy": "element_id",
                            "value": element_id
                        },
                        "text": text,  
                        "enter": enter_flag == "1"
                    })
                else:
                    
                    match = re.search(r"#Type#\s*(.+?)\s+(.+)", action_str)
                    if match:
                        label = match.group(1).strip()
                        text = match.group(2).strip()
                        plan.append({
                            "primitive": "TYPE",
                            "target": {
                                "strategy": "text",
                                "value": label
                            },
                            "text": text,
                            "enter": False
                        })
                    else:
                        plan.append({
                            "primitive": "TYPE",
                            "raw": action_str
                        })
                        
            elif action_str_clean.startswith("hover") or "#Hover#" in action_str:
                
                match = re.search(r"hover\s*\[(\d+)\]", action_str_clean)
                if match:
                    element_id = match.group(1)
                    plan.append({
                        "primitive": "HOVER",
                        "target": {
                            "strategy": "element_id",
                            "value": element_id
                        }
                    })
                else:
                    match = re.search(r"#Hover#\s*(.+)", action_str)
                    if match:
                        label = match.group(1).strip()
                        plan.append({
                            "primitive": "HOVER",
                            "target": {
                                "strategy": "text",
                                "value": label
                            }
                        })
                    else:
                        plan.append({
                            "primitive": "HOVER",
                            "raw": action_str
                        })
                        
            elif action_str_clean.startswith("scroll") or "#Scroll" in action_str:
                
                match = re.search(r"scroll\s*\[?(up|down)\]?", action_str_clean, re.I)
                if match:
                    direction = match.group(1).lower()
                    plan.append({
                        "primitive": "SCROLL",
                        "direction": direction
                    })
                else:
                    match = re.search(r"#Scroll_(up|down)#", action_str, re.I)
                    if match:
                        direction = match.group(1).lower()
                        plan.append({
                            "primitive": "SCROLL",
                            "direction": direction
                        })
                    else:
                        plan.append({
                            "primitive": "SCROLL",
                            "direction": "down"  
                        })
                        
            elif action_str_clean.startswith("goto") or "#Goto#" in action_str:
                
                match = re.search(r"goto\s*\[(.*?)\]", action_str_clean)
                if match:
                    url = match.group(1)
                    plan.append({
                        "primitive": "GOTO",
                        "url": url
                    })
                else:
                    match = re.search(r"#Goto#\s*(.+)", action_str)
                    if match:
                        url = match.group(1).strip()
                        plan.append({
                            "primitive": "GOTO",
                            "url": url
                        })
                    else:
                        plan.append({
                            "primitive": "GOTO",
                            "raw": action_str
                        })
                        
            elif action_str_clean.startswith("press") or "#Press#" in action_str:
                
                match = re.search(r"press\s*\[(.*?)\]", action_str_clean)
                if match:
                    key_comb = match.group(1)
                    plan.append({
                        "primitive": "PRESS",
                        "key_comb": key_comb
                    })
                else:
                    match = re.search(r"#Press#\s*(.+)", action_str)
                    if match:
                        key_comb = match.group(1).strip()
                        plan.append({
                            "primitive": "PRESS",
                            "key_comb": key_comb
                        })
                    else:
                        plan.append({
                            "primitive": "PRESS",
                            "raw": action_str
                        })
                        
            elif action_str_clean.startswith("select") or "#Select#" in action_str:
                
                match = re.search(r"select\s*\[(\d+)\]\s*\[(.*?)\]", action_str_clean)
                if match:
                    element_id = match.group(1)
                    option = match.group(2)
                    plan.append({
                        "primitive": "SELECT",
                        "target": {
                            "strategy": "element_id",
                            "value": element_id
                        },
                        "option": option
                    })
                else:
                    match = re.search(r"#Select#\s*(.+?)\s+(.+)", action_str)
                    if match:
                        label = match.group(1).strip()
                        option = match.group(2).strip()
                        plan.append({
                            "primitive": "SELECT",
                            "target": {
                                "strategy": "text",
                                "value": label
                            },
                            "option": option
                        })
                    else:
                        plan.append({
                            "primitive": "SELECT",
                            "raw": action_str
                        })
                        
            
            elif action_str in ["stop", "go_back", "go_home", "new_tab", "close_tab"] or                 action_str in ["#Go_backward#", "#Go_forward#", "#Exit#", "#Answer#"] or                 action_str_clean.lower().startswith("stop "):
                
                if action_str.startswith("stop") or action_str.startswith("#Answer#"):
                    
                    answer_match = re.search(r"stop\s*\[(.*?)\]", action_str)
                    if answer_match:
                        answer = answer_match.group(1)
                        plan.append({
                            "primitive": "STOP",
                            "answer": answer
                        })
                    else:
                        answer_match = re.search(r"#Answer#\s*(.+)", action_str)
                        if answer_match:
                            answer = answer_match.group(1).strip()
                            plan.append({
                                "primitive": "STOP",
                                "answer": answer
                            })
                        else:
                            plan.append({
                                "primitive": "STOP"
                            })
                elif "go_back" in action_str or "#Go_backward#" in action_str:
                    plan.append({
                        "primitive": "GO_BACK"
                    })
                elif "go_forward" in action_str or "#Go_forward#" in action_str:
                    plan.append({
                        "primitive": "GO_FORWARD"
                    })
                elif "new_tab" in action_str:
                    plan.append({
                        "primitive": "NEW_TAB"
                    })
                elif "close_tab" in action_str:
                    plan.append({
                        "primitive": "CLOSE_TAB"
                    })
                elif "#Exit#" in action_str:
                    plan.append({
                        "primitive": "STOP"
                    })
                else:
                    primitive_name = action_str.upper().replace("_", "").replace("#", "")
                    plan.append({
                        "primitive": primitive_name
                    })
            
            elif action_str_clean.lower().startswith("branch") or action_str_clean.lower().startswith("prune"):
                continue
            elif action_str_clean.strip().lower().startswith("note ") or action_str_clean.strip().lower().startswith("note["):
                
                continue
            else:
                
                plan.append({
                    "primitive": "UNKNOWN",
                    "raw": action_str
                })
        
        
        plan = [s for s in plan if s.get("primitive") != "STOP"]
        return plan
    
    def _parameterize_plan(
        self,
        plan: List[Dict[str, Any]],
        parameters: Dict[str, Any],
        action_sequence: List[Any]
    ) -> List[Dict[str, Any]]:
        """Swap literals for {{param}} placeholders aligned with schema (ids:number, text:string)."""
        if not parameters:
            return plan
        
        import copy
        parameterized_plan = copy.deepcopy(plan)
        
        
        action_values = {}
        for i, action in enumerate(action_sequence):
            action_str = str(action)
            
            id_match = re.search(r"\[(\d+)\]", action_str)
            if id_match:
                element_id = id_match.group(1)
                action_values[i] = {"element_id": element_id}
            
            type_match = re.search(r"type\s*\[.*?\]\s*\[(.*?)\]\s*\[.*?\]", action_str)
            if type_match:
                text = type_match.group(1).strip()
                if text:
                    if i not in action_values:
                        action_values[i] = {}
                    action_values[i]["text"] = text
        
        
        element_id_params = {}  
        text_params = {}  
        
        for param_name, param_def in parameters.items():
            param_type = param_def.get("type", "string")
            param_desc = param_def.get("description", "").lower()
            param_name_lower = param_name.lower()
            
            
            is_element_id = (
                param_type == "number" and (
                    "id" in param_name_lower or
                    "element" in param_name_lower or
                    "button" in param_name_lower or
                    "click" in param_name_lower or
                    "type" in param_name_lower or
                    "input" in param_name_lower or
                    "id" in param_desc or
                    "element" in param_desc or
                    "button" in param_desc
                )
            )
            
            is_text_input = (
                param_type == "string" and (
                    "text" in param_name_lower or
                    "keyword" in param_name_lower or
                    "search" in param_name_lower or
                    "input" in param_name_lower or
                    "text" in param_desc or
                    "keyword" in param_desc or
                    "search" in param_desc or
                    "input" in param_desc
                )
            )
            
            if is_element_id:
                element_id_params[param_name] = param_def
            if is_text_input:
                text_params[param_name] = param_def
        
        
        text_params_ordered = sorted(text_params.keys())
        
        input_id_params_ordered = sorted(
            [p for p in element_id_params if "input" in p.lower() or ("type" in p.lower() and "click" not in p.lower())]
        )
        if not input_id_params_ordered:
            input_id_params_ordered = sorted(element_id_params.keys())
        click_id_params_ordered = sorted(
            [p for p in element_id_params if "click" in p.lower() or "button" in p.lower()]
        )
        if not click_id_params_ordered:
            click_id_params_ordered = [p for p in sorted(element_id_params.keys()) if p not in input_id_params_ordered]
        if not click_id_params_ordered:
            click_id_params_ordered = list(element_id_params.keys())
        
        type_step_index = 0
        click_step_index = 0
        
        
        for step_idx, step in enumerate(parameterized_plan):
            primitive = step.get("primitive", "").upper()
            
            
            if primitive == "TYPE":
                
                if "target" in step and "value" in step.get("target", {}):
                    target_value = step.get("target", {}).get("value", "")
                    if target_value and target_value.isdigit():
                        matched_param = None
                        if step_idx in action_values and input_id_params_ordered:
                            action_element_id = action_values[step_idx].get("element_id")
                            if action_element_id == target_value and type_step_index < len(input_id_params_ordered):
                                matched_param = input_id_params_ordered[type_step_index]
                        if not matched_param and input_id_params_ordered and type_step_index < len(input_id_params_ordered):
                            matched_param = input_id_params_ordered[type_step_index]
                        if not matched_param and element_id_params:
                            matched_param = list(element_id_params.keys())[0]
                        if matched_param:
                            step["target"]["value"] = f"{ { {matched_param}} } "
                
                
                if "text" in step:
                    text_value = step.get("text", "")
                    if text_value and not text_value.startswith("{{"):
                        matched_param = None
                        if step_idx in action_values and text_params_ordered:
                            action_text = action_values[step_idx].get("text")
                            if action_text == text_value and type_step_index < len(text_params_ordered):
                                matched_param = text_params_ordered[type_step_index]
                        if not matched_param and text_params_ordered and type_step_index < len(text_params_ordered):
                            matched_param = text_params_ordered[type_step_index]
                        if not matched_param and text_params:
                            matched_param = list(text_params.keys())[0]
                        if matched_param:
                            step["text"] = f"{ { {matched_param}} } "
                
                type_step_index += 1
            
            
            elif primitive == "CLICK":
                
                if "target" in step and "value" in step.get("target", {}):
                    target_value = step.get("target", {}).get("value", "")
                    if target_value and target_value.isdigit():
                        matched_param = None
                        if step_idx in action_values and click_id_params_ordered and click_step_index < len(click_id_params_ordered):
                            action_element_id = action_values[step_idx].get("element_id")
                            if action_element_id == target_value:
                                matched_param = click_id_params_ordered[click_step_index]
                        if not matched_param and click_id_params_ordered and click_step_index < len(click_id_params_ordered):
                            matched_param = click_id_params_ordered[click_step_index]
                        if not matched_param and element_id_params:
                            matched_param = list(element_id_params.keys())[0]
                        if matched_param:
                            step["target"]["value"] = f"{ { {matched_param}} } "
                        click_step_index += 1
            
            
            elif primitive == "SELECT" and "option" in step:
                option_value = step.get("option", "")
                if option_value and not option_value.startswith("{{"):
                    
                    for param_name, param_def in parameters.items():
                        param_desc = param_def.get("description", "").lower()
                        if ("status" in param_desc or "type" in param_desc or 
                            "option" in param_desc or "select" in param_desc):
                            step["option"] = f"{ { {param_name}} } "
                            break
            
            
            elif primitive == "GOTO" and "url" in step:
                url_value = step.get("url", "")
                if url_value and not url_value.startswith("{{"):
                    
                    for param_name, param_def in parameters.items():
                        param_desc = param_def.get("description", "").lower()
                        if "url" in param_desc or "link" in param_desc:
                            step["url"] = f"{ { {param_name}} } "
                            break
        
        return parameterized_plan
    
    def _create_bindings(
        self,
        parameters: Dict[str, Any],
        plan: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """LLM mapping from parameters to plan step fields."""
        bindings = {}
        
        
        if not parameters:
            return bindings
        
        
        
        plan_str = json.dumps(plan, indent=2, ensure_ascii=False)
        param_names = list(parameters.keys())
        param_descriptions = {name: params.get("description", "") for name, params in parameters.items()}
        
        system_prompt = """You map parameters onto plan steps.

Each binding lists {step, field} pairs (0-based step index). Fields include text, target.value, url, option, etc.
A parameter may bind to multiple locations. Return JSON only."""

        user_prompt = f"""Create bindings for:

Parameters:
{json.dumps(param_descriptions, indent=2, ensure_ascii=False)}

Plan:
{plan_str}

Example:
{{
  "search_keyword": {{
    "bind_to": [{{"step": 0, "field": "text"}}]
  }},
  "status": {{
    "bind_to": [{{"step": 1, "field": "target.value"}}]
  }}
}}

JSON only, no prose."""
        
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = llm_utils.call_llm(self.llm_config, messages)
            response = response.strip()
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
            
            llm_bindings = json.loads(response)
            
            
            for param_name in parameters.keys():
                if param_name in llm_bindings:
                    bind_info = llm_bindings[param_name]
                    if "bind_to" in bind_info and isinstance(bind_info["bind_to"], list):
                        
                        valid_bindings = []
                        for bind_rule in bind_info["bind_to"]:
                            step_idx = bind_rule.get("step")
                            field = bind_rule.get("field")
                            if step_idx is not None and field and 0 <= step_idx < len(plan):
                                valid_bindings.append({
                                    "step": step_idx,
                                    "field": field
                                })
                        if valid_bindings:
                            bindings[param_name] = {
                                "bind_to": valid_bindings
                            }
        except Exception as e:
            print(f"[Warning] LLM binding generation failed: {e}, using fallback method")
            
            bindings = self._create_bindings_fallback(parameters, plan)
        
        
        if not bindings:
            bindings = self._create_bindings_fallback(parameters, plan)
        
        return bindings
    
    def _create_bindings_fallback(
        self,
        parameters: Dict[str, Any],
        plan: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Heuristic bindings by scanning {{param}} placeholders inside serialized steps."""
        bindings = {}
        
        
        for param_name in parameters.keys():
            param_bindings = []
            
            
            for step_idx, step in enumerate(plan):
                step_str = json.dumps(step)
                
                
                placeholder = f"{ { {param_name}} } "
                if placeholder in step_str:
                    
                    if "text" in step and placeholder in str(step.get("text", "")):
                        param_bindings.append({
                            "step": step_idx,
                            "field": "text"
                        })
                    elif "target" in step and "value" in step.get("target", {}):
                        if placeholder in str(step["target"].get("value", "")):
                            param_bindings.append({
                                "step": step_idx,
                                "field": "target.value"
                            })
                    elif "url" in step and placeholder in str(step.get("url", "")):
                        param_bindings.append({
                            "step": step_idx,
                            "field": "url"
                        })
                    elif "option" in step and placeholder in str(step.get("option", "")):
                        param_bindings.append({
                            "step": step_idx,
                            "field": "option"
                        })
                    elif "key_comb" in step and placeholder in str(step.get("key_comb", "")):
                        param_bindings.append({
                            "step": step_idx,
                            "field": "key_comb"
                        })
                    elif "answer" in step and placeholder in str(step.get("answer", "")):
                        param_bindings.append({
                            "step": step_idx,
                            "field": "answer"
                        })
            
            
            if not param_bindings:
                param_type = parameters[param_name].get("type", "string")
                param_desc = parameters[param_name].get("description", "").lower()
                
                
                for step_idx, step in enumerate(plan):
                    primitive = step.get("primitive", "").upper()
                    
                    
                    if ("search" in param_desc or "keyword" in param_desc or "text" in param_desc) and                       primitive == "TYPE" and "text" in step:
                        param_bindings.append({
                            "step": step_idx,
                            "field": "text"
                        })
                    
                    elif ("status" in param_desc or "type" in param_desc or "option" in param_desc) and                         primitive == "SELECT" and "option" in step:
                        param_bindings.append({
                            "step": step_idx,
                            "field": "option"
                        })
                    
                    elif ("url" in param_desc or "link" in param_desc) and                         primitive == "GOTO" and "url" in step:
                        param_bindings.append({
                            "step": step_idx,
                            "field": "url"
                        })
            
            if param_bindings:
                bindings[param_name] = {
                    "bind_to": param_bindings
                }
        
        return bindings
    
    def _generate_action_identity(
        self,
        context: Dict[str, Any],
        description: Dict[str, str],
        action_sequence: List[Any]
    ) -> tuple[str, str]:
        """Derive hierarchical action_id and PascalCase action_name."""
        site = context.get("site", "unknown")
        page = context.get("page", "unknown")
        summary = description.get("summary", "action")
        
        
        system_prompt = """You output a concise PascalCase verb+noun name for a UI routine.

Rules:
- 2-4 words, intent focused, no literals from the trace.
- Avoid generic tokens like Action/Do unless unavoidable.
- Never bake city names, ids, or keywords into the title.
- Avoid failure words unless the routine truly models recovery.

Return the name only."""
        
        user_prompt = f"""Name this routine (PascalCase):

Summary: {summary}
Context: {site} - {page}

Examples:
- "Filter orders by status" -> "FilterOrdersByStatus"
- "Search for products" -> "SearchProducts"

Name only, no punctuation."""
        
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = llm_utils.call_llm(self.llm_config, messages)
            action_name = response.strip()
            
            
            if "```" in action_name:
                action_name = action_name.split("```")[-1].strip()
            
            
            if not action_name or len(action_name) < 3:
                raise ValueError("Generated name too short")
            
            
            
            words = re.findall(r'\b\w+\b', action_name)
            if words:
                action_name = ''.join(word.capitalize() for word in words)
            else:
                
                action_name = summary.title().replace(" ", "").replace(".", "").replace(",", "")
                if not action_name:
                    action_name = "Action"
        except Exception as e:
            print(f"[Warning] Action name generation failed: {e}, using fallback")
            
            words = re.findall(r'\b\w+\b', summary)
            if words:
                
                key_words = [w.capitalize() for w in words[:4]]
                action_name = ''.join(key_words)
            else:
                action_name = summary.title().replace(" ", "").replace(".", "").replace(",", "")
            
            if not action_name or len(action_name) < 3:
                action_name = "Action"
        
        
        
        snake_case = re.sub(r'(?<!^)(?=[A-Z])', '_', action_name).lower()
        action_id = f"{site}.{page}.{snake_case}"
        
        return action_id, action_name
    
    def _is_failed_actspec(
        self,
        actspec: Dict[str, Any],
        action_sequence: List[Any],
        context: Dict[str, Any]
    ) -> tuple[bool, str, str]:
        """LLM classifier for failed / negative-constraint candidate traces."""
        
        
        plan = actspec.get("plan", [])
        
        
        
        for action in action_sequence:
            action_str = str(action).strip().lower()
            
            
            if action_str.startswith("stop") or "stop [" in action_str:
                print(f"[Filter] stop action detected, skipping failure classification: {action_str[:100]}")
                return False, "", "unspecified"
        
        
        for step in plan:
            primitive = step.get("primitive", "")
            raw = step.get("raw", "")
            if primitive == "STOP" or (isinstance(raw, str) and raw.strip().lower().startswith("stop")):
                print(f"[Filter] plan contains STOP primitive, skipping failure classification")
                return False, "", "unspecified"
        description = actspec.get("description", {})
        action_name = actspec.get("action_name", "")
        action_id = actspec.get("action_id", "")
        
        
        plan_str = json.dumps(plan, indent=2, ensure_ascii=False)
        description_str = json.dumps(description, indent=2, ensure_ascii=False)
        action_sequence_str = "\n".join([f"{i}: {str(a)}" for i, a in enumerate(action_sequence)])
        
        system_prompt = """You decide whether an ActSpec encodes a failed or incomplete interaction.

NOT failures:
- Explicit stop[...] primitives / actions are normal control flow (done, blocked, user abort)—set is_failed=false.
- note[...] inside UNKNOWN rows are memories for the LLM, not executable failures—set is_failed=false when that is all that remains.

Likely failures:
1. UNKNOWN primitives that are not benign notes.
2. Descriptions mentioning inability / errors / unfinished work (English or Chinese cues).
3. Names containing failed/error/attempt (unless clearly recovery flows).
4. Plans missing critical steps vs stated intent.
5. Clear context mismatch (wrong site/page assumptions).

constraint_subtype when is_failed=true:
- readiness: acted before UI stabilized (navigation/modals/refresh spam with no effect).
- disambiguation: repeated wrong picks among ambiguous similar targets.
- unspecified: everything else.

Always JSON only; stop actions => is_failed=false."""

        user_prompt = f"""Classify this ActSpec:

Action ID: {action_id}
Action Name: {action_name}

Context:
- Site: {context.get('site', 'unknown')}
- Page: {context.get('page', 'unknown')}
- URL Pattern: {context.get('url_pattern', '/')}

Actions:
{action_sequence_str}

Plan:
{plan_str}

Description:
{description_str}

Return JSON:
{{
  "is_failed": true,
  "failure_reason": "short explanation if failed else empty string",
  "constraint_subtype": "readiness | disambiguation | unspecified (required when is_failed)",
  "confidence": 0.85
}}

JSON only."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = llm_utils.call_llm(self.llm_config, messages)
            response = response.strip()
            
            
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
            
            result = json.loads(response)
            is_failed = result.get("is_failed", False)
            failure_reason = result.get("failure_reason", "")
            confidence = result.get("confidence", 0.5)
            constraint_subtype = (result.get("constraint_subtype") or "unspecified").strip().lower()
            if constraint_subtype not in ("readiness", "disambiguation", "unspecified"):
                constraint_subtype = "unspecified"
            
            
            if is_failed and confidence < 0.6:
                print(f"[Warning] Low-confidence failure verdict (confidence={confidence}): {action_id}")
            
            return is_failed, failure_reason, constraint_subtype
            
        except Exception as e:
            print(f"[Warning] LLM failure classifier error: {e}, using fallback")
            
            return self._is_failed_actspec_fallback(actspec), "", "unspecified"
    
    def _is_failed_actspec_fallback(self, actspec: Dict[str, Any]) -> bool:
        """Cheap UNKNOWN/stop heuristics when the LLM classifier fails."""
        plan = actspec.get("plan", [])
        
        
        for step in plan:
            primitive = step.get("primitive", "")
            raw = step.get("raw", "")
            if primitive == "STOP" or (isinstance(raw, str) and raw.strip().lower().startswith("stop")):
                
                return False
        
        
        for step in plan:
            if step.get("primitive") == "UNKNOWN":
                raw = step.get("raw", "") or ""
                if isinstance(raw, str) and raw.strip().lower().startswith("note"):
                    continue  
                return True
        
        
        action_name = actspec.get("action_name", "").lower()
        if any(keyword in action_name for keyword in ["failed", "error", "attempt"]):
            return True
        
        return False
    
    def _validate_actspec(self, actspec: Dict[str, Any]) -> List[str]:
        """Structural consistency checks for parameters, placeholders, and bindings."""
        errors = []
        parameters = actspec.get("parameters", {})
        plan = actspec.get("plan", [])
        bindings = actspec.get("bindings", {})
        
        
        for param_name in bindings.keys():
            if param_name not in parameters:
                errors.append(f"Binding references undefined parameter '{param_name}'")
        
        
        plan_str = json.dumps(plan, ensure_ascii=False)
        for param_name in parameters.keys():
            placeholder = f"{ { {param_name}} } "
            has_placeholder = placeholder in plan_str
            has_binding = param_name in bindings
            
            if not has_placeholder and not has_binding:
                errors.append(f"Parameter '{param_name}' lacks both placeholders and bindings")
        
        
        for step_idx, step in enumerate(plan):
            primitive = step.get("primitive", "").upper()
            
            if primitive == "TYPE" and "text" in step:
                text_value = str(step.get("text", ""))
                
                if text_value.startswith("{{") and text_value.endswith("}}"):
                    param_name = text_value[2:-2]
                    if param_name in parameters:
                        param_type = parameters[param_name].get("type", "string")
                        if param_type == "number":
                            errors.append(
                                f"plan[{step_idx}].text binds number-typed param '{param_name}' "
                                f"(expected string)"
                            )
            
            
            if primitive in ["CLICK", "TYPE"] and "target" in step:
                target_value = step.get("target", {}).get("value", "")
                if target_value and target_value.startswith("{{") and target_value.endswith("}}"):
                    param_name = target_value[2:-2]
                    if param_name in parameters:
                        param_type = parameters[param_name].get("type", "string")
                        if param_type != "number":
                            errors.append(
                                f"plan[{step_idx}].target.value uses {param_type} param '{param_name}' "
                                f"(expected number element id)"
                            )
        
        
        for param_name, binding_info in bindings.items():
            bind_to = binding_info.get("bind_to", [])
            for bind_rule in bind_to:
                step_idx = bind_rule.get("step")
                if step_idx is None or step_idx < 0 or step_idx >= len(plan):
                    errors.append(
                        f"Binding for '{param_name}' references invalid step {step_idx} "
                        f"(plan length {len(plan)})"
                    )
        
        return errors
    
    def _auto_fix_actspec(self, actspec: Dict[str, Any], errors: List[str]) -> Dict[str, Any]:
        """Patch obvious parameter/type mismatches reported by validation."""
        import copy
        fixed_actspec = copy.deepcopy(actspec)
        parameters = fixed_actspec.get("parameters", {})
        plan = fixed_actspec.get("plan", [])
        bindings = fixed_actspec.get("bindings", {})
        
        
        for step_idx, step in enumerate(plan):
            primitive = step.get("primitive", "").upper()
            
            if primitive == "TYPE" and "text" in step:
                text_value = str(step.get("text", ""))
                if text_value.startswith("{{") and text_value.endswith("}}"):
                    param_name = text_value[2:-2]
                    if param_name in parameters:
                        param_type = parameters[param_name].get("type", "string")
                        if param_type == "number":
                            
                            for p_name, p_def in parameters.items():
                                if p_def.get("type") == "string" and (
                                    "text" in p_name.lower() or
                                    "keyword" in p_name.lower() or
                                    "search" in p_name.lower() or
                                    "input" in p_name.lower()
                                ):
                                    step["text"] = f"{ { {p_name}} } "
                                    
                                    if p_name not in bindings:
                                        bindings[p_name] = {"bind_to": []}
                                    bindings[p_name]["bind_to"].append({
                                        "step": step_idx,
                                        "field": "text"
                                    })
                                    
                                    if param_name in bindings:
                                        bindings[param_name]["bind_to"] = [
                                            b for b in bindings[param_name]["bind_to"]
                                            if not (b.get("step") == step_idx and b.get("field") == "text")
                                        ]
                                    break
        
        
        for step_idx, step in enumerate(plan):
            primitive = step.get("primitive", "").upper()
            
            if primitive in ["CLICK", "TYPE"] and "target" in step:
                target_value = step.get("target", {}).get("value", "")
                if target_value and target_value.startswith("{{") and target_value.endswith("}}"):
                    param_name = target_value[2:-2]
                    if param_name in parameters:
                        param_type = parameters[param_name].get("type", "string")
                        if param_type != "number":
                            
                            for p_name, p_def in parameters.items():
                                if p_def.get("type") == "number" and (
                                    "id" in p_name.lower() or
                                    "element" in p_name.lower() or
                                    "button" in p_name.lower() or
                                    "click" in p_name.lower() or
                                    "type" in p_name.lower() or
                                    "input" in p_name.lower()
                                ):
                                    step["target"]["value"] = f"{ { {p_name}} } "
                                    
                                    if p_name not in bindings:
                                        bindings[p_name] = {"bind_to": []}
                                    bindings[p_name]["bind_to"].append({
                                        "step": step_idx,
                                        "field": "target.value"
                                    })
                                    
                                    if param_name in bindings:
                                        bindings[param_name]["bind_to"] = [
                                            b for b in bindings[param_name]["bind_to"]
                                            if not (b.get("step") == step_idx and b.get("field") == "target.value")
                                        ]
                                    break
        
        fixed_actspec["bindings"] = bindings
        return fixed_actspec
    
    def _post_process_parameterize(
        self,
        actspec: Dict[str, Any],
        action_sequence: List[Any]
    ) -> Dict[str, Any]:
        """Second pass: replace leftover literals with {{param}} using heuristics + optional LLM."""
        import copy
        fixed_actspec = copy.deepcopy(actspec)
        plan = fixed_actspec.get("plan", [])
        parameters = fixed_actspec.get("parameters", {})
        bindings = fixed_actspec.get("bindings", {})
        
        if not plan or not parameters:
            return fixed_actspec
        
        
        has_unparameterized = False
        for step in plan:
            
            if "target" in step and "value" in step.get("target", {}):
                value = step.get("target", {}).get("value", "")
                if value and not self._is_placeholder(value):
                    has_unparameterized = True
                    break
            
            if "text" in step:
                text = step.get("text", "")
                if text and not self._is_placeholder(text):
                    has_unparameterized = True
                    break
            
            if "option" in step:
                option = step.get("option", "")
                if option and not self._is_placeholder(option):
                    has_unparameterized = True
                    break
            
            if "url" in step:
                url = step.get("url", "")
                if url and not self._is_placeholder(url):
                    has_unparameterized = True
                    break
        
        if not has_unparameterized:
            return fixed_actspec
        
        
        print(f"[PostProcess] Unparameterized literals detected, repairing ActSpec: {actspec.get('action_id', 'unknown')}")
        
        
        action_values = {}
        for i, action in enumerate(action_sequence):
            action_str = str(action)
            
            id_match = re.search(r"\[(\d+)\]", action_str)
            if id_match:
                element_id = id_match.group(1)
                action_values[i] = {"element_id": element_id}
            
            type_match = re.search(r"type\s*\[.*?\]\s*\[(.*?)\]\s*\[.*?\]", action_str)
            if type_match:
                text = type_match.group(1).strip()
                if text:
                    if i not in action_values:
                        action_values[i] = {}
                    action_values[i]["text"] = text
        
        
        element_id_params = {}  
        text_params = {}  
        enum_params = {}  
        url_params = {}  
        
        for param_name, param_def in parameters.items():
            param_type = param_def.get("type", "string")
            param_desc = param_def.get("description", "").lower()
            param_name_lower = param_name.lower()
            
            
            is_element_id_param = False
            if param_type == "number" and (
                "id" in param_name_lower or
                "element" in param_name_lower or
                "button" in param_name_lower or
                "click" in param_name_lower or
                "type" in param_name_lower or
                "input" in param_name_lower
            ):
                is_element_id_param = True
            elif param_type == "enum":
                
                candidates = param_def.get("candidates", [])
                if candidates and all(isinstance(c, (int, float)) or (isinstance(c, str) and c.isdigit()) for c in candidates):
                    
                    if ("id" in param_name_lower or
                        "element" in param_name_lower or
                        "button" in param_name_lower or
                        "click" in param_name_lower or
                        "type" in param_name_lower or
                        "input" in param_name_lower):
                        is_element_id_param = True
                    else:
                        enum_params[param_name] = param_def
                else:
                    enum_params[param_name] = param_def
            
            if is_element_id_param:
                element_id_params[param_name] = param_def
            elif param_type == "string" and (
                "text" in param_name_lower or
                "keyword" in param_name_lower or
                "search" in param_name_lower or
                "input" in param_name_lower
            ):
                text_params[param_name] = param_def
            elif param_type == "string" and (
                "url" in param_name_lower or
                "link" in param_name_lower
            ):
                url_params[param_name] = param_def
        
        
        for step_idx, step in enumerate(plan):
            primitive = step.get("primitive", "").upper()
            
            
            if "target" in step and "value" in step.get("target", {}):
                target = step.get("target", {})
                strategy = target.get("strategy", "")
                value = target.get("value", "")
                
                if value and not self._is_placeholder(value):
                    
                    if strategy == "element_id":
                        
                        matched_param = None
                        
                        
                        if step_idx in action_values:
                            action_element_id = action_values[step_idx].get("element_id")
                            if action_element_id == str(value):
                                
                                if primitive == "CLICK":
                                    for param_name in element_id_params:
                                        if "click" in param_name.lower() or "button" in param_name.lower():
                                            matched_param = param_name
                                            break
                                elif primitive == "TYPE":
                                    for param_name in element_id_params:
                                        if "type" in param_name.lower() or "input" in param_name.lower():
                                            matched_param = param_name
                                            break
                        
                        
                        if not matched_param and element_id_params:
                            matched_param = list(element_id_params.keys())[0]
                        
                        
                        if not matched_param:
                            
                            if primitive == "CLICK":
                                param_name = "click_id"
                            elif primitive == "TYPE":
                                param_name = "input_id"
                            elif primitive == "HOVER":
                                param_name = "hover_id"
                            elif primitive == "SELECT":
                                param_name = "select_id"
                            else:
                                param_name = "element_id"
                            
                            
                            base_name = param_name
                            counter = 1
                            while param_name in parameters:
                                param_name = f"{base_name}_{counter}"
                                counter += 1
                            
                            
                            parameters[param_name] = {
                                "type": "number",
                                "description": f"element id for {primitive.lower()} operation",
                                "optional": False
                            }
                            element_id_params[param_name] = parameters[param_name]
                            matched_param = param_name
                            print(f"[PostProcess] Auto-created parameter: {param_name}")
                        
                        if matched_param:
                            target["value"] = f"{ { {matched_param}} } "
                            
                            if matched_param not in bindings:
                                bindings[matched_param] = {"bind_to": []}
                            
                            existing_binding = False
                            for bind_rule in bindings[matched_param]["bind_to"]:
                                if bind_rule.get("step") == step_idx and bind_rule.get("field") == "target.value":
                                    existing_binding = True
                                    break
                            if not existing_binding:
                                bindings[matched_param]["bind_to"].append({
                                    "step": step_idx,
                                    "field": "target.value"
                                })
                            _ph = "{{" + str(matched_param) + "}}"
                            print(f"[PostProcess] Fixed step[{step_idx}].target.value: {value!r} -> {_ph!r}")
            
            
            if "text" in step:
                text = step.get("text", "")
                if text and not self._is_placeholder(text):
                    matched_param = None
                    
                    
                    if step_idx in action_values:
                        action_text = action_values[step_idx].get("text")
                        if action_text == text:
                            
                            for param_name in text_params:
                                if "search" in param_name.lower() or "keyword" in param_name.lower():
                                    matched_param = param_name
                                    break
                                elif "input_text" in param_name.lower() or "text" in param_name.lower():
                                    if not matched_param:
                                        matched_param = param_name
                    
                    
                    if not matched_param and text_params:
                        matched_param = list(text_params.keys())[0]
                    
                    
                    if not matched_param:
                        matched_param = self._llm_match_parameter(
                            text, "text", primitive, text_params, parameters
                        )
                    
                    if matched_param:
                        step["text"] = f"{ { {matched_param}} } "
                        
                        if matched_param not in bindings:
                            bindings[matched_param] = {"bind_to": []}
                        
                        existing_binding = False
                        for bind_rule in bindings[matched_param]["bind_to"]:
                            if bind_rule.get("step") == step_idx and bind_rule.get("field") == "text":
                                existing_binding = True
                                break
                        if not existing_binding:
                            bindings[matched_param]["bind_to"].append({
                                "step": step_idx,
                                "field": "text"
                            })
                        _ph = "{{" + str(matched_param) + "}}"
                        print(f"[PostProcess] Fixed step[{step_idx}].text: {text!r} -> {_ph!r}")
            
            
            if "option" in step:
                option = step.get("option", "")
                if option and not self._is_placeholder(option):
                    matched_param = None
                    
                    
                    if enum_params:
                        matched_param = list(enum_params.keys())[0]
                    else:
                        
                        matched_param = self._llm_match_parameter(
                            option, "option", primitive, parameters, parameters
                        )
                    
                    if matched_param:
                        step["option"] = f"{ { {matched_param}} } "
                        
                        if matched_param not in bindings:
                            bindings[matched_param] = {"bind_to": []}
                        
                        existing_binding = False
                        for bind_rule in bindings[matched_param]["bind_to"]:
                            if bind_rule.get("step") == step_idx and bind_rule.get("field") == "option":
                                existing_binding = True
                                break
                        if not existing_binding:
                            bindings[matched_param]["bind_to"].append({
                                "step": step_idx,
                                "field": "option"
                            })
                        _ph = "{{" + str(matched_param) + "}}"
                        print(f"[PostProcess] Fixed step[{step_idx}].option: {option!r} -> {_ph!r}")
            
            
            if "url" in step:
                url = step.get("url", "")
                if url and not self._is_placeholder(url):
                    matched_param = None
                    
                    
                    if url_params:
                        matched_param = list(url_params.keys())[0]
                    else:
                        
                        matched_param = self._llm_match_parameter(
                            url, "url", primitive, parameters, parameters
                        )
                    
                    if matched_param:
                        step["url"] = f"{ { {matched_param}} } "
                        
                        if matched_param not in bindings:
                            bindings[matched_param] = {"bind_to": []}
                        
                        existing_binding = False
                        for bind_rule in bindings[matched_param]["bind_to"]:
                            if bind_rule.get("step") == step_idx and bind_rule.get("field") == "url":
                                existing_binding = True
                                break
                        if not existing_binding:
                            bindings[matched_param]["bind_to"].append({
                                "step": step_idx,
                                "field": "url"
                            })
                        _ph = "{{" + str(matched_param) + "}}"
                        print(f"[PostProcess] Fixed step[{step_idx}].url: {url!r} -> {_ph!r}")
        
        fixed_actspec["plan"] = plan
        fixed_actspec["bindings"] = bindings
        return fixed_actspec
    
    def _get_value_from_action_at_step(
        self,
        action_sequence: List[Any],
        step_idx: int,
        field: str
    ) -> Optional[Any]:
        """Pull bound fields from primitive strings for candidate population."""
        if step_idx < 0 or step_idx >= len(action_sequence):
            return None
        action_str = str(action_sequence[step_idx]).strip()
        if field == "target.value":
            
            m = re.search(r"(?:click|type|hover|select)\s*\[(\d+)\]", action_str, re.I)
            if m:
                return int(m.group(1))  
            return None
        if field == "text":
            m = re.search(r"type\s*\[.*?\]\s*\[(.*?)\]\s*\[.*?\]", action_str)
            if m:
                return m.group(1).strip()
            return None
        if field == "url":
            m = re.search(r"goto\s*\[(.*?)\]", action_str)
            if m:
                return m.group(1).strip()
            return None
        return None
    
    def _fill_parameter_candidates_from_trajectory(
        self,
        actspec: Dict[str, Any],
        action_sequence: List[Any]
    ) -> Dict[str, Any]:
        """Populate parameters[].candidates from trajectory using bindings for replay."""
        import copy
        fixed = copy.deepcopy(actspec)
        parameters = fixed.get("parameters", {})
        bindings = fixed.get("bindings", {})
        if not parameters or not bindings:
            return fixed
        for param_name, param_def in list(parameters.items()):
            binding = bindings.get(param_name, {}).get("bind_to", [])
            if not binding:
                continue
            values = []
            for b in binding:
                step_idx = b.get("step")
                field = b.get("field")
                if step_idx is None or not field:
                    continue
                val = self._get_value_from_action_at_step(action_sequence, step_idx, field)
                if val is not None:
                    values.append(val)
            if not values:
                continue
            
            seen = set()
            unique = []
            for v in values:
                key = (v, type(v))
                if key not in seen:
                    seen.add(key)
                    unique.append(v)
            param_type = param_def.get("type", "string")
            if param_type == "number" and unique:
                
                try:
                    unique = [int(u) if isinstance(u, str) and u.isdigit() else u for u in unique]
                except (ValueError, TypeError):
                    pass
            fixed["parameters"][param_name] = {**param_def, "candidates": unique}
            if param_type == "enum":
                fixed["parameters"][param_name]["type"] = "enum"
        return fixed
    
    def _is_placeholder(self, value: Any) -> bool:
        """True if value looks like {{param}}."""
        if not isinstance(value, str):
            return False
        return value.startswith("{{") and value.endswith("}}") and len(value) > 4
    
    def _llm_match_parameter(
        self,
        value: Any,
        field_type: str,
        primitive: str,
        candidate_params: Dict[str, Any],
        all_params: Dict[str, Any]
    ) -> Optional[str]:
        """Ask LLM to pick best parameter name among candidates."""
        if not candidate_params:
            return None
        
        
        if len(candidate_params) == 1:
            return list(candidate_params.keys())[0]
        
        
        param_descriptions = {}
        for param_name, param_def in candidate_params.items():
            param_descriptions[param_name] = {
                "type": param_def.get("type", "string"),
                "description": param_def.get("description", ""),
                "candidates": param_def.get("candidates", []) if param_def.get("type") == "enum" else None
            }
        
        system_prompt = """Pick exactly one parameter key from the candidate JSON.

Respect types: element ids -> number params, free text -> string, SELECT options -> enum when present, URLs -> string.
Return only the parameter name, nothing else."""

        user_prompt = f"""Observed value: {value}
Field kind: {field_type}
Primitive: {primitive}

Candidates:
{json.dumps(param_descriptions, indent=2, ensure_ascii=False)}

Return the best parameter name only."""
        
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = llm_utils.call_llm(self.llm_config, messages)
            response = response.strip()
            
            
            if "```" in response:
                response = response.split("```")[-1].strip()
            
            
            if response in candidate_params:
                return response
            else:
                
                print(f"[Warning] LLM picked unknown param '{response}', falling back to first candidate")
                return list(candidate_params.keys())[0]
        except Exception as e:
            print(f"[Warning] LLM parameter match failed: {e}, using first candidate")
            return list(candidate_params.keys())[0] if candidate_params else None
    
    def _extract_locate_strategies(
        self,
        action_sequence: List[Any],
        trajectory: List[Dict[str, Any]],
        segment_start_idx: int,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build locate.target_entries with semantic / relative / id strategies."""
        from .element_context_extractor import ElementContextExtractor
        
        extractor = ElementContextExtractor()
        locate = {
            "target_elements": []
        }
        
        
        for action_idx, action in enumerate(action_sequence):
            action_str = str(action)
            
            
            element_ids = self._extract_element_ids_from_action(action_str)
            
            
            
            trajectory_idx = self._find_action_in_trajectory(
                action_str, trajectory, segment_start_idx
            )
            
            if trajectory_idx is None:
                
                trajectory_idx = segment_start_idx + action_idx
                if trajectory_idx >= len(trajectory):
                    continue
            
            for element_id in element_ids:
                
                element_context = extractor.extract_element_context(
                    trajectory, trajectory_idx, element_id
                )
                
                if not element_context:
                    continue
                
                
                param_name = self._find_matching_param_name(element_id, parameters, action_str)
                
                
                strategies = self._generate_locate_strategies(
                    element_context, element_id, action_str, param_name
                )
                
                locate["target_elements"].append({
                    "element_id": f"{ { {param_name}} } ",
                    "strategies": strategies
                })
        
        
        if locate.get("target_elements"):
            try:
                locate = self._llm_enhance_locate_strategies(
                    locate, action_sequence, trajectory, segment_start_idx
                )
            except Exception as e:
                print(f"[Warning] LLM locate augmentation failed: {e}")
        
        return locate
    
    def _extract_element_ids_from_action(self, action_str: str) -> List[str]:
        """Collect numeric ids referenced in a primitive string."""
        element_ids = []
        
        matches = re.findall(r"(?:click|type|hover|select)\s*\[(\d+)\]", action_str, re.I)
        element_ids.extend(matches)
        return element_ids
    
    def _get_element_param_name(self, element_id: str, action_str: str) -> str:
        """Derive default parameter names from primitive kind + element id."""
        action_lower = action_str.lower()
        if "click" in action_lower:
            return "click_id"
        elif "type" in action_lower:
            return "input_id"
        elif "hover" in action_lower:
            return "hover_id"
        elif "select" in action_lower:
            return "select_id"
        else:
            return "element_id"
    
    def _find_matching_param_name(
        self,
        element_id: str,
        parameters: Dict[str, Any],
        action_str: str
    ) -> str:
        """
        Look up an existing parameter tied to this element id; otherwise synthesize a name.
        """
        
        for param_name, param_def in parameters.items():
            candidates = param_def.get("candidates", [])
            
            if candidates:
                candidate_strs = [str(c) for c in candidates]
                if element_id in candidate_strs:
                    return param_name
        
        
        return self._get_element_param_name(element_id, action_str)
    
    def _generate_locate_strategies(
        self,
        element_context: Dict[str, Any],
        element_id: str,
        action_str: str,
        param_name: str
    ) -> List[Dict[str, Any]]:
        """
        Ordered locate strategies: semantic first, relative second, raw element_id last.
        """
        strategies = []
        
        semantic_features = element_context.get("semantic_features", {})
        relative_context = element_context.get("relative_context", {})
        
        
        if semantic_features.get("role") or semantic_features.get("label"):
            semantic_strategy = {
                "strategy": "semantic",
                "priority": 1,
                "conditions": {}
            }
            
            
            if semantic_features.get("role"):
                semantic_strategy["conditions"]["role"] = semantic_features["role"]
            if semantic_features.get("label"):
                semantic_strategy["conditions"]["label"] = semantic_features["label"]
            if semantic_features.get("text"):
                semantic_strategy["conditions"]["text"] = semantic_features["text"]
            
            
            context_dict = {}
            if relative_context.get("form"):
                context_dict["parent_form"] = relative_context["form"].get("id") or relative_context["form"].get("label")
            if relative_context.get("modal"):
                context_dict["parent_modal"] = relative_context["modal"].get("id")
            if relative_context.get("region"):
                context_dict["parent_region"] = relative_context["region"]
            
            if context_dict:
                semantic_strategy["context"] = context_dict
            
            strategies.append(semantic_strategy)
        
        
        
        
        reference_element = None
        reference_type = None
        
        
        parent = relative_context.get("parent")
        if parent and parent.get("element_id"):
            reference_element = parent
            reference_type = "parent"
        
        elif relative_context.get("siblings"):
            siblings = relative_context["siblings"]
            
            priority_roles = ["image", "heading", "button", "link"]
            for role in priority_roles:
                for sibling in siblings:
                    if sibling.get("role") == role and sibling.get("element_id"):
                        reference_element = sibling
                        reference_type = "sibling"
                        break
                if reference_element:
                    break
            
            
            if not reference_element:
                for sibling in siblings:
                    sibling_id = sibling.get("element_id")
                    if sibling_id:
                        reference_element = sibling
                        reference_type = "sibling"
                        break
        
        if reference_element and reference_element.get("element_id"):
            relative_strategy = {
                "strategy": "relative_position",
                "priority": 2,
                "conditions": {
                    "relative_to": {
                        "element_id": reference_element.get("element_id"),  
                        "position": "after" if reference_type == "sibling" else "inside",
                        "distance": "adjacent"
                    },
                    "sibling_info": {
                        "role": reference_element.get("role"),
                        "label": reference_element.get("label")
                    }
                }
            }
            strategies.append(relative_strategy)
        
        
        
        element_id_strategy = {
            "strategy": "element_id",
            "priority": 3,
            "conditions": {
                "element_id": f"{ { {param_name}} } "
            }
        }
        strategies.append(element_id_strategy)
        
        return strategies
    
    def _llm_enhance_locate_strategies(
        self,
        locate: Dict[str, Any],
        action_sequence: List[Any],
        trajectory: List[Dict[str, Any]],
        segment_start_idx: int
    ) -> Dict[str, Any]:
        """
        Optional LLM pass to refine locate strategies for messy UIs.
        """
        
        
        return locate
    
    def _extract_pre_condition(
        self,
        action_sequence: List[Any],
        trajectory: List[Dict[str, Any]],
        segment_start_idx: int,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build pre-condition JSON prior to executing the segment.
        """
        
        pre_step = trajectory[segment_start_idx] if segment_start_idx < len(trajectory) else None
        if not pre_step:
            return {}
        
        pre_obs_text = pre_step.get("observation", "")
        pre_url = pre_step.get("url", "")
        
        pre_condition = {
            "url_pattern": context.get("url_pattern", ""),
            "required_elements": [],
            "required_regions": [],
            "required_modals": [],
            "page_ready": True,
            "excluded_states": []
        }
        
        
        
        required_elements = self._infer_required_elements(action_sequence, pre_obs_text)
        pre_condition["required_elements"] = required_elements
        
        
        required_regions = self._infer_required_regions(action_sequence, pre_obs_text)
        pre_condition["required_regions"] = required_regions
        
        
        has_modal = self._check_modal_exists(pre_obs_text)
        if has_modal:
            pre_condition["required_modals"] = [{"exists": True}]
        else:
            pre_condition["required_modals"] = []
        
        
        excluded_states = self._infer_excluded_states(action_sequence, pre_obs_text, context, trajectory, segment_start_idx)
        pre_condition["excluded_states"] = excluded_states
        
        
        try:
            pre_condition = self._llm_enhance_pre_condition(
                pre_condition, action_sequence, pre_obs_text, context
            )
        except Exception as e:
            print(f"[Warning] LLM pre-condition augmentation failed: {e}")
        
        return pre_condition
    
    def _infer_required_elements(
        self,
        action_sequence: List[Any],
        observation_text: str
    ) -> List[Dict[str, Any]]:
        """Heuristic required_elements derived from primitives."""
        from .accessibility_tree_parser import AccessibilityTreeParser
        
        parser = AccessibilityTreeParser()
        tree = parser.parse(observation_text)
        
        required_elements = []
        seen_signatures = set()  
        
        for action in action_sequence:
            action_str = str(action)
            
            
            element_ids = self._extract_element_ids_from_action(action_str)
            
            for element_id in element_ids:
                
                element = parser.find_element_by_id(tree, element_id)
                
                if element:
                    
                    strategy = {
                        "strategy": "semantic",
                        "conditions": {}
                    }
                    
                    if element.get("role"):
                        strategy["conditions"]["role"] = element["role"]
                    if element.get("label"):
                        strategy["conditions"]["label"] = element["label"]
                    
                    if element.get("text"):
                        strategy["conditions"]["text"] = element["text"]
                    
                    
                    signature = self._get_element_strategy_signature(strategy)
                    if signature not in seen_signatures:
                        required_elements.append(strategy)
                        seen_signatures.add(signature)
        
        return required_elements
    
    def _get_element_strategy_signature(self, strategy: Dict[str, Any]) -> str:
        """Fingerprint semantic locate rows for deduping."""
        strategy_type = strategy.get("strategy", "")
        conditions = strategy.get("conditions", {})
        
        conditions_str = ":".join(sorted([f"{k}={v}" for k, v in conditions.items()]))
        return f"{strategy_type}:{conditions_str}"
    
    def _infer_required_regions(
        self,
        action_sequence: List[Any],
        observation_text: str
    ) -> List[Dict[str, Any]]:
        """
        Infer coarse required_regions hints from the accessibility tree snapshot.
        """
        from .accessibility_tree_parser import AccessibilityTreeParser
        
        if not observation_text:
            return []
        
        try:
            parser = AccessibilityTreeParser()
            tree = parser.parse(observation_text)
            
            required_regions = []
            seen_regions = set()
            
            
            for action in action_sequence:
                action_str = str(action)
                element_ids = self._extract_element_ids_from_action(action_str)
                
                for element_id in element_ids:
                    
                    element = parser.find_element_by_id(tree, element_id)
                    if not element:
                        continue
                    
                    context = parser.get_element_context(tree, element_id)
                    region = context.get("region")
                    
                    
                    if region and region not in seen_regions:
                        required_regions.append({
                            "type": "semantic",
                            "role": region,  
                            "description": f"Required region: {region}"
                        })
                        seen_regions.add(region)
            
            
            if not required_regions:
                
                for element in tree.get("elements", []):
                    role = element.get("role", "")
                    if role in ["main", "complementary", "header", "footer", "navigation", "aside"]:
                        if role not in seen_regions:
                            required_regions.append({
                                "type": "semantic",
                                "role": role,
                                "description": f"Required region: {role}"
                            })
                            seen_regions.add(role)
            
            return required_regions
            
        except Exception as e:
            print(f"[Warning] Failed while inferring required regions: {e}")
            return []
    
    def _check_modal_exists(self, observation_text: str) -> bool:
        """
        Detect modal/dialog cues from accessibility snapshots (keywords + structure).
        """
        if not observation_text:
            return False
        
        
        modal_keywords = ["modal", "dialog", "popup", "overlay", "alertdialog"]
        observation_lower = observation_text.lower()
        if any(keyword in observation_lower for keyword in modal_keywords):
            return True
        
        
        try:
            from .accessibility_tree_parser import AccessibilityTreeParser
            
            parser = AccessibilityTreeParser()
            tree = parser.parse(observation_text)
            
            
            for element in tree.get("elements", []):
                role = element.get("role", "").lower()
                if role in ["dialog", "alertdialog"]:
                    return True
                
                
                label = element.get("label", "").lower()
                if any(keyword in label for keyword in ["modal", "dialog", "popup"]):
                    return True
            
            return False
            
        except Exception:
            
            return any(keyword in observation_lower for keyword in modal_keywords)
    
    def _llm_enhance_pre_condition(
        self,
        pre_condition: Dict[str, Any],
        action_sequence: List[Any],
        observation_text: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        LLM-assisted pre-condition enrichment.
        """
        
        
        return pre_condition
    
    def _extract_post_condition(
        self,
        action_sequence: List[Any],
        trajectory: List[Dict[str, Any]],
        segment_start_idx: int,
        segment_end_idx: int,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Construct post-conditions from before/after observations.
        """
        
        pre_step = trajectory[segment_start_idx] if segment_start_idx < len(trajectory) else None
        post_step = trajectory[segment_end_idx] if segment_end_idx < len(trajectory) else None
        
        if not pre_step or not post_step:
            return {}
        
        pre_url = pre_step.get("url", "")
        post_url = post_step.get("url", "")
        pre_obs_text = pre_step.get("observation", "")
        post_obs_text = post_step.get("observation", "")
        
        post_condition = {
            "url_change": {},
            "element_appears": [],
            "element_disappears": [],
            "text_appears": [],
            "region_updates": [],
            "modal_changes": []
        }
        
        
        if pre_url != post_url:
            post_condition["url_change"] = {
                "pattern": self._extract_url_pattern(post_url, context),
                "type": "navigate"
            }
        else:
            post_condition["url_change"] = {
                "type": "stay"
            }
        
        
        new_elements = self._detect_new_elements(pre_obs_text, post_obs_text)
        post_condition["element_appears"] = new_elements
        
        
        disappeared_elements = self._detect_disappeared_elements(pre_obs_text, post_obs_text)
        post_condition["element_disappears"] = disappeared_elements
        
        
        new_texts = self._detect_new_texts(pre_obs_text, post_obs_text)
        post_condition["text_appears"] = new_texts
        
        
        region_updates = self._detect_region_updates(pre_obs_text, post_obs_text)
        post_condition["region_updates"] = region_updates
        
        
        modal_changes = self._detect_modal_changes(pre_obs_text, post_obs_text)
        post_condition["modal_changes"] = modal_changes
        
        
        try:
            post_condition = self._llm_enhance_post_condition(
                post_condition, action_sequence, pre_obs_text, post_obs_text, context
            )
        except Exception as e:
            print(f"[Warning] LLM post-condition augmentation failed: {e}")
        
        return post_condition
    
    def _infer_excluded_states(
        self,
        action_sequence: List[Any],
        observation_text: str,
        context: Dict[str, Any],
        trajectory: List[Dict[str, Any]],
        segment_start_idx: int
    ) -> List[Dict[str, Any]]:
        """
        Heuristic excluded_states (error splash, loading gates, etc.).
        """
        excluded_states = []
        
        if not observation_text:
            return excluded_states
        
        from .accessibility_tree_parser import AccessibilityTreeParser
        
        try:
            parser = AccessibilityTreeParser()
            tree = parser.parse(observation_text)
            
            
            error_keywords = [
                "error", "failed", "invalid",
                "exception", "not found", "404", "500"
            ]
            observation_lower = observation_text.lower()
            
            
            has_error = any(keyword in observation_lower for keyword in error_keywords)
            if has_error:
                
                for element in tree.get("elements", []):
                    role = element.get("role", "").lower()
                    label = element.get("label", "").lower()
                    text = element.get("text", "").lower()
                    
                    if any(keyword in role or keyword in label or keyword in text 
                           for keyword in error_keywords):
                        excluded_states.append({
                            "type": "error",
                            "description": "Page contains error messages",
                            "element": {
                                "role": element.get("role"),
                                "label": element.get("label")
                            }
                        })
                        break
            
            
            loading_keywords = ["loading", "please wait"]
            has_loading = any(keyword in observation_lower for keyword in loading_keywords)
            if has_loading:
                excluded_states.append({
                    "type": "loading",
                    "description": "Page is in loading state"
                })
            
            
            
            current_url = trajectory[segment_start_idx].get("url", "") if segment_start_idx < len(trajectory) else ""
            error_url_patterns = [
                r"/error",
                r"/404",
                r"/500",
                r"/not-found",
                r"error=true"
            ]
            import re
            for pattern in error_url_patterns:
                if re.search(pattern, current_url, re.IGNORECASE):
                    excluded_states.append({
                        "type": "error_url",
                        "url_pattern": pattern,
                        "description": f"URL matches error pattern: {pattern}"
                    })
                    break
            
            
            
            for element in tree.get("elements", []):
                role = element.get("role", "").lower()
                if role == "form":
                    
                    children = element.get("children", [])
                    for child in children:
                        child_role = child.get("role", "").lower()
                        child_label = child.get("label", "").lower()
                        child_text = child.get("text", "").lower()
                        
                        if (child_role in ["alert", "alertdialog"] or
                            any(keyword in child_label or keyword in child_text 
                                for keyword in ["invalid", "required"])):
                            excluded_states.append({
                                "type": "form_validation_error",
                                "description": "Form contains validation errors",
                                "form_id": element.get("element_id")
                            })
                            break
            
        except Exception as e:
            print(f"[Warning] Failed while inferring excluded states: {e}")
        
        return excluded_states
    
    def _extract_url_pattern(self, url: str, context: Dict[str, Any]) -> str:
        """Normalize URL into a reusable pattern string."""
        
        url_pattern = context.get("url_pattern", "")
        if not url_pattern and url:
            
            from urllib.parse import urlparse
            parsed = urlparse(url)
            url_pattern = parsed.path or "/"
        return url_pattern
    
    def _detect_new_elements(
        self,
        pre_obs_text: str,
        post_obs_text: str
    ) -> List[Dict[str, Any]]:
        """
        Diff helper: treat differing signatures or ids as new nodes post-action.
        """
        from .accessibility_tree_parser import AccessibilityTreeParser
        
        if not pre_obs_text or not post_obs_text:
            return []
        
        try:
            parser = AccessibilityTreeParser()
            pre_tree = parser.parse(pre_obs_text)
            post_tree = parser.parse(post_obs_text)
            
            
            pre_signatures_full = set()  
            pre_element_ids = set()  
            
            for e in pre_tree.get("elements", []):
                if e.get("element_id"):
                    signature_full = self._get_element_signature(e)
                    pre_signatures_full.add(signature_full)
                    pre_element_ids.add(e.get("element_id"))
            
            new_elements = []
            seen_signatures = set()  
            
            for post_element in post_tree.get("elements", []):
                if not post_element.get("element_id"):
                    continue
                
                signature_full = self._get_element_signature(post_element)
                post_element_id = post_element.get("element_id")
                
                
                
                is_new = (signature_full not in pre_signatures_full and
                         signature_full not in seen_signatures)
                
                
                if not is_new and post_element_id not in pre_element_ids:
                    is_new = True
                
                if is_new:
                    
                    label = (post_element.get("label") or "").strip()
                    text = (post_element.get("text") or "").strip()
                    if self._is_error_or_failure_text(label) or self._is_error_or_failure_text(text):
                        seen_signatures.add(signature_full)
                        continue
                    
                    strategy = {
                        "strategy": "semantic",
                        "conditions": {}
                    }
                    
                    if post_element.get("role"):
                        strategy["conditions"]["role"] = post_element["role"]
                    if post_element.get("label"):
                        strategy["conditions"]["label"] = post_element["label"]
                    if post_element.get("text"):
                        strategy["conditions"]["text"] = post_element["text"]
                    
                    new_elements.append({
                        **strategy,
                        "required": True
                    })
                    seen_signatures.add(signature_full)
        
        except Exception as e:
            
            print(f"[Warning] Failed while detecting new elements: {e}")
            return []
        
        return new_elements
    
    def _get_element_signature(self, element: Dict[str, Any]) -> str:
        """
        Stable signature tuple mixing role/label/text/id for comparisons.
        """
        role = element.get("role", "")
        label = element.get("label", "")
        text = element.get("text", "")
        element_id = element.get("element_id", "")
        
        
        
        if element_id:
            return f"{role}:{label}:{text}:{element_id}"
        else:
            
            return f"{role}:{label}:{text}"
    
    def _get_element_signature_partial(self, element: Dict[str, Any]) -> str:
        """
        Signature without raw id (detect cosmetic id churn).
        """
        role = element.get("role", "")
        label = element.get("label", "")
        text = element.get("text", "")
        
        return f"{role}:{label}:{text}"
    
    def _detect_disappeared_elements(
        self,
        pre_obs_text: str,
        post_obs_text: str
    ) -> List[Dict[str, Any]]:
        """
        Inverse diff for nodes removed after the segment.
        """
        from .accessibility_tree_parser import AccessibilityTreeParser
        
        try:
            parser = AccessibilityTreeParser()
            pre_tree = parser.parse(pre_obs_text)
            post_tree = parser.parse(post_obs_text)
        except Exception as e:
            print(f"[Warning] Failed parsing observation: {e}")
            return []
        
        
        post_signatures_full = {
            self._get_element_signature(e) 
            for e in post_tree.get("elements", [])
            if e.get("element_id")
        }
        post_element_ids = {
            e.get("element_id")
            for e in post_tree.get("elements", [])
            if e.get("element_id")
        }
        
        disappeared_elements = []
        
        for pre_element in pre_tree.get("elements", []):
            if not pre_element.get("element_id"):
                continue
            
            signature_full = self._get_element_signature(pre_element)
            pre_element_id = pre_element.get("element_id")
            
            
            is_disappeared = signature_full not in post_signatures_full
            
            
            if not is_disappeared and pre_element_id not in post_element_ids:
                is_disappeared = True
            
            if is_disappeared:
                strategy = {
                    "strategy": "semantic",
                    "conditions": {}
                }
                
                if pre_element.get("role"):
                    strategy["conditions"]["role"] = pre_element["role"]
                if pre_element.get("label"):
                    strategy["conditions"]["label"] = pre_element["label"]
                if pre_element.get("text"):
                    strategy["conditions"]["text"] = pre_element["text"]
                
                disappeared_elements.append(strategy)
        
        return disappeared_elements
    
    def _is_error_or_failure_text(self, text: str) -> bool:
        """
        True if snippet looks like an error banner (exclude from success text_appears).
        """
        if not text or not text.strip():
            return False
        text_lower = text.strip().lower()
        error_keywords = [
            "error", "failed", "invalid",
            "exception", "not found", "404", "500",
            "loading", "please wait",
            "synchronize", "synchronization", "magento business intelligence",
        ]
        return any(kw in text_lower for kw in error_keywords)
    
    def _detect_new_texts(
        self,
        pre_obs_text: str,
        post_obs_text: str
    ) -> List[Dict[str, Any]]:
        """Diff parsed text nodes between pre/post accessibility snapshots."""
        new_texts = []
        
        if not pre_obs_text or not post_obs_text:
            return new_texts
        
        try:
            from .accessibility_tree_parser import AccessibilityTreeParser
            import re
            
            parser = AccessibilityTreeParser()
            pre_tree = parser.parse(pre_obs_text)
            post_tree = parser.parse(post_obs_text)
            
            
            pre_texts = set()
            post_texts = set()
            
            
            for element in pre_tree.get("elements", []):
                
                label = (element.get("label") or "").strip()
                text = (element.get("text") or "").strip()
                if label:
                    pre_texts.add(label.lower())
                if text and text != label:
                    pre_texts.add(text.lower())
                
                if (element.get("role") or "").strip() == "text":
                    text_content = (element.get("text") or "").strip()
                    if text_content:
                        pre_texts.add(text_content.lower())
            
            
            for element in post_tree.get("elements", []):
                
                label = (element.get("label") or "").strip()
                text = (element.get("text") or "").strip()
                if label:
                    post_texts.add(label.lower())
                if text and text != label:
                    post_texts.add(text.lower())
                
                if (element.get("role") or "").strip() == "text":
                    text_content = (element.get("text") or "").strip()
                    if text_content:
                        post_texts.add(text_content.lower())
            
            
            new_text_set = post_texts - pre_texts
            
            
            seen_texts = set()
            for element in post_tree.get("elements", []):
                label = (element.get("label") or "").strip()
                text = (element.get("text") or "").strip()
                
                
                if label and label.lower() in new_text_set and label.lower() not in seen_texts:
                    if not self._is_error_or_failure_text(label):
                        new_texts.append({
                            "text": label,
                            "required": True
                        })
                    seen_texts.add(label.lower())
                
                
                if text and text != label and text.lower() in new_text_set and text.lower() not in seen_texts:
                    if not self._is_error_or_failure_text(text):
                        new_texts.append({
                            "text": text,
                            "required": True
                        })
                    seen_texts.add(text.lower())
                
                
                if (element.get("role") or "").strip() == "text":
                    text_content = (element.get("text") or "").strip()
                    if text_content and text_content.lower() in new_text_set and text_content.lower() not in seen_texts:
                        if not self._is_error_or_failure_text(text_content):
                            new_texts.append({
                                "text": text_content,
                                "required": True
                            })
                        seen_texts.add(text_content.lower())
            
            
            
            pre_text_lines = set()
            post_text_lines = set()
            
            for line in pre_obs_text.split('\n'):
                line = (line or "").strip()
                
                match = re.match(r"^text\s+['\"]([^'\"]+)['\"]", line)
                if match:
                    text_content = (match.group(1) or "").strip()
                    if text_content:
                        pre_text_lines.add(text_content.lower())
            
            for line in post_obs_text.split('\n'):
                line = (line or "").strip()
                match = re.match(r"^text\s+['\"]([^'\"]+)['\"]", line)
                if match:
                    text_content = (match.group(1) or "").strip()
                    if text_content:
                        post_text_lines.add(text_content.lower())
            
            
            new_text_lines = post_text_lines - pre_text_lines
            
            
            for line in post_obs_text.split('\n'):
                line = (line or "").strip()
                match = re.match(r"^text\s+['\"]([^'\"]+)['\"]", line)
                if match:
                    text_content = (match.group(1) or "").strip()
                    if text_content and text_content.lower() in new_text_lines:
                        text_lower = text_content.lower()
                        
                        if text_lower not in seen_texts and not self._is_error_or_failure_text(text_content):
                            new_texts.append({
                                "text": text_content,
                                "required": True
                            })
                        seen_texts.add(text_lower)
        
        except Exception as e:
            print(f"[Warning] Failed while detecting new texts: {e}")
            
            try:
                import re
                
                pre_words = set(re.findall(r'\b\w+\b', pre_obs_text.lower()))
                post_words = set(re.findall(r'\b\w+\b', post_obs_text.lower()))
                new_words = post_words - pre_words
                
                
                if len(new_words) > 0:
                    
                    for word in list(new_words)[:10]:  
                        
                        pattern = rf'\b\w*\s*{re.escape(word)}\s*\w*\b'
                        matches = re.findall(pattern, post_obs_text, re.IGNORECASE)
                        for match in matches[:3]:  
                            match_lower = match.lower()
                            if match_lower not in [t.get("text", "").lower() for t in new_texts]:
                                if not self._is_error_or_failure_text(match):
                                    new_texts.append({
                                        "text": match,
                                        "required": True
                                    })
            except Exception:
                pass
        
        return new_texts
    
    def _detect_region_updates(
        self,
        pre_obs_text: str,
        post_obs_text: str
    ) -> List[Dict[str, Any]]:
        """
        Compare landmark regions between observations for coarse updates.
        """
        from .accessibility_tree_parser import AccessibilityTreeParser
        
        if not pre_obs_text or not post_obs_text:
            return []
        
        try:
            parser = AccessibilityTreeParser()
            pre_tree = parser.parse(pre_obs_text)
            post_tree = parser.parse(post_obs_text)
            
            region_updates = []
            
            
            region_roles = ["main", "complementary", "header", "footer", "navigation", "aside"]
            
            
            pre_regions = {}
            post_regions = {}
            
            
            for element in pre_tree.get("elements", []):
                role = element.get("role", "")
                if role in region_roles:
                    element_id = element.get("element_id", "")
                    if element_id:
                        if role not in pre_regions:
                            pre_regions[role] = []
                        pre_regions[role].append({
                            "element_id": element_id,
                            "label": element.get("label", ""),
                            "text": element.get("text", "")
                        })
            
            
            for element in post_tree.get("elements", []):
                role = element.get("role", "")
                if role in region_roles:
                    element_id = element.get("element_id", "")
                    if element_id:
                        if role not in post_regions:
                            post_regions[role] = []
                        post_regions[role].append({
                            "element_id": element_id,
                            "label": element.get("label", ""),
                            "text": element.get("text", "")
                        })
            
            
            
            for role, post_region_elements in post_regions.items():
                if role not in pre_regions:
                    
                    region_updates.append({
                        "type": "appeared",
                        "role": role,
                        "description": f"New region '{role}' appeared"
                    })
                else:
                    
                    pre_count = len(pre_regions[role])
                    post_count = len(post_region_elements)
                    
                    
                    if post_count != pre_count:
                        region_updates.append({
                            "type": "updated",
                            "role": role,
                            "description": f"Region '{role}' updated: {pre_count} -> {post_count} elements",
                            "pre_count": pre_count,
                            "post_count": post_count
                        })
                    else:
                        
                        
                        pre_labels = {e.get("label", "") for e in pre_regions[role]}
                        post_labels = {e.get("label", "") for e in post_region_elements}
                        if pre_labels != post_labels:
                            region_updates.append({
                                "type": "content_updated",
                                "role": role,
                                "description": f"Region '{role}' content updated"
                            })
            
            
            for role in pre_regions:
                if role not in post_regions:
                    region_updates.append({
                        "type": "disappeared",
                        "role": role,
                        "description": f"Region '{role}' disappeared"
                    })
            
            return region_updates
            
        except Exception as e:
            print(f"[Warning] Failed while detecting region updates: {e}")
            return []
    
    def _detect_modal_changes(
        self,
        pre_obs_text: str,
        post_obs_text: str
    ) -> List[Dict[str, Any]]:
        """
        Track modal/dialog appearance between observations.
        """
        from .accessibility_tree_parser import AccessibilityTreeParser
        
        if not pre_obs_text or not post_obs_text:
            return []
        
        try:
            parser = AccessibilityTreeParser()
            pre_tree = parser.parse(pre_obs_text)
            post_tree = parser.parse(post_obs_text)
            
            modal_changes = []
            
            
            pre_modals = []
            post_modals = []
            
            
            for element in pre_tree.get("elements", []):
                role = (element.get("role") or "").lower()
                label = (element.get("label") or "").lower()
                
                
                is_modal = (
                    role in ["dialog", "alertdialog"] or
                    any(keyword in role for keyword in ["modal", "dialog"]) or
                    any(keyword in label for keyword in ["modal", "dialog", "popup"])
                )
                
                if is_modal:
                    element_id = element.get("element_id", "") or ""
                    modal_type = "alert" if "alert" in role else "dialog" if "dialog" in role else "modal"
                    pre_modals.append({
                        "element_id": element_id,
                        "role": element.get("role") or "",
                        "label": element.get("label") or "",
                        "text": element.get("text") or "",
                        "type": modal_type
                    })
            
            
            for element in post_tree.get("elements", []):
                role = (element.get("role") or "").lower()
                label = (element.get("label") or "").lower()
                
                is_modal = (
                    role in ["dialog", "alertdialog"] or
                    any(keyword in role for keyword in ["modal", "dialog"]) or
                    any(keyword in label for keyword in ["modal", "dialog", "popup"])
                )
                
                if is_modal:
                    element_id = element.get("element_id", "") or ""
                    modal_type = "alert" if "alert" in role else "dialog" if "dialog" in role else "modal"
                    post_modals.append({
                        "element_id": element_id,
                        "role": element.get("role") or "",
                        "label": element.get("label") or "",
                        "text": element.get("text") or "",
                        "type": modal_type
                    })
            
            
            
            pre_modal_ids = {m.get("element_id") for m in pre_modals if m.get("element_id")}
            post_modal_ids = {m.get("element_id") for m in post_modals if m.get("element_id")}
            
            new_modal_ids = post_modal_ids - pre_modal_ids
            for modal in post_modals:
                if modal.get("element_id") in new_modal_ids:
                    modal_changes.append({
                        "type": "opened",
                        "role": modal.get("role") or "",
                        "label": modal.get("label") or "",
                        "description": f"Modal opened: {modal.get('label') or 'Unknown'}"
                    })
            
            
            closed_modal_ids = pre_modal_ids - post_modal_ids
            for modal in pre_modals:
                if modal.get("element_id") in closed_modal_ids:
                    modal_changes.append({
                        "type": "closed",
                        "role": modal.get("role") or "",
                        "label": modal.get("label") or "",
                        "description": f"Modal closed: {modal.get('label') or 'Unknown'}"
                    })
            
            
            existing_modal_ids = pre_modal_ids & post_modal_ids
            if existing_modal_ids:
                
                if len(pre_modals) == len(post_modals) and len(existing_modal_ids) > 0:
                    
                    modal_changes.append({
                        "type": "updated",
                        "description": "Modal content may have changed"
                    })
            
            return modal_changes
            
        except Exception as e:
            print(f"[Warning] Failed while detecting modal changes: {e}")
            return []
    
    def _llm_enhance_post_condition(
        self,
        post_condition: Dict[str, Any],
        action_sequence: List[Any],
        pre_obs_text: str,
        post_obs_text: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Let an LLM refine post_condition JSON using pre/post observation snippets."""
        import json
        
        
        has_content = (
            post_condition.get("element_appears") or
            post_condition.get("element_disappears") or
            post_condition.get("text_appears") or
            post_condition.get("region_updates") or
            post_condition.get("modal_changes") or
            (post_condition.get("url_change", {}).get("type") == "navigate")
        )
        
        
        
        
        
        action_summary = []
        for i, action in enumerate(action_sequence):
            action_str = str(action)
            action_summary.append(f"Step {i}: {action_str}")
        action_summary_str = "\n".join(action_summary)
        
        
        pre_obs_snippet = pre_obs_text[:2000] if len(pre_obs_text) > 2000 else pre_obs_text
        post_obs_snippet = post_obs_text[:2000] if len(post_obs_text) > 2000 else post_obs_text
        
        system_prompt = """You refine JSON post-conditions for web automations.

Cover URL changes, newly visible controls, disappearing loaders, important text, region refreshes, modal transitions.
Prefer semantic roles/labels/text—not bare DOM ids. Skip noise; focus on reusable signals. JSON only."""

        user_prompt = f"""Context:
- Site: {context.get('site', 'unknown')}
- Page: {context.get('page', 'unknown')}
- URL Pattern: {context.get('url_pattern', '/')}

Actions:
{action_summary_str}

Pre observation (first 2000 chars):
{pre_obs_snippet}

Post observation (first 2000 chars):
{post_obs_snippet}

Baseline post-condition JSON:
{json.dumps(post_condition, indent=2, ensure_ascii=False)}

Improve the JSON; keep schema:
{{
  "url_change": {{"type": "stay|navigate", "pattern": "optional glob"}},
  "element_appears": [...],
  "element_disappears": [...],
  "text_appears": [...],
  "region_updates": [...],
  "modal_changes": [...]
}}

JSON only, no commentary."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = llm_utils.call_llm(self.llm_config, messages)
            response = response.strip()
            
            
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
            
            
            llm_post_condition = json.loads(response)
            
            
            
            enhanced_post_condition = {
                "url_change": llm_post_condition.get("url_change", post_condition.get("url_change", {"type": "stay"})),
                "element_appears": llm_post_condition.get("element_appears", []),
                "element_disappears": llm_post_condition.get("element_disappears", []),
                "text_appears": llm_post_condition.get("text_appears", []),
                "region_updates": llm_post_condition.get("region_updates", []),
                "modal_changes": llm_post_condition.get("modal_changes", [])
            }
            
            
            if not enhanced_post_condition["element_appears"] and post_condition.get("element_appears"):
                enhanced_post_condition["element_appears"] = post_condition["element_appears"]
            if not enhanced_post_condition["element_disappears"] and post_condition.get("element_disappears"):
                enhanced_post_condition["element_disappears"] = post_condition["element_disappears"]
            if not enhanced_post_condition["text_appears"] and post_condition.get("text_appears"):
                enhanced_post_condition["text_appears"] = post_condition["text_appears"]
            if not enhanced_post_condition["region_updates"] and post_condition.get("region_updates"):
                enhanced_post_condition["region_updates"] = post_condition["region_updates"]
            if not enhanced_post_condition["modal_changes"] and post_condition.get("modal_changes"):
                enhanced_post_condition["modal_changes"] = post_condition["modal_changes"]
            
            return enhanced_post_condition
            
        except json.JSONDecodeError as e:
            print(f"[Warning] LLM post-condition response was not valid JSON: {e}")
            print(f"[Warning] LLM response snippet: {response[:500]}")
            return post_condition
        except Exception as e:
            print(f"[Warning] LLM post-condition augmentation failed: {e}")
            return post_condition
    
    def _find_action_in_trajectory(
        self,
        action_str: str,
        trajectory: List[Dict[str, Any]],
        start_idx: int
    ) -> Optional[int]:
        """Scan trajectory forward from start_idx for a matching primitive string."""
        
        search_range = min(start_idx + 20, len(trajectory))
        
        for i in range(start_idx, search_range):
            step = trajectory[i]
            step_action = step.get("action", "")
            
            
            step_action_str = str(step_action).strip()
            action_str_clean = action_str.strip()
            
            
            if step_action_str == action_str_clean:
                return i
            
            
            action_ids = self._extract_element_ids_from_action(action_str_clean)
            step_ids = self._extract_element_ids_from_action(step_action_str)
            
            if action_ids and step_ids and action_ids[0] == step_ids[0]:
                
                if (action_str_clean.lower().startswith("click") and step_action_str.lower().startswith("click")) or                   (action_str_clean.lower().startswith("type") and step_action_str.lower().startswith("type")):
                    return i
        
        return None
