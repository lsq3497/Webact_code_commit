"""
ActSpec executor: bind parameters and run plans against env.step.
"""

import copy
import json
import time
from typing import Dict, List, Any, Optional
from browser_env.actions import create_id_based_action, Action

from . import page_change_detector
from . import readiness_checker
from . import step_executor
from . import semantic_change_handler
from .post_condition_verifier import PostConditionVerifier


LOCATE_MULTIPLE_CANDIDATES = "_multiple_candidates_"


class ActSpecExecutor:
    """Bind parameters to plans and execute steps."""

    def __init__(self):
        """Stateless executor."""
        pass
    
    def execute_actspec(
        self,
        actspec: Dict[str, Any],
        parameters: Dict[str, Any],
        env: Any
    ) -> Dict[str, Any]:
        """
        Run one ActSpec end-to-end.

        Returns {"success": bool, "error": str | None}.
        """
        call_record = {"executor_success": None, "error": None}
        try:
            from .pre_condition_checker import PreConditionChecker
            from .locate_executor import LocateExecutor
            
            pre_checker = PreConditionChecker()
            locate_executor = LocateExecutor()

            
            call_record = {
                "action_id": actspec.get("action_id", ""),
                "parameters": dict(parameters) if isinstance(parameters, dict) else {},
                "pre_condition_satisfied": None,
                "post_condition_satisfied": None,
                "executor_success": None,
                "error": None,
                "partial_success": False,
                "pre_url": None,
                "post_url": None,
                "pre_text": None,
                "post_text": None,
            }
            if hasattr(env, "actspec_call_records"):
                env.actspec_call_records.append(call_record)
            
            
            
            pre_condition = actspec.get("pre_condition", {})
            if pre_condition:
                page = self._get_page_from_env(env)
                if page:
                    try:
                        
                        observation_text = None
                        if hasattr(env, 'observation'):
                            try:
                                obs = env.observation()
                                if isinstance(obs, str):
                                    observation_text = obs
                                elif isinstance(obs, dict) and 'text' in obs:
                                    t = obs['text']
                                    observation_text = t if isinstance(t, str) else (str(t[0]) if isinstance(t, (list, tuple)) and t else str(t))
                            except Exception:
                                pass
                        
                        is_satisfied, reason = pre_checker.check_pre_condition(
                            pre_condition, page or env, parameters, observation_text=observation_text
                        )
                        call_record["pre_condition_satisfied"] = bool(is_satisfied)
                        
                        
                        if not is_satisfied:
                            print(f"[ActSpec] Warning: pre-condition not met (continuing; already filtered upstream): {reason}")
                    except Exception as e:
                        
                        print(f"[Warning] Pre-condition check error: {e}, continuing")
                        call_record["pre_condition_satisfied"] = None
            
            
            
            
            pre_url = ""
            pre_text = ""
            if hasattr(env, "get_url"):
                try:
                    pre_url = env.get_url() or ""
                except Exception:
                    pass
            if hasattr(env, "observation"):
                try:
                    obs = env.observation()
                    if isinstance(obs, str):
                        pre_text = obs
                    elif isinstance(obs, dict) and "text" in obs:
                        t = obs["text"]
                        pre_text = t if isinstance(t, str) else (str(t[0]) if isinstance(t, (list, tuple)) and t else str(t))
                except Exception:
                    pass
            pre_obs = {"text": pre_text}
            call_record["pre_url"] = pre_url
            call_record["pre_text"] = pre_text
            
            
            plan = actspec.get("plan", [])
            max_llm_adjustments = max(0, len(plan) - 1)
            call_record["max_llm_adjustments"] = max_llm_adjustments
            call_record["llm_adjustment_count"] = 0
            call_record["reached_adjustment_limit"] = False
            
            
            bindings = actspec.get("bindings", {})
            bound_plan = self._bind_parameters(plan, bindings, parameters)
            locate = actspec.get("locate", {})
            locate_executor = LocateExecutor()
            page = self._get_page_from_env(env)
            observation_processor = self._get_observation_processor_from_env(env)
            target_elements = locate.get("target_elements", [])
            
            
            current_actspec = actspec
            current_plan = bound_plan
            start_step = 0
            llm_adjustment_count = 0
            max_llm_adjustments = max(0, len(current_plan) - 1)
            WAIT_RETRY_SEC = 1.5
            last_wait_retry_step = -1
            last_no_page_retry_step = -1
            
            while start_step < len(current_plan):
                step_index = start_step
                while step_index < len(current_plan):
                    step = current_plan[step_index]
                    
                    if step.get("target", {}).get("strategy") == "element_id":
                        value = step["target"].get("value", "")
                        need_locate = (
                            value is None or value == ""
                            or (isinstance(value, str) and (not value.strip() or value.strip().startswith("{{")))
                        )
                        if need_locate:
                            element_id = self._locate_element_for_step(
                                step_index, target_elements, locate_executor, page,
                                observation_processor, parameters
                            )
                            step["target"]["value"] = element_id if element_id else ""
                    
                    result = step_executor.execute_step(
                        step_index,
                        step,
                        current_plan,
                        env,
                        plan_step_to_action=self._plan_step_to_action,
                        readiness_check_fn=readiness_checker.check_readiness,
                        page_change_detector_module=page_change_detector,
                        locate_multiple_candidates_marker=LOCATE_MULTIPLE_CANDIDATES,
                    )
                    
                    if result.get("success"):
                        step_index += 1
                        continue
                    
                    failure_reason = result.get("failure_reason") or ""
                    
                    if failure_reason == "locate_multiple_candidates":
                        obs_text = self._get_observation_text_for_step(env)
                        page_desc = getattr(env, "get_url", lambda: "")() or ""
                        new_actspec, new_start, new_count = semantic_change_handler.handle_semantic_change(
                            current_actspec, step_index, env, obs_text, page_desc,
                            failure_reason, llm_adjustment_count, max_llm_adjustments,
                        )
                        if new_actspec is None:
                            call_record["llm_adjustment_count"] = llm_adjustment_count
                            call_record["reached_adjustment_limit"] = (llm_adjustment_count >= max_llm_adjustments)
                            call_record["executor_success"] = False
                            call_record["error"] = "locate_multiple_candidates_and_llm_exhausted"
                            return {"success": False, "error": "locate_multiple_candidates_and_llm_exhausted"}
                        current_actspec = new_actspec
                        current_plan = self._bind_parameters(
                            current_actspec.get("plan", []), bindings, parameters
                        )
                        start_step = new_start or step_index
                        llm_adjustment_count = new_count
                        step_index = start_step
                        break
                    
                    
                    wait_retry_done = last_wait_retry_step == step_index
                    no_page_retry_done = last_no_page_retry_step == step_index
                    if failure_reason in ("target_not_interactable", "action_exception") and not wait_retry_done:
                        time.sleep(WAIT_RETRY_SEC)
                        last_wait_retry_step = step_index
                        
                        if step.get("target", {}).get("strategy") == "element_id":
                            step["target"]["value"] = ""
                            element_id = self._locate_element_for_step(
                                step_index, target_elements, locate_executor, page,
                                observation_processor, parameters
                            )
                            step["target"]["value"] = element_id if element_id else ""
                        result2 = step_executor.execute_step(
                            step_index, step, current_plan, env,
                            plan_step_to_action=self._plan_step_to_action,
                            readiness_check_fn=readiness_checker.check_readiness,
                            page_change_detector_module=page_change_detector,
                            locate_multiple_candidates_marker=LOCATE_MULTIPLE_CANDIDATES,
                        )
                        if result2.get("success"):
                            step_index += 1
                            continue
                        result = result2
                        failure_reason = result.get("failure_reason") or ""
                    if failure_reason == "no_page_change" and not no_page_retry_done:
                        last_no_page_retry_step = step_index
                        if step.get("target", {}).get("strategy") == "element_id":
                            step["target"]["value"] = ""
                            element_id = self._locate_element_for_step(
                                step_index, target_elements, locate_executor, page,
                                observation_processor, parameters
                            )
                            step["target"]["value"] = element_id if element_id else ""
                        result2 = step_executor.execute_step(
                            step_index, step, current_plan, env,
                            plan_step_to_action=self._plan_step_to_action,
                            readiness_check_fn=readiness_checker.check_readiness,
                            page_change_detector_module=page_change_detector,
                            locate_multiple_candidates_marker=LOCATE_MULTIPLE_CANDIDATES,
                        )
                        if result2.get("success"):
                            step_index += 1
                            continue
                        result = result2
                    
                    
                    if not semantic_change_handler.cannot_consider_valid_intermediate_state(
                        current_actspec, step_index, env
                    ):
                        call_record["llm_adjustment_count"] = llm_adjustment_count
                        call_record["reached_adjustment_limit"] = (llm_adjustment_count >= max_llm_adjustments)
                        call_record["executor_success"] = False
                        call_record["error"] = failure_reason
                        return {"success": False, "error": failure_reason}
                    obs_text = self._get_observation_text_for_step(env)
                    page_desc = getattr(env, "get_url", lambda: "")() or ""
                    new_actspec, new_start, new_count = semantic_change_handler.handle_semantic_change(
                        current_actspec, step_index, env, obs_text, page_desc,
                        failure_reason, llm_adjustment_count, max_llm_adjustments,
                    )
                    if new_actspec is None:
                        call_record["llm_adjustment_count"] = llm_adjustment_count
                        call_record["reached_adjustment_limit"] = (llm_adjustment_count >= max_llm_adjustments)
                        call_record["executor_success"] = False
                        call_record["error"] = failure_reason
                        return {"success": False, "error": failure_reason}
                    current_actspec = new_actspec
                    current_plan = self._bind_parameters(
                        current_actspec.get("plan", []), bindings, parameters
                    )
                    start_step = new_start or step_index
                    llm_adjustment_count = new_count
                    step_index = start_step
                    break
                else:
                    
                    break
            
            
            call_record["llm_adjustment_count"] = llm_adjustment_count
            call_record["reached_adjustment_limit"] = (llm_adjustment_count >= max_llm_adjustments)
            post_condition = current_actspec.get("post_condition", {}) or {}
            verifier = PostConditionVerifier()
            post_ok, post_reason = verifier.verify_post_condition(
                post_condition=post_condition,
                page=self._get_page_from_env(env) or env,
                pre_url=pre_url,
                pre_obs=pre_obs,
            )
            call_record["post_condition_satisfied"] = bool(post_ok)
            call_record["executor_success"] = bool(post_ok)
            if not post_ok:
                call_record["error"] = f"post_condition_failed: {post_reason}"
                return {
                    "success": False,
                    "error": f"post_condition_failed: {post_reason}"
                }
            return {
                "success": True,
                "error": None
            }
        except Exception as e:
            call_record["executor_success"] = False
            call_record["error"] = str(e)
            call_record["reached_adjustment_limit"] = True  
            return {
                "success": False,
                "error": str(e)
            }
    
    def _get_page_from_env(self, env: Any) -> Any:
        """Resolve Playwright page from env wrapper."""
        
        if hasattr(env, 'page'):
            return env.page
        elif hasattr(env, 'webarena_env') and hasattr(env.webarena_env, 'page'):
            return env.webarena_env.page
        elif hasattr(env, 'browser') and hasattr(env.browser, 'page'):
            return env.browser.page
        else:
            return None
    
    def _get_observation_text_for_step(self, env: Any) -> str:
        """Observation text for semantic-change handling (str or dict text)."""
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
    
    def _update_plan_with_locate(
        self,
        plan: List[Dict[str, Any]],
        locate: Dict[str, Any],
        locate_executor: Any,
        page: Any,
        parameters: Dict[str, Any],
        env: Any = None
    ) -> List[Dict[str, Any]]:
        """
        Fill plan targets via locate config: try semantic strategies before numeric element_id
        so dynamic DOM ids hurt less.
        """
        import copy
        updated_plan = copy.deepcopy(plan)
        
        
        observation_processor = self._get_observation_processor_from_env(env)
        if not observation_processor:
            print("[ActSpec] Warning: observation_processor missing, skipping auto-locate")
            return updated_plan
        
        target_elements = locate.get("target_elements", [])
        
        
        for step_idx, step in enumerate(updated_plan):
            
            target = step.get("target", {})
            if target.get("strategy") == "element_id":
                value = target.get("value", "")
                
                value_s = value if isinstance(value, str) else str(value) if value is not None else ""
                if value_s.startswith("{{") and value_s.endswith("}}"):
                    param_name = value_s[2:-2]
                    
                    
                    
                    element_id = self._locate_element_for_step(
                        step_idx, target_elements, locate_executor, page, 
                        observation_processor, parameters
                    )
                    if element_id:
                        
                        step["target"]["value"] = element_id
                        provided_value = parameters.get(param_name)
                        if provided_value and str(provided_value) != str(element_id):
                            print(f"[ActSpec] step {step_idx}: locate ok, element_id={element_id} (overriding LLM value {provided_value})")
                        else:
                            print(f"[ActSpec] step {step_idx}: locate ok, element_id={element_id}")
                    else:
                        
                        
                        
                        print(f"[ActSpec] Warning: step {step_idx}: all locate strategies failed, keeping placeholder {value_s} for repair/semantic handling")
        
        return updated_plan
    
    def _get_observation_processor_from_env(self, env: Any) -> Any:
        """Observation processor from webarena env."""
        if not env:
            return None
        
        try:
            
            if hasattr(env, 'webarena_env'):
                webarena_env = env.webarena_env
                if hasattr(webarena_env, 'observation_handler'):
                    handler = webarena_env.observation_handler
                    if hasattr(handler, 'text_processor'):
                        return handler.text_processor
                
                if hasattr(webarena_env, 'info') and isinstance(webarena_env.info, dict):
                    if 'observation_processor' in webarena_env.info:
                        return webarena_env.info['observation_processor']
        except Exception as e:
            print(f"[ActSpec] Failed to get observation_processor: {e}")
        
        return None
    
    def _locate_element_for_step(
        self,
        step_idx: int,
        target_elements: List[Dict[str, Any]],
        locate_executor: Any,
        page: Any,
        observation_processor: Any,
        parameters: Dict[str, Any]
    ) -> Optional[str]:
        """
        Resolve element_id for plan step_idx using locate.target_elements (semantic first).
        Returns None or LOCATE_MULTIPLE_CANDIDATES sentinel.
        """
        
        target_element_config = None
        for te in target_elements:
            
            if "step" in te and te["step"] == step_idx:
                target_element_config = te
                break
        
        
        if not target_element_config and len(target_elements) > step_idx:
            target_element_config = target_elements[step_idx]
        
        if not target_element_config:
            return None
        
        
        strategies = target_element_config.get("strategies", [])
        if not strategies:
            return None
        
        
        strategies = sorted(strategies, key=lambda x: x.get("priority", 999))
        
        
        semantic_result = None
        for strategy in strategies:
            strategy_type = strategy.get("strategy")
            conditions = strategy.get("conditions", {})
            
            if strategy_type == "semantic":
                semantic_result = self._locate_by_semantic_and_extract_id(
                    conditions, page, observation_processor
                )
                if semantic_result is not None:
                    if semantic_result == LOCATE_MULTIPLE_CANDIDATES:
                        print(f"[ActSpec] step {step_idx}: semantic locate returned multiple candidates (no heuristic pick)")
                        return LOCATE_MULTIPLE_CANDIDATES
                    print(f"[ActSpec] step {step_idx}: semantic locate ok, element_id={semantic_result}")
                    return semantic_result
                break  
        
        
        print(f"[ActSpec] step {step_idx}: semantic locate failed, falling back to element_id strategy")
        for strategy in strategies:
            strategy_type = strategy.get("strategy")
            conditions = strategy.get("conditions", {})
            
            if strategy_type == "element_id":
                element_id_param = conditions.get("element_id", "")
                element_id_value = None
                
                
                if element_id_param.startswith("{{") and element_id_param.endswith("}}"):
                    param_name = element_id_param[2:-2]
                    if param_name in parameters and parameters[param_name]:
                        element_id_value = str(parameters[param_name])
                elif element_id_param:
                    
                    element_id_value = element_id_param
                
                
                if element_id_value:
                    if self._verify_element_id_exists(element_id_value, page, observation_processor):
                        print(f"[ActSpec] step {step_idx}: element_id strategy ok, element_id={element_id_value}")
                        return element_id_value
                    else:
                        print(f"[ActSpec] step {step_idx}: element_id={element_id_value} not present in observation, skip")
        
        print(f"[ActSpec] step {step_idx}: all locate strategies failed")
        return None
    
    def _locate_by_semantic_and_extract_id(
        self,
        conditions: Dict[str, Any],
        page: Any,
        observation_processor: Any
    ) -> Optional[str]:
        """
        Semantic match against observation_processor nodes.
        Returns one id, None, or LOCATE_MULTIPLE_CANDIDATES.
        """
        if not hasattr(observation_processor, 'obs_nodes_info'):
            return None
        
        role = conditions.get("role")
        label = conditions.get("label")
        text = conditions.get("text")
        matched_ids: List[str] = []
        
        for element_id, node_info in observation_processor.obs_nodes_info.items():
            try:
                node = observation_processor.get_node_info_by_element_id(int(element_id))
                if not node:
                    continue
                
                node_role = getattr(node, 'role', '') or ''
                node_name = getattr(node, 'name', '') or ''
                node_text = getattr(node, 'text', '') or ''
                
                if role and node_role != role:
                    continue
                
                match_found = False
                if label:
                    label_lower = label.lower().strip()
                    node_name_lower = node_name.lower().strip()
                    node_text_lower = node_text.lower().strip()
                    if label_lower == node_name_lower or label_lower == node_text_lower:
                        match_found = True
                    elif label_lower in node_name_lower or label_lower in node_text_lower:
                        match_found = True
                elif text:
                    text_lower = text.lower().strip()
                    node_name_lower = node_name.lower().strip()
                    node_text_lower = node_text.lower().strip()
                    if text_lower == node_name_lower or text_lower == node_text_lower:
                        match_found = True
                    elif text_lower in node_name_lower or text_lower in node_text_lower:
                        match_found = True
                else:
                    if role and node_role == role:
                        match_found = True
                
                if match_found:
                    if self._verify_element_exists(page, role, label or text):
                        matched_ids.append(element_id)
                        continue
                    if label:
                        label_lower = label.lower().strip()
                        if (label_lower == node_name_lower or label_lower == node_text_lower):
                            matched_ids.append(element_id)
                    elif text:
                        text_lower = text.lower().strip()
                        if (text_lower == node_name_lower or text_lower == node_text_lower):
                            matched_ids.append(element_id)
            except Exception:
                continue
        
        if len(matched_ids) == 0:
            return None
        if len(matched_ids) == 1:
            return matched_ids[0]
        return LOCATE_MULTIPLE_CANDIDATES
    
    def _verify_element_exists(self, page: Any, role: Optional[str], name: Optional[str]) -> bool:
        """
        Lenient Playwright existence probe (async-safe); defaults True if probing fails.
        """
        
        
        try:
            if role and name:
                locator = page.get_by_role(role=role, name=name, exact=False)
                
                try:
                    count = locator.count()
                    
                    if hasattr(count, '__await__'):
                        return True
                    return count > 0
                except (TypeError, AttributeError):
                    
                    return True
            elif name:
                
                try:
                    locator = page.get_by_text(name)
                    try:
                        count = locator.count()
                        if hasattr(count, '__await__'):
                            return True
                        if count > 0:
                            return True
                    except (TypeError, AttributeError):
                        return True
                except:
                    pass
                try:
                    locator = page.get_by_label(name)
                    try:
                        count = locator.count()
                        if hasattr(count, '__await__'):
                            return True
                        if count > 0:
                            return True
                    except (TypeError, AttributeError):
                        return True
                except:
                    pass
        except Exception:
            
            
            return True
        
        return False
    
    def _verify_element_id_exists(
        self,
        element_id: str,
        page: Any,
        observation_processor: Any
    ) -> bool:
        """True if element_id is listed in observation_processor."""
        if not observation_processor or not hasattr(observation_processor, 'obs_nodes_info'):
            
            return False
        
        try:
            
            element_id_int = int(element_id)
            
            
            if str(element_id_int) in observation_processor.obs_nodes_info:
                
                try:
                    node = observation_processor.get_node_info_by_element_id(element_id_int)
                    if node:
                        
                        return True
                except Exception:
                    
                    
                    return True
            
            
            return False
            
        except (ValueError, TypeError):
            
            return False
        except Exception as e:
            
            print(f"[ActSpec] Error verifying element_id presence: {e}")
            return False
    
    def _bind_parameters(
        self,
        plan: List[Dict[str, Any]],
        bindings: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Inject parameter values into plan steps per bindings; missing params leave fields untouched.
        """
        
        import copy
        bound_plan = copy.deepcopy(plan)
        
        
        for param_name, binding_info in bindings.items():
            param_value = parameters.get(param_name)
            if param_value is None:
                continue
            
            bind_to = binding_info.get("bind_to", [])
            for bind_rule in bind_to:
                step_idx = bind_rule.get("step")
                field = bind_rule.get("field")
                
                if step_idx >= len(bound_plan):
                    continue
                
                step = bound_plan[step_idx]
                
                
                if field == "text":
                    step["text"] = str(param_value)
                elif field == "target.value":
                    if "target" in step:
                        step["target"]["value"] = str(param_value)
                elif field == "url":
                    step["url"] = str(param_value)
                else:
                    
                    field_parts = field.split(".")
                    current = step
                    for part in field_parts[:-1]:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                    current[field_parts[-1]] = param_value
        
        
        plan_str = json.dumps(bound_plan)
        for param_name, param_value in parameters.items():
            placeholder = f"{ { {param_name}} } "
            plan_str = plan_str.replace(placeholder, str(param_value))
        
        bound_plan = json.loads(plan_str)
        
        return bound_plan
    
    def _execute_plan(
        self,
        plan: List[Dict[str, Any]],
        env: Any
    ) -> Dict[str, Any]:
        """Run each plan step via env.step(..., is_actspec_internal=True)."""
        try:
            for step in plan:
                primitive = step.get("primitive", "").upper()
                
                
                action_str = self._plan_step_to_action(step)
                
                if not action_str:
                    continue
                
                
                
                if hasattr(env, 'step'):
                    try:
                        
                        result = env.step(action_str, is_actspec_internal=True)
                        
                        if result is False:
                            
                            return {
                                "success": False,
                                "error": f"Invalid action: {action_str}"
                            }
                    except Exception as e:
                        return {
                            "success": False,
                            "error": f"Failed to execute action: {str(e)}"
                        }
                else:
                    return {
                        "success": False,
                        "error": "Environment does not have step method"
                    }
            
            return {
                "success": True,
                "error": None
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _plan_step_to_action(self, step: Dict[str, Any]) -> Optional[str]:
        """Serialize one plan step to env action string."""
        primitive = step.get("primitive", "").upper()
        
        if primitive == "CLICK":
            target = step.get("target", {})
            strategy = target.get("strategy", "")
            value = target.get("value", "")
            
            if strategy == "element_id":
                return f"click [{value}]"
            elif strategy == "text":
                
                return None
            else:
                return None
        
        elif primitive == "TYPE":
            target = step.get("target", {})
            strategy = target.get("strategy", "")
            value = target.get("value", "")
            text = step.get("text", "")
            enter = step.get("enter", False)
            enter_flag = "1" if enter else "0"
            
            if strategy == "element_id":
                return f"type [{value}] [{text}] [{enter_flag}]"
            else:
                return None
        
        elif primitive == "SCROLL":
            direction = step.get("direction", "down")
            return f"scroll [{direction}]"
        
        elif primitive == "GOTO":
            
            if "raw" in step and step["raw"]:
                raw_str = step["raw"].strip()
                
                if raw_str.startswith("goto [") and raw_str.count("[") >= 2:
                    return raw_str
                
                elif raw_str.startswith("goto "):
                    url = raw_str[5:].strip()  
                    return f"goto [{url}] [0]"  
                
                else:
                    url = raw_str
            else:
                url = step.get("url", "")
            
            return f"goto [{url}] [0]"
        
        elif primitive == "STOP":
            return "stop"
        
        elif primitive == "GOBACK":
            return "go_back"
        
        elif primitive == "GOHOME":
            return "go_home"
        
        else:
            return None
