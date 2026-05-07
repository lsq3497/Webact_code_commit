"""
ActSpec执行器：参数绑定和执行
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

# 语义定位返回多候选时的专用标记，表示结构性歧义；上层禁止 retry/wait，直接进入语义变化处理
LOCATE_MULTIPLE_CANDIDATES = "_multiple_candidates_"


class ActSpecExecutor:
    """ActSpec执行器，负责参数绑定和执行"""
    
    def __init__(self):
        """初始化ActSpec执行器"""
        pass
    
    def execute_actspec(
        self,
        actspec: Dict[str, Any],
        parameters: Dict[str, Any],
        env: Any
    ) -> Dict[str, Any]:
        """
        执行ActSpec
        
        Args:
            actspec: ActSpec字典
            parameters: 参数字典
            env: 环境对象（需要实现step方法）
        
        Returns:
            Status字典：{"success": bool, "error": str | None}
        """
        call_record = {"executor_success": None, "error": None}
        try:
            from .pre_condition_checker import PreConditionChecker
            from .locate_executor import LocateExecutor
            
            pre_checker = PreConditionChecker()
            locate_executor = LocateExecutor()

            # 为本次ActSpec调用准备离线评估记录（如果环境支持）
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
            
            # 1. 记录Pre-condition检查结果（仅用于离线评估，不阻止执行）
            # 注意：Pre-condition已经在过滤阶段检查过了，这里只记录结果
            pre_condition = actspec.get("pre_condition", {})
            if pre_condition:
                page = self._get_page_from_env(env)
                if page:
                    try:
                        # 获取observation文本用于检查
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
                        # 不再因为pre-condition不满足而失败，因为已经在过滤阶段检查过了
                        # 如果pre-condition不满足，记录警告但继续执行
                        if not is_satisfied:
                            print(f"[ActSpec] 警告：Pre-condition不满足（但继续执行，因为已在过滤阶段检查）: {reason}")
                    except Exception as e:
                        # 如果检查出错，记录但继续执行
                        print(f"[警告] Pre-condition检查出错: {e}，继续执行")
                        call_record["pre_condition_satisfied"] = None
            
            # 2. 不再一次性整条 plan 的 Locate；改为每步执行前按 step 调用 Locate（见下方 step 循环）
            
            # 3. 记录执行前状态（使用 env 的同步接口，避免异步 page.content() 返回 coroutine）
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
            
            # 用于离线统计：LLM 调整次数与上限（达到上限视为复用无效，记为失败）
            plan = actspec.get("plan", [])
            max_llm_adjustments = max(0, len(plan) - 1)
            call_record["max_llm_adjustments"] = max_llm_adjustments
            call_record["llm_adjustment_count"] = 0
            call_record["reached_adjustment_limit"] = False
            
            # 4. 参数绑定（占位符 target.value 在每步执行前由本 step 的 Locate 结果覆盖）
            bindings = actspec.get("bindings", {})
            bound_plan = self._bind_parameters(plan, bindings, parameters)
            locate = actspec.get("locate", {})
            locate_executor = LocateExecutor()
            page = self._get_page_from_env(env)
            observation_processor = self._get_observation_processor_from_env(env)
            target_elements = locate.get("target_elements", [])
            
            # 5. Step 级执行循环：每步 Locate → StepExecutor → 失败时自动修复或语义变化
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
                    # 每步执行前：为本 step 做 Locate，写入 step["target"]["value"]（仅内存副本）
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
                    # 多候选：禁止 retry/wait，直接进入语义变化处理
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
                    
                    # 自动修复：Wait → Retry（仅一次）或 单次 Retry（no_page_change）
                    wait_retry_done = last_wait_retry_step == step_index
                    no_page_retry_done = last_no_page_retry_step == step_index
                    if failure_reason in ("target_not_interactable", "action_exception") and not wait_retry_done:
                        time.sleep(WAIT_RETRY_SEC)
                        last_wait_retry_step = step_index
                        # 重试本 step（会再次 Locate）
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
                    
                    # 自动修复已穷尽：合理中间态判定后语义变化
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
                    # 内层循环正常结束，所有 step 完成
                    break
            
            # ActSpec 成功判定：必须通过 Post-condition 验证（与论文一致）
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
            call_record["reached_adjustment_limit"] = True  # 异常视为达到上限，统计记为失败
            return {
                "success": False,
                "error": str(e)
            }
    
    def _get_page_from_env(self, env: Any) -> Any:
        """从环境对象中获取Page对象"""
        # 支持多种环境对象结构
        if hasattr(env, 'page'):
            return env.page
        elif hasattr(env, 'webarena_env') and hasattr(env.webarena_env, 'page'):
            return env.webarena_env.page
        elif hasattr(env, 'browser') and hasattr(env.browser, 'page'):
            return env.browser.page
        else:
            return None
    
    def _get_observation_text_for_step(self, env: Any) -> str:
        """获取当前页面 observation 文本，供语义变化处理使用（同步获取）。兼容 obs 为 str 或 dict、text 为 str/list。"""
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
        使用Locate更新plan中的target
        
        定位策略优先级：
        1. 优先使用语义定位策略（semantic）- 即使LLM提供了element_id参数，也会先尝试语义定位
        2. 如果语义定位失败，再使用element_id策略作为备选，并验证元素是否存在
        
        这样可以避免因为元素ID动态变化导致的执行失败，提高ActSpec的稳定性和抗UI漂移能力。
        """
        import copy
        updated_plan = copy.deepcopy(plan)
        
        # 获取observation_processor（用于从定位到的元素中提取element_id）
        observation_processor = self._get_observation_processor_from_env(env)
        if not observation_processor:
            print("[ActSpec] 警告：无法获取observation_processor，跳过自动定位")
            return updated_plan
        
        target_elements = locate.get("target_elements", [])
        
        # 遍历plan的每个步骤
        for step_idx, step in enumerate(updated_plan):
            # 检查这个步骤是否有未提供的参数占位符
            target = step.get("target", {})
            if target.get("strategy") == "element_id":
                value = target.get("value", "")
                # 检查是否是占位符（value 可能为 int，先转为 str）
                value_s = value if isinstance(value, str) else str(value) if value is not None else ""
                if value_s.startswith("{{") and value_s.endswith("}}"):
                    param_name = value_s[2:-2]
                    # 调用_locate_element_for_step，它会：
                    # 1. 优先使用语义定位策略（即使参数已提供）
                    # 2. 如果语义定位失败，再使用element_id策略作为备选，并验证元素是否存在
                    element_id = self._locate_element_for_step(
                        step_idx, target_elements, locate_executor, page, 
                        observation_processor, parameters
                    )
                    if element_id:
                        # 更新plan中的占位符
                        step["target"]["value"] = element_id
                        provided_value = parameters.get(param_name)
                        if provided_value and str(provided_value) != str(element_id):
                            print(f"[ActSpec] 步骤 {step_idx}: 定位成功，找到 element_id={element_id}（覆盖LLM提供的值 {provided_value}）")
                        else:
                            print(f"[ActSpec] 步骤 {step_idx}: 定位成功，找到 element_id={element_id}")
                    else:
                        # 所有定位策略均失败（包括语义定位和element_id验证）
                        # 为避免未验证的 element_id 误点错误元素，这里不再使用参数中的 element_id 作为兜底
                        # 保持占位符不变，让后续执行阶段的自动修复 / 语义变化处理来接管
                        print(f"[ActSpec] 警告：步骤 {step_idx}: 所有定位策略均失败，保持占位符 {value_s}，后续交由自动修复或语义变化处理")
        
        return updated_plan
    
    def _get_observation_processor_from_env(self, env: Any) -> Any:
        """从环境对象中获取observation_processor"""
        if not env:
            return None
        
        try:
            # 尝试从 webarena_env 获取
            if hasattr(env, 'webarena_env'):
                webarena_env = env.webarena_env
                if hasattr(webarena_env, 'observation_handler'):
                    handler = webarena_env.observation_handler
                    if hasattr(handler, 'text_processor'):
                        return handler.text_processor
                # 尝试从 info 中获取
                if hasattr(webarena_env, 'info') and isinstance(webarena_env.info, dict):
                    if 'observation_processor' in webarena_env.info:
                        return webarena_env.info['observation_processor']
        except Exception as e:
            print(f"[ActSpec] 获取observation_processor失败: {e}")
        
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
        为指定步骤定位元素并返回element_id
        
        定位策略优先级：
        1. 优先使用语义定位策略（semantic）- 最稳定，抗UI漂移
        2. 如果语义定位失败，再使用element_id策略作为备选
        
        即使提供了element_id参数，也会先尝试语义定位，只有在语义定位失败时
        才会使用element_id参数，并且会验证element_id对应的元素是否存在。
        
        Args:
            step_idx: plan步骤索引
            target_elements: locate配置中的target_elements列表
            locate_executor: LocateExecutor实例（保留以备将来使用）
            page: Playwright Page对象
            observation_processor: ObservationProcessor实例
            parameters: 参数字典
        
        Returns:
            element_id字符串，如果定位失败则返回None
        """
        # 查找对应步骤的target_element配置
        target_element_config = None
        for te in target_elements:
            # 支持两种格式：使用step字段或按索引匹配
            if "step" in te and te["step"] == step_idx:
                target_element_config = te
                break
        
        # 如果没有找到，尝试使用第一个target_element（向后兼容）
        if not target_element_config and len(target_elements) > step_idx:
            target_element_config = target_elements[step_idx]
        
        if not target_element_config:
            return None
        
        # 使用LocateExecutor定位元素
        strategies = target_element_config.get("strategies", [])
        if not strategies:
            return None
        
        # 按优先级排序策略
        strategies = sorted(strategies, key=lambda x: x.get("priority", 999))
        
        # 先尝试所有语义策略（priority 1）
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
                        print(f"[ActSpec] 步骤 {step_idx}: 语义定位返回多候选，不启发式选择，返回多候选标记")
                        return LOCATE_MULTIPLE_CANDIDATES
                    print(f"[ActSpec] 步骤 {step_idx}: 语义定位成功，找到 element_id={semantic_result}")
                    return semantic_result
                break  # 语义策略已尝试完毕（无候选或多候选已在上方返回）
        
        # 语义定位失败，尝试element_id策略作为备选（priority 2）
        print(f"[ActSpec] 步骤 {step_idx}: 语义定位失败，尝试使用element_id策略作为备选")
        for strategy in strategies:
            strategy_type = strategy.get("strategy")
            conditions = strategy.get("conditions", {})
            
            if strategy_type == "element_id":
                element_id_param = conditions.get("element_id", "")
                element_id_value = None
                
                # 解析element_id参数
                if element_id_param.startswith("{{") and element_id_param.endswith("}}"):
                    param_name = element_id_param[2:-2]
                    if param_name in parameters and parameters[param_name]:
                        element_id_value = str(parameters[param_name])
                elif element_id_param:
                    # 直接提供了element_id（非占位符）
                    element_id_value = element_id_param
                
                # 如果提供了element_id，验证元素是否存在
                if element_id_value:
                    if self._verify_element_id_exists(element_id_value, page, observation_processor):
                        print(f"[ActSpec] 步骤 {step_idx}: element_id策略成功，找到 element_id={element_id_value}")
                        return element_id_value
                    else:
                        print(f"[ActSpec] 步骤 {step_idx}: element_id={element_id_value} 对应的元素不存在，跳过")
        
        print(f"[ActSpec] 步骤 {step_idx}: 所有定位策略均失败")
        return None
    
    def _locate_by_semantic_and_extract_id(
        self,
        conditions: Dict[str, Any],
        page: Any,
        observation_processor: Any
    ) -> Optional[str]:
        """
        使用语义策略定位元素，并从observation_processor中提取element_id。
        若匹配到多个候选，返回 LOCATE_MULTIPLE_CANDIDATES，不启发式选择。
        
        Returns:
            element_id 字符串；0 个匹配返回 None；1 个匹配返回该 id；多个匹配返回 LOCATE_MULTIPLE_CANDIDATES
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
        验证元素是否真的存在于页面中（使用Playwright）
        
        注意：由于page可能是异步的，这里使用宽松的验证策略。
        如果验证失败，仍然返回True，因为元素已经在obs_nodes_info中存在。
        
        Args:
            page: Playwright Page对象（可能是同步或异步）
            role: 元素role
            name: 元素名称
        
        Returns:
            True如果元素存在或验证失败，False否则
        """
        # 由于我们已经从obs_nodes_info中找到了元素，验证步骤可以更宽松
        # 如果page是异步的，locator.count()可能需要await，这里简化处理
        try:
            if role and name:
                locator = page.get_by_role(role=role, name=name, exact=False)
                # 尝试获取count，如果是异步的可能会失败，但会被except捕获
                try:
                    count = locator.count()
                    # 如果count是协程，说明是异步的，直接返回True
                    if hasattr(count, '__await__'):
                        return True
                    return count > 0
                except (TypeError, AttributeError):
                    # 可能是异步API，无法同步调用，返回True（因为obs_nodes_info中已存在）
                    return True
            elif name:
                # 尝试多种方式查找
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
            # 如果验证失败，仍然返回True（因为obs_nodes_info中已经存在）
            # 这样可以避免因为Playwright API问题导致的误判
            return True
        
        return False
    
    def _verify_element_id_exists(
        self,
        element_id: str,
        page: Any,
        observation_processor: Any
    ) -> bool:
        """
        验证element_id对应的元素是否存在于当前页面
        
        Args:
            element_id: 元素ID字符串
            page: Playwright Page对象（可能是同步或异步）
            observation_processor: ObservationProcessor实例
        
        Returns:
            True如果元素存在，False否则
        """
        if not observation_processor or not hasattr(observation_processor, 'obs_nodes_info'):
            # 如果没有observation_processor，无法验证，返回False
            return False
        
        try:
            # 尝试将element_id转换为整数
            element_id_int = int(element_id)
            
            # 检查element_id是否在obs_nodes_info中存在
            if str(element_id_int) in observation_processor.obs_nodes_info:
                # 进一步验证：尝试获取节点信息
                try:
                    node = observation_processor.get_node_info_by_element_id(element_id_int)
                    if node:
                        # 元素存在，返回True
                        return True
                except Exception:
                    # 如果获取节点信息失败，但element_id在obs_nodes_info中，仍然返回True
                    # 因为obs_nodes_info已经包含了当前页面的所有元素信息
                    return True
            
            # element_id不在obs_nodes_info中，说明元素不存在
            return False
            
        except (ValueError, TypeError):
            # element_id不是有效的整数，无法验证
            return False
        except Exception as e:
            # 其他错误，保守处理，返回False
            print(f"[ActSpec] 验证element_id存在性时出错: {e}")
            return False
    
    def _bind_parameters(
        self,
        plan: List[Dict[str, Any]],
        bindings: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        参数绑定：将参数值注入到 plan 中。
        若某参数在 parameters 中不存在，则不写入任何值（plan/locate 中原有的 target.value 或文本保持不动）。

        Args:
            plan: 可执行计划
            bindings: 参数绑定规则
            parameters: 参数字典

        Returns:
            绑定后的 plan
        """
        # 深拷贝plan
        import copy
        bound_plan = copy.deepcopy(plan)
        
        # 遍历绑定规则
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
                
                # 根据field路径设置值
                if field == "text":
                    step["text"] = str(param_value)
                elif field == "target.value":
                    if "target" in step:
                        step["target"]["value"] = str(param_value)
                elif field == "url":
                    step["url"] = str(param_value)
                else:
                    # 支持嵌套字段（如 "target.value"）
                    field_parts = field.split(".")
                    current = step
                    for part in field_parts[:-1]:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                    current[field_parts[-1]] = param_value
        
        # 替换plan中的占位符
        plan_str = json.dumps(bound_plan)
        for param_name, param_value in parameters.items():
            placeholder = f"{{{{{param_name}}}}}"
            plan_str = plan_str.replace(placeholder, str(param_value))
        
        bound_plan = json.loads(plan_str)
        
        return bound_plan
    
    def _execute_plan(
        self,
        plan: List[Dict[str, Any]],
        env: Any
    ) -> Dict[str, Any]:
        """
        执行plan
        
        Args:
            plan: 可执行计划
            env: 环境对象
        
        Returns:
            Status字典
        """
        try:
            for step in plan:
                primitive = step.get("primitive", "").upper()
                
                # 将plan step转换为action字符串
                action_str = self._plan_step_to_action(step)
                
                if not action_str:
                    continue
                
                # 执行action
                # 直接传递字符串给env.step()，并标记为ActSpec内部action
                if hasattr(env, 'step'):
                    try:
                        # 直接传递字符串，并标记为ActSpec内部action（不增加step计数）
                        result = env.step(action_str, is_actspec_internal=True)
                        # env.step可能返回status字典或None，需要检查
                        if result is False:
                            # 如果返回False，表示action无效
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
        """
        将plan step转换为action字符串
        
        Args:
            step: plan step字典
        
        Returns:
            action字符串或None
        """
        primitive = step.get("primitive", "").upper()
        
        if primitive == "CLICK":
            target = step.get("target", {})
            strategy = target.get("strategy", "")
            value = target.get("value", "")
            
            if strategy == "element_id":
                return f"click [{value}]"
            elif strategy == "text":
                # 需要查找element_id，这里简化处理
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
            # 优先使用raw字段（如果存在），否则使用url字段
            if "raw" in step and step["raw"]:
                raw_str = step["raw"].strip()
                # 如果raw字符串已经是完整的goto命令格式（goto [url] [flag]），直接返回
                if raw_str.startswith("goto [") and raw_str.count("[") >= 2:
                    return raw_str
                # 如果raw字符串是"goto url"格式，提取URL并格式化为标准格式
                elif raw_str.startswith("goto "):
                    url = raw_str[5:].strip()  # 移除"goto "前缀
                    return f"goto [{url}] [0]"  # 默认flag为0（不在新标签页打开）
                # 否则将raw_str作为URL
                else:
                    url = raw_str
            else:
                url = step.get("url", "")
            # 默认flag为0（不在新标签页打开）
            return f"goto [{url}] [0]"
        
        elif primitive == "STOP":
            return "stop"
        
        elif primitive == "GOBACK":
            return "go_back"
        
        elif primitive == "GOHOME":
            return "go_home"
        
        else:
            return None
