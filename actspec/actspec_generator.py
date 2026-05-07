"""
ActSpec生成器：将切分的action序列转换为符合规范的ActSpec JSON
"""

import copy
import json
import re
from typing import Dict, List, Any, Optional
from llms import lm_config, utils as llm_utils

from .trace_segmenter import is_page_change_action
from .negative_constraint_utils import build_action_history_prefix


class ActSpecGenerator:
    """ActSpec生成器，将segment转换为ActSpec JSON"""
    
    def __init__(self, llm_config: Optional[lm_config.LMConfig] = None):
        """
        初始化ActSpec生成器
        
        Args:
            llm_config: LLM配置，如果为None则使用默认配置
        """
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
        """
        生成单个ActSpec
        
        Args:
            action_sequence: action序列（从segment中提取）
            context: 上下文信息（site, page, url）
            task_info: 任务信息
            trajectory: 完整轨迹（用于提取Locate/Pre/Post）
            segment_start_idx: segment在trajectory中的起始索引
            segment_end_idx: segment在trajectory中的结束索引
        
        Returns:
            ActSpec字典，包含7类信息
        """
        
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
                print(f"[ActSpec] 构建 action_history_prefix 失败: {e}")
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
            print(f"[Warning] ActSpec验证发现问题 ({actspec.get('action_id', 'unknown')}):")
            for error in validation_errors:
                print(f"  - {error}")
            
            actspec = self._auto_fix_actspec(actspec, validation_errors)
        
        
        if is_failed and plan_before_param:
            actspec["plan"] = plan_before_param
        
        return actspec
    
    def _extract_parameters(self, action_sequence: List[Any]) -> Dict[str, Any]:
        """
        提取参数定义
        
        Args:
            action_sequence: action序列
        
        Returns:
            参数定义字典
        """
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
        
        text_values_str = ", ".join(text_values[:5]) if text_values else "无"
        element_ids_str = ", ".join(element_ids[:5]) if element_ids else "无"
        
        system_prompt = """你是一个参数提取专家。分析action序列，识别可以参数化的值。

参数类型可以是：
- enum: 枚举值（如状态、类型等），需要提供candidates列表
- string: 字符串（如搜索关键词、输入文本等）
- number: 数字（整数或浮点数）
- boolean: 布尔值

参数提取原则：
1. 识别在多次使用中可能变化的值（如搜索关键词、筛选条件）
2. 识别具有业务语义的值（如订单状态、用户类型）
3. 避免参数化固定值（如固定的按钮文本、固定的URL路径）
4. 对于枚举类型，需要从上下文中推断可能的候选值

请返回JSON格式的参数定义。"""
        
        user_prompt = f"""分析以下action序列，识别可参数化的值：

Action序列：
{action_summary}

序列中的文本值：{text_values_str}
序列中的element_id：{element_ids_str}

注意：
- 对于TYPE操作，如果element_id可能变化，应该提取为参数（如type_id、input_id等）
- 对于CLICK操作，如果element_id可能变化，应该提取为参数（如button_id、click_id等）
- 对于TYPE操作中的文本，应该提取为参数（如search_keyword、input_text等）

返回JSON格式，例如：
{ 
  "status": { 
    "type": "enum",
    "candidates": ["pending", "paid", "shipped"],
    "description": "order status to filter",
    "optional": false
  } ,
  "search_keyword": { 
    "type": "string",
    "optional": false,
    "description": "search keyword to query"
  } ,
  "page_number": { 
    "type": "number",
    "optional": true,
    "description": "page number for pagination"
  } 
} 

只返回JSON，不要其他文字。"""
        
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
        """
        提取上下文信息
        
        Args:
            action_sequence: action序列
            context: 基础上下文（site, page, url）
            task_info: 任务信息
        
        Returns:
            上下文信息字典
        """
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
        """
        生成语义描述（使用LLM）
        
        Args:
            action_sequence: action序列
            context: 上下文信息
        
        Returns:
            描述字典（summary, when_to_use, effect）
        """
        action_strs = [str(action) for action in action_sequence]
        action_summary = "\n".join([f"{i}: {a}" for i, a in enumerate(action_strs)])
        
        system_prompt = """你是一个动作描述专家。为给定的action序列生成自然语言描述。

描述应该：
1. 强调「意图」而不是「怎么点」
2. 明确输入 → 输出效果
3. 使用业务语义，而不是DOM语义
4. **重要**：不要包含具体的值（如具体的数字、具体的搜索关键词等），应该使用抽象的描述
5. 例如：不要说"输入Worcester"，而要说"输入搜索关键词"
6. 例如：不要说"点击element_id=79"，而要说"点击目标元素"
7. 例如：不要说"输入数字93和1"，而要说"输入搜索参数" """
        
        user_prompt = f"""为以下action序列生成描述：

上下文：
- Site: {context.get('site', 'unknown')}
- Page: {context.get('page', 'unknown')}

Action序列：
{action_summary}

返回JSON格式：
{ 
  "summary": "简短总结（一句话）",
  "when_to_use": "什么时候使用这个动作",
  "effect": "执行后的效果"
} 

只返回JSON，不要其他文字。"""
        
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
                "summary": "执行一系列操作",
                "when_to_use": "需要执行这些操作时",
                "effect": "完成操作"
            }
        
        return description
    
    def _create_executable_plan(self, action_sequence: List[Any]) -> List[Dict[str, Any]]:
        """
        创建可执行计划
        
        Args:
            action_sequence: action序列
        
        Returns:
            可执行计划列表
        """
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
        """
        参数化plan：将plan中应该参数化的值替换为占位符
        
        改进点：
        1. 确保所有element_id都被参数化（特别是TYPE操作的target.value）
        2. 确保所有文本输入都被参数化（TYPE操作的text字段）
        3. 确保参数类型匹配（text字段使用string类型，target.value使用number类型）
        
        Args:
            plan: 可执行计划
            parameters: 参数定义
            action_sequence: 原始action序列
        
        Returns:
            参数化后的plan
        """
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
        """
        创建参数绑定规则
        
        Args:
            parameters: 参数定义
            plan: 可执行计划
        
        Returns:
            绑定规则字典
        """
        bindings = {}
        
        
        if not parameters:
            return bindings
        
        
        
        plan_str = json.dumps(plan, indent=2, ensure_ascii=False)
        param_names = list(parameters.keys())
        param_descriptions = {name: params.get("description", "") for name, params in parameters.items()}
        
        system_prompt = """你是一个参数绑定专家。分析可执行计划，确定每个参数应该绑定到plan的哪个位置。

绑定规则：
1. 参数应该绑定到plan中对应的字段（如text、target.value、url、option等）
2. 一个参数可以绑定到多个位置（如果它在plan中出现多次）
3. 绑定位置使用step索引（从0开始）和字段路径（如"text"、"target.value"、"url"等）

返回JSON格式的绑定规则。"""
        
        user_prompt = f"""分析以下可执行计划，为每个参数创建绑定规则：

参数列表：
{json.dumps(param_descriptions, indent=2, ensure_ascii=False)}

可执行计划：
{plan_str}

返回JSON格式，例如：
{ 
  "search_keyword": { 
    "bind_to": [
      { "step": 0, "field": "text"} 
    ]
  } ,
  "status": { 
    "bind_to": [
      { "step": 1, "field": "target.value"} 
    ]
  } 
} 

只返回JSON，不要其他文字。"""
        
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
        """
        创建参数绑定规则（fallback方法，基于占位符匹配）
        
        Args:
            parameters: 参数定义
            plan: 可执行计划
        
        Returns:
            绑定规则字典
        """
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
        """
        生成action_id和action_name
        
        Args:
            context: 上下文信息
            description: 描述信息
            action_sequence: action序列
        
        Returns:
            (action_id, action_name) 元组
        """
        site = context.get("site", "unknown")
        page = context.get("page", "unknown")
        summary = description.get("summary", "action")
        
        
        system_prompt = """你是一个命名专家。根据动作描述生成一个简洁、清晰的PascalCase动作名称。

命名规则：
1. 使用PascalCase（首字母大写的驼峰命名）
2. 名称应该简洁（2-4个单词）
3. 名称应该反映动作的核心意图
4. 避免使用过于通用的词汇（如"Action"、"Do"等）
5. 使用动词+名词的组合（如"FilterOrders"、"SearchProducts"）
6. **重要**：不要包含具体的值（如具体的数字、具体的搜索关键词等），应该使用抽象的描述
7. 例如：不要命名为"SearchWorcester"，而应该命名为"SearchByKeyword"
8. 例如：不要命名为"ClickElement79"，而应该命名为"ClickElement"
9. 避免使用失败相关的词汇（如"Failed"、"Error"、"Attempt"等），除非动作本身就是处理失败场景的

只返回名称，不要其他文字。"""
        
        user_prompt = f"""为以下动作生成PascalCase名称：

动作描述：{summary}
上下文：{site} - {page}

示例：
- "Filter orders by status" -> "FilterOrdersByStatus"
- "Search for products" -> "SearchProducts"
- "Navigate to user profile" -> "NavigateToUserProfile"
- "Submit login form" -> "SubmitLoginForm"

只返回名称，不要其他文字。"""
        
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
        """
        使用LLM检测ActSpec是否表示失败场景
        
        Args:
            actspec: ActSpec字典
            action_sequence: 原始action序列
            context: 上下文信息
        
        Returns:
            (is_failed: bool, failure_reason: str, constraint_subtype: str) 元组。
            constraint_subtype 为 "readiness" | "disambiguation" | "unspecified" 之一。
        """
        
        
        plan = actspec.get("plan", [])
        
        
        
        for action in action_sequence:
            action_str = str(action).strip().lower()
            
            
            if action_str.startswith("stop") or "stop [" in action_str:
                print(f"[过滤] 检测到stop类型的action，跳过失败检测: {action_str[:100]}")
                return False, "", "unspecified"
        
        
        for step in plan:
            primitive = step.get("primitive", "")
            raw = step.get("raw", "")
            if primitive == "STOP" or (isinstance(raw, str) and raw.strip().lower().startswith("stop")):
                print(f"[过滤] Plan中包含stop primitive，跳过失败检测")
                return False, "", "unspecified"
        description = actspec.get("description", {})
        action_name = actspec.get("action_name", "")
        action_id = actspec.get("action_id", "")
        
        
        plan_str = json.dumps(plan, indent=2, ensure_ascii=False)
        description_str = json.dumps(description, indent=2, ensure_ascii=False)
        action_sequence_str = "\n".join([f"{i}: {str(a)}" for i, a in enumerate(action_sequence)])
        
        system_prompt = """你是一个轨迹分析专家。你的任务是判断给定的ActSpec是否表示一个失败或未完成的场景。

**重要：以下情况不应该被认为是失败场景**：
- **stop类型的action**：如果action序列或plan中包含"stop [N/A - ...]"或类似的stop action，这是手动实现的正常停止，不应该被认为是失败。stop action通常用于表示任务完成、无法继续、或需要停止的情况，这些都是正常的控制流，不是错误。
- **note类型的action**：note [内容] 是文本备忘录，供 LLM 后续行动记忆，不是可执行操作，不应视为失败。若 plan 中仅有 note/UNKNOWN 且 raw 为 "note [...]"，应返回 is_failed=false。

失败场景的特征包括（但不限于）：
1. **Plan中包含UNKNOWN primitive**：表示无法执行或未知的操作（但排除stop类型的正常停止）
2. **描述中明确提到失败**：如"未能成功"、"未成功"、"无法"、"不能"、"失败"、"错误"等
3. **Action名称中包含失败标识**：如"failed"、"error"、"attempt"、"失败"、"错误"等
4. **描述中暗示操作未完成**：如"意识到需要..."、"转而寻找..."、"技术限制"等
5. **Plan中缺少关键步骤**：操作序列不完整，无法达成预期目标
6. **上下文不匹配导致的失败**：如在错误的平台上尝试操作

**负约束子类型（仅当 is_failed=true 时填写 constraint_subtype）**：
- **readiness**：页面尚未就绪时就执行了动作。典型信号：刚发生 URL 跳转/弹窗/刷新后多次 click/type，但无有效状态变化（避免抢跑）。
- **disambiguation**：页面上存在多个相似 target，agent 反复选错同一类。典型信号：多次对同类元素（相同 role/text）点击均未触发预期变化。
- **unspecified**：其他失败类型，无法明确归入上述两类。

请仔细分析给定的ActSpec，判断它是否表示失败场景。特别注意：如果包含stop类型的action，应该返回is_failed=false。"""
        
        user_prompt = f"""分析以下ActSpec，判断它是否表示失败场景：

Action ID: {action_id}
Action Name: {action_name}

上下文：
- Site: {context.get('site', 'unknown')}
- Page: {context.get('page', 'unknown')}
- URL Pattern: {context.get('url_pattern', '/')}

Action序列：
{action_sequence_str}

Plan（可执行计划）：
{plan_str}

Description（描述）：
{description_str}

请返回JSON格式：
{ 
  "is_failed": true/false,
  "failure_reason": "如果is_failed为true，说明失败原因；如果为false，可以为空字符串",
  "constraint_subtype": "仅当is_failed为true时必填，且必须为以下之一：readiness（页面未就绪时就执行了动作，如刚跳转/弹窗后立即操作导致无效果）、disambiguation（页面上多个相似元素时反复选错同一类目标）、unspecified（其他失败类型）",
  "confidence": 0.0-1.0之间的置信度
} 

只返回JSON，不要其他文字。"""
        
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
                print(f"[Warning] 低置信度的失败判断 (confidence={confidence}): {action_id}")
            
            return is_failed, failure_reason, constraint_subtype
            
        except Exception as e:
            print(f"[Warning] LLM失败检测失败: {e}, 使用fallback方法")
            
            return self._is_failed_actspec_fallback(actspec), "", "unspecified"
    
    def _is_failed_actspec_fallback(self, actspec: Dict[str, Any]) -> bool:
        """
        Fallback失败检测方法（仅在LLM失败时使用）
        
        Args:
            actspec: ActSpec字典
        
        Returns:
            如果是失败场景，返回True
        """
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
        if any(keyword in action_name for keyword in ["failed", "error", "attempt", "失败", "错误"]):
            return True
        
        return False
    
    def _validate_actspec(self, actspec: Dict[str, Any]) -> List[str]:
        """
        验证ActSpec的正确性
        
        Args:
            actspec: ActSpec字典
        
        Returns:
            错误列表，如果为空则表示验证通过
        """
        errors = []
        parameters = actspec.get("parameters", {})
        plan = actspec.get("plan", [])
        bindings = actspec.get("bindings", {})
        
        
        for param_name in bindings.keys():
            if param_name not in parameters:
                errors.append(f"绑定中引用的参数 '{param_name}' 未在parameters中定义")
        
        
        plan_str = json.dumps(plan, ensure_ascii=False)
        for param_name in parameters.keys():
            placeholder = f"{ { {param_name}} } "
            has_placeholder = placeholder in plan_str
            has_binding = param_name in bindings
            
            if not has_placeholder and not has_binding:
                errors.append(f"参数 '{param_name}' 在plan中没有占位符，也没有绑定规则")
        
        
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
                                f"plan[{step_idx}].text 使用了number类型参数 '{param_name}'，"
                                f"应该使用string类型参数"
                            )
            
            
            if primitive in ["CLICK", "TYPE"] and "target" in step:
                target_value = step.get("target", {}).get("value", "")
                if target_value and target_value.startswith("{{") and target_value.endswith("}}"):
                    param_name = target_value[2:-2]
                    if param_name in parameters:
                        param_type = parameters[param_name].get("type", "string")
                        if param_type != "number":
                            errors.append(
                                f"plan[{step_idx}].target.value 使用了{param_type}类型参数 '{param_name}'，"
                                f"应该使用number类型参数（element_id）"
                            )
        
        
        for param_name, binding_info in bindings.items():
            bind_to = binding_info.get("bind_to", [])
            for bind_rule in bind_to:
                step_idx = bind_rule.get("step")
                if step_idx is None or step_idx < 0 or step_idx >= len(plan):
                    errors.append(
                        f"参数 '{param_name}' 的绑定规则中step索引 {step_idx} 无效"
                        f"（plan长度为{len(plan)}）"
                    )
        
        return errors
    
    def _auto_fix_actspec(self, actspec: Dict[str, Any], errors: List[str]) -> Dict[str, Any]:
        """
        自动修复ActSpec中的常见问题
        
        Args:
            actspec: ActSpec字典
            errors: 验证错误列表
        
        Returns:
            修复后的ActSpec字典
        """
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
        """
        二次检查：修复plan中未参数化的实际值
        
        检查plan中所有的value字段，如果它们不是占位符格式（{{param_name}}），
        而是实际值（数字、字符串等），则根据strategy类型和参数定义，
        自动替换为合适的占位符。
        
        Args:
            actspec: ActSpec字典
            action_sequence: 原始action序列（用于辅助匹配）
        
        Returns:
            修复后的ActSpec字典
        """
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
        
        
        print(f"[PostProcess] 检测到未参数化的值，开始修复 ActSpec: {actspec.get('action_id', 'unknown')}")
        
        
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
                            print(f"[PostProcess] 自动创建参数: {param_name}")
                        
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
                            print(f"[PostProcess] 修复 step[{step_idx}].target.value: '{value}' -> '{ { {matched_param}} } '")
            
            
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
                        print(f"[PostProcess] 修复 step[{step_idx}].text: '{text}' -> '{ { {matched_param}} } '")
            
            
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
                        print(f"[PostProcess] 修复 step[{step_idx}].option: '{option}' -> '{ { {matched_param}} } '")
            
            
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
                        print(f"[PostProcess] 修复 step[{step_idx}].url: '{url}' -> '{ { {matched_param}} } '")
        
        fixed_actspec["plan"] = plan
        fixed_actspec["bindings"] = bindings
        return fixed_actspec
    
    def _get_value_from_action_at_step(
        self,
        action_sequence: List[Any],
        step_idx: int,
        field: str
    ) -> Optional[Any]:
        """
        从 action_sequence 中指定 step 的 action 里提取指定字段的值，用于填充 candidates。
        field 为 "target.value" 时提取 element_id；为 "text" 时提取 type 的文本。
        """
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
        """
        从轨迹 action_sequence 中按 bindings 提取每个参数的实际取值，写入 parameters 的 candidates，
        确保保存的 ActSpec 能直接复用动作轨迹（执行时可用 candidates 中的 value 填充 plan 的 target.value）。
        """
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
        """
        检查值是否是占位符格式（{{param_name}}）
        
        Args:
            value: 要检查的值
        
        Returns:
            如果是占位符格式，返回True
        """
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
        """
        使用LLM辅助匹配参数
        
        Args:
            value: 实际值
            field_type: 字段类型（"element_id", "text", "option", "url"）
            primitive: primitive类型（"CLICK", "TYPE", "SELECT"等）
            candidate_params: 候选参数字典
            all_params: 所有参数字典
        
        Returns:
            匹配的参数名，如果没有匹配则返回None
        """
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
        
        system_prompt = """你是一个参数匹配专家。根据实际值和字段类型，从候选参数中选择最合适的参数。

匹配原则：
1. 根据字段类型选择参数类型：
   - element_id (strategy=element_id): 应该使用number类型参数
   - text: 应该使用string类型参数
   - option: 应该使用enum类型参数（如果有）
   - url: 应该使用string类型参数
2. 根据参数描述和实际值，选择语义最匹配的参数
3. 如果参数是enum类型，检查实际值是否在candidates中

只返回参数名，不要其他文字。"""
        
        user_prompt = f"""实际值: {value}
字段类型: {field_type}
Primitive类型: {primitive}

候选参数:
{json.dumps(param_descriptions, indent=2, ensure_ascii=False)}

请选择最合适的参数名。只返回参数名，不要其他文字。"""
        
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
                
                print(f"[Warning] LLM返回的参数名 '{response}' 不在候选列表中，使用第一个候选参数")
                return list(candidate_params.keys())[0]
        except Exception as e:
            print(f"[Warning] LLM参数匹配失败: {e}, 使用第一个候选参数")
            return list(candidate_params.keys())[0] if candidate_params else None
    
    def _extract_locate_strategies(
        self,
        action_sequence: List[Any],
        trajectory: List[Dict[str, Any]],
        segment_start_idx: int,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        提取多策略定位器
        
        Args:
            action_sequence: action序列
            trajectory: 完整轨迹
            segment_start_idx: segment在trajectory中的起始索引
        
        Returns:
            Locate字典
        """
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
                print(f"[Warning] LLM增强定位策略失败: {e}")
        
        return locate
    
    def _extract_element_ids_from_action(self, action_str: str) -> List[str]:
        """从action字符串中提取element_id列表"""
        element_ids = []
        
        matches = re.findall(r"(?:click|type|hover|select)\s*\[(\d+)\]", action_str, re.I)
        element_ids.extend(matches)
        return element_ids
    
    def _get_element_param_name(self, element_id: str, action_str: str) -> str:
        """根据action类型和element_id生成参数名"""
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
        从已定义的 parameters 中查找匹配的参数名
        
        优先查找包含该 element_id 的参数，如果找不到则使用默认命名规则
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
        生成定位策略列表（按优先级排序）
        
        策略优先级：
        1. semantic（语义特征）- 最稳定，抗UI漂移
        2. relative_position（相对位置）- 中等稳定（如果可用）
        3. element_id（元素ID）- 最不稳定，但作为fallback
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
        使用LLM增强定位策略（可选，用于复杂场景）
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
        提取执行前适用条件
        
        Args:
            action_sequence: action序列
            trajectory: 完整轨迹
            segment_start_idx: segment在trajectory中的起始索引
            context: 上下文信息
        
        Returns:
            Pre-condition字典
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
            print(f"[Warning] LLM增强pre-condition失败: {e}")
        
        return pre_condition
    
    def _infer_required_elements(
        self,
        action_sequence: List[Any],
        observation_text: str
    ) -> List[Dict[str, Any]]:
        """从action序列推断必须存在的元素"""
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
        """生成元素策略的签名用于去重"""
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
        从action序列推断必须存在的区域
        
        实现：从 accessibility tree 中提取区域信息
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
            print(f"[Warning] 推断必需区域时出错: {e}")
            return []
    
    def _check_modal_exists(self, observation_text: str) -> bool:
        """
        检查是否有modal存在
        
        改进：使用更智能的检测方法，不仅检查关键词，还检查 accessibility tree 结构
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
        使用LLM增强pre-condition
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
        提取执行后验证条件
        
        Args:
            action_sequence: action序列
            trajectory: 完整轨迹
            segment_start_idx: segment在trajectory中的起始索引
            segment_end_idx: segment在trajectory中的结束索引
            context: 上下文信息
        
        Returns:
            Post-condition字典
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
            print(f"[Warning] LLM增强post-condition失败: {e}")
        
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
        推断应该排除的状态
        
        排除状态是指不应该执行ActSpec的状态，例如：
        - 错误页面
        - 加载中页面
        - 特定的错误URL
        - 有错误消息显示的页面
        """
        excluded_states = []
        
        if not observation_text:
            return excluded_states
        
        from .accessibility_tree_parser import AccessibilityTreeParser
        
        try:
            parser = AccessibilityTreeParser()
            tree = parser.parse(observation_text)
            
            
            error_keywords = [
                "error", "错误", "failed", "失败", "invalid", "无效",
                "exception", "异常", "not found", "404", "500"
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
            
            
            loading_keywords = ["loading", "加载中", "please wait", "请稍候"]
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
                                for keyword in ["invalid", "required", "错误", "无效"])):
                            excluded_states.append({
                                "type": "form_validation_error",
                                "description": "Form contains validation errors",
                                "form_id": element.get("element_id")
                            })
                            break
            
        except Exception as e:
            print(f"[Warning] 推断排除状态时出错: {e}")
        
        return excluded_states
    
    def _extract_url_pattern(self, url: str, context: Dict[str, Any]) -> str:
        """从URL提取模式"""
        
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
        检测新出现的元素
        
        放宽检测逻辑：
        1. 只要element_id变化就认为是新元素（即使语义特征相同）
        2. 只要完整签名不匹配就认为是新元素（不再检查部分签名）
        3. 检测所有有element_id的元素，包括状态变化的元素
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
            
            print(f"[Warning] 检测新元素时出错: {e}")
            return []
        
        return new_elements
    
    def _get_element_signature(self, element: Dict[str, Any]) -> str:
        """
        生成元素的签名（用于比较）
        
        改进：添加更多信息以提高唯一性，包括 text 和 element_id
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
        生成元素的部分签名（不包含element_id，用于检测相似元素）
        
        用于检测element_id变化但元素实际相同的情况
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
        检测消失的元素
        
        放宽检测逻辑：
        1. 只要完整签名不匹配就认为是消失的元素（不再检查部分签名）
        2. 只要element_id不在post中就认为是消失的元素（即使语义特征可能相同）
        """
        from .accessibility_tree_parser import AccessibilityTreeParser
        
        try:
            parser = AccessibilityTreeParser()
            pre_tree = parser.parse(pre_obs_text)
            post_tree = parser.parse(post_obs_text)
        except Exception as e:
            print(f"[Warning] 解析observation时出错: {e}")
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
        判断文本是否为错误/失败类提示，不应作为 post_condition.text_appears 的成功指标。
        与 _infer_excluded_states 中的错误检测保持一致，避免将错误信息误记为“执行后出现的内容”。
        """
        if not text or not text.strip():
            return False
        text_lower = text.strip().lower()
        error_keywords = [
            "error", "错误", "failed", "失败", "invalid", "无效",
            "exception", "异常", "not found", "404", "500",
            "loading", "加载中", "please wait", "请稍候",
            "synchronize", "synchronization", "magento business intelligence",  
        ]
        return any(kw in text_lower for kw in error_keywords)
    
    def _detect_new_texts(
        self,
        pre_obs_text: str,
        post_obs_text: str
    ) -> List[Dict[str, Any]]:
        """
        检测新出现的文本
        
        放宽检测逻辑：
        1. 检测所有文本变化，不仅仅是关键词
        2. 提取执行前后observation中的所有文本元素，比较差异
        3. 记录所有新增的文本内容
        """
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
            print(f"[Warning] 检测新文本时出错: {e}")
            
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
        检测区域更新
        
        实现：比较执行前后的区域状态，检测区域内容的变化
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
            print(f"[Warning] 检测区域更新时出错: {e}")
            return []
    
    def _detect_modal_changes(
        self,
        pre_obs_text: str,
        post_obs_text: str
    ) -> List[Dict[str, Any]]:
        """
        检测modal变化
        
        实现：比较执行前后的 modal 状态，检测 modal 的出现、消失或内容变化
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
            print(f"[Warning] 检测modal变化时出错: {e}")
            return []
    
    def _llm_enhance_post_condition(
        self,
        post_condition: Dict[str, Any],
        action_sequence: List[Any],
        pre_obs_text: str,
        post_obs_text: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        使用LLM增强post-condition
        
        分析action执行前后的页面状态变化，提取执行后验证条件。
        如果基础检测方法没有检测到变化，LLM会尝试从语义层面识别状态变化。
        """
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
        
        system_prompt = """你是一个状态变化分析专家。分析action执行前后的页面状态变化，提取执行后验证条件（post-condition）。

Post-condition应该包括：
1. URL/route变化（如果有导航到新页面）
2. 新元素出现（确认操作成功的元素、新出现的按钮、链接等）
3. 元素消失（如loading modal、临时提示等）
4. 重要文本出现（如确认消息、错误提示、成功提示等）
5. 区域更新（form、table、main区域等的内容变化）
6. Modal变化（打开、关闭、更新）

注意：
- 只记录重要的、可复用的状态变化
- 避免过于具体的变化（如具体的element_id），使用语义特征（role、label、text）
- 如果操作后页面基本没有变化（如只是切换了选中状态），可以只记录URL变化或保持为空
- 优先识别操作成功的标志（如新页面、确认消息、新功能区域出现等）
- 如果检测到错误状态，可以在text_appears中记录错误文本

返回JSON格式的post-condition，格式如下：
{
  "url_change": {
    "type": "stay" | "navigate",
    "pattern": "URL模式（如果是navigate）"
  },
  "element_appears": [
    {
      "strategy": "semantic",
      "conditions": {
        "role": "元素角色",
        "label": "元素标签",
        "text": "元素文本"
      },
      "required": true
    }
  ],
  "element_disappears": [
    {
      "strategy": "semantic",
      "conditions": {
        "role": "元素角色",
        "label": "元素标签",
        "text": "元素文本"
      }
    }
  ],
  "text_appears": [
    {
      "text": "出现的文本内容",
      "required": true
    }
  ],
  "region_updates": [
    {
      "type": "appeared" | "updated" | "disappeared",
      "role": "区域角色（main、complementary等）",
      "description": "描述"
    }
  ],
  "modal_changes": [
    {
      "type": "opened" | "closed" | "updated",
      "role": "modal角色",
      "label": "modal标签",
      "description": "描述"
    }
  ]
}"""
        
        user_prompt = f"""上下文：
- Site: {context.get('site', 'unknown')}
- Page: {context.get('page', 'unknown')}
- URL Pattern: {context.get('url_pattern', '/')}

Action序列：
{action_summary_str}

执行前页面观察（Accessibility Tree文本，前2000字符）：
{pre_obs_snippet}

执行后页面观察（Accessibility Tree文本，前2000字符）：
{post_obs_snippet}

当前Post-condition（基础检测结果）：
{json.dumps(post_condition, indent=2, ensure_ascii=False)}

请分析执行前后的状态变化，完善post-condition。确保：
1. 包含所有重要的状态变化
2. 识别确认操作成功的标志（如新元素、文本、URL变化等）
3. 识别错误状态的标志（如果有）
4. 使用语义特征而不是具体的element_id（提高复用性）
5. 如果基础检测已经识别了一些变化，可以在此基础上补充

只返回JSON格式的post-condition，不要其他文字。"""
        
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
            print(f"[Warning] LLM返回的post-condition不是有效的JSON: {e}")
            print(f"[Warning] LLM响应: {response[:500]}")
            return post_condition
        except Exception as e:
            print(f"[Warning] LLM增强post-condition失败: {e}")
            return post_condition
    
    def _find_action_in_trajectory(
        self,
        action_str: str,
        trajectory: List[Dict[str, Any]],
        start_idx: int
    ) -> Optional[int]:
        """
        在trajectory中查找对应的action
        
        Args:
            action_str: action字符串
            trajectory: 完整轨迹
            start_idx: 搜索的起始索引
        
        Returns:
            找到的trajectory索引，如果找不到则返回None
        """
        
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
