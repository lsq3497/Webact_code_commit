"""
轨迹切分器：使用LLM分析轨迹，识别可复用的action序列

设计约定：ActSpec 仅记录**单页内**的动作序列并复用。goto、go_back、new_tab 等会导致 URL/页面
变化的 action 不纳入记录与复用，仅作为**片段切分标志**：该类动作之后的部分切分为新的 segment，
并记录新页面的上下文。
"""

import json
import re
import uuid
from typing import Dict, List, Any, Optional
from llms import lm_config, utils as llm_utils


# 导致页面/URL 变化的 action 前缀（用于切分与排除，不纳入 segment 的 actions）
PAGE_CHANGE_ACTION_PREFIXES = (
    "goto ",
    "goto[",
    "go_back",
    "go_forward",
    "new_tab",
    "close_tab",
    "close tab",
    "go_home",
)


def is_page_change_action(action_str: str) -> bool:
    """判断是否为会导致页面变化的 action（用作切分标志，不纳入记录与复用）。"""
    if not action_str or not isinstance(action_str, str):
        return False
    al = action_str.strip().lower()
    return any(al.startswith(prefix) for prefix in PAGE_CHANGE_ACTION_PREFIXES)


class TraceSegmenter:
    """轨迹切分器，将完整的trajectory切分为可复用的segment"""
    
    def __init__(self, llm_config: Optional[lm_config.LMConfig] = None):
        """
        初始化轨迹切分器
        
        Args:
            llm_config: LLM配置，如果为None则使用默认配置
        """
        self.llm_config = llm_config
        if self.llm_config is None:
            # 默认配置
            self.llm_config = lm_config.LMConfig(
                provider="openai",
                model="gpt-4-turbo",
                mode="chat",
                gen_config={
                    "temperature": 0.1,
                    "max_tokens": 2000,
                    "top_p": 1.0,
                    "context_length": 0,
                }
            )
    
    def segment_trajectory(
        self, 
        trajectory: List[Dict[str, Any]], 
        task_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        切分轨迹为可复用的segment
        
        Args:
            trajectory: 完整的轨迹数据
            task_info: 任务信息（包含task_id, sites, intent等）
        
        Returns:
            List[Segment]: 切分后的segment列表，每个segment包含：
                - segment_id: str
                - actions: List[str] (标准化的action字符串列表)
                - context: Dict (site, page, url)
                - description: str
                - segment_type: str
        """
        # 验证输入
        if not trajectory:
            print("[Warning] Empty trajectory provided")
            return []
        
        if not isinstance(trajectory, list):
            print(f"[Warning] Invalid trajectory type: {type(trajectory)}")
            return []
        
        # 确保task_info是dict
        if not isinstance(task_info, dict):
            task_info = {}
        
        try:
            # 调用LLM进行切分分析
            llm_segments = self._call_llm_for_segmentation(trajectory, task_info)
            
            # 如果LLM切分失败或返回空，使用fallback
            if not llm_segments:
                print("[Info] LLM segmentation returned empty, using fallback")
                llm_segments = self._fallback_segmentation(trajectory)
            
            # 提取action序列并组装为Segment结构
            segments = self._extract_action_sequences(llm_segments, trajectory, task_info)
            
            # 过滤掉粒度不足的segment
            valid_segments = [s for s in segments if self._validate_segment_granularity(s)]
            
            return valid_segments
        except Exception as e:
            print(f"[Error] Failed to segment trajectory: {e}")
            import traceback
            traceback.print_exc()
            # 尝试使用fallback
            try:
                llm_segments = self._fallback_segmentation(trajectory)
                segments = self._extract_action_sequences(llm_segments, trajectory, task_info)
                return segments
            except Exception as e2:
                print(f"[Error] Fallback segmentation also failed: {e2}")
                return []
    
    def _normalize_action(self, action: Any) -> str:
        """
        将action标准化为字符串格式
        
        Args:
            action: 可能是字符串、Action对象、dict等
        
        Returns:
            标准化的action字符串
        """
        if action is None:
            return "NONE"
        
        # 如果已经是字符串，直接返回
        if isinstance(action, str):
            return action.strip() if action.strip() else "NONE"
        
        # 如果是Action对象（TypedDict，有action_type字段）
        if isinstance(action, dict) and "action_type" in action:
            try:
                from browser_env.actions import action2str
                # 尝试使用action2str转换
                action_str = action2str(action, action_set_tag="id_accessibility_tree", semantic_element="")
                # 移除语义信息部分，只保留action部分
                if " where " in action_str:
                    action_str = action_str.split(" where ")[0].strip()
                return action_str
            except Exception as e:
                # 如果转换失败，尝试手动构建
                try:
                    action_type = action.get("action_type", -1)
                    element_id = action.get("element_id", "")
                    
                    # 根据action_type构建字符串
                    if action_type == 0:  # CLICK
                        return f"click [{element_id}]"
                    elif action_type == 1:  # TYPE
                        text = action.get("text", [])
                        if isinstance(text, list):
                            # 尝试转换text列表
                            try:
                                from browser_env.actions import _id2key
                                text_str = "".join([_id2key[i] for i in text if i < len(_id2key)])
                            except:
                                text_str = str(text)
                        else:
                            text_str = str(text)
                        return f"type [{element_id}] [{text_str}] [0]"
                    elif action_type == 2:  # HOVER
                        return f"hover [{element_id}]"
                    elif action_type == 3:  # SCROLL
                        direction = action.get("direction", "down")
                        return f"scroll [{direction}]"
                    elif action_type == 4:  # KEY_PRESS
                        key_comb = action.get("key_comb", "")
                        return f"press [{key_comb}]"
                    elif action_type == 5:  # GOTO_URL
                        url = action.get("url", "")
                        return f"goto [{url}]"
                    else:
                        # 其他类型，使用字符串表示
                        return str(action)
                except Exception as e2:
                    # 如果都失败了，返回字符串表示
                    return str(action)
        
        # 如果是普通dict，尝试提取action相关字段
        if isinstance(action, dict):
            # 尝试提取常见的action字段
            if "action" in action:
                return self._normalize_action(action["action"])
            elif "action_type" in action:
                return self._normalize_action(action)  # 递归处理
            else:
                # 普通dict，转换为字符串
                return str(action)
        
        # 其他类型，转换为字符串
        return str(action)
    
    def _call_llm_for_segmentation(
        self, 
        trajectory: List[Dict[str, Any]], 
        task_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        调用LLM进行轨迹切分分析
        
        Args:
            trajectory: 完整轨迹
            task_info: 任务信息
        
        Returns:
            LLM返回的切分结果列表
        """
        # 验证trajectory格式
        if not trajectory or not isinstance(trajectory, list):
            print("[Warning] Invalid trajectory format, using empty segments")
            return []
        
        # 构建轨迹摘要（只包含action信息，减少token消耗）
        trajectory_summary = []
        valid_steps = 0
        for i, step in enumerate(trajectory):
            # 支持多种trajectory格式
            # 格式1: {"action": ..., "observation": ..., "url": ...}
            # 格式2: Action对象（TypedDict）
            # 格式3: StateInfo对象
            try:
                if isinstance(step, dict):
                    action = step.get("action")
                    url = step.get("url", "")
                else:
                    # 如果是Action对象或其他格式
                    action = step
                    url = ""
                
                # 标准化action为字符串
                action_str = self._normalize_action(action)
                
                # 只记录有效的action（非NONE）
                if action_str and action_str != "NONE":
                    trajectory_summary.append({
                        "step": i,
                        "action": action_str,
                        "url": url if url else "",
                    })
                    valid_steps += 1
            except Exception as e:
                print(f"[Warning] Failed to process step {i}: {e}, skipping")
                continue
        
        # 如果没有有效的action，返回空列表
        if valid_steps == 0:
            print("[Warning] No valid actions found in trajectory")
            return []
        
        # 构建prompt
        system_prompt = """你是一个轨迹分析专家。你的任务是将用户的操作轨迹切分为可复用的动作序列（segment）。

每个segment应该：
1. 完成一个独立的子任务（如搜索、筛选、导航、表单填写等）
2. 可以被参数化后在其他场景复用
3. 包含2-10个连续的primitive action
4. **仅包含单页内的操作**：goto、go_back、go_forward、new_tab、close_tab、go_home 等会导致页面/URL变化的动作会在后续作为切分边界单独处理，不要在同一个 segment 中跨页面混合操作。

segment_type可以是以下类型之一：
- search: 搜索操作
- filter: 筛选/过滤操作
- nav: 导航操作（点击链接、返回等）
- form: 表单填写操作
- other: 其他类型

请分析以下轨迹，返回JSON格式的切分结果。"""
        
        user_prompt = f"""任务信息：
- Task ID: {task_info.get('task_id', 'unknown')}
- Sites: {task_info.get('sites', [])}
- Intent: {task_info.get('intent', 'unknown')}

轨迹数据：
{json.dumps(trajectory_summary, indent=2, ensure_ascii=False)}

请返回JSON格式的切分结果，格式如下：
{{
  "segments": [
    {{
      "start_step": 0,
      "end_step": 3,
      "segment_type": "search",
      "description": "在搜索框中输入关键词并点击搜索按钮"
    }},
    ...
  ]
}}

只返回JSON，不要其他文字。"""
        
        # 调用LLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = llm_utils.call_llm(self.llm_config, messages)
            
            # 解析JSON响应
            # 尝试提取JSON部分（可能包含markdown代码块）
            response = response.strip()
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
            
            result = json.loads(response)
            return result.get("segments", [])
        except Exception as e:
            print(f"[Warning] LLM segmentation failed: {e}, using fallback")
            # 如果LLM调用失败，使用简单的fallback：每个action作为一个segment
            return self._fallback_segmentation(trajectory)
    
    def _fallback_segmentation(
        self, 
        trajectory: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        简单的fallback切分策略：将连续的相同类型action合并为一个segment
        """
        segments = []
        current_segment = None
        
        for i, step in enumerate(trajectory):
            try:
                # 支持多种trajectory格式
                if isinstance(step, dict):
                    action = step.get("action")
                else:
                    action = step
                
                # 标准化action
                action_str = self._normalize_action(action)
                
                # 跳过无效的action
                if not action_str or action_str == "NONE":
                    continue
                
                # 简单判断action类型
                action_lower = action_str.lower()
                if action_lower.startswith("click"):
                    segment_type = "nav"
                elif action_lower.startswith("type"):
                    segment_type = "form"
                elif action_lower.startswith("scroll"):
                    segment_type = "nav"
                elif action_lower.startswith("goto"):
                    segment_type = "nav"
                else:
                    segment_type = "other"
                
                if current_segment is None or current_segment["segment_type"] != segment_type:
                    if current_segment is not None:
                        segments.append(current_segment)
                    current_segment = {
                        "start_step": i,
                        "end_step": i,
                        "segment_type": segment_type,
                        "description": f"{segment_type}操作"
                    }
                else:
                    current_segment["end_step"] = i
            except Exception as e:
                print(f"[Warning] Failed to process step {i} in fallback segmentation: {e}")
                continue
        
        if current_segment is not None:
            segments.append(current_segment)
        
        return segments
    
    def _extract_action_sequences(
        self,
        llm_segments: List[Dict[str, Any]],
        trajectory: List[Dict[str, Any]],
        task_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        提取action序列并组装为Segment结构
        
        Args:
            llm_segments: LLM返回的切分结果
            trajectory: 完整轨迹
            task_info: 任务信息
        
        Returns:
            List[Segment]: 结构化segment列表
        """
        segments = []
        
        # 验证输入
        if not llm_segments:
            print("[Warning] No segments from LLM, returning empty list")
            return []
        
        if not trajectory:
            print("[Warning] Empty trajectory, returning empty list")
            return []
        
        for seg_idx, seg_info in enumerate(llm_segments):
            try:
                start_step = max(0, seg_info.get("start_step", 0))
                end_step = min(seg_info.get("end_step", len(trajectory) - 1), len(trajectory) - 1)
                segment_type = seg_info.get("segment_type", "other")
                description = seg_info.get("description", f"{segment_type}操作")
                
                # 验证step范围
                if start_step > end_step:
                    print(f"[Warning] Invalid step range [{start_step}, {end_step}], skipping segment {seg_idx}")
                    continue
                
                if start_step >= len(trajectory) or end_step < 0:
                    print(f"[Warning] Step range [{start_step}, {end_step}] out of bounds, skipping segment {seg_idx}")
                    continue
                
                # 按「页面变化 action」切分：goto/go_back/new_tab 等不纳入 actions，仅作切分点；
                # 该类 action 之后的部分作为新 segment，使用新 URL 作为 context。
                sub_segment_start = start_step
                actions = []
                context_urls = []
                
                for i in range(start_step, min(end_step + 1, len(trajectory))):
                    try:
                        step = trajectory[i]
                        if isinstance(step, dict):
                            action = step.get("action")
                            url = step.get("url", "")
                        else:
                            action = step
                            url = ""
                        
                        action_str = self._normalize_action(action)
                        
                        if is_page_change_action(action_str):
                            # 先提交当前子片段（仅包含本页内的 actions，不含本次 goto 等）
                            if actions:
                                from .url_utils import extract_site_and_page_from_url
                                sites = task_info.get("sites", []) if isinstance(task_info, dict) else []
                                # 使用本片段内「第一个」URL 作为上下文，表示这段动作最初发生的页面
                                context_url = context_urls[0] if context_urls else ""
                                site, page = extract_site_and_page_from_url(
                                    context_url, sites=sites, include_port=True
                                )
                                segment = {
                                    "segment_id": str(uuid.uuid4()),
                                    "actions": list(actions),
                                    "context": {"site": site, "page": page, "url": context_url},
                                    "description": description,
                                    "segment_type": segment_type,
                                    "start_step": sub_segment_start,
                                    "end_step": i - 1,
                                }
                                segments.append(segment)
                            sub_segment_start = i + 1
                            actions = []
                            context_urls = []
                            continue
                        
                        if action_str and action_str != "NONE":
                            actions.append(action_str)
                        if url and isinstance(url, str) and url.strip():
                            context_urls.append(url.strip())
                    except Exception as e:
                        print(f"[Warning] Failed to extract action from step {i}: {e}, skipping")
                        continue
                
                valid_action_count = len(actions)
                if valid_action_count == 0:
                    continue
                
                from .url_utils import extract_site_and_page_from_url
                sites = task_info.get("sites", []) if isinstance(task_info, dict) else []
                # 同样对整段最后的子片段，使用其内部第一个URL作为上下文
                context_url = context_urls[0] if context_urls else ""
                site, page = extract_site_and_page_from_url(
                    context_url, sites=sites, include_port=True
                )
                segment = {
                    "segment_id": str(uuid.uuid4()),
                    "actions": actions,
                    "context": {"site": site, "page": page, "url": context_url},
                    "description": description,
                    "segment_type": segment_type,
                    "start_step": sub_segment_start,
                    "end_step": end_step,
                }
                segments.append(segment)
            except Exception as e:
                print(f"[Warning] Failed to process segment {seg_idx}: {e}, skipping")
                continue
        
        return segments
    
    def _validate_segment_granularity(self, segment: Dict[str, Any]) -> bool:
        """
        验证segment是否具有足够的粒度
        
        Args:
            segment: segment字典
        
        Returns:
            如果粒度足够，返回True
        """
        actions = segment.get("actions", [])
        if len(actions) < 2:
            return False
        
        # 检查是否只包含重复操作
        if len(actions) >= 2:
            # 提取所有action的类型和element_id
            action_types = []
            element_ids = []
            for action_str in actions:
                action_lower = action_str.lower()
                if action_lower.startswith("click"):
                    action_types.append("click")
                    id_match = re.search(r"\[(\d+)\]", action_str)
                    if id_match:
                        element_ids.append(id_match.group(1))
                elif action_lower.startswith("type"):
                    action_types.append("type")
                    id_match = re.search(r"type\s*\[(\d+)\]", action_str)
                    if id_match:
                        element_ids.append(id_match.group(1))
                elif action_lower.startswith("scroll"):
                    action_types.append("scroll")
                elif action_lower.startswith("goto"):
                    action_types.append("goto")
                else:
                    action_types.append("other")
            
            # 如果所有action都是同一类型且element_id相同，认为是重复操作
            if len(set(action_types)) == 1 and len(set(element_ids)) == 1 and len(element_ids) > 0:
                return False
        
        return True
