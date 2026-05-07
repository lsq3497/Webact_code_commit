"""
Trace segmenter: uses an LLM to split trajectories into reusable action sequences.

Design: ActSpec records and reuses **single-page** action sequences only. Actions that change URL/page
(goto, go_back, new_tab, etc.) are not recorded for reuse; they act as **segment boundaries**. Everything
after such an action starts a new segment with fresh page context.
"""

import json
import re
import uuid
from typing import Dict, List, Any, Optional
from llms import lm_config, utils as llm_utils



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
    """True if this action changes the page (boundary only; not recorded for reuse)."""
    if not action_str or not isinstance(action_str, str):
        return False
    al = action_str.strip().lower()
    return any(al.startswith(prefix) for prefix in PAGE_CHANGE_ACTION_PREFIXES)


class TraceSegmenter:
    """Splits a full trajectory into reusable segments."""

    def __init__(self, llm_config: Optional[lm_config.LMConfig] = None):
        """
        Args:
            llm_config: LM config; if None, a default OpenAI chat config is used.
        """
        self.llm_config = llm_config
        if self.llm_config is None:
            
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
        Segment trajectory into reusable pieces.

        Args:
            trajectory: Full trajectory records.
            task_info: Task metadata (task_id, sites, intent, etc.).

        Returns:
            List of segment dicts, each with segment_id, actions (normalized strings),
            context {site, page, url}, description, segment_type.
        """
        
        if not trajectory:
            print("[Warning] Empty trajectory provided")
            return []
        
        if not isinstance(trajectory, list):
            print(f"[Warning] Invalid trajectory type: {type(trajectory)}")
            return []
        
        
        if not isinstance(task_info, dict):
            task_info = {}
        
        try:
            
            llm_segments = self._call_llm_for_segmentation(trajectory, task_info)
            
            
            if not llm_segments:
                print("[Info] LLM segmentation returned empty, using fallback")
                llm_segments = self._fallback_segmentation(trajectory)
            
            
            segments = self._extract_action_sequences(llm_segments, trajectory, task_info)
            
            
            valid_segments = [s for s in segments if self._validate_segment_granularity(s)]
            
            return valid_segments
        except Exception as e:
            print(f"[Error] Failed to segment trajectory: {e}")
            import traceback
            traceback.print_exc()
            
            try:
                llm_segments = self._fallback_segmentation(trajectory)
                segments = self._extract_action_sequences(llm_segments, trajectory, task_info)
                return segments
            except Exception as e2:
                print(f"[Error] Fallback segmentation also failed: {e2}")
                return []
    
    def _normalize_action(self, action: Any) -> str:
        """
        Normalize an action to a string (str, dict with action_type, etc.).
        """
        if action is None:
            return "NONE"
        
        
        if isinstance(action, str):
            return action.strip() if action.strip() else "NONE"
        
        
        if isinstance(action, dict) and "action_type" in action:
            try:
                from browser_env.actions import action2str
                
                action_str = action2str(action, action_set_tag="id_accessibility_tree", semantic_element="")
                
                if " where " in action_str:
                    action_str = action_str.split(" where ")[0].strip()
                return action_str
            except Exception as e:
                
                try:
                    action_type = action.get("action_type", -1)
                    element_id = action.get("element_id", "")
                    
                    
                    if action_type == 0:  
                        return f"click [{element_id}]"
                    elif action_type == 1:  
                        text = action.get("text", [])
                        if isinstance(text, list):
                            
                            try:
                                from browser_env.actions import _id2key
                                text_str = "".join([_id2key[i] for i in text if i < len(_id2key)])
                            except:
                                text_str = str(text)
                        else:
                            text_str = str(text)
                        return f"type [{element_id}] [{text_str}] [0]"
                    elif action_type == 2:  
                        return f"hover [{element_id}]"
                    elif action_type == 3:  
                        direction = action.get("direction", "down")
                        return f"scroll [{direction}]"
                    elif action_type == 4:  
                        key_comb = action.get("key_comb", "")
                        return f"press [{key_comb}]"
                    elif action_type == 5:  
                        url = action.get("url", "")
                        return f"goto [{url}]"
                    else:
                        
                        return str(action)
                except Exception as e2:
                    
                    return str(action)
        
        
        if isinstance(action, dict):
            
            if "action" in action:
                return self._normalize_action(action["action"])
            elif "action_type" in action:
                return self._normalize_action(action)  
            else:
                
                return str(action)
        
        
        return str(action)
    
    def _call_llm_for_segmentation(
        self, 
        trajectory: List[Dict[str, Any]], 
        task_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Call the LLM to propose segment boundaries; returns a list of segment dicts."""
        
        if not trajectory or not isinstance(trajectory, list):
            print("[Warning] Invalid trajectory format, using empty segments")
            return []
        
        
        trajectory_summary = []
        valid_steps = 0
        for i, step in enumerate(trajectory):
            
            
            
            
            try:
                if isinstance(step, dict):
                    action = step.get("action")
                    url = step.get("url", "")
                else:
                    
                    action = step
                    url = ""
                
                
                action_str = self._normalize_action(action)
                
                
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
        
        
        if valid_steps == 0:
            print("[Warning] No valid actions found in trajectory")
            return []
        
        
        system_prompt = """You are a trajectory analyst. Split the user's action trace into reusable segments.

Each segment should:
1. Complete one coherent sub-task (search, filter, navigation, form fill, etc.).
2. Be parameterizable for reuse in other contexts.
3. Contain 2–10 consecutive primitive actions.
4. **Stay on one page only**: actions that change URL/page (goto, go_back, go_forward, new_tab, close_tab, go_home) are handled as boundaries elsewhere—do not mix cross-page steps in one segment.

segment_type must be one of:
- search
- filter
- nav (links, back, etc.)
- form
- other

Analyze the trace below and return JSON only."""

        user_prompt = f"""Task:
- Task ID: {task_info.get('task_id', 'unknown')}
- Sites: {task_info.get('sites', [])}
- Intent: {task_info.get('intent', 'unknown')}

Trajectory:
{json.dumps(trajectory_summary, indent=2, ensure_ascii=False)}

Return JSON only, shaped like:
{{
  "segments": [
    {{
      "start_step": 0,
      "end_step": 3,
      "segment_type": "search",
      "description": "Enter keywords in the search box and click search"
    }},
    ...
  ]
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
            
            result = json.loads(response)
            return result.get("segments", [])
        except Exception as e:
            print(f"[Warning] LLM segmentation failed: {e}, using fallback")
            
            return self._fallback_segmentation(trajectory)
    
    def _fallback_segmentation(
        self, 
        trajectory: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Fallback: merge consecutive actions of the same coarse type into one segment."""
        segments = []
        current_segment = None
        
        for i, step in enumerate(trajectory):
            try:
                
                if isinstance(step, dict):
                    action = step.get("action")
                else:
                    action = step
                
                
                action_str = self._normalize_action(action)
                
                
                if not action_str or action_str == "NONE":
                    continue
                
                
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
                        "description": f"{segment_type} operation"
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
        """Materialize action lists and segment structs from LLM segment metadata."""
        segments = []
        
        
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
                description = seg_info.get("description", f"{segment_type} operation")
                
                
                if start_step > end_step:
                    print(f"[Warning] Invalid step range [{start_step}, {end_step}], skipping segment {seg_idx}")
                    continue
                
                if start_step >= len(trajectory) or end_step < 0:
                    print(f"[Warning] Step range [{start_step}, {end_step}] out of bounds, skipping segment {seg_idx}")
                    continue
                
                
                
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
                            
                            if actions:
                                from .url_utils import extract_site_and_page_from_url
                                sites = task_info.get("sites", []) if isinstance(task_info, dict) else []
                                
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
        """Return True if the segment is granular enough (not a trivial single-element repeat)."""
        actions = segment.get("actions", [])
        if len(actions) < 2:
            return False
        
        
        if len(actions) >= 2:
            
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
            
            
            if len(set(action_types)) == 1 and len(set(element_ids)) == 1 and len(element_ids) > 0:
                return False
        
        return True
