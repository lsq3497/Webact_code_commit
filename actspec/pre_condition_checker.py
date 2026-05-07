"""
Pre-condition检查器：检查Pre-condition是否满足
"""

import re
from typing import Dict, Any, Tuple, Optional
from urllib.parse import urlparse


def match_url_pattern(url: str, pattern: str) -> bool:
    """匹配URL与模式（支持 path、path+query 及任意 {xxx} 占位符）。

    占位符：将模式中任意 {...} 视为占位符，匹配一段非 /?#& 的字符（path 段或 query 值）。
    无占位符时做整段匹配（等价末尾 $）；有占位符时同样整段匹配。
    若 pattern 含 '?'，则用 path + '?' + query 参与匹配；否则仅用 path。
    """
    if not pattern:
        return True
    parsed = urlparse(url)
    path = parsed.path or "/"
    query = parsed.query
    if "?" in pattern:
        url_part = path + ("?" + query if query else "")
    else:
        url_part = path
    placeholder_regex = r"[^/?#&]+"
    temp = "\x00PL\x00"
    pattern_with_placeholders = re.sub(r"\{[^}]*\}", temp, pattern)
    pattern_escaped = re.escape(pattern_with_placeholders)
    pattern_regex = pattern_escaped.replace(re.escape(temp), placeholder_regex)
    pattern_regex += r"$"
    return bool(re.match(pattern_regex, url_part))


class PreConditionChecker:
    """检查Pre-condition是否满足"""
    
    def check_pre_condition(
        self,
        pre_condition: Dict[str, Any],
        page: Any,  # Playwright Page对象、环境对象或observation文本
        parameters: Dict[str, Any],
        observation_text: Optional[str] = None  # 可选的observation文本（accessibility tree）
    ) -> Tuple[bool, str]:
        """
        检查pre-condition是否满足
        
        Args:
            pre_condition: Pre-condition配置
            page: Playwright Page对象或环境对象
            parameters: 参数字典
            observation_text: 可选的observation文本（accessibility tree格式）
        
        Returns:
            (is_satisfied: bool, reason: str)
        """
        # 1. 检查URL模式
        current_url = self._get_page_url(page)
        url_pattern = pre_condition.get("url_pattern", "")
        if url_pattern and not self._match_url_pattern(current_url, url_pattern):
            return False, f"URL不匹配: 当前={current_url}, 要求={url_pattern}"
        
        # 2. 获取observation文本（优先使用传入的observation_text，否则尝试从page/env获取）
        # 仅当未传入（None）时回退；空字符串视为显式传入，避免在 env.observation() 内再次调用 observation() 导致递归
        obs_text = observation_text
        if obs_text is None:
            obs_text = self._get_observation_text(page)
        
        # 3. 检查必须存在的元素
        required_elements = pre_condition.get("required_elements", [])
        for element_config in required_elements:
            if not self._check_element_exists(element_config, page, obs_text):
                return False, f"必需元素不存在: {element_config}"
        
        # 4. 检查必须存在的区域
        required_regions = pre_condition.get("required_regions", [])
        for region_config in required_regions:
            if not self._check_region_exists(region_config, page, obs_text):
                return False, f"必需区域不存在: {region_config}"
        
        # 5. 检查modal状态
        required_modals = pre_condition.get("required_modals", [])
        if required_modals:
            for modal_config in required_modals:
                if not self._check_modal_state(modal_config, page):
                    return False, f"Modal状态不匹配: {modal_config}"
        
        # 6. 检查排除的状态
        excluded_states = pre_condition.get("excluded_states", [])
        for excluded_state in excluded_states:
            if self._check_excluded_state(excluded_state, page):
                return False, f"处于排除状态: {excluded_state}"
        
        return True, "Pre-condition满足"
    
    def _get_page_url(self, page: Any) -> str:
        """获取页面URL"""
        if page is None:
            return ""
        if hasattr(page, 'url'):
            try:
                return page.url
            except Exception:
                pass
        elif hasattr(page, 'page') and hasattr(page.page, 'url'):
            try:
                return page.page.url
            except Exception:
                pass
        elif hasattr(page, 'get_url'):
            try:
                return page.get_url()
            except Exception:
                pass
        elif hasattr(page, 'online_url'):
            try:
                return page.online_url
            except Exception:
                pass
        return ""
    
    def _get_observation_text(self, page: Any) -> Optional[str]:
        """获取observation文本（accessibility tree格式）"""
        if page is None:
            return None
        
        # 尝试从env对象获取observation
        if hasattr(page, 'observation'):
            try:
                obs = page.observation()
                if isinstance(obs, str):
                    return obs
                elif isinstance(obs, dict) and 'text' in obs:
                    return obs['text']
            except Exception:
                pass
        
        # 尝试从webarena_env获取
        if hasattr(page, 'webarena_env'):
            try:
                if hasattr(page.webarena_env, 'obs'):
                    obs = page.webarena_env.obs
                    if isinstance(obs, dict) and 'text' in obs:
                        content = obs['text']
                        if isinstance(content, tuple) and len(content) > 0:
                            return content[0]  # 第一个元素是文本
                        elif isinstance(content, str):
                            return content
            except Exception:
                pass
        
        return None
    
    def _match_url_pattern(self, url: str, pattern: str) -> bool:
        """匹配URL模式，委托给模块级 match_url_pattern。"""
        return match_url_pattern(url, pattern)
    
    def _check_element_exists(
        self,
        element_config: Dict[str, Any],
        page: Any,
        observation_text: Optional[str] = None
    ) -> bool:
        """检查元素是否存在"""
        strategy = element_config.get("strategy", "")
        conditions = element_config.get("conditions", {})
        
        if strategy == "semantic":
            role = conditions.get("role")
            label = conditions.get("label")
            text = conditions.get("text")
            
            # 优先使用accessibility tree文本进行匹配（更可靠）
            if observation_text:
                return self._check_element_in_accessibility_tree(
                    role, label, text, observation_text
                )
            
            # 如果没有observation文本，尝试使用Playwright API
            # 尝试多种匹配方式，提高匹配成功率
            # 优先级：text > label，因为text通常更可靠
            
            # 1. 如果有role和text，优先使用text（text通常更可靠）
            if role and text:
                try:
                    # 先尝试精确匹配
                    locator = page.get_by_role(role=role, name=text, exact=True)
                    if locator.count() > 0:
                        return True
                except Exception:
                    pass
                try:
                    # 再尝试模糊匹配（不要求完全匹配）
                    locator = page.get_by_role(role=role, name=text, exact=False)
                    if locator.count() > 0:
                        return True
                except Exception:
                    pass
            
            # 2. 如果有role和label，尝试使用label
            if role and label:
                try:
                    locator = page.get_by_role(role=role, name=label, exact=True)
                    if locator.count() > 0:
                        return True
                except Exception:
                    pass
                try:
                    locator = page.get_by_role(role=role, name=label, exact=False)
                    if locator.count() > 0:
                        return True
                except Exception:
                    pass
            
            # 3. 如果有text，尝试直接通过text查找（不限制role）
            if text:
                try:
                    locator = page.get_by_text(text, exact=True)
                    if locator.count() > 0:
                        return True
                except Exception:
                    pass
                try:
                    locator = page.get_by_text(text, exact=False)
                    if locator.count() > 0:
                        return True
                except Exception:
                    pass
            
            # 4. 如果有label，尝试通过label查找
            if label:
                try:
                    locator = page.get_by_label(label)
                    if locator.count() > 0:
                        return True
                except Exception:
                    pass
            
            # 5. 如果有role但没有name，尝试通过role查找（宽松匹配）
            if role and not label and not text:
                try:
                    # 使用模糊匹配，查找所有该role的元素
                    locator = page.locator(f'[role="{role}"]')
                    if locator.count() > 0:
                        return True
                except Exception:
                    pass
        
        return False
    
    def _check_element_in_accessibility_tree(
        self,
        role: Optional[str],
        label: Optional[str],
        text: Optional[str],
        observation_text: str
    ) -> bool:
        """在accessibility tree文本中查找元素"""
        from .accessibility_tree_parser import AccessibilityTreeParser
        
        parser = AccessibilityTreeParser()
        tree = parser.parse(observation_text)
        
        # 遍历所有元素，查找匹配的元素
        for element in tree.get("elements", []):
            element_role = element.get("role")
            element_label = element.get("label")
            element_text = element.get("text")
            
            # 匹配逻辑：role必须匹配，label或text必须匹配
            role_match = not role or element_role == role
            
            if role_match:
                # 如果指定了text，优先匹配text
                if text:
                    if element_text and text.lower() in element_text.lower():
                        return True
                    if element_label and text.lower() in element_label.lower():
                        return True
                
                # 如果指定了label，匹配label
                if label:
                    if element_label and label.lower() in element_label.lower():
                        return True
                    if element_text and label.lower() in element_text.lower():
                        return True
                
                # 如果只指定了role，匹配成功
                if not text and not label:
                    return True
        
        return False
    
    def _check_region_exists(
        self,
        region_config: Dict[str, Any],
        page: Any,
        observation_text: Optional[str] = None
    ) -> bool:
        """检查区域是否存在"""
        # 简化实现：返回True
        # 可以根据需要扩展
        return True
    
    def _check_modal_state(
        self,
        modal_config: Dict[str, Any],
        page: Any
    ) -> bool:
        """检查modal状态"""
        # 简化实现：返回True
        # 可以根据需要扩展
        return True
    
    def _check_excluded_state(
        self,
        excluded_state: Dict[str, Any],
        page: Any
    ) -> bool:
        """检查是否处于排除状态"""
        url_pattern = excluded_state.get("url_pattern", "")
        if url_pattern:
            current_url = self._get_page_url(page)
            return self._match_url_pattern(current_url, url_pattern)
        return False

