"""
Post-condition验证器：验证Post-condition是否满足
"""

from typing import Dict, Any, Tuple


class PostConditionVerifier:
    """验证Post-condition是否满足"""
    
    def verify_post_condition(
        self,
        post_condition: Dict[str, Any],
        page: Any,  
        pre_url: str,
        pre_obs: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        验证post-condition是否满足
        
        Returns:
            (is_satisfied: bool, reason: str)
        """
        post_url = self._get_page_url(page)
        post_obs = {"text": self._get_page_content(page)}
        
        
        url_change = post_condition.get("url_change", {})
        if url_change:
            expected_type = url_change.get("type")
            if expected_type == "navigate" and post_url == pre_url:
                return False, "期望URL变化，但URL未变化"
            elif expected_type == "stay" and post_url != pre_url:
                return False, "期望URL不变，但URL已变化"
            elif expected_type == "navigate":
                pattern = url_change.get("pattern")
                if pattern and not self._match_url_pattern(post_url, pattern):
                    return False, f"URL变化不符合模式: {pattern}"
        
        
        element_appears = post_condition.get("element_appears", [])
        for element_config in element_appears:
            if element_config.get("required", True):
                if not self._check_element_exists(element_config, page):
                    return False, f"必需的新元素未出现: {element_config}"
        
        
        element_disappears = post_condition.get("element_disappears", [])
        for element_config in element_disappears:
            if self._check_element_exists(element_config, page):
                return False, f"期望消失的元素仍存在: {element_config}"
        
        
        text_appears = post_condition.get("text_appears", [])
        post_text = post_obs.get("text", "")
        pre_text = pre_obs.get("text", "")
        for text_config in text_appears:
            if text_config.get("required", True):
                text = text_config.get("text")
                if text not in post_text or text in pre_text:
                    return False, f"必需的文本未出现: {text}"
        
        return True, "Post-condition满足"
    
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
        return ""
    
    def _get_page_content(self, page: Any) -> str:
        """获取页面内容"""
        if page is None:
            return ""
        try:
            if hasattr(page, 'content'):
                content = page.content()
                if callable(content):
                    return content()
                return str(content)
            elif hasattr(page, 'page') and hasattr(page.page, 'content'):
                content = page.page.content()
                if callable(content):
                    return content()
                return str(content)
            else:
                return ""
        except Exception:
            return ""
    
    def _match_url_pattern(self, url: str, pattern: str) -> bool:
        """匹配URL模式，委托 pre_condition_checker 的 match_url_pattern。"""
        from actspec.pre_condition_checker import match_url_pattern
        return match_url_pattern(url, pattern)
    
    def _check_element_exists(
        self,
        element_config: Dict[str, Any],
        page: Any
    ) -> bool:
        """检查元素是否存在"""
        strategy = element_config.get("strategy", "")
        conditions = element_config.get("conditions", {})
        
        if strategy == "semantic":
            role = conditions.get("role")
            label = conditions.get("label")
            text = conditions.get("text")
            
            try:
                if role and (label or text):
                    name = label or text
                    locator = page.get_by_role(role=role, name=name)
                    return locator.count() > 0
                elif label:
                    locator = page.get_by_label(label)
                    return locator.count() > 0
                elif text:
                    locator = page.get_by_text(text)
                    return locator.count() > 0
            except Exception:
                pass
        
        return False

