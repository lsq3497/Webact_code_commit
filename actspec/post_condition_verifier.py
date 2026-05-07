"""
Post-condition verifier.
"""

from typing import Dict, Any, Tuple


class PostConditionVerifier:
    """Check ActSpec post-conditions after an action."""

    def verify_post_condition(
        self,
        post_condition: Dict[str, Any],
        page: Any,  
        pre_url: str,
        pre_obs: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Returns (ok, reason)."""
        post_url = self._get_page_url(page)
        post_obs = {"text": self._get_page_content(page)}
        
        
        url_change = post_condition.get("url_change", {})
        if url_change:
            expected_type = url_change.get("type")
            if expected_type == "navigate" and post_url == pre_url:
                return False, "Expected URL to change but it did not"
            elif expected_type == "stay" and post_url != pre_url:
                return False, "Expected URL to stay the same but it changed"
            elif expected_type == "navigate":
                pattern = url_change.get("pattern")
                if pattern and not self._match_url_pattern(post_url, pattern):
                    return False, f"URL change does not match pattern: {pattern}"
        
        
        element_appears = post_condition.get("element_appears", [])
        for element_config in element_appears:
            if element_config.get("required", True):
                if not self._check_element_exists(element_config, page):
                    return False, f"Required new element did not appear: {element_config}"
        
        
        element_disappears = post_condition.get("element_disappears", [])
        for element_config in element_disappears:
            if self._check_element_exists(element_config, page):
                return False, f"Element expected to disappear is still present: {element_config}"
        
        
        text_appears = post_condition.get("text_appears", [])
        post_text = post_obs.get("text", "")
        pre_text = pre_obs.get("text", "")
        for text_config in text_appears:
            if text_config.get("required", True):
                text = text_config.get("text")
                if text not in post_text or text in pre_text:
                    return False, f"Required text did not appear: {text}"
        
        return True, "Post-condition satisfied"
    
    def _get_page_url(self, page: Any) -> str:
        """Resolve URL from page wrapper."""
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
        """Serialized page content when available."""
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
        """Delegate to pre_condition_checker.match_url_pattern."""
        from actspec.pre_condition_checker import match_url_pattern
        return match_url_pattern(url, pattern)
    
    def _check_element_exists(
        self,
        element_config: Dict[str, Any],
        page: Any
    ) -> bool:
        """Existence check via semantic Playwright locators."""
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

