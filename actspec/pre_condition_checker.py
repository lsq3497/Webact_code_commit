"""
Pre-condition checker: test whether pre-conditions hold.
"""

import re
from typing import Dict, Any, Tuple, Optional
from urllib.parse import urlparse


def match_url_pattern(url: str, pattern: str) -> bool:
    """Match URL path (and query when pattern includes '?') against pattern with optional {placeholders}.

    Each {...} matches one non-empty segment without /?#&.
    Pattern is anchored at end ($).
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
    """Evaluate ActSpec pre-conditions."""

    def check_pre_condition(
        self,
        pre_condition: Dict[str, Any],
        page: Any,  
        parameters: Dict[str, Any],
        observation_text: Optional[str] = None  
    ) -> Tuple[bool, str]:
        """
        Args:
            pre_condition: Pre-condition block from ActSpec.
            page: Playwright Page or env wrapper.
            parameters: Bound parameters (reserved).
            observation_text: Optional a11y-tree text.

        Returns:
            (ok, reason).
        """
        
        current_url = self._get_page_url(page)
        url_pattern = pre_condition.get("url_pattern", "")
        if url_pattern and not self._match_url_pattern(current_url, url_pattern):
            return False, f"URL mismatch: current={current_url}, expected pattern={url_pattern}"
        
        
        
        obs_text = observation_text
        if obs_text is None:
            obs_text = self._get_observation_text(page)
        
        
        required_elements = pre_condition.get("required_elements", [])
        for element_config in required_elements:
            if not self._check_element_exists(element_config, page, obs_text):
                return False, f"Required element missing: {element_config}"
        
        
        required_regions = pre_condition.get("required_regions", [])
        for region_config in required_regions:
            if not self._check_region_exists(region_config, page, obs_text):
                return False, f"Required region missing: {region_config}"
        
        
        required_modals = pre_condition.get("required_modals", [])
        if required_modals:
            for modal_config in required_modals:
                if not self._check_modal_state(modal_config, page):
                    return False, f"Modal state mismatch: {modal_config}"
        
        
        excluded_states = pre_condition.get("excluded_states", [])
        for excluded_state in excluded_states:
            if self._check_excluded_state(excluded_state, page):
                return False, f"In excluded state: {excluded_state}"
        
        return True, "Pre-condition satisfied"
    
    def _get_page_url(self, page: Any) -> str:
        """Resolve page URL from page or env."""
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
        """Observation text (accessibility tree) if available."""
        if page is None:
            return None
        
        
        if hasattr(page, 'observation'):
            try:
                obs = page.observation()
                if isinstance(obs, str):
                    return obs
                elif isinstance(obs, dict) and 'text' in obs:
                    return obs['text']
            except Exception:
                pass
        
        
        if hasattr(page, 'webarena_env'):
            try:
                if hasattr(page.webarena_env, 'obs'):
                    obs = page.webarena_env.obs
                    if isinstance(obs, dict) and 'text' in obs:
                        content = obs['text']
                        if isinstance(content, tuple) and len(content) > 0:
                            return content[0]  
                        elif isinstance(content, str):
                            return content
            except Exception:
                pass
        
        return None
    
    def _match_url_pattern(self, url: str, pattern: str) -> bool:
        """Delegate to module-level match_url_pattern."""
        return match_url_pattern(url, pattern)
    
    def _check_element_exists(
        self,
        element_config: Dict[str, Any],
        page: Any,
        observation_text: Optional[str] = None
    ) -> bool:
        """Whether an element matching config exists."""
        strategy = element_config.get("strategy", "")
        conditions = element_config.get("conditions", {})
        
        if strategy == "semantic":
            role = conditions.get("role")
            label = conditions.get("label")
            text = conditions.get("text")
            
            
            if observation_text:
                return self._check_element_in_accessibility_tree(
                    role, label, text, observation_text
                )
            
            
            
            
            
            
            if role and text:
                try:
                    
                    locator = page.get_by_role(role=role, name=text, exact=True)
                    if locator.count() > 0:
                        return True
                except Exception:
                    pass
                try:
                    
                    locator = page.get_by_role(role=role, name=text, exact=False)
                    if locator.count() > 0:
                        return True
                except Exception:
                    pass
            
            
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
            
            
            if label:
                try:
                    locator = page.get_by_label(label)
                    if locator.count() > 0:
                        return True
                except Exception:
                    pass
            
            
            if role and not label and not text:
                try:
                    
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
        """Match semantic fields inside parsed accessibility tree text."""
        from .accessibility_tree_parser import AccessibilityTreeParser
        
        parser = AccessibilityTreeParser()
        tree = parser.parse(observation_text)
        
        
        for element in tree.get("elements", []):
            element_role = element.get("role")
            element_label = element.get("label")
            element_text = element.get("text")
            
            
            role_match = not role or element_role == role
            
            if role_match:
                
                if text:
                    if element_text and text.lower() in element_text.lower():
                        return True
                    if element_label and text.lower() in element_label.lower():
                        return True
                
                
                if label:
                    if element_label and label.lower() in element_label.lower():
                        return True
                    if element_text and label.lower() in element_text.lower():
                        return True
                
                
                if not text and not label:
                    return True
        
        return False
    
    def _check_region_exists(
        self,
        region_config: Dict[str, Any],
        page: Any,
        observation_text: Optional[str] = None
    ) -> bool:
        """Region presence check (stub returns True)."""
        
        
        return True
    
    def _check_modal_state(
        self,
        modal_config: Dict[str, Any],
        page: Any
    ) -> bool:
        """Modal state check (stub returns True)."""
        
        
        return True
    
    def _check_excluded_state(
        self,
        excluded_state: Dict[str, Any],
        page: Any
    ) -> bool:
        """True if URL matches excluded_state pattern."""
        url_pattern = excluded_state.get("url_pattern", "")
        if url_pattern:
            current_url = self._get_page_url(page)
            return self._match_url_pattern(current_url, url_pattern)
        return False

