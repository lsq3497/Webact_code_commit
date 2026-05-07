"""
Accessibility tree parser: parse a11y tree text into element records.
"""

import re
from typing import Dict, List, Any, Optional


class AccessibilityTreeParser:
    """Parse accessibility tree text into elements."""

    def parse(self, observation_text: str) -> Dict[str, Any]:
        """Build an element tree from accessibility tree text."""
        lines = observation_text.split('\n')
        elements = []
        stack = []  
        
        for line in lines:
            if not line.strip():
                continue
            
            
            indent_level = len(line) - len(line.lstrip('\t'))
            
            
            element = self._parse_line(line.strip())
            if not element:
                continue
            
            
            while len(stack) > indent_level:
                stack.pop()
            
            
            if stack:
                element['parent_id'] = stack[-1].get('element_id')
                if 'children' not in stack[-1]:
                    stack[-1]['children'] = []
                stack[-1]['children'].append(element)
            else:
                element['parent_id'] = None
            
            
            stack.append(element)
            elements.append(element)
        
        return {
            "elements": elements,
            "root": elements[0] if elements else None
        }
    
    def _parse_line(self, line: str) -> Optional[Dict[str, Any]]:
        """
        Parse one line of accessibility tree text.
        Examples:
        - link [79] 'Forums' [url: http://...]
        - searchbox [93] 'Search query'
        """
        
        pattern1 = r"^(\w+)\s+\[(\d+)\]\s+'([^']*)'\s*(?:\[url:\s*([^\]]+)\])?"
        match1 = re.match(pattern1, line)
        if match1:
            role, element_id, label, url = match1.groups()
            return {
                "role": role,
                "element_id": element_id,
                "label": label,
                "url": url,
                "text": label
            }
        
        
        pattern2 = r"^(\w+)\s+\[(\d+)\]\s+'([^']*)'"
        match2 = re.match(pattern2, line)
        if match2:
            role, element_id, label = match2.groups()
            return {
                "role": role,
                "element_id": element_id,
                "label": label,
                "text": label
            }
        
        
        pattern3 = r"^(\w+)\s+'([^']*)'"
        match3 = re.match(pattern3, line)
        if match3:
            role, label = match3.groups()
            return {
                "role": role,
                "element_id": None,
                "label": label,
                "text": label
            }
        
        
        pattern4 = r"^(\w+)$"
        match4 = re.match(pattern4, line)
        if match4:
            role = match4.group(1)
            return {
                "role": role,
                "element_id": None,
                "label": None,
                "text": None
            }
        
        return None
    
    def find_element_by_id(self, tree: Dict[str, Any], element_id: str) -> Optional[Dict[str, Any]]:
        """Find element by element_id."""
        for element in tree.get("elements", []):
            if element.get("element_id") == element_id:
                return element
        return None
    
    def get_element_context(self, tree: Dict[str, Any], element_id: str) -> Dict[str, Any]:
        """Parent, siblings, region context for an element."""
        element = self.find_element_by_id(tree, element_id)
        if not element:
            return {}
        
        context = {
            "element": element,
            "parent": None,
            "siblings": [],
            "ancestors": [],
            "region": None
        }
        
        
        if element.get("parent_id"):
            parent = self.find_element_by_id(tree, element["parent_id"])
            context["parent"] = parent
            
            
            current = parent
            while current:
                context["ancestors"].append(current)
                if current.get("parent_id"):
                    current = self.find_element_by_id(tree, current["parent_id"])
                else:
                    break
        
        
        if context["parent"]:
            siblings = context["parent"].get("children", [])
            context["siblings"] = [s for s in siblings if s.get("element_id") != element_id]
        
        
        for ancestor in context["ancestors"]:
            role = ancestor.get("role", "")
            if role in ["main", "complementary", "header", "footer", "navigation"]:
                context["region"] = role
                break
        
        return context

