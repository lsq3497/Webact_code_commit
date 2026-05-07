"""
Element context extractor: pull semantic / structural context around an element from a trace.
"""

from typing import Dict, List, Any
from .accessibility_tree_parser import AccessibilityTreeParser


class ElementContextExtractor:
    """Extract element-centric context from trajectory observations."""
    
    def __init__(self):
        self.parser = AccessibilityTreeParser()
    
    def extract_element_context(
        self,
        trajectory: List[Dict[str, Any]],
        action_step_idx: int,
        element_id: str
    ) -> Dict[str, Any]:
        """Context dict for element_id at action_step_idx using that step's observation."""
        
        if action_step_idx >= len(trajectory):
            return {}
        
        step = trajectory[action_step_idx]
        observation_text = step.get("observation", "")
        
        if not observation_text:
            return {}
        
        
        tree = self.parser.parse(observation_text)
        
        
        element = self.parser.find_element_by_id(tree, element_id)
        if not element:
            return {}
        
        context = self.parser.get_element_context(tree, element_id)
        
        
        semantic_features = self._extract_semantic_features(element)
        
        
        relative_context = self._extract_relative_context(context, tree)
        
        return {
            "element_id": element_id,
            "semantic_features": semantic_features,
            "relative_context": relative_context
        }
    
    def _extract_semantic_features(self, element: Dict) -> Dict[str, Any]:
        """Role/label/text/url snapshot."""
        return {
            "role": element.get("role"),
            "label": element.get("label"),
            "text": element.get("text"),
            "url": element.get("url")
        }
    
    def _extract_relative_context(
        self,
        context: Dict[str, Any],
        tree: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Parent/siblings/region/form/modal hints."""
        relative_context = {
            "parent": None,
            "siblings": [],
            "region": context.get("region"),
            "form": None,
            "modal": None
        }
        
        
        parent = context.get("parent")
        if parent:
            relative_context["parent"] = {
                "role": parent.get("role"),
                "element_id": parent.get("element_id"),
                "label": parent.get("label")
            }
        
        
        siblings = context.get("siblings", [])
        relative_context["siblings"] = [
            {
                "role": s.get("role"),
                "element_id": s.get("element_id"),
                "label": s.get("label")
            }
            for s in siblings
        ]
        
        
        
        ancestors = context.get("ancestors", [])
        for ancestor in ancestors:
            if ancestor.get("role") == "form":
                relative_context["form"] = {
                    "id": ancestor.get("element_id"),
                    "label": ancestor.get("label"),
                    "text": ancestor.get("text"),
                    "url": ancestor.get("url")
                }
                break
        
        
        
        for ancestor in ancestors:
            role = ancestor.get("role", "").lower()
            label = ancestor.get("label", "").lower()
            if "dialog" in role or "modal" in role or "modal" in label:
                modal_type = "dialog" if "dialog" in role else "modal"
                
                if "alert" in role:
                    modal_type = "alert"
                
                relative_context["modal"] = {
                    "id": ancestor.get("element_id"),
                    "label": ancestor.get("label"),
                    "text": ancestor.get("text"),
                    "type": modal_type,
                    "role": ancestor.get("role")
                }
                break
        
        return relative_context

