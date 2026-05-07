"""
元素上下文抽取器：从轨迹中抽取元素的上下文信息
"""

from typing import Dict, List, Any
from .accessibility_tree_parser import AccessibilityTreeParser


class ElementContextExtractor:
    """从轨迹中抽取元素的上下文信息"""
    
    def __init__(self):
        self.parser = AccessibilityTreeParser()
    
    def extract_element_context(
        self,
        trajectory: List[Dict[str, Any]],
        action_step_idx: int,
        element_id: str
    ) -> Dict[str, Any]:
        """
        抽取指定元素在指定步骤的上下文信息
        
        Args:
            trajectory: 完整轨迹
            action_step_idx: action执行的步骤索引
            element_id: 元素ID（字符串格式，如"79"）
        
        Returns:
            元素上下文字典
        """
        # 1. 获取action执行前的observation（字符串格式）
        if action_step_idx >= len(trajectory):
            return {}
        
        step = trajectory[action_step_idx]
        observation_text = step.get("observation", "")
        
        if not observation_text:
            return {}
        
        # 2. 解析Accessibility Tree
        tree = self.parser.parse(observation_text)
        
        # 3. 查找元素及其上下文
        element = self.parser.find_element_by_id(tree, element_id)
        if not element:
            return {}
        
        context = self.parser.get_element_context(tree, element_id)
        
        # 4. 提取语义特征
        semantic_features = self._extract_semantic_features(element)
        
        # 5. 提取相对上下文
        relative_context = self._extract_relative_context(context, tree)
        
        return {
            "element_id": element_id,
            "semantic_features": semantic_features,
            "relative_context": relative_context
        }
    
    def _extract_semantic_features(self, element: Dict) -> Dict[str, Any]:
        """提取语义特征"""
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
        """提取相对上下文"""
        relative_context = {
            "parent": None,
            "siblings": [],
            "region": context.get("region"),
            "form": None,
            "modal": None
        }
        
        # 提取父元素信息
        parent = context.get("parent")
        if parent:
            relative_context["parent"] = {
                "role": parent.get("role"),
                "element_id": parent.get("element_id"),
                "label": parent.get("label")
            }
        
        # 提取兄弟元素信息
        siblings = context.get("siblings", [])
        relative_context["siblings"] = [
            {
                "role": s.get("role"),
                "element_id": s.get("element_id"),
                "label": s.get("label")
            }
            for s in siblings
        ]
        
        # 判断是否在form中（通过祖先元素判断）
        # 改进：提取更详细的form信息
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
        
        # 判断是否在modal中（通过祖先元素判断，通常modal有特定role或label）
        # 改进：提取更详细的modal信息，包括类型和状态
        for ancestor in ancestors:
            role = ancestor.get("role", "").lower()
            label = ancestor.get("label", "").lower()
            if "dialog" in role or "modal" in role or "modal" in label:
                modal_type = "dialog" if "dialog" in role else "modal"
                # 检查是否是alertdialog
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

