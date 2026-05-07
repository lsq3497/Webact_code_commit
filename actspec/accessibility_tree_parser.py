"""
Accessibility Tree解析器：解析Accessibility Tree文本，提取元素信息
"""

import re
from typing import Dict, List, Any, Optional


class AccessibilityTreeParser:
    """解析Accessibility Tree文本，提取元素信息"""
    
    def parse(self, observation_text: str) -> Dict[str, Any]:
        """
        解析Accessibility Tree文本，构建元素树
        
        Args:
            observation_text: Accessibility Tree文本字符串
        
        Returns:
            元素树字典，包含所有元素及其层级关系
        """
        lines = observation_text.split('\n')
        elements = []
        stack = []  # 用于跟踪层级结构
        
        for line in lines:
            if not line.strip():
                continue
            
            # 计算缩进级别
            indent_level = len(line) - len(line.lstrip('\t'))
            
            # 解析元素
            element = self._parse_line(line.strip())
            if not element:
                continue
            
            # 调整stack以匹配当前缩进级别
            while len(stack) > indent_level:
                stack.pop()
            
            # 设置父元素
            if stack:
                element['parent_id'] = stack[-1].get('element_id')
                if 'children' not in stack[-1]:
                    stack[-1]['children'] = []
                stack[-1]['children'].append(element)
            else:
                element['parent_id'] = None
            
            # 添加到stack和elements列表
            stack.append(element)
            elements.append(element)
        
        return {
            "elements": elements,
            "root": elements[0] if elements else None
        }
    
    def _parse_line(self, line: str) -> Optional[Dict[str, Any]]:
        """
        解析单行Accessibility Tree文本
        
        格式示例：
        - link [79] 'Forums' [url: http://...]
        - searchbox [93] 'Search query'
        - button [113] 'MarvelsGrantMan136'
        - text 'Postmill'
        """
        # 匹配格式：role [id] 'label' [url: ...]
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
        
        # 匹配格式：role [id] 'label'（无URL）
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
        
        # 匹配格式：role 'label'（无ID，如text元素）
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
        
        # 匹配格式：role（无ID和label，如main、complementary等容器）
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
        """根据element_id查找元素"""
        for element in tree.get("elements", []):
            if element.get("element_id") == element_id:
                return element
        return None
    
    def get_element_context(self, tree: Dict[str, Any], element_id: str) -> Dict[str, Any]:
        """获取元素的上下文信息（父元素、兄弟元素、区域等）"""
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
        
        # 查找父元素
        if element.get("parent_id"):
            parent = self.find_element_by_id(tree, element["parent_id"])
            context["parent"] = parent
            
            # 查找祖先元素
            current = parent
            while current:
                context["ancestors"].append(current)
                if current.get("parent_id"):
                    current = self.find_element_by_id(tree, current["parent_id"])
                else:
                    break
        
        # 查找兄弟元素
        if context["parent"]:
            siblings = context["parent"].get("children", [])
            context["siblings"] = [s for s in siblings if s.get("element_id") != element_id]
        
        # 识别区域（main、complementary、header等）
        for ancestor in context["ancestors"]:
            role = ancestor.get("role", "")
            if role in ["main", "complementary", "header", "footer", "navigation"]:
                context["region"] = role
                break
        
        return context

