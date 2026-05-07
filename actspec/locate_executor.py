"""
Locate执行器：执行多策略定位器，找到目标元素
"""

from typing import Dict, List, Any, Optional


class LocateExecutor:
    """执行多策略定位器，找到目标元素"""
    
    def locate_element(
        self,
        locate_config: Dict[str, Any],
        page: Any,  
        parameters: Dict[str, Any]
    ) -> Optional[Any]:
        """
        根据定位配置找到元素
        
        Args:
            locate_config: Locate配置
            page: Playwright Page对象
            parameters: 参数字典
        
        Returns:
            Playwright Locator对象，如果找不到则返回None
        """
        target_elements = locate_config.get("target_elements", [])
        
        for target_element in target_elements:
            strategies = target_element.get("strategies", [])
            
            
            strategies.sort(key=lambda x: x.get("priority", 999))
            
            
            for strategy in strategies:
                locator = self._try_strategy(strategy, page, parameters)
                if locator and self._check_locator_exists(locator):
                    return locator
        
        return None
    
    def _try_strategy(
        self,
        strategy: Dict[str, Any],
        page: Any,
        parameters: Dict[str, Any]
    ) -> Optional[Any]:
        """尝试单个定位策略"""
        strategy_type = strategy.get("strategy")
        conditions = strategy.get("conditions", {})
        
        if strategy_type == "semantic":
            return self._locate_by_semantic(conditions, page)
        elif strategy_type == "relative_position":
            return self._locate_by_relative_position(conditions, page, parameters)
        elif strategy_type == "element_id":
            return self._locate_by_element_id(conditions, page, parameters)
        else:
            return None
    
    def _locate_by_semantic(
        self,
        conditions: Dict[str, Any],
        page: Any
    ) -> Optional[Any]:
        """基于语义特征定位"""
        role = conditions.get("role")
        label = conditions.get("label")
        text = conditions.get("text")
        
        
        if role and (label or text):
            name = label or text
            try:
                return page.get_by_role(role=role, name=name)
            except Exception:
                pass
        
        
        if label:
            try:
                return page.get_by_label(label)
            except Exception:
                pass
        
        
        if text:
            try:
                return page.get_by_text(text)
            except Exception:
                pass
        
        return None
    
    def _locate_by_relative_position(
        self,
        conditions: Dict[str, Any],
        page: Any,
        parameters: Dict[str, Any]
    ) -> Optional[Any]:
        """
        基于相对位置定位
        
        实现逻辑：
        1. 找到参考元素（relative_to.element_id）
        2. 根据 position 和 distance 找到目标元素
        3. 返回目标元素的 locator
        """
        relative_to = conditions.get("relative_to", {})
        sibling_info = conditions.get("sibling_info", {})
        
        if not relative_to:
            return None
        
        
        reference_element_id = relative_to.get("element_id", "")
        position = relative_to.get("position", "after")  
        distance = relative_to.get("distance", "adjacent")  
        
        if not reference_element_id:
            return None
        
        try:
            
            
            reference_locator = None
            
            
            try:
                reference_locator = page.locator(f'[data-testid="{reference_element_id}"]')
                if not self._check_locator_exists(reference_locator):
                    reference_locator = None
            except Exception:
                pass
            
            
            if not reference_locator and sibling_info:
                role = sibling_info.get("role")
                label = sibling_info.get("label")
                if role and label:
                    try:
                        reference_locator = page.get_by_role(role=role, name=label)
                        if not self._check_locator_exists(reference_locator):
                            reference_locator = None
                    except Exception:
                        pass
            
            if not reference_locator:
                return None
            
            
            
            if position == "after":
                
                if distance == "adjacent":
                    
                    try:
                        
                        target_locator = reference_locator.locator("xpath=following-sibling::*[1]")
                        if self._check_locator_exists(target_locator):
                            return target_locator
                    except Exception:
                        pass
                    
                    
                    try:
                        
                        target_locator = reference_locator.locator("~ *").first
                        if self._check_locator_exists(target_locator):
                            return target_locator
                    except Exception:
                        pass
                else:
                    
                    try:
                        target_locator = reference_locator.locator("xpath=following-sibling::*")
                        if self._check_locator_exists(target_locator):
                            return target_locator.first
                    except Exception:
                        pass
            
            elif position == "before":
                
                if distance == "adjacent":
                    try:
                        target_locator = reference_locator.locator("xpath=preceding-sibling::*[1]")
                        if self._check_locator_exists(target_locator):
                            return target_locator
                    except Exception:
                        pass
            
            
            return None
            
        except Exception as e:
            
            return None
    
    def _locate_by_element_id(
        self,
        conditions: Dict[str, Any],
        page: Any,
        parameters: Dict[str, Any]
    ) -> Optional[Any]:
        """
        基于element_id定位
        
        改进：支持多种定位方式
        1. 尝试 data-testid 属性
        2. 尝试 id 属性
        3. 尝试通过 accessibility tree 查找（如果环境支持）
        """
        element_id = conditions.get("element_id", "")
        
        
        if element_id.startswith("{{") and element_id.endswith("}}"):
            param_name = element_id[2:-2]
            element_id = parameters.get(param_name, element_id)
        
        if not element_id:
            return None
        
        
        locators_to_try = [
            
            lambda: page.locator(f'[data-testid="{element_id}"]'),
            
            lambda: page.locator(f'#element_{element_id}'),
            
            
        ]
        
        for locator_func in locators_to_try:
            try:
                locator = locator_func()
                if locator and self._check_locator_exists(locator):
                    return locator
            except Exception:
                continue
        
        return None
    
    def _check_locator_exists(self, locator: Any) -> bool:
        """检查locator是否存在"""
        try:
            count = locator.count()
            return count > 0
        except Exception:
            return False

