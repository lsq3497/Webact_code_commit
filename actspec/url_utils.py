"""
URL解析工具函数
用于统一ActSpec生成和查询阶段的URL解析逻辑，确保双向推导一致性
"""
from urllib.parse import urlparse
from typing import Tuple, Optional, List


def extract_site_and_page_from_url(
    url: str, 
    sites: Optional[List[str]] = None,
    include_port: bool = True
) -> Tuple[str, str]:
    """
    从URL提取site和page信息（统一方法）
    
    支持：
    1. 端口号提取：对于 ec2-3-129-7-246.us-east-2.compute.amazonaws.com:9999，
       如果include_port=True，site会包含端口信息（如 "amazonaws:9999"）
    2. 路径后缀提取：对于 /f/singularity/69404/...，会提取 "singularity" 作为 page
    3. 智能site识别：支持从hostname提取主域名
    
    Args:
        url: 当前URL
        sites: 可用的site列表（可选）
        include_port: 是否在site中包含端口号（默认True）
    
    Returns:
        (site, page) 元组
    """
    if not url:
        return ("unknown", "unknown")
    
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        port = parsed.port
        path = parsed.path or ""
        
        
        site = "unknown"
        if sites:
            for s in sites:
                if s.lower() in hostname.lower() or s.lower() in url.lower():
                    site = s
                    break
        
        
        if site == "unknown":
            
            hostname_parts = hostname.replace("www.", "").split(".")
            if len(hostname_parts) >= 2:
                
                
                main_domain = hostname_parts[-2]
                
                
                if main_domain in ["com", "org", "net", "edu", "gov", "io", "co", "xyz"]:
                    if len(hostname_parts) >= 3:
                        
                        site = hostname_parts[-3]
                    else:
                        site = main_domain
                else:
                    site = main_domain
        
        
        if include_port and port:
            site = f"{site}:{port}"
        
        
        page = "unknown"
        if path:
            
            path_parts = [p for p in path.split("/") if p]
            if path_parts:
                
                
                for part in path_parts:
                    
                    if len(part) > 1 and not part.isdigit():
                        
                        if not (len(part) > 10 and all(c in "0123456789abcdef-" for c in part.lower())):
                            page = part
                            break
                
                
                if page == "unknown" and path_parts:
                    page = path_parts[0]
            else:
                page = "home"
        else:
            page = "home"
        
        return (site, page)
    except Exception as e:
        print(f"[Warning] Failed to extract site/page from URL {url}: {e}")
        return ("unknown", "unknown")


def normalize_site_for_matching(site: str) -> str:
    """
    标准化site字符串用于匹配
    
    例如：
    - "amazonaws:9999" -> "amazonaws"
    - "amazonaws" -> "amazonaws"
    
    Args:
        site: site字符串
    
    Returns:
        标准化后的site（去除端口号）
    """
    if ":" in site:
        return site.split(":")[0]
    return site


def sites_match(query_site: str, actspec_site: str, flexible: bool = True) -> bool:
    """
    检查两个site是否匹配
    
    支持灵活匹配：
    - 如果flexible=True，支持部分匹配（如 "amazonaws" 匹配 "amazonaws:9999"）
    - 如果flexible=False，严格相等匹配
    
    Args:
        query_site: 查询的site
        actspec_site: ActSpec的site
        flexible: 是否使用灵活匹配（默认True）
    
    Returns:
        是否匹配
    """
    if not query_site or not actspec_site:
        return False
    
    if flexible:
        
        query_normalized = normalize_site_for_matching(query_site)
        actspec_normalized = normalize_site_for_matching(actspec_site)
        return query_normalized == actspec_normalized
    else:
        
        return query_site == actspec_site

