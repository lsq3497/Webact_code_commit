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
        
        # 从sites列表中匹配site
        site = "unknown"
        if sites:
            for s in sites:
                if s.lower() in hostname.lower() or s.lower() in url.lower():
                    site = s
                    break
        
        # 如果没匹配到，尝试从hostname提取
        if site == "unknown":
            # 提取主域名部分（去掉www等前缀）
            hostname_parts = hostname.replace("www.", "").split(".")
            if len(hostname_parts) >= 2:
                # 对于 amazonaws.com 这样的域名，提取 amazonaws
                # 对于 postmill.xyz 这样的域名，提取 postmill
                main_domain = hostname_parts[-2]
                
                # 如果主域名是常见的通用域名，尝试提取更前面的部分
                if main_domain in ["com", "org", "net", "edu", "gov", "io", "co", "xyz"]:
                    if len(hostname_parts) >= 3:
                        # 例如：compute.amazonaws.com -> amazonaws
                        site = hostname_parts[-3]
                    else:
                        site = main_domain
                else:
                    site = main_domain
        
        # 如果include_port为True且存在端口号，将端口号添加到site
        if include_port and port:
            site = f"{site}:{port}"
        
        # 从路径推断page
        page = "unknown"
        if path:
            # 提取路径的非空部分
            path_parts = [p for p in path.split("/") if p]
            if path_parts:
                # 跳过单字符路径段（如 "f"），寻找有意义的路径段
                # 例如：/f/singularity/69404/... -> page = "singularity"
                for part in path_parts:
                    # 跳过单字符路径段和数字ID
                    if len(part) > 1 and not part.isdigit():
                        # 检查是否是UUID格式
                        if not (len(part) > 10 and all(c in "0123456789abcdef-" for c in part.lower())):
                            page = part
                            break
                
                # 如果所有路径段都是单字符或数字，使用第一个非空路径段
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
        # 灵活匹配：去除端口号后比较
        query_normalized = normalize_site_for_matching(query_site)
        actspec_normalized = normalize_site_for_matching(actspec_site)
        return query_normalized == actspec_normalized
    else:
        # 严格匹配
        return query_site == actspec_site

