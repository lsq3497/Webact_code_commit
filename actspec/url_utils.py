"""
URL parsing helpers.
Shared between ActSpec generation and lookup so site/page derivation stays consistent both ways.
"""
from urllib.parse import urlparse
from typing import Tuple, Optional, List


def extract_site_and_page_from_url(
    url: str, 
    sites: Optional[List[str]] = None,
    include_port: bool = True
) -> Tuple[str, str]:
    """
    Extract site and page from a URL (single canonical path).

    Supports:
    1. Port in site: for ec2-3-129-7-246.us-east-2.compute.amazonaws.com:9999,
       if include_port=True, site may include the port (e.g. "amazonaws:9999").
    2. Path segment as page: for /f/singularity/69404/..., "singularity" is taken as page.
    3. Site from hostname: derive the registrable-style label from the hostname.

    Args:
        url: Current URL.
        sites: Optional list of known site keys to match.
        include_port: Whether to append :port to site (default True).

    Returns:
        (site, page) tuple.
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
    Normalize a site string for comparison.

    Examples:
    - "amazonaws:9999" -> "amazonaws"
    - "amazonaws" -> "amazonaws"

    Args:
        site: Site string.

    Returns:
        Normalized site without port suffix.
    """
    if ":" in site:
        return site.split(":")[0]
    return site


def sites_match(query_site: str, actspec_site: str, flexible: bool = True) -> bool:
    """
    Check whether two site strings refer to the same logical site.

    Flexible matching:
    - If flexible=True, partial match after normalization (e.g. "amazonaws" matches "amazonaws:9999").
    - If flexible=False, require exact string equality.

    Args:
        query_site: Site from the query context.
        actspec_site: Site stored on the ActSpec.
        flexible: Use normalized flexible matching (default True).

    Returns:
        True if they match.
    """
    if not query_site or not actspec_site:
        return False
    
    if flexible:
        
        query_normalized = normalize_site_for_matching(query_site)
        actspec_normalized = normalize_site_for_matching(actspec_site)
        return query_normalized == actspec_normalized
    else:
        
        return query_site == actspec_site
