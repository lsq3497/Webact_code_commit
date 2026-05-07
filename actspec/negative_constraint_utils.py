"""
Negative-constraint helpers: subtype constants, heuristics from traces,
and action_history_prefix fingerprint helpers.

Two conservative constraint families:
1. Readiness — acted before the page was stable; forbid π(a, tgt) in that unstable context.
2. Disambiguation — repeated wrong picks among similar targets; forbid certain descriptors in that context.
"""

from typing import Any, Dict, List, Optional

MAX_ACTION_HISTORY_PREFIX_LEN = 3


def _normalize_action_type_from_str(action: Any) -> Optional[str]:
    """Parse lowercase verb from an action string, e.g. \"click [79]\" -> \"click\"."""
    if not action:
        return None
    s = str(action).strip().lower()
    if not s:
        return None
    
    i = 0
    while i < len(s) and not s[i].isalpha():
        i += 1
    if i >= len(s):
        return None
    j = i
    while j < len(s) and s[j].isalpha():
        j += 1
    t = s[i:j]
    if not t:
        return None
    return t


def build_action_history_prefix(
    trajectory: List[Dict[str, Any]],
    segment_start: int,
    max_prefix_len: int = MAX_ACTION_HISTORY_PREFIX_LEN,
) -> List[str]:
    """
    Last N action-type tokens before segment_start (order only, no ids/text/urls).
    Empty when segment_start <= 1 (trace start).
    """
    if not trajectory or segment_start is None:
        return []
    
    if segment_start <= 1:
        return []

    types: List[str] = []
    
    for idx in range(segment_start - 1, -1, -1):
        step = trajectory[idx]
        if not isinstance(step, dict):
            continue
        action = step.get("action")
        t = _normalize_action_type_from_str(action)
        if not t:
            continue
        types.append(t)
        if len(types) >= max_prefix_len:
            break

    if not types:
        return []
    
    return list(reversed(types))



CONSTRAINT_SUBTYPE_READINESS = "readiness"
CONSTRAINT_SUBTYPE_DISAMBIGUATION = "disambiguation"
CONSTRAINT_SUBTYPE_UNSPECIFIED = "unspecified"

VALID_CONSTRAINT_SUBTYPES = frozenset({
    CONSTRAINT_SUBTYPE_READINESS,
    CONSTRAINT_SUBTYPE_DISAMBIGUATION,
    CONSTRAINT_SUBTYPE_UNSPECIFIED,
})


def normalize_constraint_subtype(value: Any) -> str:
    """Normalize arbitrary input to a valid constraint_subtype."""
    if value in VALID_CONSTRAINT_SUBTYPES:
        return value
    s = (value or "").strip().lower()
    if s in ("readiness", "ready"):
        return CONSTRAINT_SUBTYPE_READINESS
    if s in ("disambiguation", "disambiguate"):
        return CONSTRAINT_SUBTYPE_DISAMBIGUATION
    return CONSTRAINT_SUBTYPE_UNSPECIFIED


def infer_constraint_subtype_from_trajectory(
    trajectory: List[Dict[str, Any]],
    segment_start: int,
    segment_end: int,
    failure_reason: str,
    actions_in_segment: List[str],
) -> str:
    """
    Heuristic subtype when LLM leaves constraint unspecified.

    Args:
        trajectory: Full trace rows (url, action, observation, ...).
        segment_start / segment_end: Inclusive slice indices.
        failure_reason: Known failure text.
        actions_in_segment: Normalized action strings inside the segment.
    """
    if not trajectory or segment_start < 0 or segment_end >= len(trajectory):
        return CONSTRAINT_SUBTYPE_UNSPECIFIED

    reason_lower = (failure_reason or "").lower()
    actions_lower = [a.lower() for a in actions_in_segment if a]

    
    
    url_changed_in_or_before_segment = False
    prev_url = None
    for i in range(max(0, segment_start - 1), min(segment_end + 1, len(trajectory))):
        step = trajectory[i] if isinstance(trajectory[i], dict) else {}
        url = (step.get("url") or "").strip()
        if prev_url is not None and url and url != prev_url:
            url_changed_in_or_before_segment = True
            break
        if url and url != "about:blank":
            prev_url = url
    if segment_start > 0:
        prev_step = trajectory[segment_start - 1] if isinstance(trajectory[segment_start - 1], dict) else {}
        action_str = str(prev_step.get("action") or "").lower()
        if "goto" in action_str or "go_back" in action_str or "go_forward" in action_str:
            url_changed_in_or_before_segment = True

    has_click_or_type = any(
        "click" in a or "type" in a or "scroll" in a
        for a in actions_lower
    )
    readiness_keywords = (
        "not ready", "not loaded", "still loading", "page not", "transition", "refresh",
        "no_page_change", "target_not_interactable", "no effect", "no change",
    )
    reason_suggests_readiness = any(k in reason_lower for k in readiness_keywords)

    if url_changed_in_or_before_segment and has_click_or_type and reason_suggests_readiness:
        return CONSTRAINT_SUBTYPE_READINESS

    
    
    click_count = sum(1 for a in actions_lower if "click" in a)
    type_count = sum(1 for a in actions_lower if "type" in a)
    disambiguation_keywords = (
        "ambiguous", "wrong pick", "multiple similar", "similar target", "wrong target", "locate_multiple",
        "multiple candidates", "consistent incorrect", "repeated",
    )
    reason_suggests_disambiguation = any(k in reason_lower for k in disambiguation_keywords)

    if (click_count >= 2 or type_count >= 2) and reason_suggests_disambiguation:
        return CONSTRAINT_SUBTYPE_DISAMBIGUATION

    return CONSTRAINT_SUBTYPE_UNSPECIFIED


def build_unstable_state_for_readiness(
    trajectory: List[Dict[str, Any]],
    segment_start: int,
    segment_end: int,
) -> Dict[str, Any]:
    """Build context.unstable_state for readiness constraints when inferable; else {}."""
    if not trajectory or segment_start < 0:
        return {}
    
    prev_idx = segment_start - 1
    if prev_idx < 0:
        return {}
    prev = trajectory[prev_idx] if isinstance(trajectory[prev_idx], dict) else {}
    action_str = str(prev.get("action") or "").lower()
    if "goto" in action_str or "go_back" in action_str or "go_forward" in action_str:
        return {"type": "url_transition"}
    return {"type": "unstable"}  
