"""
ActSpec library: save, load, query specs and negative constraints.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from .negative_constraint_utils import (
    infer_constraint_subtype_from_trajectory,
    build_unstable_state_for_readiness,
    normalize_constraint_subtype,
)


class ActSpecLibrary:
    """Filesystem-backed ActSpec store."""

    def __init__(self, base_path: str = "temp_library"):
        """Create library root at base_path."""
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def save_actspec(
        self,
        actspec: Dict[str, Any],
        library_path: Optional[str] = None
    ) -> str:
        """Persist actspec JSON; optional explicit library_path or new timestamp dir."""
        
        if library_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            lib_path = self._get_library_path(self.base_path, timestamp)
        else:
            lib_path = Path(library_path)
        
        lib_path.mkdir(parents=True, exist_ok=True)
        
        
        action_id = actspec.get("action_id", "unknown")
        
        safe_filename = action_id.replace("/", "_").replace("\\", "_").replace(":", "_")
        filename = f"{safe_filename}.json"
        file_path = lib_path / filename
        
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(actspec, f, indent=2, ensure_ascii=False)
        
        
        self._update_index(actspec, lib_path, filename)
        
        return str(file_path)
    
    def load_library(self, library_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load non-disabled specs from index.json (latest timestamp dir if path omitted)."""
        if library_path is None:
            
            lib_path = self._get_latest_library_path()
        else:
            lib_path = Path(library_path)
        
        if not lib_path.exists():
            return []
        
        
        index_file = lib_path / "index.json"
        if not index_file.exists():
            return []
        
        with open(index_file, 'r', encoding='utf-8') as f:
            index = json.load(f)
        
        
        actspecs: List[Dict[str, Any]] = []
        for entry in index:
            file_path = lib_path / entry["file"]
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    actspec = json.load(f)
                    metadata = actspec.get("metadata", {})
                    
                    
                    if metadata.get("disabled", False):
                        continue
                    
                    actspecs.append(actspec)
        
        
        def _sort_key(spec):
            meta = spec.get("metadata", {})
            usage = int(meta.get("usage_count", 0) or 0)
            conf = float(meta.get("confidence", 1.0))
            
            return (-conf, usage)
        actspecs.sort(key=_sort_key)
        
        return actspecs
    
    def query_actspecs(
        self,
        context: Dict[str, Any],
        library_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Filter library entries by site/page/url_pattern/action_history_prefix."""
        if library_path is None:
            lib_path = self._get_latest_library_path()
        else:
            lib_path = Path(library_path)
        
        if not lib_path.exists():
            return []
        
        
        index_file = lib_path / "index.json"
        if not index_file.exists():
            return []
        
        with open(index_file, 'r', encoding='utf-8') as f:
            index = json.load(f)
        
        
        from .url_utils import sites_match
        
        matched_entries = []
        query_site = context.get("site", "")
        query_page = context.get("page", "")
        query_url_pattern = context.get("url_pattern", "")
        query_history_types: List[str] = context.get("action_history_types") or []
        if not isinstance(query_history_types, list):
            query_history_types = []
        
        for entry in index:
            
            
            
            actspec_site = entry.get("site", "")
            if query_site:
                if not sites_match(query_site, actspec_site, flexible=True):
                    continue
            
            
            
            actspec_page = entry.get("page", "")
            if query_page:
                
                if query_page != actspec_page and query_page not in actspec_page and actspec_page not in query_page:
                    continue
            
            matched_entries.append(entry)
        
        
        actspecs: List[Dict[str, Any]] = []
        for entry in matched_entries:
            file_path = lib_path / entry["file"]
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    actspec = json.load(f)
                    
                    
                    if query_url_pattern:
                        actspec_url_pattern = actspec.get("context", {}).get("url_pattern", "")
                        if actspec_url_pattern and query_url_pattern not in actspec_url_pattern:
                            continue
                    
                    
                    metadata = actspec.get("metadata", {})
                    if metadata.get("disabled", False):
                        continue

                    
                    
                    
                    prefix = actspec.get("action_history_prefix")
                    if isinstance(prefix, list) and prefix:
                        if not _history_prefix_matches(prefix, query_history_types):
                            continue

                    actspecs.append(actspec)
        
        
        def _sort_key(spec):
            meta = spec.get("metadata", {})
            usage = int(meta.get("usage_count", 0) or 0)
            conf = float(meta.get("confidence", 1.0))
            return (-conf, usage)
        actspecs.sort(key=_sort_key)
        
        return actspecs
    
    def _get_library_path(self, base_path: Path, timestamp: str) -> Path:
        """base_path / timestamp directory."""
        return base_path / timestamp
    
    def _get_latest_library_path(self) -> Path:
        """Newest %Y%m%d_%H%M%S child under base_path, else base_path."""
        if not self.base_path.exists():
            return self.base_path
        
        
        timestamp_dirs = []
        for item in self.base_path.iterdir():
            if item.is_dir():
                try:
                    
                    datetime.strptime(item.name, "%Y%m%d_%H%M%S")
                    timestamp_dirs.append(item)
                except ValueError:
                    
                    continue
        
        if not timestamp_dirs:
            return self.base_path
        
        
        timestamp_dirs.sort(key=lambda x: x.name, reverse=True)
        return timestamp_dirs[0]
    
    def _update_index(
        self,
        actspec: Dict[str, Any],
        library_path: Path,
        filename: str
    ) -> None:
        """Upsert index.json entry for saved actspec file."""
        index_file = library_path / "index.json"
        
        
        index = []
        if index_file.exists():
            with open(index_file, 'r', encoding='utf-8') as f:
                index = json.load(f)
        
        
        action_id = actspec.get("action_id", "")
        existing_idx = None
        for i, entry in enumerate(index):
            if entry.get("action_id") == action_id:
                existing_idx = i
                break
        
        
        context = actspec.get("context", {})
        new_entry = {
            "action_id": action_id,
            "site": context.get("site", "unknown"),
            "page": context.get("page", "unknown"),
            "file": filename
        }
        
        
        if existing_idx is not None:
            index[existing_idx] = new_entry
        else:
            index.append(new_entry)
        
        
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2, ensure_ascii=False)
    
    def load_negative_constraints(self, library_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load all negative constraint JSON files referenced by negative_constraints_index.json."""
        if library_path is None:
            
            lib_path = self._get_latest_library_path()
        else:
            lib_path = Path(library_path)
        
        if not lib_path.exists():
            return []
        
        
        index_file = lib_path / "negative_constraints_index.json"
        if not index_file.exists():
            return []
        
        with open(index_file, 'r', encoding='utf-8') as f:
            index = json.load(f)
        
        
        constraints = []
        for entry in index:
            file_path = lib_path / entry["file"]
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    constraint = json.load(f)
                    constraints.append(constraint)
        
        return constraints
    
    def save_negative_constraint(
        self,
        constraint: Dict[str, Any],
        library_path: Optional[str] = None
    ) -> str:
        """Write one negative constraint under negative_constraints/ and update index."""
        
        if library_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            lib_path = self._get_library_path(self.base_path, timestamp)
        else:
            lib_path = Path(library_path)
        
        lib_path.mkdir(parents=True, exist_ok=True)
        
        
        negative_dir = lib_path / "negative_constraints"
        negative_dir.mkdir(parents=True, exist_ok=True)
        
        
        constraint_id = constraint.get("constraint_id", "unknown")
        
        safe_filename = constraint_id.replace("/", "_").replace("\\", "_").replace(":", "_")
        filename = f"{safe_filename}.json"
        file_path = negative_dir / filename
        
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(constraint, f, indent=2, ensure_ascii=False)
        
        
        self._update_negative_constraints_index(constraint, lib_path, f"negative_constraints/{filename}")
        
        return str(file_path)
    
    def _update_negative_constraints_index(
        self,
        constraint: Dict[str, Any],
        library_path: Path,
        filename: str
    ) -> None:
        """Upsert negative_constraints_index.json."""
        index_file = library_path / "negative_constraints_index.json"
        
        
        index = []
        if index_file.exists():
            with open(index_file, 'r', encoding='utf-8') as f:
                index = json.load(f)
        
        
        constraint_id = constraint.get("constraint_id", "")
        existing_idx = None
        for i, entry in enumerate(index):
            if entry.get("constraint_id") == constraint_id:
                existing_idx = i
                break
        
        
        context = constraint.get("context", {})
        new_entry = {
            "constraint_id": constraint_id,
            "site": context.get("site", "unknown"),
            "page": context.get("page", "unknown"),
            "constraint_subtype": constraint.get("constraint_subtype", "unspecified"),
            "file": filename
        }
        
        
        if existing_idx is not None:
            index[existing_idx] = new_entry
        else:
            index.append(new_entry)
        
        
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2, ensure_ascii=False)

    

    def _resolve_library_path(self, library_path: Optional[str] = None) -> Path:
        """Use explicit library_path or fall back to latest timestamp directory."""
        if library_path:
            lib_path = Path(library_path)
        else:
            lib_path = self._get_latest_library_path()
        return lib_path

    def _load_index(
        self,
        lib_path: Path,
    ) -> Tuple[Path, list]:
        """Read index.json or empty list."""
        index_file = lib_path / "index.json"
        if not index_file.exists():
            return index_file, []
        with open(index_file, "r", encoding="utf-8") as f:
            index = json.load(f)
        return index_file, index

    def update_stats_batch(
        self,
        stats: Dict[str, Dict[str, int]],
        library_path: Optional[str] = None,
        convert_to_negative_constraints: bool = True,
    ) -> None:
        """
        Apply incremental usage/success/fail counters per action_id; optional auto negative-constraint conversion.
        """
        if not stats:
            return

        lib_path = self._resolve_library_path(library_path)
        if not lib_path.exists():
            return

        index_file, index = self._load_index(lib_path)
        if not index:
            return

        
        action_to_file: Dict[str, str] = {}
        for entry in index:
            aid = entry.get("action_id")
            fpath = entry.get("file")
            if aid and fpath:
                action_to_file[aid] = fpath

        for action_id, delta in stats.items():
            if action_id not in action_to_file:
                continue
            file_path = lib_path / action_to_file[action_id]
            if not file_path.exists():
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    actspec = json.load(f)
            except Exception:
                continue

            metadata = actspec.get("metadata", {}) or {}
            usage_prev = int(metadata.get("usage_count", 0) or 0)
            success_prev = int(metadata.get("success_count", 0) or 0)
            fail_prev = int(metadata.get("fail_count", 0) or 0)

            usage_new = int(delta.get("usage_count", 0) or 0)
            success_new = int(delta.get("success_count", 0) or 0)
            fail_new = int(delta.get("fail_count", 0) or 0)

            usage = usage_prev + usage_new
            success = success_prev + success_new
            fail = fail_prev + fail_new

            
            fail_rate = (fail / usage) if usage > 0 else 0.0

            disabled_prev = bool(metadata.get("disabled", False))

            metadata["usage_count"] = usage
            metadata["success_count"] = success
            metadata["fail_count"] = fail
            metadata["fail_rate"] = round(fail_rate, 4)

            
            
            
            disabled = metadata.get("disabled", False)
            if usage < 5 and fail >= 3:
                disabled = True
            elif usage >= 5 and fail_rate >= 0.4:
                disabled = True
            metadata["disabled"] = disabled

            
            
            if (
                convert_to_negative_constraints
                and (not disabled_prev)
                and disabled
                and not metadata.get("converted_to_negative_constraint")
            ):
                try:
                    constraint = _build_negative_constraint_from_actspec_struct(actspec)
                    if constraint:
                        
                        self.save_negative_constraint(constraint, str(lib_path))
                        metadata["converted_to_negative_constraint"] = True
                        print(f"[NegativeConstraint] ActSpec {actspec.get('action_id', 'unknown')} disabled after failures; "
                              f"converted to negative constraint {constraint.get('constraint_id', 'unknown')}")
                except Exception as e:
                    print(f"[NegativeConstraint] Failed to convert ActSpec to negative constraint ({actspec.get('action_id', 'unknown')}): {e}")

            actspec["metadata"] = metadata

            
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(actspec, f, indent=2, ensure_ascii=False)
            except Exception:
                
                continue

    def update_confidence(
        self,
        action_id: str,
        post_success: bool,
        s_max: float = 10.0,
        lambda_penalty: float = 2.0,
        library_path: Optional[str] = None,
    ) -> Optional[float]:
        """
        Bump confidence from post-condition outcome (ranking only; disabling uses update_stats_batch).
        """
        lib_path = self._resolve_library_path(library_path)
        if not lib_path.exists():
            return None

        index_file, index = self._load_index(lib_path)
        if not index:
            return None

        
        file_path = None
        for entry in index:
            if entry.get("action_id") == action_id:
                file_path = lib_path / entry.get("file", "")
                break

        if not file_path or not file_path.exists():
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                actspec = json.load(f)
        except Exception:
            return None

        metadata = actspec.get("metadata", {}) or {}
        current_confidence = float(metadata.get("confidence", 1.0))

        
        if post_success:
            
            new_confidence = current_confidence + 1.0
        else:
            
            new_confidence = current_confidence - lambda_penalty

        
        new_confidence = min(new_confidence, s_max)

        
        new_confidence = max(new_confidence, 0.0)

        
        metadata["confidence"] = round(new_confidence, 2)
        if new_confidence <= 0.0:
            metadata["disabled"] = True

        actspec["metadata"] = metadata

        
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(actspec, f, indent=2, ensure_ascii=False)
            return new_confidence
        except Exception:
            return None

    def update_confidence_batch(
        self,
        updates: Dict[str, bool],
        s_max: float = 10.0,
        lambda_penalty: float = 2.0,
        library_path: Optional[str] = None,
    ) -> Dict[str, float]:
        """Call update_confidence for many action_ids; returns {id: new_score}."""
        results = {}
        for action_id, post_success in updates.items():
            new_confidence = self.update_confidence(
                action_id=action_id,
                post_success=post_success,
                s_max=s_max,
                lambda_penalty=lambda_penalty,
                library_path=library_path,
            )
            if new_confidence is not None:
                results[action_id] = new_confidence
        return results

    def add_failed_actspec_as_negative_constraint(
        self,
        actspec: Dict[str, Any],
        library_path: Optional[str] = None,
        trajectory: Optional[List[Dict[str, Any]]] = None,
        failure_reason: str = "",
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Persist a negative constraint from a failed runtime ActSpec."""
        if not isinstance(actspec, dict):
            return None
        constraint = _build_negative_constraint_from_failed_trace_struct(
            actspec=actspec,
            trajectory=trajectory or [],
            failure_reason=failure_reason,
            runtime_context=runtime_context or {},
        )
        if not constraint:
            constraint = _build_negative_constraint_from_actspec_struct(actspec)
        if not constraint:
            return None
        try:
            return self.save_negative_constraint(constraint, library_path=library_path)
        except Exception:
            return None



def _history_prefix_matches(
    prefix: List[str],
    history: List[str],
) -> bool:
    """True if prefix occurs as a contiguous subsequence in history (lowercase verbs)."""
    if not prefix:
        return True
    if not history:
        return False
    n, m = len(history), len(prefix)
    if n < m:
        return False
    
    for start in range(0, n - m + 1):
        if history[start:start + m] == prefix:
            return True
    return False


def _actspec_id_to_constraint_id(action_id: str) -> str:
    """Map action_id to constraint_id (unknown.* -> constraint.*, else prefix constraint.)."""
    if not action_id:
        return "constraint.unknown"
    if action_id.startswith("unknown."):
        return action_id.replace("unknown.", "constraint.", 1)
    if not action_id.startswith("constraint."):
        return f"constraint.{action_id}"
    return action_id


def _build_negative_constraint_from_actspec_struct(
    actspec: Dict[str, Any],
) -> Dict[str, Any]:
    """Conservative struct-only conversion when no failure trace (subtype unspecified)."""
    action_id = actspec.get("action_id", "")
    context = actspec.get("context", {}) or {}
    pre_condition = actspec.get("pre_condition", {}) or {}
    forbidden_plan = actspec.get("plan", []) or []
    action_history_prefix = actspec.get("action_history_prefix", []) or []

    constraint_id = _actspec_id_to_constraint_id(action_id)
    failure_reason = actspec.get("failure_reason") or "ActSpec disabled after repeated failures; auto negative constraint."

    constraint = {
        "constraint_id": constraint_id,
        "type": "negative_constraint",
        "constraint_subtype": "unspecified",
        "context": context,
        "pre_condition": pre_condition,
        "action_history_prefix": action_history_prefix,
        "forbidden_plan": forbidden_plan,
        "description": {
            "failure_reason": failure_reason,
            "llm_judgment": {
                "is_failed": True,
                "failure_reason": failure_reason,
                "original_description": actspec.get("description", {}),
            },
        },
        "metadata": {
            "source": "online_adaptation",
            "generated_by": "actspec_runtime_stats",
            "original_action_id": action_id,
        },
    }
    return constraint


def _build_negative_constraint_from_failed_trace_struct(
    actspec: Dict[str, Any],
    trajectory: List[Dict[str, Any]],
    failure_reason: str,
    runtime_context: Dict[str, Any],
) -> Dict[str, Any]:
    """Build negative constraint from failed trace signals (Algorithm 3 style)."""
    action_id = actspec.get("action_id", "")
    context = dict(actspec.get("context", {}) or {})
    context.update(runtime_context or {})
    pre_condition = actspec.get("pre_condition", {}) or {}
    forbidden_plan = actspec.get("plan", []) or []
    action_history_prefix = actspec.get("action_history_prefix", []) or []

    segment_end = max(0, len(trajectory) - 1)
    window = max(1, len(forbidden_plan) * 2)
    segment_start = max(0, segment_end - window + 1)
    actions_in_segment = [
        str((step or {}).get("action", ""))
        for step in trajectory[segment_start:segment_end + 1]
        if isinstance(step, dict) and step.get("action") is not None
    ]
    constraint_subtype = normalize_constraint_subtype(
        infer_constraint_subtype_from_trajectory(
            trajectory=trajectory,
            segment_start=segment_start,
            segment_end=segment_end,
            failure_reason=failure_reason or "",
            actions_in_segment=actions_in_segment,
        )
    )
    if constraint_subtype == "readiness":
        context["unstable_state"] = build_unstable_state_for_readiness(
            trajectory=trajectory,
            segment_start=segment_start,
            segment_end=segment_end,
        )

    constraint_id = _actspec_id_to_constraint_id(action_id)
    reason = failure_reason or "failed trace indicates this routine is infeasible in current context"
    return {
        "constraint_id": constraint_id,
        "type": "negative_constraint",
        "constraint_subtype": constraint_subtype,
        "phi": {
            "delta_context": {
                "site": context.get("site"),
                "page": context.get("page"),
                "url": context.get("url"),
                "unstable_state": context.get("unstable_state"),
            }
        },
        "pi": {
            "forbidden_plan_pattern": forbidden_plan,
            "action_history_prefix": action_history_prefix,
        },
        "context": context,
        "pre_condition": pre_condition,
        "action_history_prefix": action_history_prefix,
        "forbidden_plan": forbidden_plan,
        "description": {
            "failure_reason": reason,
            "derived_from": "failed_trace",
            "phi_delta": {
                "segment_start": segment_start,
                "segment_end": segment_end,
                "subtype": constraint_subtype,
            },
        },
        "metadata": {
            "source": "online_adaptation",
            "generated_by": "failed_trace_extractor",
            "original_action_id": action_id,
        },
    }


class NegativeConstraintFilter:
    """Drop forbidden plans / specs during planning or primitive execution."""

    def __init__(self, constraints: List[Dict[str, Any]]):
        """constraints: loaded negative constraint dicts."""
        self.constraints = constraints

    def _constraint_applies(self, constraint: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """
        Context match, optional pre_condition on env, then optional action_history_prefix subsequence match.
        """
        if not self._context_matches(
            constraint.get("context", {}), context, constraint.get("constraint_subtype", "unspecified")
        ):
            return False

        
        
        pre_condition = constraint.get("pre_condition") or {}
        if pre_condition:
            try:
                from actspec.pre_condition_checker import PreConditionChecker
                pre_checker = PreConditionChecker()
                page_or_env = context.get("pre_condition_page")
                observation_text = context.get("pre_condition_observation_text")
                is_satisfied, reason = pre_checker.check_pre_condition(
                    pre_condition, page_or_env, {}, observation_text=observation_text
                )
                if not is_satisfied:
                    
                    return False
            except Exception as e:
                
                constraint_id = constraint.get("constraint_id", "unknown")
                print(f"[NegativeConstraint] pre-condition check error (constraint_id={constraint_id}): {e}; keeping constraint for matching")

        
        prefix = constraint.get("action_history_prefix")
        if not isinstance(prefix, list) or not prefix:
            
            return True

        runtime_history: List[str] = context.get("action_history_types") or []
        if not isinstance(runtime_history, list):
            runtime_history = []
        return _history_prefix_matches(prefix, runtime_history)
    
    def is_forbidden(
        self,
        plan: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> bool:
        """True if any applying constraint forbids this exact plan prefix."""
        for constraint in self.constraints:
            if self._constraint_applies(constraint, context):
                if self._plan_matches(constraint.get("forbidden_plan", []), plan):
                    return True
        return False
    
    def is_actspec_forbidden(
        self,
        actspec: Dict[str, Any],
        context: Dict[str, Any]
    ) -> bool:
        """True if this ActSpec's plan is forbidden."""
        plan = actspec.get("plan", [])
        return self.is_forbidden(plan, context)
    
    def filter_actspecs(
        self,
        actspecs: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Remove specs whose plans hit a negative constraint."""
        filtered = []
        for actspec in actspecs:
            if not self.is_actspec_forbidden(actspec, context):
                filtered.append(actspec)
            else:
                constraint_id = self._find_matching_constraint_id(actspec, context)
                print(f"[NegativeConstraint] ActSpec {actspec.get('action_id')} blocked (constraint: {constraint_id})")
        return filtered
    
    def _context_matches(
        self,
        constraint_context: Dict[str, Any],
        current_context: Dict[str, Any],
        constraint_subtype: str = "",
    ) -> bool:
        """Site/page match; readiness with unstable_state requires matching unstable signal."""
        
        constraint_site = constraint_context.get("site", "unknown")
        constraint_page = constraint_context.get("page", "unknown")
        current_site = current_context.get("site", "unknown")
        current_page = current_context.get("page", "unknown")
        
        
        if constraint_site != "unknown" and constraint_site != current_site:
            return False
        
        
        if constraint_page != "unknown" and constraint_page != current_page:
            return False
        
        
        
        if constraint_subtype == "readiness" and constraint_context.get("unstable_state"):
            if "unstable_state" in current_context and not current_context.get("unstable_state"):
                return False
        
        return True
    
    def _plan_matches(
        self,
        forbidden_plan: List[Dict[str, Any]],
        plan: List[Dict[str, Any]]
    ) -> bool:
        """Prefix match against forbidden_plan; placeholder-only first steps do not apply at plan time."""
        if len(plan) < len(forbidden_plan):
            return False
        if not forbidden_plan:
            return False

        
        first_forbidden = forbidden_plan[0]
        first_primitive = (first_forbidden.get("primitive") or "").upper()
        if first_primitive in ("CLICK", "TYPE", "HOVER", "UNKNOWN"):
            target = first_forbidden.get("target") or {}
            if target.get("strategy") == "element_id":
                val = target.get("value")
                if self._is_parameter_placeholder(val):
                    return False

        for i, forbidden_step in enumerate(forbidden_plan):
            if i >= len(plan):
                return False
            if not self._plan_step_matches(forbidden_step, plan[i]):
                return False
        return True
    
    def _find_matching_constraint_id(
        self,
        actspec: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """constraint_id that blocked this actspec, else unknown."""
        plan = actspec.get("plan", [])
        for constraint in self.constraints:
            if self._constraint_applies(constraint, context):
                if self._plan_matches(constraint.get("forbidden_plan", []), plan):
                    return constraint.get("constraint_id", "unknown")
        return "unknown"

    def get_forbidden_element_ids_for_observation(self, context: Dict[str, Any]) -> Set[str]:
        """Element ids to prune from observation when constraints apply (concrete ids only)."""
        forbidden_ids: Set[str] = set()
        for constraint in self.constraints:
            if not self._constraint_applies(constraint, context):
                continue
            for step in constraint.get("forbidden_plan", []):
                target = step.get("target") or {}
                if target.get("strategy") != "element_id":
                    continue
                val = target.get("value")
                if not val or self._is_parameter_placeholder(val):
                    continue
                forbidden_ids.add(str(val))
        return forbidden_ids
    
    def is_primitive_action_forbidden(
        self,
        action_cmd: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """(forbidden, info) for a single primitive action_cmd."""
        
        plan_step = self._action_cmd_to_plan_step(action_cmd)
        if not plan_step:
            
            return (False, None)
        
        
        for constraint in self.constraints:
            if not self._constraint_applies(constraint, context):
                continue
            
            
            forbidden_plan = constraint.get("forbidden_plan", [])
            if not forbidden_plan:
                continue
            
            
            first_step = forbidden_plan[0]
            if self._plan_step_matches(first_step, plan_step):
                
                constraint_info = {
                    "constraint_id": constraint.get("constraint_id", "unknown"),
                    "failure_reason": constraint.get("description", {}).get("failure_reason", "Action forbidden by negative constraint"),
                    "description": constraint.get("description", {})
                }
                return (True, constraint_info)
        
        return (False, None)
    
    def _action_cmd_to_plan_step(self, action_cmd: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Map browser_env action_cmd dict to plan step shape."""
        
        
        
        
        action_type_to_primitive = {
            1: "SCROLL",      
            6: "CLICK",       
            7: "TYPE",        
            8: "HOVER",       
            9: "GOTO",        
            10: "GOTO",       
            11: "GOTO",       
            12: "GOBACK",     
            13: "GOFORWARD",  
            14: "STOP",       
        }
        
        action_type = action_cmd.get("action_type")
        if action_type not in action_type_to_primitive:
            return None
        
        primitive = action_type_to_primitive[action_type]
        plan_step = {"primitive": primitive}
        
        
        if primitive == "CLICK" or primitive == "TYPE" or primitive == "HOVER":
            element_id = action_cmd.get("element_id")
            if element_id:
                plan_step["target"] = {
                    "strategy": "element_id",
                    "value": str(element_id)
                }
        
        if primitive == "TYPE":
            text = action_cmd.get("text", "")
            if isinstance(text, list):
                
                text = "".join([chr(c) if isinstance(c, int) and 32 <= c < 128 else str(c) for c in text])
            plan_step["text"] = text
        
        if primitive == "SCROLL":
            direction = action_cmd.get("direction", "down")
            plan_step["direction"] = direction
        
        if primitive == "GOTO":
            url = action_cmd.get("url", "")
            plan_step["url"] = url
        
        return plan_step
    
    def _plan_step_matches(
        self,
        forbidden_step: Dict[str, Any],
        plan_step: Dict[str, Any]
    ) -> bool:
        """Structural equality for forbidden vs candidate step."""
        
        forbidden_primitive = forbidden_step.get("primitive", "").upper()
        plan_primitive = plan_step.get("primitive", "").upper()
        
        if forbidden_primitive != plan_primitive:
            return False
        
        
        if forbidden_primitive in ["NOTE", "PRUNE"]:
            
            if plan_primitive != forbidden_primitive:
                return False
            
            forbidden_raw = forbidden_step.get("raw", "")
            plan_raw = plan_step.get("raw", "")
            if forbidden_raw and plan_raw:
                
                if forbidden_raw != plan_raw:
                    return False
            return True
        
        
        
        if forbidden_primitive == "UNKNOWN":
            
            forbidden_target = forbidden_step.get("target", {})
            if forbidden_target.get("strategy") == "element_id":
                
                forbidden_element_id = forbidden_target.get("value", "")
                plan_element_id = plan_step.get("target", {}).get("value", "")
                
                
                if self._is_parameter_placeholder(forbidden_element_id):
                    return False
                
                
                if forbidden_element_id and plan_element_id != forbidden_element_id:
                    return False
                return True
            else:
                
                return False
        
        
        if forbidden_primitive in ["CLICK", "TYPE", "HOVER"]:
            forbidden_target = forbidden_step.get("target", {})
            plan_target = plan_step.get("target", {})
            
            
            if forbidden_target:
                
                forbidden_strategy = forbidden_target.get("strategy", "")
                plan_strategy = plan_target.get("strategy", "")
                if forbidden_strategy and forbidden_strategy != plan_strategy:
                    return False
                
                
                if forbidden_strategy == "element_id":
                    forbidden_element_id = forbidden_target.get("value", "")
                    plan_element_id = plan_target.get("value", "")
                    
                    
                    if self._is_parameter_placeholder(forbidden_element_id):
                        return False
                    else:
                        
                        if forbidden_element_id and plan_element_id != forbidden_element_id:
                            return False
            
            
            if forbidden_primitive == "TYPE":
                forbidden_text = forbidden_step.get("text", "")
                plan_text = plan_step.get("text", "")
                
                
                if forbidden_text:
                    
                    if self._is_parameter_placeholder(forbidden_text):
                        pass  
                    else:
                        
                        if forbidden_text != plan_text:
                            return False
        
        
        if forbidden_primitive == "SCROLL":
            forbidden_direction = forbidden_step.get("direction", "")
            plan_direction = plan_step.get("direction", "")
            
            
            if forbidden_direction and forbidden_direction != plan_direction:
                return False
        
        
        if forbidden_primitive == "GOTO":
            forbidden_url = forbidden_step.get("url", "")
            plan_url = plan_step.get("url", "")
            
            
            if forbidden_url:
                
                if self._is_parameter_placeholder(forbidden_url):
                    pass  
                else:
                    
                    if forbidden_url != plan_url:
                        return False
        
        return True
    
    def _is_parameter_placeholder(self, value: str) -> bool:
        """True for {{param}} placeholders."""
        return isinstance(value, str) and value.startswith("{{") and value.endswith("}}")
