"""
ActSpec库管理器：保存、加载、查询ActSpec
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
    """ActSpec库管理器"""
    
    def __init__(self, base_path: str = "temp_library"):
        """
        初始化ActSpec库管理器
        
        Args:
            base_path: 库的基础路径
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def save_actspec(
        self,
        actspec: Dict[str, Any],
        library_path: Optional[str] = None
    ) -> str:
        """
        保存ActSpec到库
        
        Args:
            actspec: ActSpec字典
            library_path: 库路径，如果为None则使用时间戳目录
        
        Returns:
            保存的文件路径
        """
        # 获取库路径
        if library_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            lib_path = self._get_library_path(self.base_path, timestamp)
        else:
            lib_path = Path(library_path)
        
        lib_path.mkdir(parents=True, exist_ok=True)
        
        # 获取action_id作为文件名
        action_id = actspec.get("action_id", "unknown")
        # 清理action_id，使其适合作为文件名
        safe_filename = action_id.replace("/", "_").replace("\\", "_").replace(":", "_")
        filename = f"{safe_filename}.json"
        file_path = lib_path / filename
        
        # 保存ActSpec
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(actspec, f, indent=2, ensure_ascii=False)
        
        # 更新index.json
        self._update_index(actspec, lib_path, filename)
        
        return str(file_path)
    
    def load_library(self, library_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        加载整个库
        
        Args:
            library_path: 库路径，如果为None则加载最新的时间戳目录
        
        Returns:
            ActSpec列表
        """
        if library_path is None:
            # 查找最新的时间戳目录
            lib_path = self._get_latest_library_path()
        else:
            lib_path = Path(library_path)
        
        if not lib_path.exists():
            return []
        
        # 加载index.json
        index_file = lib_path / "index.json"
        if not index_file.exists():
            return []
        
        with open(index_file, 'r', encoding='utf-8') as f:
            index = json.load(f)
        
        # 加载所有ActSpec文件
        actspecs: List[Dict[str, Any]] = []
        for entry in index:
            file_path = lib_path / entry["file"]
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    actspec = json.load(f)
                    metadata = actspec.get("metadata", {})
                    
                    # 过滤已被禁用的ActSpec（淘汰仅由统计规则 update_stats_batch 触发）
                    if metadata.get("disabled", False):
                        continue
                    
                    actspecs.append(actspec)
        
        # 排序：从未被调用过的(usage_count==0) 最高优先级，其次按置信度降序
        def _sort_key(spec):
            meta = spec.get("metadata", {})
            usage = int(meta.get("usage_count", 0) or 0)
            conf = float(meta.get("confidence", 1.0))
            # 未使用过的排最前(0)，再按置信度降序(-conf)
            return (-conf, usage)
        actspecs.sort(key=_sort_key)
        
        return actspecs
    
    def query_actspecs(
        self,
        context: Dict[str, Any],
        library_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        根据上下文查询匹配的ActSpec
        
        Args:
            context: 上下文信息（site, page, url_pattern等）
            library_path: 库路径，如果为None则使用最新的时间戳目录
        
        Returns:
            匹配的ActSpec列表
        """
        if library_path is None:
            lib_path = self._get_latest_library_path()
        else:
            lib_path = Path(library_path)
        
        if not lib_path.exists():
            return []
        
        # 加载index.json
        index_file = lib_path / "index.json"
        if not index_file.exists():
            return []
        
        with open(index_file, 'r', encoding='utf-8') as f:
            index = json.load(f)
        
        # 过滤匹配的ActSpec（使用灵活的匹配策略 + 动作历史前缀指纹）
        from .url_utils import sites_match
        
        matched_entries = []
        query_site = context.get("site", "")
        query_page = context.get("page", "")
        query_url_pattern = context.get("url_pattern", "")
        query_history_types: List[str] = context.get("action_history_types") or []
        if not isinstance(query_history_types, list):
            query_history_types = []
        
        for entry in index:
            # 灵活匹配site（支持端口号部分匹配）
            # 例如：查询 "amazonaws" 可以匹配 ActSpec 的 "amazonaws:9999"
            # 或者查询 "amazonaws:9999" 可以匹配 ActSpec 的 "amazonaws"
            actspec_site = entry.get("site", "")
            if query_site:
                if not sites_match(query_site, actspec_site, flexible=True):
                    continue
            
            # 匹配page（如果提供，支持部分匹配）
            # 例如：查询 "singularity" 可以匹配 ActSpec 的 "singularity" 或包含 "singularity" 的page
            actspec_page = entry.get("page", "")
            if query_page:
                # 支持精确匹配或包含匹配
                if query_page != actspec_page and query_page not in actspec_page and actspec_page not in query_page:
                    continue
            
            matched_entries.append(entry)
        
        # 加载匹配的ActSpec文件
        actspecs: List[Dict[str, Any]] = []
        for entry in matched_entries:
            file_path = lib_path / entry["file"]
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    actspec = json.load(f)
                    
                    # 进一步检查url_pattern匹配（如果提供）
                    if query_url_pattern:
                        actspec_url_pattern = actspec.get("context", {}).get("url_pattern", "")
                        if actspec_url_pattern and query_url_pattern not in actspec_url_pattern:
                            continue
                    
                    # 过滤已被禁用的ActSpec（淘汰仅由统计规则 update_stats_batch 触发）
                    metadata = actspec.get("metadata", {})
                    if metadata.get("disabled", False):
                        continue

                    # 动作历史前缀匹配：若 ActSpec 带有非空 action_history_prefix，
                    # 则当前历史类型序列中必须包含该前缀作为连续子序列；
                    # 若该字段缺失或为空列表，则视为对历史无要求。
                    prefix = actspec.get("action_history_prefix")
                    if isinstance(prefix, list) and prefix:
                        if not _history_prefix_matches(prefix, query_history_types):
                            continue

                    actspecs.append(actspec)
        
        # 排序：从未被调用过的(usage_count==0) 最高优先级，其次按置信度降序
        def _sort_key(spec):
            meta = spec.get("metadata", {})
            usage = int(meta.get("usage_count", 0) or 0)
            conf = float(meta.get("confidence", 1.0))
            return (-conf, usage)
        actspecs.sort(key=_sort_key)
        
        return actspecs
    
    def _get_library_path(self, base_path: Path, timestamp: str) -> Path:
        """
        获取库路径
        
        Args:
            base_path: 基础路径
            timestamp: 时间戳
        
        Returns:
            库路径
        """
        return base_path / timestamp
    
    def _get_latest_library_path(self) -> Path:
        """
        获取最新的库路径（按时间戳排序）
        
        Returns:
            最新的库路径
        """
        if not self.base_path.exists():
            return self.base_path
        
        # 获取所有时间戳目录
        timestamp_dirs = []
        for item in self.base_path.iterdir():
            if item.is_dir():
                try:
                    # 尝试解析时间戳
                    datetime.strptime(item.name, "%Y%m%d_%H%M%S")
                    timestamp_dirs.append(item)
                except ValueError:
                    # 不是时间戳格式，跳过
                    continue
        
        if not timestamp_dirs:
            return self.base_path
        
        # 按名称排序（时间戳格式可以按字符串排序）
        timestamp_dirs.sort(key=lambda x: x.name, reverse=True)
        return timestamp_dirs[0]
    
    def _update_index(
        self,
        actspec: Dict[str, Any],
        library_path: Path,
        filename: str
    ) -> None:
        """
        更新或追加index.json
        
        Args:
            actspec: ActSpec字典
            library_path: 库路径
            filename: 文件名
        """
        index_file = library_path / "index.json"
        
        # 加载现有index
        index = []
        if index_file.exists():
            with open(index_file, 'r', encoding='utf-8') as f:
                index = json.load(f)
        
        # 检查是否已存在
        action_id = actspec.get("action_id", "")
        existing_idx = None
        for i, entry in enumerate(index):
            if entry.get("action_id") == action_id:
                existing_idx = i
                break
        
        # 准备新条目
        context = actspec.get("context", {})
        new_entry = {
            "action_id": action_id,
            "site": context.get("site", "unknown"),
            "page": context.get("page", "unknown"),
            "file": filename
        }
        
        # 更新或追加
        if existing_idx is not None:
            index[existing_idx] = new_entry
        else:
            index.append(new_entry)
        
        # 保存index.json
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2, ensure_ascii=False)
    
    def load_negative_constraints(self, library_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        加载负约束库
        
        Args:
            library_path: 库路径，如果为None则使用最新的时间戳目录
        
        Returns:
            负约束列表
        """
        if library_path is None:
            # 查找最新的时间戳目录
            lib_path = self._get_latest_library_path()
        else:
            lib_path = Path(library_path)
        
        if not lib_path.exists():
            return []
        
        # 加载negative_constraints_index.json
        index_file = lib_path / "negative_constraints_index.json"
        if not index_file.exists():
            return []
        
        with open(index_file, 'r', encoding='utf-8') as f:
            index = json.load(f)
        
        # 加载所有负约束文件
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
        """
        保存负约束到库
        
        Args:
            constraint: 负约束字典
            library_path: 库路径，如果为None则使用时间戳目录
        
        Returns:
            保存的文件路径
        """
        # 获取库路径
        if library_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            lib_path = self._get_library_path(self.base_path, timestamp)
        else:
            lib_path = Path(library_path)
        
        lib_path.mkdir(parents=True, exist_ok=True)
        
        # 创建负约束子目录
        negative_dir = lib_path / "negative_constraints"
        negative_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取constraint_id作为文件名
        constraint_id = constraint.get("constraint_id", "unknown")
        # 清理constraint_id，使其适合作为文件名
        safe_filename = constraint_id.replace("/", "_").replace("\\", "_").replace(":", "_")
        filename = f"{safe_filename}.json"
        file_path = negative_dir / filename
        
        # 保存负约束
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(constraint, f, indent=2, ensure_ascii=False)
        
        # 更新负约束索引
        self._update_negative_constraints_index(constraint, lib_path, f"negative_constraints/{filename}")
        
        return str(file_path)
    
    def _update_negative_constraints_index(
        self,
        constraint: Dict[str, Any],
        library_path: Path,
        filename: str
    ) -> None:
        """
        更新或追加negative_constraints_index.json
        
        Args:
            constraint: 负约束字典
            library_path: 库路径
            filename: 文件名（相对于库路径）
        """
        index_file = library_path / "negative_constraints_index.json"
        
        # 加载现有index
        index = []
        if index_file.exists():
            with open(index_file, 'r', encoding='utf-8') as f:
                index = json.load(f)
        
        # 检查是否已存在
        constraint_id = constraint.get("constraint_id", "")
        existing_idx = None
        for i, entry in enumerate(index):
            if entry.get("constraint_id") == constraint_id:
                existing_idx = i
                break
        
        # 准备新条目（含论文两类子类型，便于筛选与统计）
        context = constraint.get("context", {})
        new_entry = {
            "constraint_id": constraint_id,
            "site": context.get("site", "unknown"),
            "page": context.get("page", "unknown"),
            "constraint_subtype": constraint.get("constraint_subtype", "unspecified"),
            "file": filename
        }
        
        # 更新或追加
        if existing_idx is not None:
            index[existing_idx] = new_entry
        else:
            index.append(new_entry)
        
        # 保存index.json
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2, ensure_ascii=False)

    # ============ 以下为在线统计与置信度更新相关的新接口 ============

    def _resolve_library_path(self, library_path: Optional[str] = None) -> Path:
        """
        解析并返回实际的库路径。
        - 如果传入的是时间戳子目录，直接使用；
        - 否则回退到当前 base_path 的最新时间戳子目录。
        """
        if library_path:
            lib_path = Path(library_path)
        else:
            lib_path = self._get_latest_library_path()
        return lib_path

    def _load_index(
        self,
        lib_path: Path,
    ) -> Tuple[Path, list]:
        """加载给定库路径下的 index.json。"""
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
        批量更新多个 ActSpec 的统计信息。

        Args:
            stats: 形如 {action_id: {"usage_count": x, "success_count": y, "fail_count": z}} 的字典，
                   所有值为“本次评估新增的增量”，不是绝对值。
            library_path: 库路径；若为 None，则使用最近一次的库目录。
        """
        if not stats:
            return

        lib_path = self._resolve_library_path(library_path)
        if not lib_path.exists():
            return

        index_file, index = self._load_index(lib_path)
        if not index:
            return

        # 建立 action_id -> 文件路径 映射
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

            # 重新计算失败率
            fail_rate = (fail / usage) if usage > 0 else 0.0

            disabled_prev = bool(metadata.get("disabled", False))

            metadata["usage_count"] = usage
            metadata["success_count"] = success
            metadata["fail_count"] = fail
            metadata["fail_rate"] = round(fail_rate, 4)

            # 根据规则更新 disabled 标记：
            # - 使用次数 <5 且失败次数 >=3 -> 禁用
            # - 使用次数 >=5 且失败率 >= 0.4 -> 禁用
            disabled = metadata.get("disabled", False)
            if usage < 5 and fail >= 3:
                disabled = True
            elif usage >= 5 and fail_rate >= 0.4:
                disabled = True
            metadata["disabled"] = disabled

            # 当 ActSpec 首次因失败率规则被禁用时，可选择将其转换为负约束写入当前库。
            # 消融实验中应关闭该功能，避免执行失败污染负约束库。
            if (
                convert_to_negative_constraints
                and (not disabled_prev)
                and disabled
                and not metadata.get("converted_to_negative_constraint")
            ):
                try:
                    constraint = _build_negative_constraint_from_actspec_struct(actspec)
                    if constraint:
                        # 保存到同一库目录下的 negative_constraints 子目录
                        self.save_negative_constraint(constraint, str(lib_path))
                        metadata["converted_to_negative_constraint"] = True
                        print(f"[负约束] ActSpec {actspec.get('action_id', 'unknown')} 多次失败已被禁用，"
                              f"自动转换为负约束 {constraint.get('constraint_id', 'unknown')}")
                except Exception as e:
                    print(f"[负约束] 将 ActSpec 转换为负约束失败 ({actspec.get('action_id', 'unknown')}): {e}")

            actspec["metadata"] = metadata

            # 写回文件
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(actspec, f, indent=2, ensure_ascii=False)
            except Exception:
                # 出错时跳过该 ActSpec，避免影响其他更新
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
        根据 Post 条件验证结果更新单个 ActSpec 的置信度。
        仅用于排序：置信度只影响备选列表顺序，不触发淘汰（淘汰仅由 update_stats_batch 的统计规则触发）。
        
        更新规则：
        - Post 成功：s(α) ← s(α) + 1
        - Post 失败：s(α) ← s(α) - λ
        - 上界裁剪：s(α) ≤ s_max，下界保护 s(α) ≥ 0
        
        Args:
            action_id: ActSpec 的 action_id
            post_success: Post 条件是否通过（True=成功，False=失败）
            s_max: 置信度上界（默认 10.0）
            lambda_penalty: 失败惩罚系数（默认 2.0）
            library_path: 库路径；若为 None，则使用最近一次的库目录
        
        Returns:
            更新后的置信度值，如果 ActSpec 不存在则返回 None
        """
        lib_path = self._resolve_library_path(library_path)
        if not lib_path.exists():
            return None

        index_file, index = self._load_index(lib_path)
        if not index:
            return None

        # 查找对应的文件路径
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

        # 根据 Post 条件验证结果更新置信度
        if post_success:
            # Post 成功：s(α) ← s(α) + 1
            new_confidence = current_confidence + 1.0
        else:
            # Post 失败：s(α) ← s(α) - λ
            new_confidence = current_confidence - lambda_penalty

        # 上界裁剪：s(α) ≤ s_max
        new_confidence = min(new_confidence, s_max)

        # 下界保护：置信度不低于 0（仅影响排序，不触发淘汰）
        new_confidence = max(new_confidence, 0.0)

        # 更新 metadata（不再根据 confidence==0 设置 disabled）
        metadata["confidence"] = round(new_confidence, 2)
        if new_confidence <= 0.0:
            metadata["disabled"] = True

        actspec["metadata"] = metadata

        # 写回文件
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
        """
        批量更新多个 ActSpec 的置信度。
        
        Args:
            updates: 形如 {action_id: post_success} 的字典，post_success 为 True/False
            s_max: 置信度上界（默认 10.0）
            lambda_penalty: 失败惩罚系数（默认 2.0）
            library_path: 库路径；若为 None，则使用最近一次的库目录
        
        Returns:
            形如 {action_id: new_confidence} 的字典，包含更新后的置信度值
        """
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
        """将失败 ActSpec 立即转为负约束并落库。"""
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
    """
    检查 history 中是否包含 prefix 作为连续子序列。
    prefix 与 history 均为动作类型字符串（小写）。
    """
    if not prefix:
        return True
    if not history:
        return False
    n, m = len(history), len(prefix)
    if n < m:
        return False
    # 滑动窗口，检查任意连续子序列是否与前缀完全一致
    for start in range(0, n - m + 1):
        if history[start:start + m] == prefix:
            return True
    return False


def _actspec_id_to_constraint_id(action_id: str) -> str:
    """
    将 ActSpec 的 action_id 转换为负约束的 constraint_id。
    规则：前缀替换 unknown. -> constraint.，否则加上 constraint. 前缀。
    """
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
    """
    将一个已存在的 ActSpec 结构（通常是被禁用的多次失败 ActSpec）
    转换为一个结构上兼容的负约束。

    注意：这里没有具体失败片段的轨迹，只能保守地：
    - 直接复用 ActSpec 的 context / pre_condition / plan / action_history_prefix；
    - constraint_subtype 置为 unspecified。
    """
    action_id = actspec.get("action_id", "")
    context = actspec.get("context", {}) or {}
    pre_condition = actspec.get("pre_condition", {}) or {}
    forbidden_plan = actspec.get("plan", []) or []
    action_history_prefix = actspec.get("action_history_prefix", []) or []

    constraint_id = _actspec_id_to_constraint_id(action_id)
    failure_reason = actspec.get("failure_reason") or "ActSpec 多次失败，被禁用后自动转换为负约束。"

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
    """
    依据失败轨迹信号构建负约束（贴近论文 Algorithm 3）：
    c = <phi(Delta), pi(a,tgt)>.
    """
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
    """负约束过滤器，用于在planning阶段过滤被禁止的action"""
    
    def __init__(self, constraints: List[Dict[str, Any]]):
        """
        初始化负约束过滤器
        
        Args:
            constraints: 负约束列表
        """
        self.constraints = constraints

    def _constraint_applies(self, constraint: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """
        检查负约束是否适用于当前上下文。

        规则：
        1. 先做上下文匹配；
        2. 再按“包含匹配”检查当前动作历史：
           - 若约束带有非空 action_history_prefix，则当前已执行完的历史动作类型序列中，
             必须包含该前缀作为连续子序列才视为匹配；
           - 若约束未带该字段或为空列表，则视为对历史无要求，仅依赖上下文。
        """
        if not self._context_matches(
            constraint.get("context", {}), context, constraint.get("constraint_subtype", "unspecified")
        ):
            return False

        # 若负约束声明了 pre_condition，则沿用 ActSpec 的 PreConditionChecker 语义：
        # 仅当当前页面/状态满足 pre_condition 时，该负约束才参与后续匹配。
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
                    # 不满足 pre_condition，则该负约束在当前步骤不生效
                    return False
            except Exception as e:
                # 保守策略：pre_condition 检查异常时不禁用该约束，仍然继续后续匹配
                constraint_id = constraint.get("constraint_id", "unknown")
                print(f"[负约束] pre-condition 检查出错 (constraint_id={constraint_id}): {e}，保守保留该约束继续匹配")

        # 动作历史前缀宽松匹配（包含匹配）
        prefix = constraint.get("action_history_prefix")
        if not isinstance(prefix, list) or not prefix:
            # 无指纹或空指纹：视为对历史无要求
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
        """
        检查给定的plan是否被负约束禁止
        
        Args:
            plan: 要检查的plan（primitive序列）
            context: 上下文信息
        
        Returns:
            如果被禁止，返回True
        """
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
        """
        检查给定的ActSpec是否被负约束禁止
        
        Args:
            actspec: 要检查的ActSpec
            context: 上下文信息
        
        Returns:
            如果被禁止，返回True
        """
        plan = actspec.get("plan", [])
        return self.is_forbidden(plan, context)
    
    def filter_actspecs(
        self,
        actspecs: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        过滤被负约束禁止的ActSpec
        
        Args:
            actspecs: ActSpec列表
            context: 上下文信息
        
        Returns:
            过滤后的ActSpec列表
        """
        filtered = []
        for actspec in actspecs:
            if not self.is_actspec_forbidden(actspec, context):
                filtered.append(actspec)
            else:
                constraint_id = self._find_matching_constraint_id(actspec, context)
                print(f"[负约束] ActSpec {actspec.get('action_id')} 被负约束禁止 (constraint: {constraint_id})")
        return filtered
    
    def _context_matches(
        self,
        constraint_context: Dict[str, Any],
        current_context: Dict[str, Any],
        constraint_subtype: str = "",
    ) -> bool:
        """检查上下文是否匹配。对 readiness 约束且含 unstable_state 时，仅在当前也为未稳定状态时匹配。"""
        # 检查site和page是否匹配
        constraint_site = constraint_context.get("site", "unknown")
        constraint_page = constraint_context.get("page", "unknown")
        current_site = current_context.get("site", "unknown")
        current_page = current_context.get("page", "unknown")
        
        # 如果约束的site不是unknown，必须匹配
        if constraint_site != "unknown" and constraint_site != current_site:
            return False
        
        # 如果约束的page不是unknown，必须匹配
        if constraint_page != "unknown" and constraint_page != current_page:
            return False
        
        # Readiness 约束：若约束中指明了 unstable_state，则仅在当前上下文也为“未稳定”时生效。
        # 若 env 未提供 unstable_state 键，则保守地仍应用该约束（避免漏拦）。
        if constraint_subtype == "readiness" and constraint_context.get("unstable_state"):
            if "unstable_state" in current_context and not current_context.get("unstable_state"):
                return False
        # 可以添加更复杂的URL pattern匹配逻辑
        return True
    
    def _plan_matches(
        self,
        forbidden_plan: List[Dict[str, Any]],
        plan: List[Dict[str, Any]]
    ) -> bool:
        """检查 plan 是否匹配被禁止的 plan。
        阶段1改造：对 CLICK/TYPE/HOVER 优先做 element_id 精确匹配；
        若 forbidden_plan 首步为占位符（无具体 element_id），planning 阶段不采用该约束，返回 False。
        """
        if len(plan) < len(forbidden_plan):
            return False
        if not forbidden_plan:
            return False

        # 首步为占位符时，planning 阶段不采用该负约束（避免按 primitive 前缀误杀整条 ActSpec）
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
        """查找匹配的负约束ID"""
        plan = actspec.get("plan", [])
        for constraint in self.constraints:
            if self._constraint_applies(constraint, context):
                if self._plan_matches(constraint.get("forbidden_plan", []), plan):
                    return constraint.get("constraint_id", "unknown")
        return "unknown"

    def get_forbidden_element_ids_for_observation(self, context: Dict[str, Any]) -> Set[str]:
        """
        获取在 observe 阶段应隐藏的 element_id 集合。
        仅当 context + action_history_prefix 匹配且 forbidden_plan 中含具体 element_id 时纳入。
        用于在 observation 返回前将失败过的候选元素从 DOM 中隐藏。
        """
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
        """
        检查单个primitive action是否被负约束禁止
        
        Args:
            action_cmd: primitive action命令字典（包含action_type, element_id等）
            context: 上下文信息（site, page, url）
        
        Returns:
            (是否被禁止, 匹配的负约束信息) 元组
            如果被禁止，返回 (True, constraint_info)，其中 constraint_info 包含：
            - constraint_id: 负约束ID
            - failure_reason: 失败原因
            - description: 描述信息
            如果未被禁止，返回 (False, None)
        """
        # 将 action_cmd 转换为 plan step 格式
        plan_step = self._action_cmd_to_plan_step(action_cmd)
        if not plan_step:
            # 如果无法转换，认为不被禁止（可能是未知的action类型）
            return (False, None)
        
        # 检查是否匹配任何负约束的 forbidden_plan 的第一个 step
        for constraint in self.constraints:
            if not self._constraint_applies(constraint, context):
                continue
            
            # 检查 forbidden_plan 的第一个 step 是否匹配
            forbidden_plan = constraint.get("forbidden_plan", [])
            if not forbidden_plan:
                continue
            
            # 检查第一个 step 是否匹配
            first_step = forbidden_plan[0]
            if self._plan_step_matches(first_step, plan_step):
                # 找到匹配的负约束
                constraint_info = {
                    "constraint_id": constraint.get("constraint_id", "unknown"),
                    "failure_reason": constraint.get("description", {}).get("failure_reason", "该操作被负约束禁止"),
                    "description": constraint.get("description", {})
                }
                return (True, constraint_info)
        
        return (False, None)
    
    def _action_cmd_to_plan_step(self, action_cmd: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        将 action_cmd 转换为 plan step 格式
        
        Args:
            action_cmd: primitive action命令字典
        
        Returns:
            plan step 字典，如果无法转换则返回 None
        """
        # 使用硬编码的映射（避免循环导入）
        # ActionTypes 的值定义在 browser_env/actions.py 中：
        # SCROLL = 1, CLICK = 6, TYPE = 7, HOVER = 8, PAGE_FOCUS = 9, NEW_TAB = 10,
        # GOTO_URL = 11, GO_BACK = 12, GO_FORWARD = 13, STOP = 14
        action_type_to_primitive = {
            1: "SCROLL",      # SCROLL
            6: "CLICK",       # CLICK
            7: "TYPE",        # TYPE
            8: "HOVER",       # HOVER
            9: "GOTO",        # PAGE_FOCUS (简化处理为 GOTO)
            10: "GOTO",       # NEW_TAB (简化处理为 GOTO)
            11: "GOTO",       # GOTO_URL
            12: "GOBACK",     # GO_BACK
            13: "GOFORWARD",  # GO_FORWARD
            14: "STOP",       # STOP
        }
        
        action_type = action_cmd.get("action_type")
        if action_type not in action_type_to_primitive:
            return None
        
        primitive = action_type_to_primitive[action_type]
        plan_step = {"primitive": primitive}
        
        # 根据 primitive 类型添加其他字段
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
                # 如果是字符ID列表，需要转换（这里简化处理）
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
        """
        检查 plan step 是否匹配被禁止的 step
        
        Args:
            forbidden_step: 被禁止的 step
            plan_step: 要检查的 step
        
        Returns:
            如果匹配，返回 True
        """
        # 检查 primitive 类型是否匹配
        forbidden_primitive = forbidden_step.get("primitive", "").upper()
        plan_primitive = plan_step.get("primitive", "").upper()
        
        if forbidden_primitive != plan_primitive:
            return False
        
        # 如果 primitive 是 NOTE 或 PRUNE，需要匹配相同的 primitive 和 raw 内容
        if forbidden_primitive in ["NOTE", "PRUNE"]:
            # NOTE 和 PRUNE 必须匹配相同的 primitive
            if plan_primitive != forbidden_primitive:
                return False
            # 还需要检查 raw 字段是否匹配（如果都有）
            forbidden_raw = forbidden_step.get("raw", "")
            plan_raw = plan_step.get("raw", "")
            if forbidden_raw and plan_raw:
                # 对于 note 和 prune，需要匹配相同的 raw 内容
                if forbidden_raw != plan_raw:
                    return False
            return True
        
        # 如果 primitive 是 UNKNOWN，需要更严格的匹配条件
        # 只有当负约束明确指定了element_id或其他具体条件时，才匹配
        if forbidden_primitive == "UNKNOWN":
            # 检查是否有element_id约束
            forbidden_target = forbidden_step.get("target", {})
            if forbidden_target.get("strategy") == "element_id":
                # 如果指定了element_id，只匹配该element_id的操作
                forbidden_element_id = forbidden_target.get("value", "")
                plan_element_id = plan_step.get("target", {}).get("value", "")
                
                # 如果负约束使用参数占位符（如 {{click_id}}），视为过度宽泛，不予采用（不匹配）
                if self._is_parameter_placeholder(forbidden_element_id):
                    return False
                
                # 否则精确匹配element_id
                if forbidden_element_id and plan_element_id != forbidden_element_id:
                    return False
                return True
            else:
                # 如果只有UNKNOWN且没有element_id约束，不应该匹配（太宽泛）
                return False
        
        # 对于 CLICK, TYPE, HOVER 等需要 element_id 的操作，检查 element_id 是否匹配
        if forbidden_primitive in ["CLICK", "TYPE", "HOVER"]:
            forbidden_target = forbidden_step.get("target", {})
            plan_target = plan_step.get("target", {})
            
            # 如果负约束中指定了 target，必须完全匹配
            if forbidden_target:
                # 检查 strategy 是否匹配
                forbidden_strategy = forbidden_target.get("strategy", "")
                plan_strategy = plan_target.get("strategy", "")
                if forbidden_strategy and forbidden_strategy != plan_strategy:
                    return False
                
                # 如果指定了 element_id，必须精确匹配
                if forbidden_strategy == "element_id":
                    forbidden_element_id = forbidden_target.get("value", "")
                    plan_element_id = plan_target.get("value", "")
                    
                    # 如果负约束使用参数占位符（如 {{click_id}}），视为过度宽泛，不予采用（不匹配）
                    if self._is_parameter_placeholder(forbidden_element_id):
                        return False
                    else:
                        # 否则精确匹配 element_id
                        if forbidden_element_id and plan_element_id != forbidden_element_id:
                            return False
            
            # 对于 TYPE 操作，还需要检查 text 字段是否匹配
            if forbidden_primitive == "TYPE":
                forbidden_text = forbidden_step.get("text", "")
                plan_text = plan_step.get("text", "")
                
                # 如果负约束指定了 text，必须匹配
                if forbidden_text:
                    # 如果负约束使用参数占位符，匹配任何 text
                    if self._is_parameter_placeholder(forbidden_text):
                        pass  # 参数占位符匹配任何值
                    else:
                        # 否则精确匹配 text
                        if forbidden_text != plan_text:
                            return False
        
        # 对于 SCROLL 操作，检查 direction 是否匹配
        if forbidden_primitive == "SCROLL":
            forbidden_direction = forbidden_step.get("direction", "")
            plan_direction = plan_step.get("direction", "")
            
            # 如果负约束指定了 direction，必须匹配
            if forbidden_direction and forbidden_direction != plan_direction:
                return False
        
        # 对于 GOTO 操作，检查 url 是否匹配
        if forbidden_primitive == "GOTO":
            forbidden_url = forbidden_step.get("url", "")
            plan_url = plan_step.get("url", "")
            
            # 如果负约束指定了 url，必须匹配
            if forbidden_url:
                # 如果负约束使用参数占位符，匹配任何 url
                if self._is_parameter_placeholder(forbidden_url):
                    pass  # 参数占位符匹配任何值
                else:
                    # 否则精确匹配 url
                    if forbidden_url != plan_url:
                        return False
        
        return True
    
    def _is_parameter_placeholder(self, value: str) -> bool:
        """
        检查是否是参数占位符（如 {{param_name}}）
        
        Args:
            value: 要检查的值
        
        Returns:
            如果是参数占位符，返回 True
        """
        return isinstance(value, str) and value.startswith("{{") and value.endswith("}}")
