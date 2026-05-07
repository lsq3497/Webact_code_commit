import os
import time
import re
import argparse
import shutil
import sys
import json
import tempfile
import pandas as pd
import signal
import threading
from pathlib import Path
from typing import Optional
from datetime import datetime
from collections import Counter, defaultdict

from AgentOccam.env import WebArenaEnvironmentWrapper

from AgentOccam.AgentOccam import AgentOccam
from webagents_step.utils.data_prep import *
from webagents_step.agents.step_agent import StepAgent

from AgentOccam.prompts import AgentOccam_prompt
from webagents_step.prompts.webarena import step_fewshot_template_adapted, step_fewshot_template

from AgentOccam.utils import EVALUATOR_DIR


class TeeOutput:
    """将 sys.stdout/sys.stderr 同时写入控制台和文件，实现终端输出实时保存。"""

    def __init__(self, stream, log_file):
        self._stream = stream
        self._log_file = log_file

    def write(self, data):
        try:
            self._stream.write(data)
            self._stream.flush()
        except Exception:
            pass
        try:
            self._log_file.write(data)
            self._log_file.flush()
        except Exception:
            pass

    def flush(self):
        try:
            self._stream.flush()
        except Exception:
            pass
        try:
            self._log_file.flush()
        except Exception:
            pass

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _start_console_logging():
    """
    在 Temp 目录下创建以时间戳命名的 txt 日志文件，并将 stdout/stderr 重定向到该文件（同时保留控制台输出）。
    返回 (log_file_handle, original_stdout, original_stderr)，用于在 run() 结束时恢复。
    """
    current_dir = Path(__file__).resolve().parent
    main_dir = current_dir.parent
    temp_base_dir = main_dir / "Temp"
    os.makedirs(temp_base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    log_path = temp_base_dir / f"eval_console_{timestamp}.txt"
    try:
        log_file = open(log_path, "w", encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"[警告] 无法创建终端日志文件 {log_path}: {e}", file=sys.__stderr__)
        return None, None, None
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeOutput(sys.stdout, log_file)
    sys.stderr = TeeOutput(sys.stderr, log_file)
    # 在日志文件开头写入一行标识
    print(f"[终端日志] 开始时间: {datetime.now().isoformat()}, 日志文件: {log_path}")
    return log_file, original_stdout, original_stderr


def _stop_console_logging(log_file, original_stdout, original_stderr):
    """恢复 stdout/stderr 并关闭日志文件。"""
    if log_file is None:
        return
    try:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()
    except Exception:
        pass


# 加载URL映射并替换占位符
def load_url_mapping():
    """加载 Webarena_website.json 中的URL映射"""
    current_dir = Path(__file__).resolve().parent
    url_mapping_file = current_dir / "Webarena_website.json"
    if not url_mapping_file.exists():
        raise FileNotFoundError(f"URL mapping file not found: {url_mapping_file}")
    
    with open(url_mapping_file, "r", encoding="utf-8") as f:
        url_mapping = json.load(f)
    
    return url_mapping

def replace_placeholders(data, url_mapping):
    """
    递归替换数据中的所有占位符（格式：__XXX__）
    占位符会被替换为 url_mapping 中对应的 XXX_URL 的值
    """
    if isinstance(data, dict):
        return {key: replace_placeholders(value, url_mapping) for key, value in data.items()}
    elif isinstance(data, list):
        return [replace_placeholders(item, url_mapping) for item in data]
    elif isinstance(data, str):
        result = data
        # 先构建占位符到URL的映射
        placeholder_to_url = {}
        for placeholder_key, url_value in url_mapping.items():
            # 从 "XXX_URL" 提取 "XXX"
            if placeholder_key.endswith("_URL"):
                placeholder_name = placeholder_key[:-4]  # 去掉 "_URL"
                placeholder_pattern = f"__{placeholder_name}__"
                placeholder_to_url[placeholder_pattern] = url_value
        
        # 按长度从长到短排序，优先替换更长的占位符（避免部分匹配问题）
        sorted_patterns = sorted(placeholder_to_url.keys(), key=len, reverse=True)
        
        # 替换所有占位符
        for placeholder_pattern in sorted_patterns:
            url_value = placeholder_to_url[placeholder_pattern]
            if placeholder_pattern in result:
                # 如果完全匹配占位符
                if result == placeholder_pattern:
                    result = url_value
                # 如果占位符后跟路径（以/开头）
                elif result.startswith(placeholder_pattern + "/"):
                    path = result[len(placeholder_pattern):]
                    # 确保URL和路径正确拼接（处理URL末尾的/和路径开头的/）
                    if url_value.endswith("/"):
                        result = url_value.rstrip("/") + path
                    else:
                        result = url_value + path
                # 如果占位符在字符串开头，后跟其他内容
                elif result.startswith(placeholder_pattern):
                    remaining = result[len(placeholder_pattern):]
                    result = url_value + remaining
                else:
                    # 占位符在字符串中间或其他位置，直接替换
                    result = result.replace(placeholder_pattern, url_value)
        return result
    else:
        return data

# 尝试加载统一配置
config_dir = None
try:
    current_dir = Path(__file__).resolve().parent
    config_dir = current_dir / "config"
    if str(config_dir) not in sys.path:
        sys.path.insert(0, str(config_dir.parent))
    from config.config_loader import get_config
    _unified_config = get_config(str(config_dir))
    _has_unified_config = True
except (ImportError, FileNotFoundError):
    _unified_config = None
    _has_unified_config = False


def _apply_unified_config_overrides(config: DotDict, unified_config) -> DotDict:
    """
    用统一配置覆盖 AgentOccam 配置文件中的相同配置项
    统一配置（config/config.yaml）优先级最高，会覆盖 AgentOccam/configs/*.yml 中的配置
    """
    # 1. 覆盖模型配置（最高优先级）
    if hasattr(config, "agent") and config.agent.type == "AgentOccam":
        # Actor 模型 - 统一配置优先
        if hasattr(config.agent, "actor"):
            actor_model = unified_config.get_model("agent_actor", "default")
            if actor_model:
                config.agent.actor.model = actor_model
        
        # Critic 模型 - 统一配置优先
        if hasattr(config.agent, "critic"):
            critic_model = unified_config.get_model("agent_critic", "default")
            if critic_model:
                config.agent.critic.model = critic_model
        
        # Judge 模型 - 统一配置优先
        if hasattr(config.agent, "judge"):
            judge_model = unified_config.get_model("agent_judge", "default")
            if judge_model:
                config.agent.judge.model = judge_model
    
    # 2. 覆盖浏览器配置
    if hasattr(config, "env"):
        browser_config = unified_config.get("browser", {})
        if browser_config:
            if "headless" in browser_config:
                config.env.headless = browser_config["headless"]
            if "max_browser_rows" in browser_config:
                config.env.max_browser_rows = browser_config["max_browser_rows"]
            if "observation_type" in browser_config:
                config.env.observation_type = browser_config["observation_type"]
            if "current_viewport_only" in browser_config:
                config.env.current_viewport_only = browser_config["current_viewport_only"]
    
    # 3. 覆盖日志配置（顶层）
    logging_config = unified_config.get("logging", {})
    if logging_config:
        if "enabled" in logging_config:
            config.logging = logging_config["enabled"]
        if "logdir" in logging_config:
            config.logdir = logging_config["logdir"]
        if "logname" in logging_config:
            config.logname = logging_config["logname"]
        if "verbose" in logging_config:
            config.verbose = logging_config["verbose"]
        if "debug" in logging_config:
            config.debug = logging_config["debug"]
        
        # 同时覆盖 agent.others 中的日志配置
        if hasattr(config, "agent") and hasattr(config.agent, "others"):
            if "logname" in logging_config:
                config.agent.others.logname = logging_config["logname"]
            if "enabled" in logging_config:
                config.agent.others.logging = logging_config["enabled"]
            if "verbose" in logging_config:
                config.agent.others.verbose = logging_config["verbose"]
            if "debug" in logging_config:
                config.agent.others.debug = logging_config["debug"]
    
    # 4. 覆盖默认任务配置
    default_task_config = unified_config.get("default_task", {})
    if default_task_config:
        if "max_steps" in default_task_config:
            # 覆盖顶层 max_steps
            config.max_steps = default_task_config["max_steps"]
            # 同时覆盖 agent.others.max_steps
            if hasattr(config, "agent") and hasattr(config.agent, "others"):
                config.agent.others.max_steps = default_task_config["max_steps"]
        if "relative_task_dir" in default_task_config:
            if hasattr(config, "env"):
                config.env.relative_task_dir = default_task_config["relative_task_dir"]
        if "timeout" in default_task_config:
            # 设置任务执行超时时间
            config.task_timeout = default_task_config["timeout"]
    
    # 5. 覆盖ActSpec配置
    actspec_config = unified_config.get("actspec", {})
    if actspec_config:
        # 将actspec配置直接添加到config对象中
        config.actspec = actspec_config
    
    return config


def _apply_agent_model_cli_override(config: DotDict, model: Optional[str]) -> DotDict:
    """
    命令行 --agent-model：将 Actor/Critic/Judge 设为同一模型。
    优先级高于 config/config.yaml 中的 models.agent_* 配置。
    """
    if not model or not str(model).strip():
        return config
    model = str(model).strip()
    if hasattr(config, "agent") and getattr(config.agent, "type", None) == "AgentOccam":
        for role in ("actor", "critic", "judge"):
            if hasattr(config.agent, role):
                getattr(config.agent, role).model = model
        print(f"[配置] --agent-model 已生效（覆盖统一配置）: Actor / Critic / Judge -> {model}")
    return config


def extract_primitive_actions(trajectory):
    """
    从 trajectory 中提取 primitive action 调用次数
    返回一个字典，包含每种 action 的调用次数
    注意：ActSpec内部执行的action不会被统计（通过_actspec_internal标记过滤）
    支持两种 step 格式：
    1. 含 action_type 字段（与 ActionTypes 枚举对应）
    2. 含 action 字符串（如 "click [607]"、"type [123] hello"）
    """
    from browser_env.actions import ActionTypes
    
    action_counter = Counter()
    # ActionTypes到action名称的映射
    ACTION_TYPE_MAP = {
        ActionTypes.CLICK: "click",
        ActionTypes.TYPE: "type",
        ActionTypes.SCROLL: "scroll",
        ActionTypes.GOTO_URL: "goto",
        ActionTypes.GO_BACK: "go_back",
        ActionTypes.GO_FORWARD: "go_forward",
        ActionTypes.STOP: "stop",
        ActionTypes.HOVER: "hover",
        ActionTypes.PAGE_FOCUS: "goto",
        ActionTypes.NEW_TAB: "goto",
    }
    
    ACTION_WITH_ID_LIST = ["click", "type", "scroll", "goto", "note", "stop", "branch", "prune", "go_back", "go_home"]
    ACTION_WITHOUT_ID_LIST = ["stop", "go_back", "go_home"]
    
    def parse_action_string(action_str):
        """从 action 字符串解析出 action 类型名，如 'click [607]' -> 'click'。未识别返回 None。"""
        if not action_str or not isinstance(action_str, str):
            return None
        action_str = action_str.strip()
        for action_type_name in ACTION_WITH_ID_LIST:
            if f"{action_type_name} [" in action_str:
                return action_type_name
        for action_type_name in ACTION_WITHOUT_ID_LIST:
            if action_str == action_type_name or action_str.startswith(action_type_name + " "):
                return action_type_name
        return None
    
    for step_data in trajectory:
        # 跳过ActSpec内部执行的action
        if isinstance(step_data, dict) and step_data.get("_actspec_internal", False):
            continue
        if not isinstance(step_data, dict):
            continue
        
        matched = False
        # 优先使用 action_type 字段（与 browser_env ActionTypes 一致）
        if "action_type" in step_data:
            action_type = step_data.get("action_type")
            if action_type in ACTION_TYPE_MAP:
                action_name = ACTION_TYPE_MAP[action_type]
                action_counter[action_name] += 1
                matched = True
        
        # 若无 action_type 或未在映射中，则从 action 字符串解析（兼容当前日志格式）
        if not matched and "action" in step_data:
            action_name = parse_action_string(step_data["action"])
            if action_name:
                action_counter[action_name] += 1
    
    return dict(action_counter)

def extract_token_usage(trajectory, agent):
    """
    从 trajectory 和 agent 中提取 token 使用统计
    返回一个字典，包含 token 使用信息
    """
    token_stats = {
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_tokens": 0,
        "actor_tokens": 0,
        "critic_tokens": 0,
        "judge_tokens": 0,
    }
    
    # 优先从 trajectory 中提取 token 使用信息（更准确，因为记录了每一步的token使用）
    for step_data in trajectory:
        # 提取每步的token使用
        if "token_usage" in step_data:
            usage = step_data["token_usage"]
            token_stats["total_input_tokens"] += usage.get("input_tokens", 0)
            token_stats["total_output_tokens"] += usage.get("output_tokens", 0)
            token_stats["total_tokens"] += usage.get("total_tokens", 0)
        
        # 提取actor/critic/judge的token使用
        if "actor_token_usage" in step_data:
            actor_usage = step_data["actor_token_usage"]
            token_stats["actor_tokens"] += actor_usage.get("total_tokens", 0)
        
        if "critic_token_usage" in step_data:
            critic_usage = step_data["critic_token_usage"]
            token_stats["critic_tokens"] += critic_usage.get("total_tokens", 0)
        
        if "judge_token_usage" in step_data:
            judge_usage = step_data["judge_token_usage"]
            token_stats["judge_tokens"] += judge_usage.get("total_tokens", 0)
    
    # 如果从trajectory中没有提取到token使用，尝试从agent对象中获取（向后兼容）
    if token_stats["total_tokens"] == 0:
        if hasattr(agent, 'actor') and agent.actor:
            if hasattr(agent.actor, 'token_usage'):
                token_stats["actor_tokens"] = agent.actor.token_usage.get("total_tokens", 0)
                token_stats["total_tokens"] += token_stats["actor_tokens"]
                token_stats["total_input_tokens"] += agent.actor.token_usage.get("input_tokens", 0)
                token_stats["total_output_tokens"] += agent.actor.token_usage.get("output_tokens", 0)
        
        if hasattr(agent, 'critic') and agent.critic:
            if hasattr(agent.critic, 'token_usage'):
                token_stats["critic_tokens"] = agent.critic.token_usage.get("total_tokens", 0)
                token_stats["total_tokens"] += token_stats["critic_tokens"]
                token_stats["total_input_tokens"] += agent.critic.token_usage.get("input_tokens", 0)
                token_stats["total_output_tokens"] += agent.critic.token_usage.get("output_tokens", 0)
        
        if hasattr(agent, 'judge') and agent.judge:
            if hasattr(agent.judge, 'token_usage'):
                token_stats["judge_tokens"] = agent.judge.token_usage.get("total_tokens", 0)
                token_stats["total_tokens"] += token_stats["judge_tokens"]
                token_stats["total_input_tokens"] += agent.judge.token_usage.get("input_tokens", 0)
                token_stats["total_output_tokens"] += agent.judge.token_usage.get("output_tokens", 0)
    
    return token_stats

def generate_statistics_report(dstdir):
    """
    生成统计报告，包括：
    1. 任务完成率
    2. primitive action 调用次数统计
    3. LLM token 开销统计
    4. ActSpec执行统计（如果启用了test_mode）
    """
    summary_file = os.path.join(dstdir, "summary.csv")
    if not os.path.exists(summary_file):
        print("[警告] 未找到 summary.csv 文件，无法生成统计报告")
        return
    
    # 读取 summary.csv
    try:
        df_summary = pd.read_csv(summary_file)
    except Exception as e:
        print(f"[错误] 读取 summary.csv 失败: {e}")
        return
    
    # 1. 计算任务完成率
    total_tasks = len(df_summary)
    if total_tasks == 0:
        print("[警告] summary.csv 中没有任务数据")
        return
    
    # 计算任务完成率
    if "success" in df_summary.columns:
        success_col = df_summary["success"]
        # 处理不同的数据类型
        if success_col.dtype in [int, float]:
            success_count = int(success_col.sum())
        elif success_col.dtype == bool:
            success_count = int(success_col.sum())
        else:
            # 处理字符串类型
            success_count = sum(1 for x in success_col if 
                              (isinstance(x, (int, float)) and x > 0) or 
                              (isinstance(x, str) and x.lower() in ['true', '1', '1.0', 'yes']))
    else:
        # 如果没有 success 列，尝试从其他列推断
        success_count = 0
        if "reward" in df_summary.columns:
            reward_col = df_summary["reward"]
            success_count = sum(1 for x in reward_col if (isinstance(x, (int, float)) and x > 0))
    
    completion_rate = (success_count / total_tasks * 100) if total_tasks > 0 else 0
    
    # 2. 统计 primitive action 调用次数
    action_stats = Counter()
    total_actions = 0
    
    for _, row in df_summary.iterrows():
        logfile = row.get("logfile", "")
        if not logfile:
            continue
        
        log_file_path = os.path.join(dstdir, logfile)
        if not os.path.exists(log_file_path):
            continue
        
        try:
            with open(log_file_path, "r", encoding="utf-8") as f:
                log_data = json.load(f)
            
            trajectory = log_data.get("trajectory", [])
            task_action_stats = extract_primitive_actions(trajectory)
            for action_type, count in task_action_stats.items():
                action_stats[action_type] += count
                total_actions += count
        except Exception as e:
            print(f"[警告] 读取日志文件 {log_file_path} 失败: {e}")
            continue
    
    # 3. 统计 LLM token 开销
    token_stats = {
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_tokens": 0,
        "actor_tokens": 0,
        "critic_tokens": 0,
        "judge_tokens": 0,
    }
    
    # 4. 统计 ActSpec 执行情况（如果启用了test_mode）
    actspec_stats = {
        "total_calls": 0,
        "success_count": 0,
        "fail_count": 0,
        "success_rate": 0.0,
    }
    actspec_enabled = False  # 标记是否启用了ActSpec test_mode
    
    # 尝试从日志文件中提取 token 使用信息和 ActSpec 调用信息
    for _, row in df_summary.iterrows():
        logfile = row.get("logfile", "")
        if not logfile:
            continue
        
        log_file_path = os.path.join(dstdir, logfile)
        if not os.path.exists(log_file_path):
            continue
        
        try:
            with open(log_file_path, "r", encoding="utf-8") as f:
                log_data = json.load(f)
            
            trajectory = log_data.get("trajectory", [])
            for step_data in trajectory:
                if "token_usage" in step_data:
                    usage = step_data["token_usage"]
                    token_stats["total_input_tokens"] += usage.get("input_tokens", 0)
                    token_stats["total_output_tokens"] += usage.get("output_tokens", 0)
                    token_stats["total_tokens"] += usage.get("total_tokens", 0)
                # 检查是否有 actor/critic/judge 的 token 使用
                if "actor_token_usage" in step_data:
                    actor_usage = step_data["actor_token_usage"]
                    token_stats["actor_tokens"] += actor_usage.get("total_tokens", actor_usage.get("total", 0))
                if "critic_token_usage" in step_data:
                    critic_usage = step_data["critic_token_usage"]
                    token_stats["critic_tokens"] += critic_usage.get("total_tokens", critic_usage.get("total", 0))
                if "judge_token_usage" in step_data:
                    judge_usage = step_data["judge_token_usage"]
                    token_stats["judge_tokens"] += judge_usage.get("total_tokens", judge_usage.get("total", 0))
            
            # 统计 ActSpec 调用情况
            actspec_calls = log_data.get("actspec_calls", [])
            if actspec_calls and len(actspec_calls) > 0:
                actspec_enabled = True  # 如果至少有一个日志文件包含actspec_calls，说明启用了test_mode
                for call_record in actspec_calls:
                    if not isinstance(call_record, dict):
                        continue
                    
                    actspec_stats["total_calls"] += 1
                    
                    # 判断成功/失败（与离线评估一致）：仅看是否未达步数上限且 executor 成功完成
                    # - executor_success 为 True → 成功；否则 → 失败（不再依据 post_condition_satisfied）
                    executor_success = call_record.get("executor_success")
                    reached_limit = call_record.get("reached_adjustment_limit")
                    if executor_success is True and reached_limit is not True:
                        actspec_stats["success_count"] += 1
                    else:
                        actspec_stats["fail_count"] += 1
        except Exception as e:
            continue
    
    # Token fallback：若各步只有 actor/critic/judge 细分而无 token_usage，用三者之和作为总 Token
    if token_stats["total_tokens"] == 0 and (
        token_stats["actor_tokens"] > 0 or token_stats["critic_tokens"] > 0 or token_stats["judge_tokens"] > 0
    ):
        token_stats["total_tokens"] = (
            token_stats["actor_tokens"] + token_stats["critic_tokens"] + token_stats["judge_tokens"]
        )
    
    # 计算 ActSpec 成功率
    if actspec_stats["total_calls"] > 0:
        actspec_stats["success_rate"] = (actspec_stats["success_count"] / actspec_stats["total_calls"] * 100)
    
    # 生成统计报告
    report = {
        "统计时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "任务完成率": {
            "总任务数": total_tasks,
            "成功任务数": int(success_count),
            "完成率": f"{completion_rate:.2f}%"
        },
        "Primitive Action 调用统计": {
            "总调用次数": total_actions,
            "各类型调用次数": dict(action_stats),
            "平均每任务调用次数": f"{(total_actions / total_tasks):.2f}" if total_tasks > 0 else "0"
        },
        "LLM Token 开销统计": {
            "总输入 Token": token_stats["total_input_tokens"],
            "总输出 Token": token_stats["total_output_tokens"],
            "总 Token": token_stats["total_tokens"],
            "Actor Token": token_stats["actor_tokens"],
            "Critic Token": token_stats["critic_tokens"],
            "Judge Token": token_stats["judge_tokens"],
            "平均每任务 Token": f"{(token_stats['total_tokens'] / total_tasks):.2f}" if total_tasks > 0 and token_stats["total_tokens"] > 0 else "0"
        }
    }
    
    # 如果启用了ActSpec test_mode，添加ActSpec统计
    if actspec_enabled:
        report["ActSpec执行统计"] = {
            "总调用次数": actspec_stats["total_calls"],
            "成功次数": actspec_stats["success_count"],
            "失败次数": actspec_stats["fail_count"],
            "成功率": f"{actspec_stats['success_rate']:.2f}%",
            "平均每任务调用次数": f"{(actspec_stats['total_calls'] / total_tasks):.2f}" if total_tasks > 0 else "0"
        }
    
    # 保存统计报告
    report_file = os.path.join(dstdir, "statistics_report.json")
    with open(report_file, "w", encoding="utf-8", errors='replace') as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    
    # 同时保存为可读的文本格式
    report_text_file = os.path.join(dstdir, "statistics_report.txt")
    with open(report_text_file, "w", encoding="utf-8", errors='replace') as f:
        f.write("=" * 80 + "\n")
        f.write("测试统计报告\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"统计时间: {report['统计时间']}\n\n")
        
        f.write("1. 任务完成率\n")
        f.write("-" * 80 + "\n")
        f.write(f"  总任务数: {report['任务完成率']['总任务数']}\n")
        f.write(f"  成功任务数: {report['任务完成率']['成功任务数']}\n")
        f.write(f"  完成率: {report['任务完成率']['完成率']}\n\n")
        
        f.write("2. Primitive Action 调用统计\n")
        f.write("-" * 80 + "\n")
        f.write(f"  总调用次数: {report['Primitive Action 调用统计']['总调用次数']}\n")
        f.write(f"  平均每任务调用次数: {report['Primitive Action 调用统计']['平均每任务调用次数']}\n")
        f.write("  各类型调用次数:\n")
        for action_type, count in sorted(report['Primitive Action 调用统计']['各类型调用次数'].items()):
            f.write(f"    {action_type}: {count}\n")
        f.write("\n")
        
        f.write("3. LLM Token 开销统计\n")
        f.write("-" * 80 + "\n")
        f.write(f"  总输入 Token: {report['LLM Token 开销统计']['总输入 Token']}\n")
        f.write(f"  总输出 Token: {report['LLM Token 开销统计']['总输出 Token']}\n")
        f.write(f"  总 Token: {report['LLM Token 开销统计']['总 Token']}\n")
        if report['LLM Token 开销统计']['Actor Token'] > 0:
            f.write(f"  Actor Token: {report['LLM Token 开销统计']['Actor Token']}\n")
        if report['LLM Token 开销统计']['Critic Token'] > 0:
            f.write(f"  Critic Token: {report['LLM Token 开销统计']['Critic Token']}\n")
        if report['LLM Token 开销统计']['Judge Token'] > 0:
            f.write(f"  Judge Token: {report['LLM Token 开销统计']['Judge Token']}\n")
        f.write(f"  平均每任务 Token: {report['LLM Token 开销统计']['平均每任务 Token']}\n")
        f.write("\n")
        
        # 如果启用了ActSpec test_mode，添加ActSpec统计
        if "ActSpec执行统计" in report:
            f.write("4. ActSpec执行统计\n")
            f.write("-" * 80 + "\n")
            f.write(f"  总调用次数: {report['ActSpec执行统计']['总调用次数']}\n")
            f.write(f"  成功次数: {report['ActSpec执行统计']['成功次数']}\n")
            f.write(f"  失败次数: {report['ActSpec执行统计']['失败次数']}\n")
            f.write(f"  成功率: {report['ActSpec执行统计']['成功率']}\n")
            f.write(f"  平均每任务调用次数: {report['ActSpec执行统计']['平均每任务调用次数']}\n")
            f.write("\n")
    
    print(f"[统计] 统计报告已保存到: {report_file}")
    print(f"[统计] 文本报告已保存到: {report_text_file}")
    print(f"[统计] 任务完成率: {completion_rate:.2f}% ({success_count}/{total_tasks})")
    print(f"[统计] 总 Action 调用次数: {total_actions}")
    print(f"[统计] 总 Token 使用: {token_stats['total_tokens']}")
    if actspec_enabled:
        print(f"[统计] ActSpec总调用次数: {actspec_stats['total_calls']} (成功: {actspec_stats['success_count']}, 失败: {actspec_stats['fail_count']}, 成功率: {actspec_stats['success_rate']:.2f}%)")

def run():
    parser = argparse.ArgumentParser(
        description="Evaluate AgentOccam on WebArena tasks"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="yaml config file location"
    )
    parser.add_argument(
        "--tasks-file", type=str, default=None,
        help="Path to JSON file containing task list (array of task objects). If provided, tasks will be loaded directly from this file and run without copying to Temp. Required when using --resume-from or --task-id."
    )
    parser.add_argument(
        "--resume-from", type=str, default=None,
        help="Task ID to resume from (e.g. 127). Used with --tasks-file: load the dataset, then run from this task id (inclusive) to the end. Example: --resume-from 127"
    )
    parser.add_argument(
        "--task-id", type=str, default=None,
        help="Run only this single task ID from the dataset. Must be used with --tasks-file. Example: --task-id 127"
    )
    parser.add_argument(
        "--agent-model", type=str, default=None,
        help="将 Actor、Critic、Judge 统一设为该模型（OpenRouter 格式，如 openai/gpt-5.4-mini）。优先级高于 config/config.yaml 中的 models 配置。"
    )
    parser.add_argument(
        "--logname", type=str, default=None,
        help="指定日志子目录名（相对于 config 中的 logdir），用于续跑同一轮实验：与上次运行的文件夹同名时会自动跳过已存在的 <task_id>.json。例：--logname 20260507-091939-298185",
    )
    args = parser.parse_args()

    # 启动终端输出自动保存：将本次运行的所有 print 实时写入 Temp 下以时间戳命名的 txt
    log_file, orig_stdout, orig_stderr = _start_console_logging()
    try:
        _run_impl(args, log_file, orig_stdout, orig_stderr)
    finally:
        _stop_console_logging(log_file, orig_stdout, orig_stderr)


def _run_impl(args, _console_log_file, _orig_stdout, _orig_stderr):
    """run() 的实际逻辑，便于在 finally 中可靠地关闭终端日志。"""
    # --resume-from 与 --task-id 必须与 --tasks-file 同时使用
    if (args.resume_from or args.task_id) and not args.tasks_file:
        raise ValueError("--resume-from 和 --task-id 必须与 --tasks-file 同时指定。请先使用 --tasks-file 指定数据集路径。")
    with open(args.config, "r", encoding='utf-8') as file:
        config = DotDict(yaml.safe_load(file))
    
    # 用统一配置覆盖 AgentOccam 配置文件中的相同配置项
    if _has_unified_config and _unified_config:
        config = _apply_unified_config_overrides(config, _unified_config)
    config = _apply_agent_model_cli_override(config, getattr(args, "agent_model", None))
    logname_cli = getattr(args, "logname", None)
    if logname_cli is not None and str(logname_cli).strip():
        config.logname = str(logname_cli).strip()
        if hasattr(config, "agent") and hasattr(config.agent, "others"):
            config.agent.others.logname = config.logname
        print(f"[配置] --logname 已生效: {config.logname}")
    
    # 初始化日志目录变量
    dstdir = None
    
    if config.logging:
        # 确保 logdir 存在
        os.makedirs(config.logdir, exist_ok=True)
        
        # 如果 logname 为空，使用时间戳创建子目录
        if config.logname:
            dstdir = os.path.join(config.logdir, config.logname)
        else:
            # 使用时间戳格式：YYYYMMDD-HHMMSS-ffffff（包含微秒以避免冲突）
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
            dstdir = os.path.join(config.logdir, timestamp)
        
        # 创建日志目录
        os.makedirs(dstdir, exist_ok=True)
        print(f"[日志] 日志将保存到: {os.path.abspath(dstdir)}")
        
        # 将日志目录路径传递到 agent.others 中，供 Actor/Critic/Judge 使用
        if hasattr(config, "agent") and hasattr(config.agent, "others"):
            config.agent.others.logdir_path = dstdir
    random.seed(42)
    
    config_file_list = []
    temp_dir = None
    task_configs_from_file = None  # 使用 --tasks-file 时存放任务 dict 列表，供循环中写入临时文件
    tasks_file_path = None         # 使用 --tasks-file 时的数据集路径，用于日志中的 task 字段

    # 如果提供了 --tasks-file 参数，直接从该文件加载任务（不复制到 Temp），并可配合 --task-id / --resume-from 筛选
    if args.tasks_file:
        if not os.path.exists(args.tasks_file):
            raise FileNotFoundError(f"Tasks file not found: {args.tasks_file}")
        print(f"[信息] 从数据集加载任务: {args.tasks_file}")
        with open(args.tasks_file, "r", encoding="utf-8") as f:
            tasks_data = json.load(f)
        if not isinstance(tasks_data, list):
            raise ValueError(f"Tasks file must contain a JSON array, got {type(tasks_data)}")
        url_mapping = load_url_mapping()
        tasks_data = replace_placeholders(tasks_data, url_mapping)

        # 按 task_id 排序，便于 --resume-from 截取
        def _task_id_key(t):
            tid = t.get("task_id")
            if isinstance(tid, int):
                return tid
            if isinstance(tid, str) and tid.isdigit():
                return int(tid)
            return 0
        tasks_data.sort(key=_task_id_key)

        # --task-id: 只跑指定任务
        if args.task_id:
            try:
                tid = int(args.task_id)
            except ValueError:
                tid = args.task_id
            filtered = [t for t in tasks_data if t.get("task_id") == tid or t.get("task_id") == args.task_id]
            if not filtered:
                raise ValueError(f"--task-id {args.task_id}: 数据集中未找到 task_id={args.task_id} 的任务")
            tasks_data = filtered
            print(f"[信息] 仅运行任务: {args.task_id}")
        # --resume-from: 从该 task_id（含）开始跑到最后
        elif args.resume_from:
            try:
                resume_id = int(args.resume_from)
            except ValueError:
                resume_id = args.resume_from
            start_idx = None
            for i, t in enumerate(tasks_data):
                tt = t.get("task_id")
                if isinstance(tt, str) and tt.isdigit():
                    tt = int(tt)
                if tt == resume_id or (isinstance(tt, int) and isinstance(resume_id, int) and tt >= resume_id):
                    start_idx = i
                    break
            if start_idx is None:
                raise ValueError(f"--resume-from {args.resume_from}: 数据集中未找到 task_id>={args.resume_from} 的任务")
            tasks_data = tasks_data[start_idx:]
            print(f"[信息] 从 task_id={args.resume_from} 开始，共 {len(tasks_data)} 个任务")
        else:
            print(f"[信息] 运行全部 {len(tasks_data)} 个任务")

        task_configs_from_file = list(tasks_data)
        tasks_file_path = args.tasks_file
        # 使用单个临时文件，循环中每次写入当前任务（供 env 读取），不复制整个数据集到 Temp
        current_dir = Path(__file__).resolve().parent
        main_dir = current_dir.parent
        temp_base_dir = main_dir / "Temp"
        os.makedirs(temp_base_dir, exist_ok=True)
        temp_dir = tempfile.mkdtemp(prefix="eval_tasks_", dir=str(temp_base_dir))
        single_task_path = os.path.join(temp_dir, "current_task.json")
        config_file_list = [single_task_path] * len(task_configs_from_file)
        print(f"[信息] 已加载 {len(config_file_list)} 个任务")

    # 验证ActSpec配置：auto_generate 和 test_mode.enabled 互斥
    actspec_config = getattr(config, 'actspec', {})
    auto_generate = actspec_config.get('auto_generate', False)
    test_mode_enabled = actspec_config.get('test_mode', {}).get('enabled', False)
    
    if auto_generate and test_mode_enabled:
        raise ValueError(
            "ActSpec配置错误：auto_generate 和 test_mode.enabled 不能同时为 true。\n"
            "请选择一种模式：\n"
            "  - 训练模式：设置 auto_generate=true, test_mode.enabled=false\n"
            "  - 测试模式：设置 auto_generate=false, test_mode.enabled=true"
        )
    
    # 根据配置确定模式
    is_training_mode = auto_generate
    is_testing_mode = test_mode_enabled
    
    # 将模式信息存储到config中，供后续使用
    config._actspec_training_mode = is_training_mode
    config._actspec_testing_mode = is_testing_mode
    
    if actspec_config.get('enabled', False):
        mode_str = "训练模式" if is_training_mode else ("测试模式" if is_testing_mode else "未启用")
        print(f"[ActSpec] 当前模式: {mode_str} (auto_generate={auto_generate}, test_mode.enabled={test_mode_enabled})")

    # Global primitive budget B across tasks (paper-aligned optional setting)
    global_primitive_budget = int(actspec_config.get("global_primitive_budget", 0) or 0)
    remaining_primitive_budget = global_primitive_budget if global_primitive_budget > 0 else None
    if remaining_primitive_budget is not None:
        print(f"[Budget] Global primitive budget B={remaining_primitive_budget}")

    if not config_file_list:
        # 原有的从 config_files 目录加载任务的逻辑
        task_ids = config.env.task_ids
        if hasattr(config.env, "relative_task_dir"):
            relative_task_dir = config.env.relative_task_dir
        else:
            relative_task_dir = "tasks"
        if task_ids == "all" or task_ids == ["all"]:
            task_ids = [filename[:-len(".json")] for filename in os.listdir(f"config_files/{relative_task_dir}") if filename.endswith(".json")]
        
        # 加载URL映射用于替换占位符
        url_mapping = load_url_mapping()
        
        # 创建临时目录存储替换后的任务配置文件
        current_dir = Path(__file__).resolve().parent
        main_dir = current_dir.parent  # 主目录（WebAct-demo）
        temp_base_dir = main_dir / "Temp"
        os.makedirs(temp_base_dir, exist_ok=True)
        temp_dir = tempfile.mkdtemp(prefix="config_files_", dir=str(temp_base_dir))
        
        for task_id in task_ids:
            config_file_path = f"config_files/{relative_task_dir}/{task_id}.json"
            # 读取配置文件并替换占位符
            with open(config_file_path, "r", encoding="utf-8") as f:
                task_config = json.load(f)
            # 替换占位符
            task_config = replace_placeholders(task_config, url_mapping)
            # 写入临时文件
            safe_task_id = str(task_id).replace("/", "_").replace("\\", "_")
            temp_config_file = os.path.join(temp_dir, f"{safe_task_id}.json")
            with open(temp_config_file, "w", encoding="utf-8", errors='replace') as f:
                json.dump(task_config, f, indent=2, ensure_ascii=False)
            config_file_list.append(temp_config_file)

    fullpage = config.env.fullpage if hasattr(config.env, "fullpage") else True
    current_viewport_only = not fullpage

    if config.agent.type == "AgentOccam":
        agent_init = lambda: AgentOccam(
            prompt_dict = {k: v for k, v in AgentOccam_prompt.__dict__.items() if isinstance(v, dict)},
            config = config.agent,
        )
    elif config.agent.type == "AgentOccam-SteP":
            agent_init = lambda: StepAgent(
            root_action = config.agent.root_action,
            action_to_prompt_dict = {k: v for k, v in step_fewshot_template_adapted.__dict__.items() if isinstance(v, dict)},
            low_level_action_list = config.agent.low_level_action_list,
            max_actions=config.env.max_env_steps,
            verbose=config.verbose,
            logging=config.logging,
            debug=config.debug,
            model=config.agent.model_name,
            prompt_mode=config.agent.prompt_mode,
            )    
    elif config.agent.type == "SteP-replication":
        agent_init = lambda: StepAgent(
            root_action = config.agent.root_action,
            action_to_prompt_dict = {k: v for k, v in step_fewshot_template.__dict__.items() if isinstance(v, dict)},
            low_level_action_list = config.agent.low_level_action_list,
            max_actions=config.env.max_env_steps,
            verbose=config.verbose,
            logging=config.logging,
            debug=config.debug,
            model=config.agent.model_name,
            prompt_mode=config.agent.prompt_mode,
        )
    else:
        raise NotImplementedError(f"{config.agent.type} not implemented")

    
    for i, config_file in enumerate(config_file_list):
        if remaining_primitive_budget is not None and remaining_primitive_budget <= 0:
            print("[Budget] Global primitive budget exhausted; stop remaining tasks.")
            break
        # 使用 --tasks-file 时，每次将当前任务写入临时文件供 env 读取
        if task_configs_from_file is not None:
            with open(config_file, "w", encoding="utf-8", errors="replace") as f:
                json.dump(task_configs_from_file[i], f, indent=2, ensure_ascii=False)
        with open(config_file, "r", encoding='utf-8') as f:
            task_config = json.load(f)
            print(f"Task {task_config['task_id']}.")
        # 如果启用了日志，检查是否已存在该任务的日志文件
        if config.logging and os.path.exists(os.path.join(dstdir, f"{task_config['task_id']}.json")):
            print(f"Skip {task_config['task_id']}.")
            continue
        if task_config['task_id'] in list(range(600, 650))+list(range(681, 689)):
            print("Reddit post task. Sleep 30s.")
            time.sleep(30)
        env = WebArenaEnvironmentWrapper(config_file=config_file, 
                                        max_browser_rows=config.env.max_browser_rows, 
                                        max_steps=config.max_steps, 
                                        slow_mo=1, 
                                        observation_type="accessibility_tree", 
                                        current_viewport_only=current_viewport_only, 
                                        viewport_size={"width": 1920, "height": 1080}, 
                                        headless=config.env.headless,
                                        global_config=config,
                                        llm_config=_unified_config if _has_unified_config else None)
        
        agent = agent_init()
        objective = env.get_objective()
        
        # 应用action级别的timeout控制（单个action的最大等待时间）
        action_timeout = getattr(config, 'task_timeout', 300)  # 默认5分钟，作为单个action的超时时间
        timeout_occurred = [False]  # 使用列表以便在嵌套函数中修改，每个任务开始时重置
        status = None  # 初始化status变量
        
        # 确保环境状态被重置（避免前一个任务的影响）
        env.is_done = False
        if hasattr(env, 'timeout_occurred'):
            env.timeout_occurred[0] = False
        
        # 将超时时间和标志传递给环境，以便在操作中检查
        env.action_timeout = action_timeout
        env.timeout_occurred = timeout_occurred
        # 同时通过 global_config 传递超时时间和标志，以便浏览器环境访问
        if hasattr(config, 'env'):
            if not hasattr(config.env, 'action_timeout'):
                config.env.action_timeout = action_timeout
            if not hasattr(config.env, 'timeout_occurred'):
                config.env.timeout_occurred = timeout_occurred
            else:
                # 重置超时标志
                config.env.timeout_occurred[0] = False
        
        try:
            status = agent.act(objective=objective, env=env)
        except KeyboardInterrupt:
            # 处理用户中断
            print(f"[错误] 任务 {task_config['task_id']} 被用户中断")
            if timeout_occurred[0]:
                status = {"success": 0, "reward": 0, "done": True, "timeout": True}
            else:
                raise
        except TimeoutError as e:
            # 处理action超时
            print(f"[错误] 任务 {task_config['task_id']} 因action超时被中断: {e}")
            timeout_occurred[0] = True
            status = {"success": 0, "reward": 0, "done": True, "timeout": True}
        except Exception as e:
            if timeout_occurred[0]:
                print(f"[错误] 任务 {task_config['task_id']} 因超时中断: {e}")
                status = {"success": 0, "reward": 0, "done": True, "timeout": True}
            else:
                # 非超时异常，重新抛出
                print(f"[错误] 任务 {task_config['task_id']} 执行出错: {e}")
                raise
        finally:
            # 如果超时，更新status
            if timeout_occurred[0]:
                if status is None:
                    status = {}
                elif not isinstance(status, dict):
                    status = {}
                status["timeout"] = True
                status["done"] = True
                if "success" not in status:
                    status["success"] = 0
                if "reward" not in status:
                    status["reward"] = 0
                print(f"[信息] 任务 {task_config['task_id']} 已标记为超时，继续下一个任务")
            
            # 确保环境被关闭，强制清理所有资源
            try:
                # 先尝试正常关闭
                env.close()
            except Exception as e:
                # 捕获并忽略 greenlet 错误和其他清理错误
                # 这些错误通常发生在超时后环境已损坏的情况下
                error_str = str(e).lower()
                if "greenlet" in error_str or "cannot switch" in error_str:
                    # 忽略 greenlet 错误，这是预期的超时后行为
                    pass
                else:
                    print(f"[警告] 关闭环境时出错: {e}")
                # 不再尝试强制关闭 page.client，因为这会导致 greenlet 错误
                # 环境已经标记为损坏，会在下次任务时重新创建
            
            # 确保status不为None
            if status is None:
                status = env.status() if hasattr(env, 'status') else {"success": 0, "reward": 0, "done": True}

        if config.logging:
            # 使用 --tasks-file 时，日志中 task 字段记录为 数据集路径#task_id，便于溯源
            task_ref = f"{tasks_file_path}#{task_config['task_id']}" if tasks_file_path else config_file
            log_file = os.path.join(dstdir, f"{task_config['task_id']}.json")
            trajectory = agent.get_trajectory()
            
            # 提取统计信息
            action_stats = extract_primitive_actions(trajectory)
            token_stats = extract_token_usage(trajectory, agent)
            if remaining_primitive_budget is not None:
                used_primitives = int(sum(action_stats.values()))
                remaining_primitive_budget -= used_primitives
                print(
                    f"[Budget] task={task_config['task_id']} used={used_primitives}, "
                    f"remaining={max(remaining_primitive_budget, 0)}"
                )
            
            log_data = {
                "task": task_ref,
                "id": task_config['task_id'],
                "model": config.agent.actor.model if hasattr(config.agent, "actor") else config.agent.model_name,
                "type": config.agent.type,
                "trajectory": trajectory,
                # 记录环境侧的 ActSpec 调用信息，供离线复用评估使用
                "actspec_calls": getattr(env, "actspec_call_records", []),
                "statistics": {
                    "action_counts": action_stats,
                    "token_usage": token_stats,
                }
            }
            summary_file = os.path.join(dstdir, "summary.csv")
            # 使用 os.path 提取相对路径，确保跨平台兼容性
            logfile_rel_path = os.path.relpath(log_file, dstdir).replace("\\", "/")
            summary_data = {
                "task": task_ref,
                "task_id": task_config['task_id'],
                "model": config.agent.actor.model if hasattr(config.agent, "actor") else config.agent.model_name,
                "type": config.agent.type,
                "logfile": logfile_rel_path,
            }
            if status:
                summary_data.update(status)
            
            # 添加统计信息到 summary_data
            summary_data["total_actions"] = sum(action_stats.values())
            summary_data["total_tokens"] = token_stats["total_tokens"]
            for action_type, count in action_stats.items():
                summary_data[f"action_{action_type}"] = count
            
            print(f"[eval_webarena] 准备调用 log_run，task_id={task_config['task_id']}")
            print(f"[eval_webarena] log_file={log_file}")
            print(f"[eval_webarena] log_data 类型: {type(log_data)}")
            if isinstance(log_data, dict):
                print(f"[eval_webarena] log_data 包含的键: {list(log_data.keys())}")
            print(f"[eval_webarena] summary_data 类型: {type(summary_data)}")
            if isinstance(summary_data, dict):
                print(f"[eval_webarena] summary_data 包含的键: {list(summary_data.keys())}")
            
            log_run(
                log_file=log_file,
                log_data=log_data,
                summary_file=summary_file,
                summary_data=summary_data,
            )
            
            print(f"[eval_webarena] log_run 调用完成，task_id={task_config['task_id']}")
            
            # 训练阶段：在独立线程中生成 ActSpec（避免阻塞主评测流程）
            def _async_generate_actspec(
                task_cfg_snapshot,
                trajectory_snapshot,
                config_snapshot,
            ):
                try:
                    actspec_enabled = getattr(config_snapshot, 'actspec', {}).get('enabled', False)
                    actspec_auto_generate = getattr(config_snapshot, 'actspec', {}).get('auto_generate', False)
                    is_training = getattr(config_snapshot, '_actspec_training_mode', False)
                    
                    if not (actspec_enabled and actspec_auto_generate and is_training):
                        return
                    
                    print(f"[ActSpec][异步] 开始为任务 {task_cfg_snapshot['task_id']} 生成ActSpec...")
                    from actspec import TraceSegmenter, ActSpecGenerator, ActSpecLibrary
                    from llms import lm_config
                    
                    actspec_config = getattr(config_snapshot, 'actspec', {})
                    library_path = actspec_config.get('library_path', 'temp_library')
                    
                    if not hasattr(config_snapshot, '_actspec_timestamp'):
                        config_snapshot._actspec_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    timestamp_local = config_snapshot._actspec_timestamp
                    library_full_path = os.path.join(library_path, timestamp_local)
                    
                    llm_config_dict = actspec_config.get('llm_config', {})
                    llm_cfg = lm_config.LMConfig(
                        provider=llm_config_dict.get('provider', 'openai'),
                        model=llm_config_dict.get('model', 'gpt-4-turbo'),
                        mode='chat',
                        gen_config={
                            'temperature': llm_config_dict.get('temperature', 0.1),
                            'max_tokens': llm_config_dict.get('max_tokens', 3000),
                            'top_p': llm_config_dict.get('top_p', 1.0),
                        }
                    )
                    
                    segmenter = TraceSegmenter(llm_cfg)
                    generator = ActSpecGenerator(llm_cfg)
                    library = ActSpecLibrary(library_path)
                    
                    if not trajectory_snapshot or not isinstance(trajectory_snapshot, list):
                        print(f"[ActSpec][异步] 无效的trajectory格式，跳过ActSpec生成")
                        return
                    if len(trajectory_snapshot) == 0:
                        print(f"[ActSpec][异步] 空的trajectory，跳过ActSpec生成")
                        return
                    
                    segments = segmenter.segment_trajectory(trajectory_snapshot, task_cfg_snapshot)
                    print(f"[ActSpec][异步] 切分出 {len(segments)} 个segment")
                    if not segments:
                        print(f"[ActSpec][异步] 未切分出有效的segment，跳过ActSpec生成")
                        return
                    
                    actspec_count = 0
                    for seg_idx, segment in enumerate(segments):
                        try:
                            if not isinstance(segment, dict):
                                print(f"[ActSpec][异步] Segment {seg_idx} 格式无效，跳过")
                                continue
                            
                            actions = segment.get('actions', [])
                            context = segment.get('context', {})
                            
                            if not actions or not isinstance(actions, list):
                                print(f"[ActSpec][异步] Segment {seg_idx} 没有有效的actions，跳过")
                                continue
                            
                            if not isinstance(context, dict):
                                print(f"[ActSpec][异步] Segment {seg_idx} context格式无效，使用默认值")
                                context = {"site": "unknown", "page": "unknown", "url": ""}
                            
                            actspec = generator.generate_actspec(
                                actions,
                                context,
                                task_cfg_snapshot
                            )
                            
                            if not actspec or not isinstance(actspec, dict):
                                print(f"[ActSpec][异步] Segment {seg_idx} 生成的ActSpec无效，跳过")
                                continue
                            
                            library.save_actspec(actspec, library_full_path)
                            actspec_count += 1
                            print(f"[ActSpec][异步] 已生成ActSpec: {actspec.get('action_id', 'unknown')}")
                        except Exception as e:
                            print(f"[ActSpec][异步] Segment {seg_idx} 生成ActSpec失败: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    print(f"[ActSpec][异步] 任务 {task_cfg_snapshot['task_id']} 共生成 {actspec_count}/{len(segments)} 个ActSpec，保存到 {library_full_path}")
                except Exception as e:
                    print(f"[警告][ActSpec][异步] 生成过程出错: {e}")
                    import traceback
                    traceback.print_exc()

            # 启动训练阶段的异步 ActSpec 生成线程
            try:
                threading.Thread(
                    target=_async_generate_actspec,
                    args=(task_config, trajectory, config),
                    daemon=True,
                ).start()
            except Exception as e:
                print(f"[警告] 启动异步ActSpec生成线程失败: {e}")

            # 测试阶段：在独立线程中做 ActSpec 复用离线评估 + 置信度更新
            def _async_evaluate_actspec_reuse(log_file_path, config_snapshot):
                try:
                    actspec_cfg = getattr(config_snapshot, 'actspec', {})
                    enabled = actspec_cfg.get('enabled', False)
                    test_mode = actspec_cfg.get('test_mode', {}).get('enabled', False)
                    is_testing = getattr(config_snapshot, '_actspec_testing_mode', False)
                    if not (enabled and test_mode and is_testing):
                        return
                    
                    library_path_local = actspec_cfg.get('test_mode', {}).get('library_path', '') or actspec_cfg.get('library_path', 'temp_library')
                    convert_to_negative_constraints = actspec_cfg.get('convert_failures_to_negative_constraints', True)
                    
                    from actspec.actspec_offline_evaluator import evaluate_and_update_library_for_log
                    from llms import lm_config
                    
                    llm_cfg_dict = actspec_cfg.get('llm_config', {})
                    llm_cfg = lm_config.LMConfig(
                        provider=llm_cfg_dict.get('provider', 'openai'),
                        model=llm_cfg_dict.get('model', 'gpt-4-turbo'),
                        mode='chat',
                        gen_config={
                            'temperature': llm_cfg_dict.get('temperature', 0.1),
                            'max_tokens': llm_cfg_dict.get('max_tokens', 2000),
                            'top_p': llm_cfg_dict.get('top_p', 1.0),
                        }
                    )
                    
                    print(f"[ActSpec][异步] 开始对日志 {log_file_path} 进行离线复用评估...")
                    evaluate_and_update_library_for_log(
                        log_file=log_file_path,
                        library_path=library_path_local,
                        llm_cfg=llm_cfg,
                        convert_to_negative_constraints=convert_to_negative_constraints,
                    )
                    print(
                        f"[ActSpec][异步] 日志 {log_file_path} 的离线复用评估完成，并已更新库。"
                        f" convert_to_negative_constraints={convert_to_negative_constraints}"
                    )
                except Exception as e:
                    print(f"[警告][ActSpec][异步] 离线复用评估出错: {e}")
                    import traceback
                    traceback.print_exc()

            try:
                threading.Thread(
                    target=_async_evaluate_actspec_reuse,
                    args=(log_file, config),
                    daemon=True,
                ).start()
            except Exception as e:
                print(f"[警告] 启动异步ActSpec复用评估线程失败: {e}")
            
            # 每个任务执行完成后立即更新统计报告
            print(f"[统计] 任务 {task_config['task_id']} 完成，更新统计报告...")
            try:
                generate_statistics_report(dstdir)
            except Exception as e:
                print(f"[警告] 更新统计报告失败: {e}")
    
    # 清理临时目录
    if temp_dir and os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            print(f"[信息] 已清理临时目录: {temp_dir}")
        except Exception as e:
            print(f"[警告] 清理临时目录失败: {e}")
    
    # 最终生成一次统计报告（确保所有任务都包含在内）
    if config.logging and dstdir:
        print("\n[统计] 生成最终统计报告...")
        try:
            generate_statistics_report(dstdir)
        except Exception as e:
            print(f"[警告] 生成最终统计报告失败: {e}")
    
if __name__ == "__main__":
    run()
