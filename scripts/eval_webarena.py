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

from AgentCore.env import WebArenaEnvironmentWrapper

from AgentCore.AgentCore import AgentCore
from webagents_step.utils.data_prep import *
from webagents_step.agents.step_agent import StepAgent

from AgentCore.prompts import AgentPrompt
from webagents_step.prompts.webarena import step_fewshot_template_adapted, step_fewshot_template

from AgentCore.utils import EVALUATOR_DIR


class TeeOutput:
    """Mirror sys.stdout/sys.stderr to both console and file in real time."""

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
    Create a timestamped txt log under Temp and redirect stdout/stderr to it,
    while still preserving console output.
    Return (log_file_handle, original_stdout, original_stderr) for restoration.
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
        print(f"[Warning] Failed to create terminal log file {log_path}: {e}", file=sys.__stderr__)
        return None, None, None
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeOutput(sys.stdout, log_file)
    sys.stderr = TeeOutput(sys.stderr, log_file)
    
    print(f"[TerminalLog] Start time: {datetime.now().isoformat()}, log file: {log_path}")
    return log_file, original_stdout, original_stderr


def _stop_console_logging(log_file, original_stdout, original_stderr):
    """Restore stdout/stderr and close the terminal log file."""
    if log_file is None:
        return
    try:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()
    except Exception:
        pass



def load_url_mapping():
    """Load URL mappings from Webarena_website.json."""
    current_dir = Path(__file__).resolve().parent
    url_mapping_file = current_dir / "Webarena_website.json"
    if not url_mapping_file.exists():
        raise FileNotFoundError(f"URL mapping file not found: {url_mapping_file}")
    
    with open(url_mapping_file, "r", encoding="utf-8") as f:
        url_mapping = json.load(f)
    
    return url_mapping

def replace_placeholders(data, url_mapping):
    """
    Recursively replace placeholders in data (format: __XXX__).
    Each placeholder is mapped to XXX_URL from url_mapping.
    """
    if isinstance(data, dict):
        return {key: replace_placeholders(value, url_mapping) for key, value in data.items()}
    elif isinstance(data, list):
        return [replace_placeholders(item, url_mapping) for item in data]
    elif isinstance(data, str):
        result = data
        
        placeholder_to_url = {}
        for placeholder_key, url_value in url_mapping.items():
            
            if placeholder_key.endswith("_URL"):
                placeholder_name = placeholder_key[:-4]  
                placeholder_pattern = f"__{placeholder_name}__"
                placeholder_to_url[placeholder_pattern] = url_value
        
        
        sorted_patterns = sorted(placeholder_to_url.keys(), key=len, reverse=True)
        
        
        for placeholder_pattern in sorted_patterns:
            url_value = placeholder_to_url[placeholder_pattern]
            if placeholder_pattern in result:
                
                if result == placeholder_pattern:
                    result = url_value
                
                elif result.startswith(placeholder_pattern + "/"):
                    path = result[len(placeholder_pattern):]
                    
                    if url_value.endswith("/"):
                        result = url_value.rstrip("/") + path
                    else:
                        result = url_value + path
                
                elif result.startswith(placeholder_pattern):
                    remaining = result[len(placeholder_pattern):]
                    result = url_value + remaining
                else:
                    
                    result = result.replace(placeholder_pattern, url_value)
        return result
    else:
        return data


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
    Override AgentCore config values with unified config values.
    Unified config (config/config.yaml) has higher priority than
    agent_core/configs/*.yml.
    """
    
    if hasattr(config, "agent") and config.agent.type == "Agent":
        
        if hasattr(config.agent, "actor"):
            actor_model = unified_config.get_model("agent_actor", "default")
            if actor_model:
                config.agent.actor.model = actor_model
        
        
        if hasattr(config.agent, "critic"):
            critic_model = unified_config.get_model("agent_critic", "default")
            if critic_model:
                config.agent.critic.model = critic_model
        
        
        if hasattr(config.agent, "judge"):
            judge_model = unified_config.get_model("agent_judge", "default")
            if judge_model:
                config.agent.judge.model = judge_model
    
    
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
        
        
        if hasattr(config, "agent") and hasattr(config.agent, "others"):
            if "logname" in logging_config:
                config.agent.others.logname = logging_config["logname"]
            if "enabled" in logging_config:
                config.agent.others.logging = logging_config["enabled"]
            if "verbose" in logging_config:
                config.agent.others.verbose = logging_config["verbose"]
            if "debug" in logging_config:
                config.agent.others.debug = logging_config["debug"]
    
    
    default_task_config = unified_config.get("default_task", {})
    if default_task_config:
        if "max_steps" in default_task_config:
            
            config.max_steps = default_task_config["max_steps"]
            
            if hasattr(config, "agent") and hasattr(config.agent, "others"):
                config.agent.others.max_steps = default_task_config["max_steps"]
        if "relative_task_dir" in default_task_config:
            if hasattr(config, "env"):
                config.env.relative_task_dir = default_task_config["relative_task_dir"]
        if "timeout" in default_task_config:
            
            config.task_timeout = default_task_config["timeout"]
    
    
    actspec_config = unified_config.get("actspec", {})
    if actspec_config:
        
        config.actspec = actspec_config
    
    return config


def _apply_agent_model_cli_override(config: DotDict, model: Optional[str]) -> DotDict:
    """
    CLI --agent-model sets Actor/Critic/Judge to one model.
    This has higher priority than models.agent_* in config/config.yaml.
    """
    if not model or not str(model).strip():
        return config
    model = str(model).strip()
    if hasattr(config, "agent") and getattr(config.agent, "type", None) == "Agent":
        for role in ("actor", "critic", "judge"):
            if hasattr(config.agent, role):
                getattr(config.agent, role).model = model
        print(f"[Config] --agent-model applied: Actor / Critic / Judge -> {model}")
    return config


def extract_primitive_actions(trajectory):
    """
    Extract primitive action counts from trajectory.
    Returns a dict with invocation counts per action type.
    Note: actions executed internally by ActSpec are filtered by _actspec_internal.
    Supported step formats:
    1. action_type field (mapped via ActionTypes)
    2. action string (e.g. "click [607]", "type [123] hello")
    """
    from browser_env.actions import ActionTypes
    
    action_counter = Counter()
    
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
        """Parse action type from action string, e.g. 'click [607]' -> 'click'. Return None if unmatched."""
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
        
        if isinstance(step_data, dict) and step_data.get("_actspec_internal", False):
            continue
        if not isinstance(step_data, dict):
            continue
        
        matched = False
        
        if "action_type" in step_data:
            action_type = step_data.get("action_type")
            if action_type in ACTION_TYPE_MAP:
                action_name = ACTION_TYPE_MAP[action_type]
                action_counter[action_name] += 1
                matched = True
        
        
        if not matched and "action" in step_data:
            action_name = parse_action_string(step_data["action"])
            if action_name:
                action_counter[action_name] += 1
    
    return dict(action_counter)

def extract_token_usage(trajectory, agent):
    """
    Extract token usage from trajectory and agent.
    Returns a dictionary with aggregated token usage stats.
    """
    token_stats = {
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_tokens": 0,
        "actor_tokens": 0,
        "critic_tokens": 0,
        "judge_tokens": 0,
    }
    
    
    for step_data in trajectory:
        
        if "token_usage" in step_data:
            usage = step_data["token_usage"]
            token_stats["total_input_tokens"] += usage.get("input_tokens", 0)
            token_stats["total_output_tokens"] += usage.get("output_tokens", 0)
            token_stats["total_tokens"] += usage.get("total_tokens", 0)
        
        
        if "actor_token_usage" in step_data:
            actor_usage = step_data["actor_token_usage"]
            token_stats["actor_tokens"] += actor_usage.get("total_tokens", 0)
        
        if "critic_token_usage" in step_data:
            critic_usage = step_data["critic_token_usage"]
            token_stats["critic_tokens"] += critic_usage.get("total_tokens", 0)
        
        if "judge_token_usage" in step_data:
            judge_usage = step_data["judge_token_usage"]
            token_stats["judge_tokens"] += judge_usage.get("total_tokens", 0)
    
    
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
    Generate a statistics report including:
    1. Task completion
    2. Primitive action invocation stats
    3. LLM token usage stats
    4. ActSpec execution stats (if test_mode enabled)
    """
    summary_file = os.path.join(dstdir, "summary.csv")
    if not os.path.exists(summary_file):
        print("[Warning] summary.csv not found; cannot generate statistics report")
        return
    
    
    try:
        df_summary = pd.read_csv(summary_file)
    except Exception as e:
        print(f"[Error] Failed to read summary.csv: {e}")
        return
    
    
    total_tasks = len(df_summary)
    if total_tasks == 0:
        print("[Warning] No task data in summary.csv")
        return
    
    
    if "success" in df_summary.columns:
        success_col = df_summary["success"]
        
        if success_col.dtype in [int, float]:
            success_count = int(success_col.sum())
        elif success_col.dtype == bool:
            success_count = int(success_col.sum())
        else:
            
            success_count = sum(1 for x in success_col if 
                              (isinstance(x, (int, float)) and x > 0) or 
                              (isinstance(x, str) and x.lower() in ['true', '1', '1.0', 'yes']))
    else:
        
        success_count = 0
        if "reward" in df_summary.columns:
            reward_col = df_summary["reward"]
            success_count = sum(1 for x in reward_col if (isinstance(x, (int, float)) and x > 0))
    
    completion_rate = (success_count / total_tasks * 100) if total_tasks > 0 else 0
    
    
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
            print(f"[Warning] Failed to read log file {log_file_path}: {e}")
            continue
    
    
    token_stats = {
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_tokens": 0,
        "actor_tokens": 0,
        "critic_tokens": 0,
        "judge_tokens": 0,
    }
    
    
    actspec_stats = {
        "total_calls": 0,
        "success_count": 0,
        "fail_count": 0,
        "success_rate": 0.0,
    }
    actspec_enabled = False  
    
    
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
                
                if "actor_token_usage" in step_data:
                    actor_usage = step_data["actor_token_usage"]
                    token_stats["actor_tokens"] += actor_usage.get("total_tokens", actor_usage.get("total", 0))
                if "critic_token_usage" in step_data:
                    critic_usage = step_data["critic_token_usage"]
                    token_stats["critic_tokens"] += critic_usage.get("total_tokens", critic_usage.get("total", 0))
                if "judge_token_usage" in step_data:
                    judge_usage = step_data["judge_token_usage"]
                    token_stats["judge_tokens"] += judge_usage.get("total_tokens", judge_usage.get("total", 0))
            
            
            actspec_calls = log_data.get("actspec_calls", [])
            if actspec_calls and len(actspec_calls) > 0:
                actspec_enabled = True  
                for call_record in actspec_calls:
                    if not isinstance(call_record, dict):
                        continue
                    
                    actspec_stats["total_calls"] += 1
                    
                    
                    
                    executor_success = call_record.get("executor_success")
                    reached_limit = call_record.get("reached_adjustment_limit")
                    if executor_success is True and reached_limit is not True:
                        actspec_stats["success_count"] += 1
                    else:
                        actspec_stats["fail_count"] += 1
        except Exception as e:
            continue
    
    
    if token_stats["total_tokens"] == 0 and (
        token_stats["actor_tokens"] > 0 or token_stats["critic_tokens"] > 0 or token_stats["judge_tokens"] > 0
    ):
        token_stats["total_tokens"] = (
            token_stats["actor_tokens"] + token_stats["critic_tokens"] + token_stats["judge_tokens"]
        )
    
    
    if actspec_stats["total_calls"] > 0:
        actspec_stats["success_rate"] = (actspec_stats["success_count"] / actspec_stats["total_calls"] * 100)
    
    
    report = {
        "report_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "task_completion": {
            "total_tasks": total_tasks,
            "successful_tasks": int(success_count),
            "completion_rate": f"{completion_rate:.2f}%"
        },
        "primitive_action_stats": {
            "total_invocations": total_actions,
            "invocations_by_type": dict(action_stats),
            "avg_invocations_per_task": f"{(total_actions / total_tasks):.2f}" if total_tasks > 0 else "0"
        },
        "llm_token_usage": {
            "total_input_tokens": token_stats["total_input_tokens"],
            "total_output_tokens": token_stats["total_output_tokens"],
            "total_tokens": token_stats["total_tokens"],
            "actor_tokens": token_stats["actor_tokens"],
            "critic_tokens": token_stats["critic_tokens"],
            "judge_tokens": token_stats["judge_tokens"],
            "avg_tokens_per_task": f"{(token_stats['total_tokens'] / total_tasks):.2f}" if total_tasks > 0 and token_stats["total_tokens"] > 0 else "0"
        }
    }
    
    
    if actspec_enabled:
        report["actspec_execution_stats"] = {
            "total_invocations": actspec_stats["total_calls"],
            "successful_invocations": actspec_stats["success_count"],
            "failed_invocations": actspec_stats["fail_count"],
            "success_rate": f"{actspec_stats['success_rate']:.2f}%",
            "avg_invocations_per_task": f"{(actspec_stats['total_calls'] / total_tasks):.2f}" if total_tasks > 0 else "0"
        }
    
    
    report_file = os.path.join(dstdir, "statistics_report.json")
    with open(report_file, "w", encoding="utf-8", errors='replace') as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    
    
    report_text_file = os.path.join(dstdir, "statistics_report.txt")
    with open(report_text_file, "w", encoding="utf-8", errors='replace') as f:
        f.write("=" * 80 + "\n")
        f.write("Evaluation Statistics Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Report Time: {report['report_time']}\n\n")
        
        f.write("1. Task Completion\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Total Tasks: {report['task_completion']['total_tasks']}\n")
        f.write(f"  Successful Tasks: {report['task_completion']['successful_tasks']}\n")
        f.write(f"  Completion Rate: {report['task_completion']['completion_rate']}\n\n")
        
        f.write("2. Primitive Action Stats\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Total Invocations: {report['primitive_action_stats']['total_invocations']}\n")
        f.write(f"  Avg Invocations per Task: {report['primitive_action_stats']['avg_invocations_per_task']}\n")
        f.write("  Invocations by Type:\n")
        for action_type, count in sorted(report['primitive_action_stats']['invocations_by_type'].items()):
            f.write(f"    {action_type}: {count}\n")
        f.write("\n")
        
        f.write("3. LLM Token Usage\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Total Input Tokens: {report['llm_token_usage']['total_input_tokens']}\n")
        f.write(f"  Total Output Tokens: {report['llm_token_usage']['total_output_tokens']}\n")
        f.write(f"  Total Tokens: {report['llm_token_usage']['total_tokens']}\n")
        if report['llm_token_usage']['actor_tokens'] > 0:
            f.write(f"  Actor Tokens: {report['llm_token_usage']['actor_tokens']}\n")
        if report['llm_token_usage']['critic_tokens'] > 0:
            f.write(f"  Critic Tokens: {report['llm_token_usage']['critic_tokens']}\n")
        if report['llm_token_usage']['judge_tokens'] > 0:
            f.write(f"  Judge Tokens: {report['llm_token_usage']['judge_tokens']}\n")
        f.write(f"  Avg Tokens per Task: {report['llm_token_usage']['avg_tokens_per_task']}\n")
        f.write("\n")
        
        
        if "actspec_execution_stats" in report:
            f.write("4. ActSpec Execution Stats\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Total Invocations: {report['actspec_execution_stats']['total_invocations']}\n")
            f.write(f"  Successful Invocations: {report['actspec_execution_stats']['successful_invocations']}\n")
            f.write(f"  Failed Invocations: {report['actspec_execution_stats']['failed_invocations']}\n")
            f.write(f"  Success Rate: {report['actspec_execution_stats']['success_rate']}\n")
            f.write(f"  Avg Invocations per Task: {report['actspec_execution_stats']['avg_invocations_per_task']}\n")
            f.write("\n")
    
    print(f"[Stats] Statistics report saved to: {report_file}")
    print(f"[Stats] Text report saved to: {report_text_file}")
    print(f"[Stats] Task completion rate: {completion_rate:.2f}% ({success_count}/{total_tasks})")
    print(f"[Stats] Total action invocations: {total_actions}")
    print(f"[Stats] Total token usage: {token_stats['total_tokens']}")
    if actspec_enabled:
        print(f"[Stats] ActSpec total invocations: {actspec_stats['total_calls']} (success: {actspec_stats['success_count']}, fail: {actspec_stats['fail_count']}, success rate: {actspec_stats['success_rate']:.2f}%)")

def run():
    parser = argparse.ArgumentParser(
        description="Evaluate AgentCore on WebArena tasks"
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
        help="Set Actor/Critic/Judge to the same model (OpenRouter format, e.g. openai/gpt-5.4-mini). Higher priority than models in config/config.yaml."
    )
    parser.add_argument(
        "--logname", type=str, default=None,
        help="Log subdirectory name (relative to config.logdir) for resume runs. Existing <task_id>.json will be skipped when reusing the same log folder. Example: --logname 20260507-091939-298185",
    )
    args = parser.parse_args()

    
    log_file, orig_stdout, orig_stderr = _start_console_logging()
    try:
        _run_impl(args, log_file, orig_stdout, orig_stderr)
    finally:
        _stop_console_logging(log_file, orig_stdout, orig_stderr)


def _run_impl(args, _console_log_file, _orig_stdout, _orig_stderr):
    """Core run() implementation; keep separate to reliably close terminal logs in finally."""
    
    if (args.resume_from or args.task_id) and not args.tasks_file:
        raise ValueError("--resume-from and --task-id must be used with --tasks-file. Please provide dataset path via --tasks-file.")
    with open(args.config, "r", encoding='utf-8') as file:
        config = DotDict(yaml.safe_load(file))
    
    
    if _has_unified_config and _unified_config:
        config = _apply_unified_config_overrides(config, _unified_config)
    config = _apply_agent_model_cli_override(config, getattr(args, "agent_model", None))
    logname_cli = getattr(args, "logname", None)
    if logname_cli is not None and str(logname_cli).strip():
        config.logname = str(logname_cli).strip()
        if hasattr(config, "agent") and hasattr(config.agent, "others"):
            config.agent.others.logname = config.logname
        print(f"[Config] --logname applied: {config.logname}")
    
    
    dstdir = None
    
    if config.logging:
        
        os.makedirs(config.logdir, exist_ok=True)
        
        
        if config.logname:
            dstdir = os.path.join(config.logdir, config.logname)
        else:
            
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
            dstdir = os.path.join(config.logdir, timestamp)
        
        
        os.makedirs(dstdir, exist_ok=True)
        print(f"[Log] Logs will be saved to: {os.path.abspath(dstdir)}")
        
        
        if hasattr(config, "agent") and hasattr(config.agent, "others"):
            config.agent.others.logdir_path = dstdir
    random.seed(42)
    
    config_file_list = []
    temp_dir = None
    task_configs_from_file = None  
    tasks_file_path = None         

    
    if args.tasks_file:
        if not os.path.exists(args.tasks_file):
            raise FileNotFoundError(f"Tasks file not found: {args.tasks_file}")
        print(f"[Info] Loading tasks from dataset: {args.tasks_file}")
        with open(args.tasks_file, "r", encoding="utf-8") as f:
            tasks_data = json.load(f)
        if not isinstance(tasks_data, list):
            raise ValueError(f"Tasks file must contain a JSON array, got {type(tasks_data)}")
        url_mapping = load_url_mapping()
        tasks_data = replace_placeholders(tasks_data, url_mapping)

        
        def _task_id_key(t):
            tid = t.get("task_id")
            if isinstance(tid, int):
                return tid
            if isinstance(tid, str) and tid.isdigit():
                return int(tid)
            return 0
        tasks_data.sort(key=_task_id_key)

        
        if args.task_id:
            try:
                tid = int(args.task_id)
            except ValueError:
                tid = args.task_id
            filtered = [t for t in tasks_data if t.get("task_id") == tid or t.get("task_id") == args.task_id]
            if not filtered:
                raise ValueError(f"--task-id {args.task_id}: task_id={args.task_id} not found in dataset")
            tasks_data = filtered
            print(f"[Info] Running only task: {args.task_id}")
        
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
                raise ValueError(f"--resume-from {args.resume_from}: no task with task_id>={args.resume_from} found in dataset")
            tasks_data = tasks_data[start_idx:]
            print(f"[Info] Resuming from task_id={args.resume_from}, total {len(tasks_data)} task(s)")
        else:
            print(f"[Info] Running all {len(tasks_data)} task(s)")

        task_configs_from_file = list(tasks_data)
        tasks_file_path = args.tasks_file
        
        current_dir = Path(__file__).resolve().parent
        main_dir = current_dir.parent
        temp_base_dir = main_dir / "Temp"
        os.makedirs(temp_base_dir, exist_ok=True)
        temp_dir = tempfile.mkdtemp(prefix="eval_tasks_", dir=str(temp_base_dir))
        single_task_path = os.path.join(temp_dir, "current_task.json")
        config_file_list = [single_task_path] * len(task_configs_from_file)
        print(f"[Info] Loaded {len(config_file_list)} task config(s)")

    
    actspec_config = getattr(config, 'actspec', {})
    auto_generate = actspec_config.get('auto_generate', False)
    test_mode_enabled = actspec_config.get('test_mode', {}).get('enabled', False)
    
    if auto_generate and test_mode_enabled:
        raise ValueError(
            "Invalid ActSpec config: auto_generate and test_mode.enabled cannot both be true.\n"
            "Choose one mode:\n"
            "  - Training mode: set auto_generate=true, test_mode.enabled=false\n"
            "  - Testing mode: set auto_generate=false, test_mode.enabled=true"
        )
    
    
    is_training_mode = auto_generate
    is_testing_mode = test_mode_enabled
    
    
    config._actspec_training_mode = is_training_mode
    config._actspec_testing_mode = is_testing_mode
    
    if actspec_config.get('enabled', False):
        mode_str = "training" if is_training_mode else ("testing" if is_testing_mode else "disabled")
        print(f"[ActSpec] Mode: {mode_str} (auto_generate={auto_generate}, test_mode.enabled={test_mode_enabled})")

    
    global_primitive_budget = int(actspec_config.get("global_primitive_budget", 0) or 0)
    remaining_primitive_budget = global_primitive_budget if global_primitive_budget > 0 else None
    if remaining_primitive_budget is not None:
        print(f"[Budget] Global primitive budget B={remaining_primitive_budget}")

    if not config_file_list:
        
        task_ids = config.env.task_ids
        if hasattr(config.env, "relative_task_dir"):
            relative_task_dir = config.env.relative_task_dir
        else:
            relative_task_dir = "tasks"
        if task_ids == "all" or task_ids == ["all"]:
            task_ids = [filename[:-len(".json")] for filename in os.listdir(f"config_files/{relative_task_dir}") if filename.endswith(".json")]
        
        
        url_mapping = load_url_mapping()
        
        
        current_dir = Path(__file__).resolve().parent
        main_dir = current_dir.parent  
        temp_base_dir = main_dir / "Temp"
        os.makedirs(temp_base_dir, exist_ok=True)
        temp_dir = tempfile.mkdtemp(prefix="config_files_", dir=str(temp_base_dir))
        
        for task_id in task_ids:
            config_file_path = f"config_files/{relative_task_dir}/{task_id}.json"
            
            with open(config_file_path, "r", encoding="utf-8") as f:
                task_config = json.load(f)
            
            task_config = replace_placeholders(task_config, url_mapping)
            
            safe_task_id = str(task_id).replace("/", "_").replace("\\", "_")
            temp_config_file = os.path.join(temp_dir, f"{safe_task_id}.json")
            with open(temp_config_file, "w", encoding="utf-8", errors='replace') as f:
                json.dump(task_config, f, indent=2, ensure_ascii=False)
            config_file_list.append(temp_config_file)

    fullpage = config.env.fullpage if hasattr(config.env, "fullpage") else True
    current_viewport_only = not fullpage

    if config.agent.type == "Agent":
        agent_init = lambda: AgentCore(
            prompt_dict = {k: v for k, v in AgentPrompt.__dict__.items() if isinstance(v, dict)},
            config = config.agent,
        )
    elif config.agent.type == "Baseline-SteP":
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
        
        if task_configs_from_file is not None:
            with open(config_file, "w", encoding="utf-8", errors="replace") as f:
                json.dump(task_configs_from_file[i], f, indent=2, ensure_ascii=False)
        with open(config_file, "r", encoding='utf-8') as f:
            task_config = json.load(f)
            print(f"Task {task_config['task_id']}.")
        
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
        
        
        action_timeout = getattr(config, 'task_timeout', 300)  
        timeout_occurred = [False]  
        status = None  
        
        
        env.is_done = False
        if hasattr(env, 'timeout_occurred'):
            env.timeout_occurred[0] = False
        
        
        env.action_timeout = action_timeout
        env.timeout_occurred = timeout_occurred
        
        if hasattr(config, 'env'):
            if not hasattr(config.env, 'action_timeout'):
                config.env.action_timeout = action_timeout
            if not hasattr(config.env, 'timeout_occurred'):
                config.env.timeout_occurred = timeout_occurred
            else:
                
                config.env.timeout_occurred[0] = False
        
        try:
            status = agent.act(objective=objective, env=env)
        except KeyboardInterrupt:
            
            print(f"[Error] Task {task_config['task_id']} interrupted by user")
            if timeout_occurred[0]:
                status = {"success": 0, "reward": 0, "done": True, "timeout": True}
            else:
                raise
        except TimeoutError as e:
            
            print(f"[Error] Task {task_config['task_id']} interrupted by action timeout: {e}")
            timeout_occurred[0] = True
            status = {"success": 0, "reward": 0, "done": True, "timeout": True}
        except Exception as e:
            if timeout_occurred[0]:
                print(f"[Error] Task {task_config['task_id']} failed: {e}")
                status = {"success": 0, "reward": 0, "done": True, "timeout": True}
            else:
                
                print(f"[Error] Task {task_config['task_id']} failed: {e}")
                raise
        finally:
            
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
                print(f"[Info] Task {task_config['task_id']} marked as timeout; continue to next task")
            
            
            try:
                
                env.close()
            except Exception as e:
                
                
                error_str = str(e).lower()
                if "greenlet" in error_str or "cannot switch" in error_str:
                    
                    pass
                else:
                    print(f"[Warning] {e}")
                
                
            
            
            if status is None:
                status = env.status() if hasattr(env, 'status') else {"success": 0, "reward": 0, "done": True}

        if config.logging:
            
            task_ref = f"{tasks_file_path}#{task_config['task_id']}" if tasks_file_path else config_file
            log_file = os.path.join(dstdir, f"{task_config['task_id']}.json")
            trajectory = agent.get_trajectory()
            
            
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
                
                "actspec_calls": getattr(env, "actspec_call_records", []),
                "statistics": {
                    "action_counts": action_stats,
                    "token_usage": token_stats,
                }
            }
            summary_file = os.path.join(dstdir, "summary.csv")
            
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
            
            
            summary_data["total_actions"] = sum(action_stats.values())
            summary_data["total_tokens"] = token_stats["total_tokens"]
            for action_type, count in action_stats.items():
                summary_data[f"action_{action_type}"] = count
            
            print(f"[eval_webarena] Preparing log_run, task_id={task_config['task_id']}")
            print(f"[eval_webarena] log_file={log_file}")
            print(f"[eval_webarena] log_data type: {type(log_data)}")
            if isinstance(log_data, dict):
                print(f"[eval_webarena] log_data keys: {list(log_data.keys())}")
            print(f"[eval_webarena] summary_data type: {type(summary_data)}")
            if isinstance(summary_data, dict):
                print(f"[eval_webarena] summary_data keys: {list(summary_data.keys())}")
            
            log_run(
                log_file=log_file,
                log_data=log_data,
                summary_file=summary_file,
                summary_data=summary_data,
            )
            
            print(f"[eval_webarena] log_run completed, task_id={task_config['task_id']}")
            
            
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
                    
                    print(f"[ActSpec][Async] Start generating ActSpec for task {task_cfg_snapshot['task_id']}...")
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
                        print(f"[ActSpec][Async] Invalid trajectory format, skip ActSpec generation")
                        return
                    if len(trajectory_snapshot) == 0:
                        print(f"[ActSpec][Async] Empty trajectory, skip ActSpec generation")
                        return
                    
                    segments = segmenter.segment_trajectory(trajectory_snapshot, task_cfg_snapshot)
                    print(f"[ActSpec][Async] Segmented into {len(segments)} segment(s)")
                    if not segments:
                        print(f"[ActSpec][Async] No valid segments, skip ActSpec generation")
                        return
                    
                    actspec_count = 0
                    for seg_idx, segment in enumerate(segments):
                        try:
                            if not isinstance(segment, dict):
                                print(f"[ActSpec][Async] Segment {seg_idx} has invalid format, skip")
                                continue
                            
                            actions = segment.get('actions', [])
                            context = segment.get('context', {})
                            
                            if not actions or not isinstance(actions, list):
                                print(f"[ActSpec][Async] Segment {seg_idx} has no valid actions, skip")
                                continue
                            
                            if not isinstance(context, dict):
                                print(f"[ActSpec][Async] Segment {seg_idx} has invalid context, using defaults")
                                context = {"site": "unknown", "page": "unknown", "url": ""}
                            
                            actspec = generator.generate_actspec(
                                actions,
                                context,
                                task_cfg_snapshot
                            )
                            
                            if not actspec or not isinstance(actspec, dict):
                                print(f"[ActSpec][Async] Segment {seg_idx} generated invalid ActSpec, skip")
                                continue
                            
                            library.save_actspec(actspec, library_full_path)
                            actspec_count += 1
                            print(f"[ActSpec][Async] Generated ActSpec: {actspec.get('action_id', 'unknown')}")
                        except Exception as e:
                            print(f"[ActSpec][Async] Segment {seg_idx} ActSpec generation failed: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    print(f"[ActSpec][Async] Task {task_cfg_snapshot['task_id']} generated {actspec_count}/{len(segments)} ActSpec(s), saved to {library_full_path}")
                except Exception as e:
                    print(f"[Warning][ActSpec][Async] {e}")
                    import traceback
                    traceback.print_exc()

            
            try:
                threading.Thread(
                    target=_async_generate_actspec,
                    args=(task_config, trajectory, config),
                    daemon=True,
                ).start()
            except Exception as e:
                print(f"[Warning] Failed to start async ActSpec thread: {e}")

            
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
                    
                    print(f"[ActSpec][Async] Start offline reuse evaluation for log {log_file_path}...")
                    evaluate_and_update_library_for_log(
                        log_file=log_file_path,
                        library_path=library_path_local,
                        llm_cfg=llm_cfg,
                        convert_to_negative_constraints=convert_to_negative_constraints,
                    )
                    print(
                        f"[ActSpec][Async] Offline reuse evaluation finished for {log_file_path}. Library updated."
                        f" convert_to_negative_constraints={convert_to_negative_constraints}"
                    )
                except Exception as e:
                    print(f"[Warning][ActSpec][Async] {e}")
                    import traceback
                    traceback.print_exc()

            try:
                threading.Thread(
                    target=_async_evaluate_actspec_reuse,
                    args=(log_file, config),
                    daemon=True,
                ).start()
            except Exception as e:
                print(f"[Warning] Failed to start async ActSpec thread: {e}")
            
            
            print(f"[Stats] Task {task_config['task_id']} completed, updating statistics report...")
            try:
                generate_statistics_report(dstdir)
            except Exception as e:
                print(f"[Warning] {e}")
    
    
    if temp_dir and os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            print(f"[Info] Cleaned temp directory: {temp_dir}")
        except Exception as e:
            print(f"[Warning] {e}")
    
    
    if config.logging and dstdir:
        print("\n[Stats] Generating final statistics report...")
        try:
            generate_statistics_report(dstdir)
        except Exception as e:
            print(f"[Warning] {e}")
    
if __name__ == "__main__":
    run()

