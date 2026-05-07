import json
import time
import threading
from browser_env import (
    create_id_based_action,
    create_id_based_actions,
    StateInfo,
    Trajectory,
    ActionTypes,
)
from browser_env.async_wrapper import AsyncPlaywrightWrapper
from evaluation_harness.evaluators import evaluator_router
from AgentCore.obs_opt import (
    prune_tree,
    translate_node_to_str,
    search_node_by_id,
    action_set_invisible,
)


class WebArenaEnvironmentWrapper():
    def __init__(self, config_file, max_browser_rows=300, max_steps=50, slow_mo=1, observation_type="accessibility_tree", current_viewport_only=False, viewport_size={"width": 1280, "height": 720}, headless=False, global_config=None, llm_config=None):
        
        self.webarena_env = AsyncPlaywrightWrapper(
                    headless=headless,
                    slow_mo=slow_mo,
                    observation_type=observation_type,
                    current_viewport_only=current_viewport_only,
                    viewport_size=viewport_size,
                    global_config=global_config
                )
        self.config_file = config_file
        with open(self.config_file, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        self.global_config = global_config
        
        if llm_config is None and global_config and hasattr(global_config, 'get_api_key'):
            self.llm_config = global_config
        else:
            self.llm_config = llm_config
        
        
        self.webarena_env.action_timeout = getattr(global_config, 'action_timeout', 300) if global_config else 300
        
        self.obs, self.info = self.webarena_env.reset(options={"config_file": self.config_file})
        self.terminated = False
        self.objective = self.config["intent"]
        self.url = self.config["start_url"]
        self.max_browser_rows = max_browser_rows
        self.max_steps = max_steps
        self.steps = 0
        self.is_done = False
        self.reward = 0.0
        
        self.trajectory: Trajectory = []
        self.update_webarena_metrics()
        
        
        self.actspec_executor = None
        self.actspec_library = None
        self.actspec_merger = None
        self.actspecs = []
        self.negative_constraint_filter = None
        if global_config:
            try:
                actspec_config = getattr(global_config, 'actspec', {})
                test_mode_enabled = actspec_config.get('test_mode', {}).get('enabled', False)
                is_testing = getattr(global_config, '_actspec_testing_mode', False)
                
                
                if test_mode_enabled and is_testing:
                    from actspec import ActSpecExecutor, ActSpecLibrary, ActSpecMerger, NegativeConstraintFilter
                    self.actspec_executor = ActSpecExecutor()
                    library_path = actspec_config.get('test_mode', {}).get('library_path', '')
                    if not library_path:
                        library_path = actspec_config.get('library_path', 'temp_library')
                    self.actspec_library_path = library_path
                    self.use_negative_constraints_only = actspec_config.get('test_mode', {}).get(
                        'use_negative_constraints_only', False
                    )
                    self.actspec_library = ActSpecLibrary(library_path)
                    self.actspec_merger = ActSpecMerger()
                    
                    
                    self.actspec_call_records = []
                    
                    
                    self.actspecs = self.actspec_library.load_library()
                    
                    
                    
                    negative_constraints = self.actspec_library.load_negative_constraints()
                    if negative_constraints:
                        self.negative_constraint_filter = NegativeConstraintFilter(negative_constraints)
                        print(f"[负约束] 已加载 {len(negative_constraints)} 个负约束")
                    else:
                        print(f"[负约束] 未找到负约束")
            except Exception as e:
                print(f"[警告] ActSpec初始化失败: {e}")
                import traceback
                traceback.print_exc()

    def _map_action_type_to_name(self, action_type: int) -> str:
        """
        将 ActionTypes 数字枚举转换为用于指纹的标准化动作类型字符串（小写）。
        仅用于构建运行时的动作历史序列。
        """
        mapping = {
            ActionTypes.SCROLL: "scroll",
            ActionTypes.CLICK: "click",
            ActionTypes.TYPE: "type",
            ActionTypes.HOVER: "hover",
            ActionTypes.PAGE_FOCUS: "goto",
            ActionTypes.NEW_TAB: "goto",
            ActionTypes.GOTO_URL: "goto",
            ActionTypes.GO_BACK: "go_back",
            ActionTypes.GO_FORWARD: "go_forward",
            ActionTypes.STOP: "stop",
        }
        return mapping.get(action_type, "")

    def _get_recent_action_history_types(self, max_len: int = 3) -> list:
        """
        从环境维护的 trajectory 中提取最近 max_len 个已执行 primitive 动作的类型序列（小写字符串）。
        仅统计已真正执行过的动作，用于与负约束/ActSpec 上的 action_history_prefix 进行匹配。
        """
        types = []
        
        for item in reversed(self.trajectory):
            if not isinstance(item, dict):
                continue
            if "action_type" not in item:
                continue
            t = self._map_action_type_to_name(item.get("action_type"))
            if not t:
                continue
            types.append(t)
            if len(types) >= max_len:
                break
        if not types:
            return []
        return list(reversed(types))
        
    def reset(self):
        self.obs, self.info = self.webarena_env.reset(options={"config_file": self.config_file})

    def close(self):
        self.webarena_env.close()
        
    def get_url(self):
        return self.url
    
    def get_objective(self):
        return self.objective 
    
    def get_sites(self):
        return self.config["sites"]
        
    def observation(self):
        
        if getattr(self, "_in_observation", False):
            print("[observation] 重入：跳过负约束路径，返回当前 obs 的裁剪结果")
            try:
                if "page" in self.info and hasattr(self.info["page"], "url"):
                    self.url = self.info["page"].url
            except Exception:
                pass
            if self.global_config and self.global_config.env.prune:
                root_node = self.obs["text"][1]
                DOM_root_node = prune_tree(objective=self.objective, root_node=root_node, mode="node")
                DOM_str = translate_node_to_str(node=DOM_root_node, mode="concise")
                return {"text": DOM_str, "image": self.obs["image"], "node": DOM_root_node}
            browser_content = self.obs["text"][0]
            browser_content = browser_content.split("\n")[:self.max_browser_rows]
            browser_content = "\n".join(browser_content)
            return browser_content

        self._in_observation = True
        try:
            return self._observation_impl()
        finally:
            self._in_observation = False

    def _observation_impl(self):
        
        
        print(f"[observation] 开始获取 observation...")
        
        
        
        try:
            if "page" in self.info and hasattr(self.info["page"], "url"):
                self.url = self.info["page"].url
                print(f"[observation] 从 info 中获取 URL: {self.url}")
            else:
                
                print(f"[observation] info 中没有 page，保持原有 URL: {self.url}")
        except Exception as e:
            print(f"[observation] 获取 URL 时发生异常: {e}，保持原有 URL: {self.url}")
            
            pass
        
        if self.global_config and self.global_config.env.prune:
            root_node = self.obs["text"][1]
            DOM_root_node = prune_tree(objective=self.objective, root_node=root_node, mode="node")
            
            if self.negative_constraint_filter:
                try:
                    from actspec.url_utils import extract_site_and_page_from_url
                    sites = self.config.get("sites", [])
                    site, page = extract_site_and_page_from_url(
                        self.url, sites=sites, include_port=True
                    )
                    
                    history_types = self._get_recent_action_history_types()
                    
                    observation_text = None
                    try:
                        
                        raw_obs = self.webarena_env.obs if hasattr(self.webarena_env, "obs") else None
                        if isinstance(raw_obs, dict) and "text" in raw_obs:
                            content = raw_obs["text"]
                            if isinstance(content, tuple) and len(content) > 0:
                                observation_text = content[0]
                            elif isinstance(content, str):
                                observation_text = content
                    except Exception:
                        observation_text = None

                    if not isinstance(observation_text, str) or observation_text.strip() == "":
                        observation_text = "__EMPTY_OBSERVATION__"

                    context = {
                        "site": site,
                        "page": page,
                        "url": self.url,
                        "action_history_types": history_types,
                        "pre_condition_page": self,
                        "pre_condition_observation_text": observation_text,
                    }
                    forbidden_ids = self.negative_constraint_filter.get_forbidden_element_ids_for_observation(context)
                    for eid in forbidden_ids:
                        node = search_node_by_id(DOM_root_node, eid)
                        if node is not None:
                            action_set_invisible(node)
                    DOM_str = translate_node_to_str(node=DOM_root_node, mode="concise")
                except Exception as e:
                    print(f"[负约束] filter_observation 异常: {e}")
                    DOM_str = translate_node_to_str(node=DOM_root_node, mode="concise")
            else:
                DOM_str = translate_node_to_str(node=DOM_root_node, mode="concise")
            return {"text": DOM_str, "image": self.obs["image"], "node": DOM_root_node}
        else:
            browser_content = self.obs["text"][0]
            browser_content = browser_content.split("\n")[:self.max_browser_rows] 
            browser_content = "\n".join(browser_content)
            return browser_content
    
    def done(self):
        if self.is_done:
            return True
        return False
    
    def status(self):
        return {'done': self.is_done, 'reward': self.reward, 'success': float(self.reward > 0), 'num_actions': self.steps}
    
    def _step_actspec_internal(self, action):
        """
        ActSpec 内部 step：不递增 steps、不解析 actspec、不做负约束检查。
        将 action 视为已解析的 primitive 字符串，转为 id_based_actions 并执行。
        用于 execute_actspec -> _execute_plan 中每步 plan 的 env.step(action_str, is_actspec_internal=True)。
        """
        if action is None or action == "":
            return self.status()
        try:
            action_cmds = create_id_based_actions(action)
        except Exception as e:
            print(f"[ActSpec内部] Invalid action syntax: {e}")
            return self.status()
        if not action_cmds:
            return self.status()
        for action_cmd in action_cmds:
            action_start_time = time.time()
            action_type = action_cmd.get("action_type", "UNKNOWN")
            action_details = f"action_type={action_type}"
            if "element_id" in action_cmd and action_cmd["element_id"]:
                action_details += f", element_id={action_cmd['element_id']}"
            print(f"[ActSpec内部] 执行: {action_details}")
            try:
                result = self.webarena_env.step(action_cmd)
                self.obs, _, self.terminated, _, self.info = result
                self.update_webarena_metrics(action_cmd)
                
                try:
                    if isinstance(self.info, dict) and "page" in self.info:
                        p = self.info["page"]
                        if hasattr(p, "url"):
                            self.url = p.url
                except Exception:
                    pass
                actual_elapsed = time.time() - action_start_time
                print(f"[ActSpec内部] 完成，耗时 {actual_elapsed:.2f}s")
            except TimeoutError as e:
                print(f"[警告] ActSpec内部步骤超时: {e}")
                self.is_done = True
                break
            except Exception as e:
                print(f"[ActSpec内部] 执行异常: {e}")
                error_str = str(e).lower()
                if "timeout" in error_str or "cancelled" in error_str:
                    self.is_done = True
                    break
        return self.status()
    
    def step(self, action, is_actspec_internal=False):
        """
        执行一步环境动作。
        :param action: 动作字符串（可为 actspec 或 primitive 格式）
        :param is_actspec_internal: 若为 True，表示来自 ActSpec 内部的 plan step 调用：
            不递增 self.steps、不解析 actspec、不做负约束检查，仅将 action 转为 id_based_actions 并执行。
            保证「一次 ActSpec 调用 = 一次整体 step」的步数语义。
        """
        
        if is_actspec_internal:
            return self._step_actspec_internal(action)

        
        
        if self.steps >= self.max_steps:
            print(f"Steps {self.steps} reached maximum {self.max_steps}, terminating task")
            self.is_done = True
            action_cmd = create_id_based_action(f"stop [Trajectory failed: Steps {self.steps} reached maximum {self.max_steps}.]")
            self.update_webarena_metrics(action_cmd)
            return self.status()
        
        self.steps = self.steps + 1
        print(f"\n{'*'*100}")
        print(f"[环境Step] Step {self.steps}")
        print(f"[环境Step] 接收到的action: {repr(action)}")
        print(f"[环境Step] action类型: {type(action)}")
        print(f"{'*'*100}")
        
        if self.steps > self.max_steps:
            print(f"Steps {self.steps} exceeded maximum {self.max_steps}")
            self.is_done = True
            action_cmd = create_id_based_action(f"stop [Trajectory failed: Steps {self.steps} exceeded maximum {self.max_steps}.]")
            self.update_webarena_metrics(action_cmd)
            return self.status()

        if action is None or action == "":
            action_cmds = []
        else:
            
            action_str = str(action).strip()
            if self.actspec_executor and self.actspec_merger and action_str.startswith("actspec"):
                
                parsed = self.actspec_merger.parse_actspec_action(action_str)
                if parsed:
                    action_id = parsed["action_id"]
                    parameters = parsed["parameters"]
                    
                    
                    actspec = None
                    for spec in self.actspecs:
                        if spec.get("action_id") == action_id:
                            actspec = spec
                            break
                    
                    if actspec:
                        
                        base_params = {}
                        params_def = actspec.get("parameters", {}) or {}
                        for pname, pdef in params_def.items():
                            ptype = (pdef or {}).get("type", "")
                            pname_lower = pname.lower()
                            
                            if ptype == "number":
                                continue
                            if "element_id" in pname_lower or "button_id" in pname_lower or "combobox" in pname_lower or pname_lower == "id":
                                continue
                            cands = (pdef or {}).get("candidates", [])
                            if cands:
                                base_params[pname] = cands[0]
                        
                        final_params = dict(base_params)
                        valid_param_names = set(params_def.keys())
                        for k, v in (parameters or {}).items():
                            if k in valid_param_names:
                                final_params[k] = v
                            else:
                                print(f"[ActSpec] 忽略未知参数 '{k}'，使用 ActSpec 默认值")
                        if parameters:
                            for k in parameters:
                                if k in valid_param_names and ("element_id" in k.lower() or k.lower() == "button_id"):
                                    print(f"[ActSpec] 注意：LLM 提供了结构性参数 {k}，执行时仍以 locate/plan 为准")
                        parameters = final_params
                        
                        print(f"[ActSpec] 执行ActSpec: {action_id} with parameters: {parameters}")
                        status = self.actspec_executor.execute_actspec(actspec, parameters, self)
                        
                        if status["success"]:
                            print(f"[ActSpec] ActSpec执行成功")
                            return self.status()
                        else:
                            print(f"[ActSpec] ActSpec执行失败: {status.get('error', 'unknown error')}, fallback到primitive action")
                            try:
                                if self.actspec_library:
                                    from actspec.url_utils import extract_site_and_page_from_url
                                    site, page = extract_site_and_page_from_url(
                                        self.url,
                                        sites=self.config.get("sites", []),
                                        include_port=True,
                                    )
                                    self.actspec_library.add_failed_actspec_as_negative_constraint(
                                        actspec,
                                        library_path=str(self.actspec_library.base_path),
                                        trajectory=self.trajectory if isinstance(self.trajectory, list) else [],
                                        failure_reason=str(status.get("error", "")),
                                        runtime_context={
                                            "site": site,
                                            "page": page,
                                            "url": self.url,
                                            "action_history_types": self._get_recent_action_history_types(),
                                        },
                                    )
                                    negative_constraints = self.actspec_library.load_negative_constraints(
                                        library_path=str(self.actspec_library.base_path)
                                    )
                                    if negative_constraints:
                                        from actspec import NegativeConstraintFilter
                                        self.negative_constraint_filter = NegativeConstraintFilter(negative_constraints)
                            except Exception as e:
                                print(f"[ActSpec] immediate negative-constraint extraction failed: {e}")
                            
                    else:
                        print(f"[ActSpec] 未找到ActSpec: {action_id}, fallback到primitive action")
            
            
            try:
                action_cmds = create_id_based_actions(action)
                if not action_cmds:
                    
                    self.steps = max(0, self.steps - 1)
                    return False
            except Exception as e:
                print(f"Invalid action syntax: {e}")
                action_cmds = []
        
        
        
        if self.negative_constraint_filter and action_cmds:
            
            from actspec.url_utils import extract_site_and_page_from_url
            url = self.url
            sites = self.config.get("sites", [])
            site, page = extract_site_and_page_from_url(url, sites=sites, include_port=True)
            context = {
                "site": site,
                "page": page,
                "url": url,
                
                "action_history_types": self._get_recent_action_history_types(),
            }
            
            
            filtered_action_cmds = []
            forbidden_errors = []  
            
            for action_cmd in action_cmds:
                is_forbidden, constraint_info = self.negative_constraint_filter.is_primitive_action_forbidden(
                    action_cmd, context
                )
                
                if is_forbidden:
                    
                    constraint_id = constraint_info.get("constraint_id", "unknown")
                    failure_reason = constraint_info.get("failure_reason", "该操作被负约束禁止")
                    
                    
                    action_type = action_cmd.get("action_type", "UNKNOWN")
                    element_id = action_cmd.get("element_id", "")
                    action_desc = f"action_type={action_type}"
                    if element_id:
                        action_desc += f", element_id={element_id}"
                    
                    error_msg = f"[负约束拦截] Action被禁止执行: {action_desc}。原因: {failure_reason} (constraint_id: {constraint_id})"
                    print(f"[负约束拦截] {error_msg}")
                    forbidden_errors.append(error_msg)
                else:
                    
                    filtered_action_cmds.append(action_cmd)
            
            
            if not filtered_action_cmds:
                
                
                
                
                
                
                print(f"[负约束拦截] 所有action都被拦截，但steps已计入（当前: {self.steps}/{self.max_steps}）")
                
                
                if self.steps >= self.max_steps:
                    print(f"[负约束拦截] 达到最大steps数 {self.max_steps}，终止任务")
                    self.is_done = True
                    action_cmd = create_id_based_action(f"stop [Trajectory failed: Reached maximum steps {self.max_steps} with all actions blocked by negative constraints.]")
                    self.update_webarena_metrics(action_cmd)
                    return self.status()
                
                return self.status()
            
            
            action_cmds = filtered_action_cmds
        
        for action_cmd in action_cmds:
            
            action_start_time = time.time()
            
            
            action_type = action_cmd.get("action_type", "UNKNOWN")
            action_details = f"action_type={action_type}"
            if "element_id" in action_cmd and action_cmd["element_id"]:
                action_details += f", element_id={action_cmd['element_id']}"
            if "text" in action_cmd and action_cmd["text"]:
                action_details += f", text='{action_cmd['text'][:50]}...'" if len(action_cmd["text"]) > 50 else f", text='{action_cmd['text']}'"
            print(f"[执行] 开始执行浏览器操作: {action_details}")
            print(f"[执行] action_cmd完整内容: {action_cmd}")
            
            try:
                
                result = self.webarena_env.step(action_cmd)
                self.obs, _, self.terminated, _, self.info = result
                
                try:
                    if isinstance(self.info, dict) and "page" in self.info:
                        p = self.info["page"]
                        if hasattr(p, "url"):
                            self.url = p.url
                except Exception:
                    pass
                self.update_webarena_metrics(action_cmd)
                
                
                actual_elapsed = time.time() - action_start_time
                print(f"[完成] 浏览器操作执行完成，耗时 {actual_elapsed:.2f} 秒")
                if isinstance(self.obs, dict) and "text" in self.obs:
                    raw_text = self.obs["text"]
                    
                    if isinstance(raw_text, (list, tuple)) and len(raw_text) > 0:
                        text_str = raw_text[0] if isinstance(raw_text[0], str) else str(raw_text[0])
                    else:
                        text_str = raw_text if isinstance(raw_text, str) else str(raw_text)
                    text_len = len(text_str)
                    obs_text_preview = (text_str[:300] + "...") if text_len > 300 else text_str
                    print(f"[完成] 观察文本预览 (长度: {text_len}): {obs_text_preview}")
            except TimeoutError as e:
                
                print(f"[警告] 执行步骤时检测到超时: {e}")
                self.is_done = True
                break
            except Exception as e:
                
                print(f"Error occurred while taking step: {e}")
                
                
                error_str = str(e).lower()
                if "timeout" in error_str or "cancelled" in error_str:
                    self.is_done = True
                    break
            
        return self.status()
    
    def update_webarena_metrics(self, action_cmd=None):
        
        if action_cmd:
            self.trajectory.append(action_cmd)
            if action_cmd["action_type"]== ActionTypes.STOP:
                self.is_done = True

        if not self.is_done: 
            state_info: StateInfo = {"observation": self.obs, "info": self.info}
            self.trajectory.append(state_info)
            
        if self.is_done:
            print("[评估] 任务结束 (is_done=True)，准备触发评估")
            try:
                evaluator = evaluator_router(self.config_file)
                print("[评估] 评估器已加载，trajectory 步数=%d" % len(self.trajectory))
                
                page_adapter = self.webarena_env.get_sync_page_adapter()
                if page_adapter:
                    client = page_adapter.client
                    print("[评估] 开始执行评估...")
                    llm_config = getattr(self, "llm_config", None)
                    self.reward = evaluator(trajectory=self.trajectory, config_file=self.config_file, page=page_adapter, client=client, llm_config=llm_config)
                    print("[评估] 评估执行完成，reward=%.4f" % self.reward)
                else:
                    print("[评估] 无法获取页面适配器，跳过评估，reward=0")
                    self.reward = 0
            except Exception as e:
                print("[评估] 评估过程异常: %s" % e)
                self.reward = 0

