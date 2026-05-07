# WebAct Core Code (for paper citation)

This folder is a minimal extraction from the project for anonymous paper artifact citation.

## Structure

- `actspec/`: WebAct core method implementation
  - `actspec_generator.py`: successful-trace lifting (LIFTSUCCESS)
  - `trace_segmenter.py`: trace segmentation into reusable sub-intents
  - `actspec_executor.py`: ActSpec contract execution (`Pre -> Locate -> Plan -> Post`)
  - `post_condition_verifier.py`: Post verification
  - `pre_condition_checker.py`: Pre applicability checks
  - `actspec_library.py`: ActSpec library, confidence update, disable policy, negative-constraint storage
  - `negative_constraint_utils.py`: failed-trace constraint subtype inference (`readiness/disambiguation`)
  - `actspec_offline_evaluator.py`: replay/evaluation and library update
  - `locate_executor.py`, `readiness_checker.py`, `step_executor.py`, `semantic_change_handler.py`: runtime execution utilities
  - `url_utils.py`, `element_context_extractor.py`, `accessibility_tree_parser.py`: context extraction/parsing utilities

- `agent_adapter/`: minimal adapter layer to runtime agent/environment
  - `AgentOccam.py`: planner-side action selection and ActSpec integration
  - `env.py`: environment integration, immediate failed-trace constraint extraction/injection

- `scripts/`
  - `eval_webarena.py`: evaluation loop, including optional global primitive budget `B`


