"""SWE-agent-compatible local agent loop for mini bug-fix instances."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from stage2.devbugs.catalog import MiniInstance
from stage2.devbugs.evaluate import evaluate_instance
from stage2.devbugs.parse_action import format_assistant_response, parse_thought_action
from stage2.devbugs.sandbox import materialize_repo, run_command

SYSTEM_PROMPT = (
    "You are a helpful assistant that can interact with a computer to solve tasks.\n"
    "You have access to a bash shell in a small Python repository.\n"
    "Inspect files, edit them with sed/python/vi-style tools, and run tests.\n"
    "When you believe the issue is fixed, run `python -m pytest -q` to verify.\n\n"
    "Respond with your reasoning, then a single bash command inside a fenced code block:\n"
    "```\n"
    "your-command-here\n"
    "```"
)


@dataclass
class AgentConfig:
    api_base: str
    api_key: str
    model: str
    max_steps: int = 25
    temperature: float = 0.0
    command_timeout_s: int = 60


def _chat_completion(
    cfg: AgentConfig,
    messages: list[dict[str, str]],
) -> str:
    url = f"{cfg.api_base.rstrip('/')}/chat/completions"
    payload = {
        "model": cfg.model,
        "messages": messages,
        "temperature": cfg.temperature,
        # Pin thinking mode OFF at generation time so the recorded trajectory
        # matches how project_steps.py replays it (enable_thinking=False by
        # default). Without this, the server's default template could generate
        # with <think> blocks while projection reads it thinking-off, misaligning
        # the activations with what the model actually computed. vLLM's
        # OpenAI-compatible endpoint honors chat_template_kwargs.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg.api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"]


def _issue_user_message(problem: str) -> str:
    return f"We're solving the following issue:\n\nISSUE:\n{problem.strip()}\n"


def _observation_user_message(observation: str) -> str:
    obs = observation.strip()
    if not obs:
        obs = "(no output)"
    return f"OBSERVATION:\n{obs}\n"


def _build_query_messages(
    problem: str,
    history: list[tuple[str, str]],
) -> list[dict[str, str]]:
    """Messages visible before generating the next assistant turn."""
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _issue_user_message(problem)},
    ]
    for assistant_text, observation in history:
        messages.append({"role": "assistant", "content": assistant_text})
        messages.append({"role": "user", "content": _observation_user_message(observation)})
    return messages


def run_instance(
    instance: MiniInstance,
    cfg: AgentConfig,
    *,
    work_root: Path,
    output_traj: Path,
) -> tuple[bool, int]:
    """
    Run one mini instance; write SWE-agent-compatible .traj and return (resolved, n_steps).
    """
    repo_dir = materialize_repo(instance, work_root)
    history: list[tuple[str, str]] = []
    trajectory_steps: list[dict] = []

    resolved = False
    for step_idx in range(cfg.max_steps):
        query = _build_query_messages(instance.problem_statement, history)
        try:
            raw_response = _chat_completion(cfg, query)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Model API error ({exc.code}): {body}") from exc

        thought, action = parse_thought_action(raw_response)
        assistant_text = format_assistant_response(thought, action)

        observation = ""
        if action:
            if action.strip().lower() in {"submit", "exit", "done"}:
                observation = "Submitted. Running final evaluation."
            else:
                _, observation = run_command(
                    action,
                    cwd=repo_dir,
                    timeout_s=cfg.command_timeout_s,
                )

        trajectory_steps.append(
            {
                "response": assistant_text,
                "thought": thought,
                "action": action + ("\n" if action and not action.endswith("\n") else ""),
                "observation": observation,
                "query": query,
            }
        )
        history.append((assistant_text, observation))

        if evaluate_instance(instance, repo_dir):
            resolved = True
            break

    traj_payload = {
        "trajectory": trajectory_steps,
        "info": {
            "instance_id": instance.instance_id,
            "exit_status": "resolved" if resolved else "failed",
            "resolved": resolved,
        },
    }
    output_traj.parent.mkdir(parents=True, exist_ok=True)
    output_traj.write_text(json.dumps(traj_payload, indent=2), encoding="utf-8")
    return resolved, len(trajectory_steps)
