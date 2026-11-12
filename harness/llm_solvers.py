"""LLM fallbacks.  Each is a single short call (code-as-answer) whose output is
executed and validated locally; the agent loop is the last resort."""
from __future__ import annotations

import asyncio
import json
import re
import shutil
from pathlib import Path

from . import sandbox
from .llm import LLM, LLMUnavailable, MODEL_FAST
from .log import log
from .solvers import coding_solver, data_solver, search_solver

BOILER = re.compile(r"(work only inside|when you are done|write only|for example with bash|"
                    r"use bash|use the file tools|compute it with|use the bash tool)", re.I)


def clean_instruction(ins: str) -> str:
    sents = re.split(r"(?<=[.?!])\s+", ins)
    keep = [s for s in sents if not BOILER.search(s)]
    return " ".join(keep).strip() or ins


def extract_code(text: str) -> str | None:
    m = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    if m:
        return max(m, key=len)
    if "print(" in text:
        return text
    return None


def normalize(ans: str, check: str) -> str | None:
    a = ans.strip().strip("`").strip()
    a = a.splitlines()[-1].strip() if a else a
    a = re.sub(r"^(answer|final answer|result)\s*[:=]\s*", "", a, flags=re.I).strip()
    if not a:
        return None
    if check.startswith("exact_int") or check in ("int", "integer"):
        if len(a) >= 2 and a[0] == a[-1] and a[0] in "\"'":
            a = a[1:-1].strip()
        if not re.fullmatch(r"[+-]?(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.0+)?", a):
            return None
        try:
            return str(int(a.replace(",", "").split(".", 1)[0]))
        except ValueError:
            return None
    return a


def is_infeasible(text: str) -> bool:
    return bool(re.search(r"\bINFEASIBLE\b|\bIMPOSSIBLE\b|\bINSUFFICIENT\b", text or ""))


SYS_CODE = ("You convert a task into a short Python 3 program that prints ONLY the final "
            "answer (no words). Follow the wording exactly (e.g. 'rounded down' = floor "
            "division). If the given data is insufficient or contradictory so that no exact "
            "answer exists, the program must print INFEASIBLE. Reply with one ```python block.")


async def _code_answer(llm: LLM, task, user: str, cwd: Path, model: str, max_tokens: int,
                       tag: str):
    r = await llm.chat([{"role": "system", "content": SYS_CODE},
                        {"role": "user", "content": user}],
                       model=model, max_tokens=max_tokens, tag=tag)
    task.cost += r["cost"]
    content = r["content"] or ""
    if is_infeasible(content) and "print(" not in content:
        return ("infeasible", None, "model: infeasible")
    code = extract_code(content)
    if code is None:
        a = normalize(content, task.check)
        return ("answer", a, "direct") if a else None
    rc, out = await asyncio.to_thread(sandbox.run_python, code, cwd)
    stdout = out.split("\n[stderr]\n")[0].strip()
    if is_infeasible(stdout):
        return ("infeasible", None, "program: infeasible")
    if rc != 0 or not stdout:
        return ("error", None, out[-400:])
    a = normalize(stdout, task.check)
    return ("answer", a, "code") if a else ("error", None, "bad output: " + stdout[-200:])


def _det_hint(task) -> str:
    """A maybe_infeasible signal from the deterministic phase (e.g. a declared
    column absent in the CSV), passed to the model as a fact to check.  The
    model still decides: only its INFEASIBLE counts as a claim."""
    raw = getattr(task, "raw", None) or {}
    det = raw.get("_det") if isinstance(raw, dict) else None
    if not det or det[0] != "maybe_infeasible" or not det[2]:
        return ""
    return (f"\n\nNote from an automatic check: {det[2]}. If the question cannot be "
            "answered exactly from the data as it is, the program must print INFEASIBLE.")


async def solve_math(llm: LLM, task, model=MODEL_FAST):
    cwd = sandbox.scratch_copy(task.ws)
    try:
        return await _code_answer(llm, task, clean_instruction(task.instruction), cwd,
                                  model, 350, f"math:{task.task_id}")
    finally:
        shutil.rmtree(cwd, ignore_errors=True)


async def solve_data(llm: LLM, task, model=MODEL_FAST):
    cwd = sandbox.scratch_copy(task.ws)
    try:
        files = [p for p in sorted(cwd.rglob("*")) if p.is_file() and not p.name.startswith(".")]
        previews = []
        for p in files[:6]:
            if p.suffix.lower() in (".csv", ".tsv", ".txt", ".json", ".jsonl"):
                previews.append(data_solver.preview(p, 4)[:900])
            else:
                previews.append(f"{p.relative_to(cwd)} ({p.stat().st_size} bytes)")
        user = (f"Task: {clean_instruction(task.instruction)}\n\nFiles in the current "
                f"directory (previews):\n" + "\n\n".join(previews) +
                "\n\nRead the files with the csv/json modules (no pandas). Values are "
                "strings: convert carefully. Print only the answer.")
        user += _det_hint(task)
        return await _code_answer(llm, task, user, cwd, model, 450, f"data:{task.task_id}")
    finally:
        shutil.rmtree(cwd, ignore_errors=True)


async def solve_search(llm: LLM, task, model=MODEL_FAST):
    ctx = search_solver.retrieve_context(task.instruction, task.ws)
    if not ctx:
        return None
    user = (f"Knowledge base excerpts:\n{ctx}\n\nQuestion: {clean_instruction(task.instruction)}"
            "\n\nAnswer with the exact value as written in the knowledge base and nothing "
            "else. If the knowledge base does not contain enough information, answer "
            "INFEASIBLE.")
    r = await llm.chat([{"role": "user", "content": user}], model=model, max_tokens=40,
                       tag=f"search:{task.task_id}")
    task.cost += r["cost"]
    c = (r["content"] or "").strip()
    if is_infeasible(c):
        return ("infeasible", None, "model: infeasible")
    a = normalize(c, task.check)
    return ("answer", a, "llm") if a else None


SYS_FIX = ("You fix bugs in Python code so that the tests pass and the functions behave "
           "correctly in general (hidden tests will check more cases). Never change the "
           "tests. Reply with the COMPLETE corrected file in one ```python block. If the "
           "tests are contradictory or cannot pass, reply IMPOSSIBLE.")


async def solve_coding(llm: LLM, task, model=MODEL_FAST, attempts: int = 2,
                       start_files: dict | None = None):
    """One-shot whole-file repair with test feedback."""
    test_file, targets = coding_solver.locate(task.instruction, task.ws)
    if test_file is None or not targets:
        return None
    target = targets[0].relative_to(task.ws)
    test_file = test_file.relative_to(task.ws)
    cwd = sandbox.scratch_copy(task.ws)
    try:
        if start_files:
            for n, s in start_files.items():
                (cwd / n).write_text(s, encoding="utf-8")
        test_src = (cwd / test_file).read_text(encoding="utf-8")
        for attempt in range(attempts):
            ok, out = await asyncio.to_thread(coding_solver.run_tests, cwd, cwd / test_file)
            if ok:
                return ("fixed", {target.as_posix(): (cwd / target).read_text(encoding="utf-8")}, "llm")
            src = (cwd / target).read_text(encoding="utf-8")
            user = (f"Task: {clean_instruction(task.instruction)}\n\nFile {target.as_posix()}:\n```python\n{src}\n```\n\nTests {test_file.name}:\n"
                    f"```python\n{test_src[:4000]}\n```\n\nTest output:\n```\n{out[-1800:]}\n```")
            max_t = min(3000, int(len(src) / 2.5) + 300)
            r = await llm.chat([{"role": "system", "content": SYS_FIX},
                                {"role": "user", "content": user}],
                               model=model, max_tokens=max_t,
                               tag=f"code:{task.task_id}:{attempt}")
            task.cost += r["cost"]
            content = r["content"] or ""
            code = extract_code(content)
            if code is None:
                if is_infeasible(content):
                    return ("infeasible", None, "model: impossible")
                continue
            (cwd / target).write_text(code, encoding="utf-8")
            if (cwd / test_file).read_text() != test_src:
                return None
        ok, out = await asyncio.to_thread(coding_solver.run_tests, cwd, cwd / test_file)
        if ok:
            return ("fixed", {target.as_posix(): (cwd / target).read_text(encoding="utf-8")}, "llm")
        return None
    finally:
        shutil.rmtree(cwd, ignore_errors=True)


# --------------------------------------------------------------------- agent
AGENT_SYS = ("You are a careful, terse agent working in a sandbox directory. Use the tools to "
             "inspect files and run Python. Keep tool outputs small (grep/head instead of "
             "printing big files). When finished call submit(answer). For code-fix tasks fix "
             "files with write_file, make the tests pass, then submit('DONE'). If the task "
             "cannot be solved exactly from the available data, submit('INFEASIBLE').")


def _inside(p: Path, root: Path) -> bool:
    """True if p (after resolving symlinks and ..) lies within root."""
    try:
        return p.resolve().is_relative_to(root.resolve())
    except (OSError, ValueError):
        return False


SHELL_ALLOWED = {"ls", "cat", "head", "tail", "grep", "egrep", "wc", "find", "sort", "uniq",
                 "cut", "awk", "sed", "python", "python3", "echo", "tr", "diff", "dir", "type"}
_SHELL_FORBIDDEN = re.compile(r"[;`]|&&|\|\||\$\(|>|<\(|\brm\b|\bmv\b|\bcurl\b|\bwget\b")


def check_shell(command: str) -> str | None:
    """Reject anything but read-only inspection pipelines inside the workspace."""
    if _SHELL_FORBIDDEN.search(command):
        return "denied: only simple read-only commands and pipes (|) are allowed"
    for seg in command.split("|"):
        toks = seg.strip().split()
        if not toks:
            return "denied: empty command"
        if toks[0] not in SHELL_ALLOWED:
            return f"denied: command {toks[0]!r} not allowed"
        for t in toks[1:]:
            t2 = t.strip("'\"")
            if t2.startswith(("/", "~", "\\")) or ".." in t2 or re.match(r"^[A-Za-z]:", t2):
                return "denied: paths must be relative and inside the working directory"
    if re.search(r"\bsed\b.*\s-i", command):
        return "denied: in-place edits are not allowed (use write_file)"
    return None


def _make_tools(cwd: Path, state: dict):
    def list_files(path: str = ".") -> str:
        """List files under a directory (relative)."""
        base = (cwd / path).resolve()
        if not _inside(base, cwd):
            return "denied"
        items = [str(p.relative_to(cwd)) for p in sorted(base.rglob("*")) if p.is_file()]
        return "\n".join(items[:200]) + (f"\n... {len(items)} files" if len(items) > 200 else "")

    def read_file(path: str, max_chars: int = 4000) -> str:
        """Read a text file (relative path)."""
        p = (cwd / path).resolve()
        if not _inside(p, cwd) or not p.is_file():
            return "not found"
        return p.read_text(errors="replace")[: max(200, min(int(max_chars), 8000))]

    def write_file(path: str, content: str) -> str:
        """Overwrite a text file (relative path)."""
        p = (cwd / path).resolve()
        if not _inside(p, cwd):
            return "denied"
        if p.name in state.get("protected", ()):
            return "denied: tests must not be edited"
        p.write_text(content)
        return f"wrote {len(content)} chars"

    def run_python(code: str) -> str:
        """Run a Python 3 snippet in the directory and return stdout/stderr."""
        rc, out = sandbox.run_python(code, cwd)
        return f"rc={rc}\n{out[-2500:]}"

    def run_shell(command: str) -> str:
        """Run a read-only shell command (ls, grep, cat, head, wc, python3 file.py ...)."""
        bad = check_shell(command)
        if bad:
            return bad
        rc, out = sandbox.run_shell(command, cwd)
        return f"rc={rc}\n{out[-2500:]}"

    def submit(answer: str) -> str:
        """Submit the final answer."""
        state["answer"] = str(answer)
        return "submitted"

    return {"list_files": list_files, "read_file": read_file, "write_file": write_file,
            "run_python": run_python, "run_shell": run_shell, "submit": submit}


def _schemas(funcs: dict):
    """Tool schemas; built through openJiuwen's @tool when available."""
    out = []
    oj = {}
    try:
        from openjiuwen.core.foundation.tool import tool as oj_tool
        for name, f in funcs.items():
            lf = oj_tool(f, name=name, description=(f.__doc__ or name).strip())
            info = lf.card.tool_info()
            params = info.parameters
            if hasattr(params, "model_json_schema"):
                params = params.model_json_schema()
            out.append({"name": name, "description": info.description, "parameters": params})
            oj[name] = lf
        return out, oj
    except Exception as e:  # noqa: BLE001
        log(f"[agent] openJiuwen tool wrapping unavailable ({type(e).__name__}); plain schemas")
    import inspect
    for name, f in funcs.items():
        props, req = {}, []
        for pn, p in inspect.signature(f).parameters.items():
            t = "integer" if p.annotation in (int, "int") else "string"
            props[pn] = {"type": t}
            if p.default is inspect.Parameter.empty:
                req.append(pn)
        out.append({"name": name, "description": (f.__doc__ or name).strip(),
                    "parameters": {"type": "object", "properties": props, "required": req}})
    return out, {}


async def agent_solve(llm: LLM, task, cap: float, model=MODEL_FAST, max_steps: int = 12):
    """Generic tool-using agent with a hard per-task token cap.  Context is kept
    small: old tool outputs are truncated before every call."""
    cwd = sandbox.scratch_copy(task.ws)
    state: dict = {"protected": set()}
    test_file, targets = (None, [])
    if task.task_type == "coding":
        test_file, targets = coding_solver.locate(task.instruction, cwd)
        if test_file:
            state["protected"] = {test_file.name}
    funcs = _make_tools(cwd, state)
    schemas, oj = _schemas(funcs)
    msgs = [{"role": "system", "content": AGENT_SYS},
            {"role": "user", "content": clean_instruction(task.instruction) +
             "\n(Files are in the current directory.)"}]
    spent0 = task.cost
    try:
        for step in range(max_steps):
            if task.cost - spent0 > cap:
                log(f"{task.task_id}: agent cap reached")
                break
            # compress history: keep only the last 3 tool outputs in full
            tool_idx = [i for i, m in enumerate(msgs) if m["role"] == "tool"]
            for i in tool_idx[:-3]:
                if len(msgs[i]["content"]) > 300:
                    msgs[i]["content"] = msgs[i]["content"][:300] + " …[truncated]"
            r = await llm.chat(msgs, model=model, max_tokens=900, tools=schemas,
                               tag=f"agent:{task.task_id}:{step}")
            task.cost += r["cost"]
            if not r["tool_calls"]:
                txt = r["content"] or ""
                if is_infeasible(txt):
                    state["answer"] = "INFEASIBLE"
                    break
                msgs.append({"role": "assistant", "content": txt[:1500]})
                msgs.append({"role": "user", "content": "Use the tools; finish with submit()."})
                continue
            msgs.append({"role": "assistant", "content": r["content"] or "",
                         "tool_calls": [{"id": tc["id"], "type": "function",
                                         "function": {"name": tc["name"],
                                                      "arguments": tc["arguments"]}}
                                        for tc in r["tool_calls"]]})
            for tc in r["tool_calls"]:
                try:
                    args = json.loads(tc["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                name = tc["name"]
                try:
                    if name in oj:
                        res = await oj[name].invoke(args)
                        res = res if isinstance(res, str) else str(
                            getattr(res, "data", None) or res)
                    elif name in funcs:
                        res = funcs[name](**args)
                    else:
                        res = f"unknown tool {name}"
                except Exception as e:  # noqa: BLE001
                    res = f"error: {type(e).__name__}: {e}"
                msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": str(res)[:3000]})
            if "answer" in state:
                break
        ans = state.get("answer")
        if ans is None:
            return None
        if is_infeasible(ans):
            return ("infeasible", None, "agent: infeasible")
        if task.task_type == "coding":
            if test_file is None:
                return None
            ok, _ = await asyncio.to_thread(coding_solver.run_tests, cwd, test_file)
            if not ok:
                return None
            files = {p.relative_to(cwd).as_posix(): p.read_text(encoding="utf-8") for p in targets}
            return ("fixed", files, "agent")
        a = normalize(ans, task.check)
        return ("answer", a, "agent") if a else None
    except LLMUnavailable:
        raise
    finally:
        shutil.rmtree(cwd, ignore_errors=True)
