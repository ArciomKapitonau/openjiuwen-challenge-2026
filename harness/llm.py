"""LLM access through openJiuwen with strict budget accounting.

Primary path: openjiuwen.core.foundation.llm.Model (OpenAI-compatible client).
Safety net: if the openJiuwen client cannot be constructed or raises a
non-HTTP error, a minimal urllib OpenAI-compatible client is used so that a
framework incompatibility never zeroes the run.  Both paths go through the
same BudgetLedger, so every token is accounted.
"""
from __future__ import annotations

import asyncio
import contextlib
import contextvars
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import os
import re
import time
import urllib.error
import uuid
from typing import Any

from .budget import BudgetLedger, estimate_tokens
from .log import log

MODEL_FAST = "qwen3.8-27b-noreason"        # default: dense 27B, no thinking
MODEL_FAST_ALT = "qwen3.6-35b-a3b-noreason"
MODEL_REASON = "qwen3.8-27b"               # escalation only, capped max_tokens
MODEL_REASON_ALT = "qwen3.6-35b-a3b"
ALL_MODELS = [MODEL_FAST, MODEL_FAST_ALT, MODEL_REASON, MODEL_REASON_ALT]
# if a model is unavailable (403/404), use the closest one instead
FALLBACKS = {MODEL_FAST: [MODEL_FAST_ALT], MODEL_FAST_ALT: [MODEL_FAST],
             MODEL_REASON: [MODEL_REASON_ALT, MODEL_FAST], MODEL_REASON_ALT: [MODEL_REASON, MODEL_FAST_ALT]}

_BUDGET_ERR = re.compile(r"budget|quota|exhaust|insufficient (?:funds|balance|credit)|\b402\b|payment required|token limit reached", re.I)


# Per-scope spending cap (one attempt, one vote, one diagnosis).  Set by the
# Executor with ``cap_scope``; every call inside the scope is clamped so that
# the scope's reservations never exceed its cap.
_SCOPE: contextvars.ContextVar[dict | None] = contextvars.ContextVar("llm_cap_scope",
                                                                   default=None)
MIN_CALL_OUTPUT = 30


@contextlib.contextmanager
def cap_scope(cap: float):
    """Limit the total cost of LLM calls made inside this block to ``cap`` units."""
    sc = {"cap": float(cap), "spent": 0.0, "pending": 0.0}
    tok = _SCOPE.set(sc)
    try:
        yield sc
    finally:
        _SCOPE.reset(tok)


class LLMUnavailable(Exception):
    pass


class ProxyError(Exception):
    """HTTP error of the raw transport with the provider's error code kept, so
    that e.g. 429 insufficient_quota is not retried as a rate limit and
    400 model_not_found switches the model."""

    def __init__(self, status: int, body: str, headers=None):
        self.status_code = status
        self.headers = headers
        code = message = ""
        try:
            err = json.loads(body).get("error") or {}
            if isinstance(err, dict):
                code = str(err.get("code") or err.get("type") or "")
                message = str(err.get("message") or "")
            else:
                message = str(err)
        except (ValueError, AttributeError):
            message = body[:300]
        self.code = code
        super().__init__(f"HTTP error {status} {code} {message}"[:500])


class _SetupError(Exception):
    """The SDK failed before a request could be sent."""


def classify_error(e) -> str:
    """quota: shared budget exhausted -> stop everything
    model: this model is not available (403/404 or unknown model) -> switch model
    transient: 408/409/425/429/5xx/connection problems -> back off and retry
    fatal: other 4xx (bad request, auth) -> do not retry this request"""
    if e is None:
        return "transient"
    text = str(e)
    code = str(getattr(e, "code", "") or "").lower()
    if _BUDGET_ERR.search(text) or code in {"insufficient_quota", "billing_hard_limit_reached"}:
        return "quota"
    status = _http_status(e)
    if code in {"model_not_found", "model_not_available", "invalid_model"}:
        return "model"
    if status is None:
        return "transient"
    if status in (403, 404) or (status == 400 and re.search(r"model", text, re.I)
                                and re.search(r"not (found|exist|available|support)", text, re.I)):
        return "model"
    if status in (408, 409, 425, 429) or status >= 500:
        return "transient"
    return "fatal"


def _http_status(e) -> int | None:
    if e is None:
        return None
    if isinstance(e, urllib.error.HTTPError):
        return e.code
    response = getattr(e, "response", None)
    if isinstance(getattr(response, "status_code", None), int):
        return response.status_code
    for attr in ("status_code", "status", "code"):
        v = getattr(e, attr, None)
        if isinstance(v, int) and 100 <= v < 600:
            return v
    m = re.search(r"(?:status(?:_code)?|error code|http error)\D{0,6}([1-5]\d\d)\b", str(e), re.I)
    return int(m.group(1)) if m else None


class LLM:
    def __init__(self, ledger: BudgetLedger, max_concurrency: int | None = None, *,
                 time_left=None):
        if max_concurrency is None:
            # the organizers' proxy answers 429 above 3 parallel requests per account
            try:
                max_concurrency = max(1, int(os.environ.get("LLM_CONCURRENCY", "3")))
            except ValueError:
                max_concurrency = 3
        self.ledger = ledger
        self.base = (os.environ.get("LLM_BASE_URL") or "").strip().rstrip("/").removesuffix("/chat/completions")
        self.key = os.environ.get("LLM_API_KEY") or ""
        self.sem = asyncio.Semaphore(max_concurrency)
        self.enabled = bool(self.base)
        self._oj_models: dict[str, Any] = {}
        self._oj_ok = True
        self.time_left = time_left or (lambda: float("inf"))
        self.bad_models: set[str] = set()
        self.backend_used = "none"

    # ------------------------------------------------------------ openJiuwen
    def _oj_model(self, name: str):
        if name in self._oj_models:
            return self._oj_models[name]
        from openjiuwen.core.foundation.llm import (Model, ModelClientConfig,
                                                    ModelRequestConfig)
        client_cfg = ModelClientConfig(
            client_provider="OpenAI",
            api_base=self.base,
            api_key=self.key or "EMPTY",
            verify_ssl=self.base.startswith("https"),
            timeout=180.0, max_retries=0,
        )
        req_cfg = ModelRequestConfig(model=name, temperature=0.0)
        m = Model(client_cfg, req_cfg)
        self._oj_models[name] = m
        return m

    async def _call_oj(self, model: str, messages, max_tokens, tools):
        try:
            m = self._oj_model(model)
        except Exception as error:
            raise _SetupError() from error
        resp = await m.invoke(messages, tools=tools or None, model=model,
                              temperature=0.0, max_tokens=max_tokens)
        u = resp.usage_metadata
        tool_calls = []
        for tc in (resp.tool_calls or []):
            tool_calls.append({"id": tc.id or f"call_{len(tool_calls)}",
                               "name": tc.name, "arguments": tc.arguments or "{}"})
        content = resp.content if isinstance(resp.content, str) else json.dumps(resp.content)
        usage = None
        if u is not None and (u.input_tokens or u.output_tokens):
            usage = (u.input_tokens, u.output_tokens, u.reasoning_tokens or 0)
        return content or "", tool_calls, usage, resp.reasoning_content or ""

    # ---------------------------------------------------------------- raw HTTP
    async def _call_raw(self, model, messages, max_tokens, tools):
        body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0}
        if tools:
            body["tools"] = tools
        try:
            import httpx
        except ImportError:      # stdlib safety net: never depend on one HTTP client
            data = await asyncio.to_thread(self._post_urllib, body)
        else:
            async with httpx.AsyncClient(follow_redirects=False, trust_env=False) as client:
                response = await client.post(
                    self.base + "/chat/completions", json=body, timeout=180,
                    headers={"Authorization": f"Bearer {self.key}"},
                )
                if response.status_code >= 400:
                    raise ProxyError(response.status_code, response.text, response.headers)
                data = response.json()
        msg = data["choices"][0]["message"]
        tool_calls = [
            {"id": tc.get("id") or f"call_{i}", "name": tc["function"].get("name", ""),
             "arguments": tc["function"].get("arguments") or "{}"}
            for i, tc in enumerate(msg.get("tool_calls") or [])
        ]
        usage = data.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        return (msg.get("content") or "", tool_calls,
                (usage.get("prompt_tokens"), usage.get("completion_tokens"),
                 details.get("reasoning_tokens") or 0),
                msg.get("reasoning_content") or msg.get("reasoning") or "")

    def _post_urllib(self, body: dict) -> dict:
        import urllib.request
        req = urllib.request.Request(
            self.base + "/chat/completions", data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(req, timeout=180) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise ProxyError(e.code, e.read().decode("utf-8", "replace"), e.headers) from None

    # ------------------------------------------------------------------ public
    def _fallback_model(self, model: str) -> str | None:
        chain = FALLBACKS.get(model, [])
        for m in chain:
            if m not in self.bad_models:
                return m
        return None

    async def chat(self, messages: list[dict], *, model: str = MODEL_FAST,
                   max_tokens: int = 400, tools: list[dict] | None = None,
                   tag: str = "") -> dict:
        """Return one answer and the total cost of its transport attempts."""
        prompt = json.dumps([messages, tools], ensure_ascii=False)
        estimated_input = estimate_tokens(prompt)
        schema = [{"type": "function", "function": t} for t in tools] if tools else None
        cost = 0.0
        tries = 0
        while tries < 4:
            if model in self.bad_models:
                model = self._fallback_model(model)
                if model is None:
                    raise LLMUnavailable("no available model")
            transport = "openjiuwen" if self._oj_ok else "raw"
            requested = max_tokens if "noreason" in model else min(2500, max(max_tokens, 1500))
            result, error, debit = await self._request(
                transport, messages, model, requested, schema, tag, estimated_input
            )
            cost += debit
            if isinstance(error, _SetupError):
                self._oj_ok = False
                log("[llm] openJiuwen initialization failed; using HTTP transport")
                continue
            tries += 1
            if result is not None:
                self.backend_used = transport
                return {"content": result[0], "tool_calls": result[1], "cost": cost, "model": model}
            kind, status = classify_error(error), _http_status(error)
            if kind == "quota" or status == 401:
                self.ledger.mark_dead()
                raise LLMUnavailable("quota" if kind == "quota" else "proxy authentication failed")
            if kind == "model":
                self.bad_models.add(model)
                continue
            if kind != "transient" or tries == 4:
                raise LLMUnavailable(f"proxy {kind} error" + (f" (HTTP {status})" if status else ""))
            # A failed request and its transport fallback are two reservations.
            if transport == "openjiuwen" and status is None:
                self._oj_ok = False
            delay = (5.0 if status == 429 else 1.5) * tries
            response = getattr(error, "response", None)
            headers = getattr(response, "headers", None) or getattr(error, "headers", None)
            if headers and (retry := headers.get("Retry-After")):
                try:
                    seconds = int(retry) if retry.isdigit() else (
                        parsedate_to_datetime(retry) - datetime.now(timezone.utc)
                    ).total_seconds()
                    delay = max(delay, seconds)
                except (ValueError, OverflowError, TypeError):
                    pass
            if delay >= self.time_left():
                raise LLMUnavailable("deadline")
            await asyncio.sleep(delay)
        raise LLMUnavailable("no available model")

    def _commit(self, scope, reserve, usage, meta):
        cost = self.ledger.commit(reserve, *usage, meta)
        if scope is not None:
            scope["pending"] = max(0.0, scope["pending"] - reserve)
            scope["spent"] += cost
        return cost

    async def _request(self, transport, messages, model, requested, tools, tag, estimated_input):
        # Wait for a transport slot before reserving. Cancellation in the queue
        # leaves neither a charge nor a reservation.
        async with self.sem:
            if self.time_left() <= 0:
                raise LLMUnavailable("deadline")
            if not self.enabled or self.ledger.dead:
                raise LLMUnavailable("proxy unavailable")
            scope = _SCOPE.get()
            input_cost = estimated_input / 10
            output = requested
            if scope is not None:
                left = scope["cap"] - scope["spent"] - scope["pending"] - input_cost
                if left < MIN_CALL_OUTPUT:
                    raise LLMUnavailable("cap")
                output = min(output, int(left))
            reserve = input_cost + output
            if not self.ledger.try_reserve(reserve):
                raise LLMUnavailable("budget")
            if scope is not None:
                scope["pending"] += reserve
            meta = {"request_id": uuid.uuid4().hex, "tag": tag,
                    "model": model, "transport": transport, "reserved": reserve}
            start = time.monotonic()
            call = self._call_oj if transport == "openjiuwen" else self._call_raw
            try:
                async with asyncio.timeout(min(180, self.time_left())):
                    result = await call(model, messages, output, tools)
            except _SetupError as error:
                self.ledger.release(reserve)
                if scope is not None:
                    scope["pending"] = max(0.0, scope["pending"] - reserve)
                return None, error, 0.0
            except asyncio.CancelledError:
                self._commit(scope, reserve, (estimated_input, output, 0),
                             {**meta, "accounting": "reserved", "error": "cancelled",
                              "sec": round(time.monotonic() - start, 2)})
                raise
            except Exception as error:
                status = _http_status(error)
                # Explicit client rejections have no generated answer. The
                # input may still be billed. Unclear outcomes debit the reserve.
                input_only = status in {400, 401, 402, 403, 404, 413, 422, 429}
                debit = self._commit(scope, reserve, (estimated_input, 0 if input_only else output, 0),
                                     {**meta, "accounting": "estimated_input" if input_only else "reserved",
                                      "error": classify_error(error), "http_status": status,
                                      "sec": round(time.monotonic() - start, 2)})
                return None, error, debit

            valid_response = isinstance(result, tuple) and len(result) == 4
            usage = result[2] if valid_response else None
            measured = (isinstance(usage, (tuple, list)) and len(usage) == 3
                        and all(type(v) is int and v >= 0 for v in usage) and usage[0] > 0
                        and (not self.ledger.reasoning_in_output or usage[2] <= usage[1]))
            if not measured:
                usage = (estimated_input, output, 0)
                self.ledger.mark_dead()
                meta["error"] = "missing or inconsistent usage"
            elif usage[1] + (0 if self.ledger.reasoning_in_output else usage[2]) > output:
                self.ledger.mark_dead()
                meta["error"] = "provider exceeded output cap"
            debit = self._commit(scope, reserve, usage,
                                 {**meta, "accounting": "usage" if measured else "reserved",
                                  "sec": round(time.monotonic() - start, 2)})
            if not valid_response:
                return None, ValueError("invalid proxy response"), debit
            return result, None, debit
