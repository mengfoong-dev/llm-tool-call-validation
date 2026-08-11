"""Strictly validate and coerce untrusted LLM JSON into a tool-call schema.

Target schema:
    {
        "action": "search" | "answer",
        "q": non-empty str        # required iff action == "search"
        "k": int in [1, 5]        # optional, default 3
    }

Design notes (rationale for the judgement calls a bare schema leaves open):

* Fatal vs. recoverable. 'action' and 'q' carry intent -- if they are wrong we
  cannot guess what the model meant, so they are fatal and we return ({}, errors).
  'k' is a bounded tuning knob with a documented default, so a bad 'k' is
  recoverable: we fall back to the default, record the error, and still return a
  usable call. This keeps a single bad pagination hint from throwing away an
  otherwise valid tool call.
* Report everything at once. All errors we can determine are collected in one
  pass rather than failing on the first. In an agent loop the errors list is fed
  back to the model as a repair prompt, and one round-trip with complete
  feedback beats several with partial feedback.
* Lenient input, canonical output. 'action' is matched case-insensitively after
  trimming ("  Search " -> "search") because it is a closed enum, so widening
  the accepted input cannot introduce ambiguity. What we emit is always the
  exact canonical value.
* No invented data. A non-string 'q' (e.g. 123) is rejected rather than
  stringified -- coercing it would fabricate a search query the model never
  wrote. Numeric coercion is confined to 'k', where the brief asks for it.
"""

from typing import Any, Dict, List, Optional, Tuple

VALID_ACTIONS = ("search", "answer")
K_MIN = 1
K_MAX = 5
K_DEFAULT = 3


def _coerce_int(value: Any) -> Optional[int]:
    """Best-effort integer coercion. Returns None if `value` is not an integer.

    Accepts ints, integral floats (3.0), and numeric strings (" 4 ", "4.0").
    Rejects bools: `bool` subclasses `int` in Python, so a bare isinstance check
    would silently turn `True` into k=1.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str):
        text = value.strip()
        try:
            return int(text)
        except ValueError:
            pass
        try:
            number = float(text)
        except ValueError:
            return None
        return int(number) if number.is_integer() else None
    return None


def validate_tool_call(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """
    Returns (clean, errors). 'clean' strictly follows the schema with defaults applied.
    - Trim strings; coerce numeric strings to ints.
    - Remove unknown keys.
    - If action=='answer', ignore 'q' if present (no error).
    - On fatal errors (e.g., missing/invalid 'action', or missing/empty 'q' for search),
      return ({}, errors).
    """
    if not isinstance(payload, dict):
        return {}, [f"payload: expected an object, got {type(payload).__name__}"]

    errors: List[str] = []
    clean: Dict[str, Any] = {}
    fatal = False

    # --- action: required, closed enum -----------------------------------
    action: Optional[str] = None
    if "action" not in payload:
        errors.append("action: required")
        fatal = True
    else:
        raw_action = payload["action"]
        if not isinstance(raw_action, str):
            errors.append(f"action: expected str, got {type(raw_action).__name__}")
            fatal = True
        else:
            candidate = raw_action.strip().lower()
            if candidate in VALID_ACTIONS:
                action = candidate
                clean["action"] = candidate
            else:
                errors.append(
                    f"action: expected one of {list(VALID_ACTIONS)}, "
                    f"got {raw_action.strip()!r}"
                )
                fatal = True

    # --- q: required iff action == "search" -------------------------------
    # When action is invalid we cannot know whether 'q' is required, so we skip
    # it rather than emit a second, possibly bogus error.
    if action == "search":
        raw_q = payload.get("q")
        if raw_q is None:
            errors.append("q: required when action == 'search'")
            fatal = True
        elif not isinstance(raw_q, str):
            errors.append(f"q: expected str, got {type(raw_q).__name__}")
            fatal = True
        else:
            q = raw_q.strip()
            if q:
                clean["q"] = q
            else:
                errors.append("q: must be non-empty when action == 'search'")
                fatal = True
    # action == "answer": 'q' is not part of the schema, so it is dropped
    # silently along with any other unknown key (no error, per the brief).

    # --- k: optional, int in [1, 5], default 3 ---------------------------
    # Recoverable: an unusable 'k' falls back to the default and is reported.
    if payload.get("k") is None:
        clean["k"] = K_DEFAULT
    else:
        k = _coerce_int(payload["k"])
        if k is None or not (K_MIN <= k <= K_MAX):
            errors.append(
                f"k: expected an int in [{K_MIN}, {K_MAX}], got "
                f"{payload['k']!r}; using default {K_DEFAULT}"
            )
            k = K_DEFAULT
        clean["k"] = k

    # Unknown keys are never copied into `clean`: we build the output key by key
    # from the schema (allowlist) instead of deleting from the input (denylist),
    # so a key we have never seen cannot leak through.

    if fatal:
        return {}, errors
    return clean, errors
