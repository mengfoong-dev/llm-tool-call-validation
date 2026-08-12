# Validating untrusted LLM tool-call JSON

[![tests](https://github.com/mengfoong-dev/llm-tool-call-validation/actions/workflows/test.yml/badge.svg)](https://github.com/mengfoong-dev/llm-tool-call-validation/actions/workflows/test.yml)

A small, dependency-free validator that coerces untrusted model output into a
strict tool-call schema — and, more to the point, a worked argument about the
decisions a schema alone doesn't specify.

The function is ~70 lines. The interesting part is *why* it behaves the way it
does at the edges, because that boundary is where an LLM stops being a source of
text and becomes a source of instructions. Getting it wrong doesn't throw — it
quietly executes something the model never asked for.

```python
>>> validate_tool_call({"action": "  Search ", "q": " refund policy ", "k": "4"})
({'action': 'search', 'q': 'refund policy', 'k': 4}, [])

>>> validate_tool_call({"action": "search", "k": 9})
({}, ["q: required when action == 'search'",
      'k: expected an int in [1, 5], got 9; using default 3'])
```

## The schema

| Field | Type | Required | Notes |
|---|---|---|---|
| `action` | `"search"` \| `"answer"` | yes | closed enum |
| `q` | non-empty `str` | **iff** `action == "search"` | trimmed |
| `k` | `int` in `[1, 5]` | no | default `3` |

`validate_tool_call(payload)` returns `(clean, errors)`. `clean` strictly follows
the schema with defaults applied; `errors` is a list of human-readable strings.
Strings are trimmed, numeric strings are coerced to ints, unknown keys are
dropped, and on a fatal error `clean` is `{}`.

## The seven decisions

**1. Fatal vs. recoverable, and why the line sits where it does.**
`action` and `q` carry the model's *intent* — if they're wrong there is nothing to
guess, so they're fatal and the call is rejected wholesale. `k` is a bounded
tuning knob with a documented default, so a bad `k` is recoverable: fall back to
the default, record the error, and still return a usable call. One malformed
pagination hint shouldn't discard an otherwise valid tool call.

This means `errors` can be non-empty while `clean` is populated. That's
deliberate, and it's visible rather than hidden — a caller that wants strict
behaviour treats any non-empty `errors` as a rejection.

**2. Report every error in one pass.**
In an agent loop the `errors` list is fed back to the model as a repair prompt. One
round-trip carrying complete feedback beats three carrying partial feedback, so
all determinable errors accumulate instead of returning on the first. Messages
name the offending field and the offending value, because that string ends up in a
prompt and the model has to act on it.

**3. Lenient input, canonical output.**
`action` is matched case-insensitively after trimming, so `"  Search "` is
accepted. Widening the input is safe *specifically because `action` is a closed
enum* — there's no ambiguity to introduce. What gets emitted is always the exact
canonical value. The same leniency would not be safe on a free-text field.

**4. No invented data.**
A non-string `q` (say `123`) is rejected rather than stringified. Coercing it
would fabricate a search query the model never wrote, and a silently wrong query
is worse than a loud failure — it returns plausible results for a question nobody
asked. Numeric coercion stays confined to `k`, where the schema genuinely calls
for it.

**5. Allowlist, not denylist.**
"Remove unknown keys" is implemented by building the output key-by-key from the
schema, not by deleting known-bad keys from a copy of the input. The difference
only shows up as the schema grows: with an allowlist, a field nobody anticipated
*cannot* reach the tool. With a denylist, it reaches the tool until someone
remembers to add it.

**6. `bool` is not `int`.**
`bool` subclasses `int` in Python, so `isinstance(True, int)` is `True` and a
naive check silently accepts `k=True` as `k=1`. Explicitly rejected — this is the
bug most implementations of this exercise ship with.

**7. The input is never mutated.**
Callers usually still need the raw payload for logging and replay. There's a test
pinning this, because it's the kind of property that quietly breaks during a
refactor.

## Tests

30 cases plus three invariants, no dependencies:

```
$ python test_validate_tool_call.py
30/30 cases passed
invariants passed (recoverable-vs-fatal, multi-error, no mutation)
```

Beyond the happy paths and the specified coercions, the suite pins the cases that
break naive implementations:

| Case | Expected |
|---|---|
| `k: True` | `3` + error — not `1` |
| `k: 3.5` / `"many"` / `[3]` | `3` + error |
| `k: 2.0` / `"2.0"` | `2`, no error |
| `k: None` | `3`, no error — absent ≠ invalid |
| `q: 123` on search | `({}, errors)` — never `"123"` |
| `q: "   "` on search | `({}, errors)` |
| `action: "  Search "` | `"search"` |
| `action: ["search"]` / `None` / `"delete"` | `({}, errors)` |
| `payload` is a list / `None` / a JSON *string* | `({}, errors)` |
| `{"action": "nope", "k": 99}` | `{}` **and ≥2 errors** — they accumulate |
| any valid call with extra keys | input dict unchanged afterwards |

Runs under `pytest` too, and on Python 3.9+.

## What this deliberately doesn't do

Three things I'd add before running this in production, left out to keep the
example focused:

**Emit metrics, not just errors.** Per-field rejection rates broken down by model
version are the cheapest early warning that a prompt change or model upgrade has
regressed. A jump in `action` rejections after a deploy belongs on a dashboard,
not in a log file nobody reads.

**Generate the schema rather than hand-writing it.** At one or two tools,
hand-written validation is clearer than a framework. Past roughly five, define the
schema once (Pydantic) and derive three artifacts from it: this validator, the
JSON Schema sent to the model as the tool definition, and the docs. Hand-written
validators and hand-written tool definitions drift apart, and the drift is silent
— the model gets told about a field the validator drops.

**Distinguish "malformed" from "not JSON at all."** Upstream of this function,
output truncated by a token limit needs a different repair strategy (raise the
budget, or continue) than well-formed JSON with a bad enum (re-prompt with the
errors). Retrying the wrong one burns tokens and fails anyway.

## License

MIT — see [LICENSE](LICENSE).
