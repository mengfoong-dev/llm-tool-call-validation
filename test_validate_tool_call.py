"""Tests for validate_tool_call.

Runs with plain `python test_q1_validate_tool_call.py` (no pytest required) and
also under `pytest test_q1_validate_tool_call.py`.
"""

from validate_tool_call import validate_tool_call

CASES = [
    # (name, payload, expected_clean, expect_errors)
    # --- happy paths ---
    (
        "search: full payload",
        {"action": "search", "q": "refund policy", "k": 5},
        {"action": "search", "q": "refund policy", "k": 5},
        False,
    ),
    (
        "search: k defaults to 3",
        {"action": "search", "q": "refund policy"},
        {"action": "search", "q": "refund policy", "k": 3},
        False,
    ),
    (
        "answer: q is not part of the schema",
        {"action": "answer"},
        {"action": "answer", "k": 3},
        False,
    ),
    # --- coercion / normalisation ---
    (
        "trims action, q and numeric-string k",
        {"action": "  search ", "q": "  hello world  ", "k": " 4 "},
        {"action": "search", "q": "hello world", "k": 4},
        False,
    ),
    (
        "action matched case-insensitively, emitted canonically",
        {"action": "SEARCH", "q": "x"},
        {"action": "search", "q": "x", "k": 3},
        False,
    ),
    (
        "integral float k",
        {"action": "answer", "k": 2.0},
        {"action": "answer", "k": 2},
        False,
    ),
    (
        "integral numeric string k",
        {"action": "answer", "k": "2.0"},
        {"action": "answer", "k": 2},
        False,
    ),
    (
        "unknown keys are removed",
        {"action": "answer", "k": 1, "temperature": 0.7, "__proto__": "x"},
        {"action": "answer", "k": 1},
        False,
    ),
    (
        "answer: stray q ignored without error",
        {"action": "answer", "q": "ignored"},
        {"action": "answer", "k": 3},
        False,
    ),
    (
        "null k falls back to the default without error",
        {"action": "answer", "k": None},
        {"action": "answer", "k": 3},
        False,
    ),
    # --- recoverable: bad k -> default + error, call still usable ---
    (
        "k above range",
        {"action": "search", "q": "x", "k": 9},
        {"action": "search", "q": "x", "k": 3},
        True,
    ),
    ("k below range", {"action": "answer", "k": 0}, {"action": "answer", "k": 3}, True),
    (
        "k non-numeric string",
        {"action": "answer", "k": "many"},
        {"action": "answer", "k": 3},
        True,
    ),
    (
        "k fractional",
        {"action": "answer", "k": 3.5},
        {"action": "answer", "k": 3},
        True,
    ),
    (
        "k bool True is not 1",
        {"action": "answer", "k": True},
        {"action": "answer", "k": 3},
        True,
    ),
    (
        "k list",
        {"action": "answer", "k": [3]},
        {"action": "answer", "k": 3},
        True,
    ),
    # --- fatal: action ---
    ("action missing", {"q": "x"}, {}, True),
    ("action null", {"action": None}, {}, True),
    ("action unknown value", {"action": "delete", "q": "x"}, {}, True),
    ("action empty string", {"action": "   "}, {}, True),
    ("action wrong type", {"action": ["search"], "q": "x"}, {}, True),
    # --- fatal: q when searching ---
    ("search without q", {"action": "search"}, {}, True),
    ("search with null q", {"action": "search", "q": None}, {}, True),
    ("search with empty q", {"action": "search", "q": ""}, {}, True),
    ("search with whitespace-only q", {"action": "search", "q": "   "}, {}, True),
    ("search with non-string q", {"action": "search", "q": 123}, {}, True),
    # --- fatal: payload itself ---
    ("payload is a list", [{"action": "search"}], {}, True),
    ("payload is None", None, {}, True),
    ("payload is a string", '{"action": "search"}', {}, True),
    ("payload empty dict", {}, {}, True),
]


def run() -> int:
    failures = []
    for name, payload, expected_clean, expect_errors in CASES:
        clean, errors = validate_tool_call(payload)
        problems = []
        if clean != expected_clean:
            problems.append(f"clean={clean!r} expected {expected_clean!r}")
        if bool(errors) != expect_errors:
            problems.append(
                f"errors={errors!r} but expected "
                f"{'at least one error' if expect_errors else 'none'}"
            )
        if not isinstance(errors, list) or not all(
            isinstance(e, str) for e in errors
        ):
            problems.append(f"errors is not List[str]: {errors!r}")
        if problems:
            failures.append((name, problems))
        print(f"{'FAIL' if problems else 'pass'}  {name}")
        for problem in problems:
            print(f"        {problem}")

    # Invariants that are easier to assert directly than to table-drive.
    clean, errors = validate_tool_call({"action": "search", "q": "x", "k": 9})
    assert clean and errors, "a recoverable-only failure must still return a call"

    clean, errors = validate_tool_call({"action": "nope", "q": "", "k": 99})
    assert clean == {}, "fatal must return {}"
    assert len(errors) >= 2, f"expected action and k errors together, got {errors}"

    # Input is never mutated -- the caller may still need the raw payload for logs.
    payload = {"action": "  Search ", "q": " x ", "k": "4", "extra": 1}
    snapshot = dict(payload)
    validate_tool_call(payload)
    assert payload == snapshot, "validate_tool_call must not mutate its input"

    print(f"\n{len(CASES) - len(failures)}/{len(CASES)} cases passed")
    if failures:
        print(f"{len(failures)} FAILED: {[n for n, _ in failures]}")
        return 1
    print("invariants passed (recoverable-vs-fatal, multi-error, no mutation)")
    return 0


def test_all() -> None:
    """pytest entry point."""
    assert run() == 0


if __name__ == "__main__":
    raise SystemExit(run())
