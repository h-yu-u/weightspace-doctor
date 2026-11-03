"""Small, objective smoke probe suites. Not a claim of benchmark coverage.

Use representative, pre-registered domain cases for substantive validation.
These fixtures are intentionally deterministic and free of customer data.
"""

from .schema import Case


def built_in_cases(profile: str = "general") -> list[Case]:
    if profile not in ("general", "math", "code"):
        raise ValueError("profile must be general, math or code")
    math = [
        Case("math-add", "What is 19 + 23? Reply with digits only.", "42", tags=("math",)),
        Case("math-multiply", "What is 17 * 13? Reply with digits only.", "221", tags=("math",)),
        Case("math-divide", "What is 144 / 12? Reply with digits only.", "12", tags=("math",)),
        Case("math-square", "What is 23 squared? Reply with digits only.", "529", tags=("math",)),
        Case("math-subtract", "What is 103 - 47? Reply with digits only.", "56", tags=("math",)),
        Case("math-fractions", "What is 1/2 + 1/4? Reply as a reduced fraction.",
             "3/4", tags=("math", "reasoning")),
        Case("math-series", "Compute the sum of the integers 1 through 20. "
             "Reply with digits only.", "210", tags=("math", "reasoning")),
        Case("math-equation", "Solve 3x + 7 = 28 for x. Reply with digits only.",
             "7", tags=("math", "reasoning")),
    ]
    guards = [
        Case("guard-uppercase", "Return exactly one word: write 'cat' in uppercase.",
             "CAT", split="guard", tags=("instruction",)),
        Case("guard-lowercase", "Return exactly one word: write 'HELLO' in lowercase.",
             "hello", split="guard", tags=("instruction",)),
        Case("guard-digit", "Return only the decimal integer corresponding to the binary number 1010.",
             "10", split="guard", tags=("instruction", "math")),
        Case("guard-json", "Return exactly the JSON object with one key 'ok' and boolean value true."
             " No markdown.", expected_regex=r"^\s*\{\s*\"ok\"\s*:\s*true\s*\}\s*$",
             split="guard", tags=("json", "instruction")),
        Case("guard-reverse", "Reverse the ASCII word 'straw' and output only the result.",
             "warts", split="guard", tags=("instruction",)),
    ]
    code = [
        Case("code-python-range", "What is the exact output of Python: print(sum(range(5)))? "
             "Reply with digits only.", "10", tags=("code", "python")),
        Case("code-python-slice", "What is the exact output of Python: "
             "print('abcdef'[2:5])? Reply with the characters only.",
             "cde", tags=("code", "python")),
        Case("code-python-len", "What is the output of Python: print(len(set([1, 1, 3])))? "
             "Reply with digits only.", "2", tags=("code", "python")),
        Case("code-js-array", "What is the output of JavaScript: console.log([1,2,3].length)? "
             "Reply with digits only.", "3", tags=("code", "javascript")),
        Case("code-js-template", "What is the output of JavaScript: "
             "console.log(`a${2 + 3}b`)? Reply with the characters only.",
             "a5b", tags=("code", "javascript")),
        Case("code-python-bool", "What does Python print: print(3 > 7)? "
             "Reply with the exact Python boolean spelling only.",
             "False", tags=("code", "python")),
        Case("code-python-mod", "What is the output of Python: print(17 % 5)? "
             "Reply with digits only.", "2", tags=("code", "python")),
        Case("code-json-parser", "Return exactly valid JSON: a key named 'count' "
             "with integer value 3. No markdown.",
             expected_regex=r"^\s*\{\s*\"count\"\s*:\s*3\s*\}\s*$",
             tags=("code", "json")),
    ]
    # Use the profile to choose target-quality tests while retaining unrelated
    # guard probes.  Not all cases require the same model output style.
    if profile == "math":
        return math + guards
    if profile == "code":
        return code + guards
    return math[:5] + code[:3] + guards
