"""Polynomials in x with integer coefficients: parse, write, differentiate, integrate, factor.

The three math tools (``differentiate``, ``integrate``, ``factor_polynomial``) run on
this, and execution match compares their results, so the written form is canonical:
two polynomials are equal exactly when their texts are.

Input (:func:`parse`): one variable, ``x``; integer coefficients; powers written ``x^2``
or ``x**2``; an optional ``*`` between a coefficient and ``x``; spaces anywhere; an
optional left side (``y =``, ``f(x) =``) or right side ``= 0`` is dropped, and so is
one pair of brackets around the whole polynomial; ``+ -6`` reads as ``- 6``. Anything
else (another letter, a capital X, ``x²``, a unicode minus, a decimal, a coefficient
after ``x``, inner brackets, another double sign, ``+ C``) is a :class:`PolyError`.

Output (:func:`to_text`): terms by falling power, ``3x^2 - x + 5``; a coefficient 1 is
not written; a fraction is written in brackets, ``(1/3)x^3``; the zero polynomial is
``0``. :func:`factor_text` writes ``(x + 2)(x - 3)``: factors sorted by root, a
repeated root as a power, ``(x - 1)^2``, a leading constant in front, ``2(x - 1)``.

Pure Python, no dependency; tested in ``tests/test_poly.py``.
"""

from __future__ import annotations

import re
from fractions import Fraction

MAX_DEGREE = 12
MAX_COEFFICIENT = 10**9

Poly = dict  # power -> coefficient (int or Fraction), zero coefficients left out


class PolyError(ValueError):
    """Not a polynomial in x with integer coefficients (as this module reads them)."""


_TERM = re.compile(r"^(\d+)?(\*?x(?:\^(\d+))?)?$")


def parse(text: str) -> dict[int, int]:
    """``"3x^2 - 4x + 1"`` -> ``{2: 3, 1: -4, 0: 1}``. Like terms are added."""
    if not isinstance(text, str):
        raise PolyError("the polynomial must be a string")
    if text.count("=") > 1:
        raise PolyError("more than one '='")
    if "=" in text:
        left, right = text.split("=")
        if "".join(left.split()) in ("y", "f(x)", "p(x)"):
            text = right
        elif "".join(right.split()) == "0":
            text = left  # an equation: "x^2 - x - 6 = 0"
        else:
            raise PolyError(f"unexpected left side: {left.strip()!r}")
    s = "".join(text.split()).replace("**", "^")
    if s.startswith("(") and s.endswith(")") and "(" not in s[1:-1] and ")" not in s[1:-1]:
        s = s[1:-1]  # one pair of brackets around the whole polynomial
    s = s.replace("+-", "-")  # "+ -6" is "- 6"
    if not s:
        raise PolyError("empty polynomial")
    if s.startswith("+"):
        s = s[1:]
    parts = re.split(r"([+-])", s)
    if parts[0] == "":  # a leading minus
        parts = parts[1:]
        signs, bodies = [parts[0]], [parts[1]] if len(parts) > 1 else [""]
        rest = parts[2:]
    else:
        signs, bodies, rest = ["+"], [parts[0]], parts[1:]
    for i in range(0, len(rest), 2):
        signs.append(rest[i])
        bodies.append(rest[i + 1] if i + 1 < len(rest) else "")
    out: dict[int, int] = {}
    for sign, body in zip(signs, bodies, strict=True):
        m = _TERM.match(body)
        if not body or not m or (m.group(1) is None and m.group(2) is None):
            raise PolyError(f"not a term: {body!r}")
        if m.group(1) is None and m.group(2).startswith("*"):
            raise PolyError(f"not a term: {body!r}")
        coef = int(m.group(1)) if m.group(1) is not None else 1
        power = 0 if m.group(2) is None else (int(m.group(3)) if m.group(3) else 1)
        if power > MAX_DEGREE or coef > MAX_COEFFICIENT:
            raise PolyError("degree or coefficient too large")
        out[power] = out.get(power, 0) + (coef if sign == "+" else -coef)
    return {p: c for p, c in out.items() if c != 0}


def _coef_text(c: Fraction | int, power: int) -> str:
    c = abs(Fraction(c))
    if power and c == 1:
        return ""
    if c.denominator == 1:
        return str(c.numerator)
    return f"({c.numerator}/{c.denominator})"


def to_text(p: dict) -> str:
    """The canonical written form (see the module docstring)."""
    terms = [(power, Fraction(c)) for power, c in sorted(p.items(), reverse=True) if c != 0]
    if not terms:
        return "0"
    out = ""
    for i, (power, c) in enumerate(terms):
        body = _coef_text(c, power) + ("x" if power else "") + (f"^{power}" if power > 1 else "")
        if i == 0:
            out = ("-" if c < 0 else "") + body
        else:
            out += (" - " if c < 0 else " + ") + body
    return out


def normalize(text: str) -> str:
    """:func:`parse` then :func:`to_text`."""
    return to_text(parse(text))


def differentiate(p: dict) -> dict:
    return {k - 1: c * k for k, c in p.items() if k > 0}


def integrate(p: dict) -> dict:
    """The antiderivative with constant 0; coefficients become fractions where needed."""
    out = {}
    for k, c in p.items():
        v = Fraction(c, k + 1)
        out[k + 1] = v.numerator if v.denominator == 1 else v
    return out


def _divisors(n: int) -> list[int]:
    n = abs(n)
    small = [d for d in range(1, int(n**0.5) + 1) if n % d == 0]
    return sorted({*small, *(n // d for d in small)})


def factor(p: dict) -> tuple[int, list[int]]:
    """``(lead, roots)`` with p = lead * prod(x - r), roots ascending, repeats kept.

    Raises :class:`PolyError` unless every factor is linear with an integer root
    (degree at least 1)."""
    degree = max(p, default=0)
    if not p or degree < 1:
        raise PolyError("nothing to factor: a constant")
    coeffs = [Fraction(p.get(k, 0)) for k in range(degree, -1, -1)]  # high to low
    roots: list[int] = []
    while len(coeffs) > 1:
        if coeffs[-1] == 0:
            root = 0
        else:
            lead_free = coeffs[-1] / coeffs[0]
            if lead_free.denominator != 1:
                raise PolyError("no integer-root factorisation")
            root = None
            for d in _divisors(int(lead_free)):
                for r in (d, -d):
                    value = Fraction(0)
                    for c in coeffs:
                        value = value * r + c
                    if value == 0:
                        root = r
                        break
                if root is not None:
                    break
            if root is None:
                raise PolyError("no integer-root factorisation")
        quotient, acc = [], Fraction(0)
        for c in coeffs[:-1]:
            acc = acc * root + c
            quotient.append(acc)
        roots.append(root)
        coeffs = quotient
    lead = coeffs[0]
    if lead.denominator != 1:
        raise PolyError("no integer-root factorisation")
    return int(lead), sorted(roots)


def factor_text(lead: int, roots: list[int]) -> str:
    """``(1, [-2, 3])`` -> ``"(x + 2)(x - 3)"``; repeats as powers; lead in front."""
    out = "" if lead == 1 else ("-" if lead == -1 else str(lead))
    i = 0
    while i < len(roots):
        r = roots[i]
        n = roots.count(r)
        body = "x" if r == 0 else (f"x - {r}" if r > 0 else f"x + {-r}")
        out += f"({body})" + (f"^{n}" if n > 1 else "")
        i += n
    return out
