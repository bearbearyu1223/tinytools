"""The polynomial normaliser behind differentiate, integrate and factor_polynomial."""

from fractions import Fraction

import pytest

from tinytools import poly


@pytest.mark.parametrize("text, want", [
    ("3x^2 - 4x + 1", {2: 3, 1: -4, 0: 1}),
    ("3*x**2-4*x+1", {2: 3, 1: -4, 0: 1}),
    ("  -x^3 + x ", {3: -1, 1: 1}),
    ("+5", {0: 5}),
    ("x + x", {1: 2}),
    ("x - x", {}),
    ("y = 2x^2", {2: 2}),
    ("f(x) = x^2 + 1", {2: 1, 0: 1}),
    ("0", {}),
])
def test_parse(text, want):
    assert poly.parse(text) == want


@pytest.mark.parametrize("text", ["", "x^", "2y", "x^2 + y", "1.5x", "(x + 1)(x - 2)", "x--1",
                                  "x^-2", "*x", "2*", "g(x) = x", "x = 1 = 2", "x^99", "sin x"])
def test_parse_refuses(text):
    with pytest.raises(poly.PolyError):
        poly.parse(text)


@pytest.mark.parametrize("text", ["x^2 - x - 6 = 0", "(x^2 - x - 6)", "x^2 - x + -6",
                                  "( x^2 - x - 6 ) = 0", "f(x) = (x^2 - x - 6)",
                                  "x^2 + -x - 6"])
def test_parse_reads_what_a_model_might_write(text):
    """T1b review 5: '= 0', one pair of outer brackets and '+ -6' are the same polynomial."""
    assert poly.parse(text) == {2: 1, 1: -1, 0: -6}


@pytest.mark.parametrize("text", ["3X^2 - 4x + 1", "x\u00b2 - 4x + 1", "x^2 \u2212 4x + 1",
                                  "x*3 + 1", "3.0x + 1", "x^2 + 3x + C", "x^2 - x - 6 = 1",
                                  "((x^2 - x - 6))", "(x + 2)(x - 3)", "x^2 - - 6", "x^2 -+ 6",
                                  "(x^2 - x) - 6"])
def test_parse_still_refuses(text):
    """T1b review 5: the forms that stay wrong."""
    with pytest.raises(poly.PolyError):
        poly.parse(text)


def test_parse_refuses_a_non_string():
    with pytest.raises(poly.PolyError):
        poly.parse(3)


def test_to_text_is_canonical():
    assert poly.to_text({2: 3, 1: -1, 0: 5}) == "3x^2 - x + 5"
    assert poly.to_text({3: -1, 0: -2}) == "-x^3 - 2"
    assert poly.to_text({}) == "0"
    assert poly.to_text({3: Fraction(4, 3), 1: -Fraction(1, 2)}) == "(4/3)x^3 - (1/2)x"
    # spelling never matters: the same polynomial, the same text
    assert poly.normalize("5x + x^2 + 6") == poly.normalize("x**2+5*x+6") == "x^2 + 5x + 6"


def test_differentiate_and_integrate():
    p = poly.parse("4x^2 + 4x - 18")
    assert poly.to_text(poly.differentiate(p)) == "8x + 4"
    assert poly.to_text(poly.integrate(p)) == "(4/3)x^3 + 2x^2 - 18x"
    assert poly.differentiate(poly.parse("7")) == {}
    # integrate then differentiate gives back the polynomial (constant 0)
    assert poly.differentiate(poly.integrate(p)) == p


@pytest.mark.parametrize("text, lead, roots, written", [
    ("x^2 - x - 6", 1, [-2, 3], "(x + 2)(x - 3)"),
    ("x^2 + 2x - 24", 1, [-6, 4], "(x + 6)(x - 4)"),
    ("x^2 - 16", 1, [-4, 4], "(x + 4)(x - 4)"),
    ("x^2 - 2x + 1", 1, [1, 1], "(x - 1)^2"),
    ("x^2 + 3x", 1, [-3, 0], "(x + 3)(x)"),
    ("2x^2 - 2", 2, [-1, 1], "2(x + 1)(x - 1)"),
    ("-x^2 + 1", -1, [-1, 1], "-(x + 1)(x - 1)"),
    ("x^3 - 6x^2 + 11x - 6", 1, [1, 2, 3], "(x - 1)(x - 2)(x - 3)"),
    ("x - 5", 1, [5], "(x - 5)"),
])
def test_factor(text, lead, roots, written):
    assert poly.factor(poly.parse(text)) == (lead, roots)
    assert poly.factor_text(lead, roots) == written


@pytest.mark.parametrize("text", ["x^2 + 1", "x^2 - 2", "2x^2 + 3x + 1", "7", "0"])
def test_factor_refuses_what_has_no_integer_root_factorisation(text):
    with pytest.raises(poly.PolyError):
        poly.factor(poly.parse(text))
