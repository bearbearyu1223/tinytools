"""sft-followup's pure helpers: the wording classes and the asked-parameter rule."""

import pytest

from tinytools import followup as F


@pytest.mark.parametrize("text, kind", [
    ("The given question lacks the parameters required by the function.", F.TEMPLATE),
    ("The query does not provide the required argument \"proxy\".", F.TEMPLATE),
    ("  The question lacks the required \"x\" parameter.", F.TEMPLATE),  # template wins
    ('The function "f" can do it, but it lacks the required "a" parameter.', F.LACKS),
    ("To process the return, could you please provide the reason?", F.ASKS),
    ("None of the provided functions can be used to answer this.", F.NO_FIT),
    ("The provided functions do not support retrieving prices.", F.NO_FIT),
    ('"[No Need to Ask]"', F.OTHER),
])
def test_opening_kind(text, kind):
    assert F.opening_kind(text) == kind


@pytest.mark.parametrize("reply, asked", [
    ('it lacks the required "a" parameter. Please provide it.', ["a"]),
    ('but it lacks the required "number1" and "number2" parameters.', ["number1", "number2"]),
    ('lacks the required "x", "y", and "z" parameters', ["x", "y", "z"]),
    ("The query lacks the required arguments for the function `f`.", None),
    ("<tool_call>...</tool_call>", None),
])
def test_asked_parameters(reply, asked):
    assert F.asked_parameters(reply) == asked


@pytest.mark.parametrize("options, question, found", [
    ([2], "where a=2, b=5 and c=3 ?", True),
    ([2], "where a=12", False),            # not inside a longer number
    ([2], "a=2.5", False),
    ([10.0], "a radius of 10?", True),      # a whole float as an integer
    (["La Liga"], "the 2021 la liga season", True),
    (["meters", ""], "a radius of 10", False),
    ([True], "include dissent: True", False),  # booleans are never checkable
    ([["a", "b"]], "a and b", False),
])
def test_value_in_question(options, question, found):
    assert F.value_in_question(options, question) is found


# Regression (post 9 review, round 2): string values matched inside words, so
# "in the question" over-counted (233 SFT, 231 control; 229 and 228 word-bounded).
def test_en_is_not_in_a_question_that_only_contains_sentiment_and_fresh():
    # BFCL simple_python_216: language's possible answer is ["en"]; the question never
    # names a language. Counted "in the question" for SFT seeds 1-3 and control seeds 1-3.
    q = ("Analyze the sentiment of a customer review 'I love the food here! "
         "It's always fresh and delicious.'.")
    assert not F.value_in_question(["en"], q)


def test_goog_is_not_written_in_google():
    # BFCL parallel_17: symbol's possible answer is ["GOOG", "GOOGL"]; the question says
    # "Google LLC.". Counted "in the question" for SFT seed 2.
    q = ("Get the latest closing prices and volumes for Apple Inc., Google LLC., and "
         "Microsoft Corporation in the New York Stock Exchange")
    assert not F.value_in_question(["GOOG", "GOOGL"], q)


def test_a_required_parameter_whose_value_is_absent_is_not_in_the_question():
    functions = [{"name": "sentiment_analysis", "parameters": {
        "properties": {"text": {}, "language": {}}, "required": ["text", "language"]}}]
    ground_truth = [{"sentiment_analysis": {"text": ["I love the food here!"],
                                            "language": ["en"]}}]
    q = "Analyze the sentiment of a customer review 'I love the food here!'."
    assert F.parameter_kind("language", functions, ground_truth, q) != F.IN_QUESTION


@pytest.mark.parametrize("options, question", [
    (["house_price", "house price"],                            # BFCL multiple_53
     "Predict the house prices for next 5 years based on interest rates and "
     "unemployment rates."),
    (["living room", "Living room"],                            # BFCL multiple_75
     "Which paint color is currently most popular for living rooms?"),
    (["box"], "ship two boxes"),
])
def test_a_plural_ending_still_counts(options, question):
    assert F.value_in_question(options, question)


FUNCS = [{"name": "area", "parameters": {"properties": {"radius": {}, "units": {}, "mode": {}},
                                          "required": ["radius", "mode"]}}]


@pytest.mark.parametrize("param, truth, kind", [
    ("radius", [{"area": {"radius": [10], "units": ["m", ""], "mode": ["x"]}}], F.IN_QUESTION),
    ("units", [{"area": {"radius": [10], "units": ["m", ""], "mode": ["x"]}}], F.OPTIONAL),
    ("mode", [{"area": {"radius": [10], "mode": ["fast", ""]}}], F.OMITTABLE),
    ("mode", [{"area": {"radius": [10], "mode": ["slow"]}}], F.NOT_FOUND),
    ("colour", [{"area": {"radius": [10]}}], F.NOT_IN_SCHEMA),
])
def test_parameter_kind(param, truth, kind):
    assert F.parameter_kind(param, FUNCS, truth, "Area of a circle of radius 10?") == kind


def test_reply_kind_takes_the_first_in_order():
    assert F.reply_kind([F.NOT_FOUND, F.OPTIONAL]) == F.OPTIONAL
    assert F.reply_kind([F.OPTIONAL, F.IN_QUESTION]) == F.IN_QUESTION
    with pytest.raises(ValueError):
        F.reply_kind([])


def test_text_turns_and_calls():
    msgs = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "which?"},
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": "", "tool_calls": [{"function": {}}]}]
    assert F.has_call(msgs) and F.text_turns(msgs) == 1
    assert F.first_assistant_text(msgs) == "which?"
