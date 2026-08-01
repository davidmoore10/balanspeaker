"""Parse natural-language duration expressions."""

import re

_SMALL_NUMBERS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}

_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}

_NUMBER_WORDS = frozenset(
    {
        *_SMALL_NUMBERS,
        *_TENS,
        "hundred",
        "thousand",
        "and",
        "a",
        "an",
    }
)

_UNIT_MULTIPLIERS = {
    "second": 1,
    "seconds": 1,
    "sec": 1,
    "secs": 1,
    "minute": 60,
    "minutes": 60,
    "min": 60,
    "mins": 60,
    "hour": 3600,
    "hours": 3600,
    "hr": 3600,
    "hrs": 3600,
}

_TIMER_KEYWORDS = {
    "timer",
    "countdown",
}

_TOKEN_PATTERN = re.compile(r"\d+(?:\.\d+)?|[a-z]+(?:-[a-z]+)?")


def parse_duration_seconds(text: str) -> int | None:
    """Return the first duration found in text.

    When a timer command contains a number without a unit,
    seconds are assumed.

    Supported examples include:

    - timer 5
    - five minutes
    - twenty-one seconds
    - 1.5 hours
    - half an hour
    - quarter of an hour
    - one and a half hours
    """

    normalized_text = _normalize(text)

    fractional_duration = _parse_fractional_duration(normalized_text)

    if fractional_duration is not None:
        return fractional_duration

    tokens = _TOKEN_PATTERN.findall(normalized_text)

    explicit_duration = _parse_explicit_unit_duration(tokens)

    if explicit_duration is not None:
        return explicit_duration

    unitless_value = _parse_unitless_timer_value(tokens)

    if unitless_value is None:
        return None

    return round(unitless_value)


def parse_number_phrase(text: str) -> float | None:
    """Convert a number phrase into a numeric value."""

    normalized_text = _normalize(text)

    try:
        return float(normalized_text)
    except ValueError:
        pass

    tokens = _TOKEN_PATTERN.findall(normalized_text)

    if not tokens:
        return None

    if tokens in (["a"], ["an"]):
        return 1.0

    if not all(token in _NUMBER_WORDS for token in tokens):
        return None

    current = 0
    total = 0
    found_number = False

    for token in tokens:
        if token == "and":
            continue

        if token in {"a", "an"}:
            current += 1
            found_number = True
            continue

        if token in _SMALL_NUMBERS:
            current += _SMALL_NUMBERS[token]
            found_number = True
            continue

        if token in _TENS:
            current += _TENS[token]
            found_number = True
            continue

        if token == "hundred":
            current = max(current, 1) * 100
            found_number = True
            continue

        if token == "thousand":
            total += max(current, 1) * 1000
            current = 0
            found_number = True

    if not found_number:
        return None

    return float(total + current)


def _parse_explicit_unit_duration(
    tokens: list[str],
) -> int | None:
    """Parse a number followed by a duration unit."""

    for unit_index, unit_token in enumerate(tokens):
        multiplier = _UNIT_MULTIPLIERS.get(unit_token)

        if multiplier is None:
            continue

        value = _parse_value_before_unit(
            tokens=tokens,
            unit_index=unit_index,
        )

        if value is None:
            continue

        return round(value * multiplier)

    return None


def _parse_fractional_duration(
    text: str,
) -> int | None:
    """Parse common fractional hour expressions."""

    if re.search(
        r"\b(?:a\s+)?quarter\s+of\s+an\s+hour\b",
        text,
    ):
        return 15 * 60

    if re.search(
        r"\b(?:a\s+)?quarter\s+hour\b",
        text,
    ):
        return 15 * 60

    if re.search(
        r"\bhalf\s+(?:of\s+)?an\s+hour\b",
        text,
    ):
        return 30 * 60

    if re.search(
        r"\bhalf\s+(?:an?\s+)?hour\b",
        text,
    ):
        return 30 * 60

    tokens = _TOKEN_PATTERN.findall(text)

    for index in range(len(tokens) - 3):
        if tokens[index : index + 3] != [
            "and",
            "a",
            "half",
        ]:
            continue

        unit_index = index + 3

        if unit_index >= len(tokens):
            continue

        unit = tokens[unit_index]

        if unit not in {
            "hour",
            "hours",
            "hr",
            "hrs",
        }:
            continue

        whole_number = _parse_number_suffix(tokens[:index])

        if whole_number is None:
            continue

        return round((whole_number + 0.5) * 3600)

    return None


def _parse_value_before_unit(
    *,
    tokens: list[str],
    unit_index: int,
) -> float | None:
    """Parse the number immediately preceding a unit."""

    if unit_index <= 0:
        return None

    previous_token = tokens[unit_index - 1]

    try:
        return float(previous_token)
    except ValueError:
        pass

    candidate_tokens: list[str] = []

    for index in range(unit_index - 1, -1, -1):
        token = tokens[index]

        if token not in _NUMBER_WORDS:
            break

        candidate_tokens.append(token)

    candidate_tokens.reverse()

    return _parse_number_suffix(candidate_tokens)


def _parse_number_suffix(
    tokens: list[str],
) -> float | None:
    """Parse the longest usable numeric suffix of a token list."""

    if not tokens:
        return None

    for start_index in range(len(tokens)):
        candidate_tokens = tokens[start_index:]

        if not candidate_tokens:
            continue

        candidate = " ".join(candidate_tokens)
        parsed_number = parse_number_phrase(candidate)

        if parsed_number is not None:
            return parsed_number

    return None


def _parse_unitless_timer_value(
    tokens: list[str],
) -> float | None:
    """Parse a timer number that omits its unit.

    Examples:

    - timer 5
    - timer five
    - countdown twenty
    """

    keyword_index: int | None = None

    for index, token in enumerate(tokens):
        if token in _TIMER_KEYWORDS:
            keyword_index = index
            break

    if keyword_index is None:
        return None

    remaining_tokens = tokens[keyword_index + 1 :]

    while remaining_tokens and remaining_tokens[0] in {"for", "of"}:
        remaining_tokens = remaining_tokens[1:]

    if not remaining_tokens:
        return None

    first_token = remaining_tokens[0]

    try:
        return float(first_token)
    except ValueError:
        pass

    number_tokens: list[str] = []

    for token in remaining_tokens:
        if token not in _NUMBER_WORDS:
            break

        number_tokens.append(token)

    if not number_tokens:
        return None

    return parse_number_phrase(" ".join(number_tokens))


def _normalize(text: str) -> str:
    """Normalize text for duration parsing."""

    normalized = text.strip().lower()
    normalized = normalized.replace("-", " ")

    return re.sub(r"\s+", " ", normalized)
