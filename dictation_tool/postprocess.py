"""Deterministic transcript post-processing."""

from __future__ import annotations

import re
from typing import ClassVar


class DictationPostProcessor:
    """Format raw model text into paste-ready dictation output."""

    _PUNCTUATION: ClassVar[dict[tuple[str, ...], str]] = {
        ("at", "sign"): "@",
        ("dot",): ".",
        ("comma",): ",",
        ("colon",): ":",
        ("semicolon",): ";",
        ("question", "mark"): "?",
        ("exclamation", "point"): "!",
        ("exclamation", "mark"): "!",
        ("dash",): "-",
        ("hyphen",): "-",
        ("slash",): "/",
        ("open", "parenthesis"): "(",
        ("close", "parenthesis"): ")",
        ("open", "bracket"): "[",
        ("close", "bracket"): "]",
        ("open", "brace"): "{",
        ("close", "brace"): "}",
        ("dollar", "sign"): "$",
        ("percent", "sign"): "%",
        ("hash", "tag"): "#",
        ("pound", "sign"): "#",
    }
    _EMAIL_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"\b([\w.-]+)\s+at(?:\s+sign)?\s+([\w.-]+)\s+dot\s+com\b", re.I
    )
    _URL_DOT: ClassVar[re.Pattern[str]] = re.compile(
        r"\b([a-zA-Z0-9_-]+)\s+\.\s+([a-zA-Z0-9_-]+)"
    )
    _CMD_SUBS: ClassVar[tuple[tuple[re.Pattern[str], str], ...]] = (
        (
            re.compile(
                r"\b(?:new\s+line|line\s*break|newline)\b[ \t]*[.,!?;:]?[ \t]*",
                re.I,
            ),
            "\n",
        ),
        (re.compile(r"\bnew\s+paragraph\b[ \t]*[.,!?;:]?[ \t]*", re.I), "\n\n"),
        (re.compile(r"\bbullet\s+point\b[ \t]*", re.I), "\n• "),
        (re.compile(r'[\u201C\u201D"`\uFFFD]+'), ""),
    )
    _SPACES_AROUND_DOT_AT: ClassVar[re.Pattern[str]] = re.compile(
        r"[\t \u00A0\u1680\u2000-\u200A\u202F\u205F\u3000]*"
        r"([@.])"
        r"[\t \u00A0\u1680\u2000-\u200A\u202F\u205F\u3000]*"
    )
    _SIGNOFFS: ClassVar[tuple[str, ...]] = (
        "kind regards",
        "best regards",
        "regards",
        "cheers",
    )
    _SIGNOFF_PAT: ClassVar[re.Pattern[str]] = re.compile(
        rf"(^|\n)({'|'.join(_SIGNOFFS)})\b[,.\s]*",
        re.I,
    )
    _GREETING_BREAK: ClassVar[re.Pattern[str]] = re.compile(
        r"(?i)(^|\n)(\s*(?:hi|hello|hey|kia(?:\s+ora)?|dear)\b[^\n]*?)\s*\n\n"
    )

    def __init__(self) -> None:
        self._punctuation_patterns = [
            (re.compile(r"\b" + r"\s+".join(map(re.escape, key)) + r"\b", re.I), value)
            for key, value in self._PUNCTUATION.items()
        ]
        self._punctuation_patterns.sort(key=lambda item: -item[0].pattern.count(r"\s"))
        self._spc_before = re.compile(r"[ \t]+([,.:;?!()[\]{}])")
        self._spc_after = re.compile(r"([(\[{])[ \t]+")

    def clean_model_text(self, text: str) -> str:
        """Apply command, email, URL, sign-off, and line formatting rules."""
        text = self._EMAIL_RE.sub(r"\1@\2.com", self._URL_DOT.sub(r"\1.\2", text))
        for pattern, replacement in self._CMD_SUBS:
            text = pattern.sub(replacement, text)

        text = re.sub(r"\n{2,}(?=•)", "\n", text)
        text = re.sub(r",\s*,+", ",", text)
        text = re.sub(r",\s*\n\n", ".\n\n", text)
        text = self._GREETING_BREAK.sub(
            lambda match: f"{match.group(1)}{match.group(2).rstrip(' ,.!?;:')},\n\n",
            text,
        )
        text = re.sub(r"([^\s.,!?;:])\s*\n\n", r"\1.\n\n", text)
        text = self._SPACES_AROUND_DOT_AT.sub(r"\1", text)
        text = re.sub(r"@[ \t]+", "@", text)
        text = re.sub(r"\.[ \t]+", ".", text)
        text = self._SIGNOFF_PAT.sub(
            lambda match: f"{match.group(1)}{match.group(2).title()},\n",
            text,
        )
        text = re.sub(r"[ \t]+\n", "\n", text)
        return re.sub(
            r"(^|\n)([• \t]*)([a-z])",
            lambda match: match.group(1) + match.group(2) + match.group(3).upper(),
            text,
        )

    def format_clipboard_text(self, text: str) -> str:
        """Apply spoken punctuation replacements and final trim."""
        for pattern, replacement in self._punctuation_patterns:
            text = pattern.sub(replacement, text)
        text = self._spc_before.sub(r"\1", text)
        text = self._spc_after.sub(r"\1", text)
        return text.strip()

    def process(self, text: str) -> str:
        """Run the full model-text to paste-text pipeline."""
        return self.format_clipboard_text(self.clean_model_text(text))
