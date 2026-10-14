"""Golden tests for deterministic transcript post-processing."""

import pytest

from dictation_tool.postprocess import DictationPostProcessor


@pytest.mark.parametrize(
    "spoken, expected",
    [
        ("check the site dot example dot org", "Check the site. example. org"),
        ("version two dot zero dot one", "Version two. zero. one"),
        ("visit example . org", "Visit example.org"),
    ],
)
def test_postprocess_does_not_invent_email_addresses(spoken, expected):
    assert DictationPostProcessor().process(spoken) == expected


def test_postprocess_preserves_spaces_between_sentences():
    post = DictationPostProcessor()

    assert post.process("Hello. Next sentence.") == "Hello. Next sentence."


def test_postprocess_normalises_email_spacing_without_joining_sentences():
    post = DictationPostProcessor()

    assert (
        post.process(
            "email john\u00a0.\u00a0smith\u00a0@\u00a0gmail\u00a0.\u00a0com. Next sentence."
        )
        == "Email john.smith@gmail.com. Next sentence."
    )


def test_postprocess_formats_email_address_and_spacing():
    post = DictationPostProcessor()

    assert post.process("please email john at sign gmail dot com") == (
        "Please email john@gmail.com"
    )


def test_postprocess_formats_explicit_email_at_custom_domain():
    assert (
        DictationPostProcessor().process("please email john at sign example dot org")
        == "Please email john@example.org"
    )


def test_postprocess_formats_email_with_spoken_at_and_literal_domain():
    post = DictationPostProcessor()

    assert post.process("mckenzielawrence at gmail.com") == (
        "Mckenzielawrence@gmail.com"
    )


def test_postprocess_removes_repeated_cc_bcc_tail_after_email_address():
    post = DictationPostProcessor()

    assert (
        post.process("mckinzylawrence dot gmail dot com, cc, bcc, bcc, bcc, bcc,")
        == "Mckinzylawrence@gmail.com"
    )


def test_postprocess_preserves_email_paragraph_shape():
    post = DictationPostProcessor()

    assert post.process("hi mckenzie new paragraph kind regards mckenzie") == (
        "Hi mckenzie,\n\nKind Regards,\nMckenzie"
    )


def test_postprocess_formats_bullets_and_line_capitalisation():
    post = DictationPostProcessor()

    assert post.process(
        "bullet point first item new line bullet point second item"
    ) == ("• First item\n• Second item")
