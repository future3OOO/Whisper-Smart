"""Golden tests for deterministic transcript post-processing."""

from dictation_tool.postprocess import DictationPostProcessor


def test_postprocess_formats_email_address_and_spacing():
    post = DictationPostProcessor()

    assert post.process("please email john at sign gmail dot com") == (
        "Please email john@gmail.com"
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
