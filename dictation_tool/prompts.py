# dictation_tool/prompts.py
# ---------------------------------------------------------------------------
# High-level “initial_prompt” snippets injected into Whisper to bias spelling,
# e-mail formatting, technical terms, etc.  The {terms} placeholder lets the
# CLI / GUI pass extra vocabulary at runtime – but every preset already lists
# sane defaults so you can call them verbatim.
# ---------------------------------------------------------------------------

PRESETS: dict[str, tuple[str, str]] = {
    # ──────────────────────────────────────────────────────────────────────
    "business": (
        "You are transcribing professional business dictation.  "
        "Spelling and punctuation must be perfect.  Vocabulary includes: {terms}.",
        "Q4,spreadsheet,project coordination,stakeholders",
    ),
    # ──────────────────────────────────────────────────────────────────────
    "programming": (
        "You are transcribing technical programming dictation.  "
        "Code identifiers and library names must be preserved verbatim.  "
        "Vocabulary includes: {terms}.",
        "DataFrame,pandas,numpy,microservices,API,SSL",
    ),
    # ──────────────────────────────────────────────────────────────────────
    "file-ops": (
        "You are transcribing speech about computer file management.  "
        "Paths and filenames must be exact.  Vocabulary includes: {terms}.",
        "Project_Files,report.pdf,Adobe Reader,C-drive",
    ),
    # ──────────────────────────────────────────────────────────────────────
    "tech": (
        "You are transcribing a technical discussion on software architecture.  "
        "Maintain precise terminology.  Vocabulary includes: {terms}.",
        "microservices,SSL encryption,staging environment,deployment",
    ),
    # ──────────────────────────────────────────────────────────────────────
    "general": (
        # ⭐ Generic web-search + short e-mail dictation preset
        "You are transcribing high-quality everyday dictation for web searches and "
        "short professional e-mails.  Convert phrases such as “at sign” or “at symbol” "
        "to “@”, “dot” to “.”, and remove spaces inside URLs and e-mail addresses.  "
        "Capitalise the first word of each sentence.  Insert a single line break when "
        "the speaker says “new line” and a blank line (two line breaks) when the "
        "speaker says “new paragraph”.  Vocabulary includes: {terms}.",
        "www,.com,@,Google,search for,send an email to,subject line,John Smith",
    ),
    # ──────────────────────────────────────────────────────────────────────
    "email": (
        # ⭐ Fully-optimised long-form e-mail preset
        "Transcribe a formal e-mail **and apply the following commands instead of "
        "writing them literally**:\n"
        "• “at sign”, “at symbol” → @\n"
        "• “dot”, “period”         → .\n"
        "• “new line”, “line break”      → ⏎ (single newline)\n"
        "• “new paragraph”                → ⏎⏎ (blank line)\n"
        "• “bullet point”                 → ⏎•  (bullet + space)\n"
        "Strip every space that occurs immediately before or after “@” or “.” "
        "inside addresses and URLs. Never output any quotation marks the speaker "
        "uses to indicate a command. Capitalise the first word of each sentence "
        "and use correct punctuation. Vocabulary: {terms}.",
        "Dear,Hi team,Kind regards,Best regards,McKenzie,@,gmail.com,cc,bcc",
    ),

}
