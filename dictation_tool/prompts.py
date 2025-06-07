# dictation_tool/prompts.py

PRESETS = {
    "business": (
        "This is professional business dictation with terms like {terms}. Please use proper formatting.",
        "Q4,spreadsheet,project coordination,stakeholders"
    ),
    "programming": (
        "This is technical programming dictation with terms like {terms}. Please use proper formatting.",
        "DataFrame,pandas,numpy,microservices,API,SSL"
    ),
    "file-ops": (
        "This is computer file management with terms like {terms}. Please use proper formatting.",
        "Project_Files,report.pdf,Adobe Reader,C-drive"
    ),
    "tech": (
        "This is a technical discussion about software architecture with terms like {terms}.",
        "microservices,SSL encryption,staging environment,deployment"
    ),
    "general": (
        # This is the new, optimized prompt for web, search, and email.
        # Note: The {terms} placeholder is still here for consistency, but we provide the terms directly.
        "This is a high-quality dictation for web browsing, search queries, and professional emails. It includes terms like {terms}. Please use correct capitalization and punctuation.",
        "www,.com,@,Google,search for,send an email to,subject line,John Smith"
    ),
} 