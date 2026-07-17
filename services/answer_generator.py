from typing import Any


def generate_educational_answer(
    question: str,
    matches: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Create a concise educational response from retrieved knowledge chunks.

    This version uses deterministic rules and does not call a paid LLM.
    """

    if not matches:
        return {
            "summary": (
                "VisaOPS could not find enough information in the current "
                "knowledge base to answer this question."
            ),
            "considerations": [],
            "sources": [],
        }

    combined_text = " ".join(
        match["content"].lower()
        for match in matches
    )

    summary = (
        "VisaOPS found educational guidance related to your question. "
        "The exact outcome depends on the person's documents, dates, "
        "immigration history, and individual circumstances."
    )

    considerations: list[str] = []

    if "grace period" in combined_text or "60 consecutive days" in combined_text:
        summary = (
            "Certain eligible nonimmigrant workers may receive a discretionary "
            "grace period of up to 60 consecutive days after employment ends. "
            "The available period may be shorter if the person's authorized "
            "stay expires earlier."
        )

        considerations.extend(
            [
                "Review the last day of employment.",
                "Check the expiration date on the most recent Form I-94.",
                "Review the validity dates on approval notices.",
                "Consider whether a timely petition, change of status, or departure is needed.",
            ]
        )

    elif "change-of-employer" in combined_text or "portability" in combined_text:
        summary = (
            "A change of employer may be possible through a new H-1B petition, "
            "but eligibility and the permitted employment start date depend "
            "on the person's current status and filing circumstances."
        )

        considerations.extend(
            [
                "Collect prior H-1B approval notices.",
                "Review the most recent Form I-94.",
                "Confirm when the new petition will be filed.",
                "Ask whether the case qualifies for H-1B portability.",
            ]
        )

    elif "change-of-status" in combined_text:
        summary = (
            "A change-of-status application asks USCIS to move a person from "
            "one nonimmigrant classification to another. A pending application "
            "does not automatically provide employment authorization."
        )

        considerations.extend(
            [
                "Confirm the filing and receipt dates.",
                "Review the current Form I-94 expiration date.",
                "Confirm whether the prior status must be maintained.",
                "Do not assume that a pending filing authorizes employment.",
            ]
        )

    else:
        considerations.extend(
            [
                "Review the most recent Form I-94.",
                "Confirm all filing, receipt, and decision dates.",
                "Preserve approval notices and other supporting documents.",
                "Discuss case-specific questions with a qualified attorney.",
            ]
        )

    sources = []

    for match in matches:
        source = {
            "title": match["file_name"].replace("_", " ").replace(".md", "").title(),
            "path": match["source"],
            "category": match["visa_category"],
        }

        if source not in sources:
            sources.append(source)

    return {
        "summary": summary,
        "considerations": considerations,
        "sources": sources,
    }
