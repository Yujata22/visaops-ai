from typing import Any

from google import genai


def _build_context(matches: list[dict[str, Any]]) -> str:
    sections = []

    for index, match in enumerate(matches, start=1):
        text = (
            match.get("document")
            or match.get("text")
            or match.get("content")
            or ""
        )

        metadata = match.get("metadata", {})
        source = (
            metadata.get("title")
            or metadata.get("source")
            or metadata.get("filename")
            or f"Source {index}"
        )

        sections.append(
            f"[Source {index}: {source}]\n{text}"
        )

    return "\n\n".join(sections)


def generate_gemini_answer(
    api_key: str,
    question: str,
    matches: list[dict[str, Any]],
    model_name: str = "gemini-3.5-flash",
) -> dict[str, Any]:
    """
    Generate a grounded educational response using Gemini.
    """

    if not api_key:
        return {
            "model_name": model_name,
            "answer": "",
            "error": "Gemini API key is missing.",
        }

    context = _build_context(matches)

    prompt = f"""
You are VisaOPS AI, an educational immigration information assistant.

Rules:
1. Use only the supplied knowledge-base context.
2. Do not invent laws, deadlines, eligibility rules, or USCIS procedures.
3. Clearly distinguish general information from legal advice.
4. Tell the user what facts should be verified.
5. Do not claim that an application will be approved.
6. Cite sources using labels such as [Source 1].
7. Keep the answer clear and practical.

User question:
{question}

Knowledge-base context:
{context}

Provide the response under these headings:

Educational Summary
Important Considerations
What to Verify
Sources
"""

    try:
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )

        answer = response.text or ""

        return {
            "model_name": model_name,
            "answer": answer.strip(),
            "error": None,
        }

    except Exception as exc:
        return {
            "model_name": model_name,
            "answer": "",
            "error": str(exc),
        }
