from datetime import date

import streamlit as st

from services.analysis_engine import analyze_scenario
from services.answer_generator import generate_educational_answer
from services.knowledge_retriever import (
    KnowledgeBaseError,
    VisaKnowledgeRetriever,
)
from services.timeline_engine import build_timeline


st.set_page_config(
    page_title="VisaOPS AI",
    page_icon="🛂",
    layout="wide",
)


@st.cache_resource
def load_knowledge_retriever() -> VisaKnowledgeRetriever:
    """
    Load the embedding model and Chroma collection once.
    """

    return VisaKnowledgeRetriever()


st.title("🛂 VisaOPS AI")

st.caption(
    "Explore fictional U.S. visa scenarios, timelines, risks, "
    "and educational knowledge resources."
)

st.warning(
    "VisaOPS AI provides educational information only and does not provide "
    "legal advice. It cannot determine lawful status, eligibility, "
    "employment authorization, or the outcome of a case."
)


metric_col1, metric_col2, metric_col3 = st.columns(3)

metric_col1.metric(
    "Knowledge Articles",
    "3",
)

metric_col2.metric(
    "Visa Categories",
    "3",
)

metric_col3.metric(
    "Scenario Types",
    "7",
)

st.divider()


with st.sidebar:
    st.header("VisaOPS AI")

    st.info("MVP Version: 0.5")

    with st.expander("About this product"):
        st.write(
            """
            VisaOPS is an experimental immigration scenario explorer.

            It helps users:

            - Search educational visa guidance
            - Organize important dates
            - Visualize visa timelines
            - Identify missing information
            - Prepare questions for an attorney
            """
        )

    with st.expander("Privacy guidance"):
        st.caption(
            "Do not enter passport numbers, receipt numbers, A-numbers, "
            "addresses, dates of birth, or other sensitive personal information."
        )


st.subheader("Ask VisaOPS")

st.write(
    "Search the educational visa knowledge base using a natural-language question."
)

question = st.text_input(
    "Enter a visa question",
    placeholder=(
        "Example: My H-1B employment ended. "
        "What documents and deadlines should I review?"
    ),
)

search_col1, search_col2 = st.columns([3, 1])

with search_col1:
    category_filter = st.selectbox(
        "Knowledge category",
        [
            "All categories",
            "H-1B",
            "F-1",
            "OPT",
            "General",
        ],
    )

with search_col2:
    result_count = st.selectbox(
        "Number of sources",
        [1, 2, 3],
        index=2,
    )

search_knowledge = st.button(
    "Search Knowledge Base",
    use_container_width=True,
)


if search_knowledge:
    if not question.strip():
        st.warning("Enter a question before searching.")
    else:
        try:
            retriever = load_knowledge_retriever()

            category_mapping = {
                "All categories": None,
                "H-1B": "h1b",
                "F-1": "f1",
                "OPT": "opt",
                "General": "general",
            }

            selected_category = category_mapping[category_filter]

            matches = retriever.search(
                query=question,
                number_of_results=result_count,
                visa_category=selected_category,
            )

            if not matches:
                st.info(
                    "No matching knowledge documents were found."
                )
            else:
                generated_answer = generate_educational_answer(
                    question=question,
                    matches=matches,
                )

                st.markdown("### Educational Summary")

                st.info(
                    generated_answer["summary"]
                )

                if generated_answer["considerations"]:
                    st.markdown("#### Important Considerations")

                    for consideration in generated_answer["considerations"]:
                        st.write(f"- {consideration}")

                st.markdown("### Supporting Knowledge Sources")

                for result_number, match in enumerate(
                    matches,
                    start=1,
                ):
                    display_title = (
                        match["file_name"]
                        .replace("_", " ")
                        .replace(".md", "")
                        .title()
                    )

                    with st.expander(
                        f"Source {result_number}: {display_title}",
                        expanded=result_number == 1,
                    ):
                        st.write(match["content"])

                        st.caption(
                            f"Knowledge file: {match['source']}"
                        )

                        st.caption(
                            f"Category: {match['visa_category'].upper()}"
                        )

                        st.caption(
                            "Relevant knowledge-base match"
                        )

                st.info(
                    "This response is based on retrieved educational passages. "
                    "It is not a legal conclusion or personalized legal advice."
                )

        except KnowledgeBaseError as error:
            st.error(str(error))

            st.code(
                "python scripts/ingest_documents.py",
                language="bash",
            )

        except Exception as error:
            st.error(
                "VisaOPS could not search the knowledge base."
            )

            with st.expander("Technical error"):
                st.exception(error)


st.divider()


st.subheader("Visa Timeline Simulator")

st.write(
    "Enter a fictional situation to organize important dates, "
    "identify potential concerns, and prepare questions for an attorney."
)

col1, col2 = st.columns(2)


with col1:
    visa_type = st.selectbox(
        "Current visa or status",
        [
            "H-1B",
            "F-1",
            "F-1 OPT",
            "F-1 STEM OPT",
            "H-4",
            "B-1/B-2",
            "Other",
        ],
        key="scenario_visa_type",
    )

    situation = st.selectbox(
        "What happened?",
        [
            "Employment ended",
            "Considering an H-1B transfer",
            "Change of status filed",
            "Application rejected",
            "Application denied",
            "Status expiration approaching",
            "Exploring visa options",
        ],
    )


with col2:
    event_date = st.date_input(
        "Date of the event",
        value=date.today(),
    )

    has_filing_date = st.checkbox(
        "I have an application or petition filing date"
    )

    filing_date = None

    if has_filing_date:
        filing_date = st.date_input(
            "Filing date",
            value=date.today(),
        )


details = st.text_area(
    "Describe the fictional scenario",
    placeholder=(
        "Example: The person's H-1B employment ended, and they are "
        "considering filing a change-of-status application."
    ),
    height=130,
)

privacy_confirmed = st.checkbox(
    "I confirm that I have not entered sensitive identifying information."
)


analyze = st.button(
    "Analyze Scenario",
    type="primary",
    use_container_width=True,
    disabled=not privacy_confirmed,
)


if analyze:
    timeline = build_timeline(
        visa_type=visa_type,
        situation=situation,
        event_date=event_date,
        filing_date=filing_date,
    )

    analysis = analyze_scenario(
        visa_type=visa_type,
        situation=situation,
        event_date=event_date,
        filing_date=filing_date,
        details=details,
    )

    st.divider()
    st.subheader("Scenario Summary")

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    summary_col1.metric(
        "Visa Type",
        visa_type,
    )

    summary_col2.metric(
        "Situation",
        situation,
    )

    summary_col3.metric(
        "Review Level",
        analysis["risk_level"],
    )

    st.write(
        f"**Event date:** {event_date.strftime('%b %d, %Y')}"
    )

    if filing_date:
        st.write(
            f"**Reported filing date:** "
            f"{filing_date.strftime('%b %d, %Y')}"
        )

    if filing_date and filing_date < event_date:
        st.warning(
            "The reported filing date occurs before the selected event date. "
            "This may be correct, but review the dates carefully."
        )

    st.markdown("### Visa Timeline")

    for index, timeline_event in enumerate(timeline):
        timeline_date = timeline_event["date"].strftime(
            "%b %d, %Y"
        )

        title = timeline_event["title"]
        description = timeline_event["description"]
        status = timeline_event["status"]

        timeline_message = (
            f"**{timeline_date} — {title}**\n\n"
            f"{description}"
        )

        if status == "important":
            st.warning(timeline_message)
        elif status == "completed":
            st.success(timeline_message)
        else:
            st.info(timeline_message)

        if index < len(timeline) - 1:
            st.markdown(
                "<div style='text-align: center; font-size: 28px;'>↓</div>",
                unsafe_allow_html=True,
            )

    st.markdown("### Timeline Summary")

    if len(timeline) == 1:
        st.write(
            "One event was identified for this scenario."
        )
    else:
        total_days = (
            timeline[-1]["date"] - timeline[0]["date"]
        ).days

        st.write(
            f"This scenario contains **{len(timeline)} timeline events** "
            f"covering approximately **{total_days} days**."
        )

    st.markdown("### Educational Observations")

    for observation in analysis["observations"]:
        st.write(f"- {observation}")

    st.markdown("### Possible Next Steps")

    for step_number, next_step in enumerate(
        analysis["next_steps"],
        start=1,
    ):
        st.write(f"{step_number}. {next_step}")

    st.markdown("### Questions for an Immigration Attorney")

    for attorney_question in analysis["attorney_questions"]:
        st.write(f"- {attorney_question}")

    if analysis["resources"]:
        st.markdown("### Official Resources")

        for resource in analysis["resources"]:
            st.markdown(
                f"- [{resource['title']}]({resource['url']})"
            )

    with st.expander("Scenario details"):
        st.write(
            details or "No additional scenario details were entered."
        )

    st.markdown("### Important Limitation")

    st.info(
        "This result is generated from broad educational rules and the "
        "limited information entered. It does not review the person's "
        "immigration documents or replace advice from a qualified attorney."
    )