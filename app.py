from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

if not CHROMA_DIR.exists():
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "ingest_documents.py")],
        cwd=PROJECT_ROOT,
        check=True,
    )


from __future__ import annotations
import json
import time
from datetime import date, timedelta
from typing import Any

import pandas as pd
import streamlit as st

from services.answer_generator import generate_educational_answer
from services.evaluation_engine import evaluate_answer
from services.knowledge_retriever import (
    KnowledgeBaseError,
    VisaKnowledgeRetriever,
)
from services.model_providers import (
    generate_gemini_answer,
    generate_local_answer,
    generate_rule_based_answer,
)

from services.benchmark_runner import (
    load_benchmark_questions,
    run_benchmark,
    summarize_benchmark,
)
# -------------------------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------------------------

st.set_page_config(
    page_title="VisaOPS AI",
    page_icon="🛂",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -------------------------------------------------------------------
# CUSTOM STYLING
# -------------------------------------------------------------------

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .visaops-hero {
            padding: 1.5rem;
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 14px;
            margin-bottom: 1.25rem;
        }

        .visaops-disclaimer {
            padding: 0.85rem 1rem;
            border-left: 4px solid #f59e0b;
            background: rgba(245, 158, 11, 0.08);
            border-radius: 6px;
            margin-bottom: 1rem;
        }

        .metric-card {
            padding: 1rem;
            border: 1px solid rgba(128, 128, 128, 0.22);
            border-radius: 10px;
            text-align: center;
        }

        .source-card {
            padding: 1rem;
            border: 1px solid rgba(128, 128, 128, 0.20);
            border-radius: 10px;
            margin-bottom: 0.75rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# CACHED RESOURCES
# -------------------------------------------------------------------

@st.cache_resource
def load_retriever() -> VisaKnowledgeRetriever:
    """Load the ChromaDB knowledge retriever once."""
    return VisaKnowledgeRetriever()


def get_gemini_api_key() -> str:
    """
    Retrieve the Gemini API key from Streamlit secrets.

    Expected file:
    .streamlit/secrets.toml

    Expected content:
    GEMINI_API_KEY = "your-key"
    """
    try:
        return str(st.secrets.get("GEMINI_API_KEY", "")).strip()
    except Exception:
        return ""


# -------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------

def extract_match_text(match: dict[str, Any]) -> str:
    """Extract document text from different possible retriever formats."""
    return str(
        match.get("document")
        or match.get("text")
        or match.get("content")
        or ""
    ).strip()


def extract_match_metadata(match: dict[str, Any]) -> dict[str, Any]:
    """Safely extract metadata."""
    metadata = match.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def extract_source_title(
    match: dict[str, Any],
    source_number: int,
) -> str:
    """Create a readable title for a retrieved source."""
    metadata = extract_match_metadata(match)

    return str(
        metadata.get("title")
        or metadata.get("source")
        or metadata.get("filename")
        or metadata.get("file_name")
        or metadata.get("document_name")
        or f"Source {source_number}"
    )


def format_rule_based_answer(result: Any) -> str:
    """
    Convert the existing answer generator output into displayable text.

    This supports both string and dictionary outputs.
    """
    if isinstance(result, str):
        return result.strip()

    if not isinstance(result, dict):
        return str(result).strip()

    sections: list[str] = []

    summary = (
        result.get("answer")
        or result.get("summary")
        or result.get("educational_summary")
    )

    if summary:
        sections.append(f"### Educational Summary\n\n{summary}")

    considerations = (
        result.get("considerations")
        or result.get("important_considerations")
        or result.get("observations")
    )

    if considerations:
        if isinstance(considerations, list):
            consideration_text = "\n".join(
                f"- {item}" for item in considerations
            )
        else:
            consideration_text = str(considerations)

        sections.append(
            f"### Important Considerations\n\n{consideration_text}"
        )

    next_steps = result.get("next_steps")

    if next_steps:
        if isinstance(next_steps, list):
            next_step_text = "\n".join(
                f"- {item}" for item in next_steps
            )
        else:
            next_step_text = str(next_steps)

        sections.append(f"### Next Steps\n\n{next_step_text}")

    attorney_questions = result.get("attorney_questions")

    if attorney_questions:
        if isinstance(attorney_questions, list):
            attorney_text = "\n".join(
                f"- {item}" for item in attorney_questions
            )
        else:
            attorney_text = str(attorney_questions)

        sections.append(
            f"### Questions to Verify With an Attorney\n\n{attorney_text}"
        )

    resources = result.get("resources") or result.get("sources")

    if resources:
        if isinstance(resources, list):
            resource_text = "\n".join(
                f"- {item}" for item in resources
            )
        else:
            resource_text = str(resources)

        sections.append(f"### Sources\n\n{resource_text}")

    if not sections:
        return str(result)

    return "\n\n".join(sections)


def display_retrieved_sources(
    matches: list[dict[str, Any]],
    expanded: bool = False,
) -> None:
    """Display retrieved knowledge-base passages."""
    if not matches:
        st.info("No supporting knowledge sources were retrieved.")
        return

    for index, match in enumerate(matches, start=1):
        title = extract_source_title(match, index)
        text = extract_match_text(match)
        metadata = extract_match_metadata(match)

        with st.expander(
            f"Source {index}: {title}",
            expanded=expanded,
        ):
            source_url = metadata.get("source_url")

            if source_url:
                st.caption(f"Official source: {source_url}")

            visa_category = metadata.get("visa_category")
            topic = metadata.get("topic")
            last_reviewed = metadata.get("last_reviewed")

            metadata_values = []

            if visa_category:
                metadata_values.append(
                    f"Visa category: {visa_category}"
                )

            if topic:
                metadata_values.append(f"Topic: {topic}")

            if last_reviewed:
                metadata_values.append(
                    f"Last reviewed: {last_reviewed}"
                )

            if metadata_values:
                st.caption(" | ".join(metadata_values))

            if text:
                st.write(text)
            else:
                st.warning(
                    "This source did not contain readable text."
                )


def build_basic_timeline(
    employment_end_date: date,
    filing_date: date | None,
) -> pd.DataFrame:
    """
    Build an educational timeline.

    The 60-day date is displayed as a potential discretionary grace-period
    endpoint, not as a legal conclusion.
    """
    timeline_rows = [
        {
            "Date": employment_end_date,
            "Event": "Employment end date",
            "Description": (
                "The date employment with the sponsoring employer ended."
            ),
        },
        {
            "Date": employment_end_date + timedelta(days=60),
            "Event": "Potential 60-day grace-period endpoint",
            "Description": (
                "This is only a calendar estimate. The available period "
                "may be shorter if the authorized stay expires earlier, "
                "and the grace period is discretionary."
            ),
        },
    ]

    if filing_date:
        timeline_rows.append(
            {
                "Date": filing_date,
                "Event": "Application or petition filing date",
                "Description": (
                    "The date the new filing was submitted or delivered."
                ),
            }
        )

    timeline = pd.DataFrame(timeline_rows)
    timeline = timeline.sort_values("Date").reset_index(drop=True)
    timeline["Date"] = timeline["Date"].astype(str)

    return timeline


# -------------------------------------------------------------------
# LOAD RETRIEVER
# -------------------------------------------------------------------

try:
    retriever = load_retriever()
    retriever_ready = True
    retriever_error = ""
except Exception as exc:
    retriever = None
    retriever_ready = False
    retriever_error = str(exc)


# -------------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------------

with st.sidebar:
    st.header("VisaOPS AI")

    st.write(
        "A zero-cost immigration knowledge and LLM evaluation platform."
    )

    st.divider()

    st.markdown("### System Components")

    st.markdown(
        """
        - ChromaDB retrieval
        - Sentence-transformer embeddings
        - Gemini model
        - Local FLAN-T5 model
        - Deterministic RAG baseline
        - Transparent evaluation metrics
        """
    )

    st.divider()

    if retriever_ready:
        st.success("Knowledge base connected")
    else:
        st.error("Knowledge base unavailable")

    gemini_api_key = get_gemini_api_key()

    if gemini_api_key:
        st.success("Gemini API connected")
    else:
        st.warning("Gemini API key not detected")

    st.divider()

    st.caption("MVP Version: 0.7")


# -------------------------------------------------------------------
# HERO
# -------------------------------------------------------------------

st.markdown(
    """
    <div class="visaops-hero">
        <h1>🛂 VisaOPS AI</h1>
        <p>
            Immigration knowledge retrieval, scenario education, and
            multi-model LLM evaluation in one zero-cost application.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="visaops-disclaimer">
        <strong>Educational use only:</strong>
        VisaOPS AI does not provide legal advice and does not predict
        immigration outcomes. Rules can depend on individual facts,
        government discretion, and current agency guidance. Verify
        important decisions with official sources and a qualified
        immigration attorney.
    </div>
    """,
    unsafe_allow_html=True,
)

if not retriever_ready:
    st.error(
        "VisaOPS could not connect to the local knowledge base. "
        f"Details: {retriever_error}"
    )


# -------------------------------------------------------------------
# MAIN TABS
# -------------------------------------------------------------------

ask_tab, evaluation_tab, timeline_tab = st.tabs(
    [
        "Ask VisaOPS",
        "Model Evaluation Lab",
        "Timeline Simulator",
    ]
)


# -------------------------------------------------------------------
# TAB 1: ASK VISAOPS
# -------------------------------------------------------------------

with ask_tab:
    st.subheader("Ask VisaOPS")

    st.write(
        "Search the local immigration knowledge base and generate an "
        "educational response."
    )

    question = st.text_area(
        "Your immigration question",
        value="What happens after H-1B employment ends?",
        height=110,
        key="ask_question",
    )

    ask_source_count = st.slider(
        "Number of supporting knowledge passages",
        min_value=1,
        max_value=5,
        value=3,
        key="ask_source_count",
    )

    ask_button = st.button(
        "Search Knowledge Base",
        type="primary",
        key="ask_visaops_button",
    )

    if ask_button:
        if not retriever_ready or retriever is None:
            st.error("The knowledge retriever is not available.")

        elif not question.strip():
            st.warning("Enter a question before searching.")

        else:
            try:
                with st.spinner(
                    "Searching the immigration knowledge base..."
                ):
                    matches = retriever.search(
                        query=question.strip(),
                        number_of_results=ask_source_count,
                    )

                if not matches:
                    st.warning(
                        "No relevant knowledge passages were found."
                    )

                else:
                    with st.spinner(
                        "Generating an educational response..."
                    ):
                        generated_result = generate_educational_answer(
                            question=question.strip(),
                            matches=matches,
                        )

                    formatted_answer = format_rule_based_answer(
                        generated_result
                    )

                    st.markdown("### Educational Response")
                    st.markdown(formatted_answer)

                    st.markdown("### Supporting Knowledge Sources")
                    display_retrieved_sources(matches)

            except KnowledgeBaseError as exc:
                st.error(f"Knowledge-base error: {exc}")

            except Exception as exc:
                st.error(f"Unable to generate a response: {exc}")


# -------------------------------------------------------------------
# TAB 2: MODEL EVALUATION LAB
# -------------------------------------------------------------------

with evaluation_tab:
    st.subheader("Model Evaluation Lab")

    st.write(
        "Compare response-generation approaches using the same retrieved "
        "knowledge and transparent, deterministic evaluation metrics."
    )

    evaluation_mode = st.radio(
        "Evaluation mode",
        options=[
            "Single Question",
            "Full Benchmark",
        ],
        horizontal=True,
        key="evaluation_mode",
    )

    # ---------------------------------------------------------------
    # SINGLE QUESTION MODE
    # ---------------------------------------------------------------

    if evaluation_mode == "Single Question":
        col_question, col_settings = st.columns([2, 1])

        with col_question:
            evaluation_question = st.text_area(
                "Question to benchmark",
                value="What happens after H-1B employment ends?",
                height=120,
                key="evaluation_question",
            )

        with col_settings:
            evaluation_source_count = st.slider(
                "Knowledge passages",
                min_value=1,
                max_value=5,
                value=3,
                key="evaluation_source_count",
            )

            include_local_model = st.checkbox(
                "Run local FLAN-T5 Small",
                value=False,
                disabled=True,
                help=(
                    "Local inference is disabled in the public free "
                    "deployment to avoid exceeding memory limits. It can "
                    "still be enabled locally in development."
                ),
                key="include_local_model",
            )

        st.caption(
            "The evaluator scores groundedness, source usage, safety "
            "language, completeness, readability, and latency. It does "
            "not use a paid LLM judge."
        )

        run_evaluation = st.button(
            "Run Model Evaluation",
            type="primary",
            key="run_model_evaluation",
        )

        if run_evaluation:
            if not retriever_ready or retriever is None:
                st.error("The knowledge retriever is unavailable.")

            elif not evaluation_question.strip():
                st.warning(
                    "Enter a question before running the evaluation."
                )

            else:
                try:
                    with st.spinner(
                        "Retrieving shared knowledge passages..."
                    ):
                        matches = retriever.search(
                            query=evaluation_question.strip(),
                            number_of_results=evaluation_source_count,
                        )

                    if not matches:
                        st.warning(
                            "No knowledge passages were found for evaluation."
                        )

                    else:
                        provider_functions: list[
                            tuple[str, Any]
                        ] = [
                            (
                                "Rule-Based RAG",
                                lambda: generate_rule_based_answer(
                                    question=evaluation_question.strip(),
                                    matches=matches,
                                ),
                            )
                        ]

                        if gemini_api_key:
                            provider_functions.append(
                                (
                                    "Gemini",
                                    lambda: generate_gemini_answer(
                                        api_key=gemini_api_key,
                                        question=evaluation_question.strip(),
                                        matches=matches,
                                    ),
                                )
                            )
                        else:
                            st.info(
                                "Gemini was skipped because "
                                "GEMINI_API_KEY was not found."
                            )

                        if include_local_model:
                            provider_functions.append(
                                (
                                    "FLAN-T5 Small",
                                    lambda: generate_local_answer(
                                        question=evaluation_question.strip(),
                                        matches=matches,
                                    ),
                                )
                            )

                        results: list[dict[str, Any]] = []
                        progress_bar = st.progress(0)
                        progress_text = st.empty()
                        total_providers = len(provider_functions)

                        for provider_index, (
                            display_name,
                            provider_function,
                        ) in enumerate(provider_functions, start=1):
                            progress_text.write(
                                f"Running {display_name}..."
                            )

                            start_time = time.perf_counter()

                            try:
                                response = provider_function()
                            except Exception as exc:
                                response = {
                                    "answer": "",
                                    "error": str(exc),
                                }

                            latency_seconds = (
                                time.perf_counter() - start_time
                            )

                            if response.get("error"):
                                st.error(
                                    f"{display_name} failed: "
                                    f"{response['error']}"
                                )
                            else:
                                answer = str(
                                    response.get("answer", "")
                                ).strip()

                                metrics = evaluate_answer(
                                    answer=answer,
                                    matches=matches,
                                    latency_seconds=latency_seconds,
                                )

                                results.append(
                                    {
                                        "model": display_name,
                                        "answer": answer,
                                        **metrics,
                                    }
                                )

                            progress_bar.progress(
                                provider_index / total_providers
                            )

                        progress_text.empty()
                        progress_bar.empty()

                        if not results:
                            st.error(
                                "No model completed successfully."
                            )

                        else:
                            score_table = pd.DataFrame(
                                [
                                    {
                                        "Model": result["model"],
                                        "Overall": result["overall_score"],
                                        "Groundedness": result[
                                            "groundedness"
                                        ],
                                        "Source Usage": result[
                                            "source_usage"
                                        ],
                                        "Safety": result["safety"],
                                        "Completeness": result[
                                            "completeness"
                                        ],
                                        "Readability": result[
                                            "readability"
                                        ],
                                        "Latency (sec)": result[
                                            "latency_seconds"
                                        ],
                                    }
                                    for result in results
                                ]
                            )

                            score_table = score_table.sort_values(
                                by="Overall",
                                ascending=False,
                            ).reset_index(drop=True)

                            st.markdown("### Evaluation Leaderboard")

                            st.dataframe(
                                score_table,
                                use_container_width=True,
                                hide_index=True,
                            )

                            winner = score_table.iloc[0]

                            metric_col1, metric_col2, metric_col3 = (
                                st.columns(3)
                            )

                            with metric_col1:
                                st.metric(
                                    "Top Model",
                                    str(winner["Model"]),
                                )

                            with metric_col2:
                                st.metric(
                                    "Overall Score",
                                    f"{winner['Overall']}/100",
                                )

                            with metric_col3:
                                st.metric(
                                    "Latency",
                                    f"{winner['Latency (sec)']} sec",
                                )

                            st.success(
                                f"Top evaluated model: "
                                f"{winner['Model']} with an overall "
                                f"score of {winner['Overall']}."
                            )

                            st.markdown("### Overall Score Comparison")
                            st.bar_chart(
                                score_table.set_index("Model")[["Overall"]]
                            )

                            st.markdown("### Model Responses")

                            sorted_results = sorted(
                                results,
                                key=lambda item: item["overall_score"],
                                reverse=True,
                            )

                            for result in sorted_results:
                                with st.expander(
                                    (
                                        f"{result['model']} — "
                                        f"Overall Score: "
                                        f"{result['overall_score']}"
                                    ),
                                    expanded=False,
                                ):
                                    st.markdown(result["answer"])
                                    st.divider()

                                    response_metrics = pd.DataFrame(
                                        [
                                            {
                                                "Metric": "Groundedness",
                                                "Score": result[
                                                    "groundedness"
                                                ],
                                            },
                                            {
                                                "Metric": "Source Usage",
                                                "Score": result[
                                                    "source_usage"
                                                ],
                                            },
                                            {
                                                "Metric": "Safety",
                                                "Score": result["safety"],
                                            },
                                            {
                                                "Metric": "Completeness",
                                                "Score": result[
                                                    "completeness"
                                                ],
                                            },
                                            {
                                                "Metric": "Readability",
                                                "Score": result[
                                                    "readability"
                                                ],
                                            },
                                        ]
                                    )

                                    st.dataframe(
                                        response_metrics,
                                        use_container_width=True,
                                        hide_index=True,
                                    )

                                    st.caption(
                                        f"Response latency: "
                                        f"{result['latency_seconds']} seconds"
                                    )

                            st.markdown(
                                "### Retrieved Evaluation Context"
                            )
                            display_retrieved_sources(matches)

                            csv_data = score_table.to_csv(
                                index=False
                            ).encode("utf-8")

                            json_data = json.dumps(
                                results,
                                indent=2,
                                default=str,
                            )

                            download_col1, download_col2 = st.columns(2)

                            with download_col1:
                                st.download_button(
                                    label="Download Scores CSV",
                                    data=csv_data,
                                    file_name=(
                                        "visaops_model_evaluation_scores.csv"
                                    ),
                                    mime="text/csv",
                                )

                            with download_col2:
                                st.download_button(
                                    label="Download Results JSON",
                                    data=json_data,
                                    file_name=(
                                        "visaops_model_evaluation_results.json"
                                    ),
                                    mime="application/json",
                                )

                except KnowledgeBaseError as exc:
                    st.error(f"Knowledge-base error: {exc}")

                except Exception as exc:
                    st.error(f"Evaluation failed: {exc}")

    # ---------------------------------------------------------------
    # FULL BENCHMARK MODE
    # ---------------------------------------------------------------

    else:
        st.markdown("### Full Benchmark")

        st.write(
            "Run a fixed set of immigration questions through the same "
            "retrieval, generation, and evaluation pipeline."
        )

        try:
            benchmark_questions = load_benchmark_questions()
        except Exception as exc:
            benchmark_questions = []
            st.error(f"Unable to load benchmark questions: {exc}")

        if benchmark_questions:
            benchmark_frame = pd.DataFrame(benchmark_questions)
            category_count = (
                benchmark_frame["category"].nunique()
                if "category" in benchmark_frame.columns
                else 0
            )

            info_col1, info_col2 = st.columns(2)

            with info_col1:
                st.metric(
                    "Benchmark Questions",
                    len(benchmark_questions),
                )

            with info_col2:
                st.metric(
                    "Visa Categories",
                    category_count,
                )

            with st.expander(
                "View benchmark question set",
                expanded=False,
            ):
                st.dataframe(
                    benchmark_frame,
                    use_container_width=True,
                    hide_index=True,
                )

            benchmark_source_count = st.slider(
                "Knowledge passages per benchmark question",
                min_value=1,
                max_value=5,
                value=3,
                key="benchmark_source_count",
            )

            st.caption(
                "The public benchmark compares the rule-based RAG engine "
                "and Gemini. Local FLAN-T5 is excluded to keep the hosted "
                "application within free memory limits."
            )

            run_full_benchmark = st.button(
                "Run Full Benchmark",
                type="primary",
                key="run_full_benchmark",
            )

            if run_full_benchmark:
                if not retriever_ready or retriever is None:
                    st.error(
                        "The knowledge retriever is unavailable."
                    )

                else:
                    provider_factories = [
                        (
                            "Rule-Based RAG",
                            lambda question, matches:
                            generate_rule_based_answer(
                                question=question,
                                matches=matches,
                            ),
                        )
                    ]

                    if gemini_api_key:
                        provider_factories.append(
                            (
                                "Gemini",
                                lambda question, matches:
                                generate_gemini_answer(
                                    api_key=gemini_api_key,
                                    question=question,
                                    matches=matches,
                                ),
                            )
                        )
                    else:
                        st.warning(
                            "Gemini is unavailable because its API key "
                            "was not found. The benchmark will run only "
                            "the rule-based model."
                        )

                    with st.spinner(
                        "Running the full benchmark. This may take "
                        "several minutes..."
                    ):
                        benchmark_scores, benchmark_details = (
                            run_benchmark(
                                questions=benchmark_questions,
                                retriever=retriever,
                                provider_factories=provider_factories,
                                number_of_results=benchmark_source_count,
                            )
                        )

                    if benchmark_scores.empty:
                        st.error(
                            "No benchmark results were generated."
                        )

                    else:
                        benchmark_summary = summarize_benchmark(
                            benchmark_scores
                        )

                        st.markdown("### Benchmark Leaderboard")

                        st.dataframe(
                            benchmark_summary,
                            use_container_width=True,
                            hide_index=True,
                        )

                        winner = benchmark_summary.iloc[0]

                        benchmark_col1, benchmark_col2, benchmark_col3 = (
                            st.columns(3)
                        )

                        with benchmark_col1:
                            st.metric(
                                "Top Model",
                                str(winner["Model"]),
                            )

                        with benchmark_col2:
                            st.metric(
                                "Average Overall Score",
                                f"{winner['Overall']}/100",
                            )

                        with benchmark_col3:
                            st.metric(
                                "Average Latency",
                                f"{winner['Latency (sec)']} sec",
                            )

                        st.success(
                            f"Benchmark winner: {winner['Model']} with "
                            f"an average overall score of "
                            f"{winner['Overall']}."
                        )

                        st.markdown("### Average Overall Score")

                        st.bar_chart(
                            benchmark_summary.set_index("Model")[
                                ["Overall"]
                            ]
                        )

                        st.markdown(
                            "### Average Metric Comparison"
                        )

                        metric_columns = [
                            "Groundedness",
                            "Source Usage",
                            "Safety",
                            "Completeness",
                            "Readability",
                        ]

                        st.bar_chart(
                            benchmark_summary.set_index("Model")[
                                metric_columns
                            ]
                        )

                        st.markdown(
                            "### Question-Level Results"
                        )

                        categories = sorted(
                            benchmark_scores[
                                "Category"
                            ].dropna().unique().tolist()
                        )

                        selected_category = st.selectbox(
                            "Filter by category",
                            options=["All", *categories],
                            key="benchmark_category_filter",
                        )

                        filtered_scores = benchmark_scores.copy()

                        if selected_category != "All":
                            filtered_scores = filtered_scores[
                                filtered_scores["Category"]
                                == selected_category
                            ]

                        st.dataframe(
                            filtered_scores,
                            use_container_width=True,
                            hide_index=True,
                        )

                        st.markdown(
                            "### Detailed Model Responses"
                        )

                        for detail in benchmark_details:
                            if detail.get("error"):
                                continue

                            title = (
                                f"{detail['question_id']} | "
                                f"{detail['model']} | "
                                f"{detail['category']}"
                            )

                            with st.expander(
                                title,
                                expanded=False,
                            ):
                                st.markdown(
                                    f"**Question:** {detail['question']}"
                                )
                                st.markdown(detail["answer"])

                                metrics = detail.get("metrics", {})

                                if metrics:
                                    st.caption(
                                        " | ".join(
                                            [
                                                (
                                                    "Overall: "
                                                    f"{metrics.get('overall_score')}"
                                                ),
                                                (
                                                    "Groundedness: "
                                                    f"{metrics.get('groundedness')}"
                                                ),
                                                (
                                                    "Source usage: "
                                                    f"{metrics.get('source_usage')}"
                                                ),
                                                (
                                                    "Latency: "
                                                    f"{metrics.get('latency_seconds')} sec"
                                                ),
                                            ]
                                        )
                                    )

                        summary_csv = benchmark_summary.to_csv(
                            index=False
                        ).encode("utf-8")

                        detailed_csv = benchmark_scores.to_csv(
                            index=False
                        ).encode("utf-8")

                        details_json = json.dumps(
                            benchmark_details,
                            indent=2,
                            default=str,
                        )

                        download_col1, download_col2, download_col3 = (
                            st.columns(3)
                        )

                        with download_col1:
                            st.download_button(
                                label="Download Summary CSV",
                                data=summary_csv,
                                file_name=(
                                    "visaops_benchmark_summary.csv"
                                ),
                                mime="text/csv",
                            )

                        with download_col2:
                            st.download_button(
                                label="Download Detailed CSV",
                                data=detailed_csv,
                                file_name=(
                                    "visaops_benchmark_results.csv"
                                ),
                                mime="text/csv",
                            )

                        with download_col3:
                            st.download_button(
                                label="Download JSON",
                                data=details_json,
                                file_name=(
                                    "visaops_benchmark_details.json"
                                ),
                                mime="application/json",
                            )


# -------------------------------------------------------------------
# TAB 3: TIMELINE SIMULATOR
# -------------------------------------------------------------------

with timeline_tab:
    st.subheader("Visa Timeline Simulator")

    st.write(
        "Create an educational calendar view for an employment-ending "
        "scenario and a subsequent filing."
    )

    st.warning(
        "The calculated date is not a determination of lawful status or "
        "eligibility. The grace period is discretionary and may end "
        "earlier based on the authorized-stay expiration date."
    )

    timeline_col1, timeline_col2 = st.columns(2)

    with timeline_col1:
        employment_end_date = st.date_input(
            "Employment end date",
            value=date.today(),
            key="employment_end_date",
        )

    with timeline_col2:
        include_filing_date = st.checkbox(
            "Include an application or petition filing",
            value=True,
            key="include_filing_date",
        )

        filing_date: date | None = None

        if include_filing_date:
            filing_date = st.date_input(
                "Filing or delivery date",
                value=date.today(),
                key="filing_date",
            )

    generate_timeline = st.button(
        "Generate Timeline",
        type="primary",
        key="generate_timeline",
    )

    if generate_timeline:
        timeline_df = build_basic_timeline(
            employment_end_date=employment_end_date,
            filing_date=filing_date,
        )

        estimated_grace_end = (
            employment_end_date + timedelta(days=60)
        )

        metric1, metric2, metric3 = st.columns(3)

        with metric1:
            st.metric(
                "Employment End",
                employment_end_date.strftime("%b %d, %Y"),
            )

        with metric2:
            st.metric(
                "Potential 60-Day Endpoint",
                estimated_grace_end.strftime("%b %d, %Y"),
            )

        with metric3:
            if filing_date:
                days_after_end = (
                    filing_date - employment_end_date
                ).days

                st.metric(
                    "Filing Timing",
                    f"{days_after_end} days after employment end",
                )
            else:
                st.metric("Filing Timing", "Not provided")

        st.markdown("### Scenario Timeline")

        st.dataframe(
            timeline_df,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("### Educational Observations")

        if filing_date:
            filing_difference = (
                filing_date - employment_end_date
            ).days

            if filing_difference < 0:
                st.info(
                    "The entered filing date is before the employment "
                    "end date. Confirm that both dates are correct."
                )

            elif filing_difference <= 60:
                st.info(
                    "The filing date falls within 60 calendar days of "
                    "the employment end date. This does not independently "
                    "establish eligibility or confirm maintenance of status."
                )

            else:
                st.warning(
                    "The filing date is more than 60 calendar days after "
                    "the employment end date. Obtain individualized legal "
                    "guidance regarding status, filing options, and any "
                    "possible late-filing explanation."
                )

        st.markdown("### Questions to Verify")

        st.markdown(
            """
            - What is the expiration date on the latest Form I-94?
            - Was the employment termination date recorded correctly?
            - Was the filing properly submitted and accepted by USCIS?
            - Did USCIS issue a rejection, receipt notice, request for evidence, or denial?
            - Is employment authorization available while the filing is pending?
            - Are there travel or consular-processing consequences?
            """
        )


# -------------------------------------------------------------------
# FOOTER
# -------------------------------------------------------------------

st.divider()

st.caption(
    "VisaOPS AI is an educational portfolio project. It does not replace "
    "official government guidance or individualized legal advice."
)
0
