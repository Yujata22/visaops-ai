from datetime import date
from typing import Optional


def analyze_scenario(
    visa_type: str,
    situation: str,
    event_date: date,
    filing_date: Optional[date] = None,
    details: str = "",
) -> dict:
    """
    Generate cautious, educational observations for a fictional visa scenario.

    This function does not determine immigration status, eligibility,
    work authorization, or the outcome of any immigration filing.
    """

    risk_level = "Needs Review"
    observations: list[str] = []
    next_steps: list[str] = []
    attorney_questions: list[str] = []
    resources: list[dict] = []

    if visa_type == "H-1B" and situation == "Employment ended":
        risk_level = "Time Sensitive"

        observations.extend(
            [
                (
                    "Employment termination may create a time-sensitive "
                    "immigration situation."
                ),
                (
                    "A discretionary grace period of up to 60 days may apply, "
                    "but the available period may be shorter if the person's "
                    "authorized stay expires earlier."
                ),
                (
                    "The selected event date should be compared with the "
                    "person's I-94 expiration date and petition validity dates."
                ),
            ]
        )

        next_steps.extend(
            [
                "Confirm the last day of employment.",
                "Review the most recent Form I-94 expiration date.",
                "Collect recent approval notices and employment records.",
                (
                    "Identify whether a new employer petition, change of status, "
                    "departure, or another option is being considered."
                ),
            ]
        )

        attorney_questions.extend(
            [
                (
                    "What date should be treated as the beginning of the "
                    "potential grace period?"
                ),
                (
                    "Does the I-94 expiration date shorten the potential "
                    "grace period?"
                ),
                (
                    "What filings may be available based on this timeline?"
                ),
                (
                    "When, if at all, could employment with a new employer begin?"
                ),
            ]
        )

        resources.append(
            {
                "title": (
                    "USCIS: Options for Nonimmigrant Workers Following "
                    "Termination of Employment"
                ),
                "url": (
                    "https://www.uscis.gov/archive/"
                    "options-for-nonimmigrant-workers-following-"
                    "termination-of-employment-0"
                ),
            }
        )

    elif visa_type == "H-1B" and situation == "Considering an H-1B transfer":
        risk_level = "Review Before Filing"

        observations.extend(
            [
                (
                    "An H-1B change-of-employer scenario generally requires "
                    "review of the person's current status, prior approval "
                    "notices, I-94, and the proposed employer's petition."
                ),
                (
                    "The app cannot determine whether the person qualifies "
                    "for employment portability."
                ),
            ]
        )

        next_steps.extend(
            [
                "Collect all prior H-1B approval notices.",
                "Locate the most recent I-94 record.",
                "Confirm the proposed employment start date.",
                "Ask the employer whether premium processing is being considered.",
            ]
        )

        attorney_questions.extend(
            [
                "Does this case qualify for H-1B portability?",
                "When could employment with the new employer begin?",
                "Would consular processing be required?",
                "Are there any gaps or status-maintenance concerns?",
            ]
        )

        resources.append(
            {
                "title": "USCIS: H-1B Specialty Occupations",
                "url": (
                    "https://www.uscis.gov/working-in-the-united-states/"
                    "h-1b-specialty-occupations"
                ),
            }
        )

    elif situation == "Change of status filed":
        risk_level = "Pending Filing Review"

        observations.extend(
            [
                (
                    "A pending change-of-status application does not "
                    "automatically provide employment authorization."
                ),
                (
                    "The filing date, receipt notice, requested status, "
                    "I-94 expiration date, and maintenance of the prior status "
                    "may all be relevant."
                ),
            ]
        )

        if filing_date:
            if filing_date < event_date:
                observations.append(
                    (
                        "The reported filing date occurs before the selected "
                        "event date. Confirm whether both dates are correct."
                    )
                )
            else:
                observations.append(
                    (
                        "A filing date was provided and should be verified "
                        "against the courier record and USCIS receipt notice."
                    )
                )
        else:
            observations.append(
                (
                    "No filing date was provided. The filing date is important "
                    "for understanding the scenario."
                )
            )

        next_steps.extend(
            [
                "Locate the USCIS receipt notice, if one was issued.",
                "Confirm the form type and requested status.",
                "Verify the filing date and delivery confirmation.",
                "Review the I-94 expiration date.",
            ]
        )

        attorney_questions.extend(
            [
                "Was the application timely and properly filed?",
                "What activities are permitted while the application is pending?",
                "Does the applicant need to maintain the prior status?",
                "What happens if USCIS issues a request for evidence or denial?",
            ]
        )

        resources.extend(
            [
                {
                    "title": "USCIS: Change My Nonimmigrant Status",
                    "url": (
                        "https://www.uscis.gov/visit-the-united-states/"
                        "change-my-nonimmigrant-status"
                    ),
                },
                {
                    "title": "USCIS: Form I-539",
                    "url": "https://www.uscis.gov/i-539",
                },
            ]
        )

    elif situation == "Application rejected":
        risk_level = "Immediate Document Review"

        observations.extend(
            [
                (
                    "A rejection generally means USCIS did not accept the "
                    "submission for processing."
                ),
                (
                    "A rejected submission should not automatically be treated "
                    "as having the same effect as an accepted filing."
                ),
                (
                    "The rejection notice and returned package should be "
                    "reviewed carefully before resubmission."
                ),
            ]
        )

        next_steps.extend(
            [
                "Read every page of the rejection notice.",
                "Identify the exact rejection reason.",
                "Preserve the returned package and delivery records.",
                "Confirm whether corrected forms, fees, or signatures are required.",
                "Obtain case-specific legal advice before relying on a resubmission.",
            ]
        )

        attorney_questions.extend(
            [
                "Was the original submission considered properly filed?",
                "What filing date, if any, can be relied upon?",
                "Can the application be resubmitted?",
                "Does the rejection create a status or timing concern?",
            ]
        )

        resources.append(
            {
                "title": "USCIS: Form I-539",
                "url": "https://www.uscis.gov/i-539",
            }
        )

    elif situation == "Application denied":
        risk_level = "Urgent Legal Review"

        observations.extend(
            [
                (
                    "A denial may affect the person's available options, "
                    "depending on the requested benefit and current status."
                ),
                (
                    "The decision notice should be reviewed for the effective "
                    "date, reasons, and any motion, appeal, departure, or "
                    "refiling considerations."
                ),
            ]
        )

        next_steps.extend(
            [
                "Read the complete denial decision.",
                "Record the decision date and date received.",
                "Preserve the filing package and supporting evidence.",
                "Speak with a qualified immigration attorney promptly.",
            ]
        )

        attorney_questions.extend(
            [
                "What is the effect of this denial on the person's current status?",
                "Is a motion, appeal, refiling, or departure appropriate?",
                "Is there a deadline associated with the decision?",
                "Does the denial affect employment authorization?",
            ]
        )

    elif visa_type in {"F-1", "F-1 OPT", "F-1 STEM OPT"}:
        risk_level = "Student Status Review"

        observations.extend(
            [
                (
                    "F-1 scenarios may depend on the Form I-20, SEVIS record, "
                    "program dates, employment authorization, and DSO guidance."
                ),
                (
                    "Employment authorization should not be assumed based only "
                    "on a pending application or school enrollment."
                ),
            ]
        )

        next_steps.extend(
            [
                "Review the latest Form I-20.",
                "Confirm the SEVIS status with the school's DSO.",
                "Review any EAD start and expiration dates.",
                "Preserve employment and academic records.",
            ]
        )

        attorney_questions.extend(
            [
                "Is the person's SEVIS record active?",
                "Is employment currently authorized?",
                "Does a pending application affect the student's activities?",
                "Are there travel or program-start-date concerns?",
            ]
        )

        resources.extend(
            [
                {
                    "title": "USCIS: Changing to F or M Student Status",
                    "url": (
                        "https://www.uscis.gov/working-in-the-united-states/"
                        "students-and-exchange-visitors/students-and-employment/"
                        "changing-to-a-nonimmigrant-f-or-m-student-status"
                    ),
                },
                {
                    "title": "USCIS: Optional Practical Training",
                    "url": (
                        "https://www.uscis.gov/working-in-the-united-states/"
                        "students-and-exchange-visitors/"
                        "optional-practical-training-opt-for-f-1-students"
                    ),
                },
            ]
        )

    else:
        observations.extend(
            [
                (
                    "This scenario requires additional facts before meaningful "
                    "educational guidance can be generated."
                ),
                (
                    "Important documents may include the most recent I-94, "
                    "approval notices, receipt notices, visa stamp, and "
                    "employment or school records."
                ),
            ]
        )

        next_steps.extend(
            [
                "Confirm the current immigration classification.",
                "Locate the most recent I-94.",
                "List all important events and filing dates.",
                "Review the situation with a qualified immigration professional.",
            ]
        )

        attorney_questions.extend(
            [
                "What is the person's current immigration status?",
                "What is the controlling expiration date?",
                "Are any applications or petitions currently pending?",
                "What options are available based on the complete timeline?",
            ]
        )

    if details.strip():
        observations.append(
            (
                "Additional scenario details were provided, but the current "
                "rule-based engine does not yet extract facts from free text."
            )
        )

    return {
        "risk_level": risk_level,
        "observations": observations,
        "next_steps": next_steps,
        "attorney_questions": attorney_questions,
        "resources": resources,
    }