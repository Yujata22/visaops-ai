from datetime import date, timedelta
from typing import Optional


def build_timeline(
    visa_type: str,
    situation: str,
    event_date: date,
    filing_date: Optional[date] = None,
) -> list[dict]:
    """
    Build a simple educational timeline based on a fictional visa scenario.

    This function does not determine immigration status or provide legal advice.
    """

    timeline = [
        {
            "date": event_date,
            "title": situation,
            "description": (
                f"Reported event associated with the selected "
                f"{visa_type} scenario."
            ),
            "status": "completed",
        }
    ]

    if situation == "Employment ended" and visa_type == "H-1B":
        grace_period_end = event_date + timedelta(days=60)

        timeline.append(
            {
                "date": grace_period_end,
                "title": "Potential 60-day grace-period endpoint",
                "description": (
                    "A discretionary grace period of up to 60 days may apply. "
                    "The actual period may be shorter depending on the person's "
                    "I-94 expiration date and individual circumstances."
                ),
                "status": "important",
            }
        )

    if filing_date:
        timeline.append(
            {
                "date": filing_date,
                "title": "Application or petition filed",
                "description": (
                    "The scenario reports that an immigration application "
                    "or petition was submitted on this date."
                ),
                "status": "completed",
            }
        )

    timeline.sort(
        key=lambda timeline_event: timeline_event["date"]
    )

    return timeline