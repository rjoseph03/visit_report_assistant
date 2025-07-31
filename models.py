import datetime
from pydantic import BaseModel, Field
from typing import Optional


class VisitReport(BaseModel):
    account_name: str = Field(..., description="Company the meeting was with.")
    primary_contact: str = Field(
        ..., description="Primary contact person for the meeting."
    )
    date: datetime.date = Field(..., description="Date of the meeting.")
    location: str = Field(
        ...,
        description="Location of the meeting. Main options: remote, client, igus, other.",
    )
    division: str = Field(
        ...,
        description="Division involved. Main options: e-chain, bearings, e-chain&bearings.",
    )
    subject: str = Field(..., description="Subject or title of the meeting.")
    description: str = Field(
        ..., description="Detailed description of the meeting content."
    )
    machines: Optional[str] = Field(
        None, description="(Optional) Machines that were discussed during the meeting."
    )
