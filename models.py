import datetime
from pydantic import BaseModel, Field
from enum import Enum


class VisitReport(BaseModel):
    class Division(str, Enum):
        ECHAIN = "e-chain"
        BEARINGS = "bearings"
        ECHAIN_BEARINGS = "e-chain&bearings"

    class Location(str, Enum):
        REMOTE = "Remote"
        CLIENT = "Client"
        AT_IGUS = "At igus"
        OTHER = "Other"

    Account__c: str = Field(..., description="Company the meeting was with.")
    Primary_Contact__c: str = Field(
        ..., description="Primary contact person for the meeting."
    )
    Visit_Date__c: datetime.date = Field(..., description="Date of the meeting.")
    Visit_Location__c: Location = Field(
        ...,
        description="Location of the meeting. Only allowed options: remote, client, igus, other.",
    )
    Related_Product_Division__c: Division = Field(
        ...,
        description="Division involved. Only allowed options: e-chain, bearings, e-chain&bearings.",
    )
    Name: str = Field(..., description="Subject or title of the meeting.")
    Description__c: str = Field(
        ..., description="Detailed description of the meeting content."
    )
