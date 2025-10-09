import datetime
from simple_salesforce import Salesforce
from models import VisitReport
from functools import partial


def find_account_by_name(sf: Salesforce, account_name: str) -> dict:
    query = f"SELECT Id, Name FROM Account WHERE Name LIKE '%{account_name}%'"
    results = sf.query(query)["records"]

    if len(results) == 1:
        return {
            "status": "single_found",
            "account_name": results[0]["Name"],
            "account_id": results[0]["Id"],
        }
    elif len(results) > 1:
        matched_accounts = [{"name": r["Name"], "id": r["Id"]} for r in results]
        return {
            "status": "multiple_found",
            "matched_accounts": matched_accounts,
        }
    else:
        return {"status": "not_found"}


def list_contacts_for_account(sf: Salesforce, account_name: str) -> dict:
    account_query = f"SELECT Id FROM Account WHERE Name LIKE '%{account_name}%'"
    account_results = sf.query(account_query)["records"]

    if not account_results:
        return {"contacts": []}

    account_ids = [acc["Id"] for acc in account_results]
    ids_str = ",".join(f"'{acc_id}'" for acc_id in account_ids)

    contact_query = f"""
        SELECT Name, Email, Id
        FROM Contact
        WHERE AccountId IN ({ids_str})
    """
    contact_results = sf.query(contact_query)["records"]

    contacts = [
        {"name": c.get("Name"), "email": c.get("Email"), "Id": c.get("Id")}
        for c in contact_results
    ]
    return {"contacts": contacts}


def upload_visit_report(
    sf: Salesforce,
    account_id: str,
    primary_contact_id: str,
    date: datetime.date,
    location: str,
    division: str,
    subject: str,
    description: str,
):
    object_api_name = "Visit_Report__c"

    payload = {
        "Account__c": account_id,
        "Primary_Contact__c": primary_contact_id,
        "Visit_Date__c": date,
        "Visit_Location__c": location,
        "Related_Product_Division__c": division,
        "Name": subject,
        "Description__c": description,
    }

    result = sf.__getattr__(object_api_name).create(payload)
    return result


TOOLS = [
    {
        "type": "function",
        "name": "find_account_by_name",
        "description": "Searches Salesforce for accounts whose names contain the given string and returns either a single match, multiple matches, or a not-found status.",
        "parameters": {
            "type": "object",
            "properties": {
                "sf": {
                    "type": "object",
                    "description": "An authenticated Salesforce client instance from simple_salesforce.",
                },
                "account_name": {
                    "type": "string",
                    "description": "Full or partial name of the account to search for.",
                },
            },
            "required": ["sf", "account_name"],
        },
    },
    {
        "type": "function",
        "name": "list_contacts_for_account",
        "description": "Retrieves contacts from Salesforce linked to accounts whose names contain the given string.",
        "parameters": {
            "type": "object",
            "properties": {
                "sf": {
                    "type": "object",
                    "description": "An authenticated Salesforce client instance from simple_salesforce.",
                },
                "account_name": {
                    "type": "string",
                    "description": "Full or partial name of the account whose contacts should be listed.",
                },
            },
            "required": ["sf", "account_name"],
        },
    },
    {
        "type": "function",
        "name": "upload_visit_report",
        "description": "Uploads a structured visit report to the Salesforce Visit_Report__c object.",
        "parameters": {
            "type": "object",
            "properties": {
                "account_id": {
                    "type": "string",
                    "description": "Salesforce record ID of the Account (starts with '001').",
                },
                "primary_contact_id": {
                    "type": "string",
                    "description": "Salesforce record ID of the Primary Contact (starts with '003').",
                },
                "date": {
                    "type": "string",
                    "format": "date",
                    "description": "Date of the meeting (YYYY-MM-DD).",
                },
                "location": {
                    "type": "string",
                    "description": "Picklist value for the Visit_Location__c field.",
                },
                "division": {
                    "type": "string",
                    "description": "Picklist value for Related_Product_Division__c.",
                },
                "subject": {
                    "type": "string",
                    "description": "Subject or title of the meeting (stored in Name field).",
                },
                "description": {
                    "type": "string",
                    "description": "Detailed description of the meeting.",
                },
            },
            "required": [
                "account_id",
                "primary_contact_id",
                "date",
                "location",
                "division",
                "subject",
                "description",
            ],
        },
    },
]
