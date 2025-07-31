import datetime
from models import VisitReport


def create_visit_report(data: VisitReport) -> dict:
    summary_lines = [
        f"Contact Name: {data.contact_name}",
        f"Company Name: {data.company_name}",
        f"Phone: {data.phone}",
        f"Date: {data.date}",
    ]

    if data.favorite_drink:
        summary_lines.append(f"Favorite Drink: {data.favorite_drink}")

    summary = "\n".join(summary_lines)

    print("\n[TOOL] Visit report created:")
    print(summary)

    return {
        "status": "success",
        "summary": summary,
    }


def write_visit_report_to_file(data: VisitReport, filename: str) -> None:
    with open(filename, "w") as file:
        file.write(f"Contact Name: {data.contact_name}\n")
        file.write(f"Company Name: {data.company_name}\n")
        file.write(f"Phone: {data.phone}\n")
        file.write(f"Date: {data.date}\n")
        if data.favorite_drink:
            file.write(f"Favorite Drink: {data.favorite_drink}\n")


def find_account_by_name(account_name: str) -> dict:
    matched = [
        {"name": "igus GmbH", "id": "A001"},
        {"name": "igus North America", "id": "A002"},
    ]
    results = [acc for acc in matched if account_name.lower() in acc["name"].lower()]

    if len(results) == 1:
        return {
            "status": "single_found",
            "account_name": results[0]["name"],
            "account_id": results[0]["id"],
        }
    elif len(results) > 1:
        return {
            "status": "multiple_found",
            "matched_accounts": results,
        }
    else:
        return {
            "status": "not_found",
        }


def list_contacts_for_account(account_name: str) -> dict:
    contacts = {
        "igus GmbH": [
            {"name": "Max Mustermann", "email": "max@igus.de"},
            {"name": "Erika Beispiel", "email": "erika@igus.de"},
        ],
        "igus North America": [
            {"name": "John Doe", "email": "john@igus.com"},
            {"name": "Jane Smith", "email": "jane@igus.com"},
        ],
    }
    matched = contacts.get(account_name, [])
    return {"contacts": matched}


def prepare_for_upload(
    account_name: str,
    primary_contact: str,
    date: datetime.date,
    location: str,
    division: str,
    subject: str,
    description: str,
    machines: str,
) -> dict:
    return {
        "AccountName__c": account_name,
        "PrimaryContact__c": primary_contact,
        "Date__c": date,
        "Location__c": location,
        "Division__c": division,
        "Subject__c": subject,
        "Description__c": description,
        "Machines__c": machines,
    }


TOOLS = [
    {
        "type": "function",
        "name": "find_account_by_name",
        "description": "Searches accounts/company by name to prove their validity.",
        "parameters": {
            "type": "object",
            "properties": {
                "account_name": {"type": "string"},
            },
            "required": ["account_name"],
        },
    },
    {
        "type": "function",
        "name": "list_contacts_for_account",
        "description": "Lists contacts for a account/company name to prove their validity.",
        "parameters": {
            "type": "object",
            "properties": {
                "account_name": {"type": "string"},
            },
            "required": ["account_name"],
        },
    },
    {
        "type": "function",
        "name": "prepare_for_upload",
        "description": f"Processes data gathered for a visit report to prepare it for upload.",
        "parameters": {
            "type": "object",
            "properties": {
                "account_name": {
                    "type": "string",
                    "description": "Company the meeting was with.",
                },
                "primary_contact": {
                    "type": "string",
                    "description": "Primary contact person for the meeting.",
                },
                "date": {
                    "type": "string",
                    "format": "date",
                    "description": "Date of the meeting (YYYY-MM-DD).",
                },
                "location": {
                    "type": "string",
                    "description": "Location of the meeting.",
                },
                "division": {
                    "type": "string",
                    "description": "Division involved in the meeting.",
                },
                "subject": {
                    "type": "string",
                    "description": "Subject or title of the meeting.",
                },
                "description": {
                    "type": "string",
                    "description": "Detailed description of the meeting.",
                },
                "machines": {
                    "type": "string",
                    "description": "Optional: machines discussed during the meeting.",
                    "nullable": True,
                },
            },
            "required": [
                "account_name",
                "primary_contact",
                "date",
                "location",
                "division",
                "subject",
                "description",
            ],
        },
    },
]

TOOL_MAP = {
    "find_account_by_name": find_account_by_name,
    "list_contacts_for_account": list_contacts_for_account,
    "prepare_for_upload": prepare_for_upload,
}
