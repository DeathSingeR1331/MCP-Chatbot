"""
gmail_service.py
~~~~~~~~~~~~~~~~~~

This module wraps the Gmail API and OAuth 2.0 dance to allow the MCP
chatbot to fetch unread messages and send email on behalf of a user.  A
user must first authorize the application by visiting a Google consent
screen.  Once authorized, a refresh token is stored on disk under the
`tokens/` directory, keyed by `user_id`.  Subsequent calls to Gmail
functions will automatically refresh the access token as needed.

Functions provided:

* start_oauth(user_id, redirect_uri) -> GmailAuthURLs
  Returns a URL that the frontend should redirect the user to in order to
  authorize access.  The state parameter encodes the user_id.

* finish_oauth(state, code, redirect_uri) -> Optional[str]
  Completes the OAuth dance by exchanging the authorization code for
  tokens and storing them on disk.  Returns the user_id encoded in the
  state or None on error.

* list_unread(user_id, max_results) -> Dict
  Returns a dictionary with a boolean 'authorized' key and a list of
  unread messages.  If the user has not authorized Gmail the function
  returns {'authorized': False, 'messages': []}.

* send_message(user_id, to_email, subject, body) -> Dict
  Send a plaintext email from the user's account.  Returns a dict
  containing the Gmail message id on success, or indicates lack of
  authorization.

Usage:

  from gmail_service import start_oauth, finish_oauth, list_unread, send_message

  # Step 1: redirect user to start_oauth(...).auth_url
  # Step 2: in callback handler, call finish_oauth(...) with the code
  # Step 3: call list_unread or send_message with the user_id

You must create a Google Cloud OAuth client and place the JSON file in
`mcp-chatbot/credentials.json`.  See README for detailed instructions.
"""

from __future__ import annotations

import base64
import json
import os
import secrets
from dataclasses import dataclass
from typing import Dict, List, Optional

from email.message import EmailMessage
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build

# The scopes required for listing unread messages and sending email.
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
]

# Path to the OAuth client secrets.  This file should be provided by the user.
CREDENTIALS_PATH = os.path.join(os.path.dirname(__file__), "credentials.json")

# Directory where tokens will be stored.  One token file per user ID.
TOKENS_DIR = os.path.join(os.path.dirname(__file__), "tokens")
os.makedirs(TOKENS_DIR, exist_ok=True)


@dataclass
class GmailAuthURLs:
    """Data container returned by start_oauth()."""
    auth_url: str
    state: str


def _state_for(user_id: str) -> str:
    """Encode the user_id and a random nonce into the state parameter."""
    payload = {"user_id": user_id, "nonce": secrets.token_urlsafe(8)}
    return base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()


def _parse_state(state: str) -> Dict[str, str]:
    try:
        return json.loads(base64.urlsafe_b64decode(state.encode()).decode())
    except Exception:
        return {}


def user_token_path(user_id: str) -> str:
    """Return the filesystem path to the token file for a given user."""
    return os.path.join(TOKENS_DIR, f"{user_id}.json")


def build_flow(redirect_uri: str, state: Optional[str] = None) -> Flow:
    """Construct an OAuth flow configured for our scopes and redirect URI."""
    return Flow.from_client_secrets_file(
        CREDENTIALS_PATH,
        scopes=SCOPES,
        redirect_uri=redirect_uri,
        state=state,
    )


def start_oauth(user_id: str, redirect_uri: str) -> GmailAuthURLs:
    """Start the OAuth flow and return the authorization URL and state."""
    state = _state_for(user_id)
    flow = build_flow(redirect_uri, state=state)
    auth_url, _ = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
        state=state,
    )
    return GmailAuthURLs(auth_url=auth_url, state=state)


def finish_oauth(state: str, code: str, redirect_uri: str) -> Optional[str]:
    """Complete OAuth by exchanging the code for tokens and storing them.

    Args:
        state: The opaque state parameter returned from Google.
        code: The authorization code returned from Google.
        redirect_uri: The same redirect URI passed to start_oauth().

    Returns:
        The user_id encoded in the state parameter, or None if the state
        could not be decoded.  Raises exceptions if token exchange fails.
    """
    payload = _parse_state(state)
    user_id = payload.get("user_id")
    if not user_id:
        return None
    flow = build_flow(redirect_uri, state=state)
    flow.fetch_token(code=code)
    creds: Credentials = flow.credentials
    # Persist tokens
    with open(user_token_path(user_id), "w", encoding="utf-8") as f:
        f.write(creds.to_json())
    return user_id


def _load_creds(user_id: str) -> Optional[Credentials]:
    """Load stored credentials for the given user, or None if not found."""
    path = user_token_path(user_id)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Credentials.from_authorized_user_info(data, SCOPES)


def _gmail_service(user_id: str):
    """Return an authenticated Gmail service or None if not authorised."""
    creds = _load_creds(user_id)
    if not creds:
        return None
    # google-auth handles token refresh automatically when used with the API client
    return build("gmail", "v1", credentials=creds, cache_discovery=False)


def list_unread(user_id: str, max_results: int = 10) -> Dict[str, List[Dict]]:
    """Return a simplified list of unread messages from the user's inbox."""
    svc = _gmail_service(user_id)
    if not svc:
        return {"authorized": False, "messages": []}
    resp = svc.users().messages().list(
        userId="me",
        labelIds=["INBOX", "UNREAD"],
        maxResults=max(1, min(max_results, 50)),
    ).execute()
    items: List[Dict[str, str]] = []
    for msg in resp.get("messages", []):
        m = svc.users().messages().get(
            userId="me",
            id=msg["id"],
            format="metadata",
            metadataHeaders=["From", "Subject", "Date"],
        ).execute()
        headers = {h["name"]: h["value"] for h in m.get("payload", {}).get("headers", [])}
        items.append({
            "id": msg["id"],
            "from": headers.get("From", ""),
            "subject": headers.get("Subject", "(no subject)"),
            "date": headers.get("Date", ""),
            "snippet": m.get("snippet", ""),
        })
    return {"authorized": True, "messages": items}


def send_message(user_id: str, to_email: str, subject: str, body: str) -> Dict[str, str]:
    """Send a plaintext email from the user's Gmail account."""
    svc = _gmail_service(user_id)
    if not svc:
        return {"authorized": False, "status": "not_authorized"}
    em = EmailMessage()
    em["To"] = to_email
    em["Subject"] = subject
    em.set_content(body)
    raw = base64.urlsafe_b64encode(em.as_bytes()).decode()
    result = svc.users().messages().send(userId="me", body={"raw": raw}).execute()
    return {"authorized": True, "status": "sent", "messageId": result.get("id", "")}