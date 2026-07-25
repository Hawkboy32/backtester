"""Time-based one-time passwords (TOTP) for the dashboard login — task #22.

WHY THIS IS HAND-ROLLED ON TOP OF streamlit-authenticator. That library (0.4.2)
does ship a `two_factor_auth` flag, but it is available only on
`forgot_password`, `forgot_username` and `register_user` — NOT on `login`, which
is the one flow #22 needs to protect. Its mechanism is also an emailed code sent
through the library author's cloud service, which would mean routing the user's
address and login codes through a third party. So the second factor here is a
standard RFC 6238 TOTP (Google Authenticator, Authy, 1Password, etc.), verified
locally, with nothing leaving the machine.

WHEN THIS ACTUALLY MATTERS. Today the dashboard is bound to localhost and an
attacker would already need to be on this machine, where they could read the
state files directly. 2FA earns its keep the moment the dashboard is reachable
from elsewhere — the planned Raspberry Pi + Tailscale move — because from then on
a stolen password alone would be enough to arm a trading bot. This module exists
so that step doesn't have to wait on it.

STORAGE. The TOTP secret is a credential of equal standing to the password, so it
lives in the OS keyring (Windows Credential Manager) alongside the broker
credentials, never in a file — matching accounts.py. Recovery codes are stored
only as hashes, so the file they live in cannot be replayed to get in.

NOT LOCKING YOURSELF OUT. This is a single-user local dashboard with no support
desk, so the recovery path is deliberately generous and documented in three
places:
  1. Ten single-use recovery codes, generated at enrolment.
  2. `disable(username)` from a local terminal, which needs no code at all —
     legitimate because anyone who can run it already has the machine.
  3. Enrolment is not enforced until `is_enrolled()` is true, and enrolment only
     completes after a code from the user's authenticator app has been verified
     — so a mis-scanned QR can never leave the account locked.
"""

from __future__ import annotations

import hashlib
import json
import secrets

import keyring
import keyring.errors
import pyotp

from backtester.auto_trader_state import STATE_DIR, atomic_write_text

KEYRING_SERVICE = "backtester_dashboard_2fa"
ISSUER = "Trading Bot Backtester"
RECOVERY_CODES_PATH = STATE_DIR / "recovery_codes.json"
NUM_RECOVERY_CODES = 10
# One step of clock drift either way. TOTP steps are 30s, so this tolerates a
# phone up to ~30s out from the laptop — common, and far cheaper than the
# support burden of "the code says it's wrong but it isn't".
VALID_WINDOW = 1


# --------------------------------------------------------------- secret storage


def _keyring_key(username: str) -> str:
    return f"{username}:totp_secret"


def get_secret(username: str) -> str | None:
    try:
        return keyring.get_password(KEYRING_SERVICE, _keyring_key(username))
    except keyring.errors.KeyringError:
        return None


def is_enrolled(username: str) -> bool:
    """Whether 2FA should be ENFORCED for this user. Everything keys off this, so
    a user who has never enrolled is never challenged and never locked out."""
    return bool(get_secret(username))


def begin_enrolment(username: str) -> tuple[str, str]:
    """Generate a candidate secret and its provisioning URI. Deliberately does
    NOT store anything: the secret is only persisted by complete_enrolment()
    once the user has proved their app can produce a valid code from it. That
    ordering is what makes a mis-scanned code harmless."""
    secret = pyotp.random_base32()
    uri = pyotp.TOTP(secret).provisioning_uri(name=username, issuer_name=ISSUER)
    return secret, uri


def complete_enrolment(username: str, secret: str, code: str) -> list[str] | None:
    """Verify `code` against the candidate `secret`; on success store the secret
    and return fresh recovery codes. Returns None (storing nothing) on failure."""
    if not verify_code(code, secret=secret):
        return None
    keyring.set_password(KEYRING_SERVICE, _keyring_key(username), secret)
    return _generate_recovery_codes(username)


def disable(username: str) -> None:
    """Remove 2FA entirely. The documented get-back-in path — run it from a local
    terminal. No code required, on purpose: physical access to this machine is
    already game over for a local-only app, so demanding a factor the user has
    just lost would add no security and plenty of misery."""
    try:
        keyring.delete_password(KEYRING_SERVICE, _keyring_key(username))
    except keyring.errors.PasswordDeleteError:
        pass
    codes = _load_recovery_codes()
    codes.pop(username, None)
    _save_recovery_codes(codes)


# ------------------------------------------------------------------ verifying


def verify_code(code: str, *, username: str | None = None, secret: str | None = None) -> bool:
    """Check a 6-digit TOTP. Pass `secret` during enrolment (nothing stored yet)
    or `username` afterwards. Never raises on malformed input — a stray space or
    a pasted dash is a typo, not an exception."""
    if secret is None:
        if username is None:
            return False
        secret = get_secret(username)
        if not secret:
            return False
    cleaned = (code or "").strip().replace(" ", "").replace("-", "")
    if not cleaned.isdigit():
        return False
    try:
        return pyotp.TOTP(secret).verify(cleaned, valid_window=VALID_WINDOW)
    except Exception:
        return False


# ------------------------------------------------------------- recovery codes


def _hash_code(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def _load_recovery_codes() -> dict:
    if not RECOVERY_CODES_PATH.exists():
        return {}
    try:
        return json.loads(RECOVERY_CODES_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_recovery_codes(codes: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_text(RECOVERY_CODES_PATH, json.dumps(codes, indent=2))


def _generate_recovery_codes(username: str) -> list[str]:
    """Ten single-use codes. Only their hashes are stored, so this return value is
    the one and only time the plaintext exists — show it to the user then."""
    plain = ["-".join(secrets.token_hex(2) for _ in range(2)) for _ in range(NUM_RECOVERY_CODES)]
    stored = _load_recovery_codes()
    stored[username] = [_hash_code(c) for c in plain]
    _save_recovery_codes(stored)
    return plain


def remaining_recovery_codes(username: str) -> int:
    return len(_load_recovery_codes().get(username, []))


def consume_recovery_code(username: str, code: str) -> bool:
    """Spend a recovery code. Single-use: a match is deleted before returning, so
    the same code can never be replayed."""
    cleaned = (code or "").strip().lower()
    if not cleaned:
        return False
    stored = _load_recovery_codes()
    hashes = stored.get(username, [])
    target = _hash_code(cleaned)
    if target not in hashes:
        return False
    hashes.remove(target)
    stored[username] = hashes
    _save_recovery_codes(stored)
    return True
