"""CSRF protection for the admin UI's forms.

The admin has no authentication yet (that is step 60) and binds to
localhost, which is exactly the setup a malicious web page can exploit: a
page you visit can submit a form to http://127.0.0.1:8000 and your browser
will send it. Two checks, either of which stops that:

1. **Double-submit token.** A random token is set in a cookie and repeated
   in every form. Another site can make the browser send the cookie, but
   cannot read it, so it cannot put the matching value in the form.
2. **Origin check.** Browsers attach `Origin` to cross-site POSTs; a POST
   whose Origin is not this server is refused.

The public demo (ADR-020) is shown inside a cross-site iframe (huggingface.co
embeds the Space's own domain), where a `SameSite=Strict` cookie is never
sent. There the cookie is `SameSite=None; Secure; Partitioned`, so it is kept
only for that embedding, and a browser that blocks even partitioned cookies
falls back to the Origin check alone - enough for a read-only demo with no
session to ride.
"""

from __future__ import annotations

import secrets
from urllib.parse import urlsplit

from fastapi import HTTPException, Request, Response

COOKIE = "grounded_csrf"
FIELD = "csrf_token"


def issue(request: Request) -> tuple[str, bool]:
    """The request's token, and whether it is new (so needs a cookie set)."""
    existing = request.cookies.get(COOKIE)
    if existing:
        return existing, False
    return secrets.token_urlsafe(32), True


def attach(response: Response, token: str, *, embedded: bool = False) -> None:
    if embedded:
        # Written by hand: Starlette's `partitioned=` needs Python 3.14's http.cookies.
        # token_urlsafe() output needs no quoting.
        response.headers.append(
            "set-cookie", f"{COOKIE}={token}; HttpOnly; Path=/; SameSite=None; Secure; Partitioned"
        )
    else:
        response.set_cookie(COOKIE, token, httponly=True, samesite="strict")


def check(request: Request, submitted: str | None, *, embedded: bool = False) -> None:
    origin = request.headers.get("origin")
    if origin and urlsplit(origin).netloc != request.url.netloc:
        raise HTTPException(status_code=403, detail="cross-origin form submission refused")
    cookie = request.cookies.get(COOKIE)
    if embedded and origin and not cookie:
        return  # same-origin proven above; this browser blocks cookies in the iframe
    if not cookie or not submitted or not secrets.compare_digest(cookie, submitted):
        raise HTTPException(status_code=403, detail="missing or invalid CSRF token")
