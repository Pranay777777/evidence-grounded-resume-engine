"""Who is calling, for which tenant, allowed to do what (ADR-017).

Two credentials produce the same `Principal`:

- **JWT bearer tokens** (HS256, `AUTH=jwt`): `sub`, `tenant`, `scope`
  (space-separated), optional `budget` (tokens per UTC day), plus `iss`,
  `aud` and `exp`, all required. `python -m grounded.api token` issues them.
- **API keys** (step 59, `X-API-Key`): a machine credential for one tenant
  with the `generate` scope only.

Scopes: `generate` (POST /v1/drafts), `read_evidence` (GET the evidence
API), `admin` (write, verify or reject evidence). With `AUTH=off` - local,
loopback use - an anonymous caller is the `default` tenant with
`read_evidence` and `admin`, but never `generate`: calls that cost money
always need a credential.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

import jwt
from fastapi import Depends, Header, HTTPException, Request

from grounded.api.guards import authenticate, parse_keys
from grounded.config import Settings, get_settings
from grounded.evidence.tenancy import DEFAULT_TENANT, TENANT_PATTERN

SCOPES = frozenset({"generate", "read_evidence", "admin"})
ALGORITHM = "HS256"
MIN_SECRET_BYTES = 32


@dataclass(frozen=True)
class Principal:
    subject: str
    tenant: str
    scopes: frozenset[str]
    daily_tokens: int
    anonymous: bool = False

    @property
    def name(self) -> str:
        """The budget and rate-limit identity: unique across tenants."""
        return f"{self.tenant}/{self.subject}"


LOCAL = Principal("local", DEFAULT_TENANT, frozenset({"read_evidence", "admin"}), 0, True)


class AuthError(ValueError):
    pass


def check_secret(settings: Settings) -> bytes:
    secret = settings.jwt_secret.get_secret_value().encode()
    if len(secret) < MIN_SECRET_BYTES:
        raise AuthError(
            f"JWT_SECRET must be at least {MIN_SECRET_BYTES} bytes - generate one with "
            '`python -c "import secrets; print(secrets.token_urlsafe(48))"`'
        )
    return secret


def issue(
    settings: Settings,
    subject: str,
    tenant: str,
    scopes: set[str],
    ttl: timedelta = timedelta(hours=1),
    budget: int | None = None,
    now: datetime | None = None,
) -> str:
    unknown = scopes - SCOPES
    if unknown:
        raise AuthError(f"unknown scope(s): {', '.join(sorted(unknown))}")
    if not re.fullmatch(TENANT_PATTERN, tenant):
        raise AuthError(f"tenant '{tenant}' must match {TENANT_PATTERN}")
    moment = now or datetime.now(UTC)
    claims: dict[str, Any] = {
        "sub": subject,
        "tenant": tenant,
        "scope": " ".join(sorted(scopes)),
        "iss": settings.jwt_issuer,
        "aud": settings.jwt_audience,
        "iat": moment,
        "exp": moment + ttl,
    }
    if budget is not None:
        claims["budget"] = budget
    return jwt.encode(claims, check_secret(settings), algorithm=ALGORITHM)


def verify(token: str, settings: Settings) -> Principal:
    try:
        claims = jwt.decode(
            token,
            check_secret(settings),
            algorithms=[ALGORITHM],
            audience=settings.jwt_audience,
            issuer=settings.jwt_issuer,
            options={"require": ["exp", "iat", "sub", "iss", "aud"]},
        )
    except jwt.PyJWTError as exc:
        raise AuthError(f"invalid token: {exc}") from exc
    tenant = claims.get("tenant")
    if not isinstance(tenant, str) or not re.fullmatch(TENANT_PATTERN, tenant):
        raise AuthError("invalid token: missing or malformed tenant")
    scopes = frozenset(str(claims.get("scope", "")).split()) & SCOPES
    budget = claims.get("budget", settings.default_daily_tokens)
    return Principal(str(claims["sub"]), tenant, scopes, int(budget))


def identify(request: Request, settings: Settings) -> Principal | None:
    """The caller's credential, if any; raises AuthError on a bad one."""
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        if settings.auth != "jwt":
            raise AuthError("bearer tokens need AUTH=jwt")
        return verify(header[7:].strip(), settings)
    key = authenticate(request.headers.get("x-api-key"), parse_keys(settings.api_keys))
    if key is not None:
        return Principal(key.name, key.tenant, frozenset({"generate"}), key.daily_tokens)
    if request.headers.get("x-api-key"):
        raise AuthError("unknown X-API-Key")
    return None


def current_principal(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
    x_api_key: Annotated[str | None, Header()] = None,
) -> Principal:
    settings = get_settings()
    try:
        principal = identify(request, settings)
    except AuthError as exc:
        raise HTTPException(401, str(exc), headers={"WWW-Authenticate": "Bearer"}) from exc
    if principal is not None:
        return principal
    if settings.auth == "off":
        return LOCAL
    raise HTTPException(401, "credentials required", headers={"WWW-Authenticate": "Bearer"})


PrincipalDep = Annotated[Principal, Depends(current_principal)]


def require(scope: str) -> Callable[[Principal], Principal]:
    assert scope in SCOPES, scope

    def dependency(principal: PrincipalDep) -> Principal:
        if scope not in principal.scopes:
            if principal.anonymous:
                raise HTTPException(
                    401, f"'{scope}' needs a credential", headers={"WWW-Authenticate": "Bearer"}
                )
            raise HTTPException(403, f"token lacks the '{scope}' scope")
        return principal

    return dependency
