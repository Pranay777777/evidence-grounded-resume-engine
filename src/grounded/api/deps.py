"""Request-scoped database sessions, scoped to the caller's tenant (ADR-017)."""

from __future__ import annotations

from collections.abc import Iterator

from fastapi import Request
from sqlalchemy.orm import Session

from grounded.api.auth import PrincipalDep
from grounded.evidence.tenancy import scope


def get_session(request: Request, principal: PrincipalDep) -> Iterator[Session]:
    with Session(request.app.state.engine) as session:
        yield scope(session, principal.tenant)
