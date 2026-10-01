"""Row-level tenant isolation (ADR-017).

Every evidence-store row belongs to exactly one tenant. Isolation is
enforced in two independent layers:

1. **The ORM.** A session carries its tenant in `session.info["tenant"]`.
   Every ORM query run through such a session - selects, `Session.get`,
   relationship loads - gets `tenant_id = <tenant>` added automatically, and
   new rows are stamped with the session's tenant at flush. Code cannot
   forget the filter because it never writes it.
2. **Postgres row-level security** (migration 0003). Policies on every
   tenant table compare `tenant_id` with the transaction's `app.tenant_id`,
   which the session sets when it begins. A query that somehow escaped the
   ORM filter still sees only its own tenant's rows - when the connection
   is not a superuser or table owner, who bypass RLS (see ADR-017).

A session without a tenant (the CLI's bulk tools, tests) is unscoped.
`execution_options(all_tenants=True)` lifts the filter for one statement,
for the few checks that must see across tenants (an ID already taken).
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import String, event, text
from sqlalchemy.orm import (
    Mapped,
    ORMExecuteState,
    Session,
    SessionTransaction,
    mapped_column,
    with_loader_criteria,
)

DEFAULT_TENANT = "default"
TENANT_PATTERN = r"^[a-z0-9][a-z0-9_-]{0,39}$"


class TenantScoped:
    tenant_id: Mapped[str] = mapped_column(
        String(40),
        default=DEFAULT_TENANT,
        server_default=DEFAULT_TENANT,
        nullable=False,
        index=True,
    )


class CrossTenantWriteError(RuntimeError):
    pass


def scope(session: Session, tenant: str) -> Session:
    session.info["tenant"] = tenant
    return session


def tenant_of(session: Session) -> str | None:
    value = session.info.get("tenant")
    return str(value) if value is not None else None


@event.listens_for(Session, "do_orm_execute")
def _filter(state: ORMExecuteState) -> None:
    tenant = tenant_of(state.session)
    if tenant is None or state.execution_options.get("all_tenants", False):
        return
    if state.is_select and not state.is_column_load and not state.is_relationship_load:
        state.statement = state.statement.options(
            with_loader_criteria(
                TenantScoped,
                lambda cls: cls.tenant_id == tenant,
                include_aliases=True,
                track_closure_variables=False,
            )
        )


@event.listens_for(Session, "before_flush")
def _stamp(session: Session, flush_context: Any, instances: Any) -> None:
    tenant = tenant_of(session)
    if tenant is None:
        return
    for obj in list(session.new) + list(session.dirty):
        if isinstance(obj, TenantScoped):
            if obj.tenant_id is None or (obj in session.new and obj.tenant_id == DEFAULT_TENANT):
                obj.tenant_id = tenant
            if obj.tenant_id != tenant:
                raise CrossTenantWriteError(
                    f"session for tenant '{tenant}' tried to write a '{obj.tenant_id}' row"
                )


@event.listens_for(Session, "after_begin")
def _set_postgres_tenant(
    session: Session, transaction: SessionTransaction, connection: Any
) -> None:
    tenant = tenant_of(session)
    if tenant is not None and connection.dialect.name == "postgresql":
        connection.execute(
            text("SELECT set_config('app.tenant_id', :tenant, true)"), {"tenant": tenant}
        )
