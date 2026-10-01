# ADR-017: JWT scopes and row-level tenant isolation, enforced twice

- **Status:** accepted
- **Date:** 2026-10-01

## Context

Until now the service trusted anyone who could reach it (loopback only) and
held one person's evidence. Serving several people means two things must be
true at once: a caller can do only what their credential allows, and no
query can ever return another tenant's evidence - including a query someone
writes next year and forgets to filter.

## Decision

### Credentials and scopes

- **JWT bearer tokens** (`AUTH=jwt`), HS256 with `JWT_SECRET` (at least 32
  bytes; the app refuses to start otherwise). Required claims: `sub`,
  `tenant`, `scope`, `iss`, `aud`, `exp`, `iat`; optional `budget` (tokens per
  UTC day). The algorithm is pinned, so `alg: none` and key-confusion tokens
  fail. `python -m grounded.api token` issues tokens for local use; in
  production an identity provider would sign them.
- **API keys** (ADR-016) remain a machine credential: one tenant
  (`name=digest:budget:tenant`), `generate` only.
- **Scopes**: `generate` (POST /v1/drafts), `read_evidence` (GET the evidence
  API), `admin` (write, verify, reject). 401 for a missing or bad credential,
  403 for a valid one without the scope.
- `AUTH=off` keeps local use frictionless: an anonymous caller is the
  `default` tenant with `read_evidence` and `admin` - but never `generate`,
  because calls that cost money always need a credential. The admin and UI
  pages are served only in this mode; with `AUTH=jwt` the service is API-only.

### Tenant isolation, in two independent layers

1. **ORM**: every evidence-store table has `tenant_id`. A request's session
   carries the caller's tenant, and a session event adds
   `tenant_id = <tenant>` to every ORM select - queries, `Session.get`,
   relationship loads - and stamps new rows at flush. A write of another
   tenant's row raises. Nobody writes the filter, so nobody can forget it.
2. **Postgres row-level security** (migration 0003): a policy on each table
   compares `tenant_id` with the transaction's `app.tenant_id`, which the
   session sets when it begins; `WITH CHECK` refuses writes for another
   tenant. An integration test runs raw SQL as a non-superuser role and gets
   only that tenant's rows - and none with no tenant set.

Record IDs stay globally unique. A write whose ID another tenant already
holds fails with "id already in use", checked before the insert.

## Consequences

- **RLS does not apply to superusers or table owners.** The compose stack's
  `app` role is both, so in development only the ORM layer is active. A
  production deployment should migrate as the owner and serve as a separate,
  non-owner role; the integration test shows the policies then hold.
- Global IDs leak one bit - that an ID exists somewhere. Composite
  `(tenant_id, id)` keys would remove it at the cost of rewriting every
  foreign key; recorded as a known limitation.
- The rate limiter, budgets and breaker are still per process (ADR-016).
