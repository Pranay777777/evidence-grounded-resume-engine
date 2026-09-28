# ADR-003: One write path, a local admin, and no deletes

- **Status:** accepted
- **Date:** 2026-09-28

## Context

Evidence can now arrive three ways: a YAML file, a JSON API, and an admin
form. ADR-002's central rule — an edited fact gets a new revision and loses
its verification — is only a rule if no route around it exists.

## Decision

**One write path.** The loader, the API's `PUT /evidence/{id}` and the admin
form all call `store.upsert_evidence`. There is no second implementation to
drift, and the tests prove the rule through each surface: an API edit and a
form edit both come back as revision 2, unverified.

**Requests reuse the file schema.** The API validates bodies with the same
Pydantic model as the YAML loader, so a record is valid or invalid in the
same way however it arrives.

**No deletes.** There is no `DELETE` route. A record is retired with
`reject`, which keeps it on the record — so a disproved claim cannot vanish
and quietly return.

**A local admin, bound to loopback, with CSRF protection.** Authentication
arrives in step 60. Until then the server binds to `127.0.0.1` by default,
and the container binds `0.0.0.0` behind its port mapping. Loopback alone is
not a defence: a web page you visit can make your browser POST to
`http://127.0.0.1:8000`. So every admin form carries a double-submit token
(a cookie another site can make the browser send but cannot read) and the
server refuses any POST whose `Origin` is not itself.

**Server-rendered forms.** Jinja2 with autoescaping and plain HTML forms —
no JavaScript build, nothing to keep patched. Evidence text is user input;
a test proves a `<script>` statement renders inert.

## Consequences

**Gained:** ADR-002 holds on every surface by construction, not by care; the
admin is safe against the browser-borne attacks a local tool invites.

**Given up:** the JSON API has no CSRF protection, because it accepts only
JSON bodies, which a cross-site form cannot send without a CORS preflight
the server does not grant — and it has no authentication at all until step
60, so it must stay on loopback. The admin is deliberately plain.
