# Evidence

The only source of fact the generator may cite ([ADR-001](../docs/adr/0001-grounding-constraint.md),
[ADR-002](../docs/adr/0002-evidence-model.md)). Records are written here as YAML,
validated, and loaded into the store:

```bash
python -m grounded.evidence check evidence/private/evidence.yaml   # validate, no database
python -m grounded.evidence load  evidence/private/evidence.yaml   # upsert into the store
python -m grounded.evidence list  --status unverified
python -m grounded.evidence verify <id> --method artifact --by "GitHub Actions"
```

## Where files go

| Folder | Committed? | For |
|---|---|---|
| `private/` | **No** — gitignored | Your real evidence, especially anything about an employer's internal work |
| `drafts/` | Yes | Records drawn from artifacts that are already public |

Treat employer work as confidential unless you know it is not. A fact can be
true and still not be yours to publish.

## Writing a record

**One checkable fact per record.** If the statement needs "and", it is two
records. The verifier checks each generated bullet against the records it
cites, and a compound record makes that check meaningless.

```yaml
evidence:
  - id: lakehouse-test-count          # stable forever — citations use it
    kind: metric                      # achievement | metric | responsibility | skill | education | certification
    statement: The lakehouse test suite has 499 tests.
    project: metadata-driven-lakehouse
    month: 2026-09                    # months only
    metric: {value: 499, unit: tests} # metric records only, and required for them
    skills: [pytest]                  # retrieval tags; not part of the fact
    artifact_url: https://github.com/...
    verification: {method: artifact, by: GitHub Actions}   # optional
```

- **Numbers live only in `metric` records.** The generator may quote a
  number from nowhere else.
- **State what happened, at the strength it happened.** "Contributed to"
  and "led" are different facts; write the true one.
- **Editing a statement voids its verification.** The revision increments
  and the record must be verified again before it can be cited.
- **Nothing is deleted by a load.** A record removed from the file stays in
  the store; use `reject` to retire one on the record.

## Verification methods

| Method | Means | Needs |
|---|---|---|
| `artifact` | A public link proves it — a release, a CI run, a certificate | `artifact_url` |
| `third_party` | A named person or organisation can confirm it | who, in `by` |
| `self_attested` | Your own word, labelled as such wherever it is cited | who, in `by` |

Only `verified` records are citable. Unverified ones are stored, visible, and
ignored by the generator.
