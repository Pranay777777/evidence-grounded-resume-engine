"""Closed vocabularies for evidence, enforced in the database as CHECKs."""

from __future__ import annotations

from enum import StrEnum


class EvidenceKind(StrEnum):
    """What sort of fact a record states. Drives how it may be phrased."""

    ACHIEVEMENT = "achievement"
    """Something done, with an outcome: 'Shipped v1.0.0 with CI green on
    Ubuntu and Windows.'"""

    METRIC = "metric"
    """A number with a unit. The only kind allowed to carry `metric_value`,
    and the only kind a generated bullet may quote a number from."""

    RESPONSIBILITY = "responsibility"
    """Ongoing scope in a role, without a claimed outcome."""

    SKILL = "skill"
    """Hands-on use of a technology, tied to where it was used."""

    EDUCATION = "education"
    CERTIFICATION = "certification"


class VerificationStatus(StrEnum):
    UNVERIFIED = "unverified"
    """New, or changed since it was last verified. Never cited."""

    VERIFIED = "verified"
    """Checked by the method recorded alongside it. The only status the
    generator may cite."""

    REJECTED = "rejected"
    """Checked and found wrong. Kept, so the rejection is auditable and the
    same claim is not quietly re-added."""


class VerificationMethod(StrEnum):
    """How a record was verified — shown next to every citation."""

    ARTIFACT = "artifact"
    """A public link proves it: a commit, a release, a CI run, a certificate
    URL. Requires `artifact_url`."""

    THIRD_PARTY = "third_party"
    """A named person or organisation can confirm it: a manager, a client,
    an issuer. `verified_by` names them."""

    SELF_ATTESTED = "self_attested"
    """The candidate's own word, labelled as such. Honest, weakest, and
    visible to anyone reading the output."""
