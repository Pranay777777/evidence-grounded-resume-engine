"""Render a built résumé as one self-contained, print-ready HTML page (A4)."""

from __future__ import annotations

from datetime import date
from html import escape

from grounded.resume.build import Block, Bullet, Resume

CSS = """
@page { size: A4; margin: 14mm 15mm; }
* { box-sizing: border-box; }
body { font: 9.6pt/1.32 "Segoe UI", "Helvetica Neue", Arial, sans-serif; color: #1d232a;
       max-width: 190mm; margin: 0 auto; padding: 8mm 0; }
h1 { font-size: 20pt; margin: 0; letter-spacing: .2px; }
.headline { font-size: 11pt; color: #3c4a57; margin: 1mm 0 2mm; }
.contact { font-size: 9pt; color: #3c4a57; }
.contact a { color: inherit; }
h2 { font-size: 10.5pt; text-transform: uppercase; letter-spacing: 1px; color: #0f5c8c;
     border-bottom: 1px solid #c9d5df; margin: 5mm 0 2mm; padding-bottom: .6mm; }
.block { margin: 0 0 2mm; break-inside: avoid; }
.block-head { display: flex; justify-content: space-between; gap: 4mm; }
.block-title { font-weight: 600; }
.block-title a { color: inherit; text-decoration: none; }
.block-sub { color: #56636f; font-size: 9pt; text-align: right; }
.summary { color: #56636f; font-size: 9pt; margin: .3mm 0 .6mm; }
ul { margin: .6mm 0 0; padding-left: 4.5mm; }
li { margin: .2mm 0; }
.inline { margin: .6mm 0 0; }
.inline span + span::before { content: " · "; color: #7a8792; }
.cite { font-size: 7pt; color: #7a8792; text-decoration: none; white-space: nowrap; }
.skills div { margin: .4mm 0; }
.skills b { font-weight: 600; }
.footer { margin-top: 5mm; font-size: 7.5pt; color: #7a8792; border-top: 1px solid #e3e9ee;
          padding-top: 1.5mm; }
@media print { body { padding: 0; } a { color: inherit; } }
"""


def _bullet(b: Bullet) -> str:
    mark = "<sup>†</sup>" if b.self_attested else ""
    ref = escape(b.evidence_id)
    cite = (
        f' <a class="cite" href="{escape(b.link)}">[{ref}]</a>'
        if b.link
        else f' <span class="cite">[{ref}]</span>'
    )
    return f"<li>{escape(b.text)}{mark}{cite}</li>"


def _inline(bullets: list[Bullet]) -> str:
    """Short records (degrees, certificates) on one line each section, still cited."""
    items = "".join(f"<span>{_bullet(b)[4:-5]}</span>" for b in bullets)  # strip <li></li>
    return f'<div class="inline">{items}</div>'


def _block(blk: Block, sub_right: bool) -> str:
    title = escape(blk.title)
    if blk.link:
        title = f'<a href="{escape(blk.link)}">{title}</a>'
    right = (
        f'<span class="block-sub">{escape(blk.subtitle)}</span>'
        if (sub_right and blk.subtitle)
        else ""
    )
    summary = (
        f'<div class="summary">{escape(blk.subtitle)}</div>'
        if (blk.subtitle and not sub_right)
        else ""
    )
    items = "".join(_bullet(b) for b in blk.bullets)
    return (
        f'<div class="block"><div class="block-head"><span class="block-title">{title}</span>'
        f"{right}</div>{summary}<ul>{items}</ul></div>"
    )


def render(resume: Resume, today: date | None = None) -> str:
    p = resume.profile
    contact = " · ".join(
        f'<a href="{escape(str(c.url))}">{escape(c.label)}</a>' if c.url else escape(c.label)
        for c in p.contact
    )
    parts = [
        f"<h1>{escape(p.name)}</h1>",
        f'<div class="headline">{escape(p.headline)}</div>',
        f'<div class="contact">{contact}</div>',
    ]
    if resume.experience:
        parts.append("<h2>Experience</h2>")
        parts += [_block(b, sub_right=True) for b in resume.experience]
    if resume.projects:
        parts.append("<h2>Projects</h2>")
        parts += [_block(b, sub_right=False) for b in resume.projects]
    if p.skills:
        rows = "".join(
            f"<div><b>{escape(g.group)}:</b> {escape(', '.join(g.items))}</div>" for g in p.skills
        )
        parts.append(f'<h2>Skills</h2><div class="skills">{rows}</div>')
    if resume.education:
        parts.append("<h2>Education</h2>" + _inline(resume.education))
    if resume.certifications:
        parts.append("<h2>Certifications</h2>" + _inline(resume.certifications))
    stamp = (today or date.today()).isoformat()
    has_self = any(b.self_attested for b in resume.cited)
    parts.append(
        f'<div class="footer">Generated {stamp} from {len(resume.cited)} verified evidence '
        "records; "
        "each bullet is a record's exact statement and links to its proof"
        + ("; † self-attested" if has_self else "")
        + ". github.com/Pranay777777/evidence-grounded-resume-engine</div>"
    )
    return (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{escape(p.name)} - résumé</title><style>{CSS}</style></head>"
        f"<body>{''.join(parts)}</body></html>"
    )
