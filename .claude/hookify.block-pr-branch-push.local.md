---
name: block-pr-branch-push
enabled: true
event: bash
pattern: git\s+push\s+.*submission/2026-03-25-hedge-mixer-crown-q
action: block
---

**BLOCKED: Push to the active PR branch.**

Branch `submission/2026-03-25-hedge-mixer-crown-q` is the open PR (#700) against openai/parameter-golf. Pushing here updates the PR and risks disqualification.

Create a NEW branch for V28+ work instead.
