---
name: warn-any-git-push
enabled: true
event: bash
pattern: git\s+push
action: warn
---

**Git push detected.** Verify:
- Target is `origin` (RoyiRa's fork), NOT `upstream` (openai)
- Branch is NOT `submission/2026-03-25-hedge-mixer-crown-q` (active PR #700)
- Only push to NEW branches on origin
