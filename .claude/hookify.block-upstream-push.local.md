---
name: block-upstream-push
enabled: true
event: bash
pattern: git\s+push\s+.*upstream|git\s+push\s+.*openai
action: block
---

**BLOCKED: Push to openai/parameter-golf upstream remote.**

You must NEVER push to the upstream (openai) remote. Only push to origin (RoyiRa's private fork) on NEW branches.
