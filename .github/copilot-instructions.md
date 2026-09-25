# Libera Utils — Copilot Instructions

Libera Utils is a Python utility library for the Libera Science Data Center (LASP, University
of Colorado). It supports L2 algorithm developers working on the Libera satellite mission with
shared tooling for NetCDF data I/O, telemetry packet parsing, SPICE kernel generation, Libera
file naming conventions, and AWS pipeline integration.

Detailed coding rules, package layout, testing conventions, key patterns, and AI agent
restrictions are defined in `.github/instructions/libera-utils.instructions.md`. Copilot
applies that file automatically to all files in this repository. `AGENTS.md` is the
tool-neutral entry point above both: it maps this file, the instruction file and `standards/`
to each other and describes how a change moves from ticket to merge.

The review standard is tool-neutral on purpose: Copilot, Claude and a person reading the
repository for the first time all read the same rules, so a rule cannot drift between tools.
Each rule's wording is in the Review Rules section of
`.github/instructions/libera-utils.instructions.md`, which this file already applies. Its tier,
status, evidence and do-not-flag sentence are in `standards/review-rules.md`, and the contract
a reviewer follows is `standards/review-contract.md`, starting at `standards/README.md`.
