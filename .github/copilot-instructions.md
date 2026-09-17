# Libera Utils — Copilot Instructions

Libera Utils is a Python utility library for the Libera Science Data Center (LASP, University
of Colorado). It supports L2 algorithm developers working on the Libera satellite mission with
shared tooling for NetCDF data I/O, telemetry packet parsing, SPICE kernel generation, Libera
file naming conventions, and AWS pipeline integration.

Detailed coding rules, package layout, testing conventions, key patterns, and AI agent
restrictions are defined in `.github/instructions/libera-utils.instructions.md`. Copilot
applies that file automatically to all files in this repository.

The review standard — the rules a reviewer checks, the contract it follows, and the
do-not-flag list — is in `standards/`, starting at `standards/README.md`. It is tool-neutral
on purpose: Copilot, Claude and a person reading the repository for the first time all read
the same `standards/rules.md`, so a rule cannot drift between tools.
