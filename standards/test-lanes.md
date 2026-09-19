# Test lanes

Every lane marker, path and command the corpus depends on, in one file, so that when the
lane layout changes only this file changes.

**Status: provisional.** The three-lane layout below arrives with LIBSDC-703, which is a
draft pull request and may still change. The durable source is the SDC testing approach
recorded in Confluence ("Verification and Peace of Mind Tests"), not that branch: the
principle — a test's lane is decided by what it depends on, and a pull request must not be
gated on an outage at an external service — outlives whatever the markers end up being
called. If 703 lands with a different layout, this is the only file under `standards/` to edit.
Outside `standards/` the same lane table appears in `doc/source/developer-docs/testing.md`
and in `.github/instructions/libera-utils.instructions.md`, which are the repository's own
documentation and move with 703 itself.

## What the lanes are

| Lane        | Directory           | Marker                    | Runs     | Admits                                                                      |
| ----------- | ------------------- | ------------------------- | -------- | --------------------------------------------------------------------------- |
| unit        | `tests/unit`        | none                      | every PR | One function or class. No network, no external process                      |
| integration | `tests/integration` | `pytest.mark.integration` | every PR | Several components, or a real subprocess. Everything it reads is checked in |
| e2e         | `tests/e2e`         | `pytest.mark.e2e`         | daily    | Needs a live external service, or is too slow to sit in front of a PR       |

Markers are set per module with `pytestmark`, so a module belongs to exactly one lane.

## The commands the corpus names

| Purpose                                | Command                                                            |
| -------------------------------------- | ------------------------------------------------------------------ |
| What a pull request runs, and gate 3   | `pytest -m "not e2e" tests/`                                       |
| Unit only                              | `pytest -m "not integration and not e2e" tests/`                   |
| What the daily build adds              | `pytest -m e2e tests/`                                             |
| Coverage of changed files, for gate 3b | `pytest -m "not e2e" --cov=libera_utils --cov-report=term-missing` |

## The network guard

An autouse fixture, `block_outbound_network` in `tests/conftest.py`, fails any test outside
the daily lane that opens a socket to a remote host. Mocking layers such as `responses` and
`moto` sit above the socket and are unaffected.

It exists because vigilance was not enough: 28 tests were found silently downloading kernels
from NAIF, unnoticed for months. That fact is durable even if the lane names are not.

## Who reads this file

`standards/review-contract.md` (the coverage command and the guard), the `loop-self-review`
gate ladder (gate 3 and gate 3b), and `standards/README.md` (the measured wall clock). None
of them hard-codes a marker or a path; they name this file.

**Gate 3b judges the shape of a test, not its location.** Whether a new test belongs beside
its siblings rather than in a new module, whether a changed line has a test, and whether a
new `raise` is asserted are all independent of how the lanes are arranged. Only the commands
above depend on the layout.
