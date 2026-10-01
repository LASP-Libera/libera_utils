# Test lanes

Every lane marker, path and command the corpus depends on, in one file, so that when the lane
layout changes only this file changes.

**Status: in transition, and this file describes both ends of it.** Today a pull request runs
the whole suite: `.github/workflows/_run-tests.yml` invokes `pytest` with no marker filter, and
the same workflow serves the PR build and the daily build. The `integration` marker exists and
nine modules set it, but nothing splits on it. The three-lane layout below arrives with
LIBSDC-703.

The durable source is the SDC testing approach recorded in Confluence ("Verification and Peace
of Mind Tests"), not that branch: the principle — a test's lane is decided by what it depends
on, and a pull request must not be gated on an outage at an external service — outlives
whatever the markers end up being called. If 703 lands with a different layout, this is the
only file under `standards/` to edit.

## What the lanes are

| Lane        | Directory           | Marker                    | Runs     | Admits                                                                                                                              | Today                        |
| ----------- | ------------------- | ------------------------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------- | ---------------------------- |
| unit        | `tests/unit`        | none                      | every PR | Nothing outside the checked-in test data: no network, no external process, fast. May call several levels deep rather than mock them | exists                       |
| integration | `tests/integration` | `pytest.mark.integration` | every PR | An external interface or infrastructure, such as a real subprocess. Everything it reads is checked in                               | exists, marked, not split on |
| e2e         | `tests/e2e`         | `pytest.mark.e2e`         | daily    | Needs a live external service, or is too slow to sit in front of a PR                                                               | arrives with LIBSDC-703      |

Markers are set per module with `pytestmark`, so a module belongs to exactly one lane.

### Measured

The counts `.github/scripts/check_measurements.py --lanes` re-collects. Wall clock is
machine-local and load-sensitive, so it is recorded here and never checked.

| Lane      | Tests | Wall clock             | Measured                                                                                                                                     | Command                      |
| --------- | ----- | ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------- |
| Unit lane | 1017  | **70.7 s**             | on `main` 2026-09-22 at 1001, plus the 16 tests of the two check scripts                                                                     | `pytest tests/unit/`         |
| PR lane   | 1111  | **446 s** (7 min 26 s) | same run, on 2026-09-25, at 1095 plus the 16 tests of the two check scripts; 102 s on the LIBSDC-703 base, which freezes the kernel fixtures | `pytest -m "not e2e" tests/` |

## The commands the corpus names

| Purpose                             | Command                                                            |
| ----------------------------------- | ------------------------------------------------------------------ |
| What a pull request and a build run | `pytest -m "not e2e" tests/`                                       |
| Unit only                           | `pytest -m "not integration and not e2e" tests/`                   |
| What the daily build adds           | `pytest -m e2e tests/`                                             |
| Coverage of changed files           | `pytest -m "not e2e" --cov=libera_utils --cov-report=term-missing` |

These are written for the layout the repository is moving to, and they are correct on both
sides of 703 — which is why the corpus can name them now. An unregistered marker in a `-m`
expression matches nothing, so `-m "not e2e"` today selects the whole suite, which is exactly
what a pull request currently runs. After 703 the same command selects everything but the
daily lane. Nothing has to change here when the marker lands.

`pytest -m e2e tests/` is the exception: it selects nothing until 703 creates the lane.

## The network guard

**Arrives with LIBSDC-703.** An autouse fixture, `block_outbound_network` in
`tests/conftest.py`, fails any test outside the daily lane that opens a socket to a remote
host. Mocking layers such as `responses` and `moto` sit above the socket and are unaffected.

It exists because vigilance was not enough: 28 tests were found silently downloading kernels
from NAIF, unnoticed for months. That fact is durable even if the lane names are not, and it
is the strongest argument for the split — a pull request that fails because NAIF is down has
told the author nothing about their change.

## Who reads this file

`standards/review-contract.md` (the coverage command and the guard), the plugin's
`test-suite-review` skill (the lanes and commands), the implementation reviewer at the end of a
build (through `AGENTS.md`), and `.github/scripts/check_measurements.py --lanes` (the measured lane counts). None of them hard-codes a marker or a path; they name this file.

**Test scrutiny judges the shape of a test, not its location.** Whether a new test belongs beside its
siblings rather than in a new module, whether a changed line has a test, and whether a new
`raise` is asserted are all independent of how the lanes are arranged. Only the commands above
depend on the layout, and only the daily one depends on 703.
