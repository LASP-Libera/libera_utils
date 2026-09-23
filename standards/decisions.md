# Decisions · libera_utils

Decisions that constrain this repository only. Cross-repository decisions are in the shared
corpus (`SHARED.md`), numbered in the same `D-` space. Cite a local one as
`libera_utils/D-NNN` in prose and as `D-NNN@libera_utils` in a finding key, whose citation
must contain no `/` (`review-contract.md`, Citation).

Appended by the ratchet, from convention questions people answered in a pull request thread
or a ticket. One paragraph each, with where and when.

---

### libera_utils/D-001 · The ObsID registry is data, validated at import

_2026-09-02, PR #41_

`data/obsid_registry.csv` and `data/trim_family_inputs.csv` hold the registry, not a literal
inside `obsids.py`. A table in code cannot be validated as data and a table in a comment
cannot be used at all. The registry docstring names the upstream ICIE ObsID page — by name,
with no URL (R-014) — so the next reader knows there is a document to reconcile against.

### libera_utils/D-002 · A ProductID that reaches a filename is globally unique

_2026-09-02, PR #41_

Product identifier strings land in filenames, so two products that share a string collide on
disk and the second write clobbers the first. The registry key carries the instrument and
the ProductID does not, so any ObsID asserted on both the radiometer and the camera needs
its own pair of product identifiers. This is what the uniqueness invariant test guards.

### libera_utils/D-003 · Ground CCSDS filename times are bin starts, not data spans

_2026-09-09, PR #48_

The time fields in a ground capture's filename are the bin the file was cut from. They say
nothing about the packets inside it, and under DITL they disagree with the data by years.
Searchable data times come from File Metadata at ingest, never from the archive path. Any
code that reads a time from a filename must say which of the two it means (T-010, T-011,
T-012).

### libera_utils/D-004 · The NOAA-20 SPICE configuration is test data, not shipped config

_2026-09, LIBSDC-703, changelog 5.12.0_

It moved to `tests/test_data/noaa20_spice/`. Nothing in the pipeline could select it, and its
frame kernel declares none of the measured misalignments, so it cannot produce
flight-representative geometry. It is kept as test data because it is the only kernel
generation driven by real decoded spacecraft telemetry and the only geolocation validated
against a third-party product (CERES). A downstream package that pointed `LIBERA_KERNEL_DIR`
at it must vendor the configuration itself.

### libera_utils/D-005 · Writing a product overwrites an existing object at the same key

_2026-09-19 · **provisional** · source: PR #66 review, adjudicated by mmaclay_

Reprocessing legitimately rewrites a granule at the same key, and a pipeline that halted
because the object in the bucket was newer than the file it just produced would fail on its
own success. So the write proceeds.

This is what the code did before PR #66, not something that PR decided. `CloudPath.open("w+b")`
refreshed the cache from the existing object, recorded its mtime, and on close bumped the
freshly written cache file to `original_mtime + 1` if it came out older — a step cloudpathlib
takes so that a write through `open` always counts as newer than what it is replacing. The
upload that followed then passed its `local newer than cloud` test by construction.
`OverwriteNewerCloudError` could not fire on a sequential write.

`force_overwrite_to_cloud=True` on the staged upload is therefore the faithful translation, not
a loosening. A staged temporary file carries no relationship to the object's mtime, so leaving
the argument at its default would compare a local clock against S3's and raise where the old
path could not — a behavior change wearing the default's clothes.

One check is genuinely gone: an object replaced by another writer between the open and the
close used to raise. Nothing checks for that now. The window was the duration of one write, and
nothing in the pipeline writes the same key from two processes, which is why this is recorded
rather than treated as a regression.

What would reverse it: a product whose key is not unique per reprocessing run, where a silent
overwrite would destroy a granule someone still needs.

### libera_utils/D-006 · `libera_utils` is published, so internal documents are cited by name only

_2026-08-05, PR libera_utils#41 review thread · shared D-004 until 2026-09-23_

The package ships to PyPI. No internal Confluence or Jira URL, and no internal document
content, goes into shipped source. Cite the document by name, say what it decides, and let
the reader find it. This is also why links are not used as the pointer: Confluence links
rot. The same posture applies to anything that would land under a public repository's
`standards/`.

### libera_utils/D-007 · Narrow the annotation rather than reject at runtime

_2026-09 · **established** · source: libera_utils pr-0028, pr-0012, pr-0060 · shared D-011 until 2026-09-23_

When a function accepts a type it cannot actually handle, the fix is the signature, not a
guard. The reviewer on pr-0028 was offered a runtime rejection of cloud paths and chose the
other way: "just change the typehint on `manual_ingest_data_products` to only accept a local
`Path` or `str` since that is what is actually required." Widening a signature and then
policing it inside moves the failure later and states the contract in two places that drift.
The corollary from pr-0012 is that the narrow type is usually a domain type the repository
already owns — `LiberaDataProductFilename` rather than `str`, `PathType` where an `S3Path`
can genuinely reach.

### libera_utils/D-008 · An error message says what went wrong and what to do next

_2026-09 · **established** · source: libera_utils pr-0028, pr-0060, pr-0015 · shared D-012 until 2026-09-23; its earlier wording is archived in the shared corpus_

The audience for a failure in this package is an algorithm developer outside the SDC, which
is why a message naming only internals — an IAM role ARN, a boto exception — fails: it tells
them nothing they can act on. That is the reason for the rule, not the rule itself. The rule
is that the message states the condition and the next action.

Naming the reader is sometimes the clearest way to do that, and pr-0028 dictated the
replacement text in full: "Check that you are using the profile that logs in as the L2
Developer base role. If this error persists, contact the SDC team." But it is not required,
and the other two sources do not do it. pr-0060 asked for an error telling the caller they
must supply a tag rather than defaulting to `latest`. pr-0015 asked that a warning say which
variables disagreed, and what the condition usually means — clock jamming. Neither names a
reader; both say what happened and what to do about it.
