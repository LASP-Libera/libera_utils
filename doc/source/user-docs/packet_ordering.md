# Packet Ordering

`parse_packets_to_l1a_dataset` puts each APID's `PACKET` axis in **`SRC_SEQ_CTR` order**, the order
the instrument built its packets in, whatever the order of the input files. The axis is **not** sorted
by packet time, which can step backward while `SRC_SEQ_CTR` advances. The code is in
`libera_utils.l1a.packet_ordering`.

## The counter and packet time

`SRC_SEQ_CTR` is the CCSDS packet sequence counter. It advances by one for each packet of an APID and
is 14 bits, so it wraps from 16383 back to 0. On its own a counter value does not say which wrap a
packet falls in, and the counter restarts when the instrument does.

Packet time (`PACKET_ICIE_TIME`, or the APID's configured packet time variable) is when the ICIE built
the packet. It places packets roughly, but it is not an order: in the DITL2 ground test it stepped
backward along counter order by up to 1.99 s on WFOV SCI and 0.81 s on RAD SAMPLE. Packet time is used
only to tell which wrap of the counter a packet falls in, and to place packets where the counter cannot.

## How the order is found

`sort_by_sequence_counter` gives each packet a sort key: its counter with the wraps counted.

1. Take the packets in packet-time order.
2. Read each counter step between neighbors modulo 16384, as -8192 to 8191. `16383 -> 0` is one step
   forward, and `16380 -> 2` is a step of 6, five packets lost.
3. Where a step repeats or goes backward and the two packets are more than `REORDER_WINDOW` (10 s)
   apart in packet time, cut the packets into segments there. A backward step that far apart in time
   is not a reordering the counter can explain: the counter restarted, more than 8,191 packets were
   lost, or a packet clock is wrong. The counter alone cannot tell these apart.
4. Within a segment, add up the steps, so a packet whose time is slightly out of order still lands by
   its counter.
5. Stack the segments in packet-time order, each starting at the smallest forward counter step from the
   one before it.
6. Sort the axis by the key, and drop packets with the same key that were received twice.

So the counter reorders packets only where their packet times are within 10 s of each other. Across a
gap in time of any length, a forward counter step is read as forward, with fewer than one wrap lost.
Anywhere else the packets go in packet-time order.

| Counters                  | Packet times (s)          | Order                     | Why                                                   |
| ------------------------- | ------------------------- | ------------------------- | ----------------------------------------------------- |
| 100, 101, 102, 103        | 0, 2, 1, 3                | 100, 101, 102, 103        | A backward step within 10 s: reordered by counter.    |
| 16382, 16383, 0, 1        | 0, 1, 2, 3                | 16382, 16383, 0, 1        | A wrap is one forward step.                           |
| 10, 9010, 9011            | 0, 100, 101               | 10, 9010, 9011            | 9,000 lost reads as a step back, 100 s apart: a cut.  |
| 5000, 5001, 5002, 0, 1, 2 | 0, 60, 120, 300, 360, 420 | 5000, 5001, 5002, 0, 1, 2 | A restart minutes later: a cut, placed by time.       |
| 10, 11, 13, 14            | 200, 201, 81, 82          | 13, 14, 10, 11            | The clock stepped back 120 s at a lost packet: a cut. |

The last row may well be in the wrong order: the counter says 10 and 11 were built first. With a packet
lost between them, nothing in the packets shows whether the clock stepped back or the counter
restarted, so packet time decides.

Each cut is logged at INFO, with the APID and the counters and packet times either side of it:

```text
APID 1060: SRC_SEQ_CTR goes from 5002 at 2026-07-12T00:02:00.000000 to 0 at 2026-07-12T00:05:00.000000,
180.0 s later, beyond REORDER_WINDOW (10 s). The counter restarted, more than 8,191 packets were lost, or
a packet clock is wrong; the packets from 2026-07-12T00:05:00.000000 on are placed by packet time.
```

## What is dropped and what raises

Packets next to each other in an input file with consecutive counters were built one after the other.
If packet time puts them a counter wrap apart, a packet time between them is wrong by more than 10 s:

- **One bad packet time is dropped.** A packet a wrap away from both of its neighbors, whose counters
  continue its own, is the one with the wrong time. It is dropped, with its samples, and a `UserWarning`
  and a logged warning name the APID, its counter, its packet time and its neighbors' times. For
  example, counters 10 to 14 at 0, 1, 3e8, 3, 4 s keep 10, 11, 13 and 14.
- **A clock step raises.** Where the wrap falls between two runs of packets, nothing shows which run has
  the wrong time, and the parse raises `ValueError` naming the APID, the counters and the packet times.
  For example, counters 10, 11, 12, 13 at 200, 201, 81, 82 s raise, as do two bad packet times side by
  side. This holds for ground and flight data alike.

TODO[LIBSDC-830]: carry the count of packets dropped for a bad packet time on the granule.

**Packets that share a counter.** Packets with the same key and the same contents are one packet
received twice, such as from two overlapping files, and are kept once with a `UserWarning`. Packets with
the same key and different contents raise `ValueError`, or keep the first with `ground_data=True`.
Packets that share a packet time but not a counter are all kept.

Input files are parsed in file name order, which `input_files` lists. Two packets adjacent across a file
join are then neighbors in time, so the check above does not pair the end of a later file with the start
of an earlier one. Flight PDS names carry the file's creation time, so a PDS file re-delivered later with
earlier data still sorts out of time order. If the file that sorts first then ends one counter below where
the next file starts, that join reads as consecutive counters a wrap apart and the parse raises
`ValueError`.

## What warns

After the sort, `parse_packets_to_l1a_dataset` calls `check_packet_order`. Where packet time steps
backward along the counter order, it emits a `UserWarning` and logs a warning with the count, the largest
step and the first one's counters and times. The packets stay in counter order and nothing is raised. A
dataset without `SRC_SEQ_CTR` or its packet time variable raises `ValueError`.

`nom_hk_trim.find_obsid_runs` calls `out_of_order_positions` and raises `ValueError` on a NOM-HK product
whose counter repeats or steps backward between packets within 10 s of each other. A product written in
packet-time order by a release before 5.23.0 does this; decode its L0 files again.

## Gaps the counter cannot size

Some losses cannot be measured from the packets. A forward step across a gap is read as fewer than one
wrap lost, and the lost count across a cut is unknown: a gap that hides a whole wrap of the counter looks
the same as one that hides a few packets. These gaps change only a missing-packet count, never the order,
because a cut is placed by packet time.

TODO[LIBSDC-878]: count the SSC packet completeness of each region of the day for the L1A day coverage
gate. A gap the counter cannot size counts its modulo step as missing, a lower bound, and is listed as
ambiguous with its length in packet time.

TODO[LIBSDC-830]: report the missing-packet count (counter steps above 1 between neighbors on the
`PACKET` axis) and each backward packet-time step, and carry both counts on the granule. Today no check
reports scattered lost packets: on the DITL2 golden day, 2026-07-12, WFOV SCI lost 3,760 packets in 11
drops and still wrote a product with no warning about them.

## Choosing `REORDER_WINDOW`

The window has two limits, recorded next to the constant in `packet_ordering.py`; a change to it must
keep both:

- **Lower.** It must exceed the largest backward packet-time step between neighbors in counter order,
  or those packets are placed by packet time instead of by counter. In DITL2 that step is at most 1.99 s
  across the 33 WFOV SCI files and 3.7 ms across NOM-HK.
- **Upper.** A counter restart that resumes within the window is placed by its counter, ahead of the
  packets built before it, so a longer window misorders more restarts. Losing 8,192 or more packets
  within the window also reads as a backward step placed by counter, so an APID must send fewer than
  8,192 packets in it: about 819 packets/s at 10 s, against ~85 packets/s for WFOV SCI in DITL2.

The rule does not depend on an APID's packet rate otherwise, so the APIDs that report once a minute
(TEMP-HK, PEC-SW-STAT, PEV-SW-STAT) are ordered by the same rule as WFOV SCI.

## For consumers

`{group}_packet_index` and `CAMERA_PACKET_INDEX` index the `PACKET` axis by position. `PACKET_IMAGE_ID`
lies on `PACKET` but its values index `CAMERA_TIME`, with `-1` for a packet in no image.

Anything that needs packet times in ascending order sorts them first: `numpy.searchsorted` and
`numpy.interp` return wrong values without raising on an unsorted axis. Sorting does not correct a wrong
time. Where packet time steps backward along `PACKET`, the packet time of the packets on one or both sides
of the step is wrong; L1A keeps those packets, in counter order, and does not mark which time is wrong.
Find them at the positions where `numpy.diff` of the packet time is below zero, the steps
`check_packet_order` warns about, and exclude or flag those packets before using packet time as the
independent variable.

The granule filename's start and end are the minimum and maximum of its time variable.
