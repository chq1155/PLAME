"""Utilities for reading FASTA/A3M alignment files."""

import re


def read_fasta(path, keep_gaps=True, keep_insertions=True, to_upper=True):
    """Yield (description, sequence) tuples from a FASTA/A3M file."""
    with open(path, "r") as f:
        yield from read_alignment_lines(
            f, keep_gaps=keep_gaps, keep_insertions=keep_insertions, to_upper=to_upper
        )


def read_alignment_lines(lines, keep_gaps=True, keep_insertions=True, to_upper=True):
    """Yield (description, sequence) tuples from an iterable of lines."""
    seq = desc = None

    def parse(s):
        if not keep_gaps:
            s = re.sub("-", "", s)
        if not keep_insertions:
            s = re.sub("[a-z]", "", s)
        return s.upper() if to_upper else s

    for line in lines:
        if len(line) > 0 and line[0] == ">":
            if seq is not None:
                yield desc, parse(seq)
            desc = line.strip().lstrip(">")
            seq = ""
        elif len(line) > 0 and line[0] in ("#", "\n"):
            continue
        else:
            if seq is None:
                continue
            line = line.replace("\x00", "")
            seq += line.strip()

    if seq is not None and desc is not None:
        yield desc, parse(seq)
