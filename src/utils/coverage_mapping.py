import pysam
import pandas as pd
from typing import Any


def read_bam_as_bed(file_path: str) -> pd.DataFrame:
    bam_file = pysam.AlignmentFile(file_path, "rb")

    records: list[tuple[str, int, int, int]] = []

    for chrom in bam_file.references:
        length = bam_file.get_reference_length(chrom)
        run_pos: int | None = None
        run_coverage: int | None = None

        for col in bam_file.pileup(chrom, 0, length, truncate=True):
            col_pos = col.pos
            col_coverage = col.n

            if run_coverage is None:
                run_pos = col_pos
                run_coverage = col_coverage
                continue

            if col_coverage != run_coverage:
                records.append((chrom, run_pos, col_pos, run_coverage))
                run_pos = col_pos
                run_coverage = col_coverage

        if run_coverage is not None:
            records.append((chrom, run_pos, length, run_coverage))

    bam_file.close()
    
    return pd.DataFrame(records, columns=["chrom", "start", "end", "value"])