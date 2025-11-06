
import os
import gzip
from collections import defaultdict, OrderedDict
from collections.abc import Iterator

import pandas as pd
from shortuuid import uuid

def read_fasta(path: str) -> dict[str, str]:
    fasta_dict = {}
    handle = gzip.open(path, "rt") if path.endswith(".gz") else open(path)
    desc = ""

    for line in handle:
        if line.startswith(">"):
            desc = line.strip()[1:]
            fasta_dict[desc] = ""
        else:
            fasta_dict[desc] += line.strip()

    handle.close()
    return fasta_dict


def write_fasta(fasta_dict: dict[str, str], path: str, line_length: int = 80):
    handle = gzip.open(path, "wt") if path.endswith(".gz") else open(path, "w")

    for desc, seq in fasta_dict.items():
        handle.write(f">{desc}\n")

        for i in range(0, len(seq), line_length):
            handle.write(seq[i:i + line_length] + "\n")

    handle.close()


def read_fasta_in_chunks(path: str, chunk_size: int = 1000000) -> Iterator[tuple[str, str]]:
    handle = gzip.open(path, "rt") if path.endswith(".gz") else open(path)

    # Skip to the first line that starts with ">"
    line = next(handle).strip()

    while not line.startswith(">"):
        line = next(handle).strip()

    desc = line[1:]
    seq = ""

    for line in handle:
        if line.startswith(">"):
            yield desc, seq
            desc = line.strip()[1:]
            seq = ""
        else:
            seq += line.strip()

            if len(seq) >= chunk_size:
                yield desc, seq[:chunk_size]
                seq = seq[chunk_size:]

    yield desc, seq


# Writes sequence chunks to files and returns an index file
def chunk_fasta(fasta_path: str, output_dir: str, chunk_size: int = 1000000, line_length: int = 80) -> pd.DataFrame:

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    index_dict = defaultdict(list)
    prev_desc = None

    for desc, chunk in read_fasta_in_chunks(fasta_path, chunk_size):
        if desc != prev_desc:
            prev_desc = desc
            start_idx = 0

        chunk_file_name = f"{uuid()}.txt.gz"
        chunk_file_path = os.path.join(output_dir, chunk_file_name)

        with gzip.open(chunk_file_path, "wt") as f:
            for j in range(0, len(chunk), line_length):
                f.write(chunk[j:j + line_length].upper() + "\n")

        index_dict["file_name"].append(chunk_file_name)
        index_dict["src"].append(desc)
        index_dict["start_idx"].append(start_idx)
        stop_idx = start_idx + len(chunk)
        index_dict["stop_idx"].append(stop_idx)
        start_idx = stop_idx

    return pd.DataFrame(index_dict)


class ChunkedSequenceReader:
    def __init__(self, chunk_dir: str, index_path: str):
        self.chunk_dir = chunk_dir
        self._index_df = pd.read_csv(index_path)

    @property
    def index_df(self) -> pd.DataFrame:
        return self._index_df

    def __getitem__(self, key: str) -> dict[str, slice]:
        mask = self.index_df["src"] == key
        file_path_to_slice = {}

        for __, row in self.index_df.loc[mask].iterrows():
            file_path = f"{self.chunk_dir}/{row['file_name']}"
            file_path_to_slice[file_path] = slice(row["start_idx"], row["stop_idx"])
        
        return ChunkedSequence(file_path_to_slice)
    

class ChunkedSequence:
    def __init__(self, file_path_to_start_idx: dict[str, int]):
        self._file_path_to_slice = OrderedDict(sorted(file_path_to_start_idx.items(), key=lambda x: x[1]))

        self._seq_length = 0

        for slice_ in self._file_path_to_slice.values():
            self._seq_length = max(slice_.stop, self._seq_length)

    def __getitem__(self, key: int | slice) -> str:
        if isinstance(key, int):
            key = slice(key, key + 1)
        
        seq_start_idx, seq_stop_idx, step_size = key.indices(self._seq_length)

        if seq_stop_idx == 0:
            seq_stop_idx = self._seq_length

        if seq_start_idx == self._seq_length:
            raise IndexError("index out of range")

        if step_size is not None and step_size != 1:
            raise NotImplementedError("`step` parameter is not supported")
        
        seq = ""

        for file_path, slice_ in self._file_path_to_slice.items():
            if seq_stop_idx <= slice_.start:
                break

            if seq_start_idx >= slice_.stop:
                continue
            
            chunk_start_idx = max(seq_start_idx - slice_.start, 0)
            chunk_stop_idx = max(seq_stop_idx - slice_.start, 0)
            chunk_length = chunk_stop_idx - chunk_start_idx

            handle = gzip.open(file_path, "rt") if file_path.endswith(".gz") else open(file_path)
            line_length = len(handle.readline().rstrip("\n"))                
            handle.seek(chunk_start_idx + chunk_start_idx // line_length)
            chunk = handle.read(chunk_length + chunk_length // line_length + 1).replace("\n", "")
            chunk = chunk[:chunk_length]
            handle.close()

            seq += chunk

        return seq
    
    def __len__(self) -> int:
        return self._seq_length