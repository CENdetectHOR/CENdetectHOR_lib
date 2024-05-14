from functools import partial
import multiprocessing
import math
from matrix_closure import matrix_sparsity, graph_connected_components
import numpy as np
import editdistance
from Bio.SeqRecord import SeqRecord
from multiprocessing import Pool
from dataclasses import dataclass


def build_string_distance_triu(strings: list[str]) -> np.ndarray:
    seq_triu = np.array([
        (0 if j <= i else editdistance.eval(seq_i, strings[j]))
        for i, seq_i in enumerate(strings)
        for j in range(len(strings))
    ])
    seq_triu.shape = (len(strings), len(strings))
    return seq_triu


def build_string_distance_matrix(strings: list[str]) -> np.ndarray:
    seq_triu = build_string_distance_triu(strings)
    return seq_triu + seq_triu.T


def build_string_cross_distance_matrix(
    row_strings: list[str],
    col_strings: list[str]
) -> np.ndarray:
    dist_matrix = np.array([
        editdistance.eval(row_string, col_string)
        for row_string in row_strings
        for col_string in col_strings
    ])
    dist_matrix.shape = (len(row_strings), len(col_strings))
    return dist_matrix


def build_seqs_distance_matrix(seqs: list[SeqRecord]) -> np.ndarray:
    return build_string_distance_matrix([str(seq.seq) for seq in seqs])

@dataclass
class ChunkIndex:
    row: int
    col: int

class ChunkParams:
    def get_index(self) -> ChunkIndex:
        pass

@dataclass
class ChunkParamsInternal(ChunkParams):
    row_index: int
    col_index: int
    row_strings: list[str]
    col_strings: list[str]

    def __str__(self):
        return f"[{self.row_index}:{self.col_index}]({len(self.row_strings)},{len(self.col_strings)})"
    
    def get_index(self):
        return ChunkIndex(self.row_index, self.col_index)


@dataclass
class ChunkParamsDiagonal(ChunkParams):
    index: int
    strings: list[str]

    def __str__(self):
        return f"[{self.index}:{self.index}]({len(self.strings)},{len(self.strings)})"

    def get_index(self):
        return ChunkIndex(self.index, self.index)

class ChunkResults:

    def get_data(self) -> np.ndarray:
        pass


@dataclass
class ChunkResultsInternal(ChunkResults):
    row_index: int
    col_index: int
    dist_matrix: np.ndarray

    def get_data(self) -> np.ndarray:
        return self.dist_matrix

@dataclass
class ChunkResultsDiagonal(ChunkResults):
    index: int
    dist_triu: np.ndarray

    def get_data(self) -> np.ndarray:
        return self.dist_triu



@dataclass
class JobParams:
    chunk_params: list[ChunkParams]

    def __str__(self):
        return "(" + ",".join([str(cp) for cp in self.chunk_params]) + ")"


@dataclass
class JobResult:
    chunk_results: list[ChunkResults]


class ChunkStore:
    def get(self, chunk_params: ChunkParams) -> ChunkResults:
        return None

    def set(self, chunk_params: ChunkParams, chunk_results: ChunkResults):
        pass

dummy_chunk_store = ChunkStore()

class FileSystemChunkStore:
    filename_template: str
    
    def __init__(self, filename_template) -> None:
        self.filename_template = filename_template
    
    def get(self, chunk_params: ChunkParams) -> ChunkResults:
        filename = self.filename_template.format_map(vars(chunk_params.get_index()))
        try:
            cache_chunk_results_data = np.load(filename + '.npy')
            if isinstance(chunk_params, ChunkParamsDiagonal):
                return ChunkResultsDiagonal(
                    index=chunk_params.index,
                    dist_triu=cache_chunk_results_data
                )
            if isinstance(chunk_params, ChunkParamsInternal):
                return ChunkResultsInternal(
                    row_index=chunk_params.row_index,
                    col_index=chunk_params.col_index,
                    dist_matrix=cache_chunk_results_data
                )
        except IOError:
            return None

    def set(self, chunk_params: ChunkParams, chunk_results: ChunkResults):
        filename = self.filename_template.format_map(vars(chunk_params.get_index()))
        np.save(filename, chunk_results.get_data())


def compute_chunk(chunk_params: ChunkParams) -> ChunkResults:
    if isinstance(chunk_params, ChunkParamsDiagonal):
        return ChunkResultsDiagonal(
            index=chunk_params.index,
            dist_triu=build_string_distance_triu(chunk_params.strings)
        )
    if isinstance(chunk_params, ChunkParamsInternal):
        return ChunkResultsInternal(
            row_index=chunk_params.row_index,
            col_index=chunk_params.col_index,
            dist_matrix=build_string_cross_distance_matrix(
                row_strings=chunk_params.row_strings,
                col_strings=chunk_params.col_strings
            )
        )

def compute_chunk_if_needed(
    chunk_params: ChunkParams,
    chunk_store: ChunkStore = dummy_chunk_store
) -> ChunkResults:
    cached_chunk_results = chunk_store.get(chunk_params)
    if cached_chunk_results is not None:
        return cached_chunk_results
    fresh_chunk_results = compute_chunk(chunk_params)
    chunk_store.set(chunk_params, fresh_chunk_results)
    return fresh_chunk_results

def execute_job(
    job_params: JobParams,
    chunk_store: ChunkStore = dummy_chunk_store
) -> JobResult:
    return JobResult([
        compute_chunk_if_needed(chunk_params, chunk_store)
        for chunk_params in job_params.chunk_params
    ])


def build_string_distance_matrix_by_chunks(
    strings: list[str],
    num_chunks: int = None,
    max_num_processes: int = None,
    chunk_store: ChunkStore = dummy_chunk_store
) -> np.ndarray:

    num_strings = len(strings)

    if (num_chunks is None):
        if (max_num_processes is None):
            print(f"CPUs detected: {multiprocessing.cpu_count()}")
            max_num_processes = multiprocessing.cpu_count()
        num_chunks = int(math.sqrt(max_num_processes * 2))

    print(f"# of chunks for computing distance matrix: {num_chunks}")

    chunk_size = math.ceil(num_strings / num_chunks)

    print(f"Chunk size: {chunk_size}")

    def chunk(index):
        offset = index * chunk_size
        return strings[offset:offset + chunk_size]

    internal_blocks = [
        ChunkParamsInternal(
            row_strings=chunk(row_index),
            col_strings=chunk(col_index),
            row_index=row_index,
            col_index=col_index
        )
        for col_index in range(num_chunks)
        for row_index in range(col_index)
    ]

    diagonal_blocks = [
        ChunkParamsDiagonal(
            strings=chunk(index),
            index=index
        )
        for index in range(num_chunks)
    ]

    jobs_params = [
        JobParams([internal_block])
        for internal_block in internal_blocks
    ] + [
        JobParams(diagonal_blocks[i*2:(i+1)*2])
        for i in range(math.ceil(len(diagonal_blocks) / 2))
    ]

    print(f"Num processes: {len(jobs_params)}")
    print(f"Blocks: {[str(jb) for jb in jobs_params]}")

    with Pool(max_num_processes) as p:
        job_results = p.map(partial(execute_job, chunk_store=chunk_store),jobs_params)

    results_matrix = [
        [
            np.zeros((
                chunk_size
                if row_index < num_chunks - 1 or num_strings % chunk_size == 0
                else num_strings % chunk_size,
                chunk_size
                if col_index < num_chunks - 1 or num_strings % chunk_size == 0
                else num_strings % chunk_size
            ))
            for col_index in range(num_chunks)
        ]
        for row_index in range(num_chunks)
    ]

    for job_result in job_results:
        for chunk_result in job_result.chunk_results:
            if isinstance(chunk_result, ChunkResultsDiagonal):
                results_matrix[chunk_result.index][chunk_result.index] = chunk_result.dist_triu
            elif isinstance(chunk_result, ChunkResultsInternal):
                results_matrix[chunk_result.row_index][chunk_result.col_index] = chunk_result.dist_matrix

    print(
        f"Block result shapes: {[[result.shape for result in results_row] for results_row in results_matrix]}")

    global_dist_triu = np.block(results_matrix)
    return global_dist_triu + global_dist_triu.T
    # np.zeros((num_strings, num_strings))


def build_seqs_distance_matrix_by_chunks(
    seqs: list[SeqRecord],
    num_chunks: int = None,
    max_num_processes: int = None,
    chunk_store: ChunkStore = dummy_chunk_store
) -> np.ndarray:
    return build_string_distance_matrix_by_chunks(
        strings=[str(seq.seq) for seq in seqs],
        num_chunks=num_chunks,
        max_num_processes=max_num_processes,
        chunk_store=chunk_store
    )
