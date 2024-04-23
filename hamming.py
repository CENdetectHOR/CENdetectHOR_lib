import numpy as np
import math

base_to_bits = {'A': 0, 'C': 3, 'G': 5, 'T': 6}
def block_to_bits(seq):
    bits = 0
    for base in seq:
        bits = (bits << 3) + base_to_bits[base]
    return bits


def half_bit_count(arr):
     # Make the values type-agnostic (as long as it's integers)
     # t = arr.dtype.type
     # mask = t(-1)
     # s55 = t(0x5555555555555555 & mask)  # Add more digits for 128bit support
     # s33 = t(0x3333333333333333 & mask)
     # s0F = t(0x0F0F0F0F0F0F0F0F & mask)
     # s01 = t(0x0101010101010101 & mask)

     # arr = arr - ((arr >> 1) & s55)
     # arr = (arr & s33) + ((arr >> 2) & s33)
     # arr = (arr + (arr >> 4)) & s0F
     # return ((arr * s01) >> (8 * (arr.itemsize - 1) + 1)).sum()
     return np.unpackbits(arr.view('uint8')).sum() // 2

def distance_bits(seq1, seq2):
    return half_bit_count(np.bitwise_xor(seq1, seq2))

def build_hamming_distance_matrix(dna_seqs):
    bases_for_block = 21
    seq_blocks = [
        np.array([
            block_to_bits(dna_seq[sq_index * bases_for_block : (sq_index + 1) * bases_for_block])
            for sq_index in range(math.ceil(len(dna_seq) / bases_for_block))
        ])
        for dna_seq in dna_seqs
    ]
    seq_triu = np.array([
            (0 if j <= i else distance_bits(seq_i, seq_blocks[j]))
            for i, seq_i in enumerate(seq_blocks)
            for j in range(len(seq_blocks))
    ])
    seq_triu.shape = (len(seq_blocks), len(seq_blocks))
    return seq_triu + seq_triu.T