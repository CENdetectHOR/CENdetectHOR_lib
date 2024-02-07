import csv
from Bio.SeqFeature import SeqFeature, FeatureLocation 
import numpy as np

def BED_file_to_features(BED_filename):
    with open(BED_filename) as tsv:
        return [
            SeqFeature(
                FeatureLocation(
                    int(bedLine['chromStart']),
                    int(bedLine['chromEnd']),
                    strand=(1 if bedLine['strand'] == '+' else (-1 if bedLine['strand'] == '-' else None)) if 'strand' in bedLine else None,
                    ref=bedLine['chrom']
                ),
                id=bedLine['name'], type='repeat'
            )
            for bedLine in csv.DictReader(tsv, delimiter="\t", fieldnames=[
            'chrom', 'chromStart', 'chromEnd', 'name', 'score', 'strand',
            'thickStart', 'thickEnd', 'itemRgb', 'blockCount', 'blockSizes', 'blockStarts'])
        ]


def feature_to_seq(feature, references):
    seq = feature.extract(None, references=references)
    seq.id = feature.id
    return seq 

def id_from_location(location):
    return (
        f'{location.ref}:{location.start}-{location.end}'
        + f'({"-" if location.strand == -1 else "+"})' if location.strand is not None else ''
    )

def location_to_feature(location):
    return SeqFeature(location=location, id=id_from_location(location))

def location_to_seq(location, references):
    return feature_to_seq(location_to_feature(location), references)

def extract_features_from_labels(seqs):
    seqs_as_features = []
    for seq in seqs:
        label_parts = seq.id.split(',')
        coordinates = label_parts[0]
        coordinates_parts = coordinates.split(':')
        seq_label = coordinates_parts[0]
        start_end = coordinates_parts[1].split('-')
        start = int(start_end[0])
        end = int(start_end[1])
        props = {}
        for label_index in range(len(label_parts) - 1):
            key_value = label_parts[label_index].split('=')
            props[key_value[0]] = key_value[1] if len(key_value) > 1 else True
        strand = (1 if props['strand'] == "'+'" else (-1 if props['strand'] == "'-'" else None)) if 'strand' in props else None
        seqs_as_features.append(SeqFeature(FeatureLocation(start, end, strand=strand, ref=seq_label), id=seq.id, type='repeat'))
    return seqs_as_features

def extract_indices(indexed_list):
    return [indexed_item[0] for indexed_item in indexed_list]

def sorted_locations_indeces(locations):
    indexed_locations = list(enumerate([location for location in locations]))
    sorted_indeces = []
    for curr_ref in sorted(set([location.ref for location in locations])):
        curr_ref_indexed_locations = [
            indexed_location for indexed_location in indexed_locations if indexed_location[1].ref == curr_ref
        ]
        def indexed_locations_by_strand(strand):
            return [
                indexed_location
                for indexed_location in curr_ref_indexed_locations
                if indexed_location[1].strand == strand
            ]
        def get_start(indexed_location):
            return indexed_location[1].start
        sorted_indeces.extend(
            extract_indices(sorted(indexed_locations_by_strand(1), key=get_start))
            + extract_indices(sorted(indexed_locations_by_strand(-1), key=get_start, reverse=True))
        )
    return sorted_indeces

def order_by_indices(list_to_order, reordered_indeces):
    return [list_to_order[old_index] for old_index in reordered_indeces] if list_to_order is not None else None

def sorted_locations(locations):
    order_by_indices(locations, sorted_locations_indeces(locations))

def sorted_features(features):
    order_by_indices(features, sorted_locations_indeces([feature.location for feature in features]))

def indeces_to_ordering_matrix(reordered_indeces):
    list_size = len(reordered_indeces)
    return np.array([[i == old_index for i in range(list_size)] for old_index in reordered_indeces]).T

def order_matrix_by_indeces(matrix_to_order, reordered_indeces, reorder_rows=True, reorder_cols=True):
    if matrix_to_order is None:
        return None
    ordering_matrix = indeces_to_ordering_matrix(reordered_indeces)
    matrix = matrix_to_order
    if reorder_cols:
        matrix = matrix @ ordering_matrix
    if reorder_rows:
        matrix = (matrix.T @ ordering_matrix).T
    return matrix

