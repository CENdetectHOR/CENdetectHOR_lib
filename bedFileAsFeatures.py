import csv
from Bio.SeqFeature import SeqFeature, FeatureLocation 

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
