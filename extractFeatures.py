from Bio.SeqFeature import SeqFeature, FeatureLocation 

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