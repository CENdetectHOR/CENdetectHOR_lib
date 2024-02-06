def feature_to_seq(feature, references):
    seq = feature.extract(None, references=references)
    seq.id = feature.id
    return seq
