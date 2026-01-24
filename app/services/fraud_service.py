def compute_fraud(matches):
    if not matches:
        return 0.0

    top_score = matches[0].score
    reuse_count = len(matches)

    fraud_score = (reuse_count * 0.4) + ((1 - top_score) * 0.6)
    return min(fraud_score, 1.0)
