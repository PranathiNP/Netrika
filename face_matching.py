"""Conservative frame-level matching: one detection per student ID."""
import numpy as np


def match_faces(features, known, threshold=0.50, margin=0.08):
    matches = [None] * len(features)
    candidates = {}
    for index, feature in enumerate(features):
        if feature is None:
            continue
        # Collapse duplicate templates by identity before measuring ambiguity.
        scores = {}
        for student_id, name, template in known:
            score = float(np.dot(feature, template))
            if np.isfinite(score) and (student_id not in scores or score > scores[student_id][0]):
                scores[student_id] = (score, name)
        ranked = sorted(scores.items(), key=lambda item: item[1][0], reverse=True)
        if not ranked:
            continue
        student_id, (score, name) = ranked[0]
        if score < threshold:
            continue
        if len(ranked) > 1 and score - ranked[1][1][0] < margin:
            continue
        candidates.setdefault(student_id, []).append((score, index, name))

    for student_id, contenders in candidates.items():
        contenders.sort(reverse=True)
        score, index, name = contenders[0]
        # Similar-looking detections competing for one identity are all unknown.
        # Never assign losers to their second-best student just to fill a slot.
        if len(contenders) > 1 and score - contenders[1][0] < margin:
            continue
        matches[index] = (student_id, name)
    return matches
