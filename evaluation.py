"""Validate reviewer judgments before any relevance metrics are calculated."""
import pandas as pd
from core import evaluate

def validate_labels(frame, candidate_ids):
    if not {'candidate_id','label'}.issubset(frame.columns):
        raise ValueError('The file needs candidate_id and label columns.')
    ids=frame.candidate_id.astype(str)
    if ids.duplicated().any():
        raise ValueError('Each candidate ID must appear exactly once.')
    if set(ids)!=set(candidate_ids):
        raise ValueError('The candidate IDs must match this experiment exactly.')
    try:
        values=pd.to_numeric(frame.label,errors='raise')
    except (ValueError,TypeError) as exc:
        raise ValueError('Fill every label with 0, 1 or 2.') from exc
    if values.isna().any() or not values.isin([0,1,2]).all():
        raise ValueError('Fill every label with 0, 1 or 2.')
    return dict(zip(ids,values.astype(int)))

def evaluate_methods(results,labels,k):
    return pd.DataFrame([{'method':method,**evaluate(result['ranking'],labels,k)} for method,result in results.items()])
