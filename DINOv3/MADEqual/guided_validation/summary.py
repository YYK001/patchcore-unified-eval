"""Category-equal summaries; count totals are explicitly separate from means."""
import numpy as np
from DINOv3.MADEqual.mvtec_broad6_compose2.metrics import METRIC_NAMES

METHODS = ('BASE', 'GUIDED')
SMOOTH_METHODS = (*METHODS, 'SMOOTH')
SMOOTH_COMPARISONS = (('GUIDED', 'SMOOTH'), ('SMOOTH', 'BASE'), ('GUIDED', 'BASE'))
CAPS = (.01, .05)
FPR_FIELDS = ('actual_fpr', 'defect_pixel_recall', 'region_mean_coverage',
              'small_region_mean_coverage')
CHANGE_COUNTS = ('base_fp_corrected', 'base_tn_new_fp', 'base_tp_retained',
                 'base_tp_lost', 'base_fn_new_tp')
TIE_TOLERANCE = 1e-10


def is_na(value):
    return value is None or str(value) == 'N/A'


def mean_available(values):
    valid = [float(v) for v in values if not is_na(v)]
    if not np.isfinite(valid).all():
        raise ValueError('nonfinite summary input')
    return (float(np.mean(valid)) if valid else 'N/A'), len(valid)


def difference(a, b):
    return 'N/A' if is_na(a) or is_na(b) else float(a) - float(b)


def indexed(rows, keys, expected):
    result = {tuple(row[k] for k in keys): row for row in rows}
    if len(result) != len(rows) or set(result) != set(expected):
        raise ValueError(f'incomplete/duplicate category results: keys={keys}')
    return result


def summarize(metrics, operating, complements, categories, methods=METHODS):
    methods = tuple(methods)
    if methods not in (METHODS, SMOOTH_METHODS):
        raise ValueError('unsupported method set')
    comparisons = SMOOTH_COMPARISONS if methods == SMOOTH_METHODS else (('GUIDED', 'BASE'),)
    categories = tuple(categories)
    if not categories or len(set(categories)) != len(categories):
        raise ValueError('unique nonempty categories required')
    mt = indexed(metrics, ('category', 'method'), [(c, m) for c in categories for m in methods])
    ft = indexed(operating, ('category', 'method', 'fpr_cap'),
                 [(c, m, cap) for c in categories for m in methods for cap in CAPS])
    # Existing two-method CSV schemas remain unchanged.
    expanded = methods == SMOOTH_METHODS
    ct = indexed(complements, ('category', 'fpr_cap', 'comparison') if expanded else ('category', 'fpr_cap'),
        [(c, cap, f'{a}_minus_{b}') for c in categories for cap in CAPS for a,b in comparisons] if expanded
        else [(c, cap) for c in categories for cap in CAPS])
    macro = [dict(method=m, category_count=len(categories), averaging='category_unweighted',
                  **{k: mean_available([mt[c, m][k] for c in categories])[0] for k in METRIC_NAMES})
             for m in methods]
    if any(is_na(row[k]) for row in metrics for k in METRIC_NAMES):
        raise ValueError('five formal metrics cannot be N/A')
    deltas = [dict(category=c, comparison=f'{a}_minus_{b}',
                   **{k: difference(mt[c, a][k], mt[c, b][k]) for k in METRIC_NAMES})
              for a,b in comparisons for c in categories]
    mm = {r['method']: r for r in macro}
    macro_delta = [dict(scope='category_unweighted_macro', comparison=f'{a}_minus_{b}',
                        **{k: mm[a][k] - mm[b][k] for k in METRIC_NAMES}) for a,b in comparisons]
    fmacro = []
    for m in methods:
        for cap in CAPS:
            group = [ft[c, m, cap] for c in categories]
            row = dict(method=m, fpr_cap=cap, category_count=len(categories),
                       averaging='unweighted_category_means_excluding_NA')
            for k in FPR_FIELDS:
                row[k], row[k+'_valid_category_count'] = mean_available([r[k] for r in group])
            for k in ('region_count', 'small_region_count', 'small_region_image_count'):
                row[k+'_sum'] = sum(int(r[k]) for r in group)
            fmacro.append(row)
    fdelta = [dict(category=c, fpr_cap=cap, comparison=f'{a}_minus_{b}',
                   **{k: difference(ft[c, a, cap][k], ft[c, b, cap][k]) for k in FPR_FIELDS})
              for a,b in comparisons for c in categories for cap in CAPS]
    fm = {(r['method'], r['fpr_cap']): r for r in fmacro}
    fmacro_delta = [dict(fpr_cap=cap, comparison=f'{a}_minus_{b}',
                        **{k: difference(fm[a, cap][k], fm[b, cap][k]) for k in FPR_FIELDS})
                    for a,b in comparisons for cap in CAPS]
    totals = []
    for a,b in comparisons:
        for cap in CAPS:
            row = dict(fpr_cap=cap, aggregation='pixel_count_sum_not_macro', category_count=len(categories),
                **{k: sum(int(ct[(c, cap, f'{a}_minus_{b}') if expanded else (c, cap)][k])
                          for c in categories) for k in CHANGE_COUNTS})
            if expanded: row.update(comparison=f'{a}_minus_{b}', reference_method=b, candidate_method=a)
            totals.append(row)
    changes = []
    for a,b in comparisons:
        name = f'{a}_minus_{b}'
        for cap, fields, rows in [(None, METRIC_NAMES, [r for r in deltas if r['comparison'] == name])] + [
            (cap, FPR_FIELDS[1:], [r for r in fdelta if r['fpr_cap'] == cap and r['comparison'] == name]) for cap in CAPS
        ]:
            for k in fields:
                values = [float(r[k]) for r in rows if not is_na(r[k])]
                row = dict(metric=k, fpr_cap='' if cap is None else cap,
                    improved=sum(v > TIE_TOLERANCE for v in values),
                    tied=sum(abs(v) <= TIE_TOLERANCE for v in values),
                    declined=sum(v < -TIE_TOLERANCE for v in values),
                    valid_category_count=len(values), missing_category_count=len(categories)-len(values),
                    tie_absolute_tolerance=TIE_TOLERANCE)
                if expanded: row['comparison'] = name
                changes.append(row)
    return dict(macro_metrics=macro, metric_differences=deltas, macro_differences=macro_delta,
                fixed_fpr_macro=fmacro, fixed_fpr_differences=fdelta,
                fixed_fpr_macro_differences=fmacro_delta, complementarity_totals=totals,
                category_change_counts=changes)
