import numpy as np
from DINOv3.MADEqual.hard_sample_discrimination.scoring import fit_head, fixed_fpr_diagnostics
from DINOv3.relation_reliability.superadd_host import interpolate_score

METHODS = ('ALL_STABLE', 'NORMAL_RECON')


def fit_mad(calibration_errors):
    if not calibration_errors or any(a.ndim != 3 or a.shape[0] != 2 or not np.isfinite(a).all()
                                     for a in calibration_errors):
        raise ValueError('two finite native normal calibration maps required')
    # Existing four-layer MAD fitter is reused through its existing one-map adapter.
    return [dict(fit_head([a[i] for a in calibration_errors]), group=group,
                 input='native normal reconstruction cosine error', image_count=len(calibration_errors))
            for i, group in enumerate(('low', 'high'))]


def calibrated_map(errors, states, shape):
    z = np.stack([(np.asarray(errors[i], np.float32) - np.float32(s['median'])) / np.float32(s['scale'])
                  for i, s in enumerate(states)])
    if z.shape[0] != 2 or not np.isfinite(z).all(): raise ValueError('invalid calibrated reconstruction')
    return interpolate_score(z.mean(0), shape)


def operating_sets(maps, masks, cap):
    # Minimal set-returning adapter to fixed_fpr_diagnostics' exact same
    # whole-tie operating rule. The caller verifies its FP counts against it.
    negatives = np.concatenate([np.asarray(p, np.float32)[~np.asarray(m, bool)] for p, m in zip(maps, masks)])
    if not len(negatives) or not 0 <= cap < 1: raise ValueError('invalid FPR operating point')
    k = len(negatives) - 1 - int(np.floor(cap * len(negatives)))
    boundary = np.partition(negatives, k)[k]
    return [np.asarray(p) > boundary for p in maps]


def complementarity(base, recon, masks, caps=(.01, .05)):
    diagnostics = [fixed_fpr_diagnostics(m, masks, caps) for m in (base, recon)]
    rows = []
    for j, cap in enumerate(caps):
        a, b = operating_sets(base, masks, cap), operating_sets(recon, masks, cap)
        counts = dict(base_fp_corrected=0, base_tn_new_fp=0, base_tp_retained=0, base_tp_lost=0, base_fn_new_tp=0)
        den = dict(base_fp=0, base_tn=0, base_tp=0, base_fn=0)
        for x, y, mask in zip(a, b, masks):
            t = np.asarray(mask, bool)
            for key, sel in [('base_fp_corrected', x & ~y & ~t), ('base_tn_new_fp', ~x & y & ~t),
                             ('base_tp_retained', x & y & t), ('base_tp_lost', x & ~y & t),
                             ('base_fn_new_tp', ~x & y & t)]: counts[key] += int(sel.sum())
            for key, sel in [('base_fp', x & ~t), ('base_tn', ~x & ~t), ('base_tp', x & t), ('base_fn', ~x & t)]:
                den[key] += int(sel.sum())
        for sets, diag in zip((a, b), diagnostics):
            fp = sum(int((x & ~np.asarray(t, bool)).sum()) for x, t in zip(sets, masks))
            if fp != diag[j]['false_positive_pixels']: raise RuntimeError('FPR adapter mismatch')
        row = dict(fpr_cap=cap, base_actual_fpr=diagnostics[0][j]['actual_fpr'],
                   recon_actual_fpr=diagnostics[1][j]['actual_fpr'], **counts, **den)
        for key, denominator in zip(counts, ('base_fp', 'base_tn', 'base_tp', 'base_tp', 'base_fn')):
            row[key+'_ratio'] = counts[key] / den[denominator] if den[denominator] else 'N/A'
            row[key+'_denominator'] = denominator
        rows.append(row)
    return rows
