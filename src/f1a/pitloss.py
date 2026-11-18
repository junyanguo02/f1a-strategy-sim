"""T009 descriptive pit loss, preserving unknown states and inferred origins."""
import pandas as pd

CONDITIONS = ('正常', 'VSC', 'SC', 'unknown')


def pair_flag(a, b):
    """Three-valued OR: true if observed on either lap, NA if unresolved."""
    values = pd.array([a, b], dtype='boolean')
    if values.fillna(False).any():
        return True
    return pd.NA if values.isna().any() else False


def pit_losses(laps, circuit_id):
    """One race input. Nearest <=3 clean laps each side, excluding in/out pair."""
    if laps[['year', 'round']].drop_duplicates().shape[0] != 1:
        raise ValueError('Expected one race')
    if laps.duplicated(['driver_number', 'lap']).any():
        raise ValueError('Duplicate driver/lap')
    rows = []
    for _, driver in laps.groupby('driver_number', sort=True):
        driver = driver.sort_values('lap').set_index('lap', drop=False)
        clean = driver.loc[driver.is_clean]
        for _, stop in driver.loc[driver.pit_in_filled.fillna(False)].iterrows():
            lap = int(stop.lap)
            out = driver.loc[lap+1] if lap+1 in driver.index else None
            inferred = bool(stop.pit_in_inferred)
            before = clean.loc[clean.lap < lap].tail(3)
            after = clean.loc[clean.lap > lap+1].head(3)
            reference = pd.concat([before, after])
            median = reference.lap_time_s.median() if len(reference) else pd.NA
            out_time = out.lap_time_s if out is not None else pd.NA
            out_flag = out.pit_out if out is not None else pd.NA
            raw_states = [stop.safety_car, stop.virtual_safety_car,
                          out.safety_car if out is not None else pd.NA,
                          out.virtual_safety_car if out is not None else pd.NA]
            sc = pair_flag(raw_states[0], raw_states[2])
            vsc = pair_flag(raw_states[1], raw_states[3])
            mixed = pair_flag(False, sc) & pair_flag(False, vsc)
            unknown = any(pd.isna(v) for v in raw_states)
            condition = 'unknown' if unknown else 'SC' if sc else 'VSC' if vsc else '正常'
            reasons = []
            if out is None:
                reasons.append('missing_next_lap')
            elif not inferred and (pd.isna(out_flag) or not bool(out_flag)):
                reasons.append('unconfirmed_source_out_lap')
            if pd.isna(stop.lap_time_s) or pd.isna(out_time):
                reasons.append('missing_lap_time')
            if not len(reference):
                reasons.append('no_clean_reference')
            usable = not reasons
            rows.append(dict(year=int(stop.year), round=int(stop['round']), circuit_id=circuit_id,
                driver=stop.driver, driver_number=int(stop.driver_number), lap=lap,
                loss_s=float(stop.lap_time_s + out_time - 2*median) if usable else pd.NA,
                under_sc=sc, under_vsc=vsc, mixed_sc_vsc=mixed, condition=condition,
                reference_n=len(reference), reference_before_n=len(before), reference_after_n=len(after),
                reference_laps=','.join(map(str, reference.lap.astype(int))),
                reference_time_s=median, usable=usable, unusable_reason='|'.join(reasons),
                pit_in_inferred=inferred, out_lap_inferred=inferred, out_lap=lap+1,
                source_out_flag=out_flag, in_lap_time_s=stop.lap_time_s, out_lap_time_s=out_time))
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.astype({name: 'boolean' for name in ['under_sc', 'under_vsc',
            'mixed_sc_vsc', 'usable', 'pit_in_inferred', 'out_lap_inferred', 'source_out_flag']})
        frame = frame.astype({name: 'Float64' for name in ['loss_s', 'reference_time_s',
                                                         'in_lap_time_s', 'out_lap_time_s']})
    return frame


def summarize_losses(pits, circuits, exclude_inferred=False):
    """Full circuit x condition grid; quantiles use pandas linear interpolation."""
    eligible = pits.loc[pits.usable]
    if exclude_inferred:
        eligible = eligible.loc[~eligible.out_lap_inferred]
    rows = []
    for circuit in sorted(circuits):
        for condition in CONDITIONS:
            losses = eligible.loc[(eligible.circuit_id == circuit) &
                                  (eligible.condition == condition), 'loss_s']
            n = len(losses)
            q1, q3 = (float(losses.quantile(.25)), float(losses.quantile(.75))) if n else (None, None)
            rows.append(dict(circuit_id=circuit, condition=condition, n=n,
                             median_s=float(losses.median()) if n else None,
                             q1_s=q1, q3_s=q3, iqr_s=q3-q1 if n else None, low_n=n < 5))
    return pd.DataFrame(rows)
