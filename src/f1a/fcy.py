"""Race-control episodes on the provider's race/leader lap counter (T009)."""
import pandas as pd

EPISODE_COLUMNS = ['type', 'start_lap', 'end_lap', 'n_laps', 'start_frac',
                   'followed_by', 'has_red_flag', 'start_time', 'end_time', 'end_reason']


def split_episodes(messages, total_laps):
    """Merge repeated deployments until a true state change; red flags interrupt.

    Both boundary laps are included: duration is touched leader laps, not seconds
    or complete laps. A direct VSC->SC transition can touch the same lap twice.
    """
    if total_laps <= 0 or not messages:
        raise ValueError('Positive total_laps and nonempty race-control feed required')
    ordered = sorted(messages, key=lambda m: pd.Timestamp(m['date']))
    episodes, active = [], None
    for message in ordered:
        text = message['message'].upper()
        state = None
        if text.startswith(('VIRTUAL SAFETY CAR DEPLOYED', 'VSC DEPLOYED')):
            state = 'VSC'
        elif text.startswith('SAFETY CAR DEPLOYED'):
            state = 'SC'
        elif message.get('scope') == 'Track':
            state = {'RED': 'RED', 'GREEN': 'GREEN', 'CLEAR': 'GREEN',
                     'CHEQUERED': 'FINISH'}.get(message.get('flag'))
        if state is None:
            # VSC ENDING and SC IN THIS LAP do not terminate the active phase.
            continue
        lap = message.get('lap_number')
        if lap is None or int(lap) < 1:
            raise ValueError(f'Missing/invalid race lap in state change: {message}')
        lap = int(lap)
        if state == (active or {}).get('type'):
            continue
        if active is not None:
            end = min(lap, total_laps)
            if end < active['start_lap']:
                raise ValueError('Race lap counter regressed during episode')
            active.update(end_lap=end, n_laps=end-active['start_lap']+1,
                          followed_by=active['type'] == 'VSC' and state == 'SC',
                          has_red_flag=state == 'RED', end_time=message['date'],
                          end_reason=state)
            episodes.append(active)
            active = None
        if state == 'FINISH':
            break
        if state in {'SC', 'VSC'}:
            if lap > total_laps:
                raise ValueError('Deployment outside race distance')
            active = dict(type=state, start_lap=lap, start_frac=lap/total_laps,
                          start_time=message['date'])
    if active is not None:
        active.update(end_lap=total_laps, n_laps=total_laps-active['start_lap']+1,
                      followed_by=False, has_red_flag=False, end_time=None,
                      end_reason='feed_end')
        episodes.append(active)
    return pd.DataFrame(episodes, columns=EPISODE_COLUMNS)


def start_bin(start_lap, total_laps):
    """Mutually exclusive paper-style six bins, with lap 1 taking precedence."""
    if not 1 <= start_lap <= total_laps:
        raise ValueError('Start lap outside race')
    if start_lap == 1:
        return '第1圈'
    frac = start_lap / total_laps
    return next(label for edge, label in [(0.2, '≤20%'), (0.4, '≤40%'),
                (0.6, '≤60%'), (0.8, '≤80%'), (1., '≤100%')] if frac <= edge)
