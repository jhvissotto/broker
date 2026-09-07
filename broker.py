import numpy as np
import numba as nb
from numpy import nan

spec = [
    ('posit_volume', nb.float64),
    ('posit_price', nb.float64),
    ('posit_bar', nb.int64),
    ('posit_time', nb.int64),
    ('posit_upper', nb.float64),
    ('posit_lower', nb.float64),
    ('posit_mae_chg', nb.float64),
    ('posit_mae_pct', nb.float64),

    ('trades_entry_bar', nb.int64[:]),
    ('trades_entry_time', nb.int64[:]),
    ('trades_exit_bar', nb.int64[:]),
    ('trades_exit_time', nb.int64[:]),
    ('trades_entry_price', nb.float64[:]),
    ('trades_exit_price', nb.float64[:]),
    ('trades_volume', nb.float64[:]),
    ('trades_mae_chg', nb.float64[:]),
    ('trades_mae_pct', nb.float64[:]),
    ('trades_count', nb.int64),
]

@nb.experimental.jitclass(spec)
class BROKER:

    def __init__(x, max_trades=5000):
        x.posit_volume = 0.0
        x.posit_price = nan
        x.posit_bar = -1
        x.posit_time = -1
        x.posit_upper = nan
        x.posit_lower = nan
        x.posit_mae_chg = nan
        x.posit_mae_pct = nan

        x.trades_entry_bar = np.zeros(max_trades, dtype=np.int64)
        x.trades_entry_time = np.zeros(max_trades, dtype=np.int64)
        x.trades_exit_bar = np.zeros(max_trades, dtype=np.int64)
        x.trades_exit_time = np.zeros(max_trades, dtype=np.int64)
        x.trades_entry_price = np.zeros(max_trades, dtype=np.float64)
        x.trades_exit_price = np.zeros(max_trades, dtype=np.float64)
        x.trades_volume = np.zeros(max_trades, dtype=np.float64)
        x.trades_mae_chg = np.zeros(max_trades, dtype=np.float64)
        x.trades_mae_pct = np.zeros(max_trades, dtype=np.float64)
        x.trades_count = 0

    # ---------------- abertura / gestao de posicao ----------------

    def _open_long(x, bar, time, price, volume, prc_upper=nan, prc_lower=nan):
        x.posit_volume = abs(volume)
        x.posit_price = price
        x.posit_bar = bar
        x.posit_time = time
        x.posit_upper = prc_upper
        x.posit_lower = prc_lower
        x.posit_mae_chg = 0.0
        x.posit_mae_pct = 0.0

    def _open_short(x, bar, time, price, volume, prc_upper=nan, prc_lower=nan):
        x.posit_volume = -abs(volume)
        x.posit_price = price
        x.posit_bar = bar
        x.posit_time = time
        x.posit_upper = prc_upper
        x.posit_lower = prc_lower
        x.posit_mae_chg = 0.0
        x.posit_mae_pct = 0.0

    def _partial_rise(x, bar, time, price, vol_dif, volabs_max):
        old_abs = abs(x.posit_volume)
        new_abs = min(old_abs + vol_dif, volabs_max)
        add_abs = new_abs - old_abs
        if add_abs > 0.0 and new_abs > 0.0:
            x.posit_price = (x.posit_price * old_abs + price * add_abs) / new_abs
        sign = 1.0 if x.posit_volume >= 0 else -1.0
        x.posit_volume = sign * new_abs

    def _partial_real(x, bar, time, price, vol_dif, volabs_min):
        old_abs = abs(x.posit_volume)
        new_abs = max(old_abs - vol_dif, volabs_min)
        if new_abs <= 0.0:
            x._close_position(bar, time, price)
        else:
            sign = 1.0 if x.posit_volume >= 0 else -1.0
            x.posit_volume = sign * new_abs

    def _update_posit_minmax(x, bar, time, bar_high, bar_low):
        if x.posit_volume == 0.0:
            return
        entry = x.posit_price
        vol = x.posit_volume
        sign = 1.0 if vol >= 0 else -1.0

        chg_low = (bar_low - entry) * vol
        chg_high = (bar_high - entry) * vol
        candidate_chg = min(chg_low, chg_high)
        if candidate_chg < x.posit_mae_chg:
            x.posit_mae_chg = candidate_chg

        pct_low = (bar_low - entry) / entry * sign
        pct_high = (bar_high - entry) / entry * sign
        candidate_pct = min(pct_low, pct_high) * 100.0
        if candidate_pct < x.posit_mae_pct:
            x.posit_mae_pct = candidate_pct

    def _update_trailing_upper(x, bar, time, new_upper):
        if np.isnan(new_upper):
            return
        if np.isnan(x.posit_upper) or new_upper < x.posit_upper:
            x.posit_upper = new_upper

    def _update_trailing_lower(x, bar, time, new_lower):
        if np.isnan(new_lower):
            return
        if np.isnan(x.posit_lower) or new_lower > x.posit_lower:
            x.posit_lower = new_lower

    def _close_position(x, bar, time, price):
        i = x.trades_count
        x.trades_entry_bar[i] = x.posit_bar
        x.trades_entry_time[i] = x.posit_time
        x.trades_exit_bar[i] = bar
        x.trades_exit_time[i] = time
        x.trades_entry_price[i] = x.posit_price
        x.trades_exit_price[i] = price
        x.trades_volume[i] = x.posit_volume
        x.trades_mae_chg[i] = x.posit_mae_chg
        x.trades_mae_pct[i] = x.posit_mae_pct
        x.trades_count += 1

        x.posit_volume = 0.0
        x.posit_price = nan
        x.posit_bar = -1
        x.posit_time = -1
        x.posit_upper = nan
        x.posit_lower = nan
        x.posit_mae_chg = nan
        x.posit_mae_pct = nan

    # ---------------- estado da posicao ----------------

    def _posit_has(x):
        return x.posit_volume != 0.0

    def _posit_sign_volume(x):
        sign = 1.0 if x.posit_volume >= 0 else -1.0
        return sign * x.posit_volume

    def _posit_is_long(x):
        return x.posit_volume > 0.0

    def _posit_is_short(x):
        return x.posit_volume < 0.0

    def _posit_has_upper(x):
        return not np.isnan(x.posit_upper)

    def _posit_has_lower(x):
        return not np.isnan(x.posit_lower)

    def _posit_bars_duration(x, bar):
        if x.posit_volume == 0.0:
            return 0
        return bar - x.posit_bar

    # ---------------- historico de trades ----------------

    def _trades_first_entry_bar(x):
        n = x.trades_count
        return x.trades_entry_bar[:n]

    def _trades_first_entry_time(x):
        n = x.trades_count
        return x.trades_entry_time[:n]

    def _trades_last_exit_bar(x):
        n = x.trades_count
        return x.trades_exit_bar[:n]

    def _trades_last_exit_time(x):
        n = x.trades_count
        return x.trades_exit_time[:n]

    def _trades_result_chg(x):
        n = x.trades_count
        return (x.trades_exit_price[:n] - x.trades_entry_price[:n]) * x.trades_volume[:n]

    def _trades_result_pct(x):
        n = x.trades_count
        sign = np.sign(x.trades_volume[:n])
        per_unit_chg = (x.trades_exit_price[:n] - x.trades_entry_price[:n]) * sign
        return per_unit_chg / x.trades_entry_price[:n] * 100.0

    def _trades_mae_chg(x):
        n = x.trades_count
        return x.trades_mae_chg[:n]

    def _trades_mae_pct(x):
        n = x.trades_count
        return x.trades_mae_pct[:n]
