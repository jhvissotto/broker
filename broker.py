import numpy as np
import numba as nb
from numpy import nan

spec = [
    ('posit_volume', nb.float64),
    ('posit_price', nb.float64),
    ('posit_bar', nb.int64),
    ('posit_time', nb.int64),
    ('posit_take', nb.float64),
    ('posit_stop', nb.float64),
    ('posit_min', nb.float64),
    ('posit_max', nb.float64),
    ('trail_take_dist', nb.float64),
    ('trail_stop_dist', nb.float64),

    ('trades_entry_bar', nb.int64[:]),
    ('trades_entry_time', nb.int64[:]),
    ('trades_exit_bar', nb.int64[:]),
    ('trades_exit_time', nb.int64[:]),
    ('trades_entry_price', nb.float64[:]),
    ('trades_exit_price', nb.float64[:]),
    ('trades_volume', nb.float64[:]),
    ('trades_min_price', nb.float64[:]),
    ('trades_max_price', nb.float64[:]),
    ('trades_count', nb.int64),
]

@nb.experimental.jitclass(spec)
class BROKER:

    def __init__(x, max_trades=5000):
        x.posit_volume = 0.0
        x.posit_price = nan
        x.posit_bar = -1
        x.posit_time = -1
        x.posit_take = nan
        x.posit_stop = nan
        x.posit_min = nan
        x.posit_max = nan
        x.trail_take_dist = nan
        x.trail_stop_dist = nan

        x.trades_entry_bar = np.zeros(max_trades, dtype=np.int64)
        x.trades_entry_time = np.zeros(max_trades, dtype=np.int64)
        x.trades_exit_bar = np.zeros(max_trades, dtype=np.int64)
        x.trades_exit_time = np.zeros(max_trades, dtype=np.int64)
        x.trades_entry_price = np.zeros(max_trades, dtype=np.float64)
        x.trades_exit_price = np.zeros(max_trades, dtype=np.float64)
        x.trades_volume = np.zeros(max_trades, dtype=np.float64)
        x.trades_min_price = np.zeros(max_trades, dtype=np.float64)
        x.trades_max_price = np.zeros(max_trades, dtype=np.float64)
        x.trades_count = 0

    # ---------------- abertura / gestao de posicao ----------------

    def _open_long(x, bar, time, price, volume, prc_take=nan, prc_stop=nan):
        x.posit_volume = abs(volume)
        x.posit_price = price
        x.posit_bar = bar
        x.posit_time = time
        x.posit_take = prc_take
        x.posit_stop = prc_stop
        x.posit_min = price
        x.posit_max = price

    def _open_short(x, bar, time, price, volume, prc_take=nan, prc_stop=nan):
        x.posit_volume = -abs(volume)
        x.posit_price = price
        x.posit_bar = bar
        x.posit_time = time
        x.posit_take = prc_take
        x.posit_stop = prc_stop
        x.posit_min = price
        x.posit_max = price

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
        sign = 1.0 if x.posit_volume >= 0 else -1.0
        x.posit_volume = sign * new_abs

    def _update_posit_minmax(x, bar, time, bar_high, bar_low):
        if np.isnan(x.posit_min) or bar_low < x.posit_min:
            x.posit_min = bar_low
        if np.isnan(x.posit_max) or bar_high > x.posit_max:
            x.posit_max = bar_high

    def _update_trailing_take(x, bar, time, bar_close):
        if np.isnan(x.trail_take_dist):
            return
        if x.posit_volume > 0:
            candidate = bar_close - x.trail_take_dist
            if np.isnan(x.posit_take) or candidate > x.posit_take:
                x.posit_take = candidate
        elif x.posit_volume < 0:
            candidate = bar_close + x.trail_take_dist
            if np.isnan(x.posit_take) or candidate < x.posit_take:
                x.posit_take = candidate

    def _update_trailing_stop(x, bar, time, bar_close):
        if np.isnan(x.trail_stop_dist):
            return
        if x.posit_volume > 0:
            candidate = bar_close - x.trail_stop_dist
            if np.isnan(x.posit_stop) or candidate > x.posit_stop:
                x.posit_stop = candidate
        elif x.posit_volume < 0:
            candidate = bar_close + x.trail_stop_dist
            if np.isnan(x.posit_stop) or candidate < x.posit_stop:
                x.posit_stop = candidate

    def _close_position(x, bar, time, price):
        i = x.trades_count
        x.trades_entry_bar[i] = x.posit_bar
        x.trades_entry_time[i] = x.posit_time
        x.trades_exit_bar[i] = bar
        x.trades_exit_time[i] = time
        x.trades_entry_price[i] = x.posit_price
        x.trades_exit_price[i] = price
        x.trades_volume[i] = x.posit_volume
        x.trades_min_price[i] = x.posit_min
        x.trades_max_price[i] = x.posit_max
        x.trades_count += 1

        x.posit_volume = 0.0
        x.posit_price = nan
        x.posit_bar = -1
        x.posit_time = -1
        x.posit_take = nan
        x.posit_stop = nan
        x.posit_min = nan
        x.posit_max = nan

    # ---------------- estado da posicao ----------------

    def _posit_has(x):
        return x.posit_volume != 0.0

    def _posit_sign_volume(x):
        if x.posit_volume > 0:
            return 1
        elif x.posit_volume < 0:
            return -1
        return 0

    def _posit_is_long(x):
        return x.posit_volume > 0.0

    def _posit_is_short(x):
        return x.posit_volume < 0.0

    def _posit_has_take(x):
        return not np.isnan(x.posit_take)

    def _posit_has_stop(x):
        return not np.isnan(x.posit_stop)

    def _posit_bars_duration(x, bar):
        if x.posit_volume == 0.0:
            return 0
        return bar - x.posit_bar

    # ---------------- historico de trades ----------------

    def _trades_first_entry_bar(x):
        return x.trades_entry_bar[0]

    def _trades_first_entry_time(x):
        return x.trades_entry_time[0]

    def _trades_result_chg(x):
        # total de pontos movimentados pela posicao inteira: preco * volume (com sinal)
        n = x.trades_count
        return (x.trades_exit_price[:n] - x.trades_entry_price[:n]) * x.trades_volume[:n]

    def _trades_result_pct(x):
        # retorno percentual e por unidade, nao depende do volume operado
        n = x.trades_count
        sign = np.sign(x.trades_volume[:n])
        per_unit_chg = (x.trades_exit_price[:n] - x.trades_entry_price[:n]) * sign
        return per_unit_chg / x.trades_entry_price[:n] * 100.0

    def _trades_mae_chg(x):
        # MAE total da posicao: MAE por unidade * volume absoluto
        n = x.trades_count
        sign = np.sign(x.trades_volume[:n])
        long_mae = x.trades_min_price[:n] - x.trades_entry_price[:n]
        short_mae = x.trades_entry_price[:n] - x.trades_max_price[:n]
        per_unit_mae = np.where(sign > 0, long_mae, short_mae)
        return per_unit_mae * np.abs(x.trades_volume[:n])

    def _trades_mae_pct(x):
        # MAE percentual e por unidade, nao depende do volume operado
        n = x.trades_count
        sign = np.sign(x.trades_volume[:n])
        long_mae = x.trades_min_price[:n] - x.trades_entry_price[:n]
        short_mae = x.trades_entry_price[:n] - x.trades_max_price[:n]
        per_unit_mae = np.where(sign > 0, long_mae, short_mae)
        return per_unit_mae / x.trades_entry_price[:n] * 100.0
