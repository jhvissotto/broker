_SPEC = [
    ('posit_volume',        nb.float64),
    ('posit_price',         nb.float64),
    ('posit_bar',           nb.int64),
    ('posit_time',          nb.int64),
    ('posit_upper',         nb.float64),
    ('posit_lower',         nb.float64),
    ('posit_ath',           nb.float64),
    ('posit_atl',           nb.float64),
    ('posit_realized_chg',  nb.float64),
    ('posit_volume_max',    nb.float64),
    ('trades_entry_bar',    nb.int64[:]),
    ('trades_entry_time',   nb.int64[:]),
    ('trades_exit_bar',     nb.int64[:]),
    ('trades_exit_time',    nb.int64[:]),
    ('trades_entry_price',  nb.float64[:]),
    ('trades_exit_price',   nb.float64[:]),
    ('trades_volume',       nb.float64[:]),
    ('trades_volume_max',   nb.float64[:]),
    ('trades_realized_chg', nb.float64[:]),
    ('trades_mae_chg',      nb.float64[:]),
    ('trades_mae_pct',      nb.float64[:]),
    ('trades_finalby',      nb.int64[:]),
    ('trades_count',        nb.int64),
]


@nb.experimental.jitclass(_SPEC)
class BROKER:
    def __init__(x, max_trades=5000):

        x.posit_volume = 0.0
        x.posit_price = nan
        x.posit_bar = -1
        x.posit_time = -1
        x.posit_upper = nan
        x.posit_lower = nan
        x.posit_ath = nan
        x.posit_atl = nan
        x.posit_realized_chg = 0.0
        x.posit_volume_max = 0.0

        x.trades_entry_bar = np.zeros(max_trades, dtype=np.int64)
        x.trades_entry_time = np.zeros(max_trades, dtype=np.int64)
        x.trades_exit_bar = np.zeros(max_trades, dtype=np.int64)
        x.trades_exit_time = np.zeros(max_trades, dtype=np.int64)
        x.trades_entry_price = np.zeros(max_trades, dtype=np.float64)
        x.trades_exit_price = np.zeros(max_trades, dtype=np.float64)
        x.trades_volume = np.zeros(max_trades, dtype=np.float64)
        x.trades_volume_max = np.zeros(max_trades, dtype=np.float64)
        x.trades_realized_chg = np.zeros(max_trades, dtype=np.float64)
        x.trades_mae_chg = np.zeros(max_trades, dtype=np.float64)
        x.trades_mae_pct = np.zeros(max_trades, dtype=np.float64)
        x.trades_finalby = np.zeros(max_trades, dtype=np.int64)
        x.trades_count = 0

    # ---------------- abertura / gestao de posicao ----------------

    def _open_long(x, bar, time, price, volume, prc_upper=nan, prc_lower=nan):
        x.posit_volume = abs(volume)
        x.posit_price = price
        x.posit_bar = bar
        x.posit_time = time
        x.posit_upper = prc_upper
        x.posit_lower = prc_lower
        x.posit_ath = price
        x.posit_atl = price
        x.posit_realized_chg = 0.0
        x.posit_volume_max = abs(volume)

    def _open_short(x, bar, time, price, volume, prc_upper=nan, prc_lower=nan):
        x.posit_volume = -abs(volume)
        x.posit_price = price
        x.posit_bar = bar
        x.posit_time = time
        x.posit_upper = prc_upper
        x.posit_lower = prc_lower
        x.posit_ath = price
        x.posit_atl = price
        x.posit_realized_chg = 0.0
        x.posit_volume_max = abs(volume)

    def _partial_rise(x, bar, time, price, vol_dif, volabs_max):
        old_abs = abs(x.posit_volume)
        new_abs = min(old_abs + vol_dif, volabs_max)
        add_abs = new_abs - old_abs
        if add_abs > 0.0 and new_abs > 0.0:
            x.posit_price = (x.posit_price * old_abs + price * add_abs) / new_abs
        sign = 1.0 if x.posit_volume >= 0 else -1.0
        x.posit_volume = sign * new_abs
        if new_abs > x.posit_volume_max:
            x.posit_volume_max = new_abs

    def _partial_real(x, bar, time, price, vol_dif, volabs_min, finalby=0):
        old_abs = abs(x.posit_volume)
        new_abs = max(old_abs - vol_dif, volabs_min)
        sold_abs = old_abs - new_abs
        sign = 1.0 if x.posit_volume >= 0 else -1.0  # calculado ANTES de qualquer mutacao

        if new_abs <= 0.0:
            # FIX: fechamento total via parcial. NAO soma sold_abs em posit_realized_chg
            # aqui -- x.posit_volume ainda guarda o lote inteiro (sign*old_abs) e e
            # exatamente esse lote que o _close_position vai aplicar via
            # (price - entry) * x.posit_volume. Somar aqui E deixar o _close_position
            # somar de novo duplicava o PnL desse ultimo lote (bug antigo).
            x._close_position(bar, time, price, finalby)
        else:
            if sold_abs > 0.0:
                x.posit_realized_chg += (price - x.posit_price) * sign * sold_abs
            x.posit_volume = sign * new_abs

    def _update_posit_minmax(x, bar, time, bar_high, bar_low):
        if x.posit_volume == 0.0:
            return
        if bar_high > x.posit_ath:
            x.posit_ath = bar_high
        if bar_low < x.posit_atl:
            x.posit_atl = bar_low

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

    def _hit_upper(x, bar, time, bar_high, finalby=0):
        # FIX: comparar em vez de sobrescrever. posit_ath ja pode ter um valor
        # maior acumulado por _update_posit_minmax em barras anteriores; o hit
        # so deve "subir" o ath se o valor clipado desta barra for maior ainda.
        clipped = min(bar_high, x.posit_upper)
        if clipped > x.posit_ath:
            x.posit_ath = clipped
        x._close_position(bar, time, x.posit_upper, finalby)

    def _hit_lower(x, bar, time, bar_low, finalby=0):
        # FIX: mesma logica do _hit_upper, espelhada para o lado de baixo.
        clipped = max(bar_low, x.posit_lower)
        if clipped < x.posit_atl:
            x.posit_atl = clipped
        x._close_position(bar, time, x.posit_lower, finalby)

    def _hit_upper_overnight(x, bar, time, bar_open, bar_high, finalby=0):
        # NOVO: bar_open ja abriu >= posit_upper (gap). A execucao ocorre no
        # proprio bar_open (pode desrespeitar/ultrapassar o prc_upper original),
        # e o ath tambem passa a ser regido por bar_open -- nao pelo bar_high,
        # pois a posicao ja fechou na abertura e o resto da barra nunca chega
        # a ser "vivido" pela posicao.
        if bar_open > x.posit_ath:
            x.posit_ath = bar_open
        x._close_position(bar, time, bar_open, finalby)

    def _hit_lower_overnight(x, bar, time, bar_open, bar_low, finalby=0):
        # NOVO: espelho do _hit_upper_overnight para o lado de baixo.
        if bar_open < x.posit_atl:
            x.posit_atl = bar_open
        x._close_position(bar, time, bar_open, finalby)

    def _close_position(x, bar, time, price, finalby=0):
        sign = 1.0 if x.posit_volume >= 0 else -1.0
        entry = x.posit_price

        mae_chg_ath = (x.posit_ath - entry) * sign
        mae_chg_atl = (x.posit_atl - entry) * sign
        mae_chg = min(mae_chg_ath, mae_chg_atl)

        mae_pct_ath = (x.posit_ath - entry) / entry * sign * 100.0
        mae_pct_atl = (x.posit_atl - entry) / entry * sign * 100.0
        mae_pct = min(mae_pct_ath, mae_pct_atl)

        i = x.trades_count
        x.trades_entry_bar[i] = x.posit_bar
        x.trades_entry_time[i] = x.posit_time
        x.trades_exit_bar[i] = bar
        x.trades_exit_time[i] = time
        x.trades_entry_price[i] = entry
        x.trades_exit_price[i] = price
        x.trades_volume[i] = x.posit_volume
        x.trades_volume_max[i] = x.posit_volume_max
        x.trades_realized_chg[i] = x.posit_realized_chg + (price - entry) * x.posit_volume
        x.trades_mae_chg[i] = mae_chg
        x.trades_mae_pct[i] = mae_pct
        x.trades_finalby[i] = finalby
        x.trades_count += 1

        x.posit_volume = 0.0
        x.posit_price = nan
        x.posit_bar = -1
        x.posit_time = -1
        x.posit_upper = nan
        x.posit_lower = nan
        x.posit_ath = nan
        x.posit_atl = nan
        x.posit_realized_chg = 0.0
        x.posit_volume_max = 0.0

    # ---------------- estado da posicao ----------------

    def _posit_has(x):
        return x.posit_volume != 0.0

    def _posit_sign_volume(x):
        return x.posit_volume

    def _posit_is_long(x):
        return x.posit_volume > 0.0

    def _posit_is_short(x):
        return x.posit_volume < 0.0

    def _posit_has_upper(x):
        return not np.isnan(x.posit_upper)

    def _posit_has_lower(x):
        return not np.isnan(x.posit_lower)

    def _posit_entry_bar(x):
        return x.posit_bar

    def _posit_prc_upper(x):
        return x.posit_upper

    def _posit_prc_lower(x):
        return x.posit_lower

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
        return x.trades_realized_chg[:n]

    def _trades_result_pct(x):
        n = x.trades_count
        return x.trades_realized_chg[:n] / (x.trades_entry_price[:n] * x.trades_volume_max[:n]) * 100.0

    def _trades_mae_chg(x):
        n = x.trades_count
        return x.trades_mae_chg[:n]

    def _trades_mae_pct(x):
        n = x.trades_count
        return x.trades_mae_pct[:n]

    def _trades_finalby(x):
        n = x.trades_count
        return x.trades_finalby[:n]
