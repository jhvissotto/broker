import numpy as np
import numba as nb
from numpy import nan

BK_FINAL_NOID       = 0  # as flags nao devem usadas para mudar/rotear o comportamento no código, ela são marcadores para análises de dados apenas
BK_FINAL_PARTIAL    = 1
BK_FINAL_SIGNAL     = 2
BK_FINAL_DURAT      = 3
BK_FINAL_UPPER      = 4
BK_FINAL_LOWER      = 5
BK_FINAL_UPPER_ON   = 6
BK_FINAL_LOWER_ON   = 7
BK_FINAL_FLAT       = 8

@nb.experimental.jitclass
class ASSIGN_BROKER():
    def __init__(x, max_trades=5000): ... # max_trades nao é requisito, é alocação de memória apenas
    
    def _open_long(x,  bar, time, price, volume, prc_upper=nan, prc_lower=nan)  -> None: ...  # prc_upper = long_take  e  prc_lower = short_stop
    def _open_short(x, bar, time, price, volume, prc_upper=nan, prc_lower=nan)  -> None: ...  # prc_upper = long_stop  e  prc_lower = short_take
    def _partial_rise(x, bar, time, price, vol_dif, volabs_max)                 -> None: ...
    def _partial_real(x, bar, time, price, vol_dif, volabs_min, finalby=0)      -> None: ...  # se volabs_min for 0, entao a parcial pode fechar a posicao
    def _update_posit_minmax(x, bar, time, bar_high, bar_low)                   -> None: ...  # atualiza ath e atl da posicao
    def _update_trailing_upper(x, bar, time, new_upper)                         -> None: ...  # pode apenas descer
    def _update_trailing_lower(x, bar, time, new_lower)                         -> None: ...  # pode apenas subir
    def _hit_upper(x, bar, time, bar_high, finalby=0)                           -> None: ...  # ath da posicao não é o bar_high, ath é clipado por prc_upper
    def _hit_lower(x, bar, time, bar_low,  finalby=0)                           -> None: ...  # atl da posicao não é o bar_low,  atl é clipado por prc_lower
    def _hit_upper_overnight(x, bar, time, bar_open, bar_high, finalby=0)       -> None: ...  # bar_open da primeira barra do dia pode abrir muito acima  do prc_upper, hitando e fechando a posicao imediamente,  podendo ath da posicao ficar acima  de prc_upper  
    def _hit_lower_overnight(x, bar, time, bar_open, bar_low,  finalby=0)       -> None: ...  # bar_open da primeira barra do dia pode abrir muito abaixo do prc_lower, hitando e fechando a posicao imediamente,  podendo atl da posicao ficar abaixo de prc_lower
    def _close_position(x, bar, time, price, finalby=0)                         -> None: ...  

    def _posit_has(x)               ->  bool: ...
    def _posit_is_long(x)           ->  bool: ...
    def _posit_is_short(x)          ->  bool: ...
    def _posit_has_upper(x)         ->  bool: ...
    def _posit_has_lower(x)         ->  bool: ...
    def _posit_entry_bar(x)         ->   int: ...  
    def _posit_sign_volume(x)       ->   int: ...  # (sinal da posicao) x (volume da posicao)
    def _posit_prc_upper(x)         -> float: ...  # long_take ou short_stop  (sera usado para checar se a barra hitou prc_upper)
    def _posit_prc_lower(x)         -> float: ...  # long_stop ou short_take  (sera usado para checar se a barra hitou prc_lower)

    def _trades_first_entry_bar(x)  -> np.ndarray: ...  # em caso de aumentos parciais, a primeira entrada de cada trade é a mandatória
    def _trades_first_entry_time(x) -> np.ndarray: ...  # em caso de aumentos parciais, a primeira entrada de cada trade é a mandatória
    def _trades_last_exit_bar(x)    -> np.ndarray: ...  # em caso de realizacoes parciais, a ultima saida de cada trade é a mandatória
    def _trades_last_exit_time(x)   -> np.ndarray: ...  # em caso de realizacoes parciais, a ultima saida de cada trade é a mandatória
    def _trades_result_chg(x)       -> np.ndarray: ...  
    def _trades_result_pct(x)       -> np.ndarray: ...  
    def _trades_mae_chg(x)          -> np.ndarray: ...  # a referência não é o menor preço, é o maior prejuizo não realizado 
    def _trades_mae_pct(x)          -> np.ndarray: ...  # a referência não é o menor preço, é o maior prejuizo não realizado
    def _trades_finalby(x)          -> np.ndarray: ...  # tags indentificando o motivo do fechamento do trade