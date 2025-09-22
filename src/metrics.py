# src/metrics.py
import numpy as np
def sharpe(ret, rf=0.0):
    excess = ret - rf/252
    return np.sqrt(252)*excess.mean()/excess.std()

# DSR: usar fórmula do paper (ou pacote/implementação auxiliar)
# White's Reality Check: bootstrap sobre a diferença de performance vs. benchmark
