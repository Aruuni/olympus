"""Multi-agent TD3 learner used by the Astraea training recreation.

Multi-flow CTDE: parameter-shared deterministic actor on each flow's stacked
local state, twin centralized critics that additionally see the 12-value
global state (paper Table 2), trained on the global convergence reward
(paper Eq. 8) assembled in the learner.
"""

MULTI_AGENT = True
