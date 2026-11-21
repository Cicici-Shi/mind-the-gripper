#!/bin/bash
# maskinfo-pi parameters: the single source read by every job (eval "$(tools/mi_pi_params.sh)").
# Values follow astra_maskinfo_pi_rulings_2026-11-08.md; the ones marked FREEZE are placeholders to be confirmed
# when protocol_maskinfo_pi.md v1 is frozen (they are configurable, nothing else hard-codes them).
#   MI_QUERY_EVERY 5  : pi0 / pi0.5 replan 5 (pi0 chunk 50, pi0.5 chunk 10, first 5 executed)
#   MI_LAG_STEPS  15  : ruling 3 (three query intervals, ~0.8 s); becomes 20 if the cached-state agentview IoU
#                       median > 0.7 in either suite (FREEZE: set after the 5-step manipulation check)
#   MI_OUTLINE_W   3  : ruling 4 (raw 3 px, "1 inner 2 outer", not re-tuned per model)
#   MI_PROTECT interactive, MI_PSET lists/mi_pset.json (7167fef3...), MI_PREP openpi
#   MI_NOISE_MASTER 5478 : astra_three_protocols_rulings_2026-11-08 §1 item 3 (common random numbers across cells)
#   MEMFRAC 0.3       : JAX preallocation on the shared h200 (task item 5)
echo "export MI_QUERY_EVERY=5 MI_LAG_STEPS=15 MI_OUTLINE_W=3 MI_PROTECT=interactive MI_PREP=openpi MI_NOISE_MASTER=5478 MEMFRAC=0.3"
