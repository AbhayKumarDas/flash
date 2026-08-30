## Before using this corpus

1. The placement mode split is close to `PLACEMENT_MIX`. A large skew means adaptive placement is
   failing its containment search and falling back.
2. Few samples routed to alpha. A large fraction means the bank's defects are mostly too small for
   Poisson blending, so the corpus does not exercise it.
3. The dissolved count is low.
4. The seam check in section 9 passes by eye. No metric here catches a shortcut feature.
5. `GEN_SECONDS` is your measured Module 1 cost, not the placeholder.
