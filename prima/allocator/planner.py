from typing import Dict, List, Tuple

MB_PER_MI = 1.048576  # 1 MiB = 1.048576 MB

def mb_to_mi(x_mb: float) -> int:
    return int(round(x_mb / MB_PER_MI))


def plan(margins_mb: Dict[str, float], available_mb: float | None) -> Dict[str, float]:
    """
    Inputs:
      - margins_mb: {role: MB}
      - available_mb: available memory in MB, or None when observation fails
    Output:
      - {role: MB}; this function keeps MB internally
    """
    total_need_mb = sum(margins_mb.values())
    if total_need_mb == 0:
        return {r: 0.0 for r in margins_mb}

    if available_mb is None:
        # Fall back to the predicted demand when the memory reader is unavailable.
        return {r: float(v) for r, v in margins_mb.items()}

    alloc_mb = {r: (v / total_need_mb) * available_mb for r, v in margins_mb.items()}
    return alloc_mb


def validate_or_reduce(plan_mb: Dict[str, float], margins_mb: Dict[str, float]) -> Tuple[Dict[str, int], List[str]]:
    """
    Inputs:
      - plan_mb: {role: MB}, produced by plan()
      - margins_mb: {role: MB}, predicted demand
    Output:
      - ({role: Mi}, chosen_roles)
        MB is converted to MiB only at the return boundary.
    """
    chosen = list(plan_mb.keys())
    total_budget_mb = sum(plan_mb.values())

    while True:
        ok = all(plan_mb[r] >= margins_mb[r] for r in chosen)
        if ok:
            return {r: mb_to_mi(plan_mb[r]) for r in chosen}, chosen

        if len(chosen) == 1:
            r = chosen[0]
            return {r: mb_to_mi(margins_mb[r])}, chosen

        # Remove the role with the largest predicted demand.
        worst = max(chosen, key=lambda r: margins_mb[r])
        chosen.remove(worst)

        need_sum_mb = sum(margins_mb[r] for r in chosen)
        if need_sum_mb == 0:
            plan_mb = {r: 0.0 for r in chosen}
        else:
            plan_mb = {r: (margins_mb[r] / need_sum_mb) * total_budget_mb for r in chosen}
