from math import sqrt


# Tak jak opisano tutaj: https://github.com/VainF/Torch-Pruning#high-level-pruners
# Pruning ratio to nie jest to samo co sparsity.
def calculate_pruning_ratio(for_sparsity: float) -> float:
    return 1 - sqrt(1 - for_sparsity)
