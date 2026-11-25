"""Task instruction published in the MemoryVLA paper, Table 18."""

COLORS = ("blue", "green", "pink")


def paper_instruction(order):
    first, second, third = (COLORS[index] for index in order)
    return f"Push the {first}, {second}, and {third} buttons in sequence"
