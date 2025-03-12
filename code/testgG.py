
G = []


def swap_statements(code):
    lines = code.strip().split("\n")
    return "\n".join(reversed(lines))


def rename_variable(code, old_name, new_name):
    return code.replace(old_name, new_name)


G.append(swap_statements)
G.append(rename_variable)


for g in G:
    print(f"Function {g.__name__} is in G")