def sum_numbers(n):
    total = 0
    for i in range(n + 1):
        total += i
    return total

result = sum_numbers(10)
print("Sum:", result)