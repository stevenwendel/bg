# Example 1: Basic list containment
superlist = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
sublist = [4, 5, 6]

# Using 'in' operator
print("Using 'in' operator:", sublist in superlist)  # True

# Example 2: Lists with different order
superlist2 = [[1, 2, 3], [6, 5, 4], [7, 8, 9]]
sublist2 = [4, 5, 6]

print("Order matters:", sublist2 in superlist2)  # False

# Example 3: Using any() for more complex conditions
# Check if any sublist contains the same elements (order doesn't matter)
print("Using any() with set:", 
      any(set(sublist2) == set(x) for x in superlist2))  # True

# Example 4: Lists with duplicate elements
superlist3 = [[1, 2, 2], [3, 3, 4], [5, 6, 6]]
sublist3 = [2, 1, 2]

# Using collections.Counter for element frequency
from collections import Counter
print("Using Counter:", 
      any(Counter(sublist3) == Counter(x) for x in superlist3))  # True

# Example 5: Nested lists with different depths
superlist4 = [[1, [2, 3]], [4, [5, 6]], [7, [8, 9]]]
sublist4 = [4, [5, 6]]

print("Nested lists:", sublist4 in superlist4)  # True

# Example 6: Using numpy arrays
import numpy as np
superlist_np = [np.array([1, 2, 3]), np.array([4, 5, 6])]
sublist_np = np.array([4, 5, 6])

# For numpy arrays, we need to use array_equal
print("Numpy arrays:", 
      any(np.array_equal(sublist_np, x) for x in superlist_np))  # True 