class node:
    def __init__(self):
        self.left = None
        self.right = None
        self.val = 0

A = node()
A.val = 2
B = node()

B.val = 3
A.right = B

print(A.right.val)
