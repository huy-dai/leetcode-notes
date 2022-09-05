# Data Structures

## Min heap 

heapq.heapify(lst) - Make a min heap
heapq.nlargest(n, iterable) - return a list of n largest elements in iterable
heapq.nsmallest(n, iterable) 


## Binary tree node.
```py
class TreeNode:
     def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
```

## Inorder tree traversal

1. Traverse left subtree
2. Traverse root
3. Traverse right subtree

