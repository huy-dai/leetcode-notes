# Surrounded Regions

Initial attempt, realized that this method doesn't work because it assumes at least one square is four-surrounded, which can't happen if it's in a multi-square region.

```py
class Solution:

    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """        
        def check_neighbors(x,y):
            """
            Return whether or not square can be captured
            """
            x_offset = [x-1,x+1]
            y_offset = [y-1,y+1]

            for x_entry in x_offset:
                if board[x_entry][y] == "O":
                    return False
            for y_entry in y_offset:
                if board[x][y_entry] == "O":
                    return False
            return True
        
        m = len(board)
        n = len(board[0])
        for i in range((n-1)*(m-1)+1):
            for x_coord in range(1,m-1):
                for y_coord in range(1,n-1):
                    if check_neighbors(x_coord,y_coord):
                        board[x_coord][y_coord] = "X"
#No need to check border
```

Better method is to realize that the only way O's can *not* be captured is if it is connected to an "O" square on the border. So we just need to run DFS/BFS from the border and mark all the squares which we know can't be captured. Then we proceed to capture all the remaining "O" squares left

```py
class Solution:

    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """   
        def dfs(x,y):
            """
            Invalidate all squares in board that can't be captured
            (by marking it as '~') that is connected to (x,y)
            """
            board[x][y] = "~" #Invalidate starting square   
            neighbors = [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]
            for x_neigh, y_neigh in neighbors:
                if 1 <= x_neigh < m-1 and 1 <= y_neigh < n-1: 
                    if board[x_neigh][y_neigh] == "O":
                        dfs(x_neigh,y_neigh) #Continue exploration

        visited = set()     
        m = len(board)
        n = len(board[0])
        
        #Left column
        for x_coord in range(m):
            if board[x_coord][0] == "O":
                dfs(x_coord,0)
        #Right column
        for x_coord in range(m):
            if board[x_coord][n-1] == "O":
                dfs(x_coord,n-1)
        #Top row
        for y_coord in range(n):
            if board[0][y_coord] == "O":
                dfs(0,y_coord)
        #Bottom row
        for y_coord in range(n):
            if board[m-1][y_coord] == "O":
                dfs(m-1,y_coord)
        
        for x_coord in range(m):
            for y_coord in range(n):
                if board[x_coord][y_coord] == "O":
                    board[x_coord][y_coord] = "X" #Safely capture it
                if board[x_coord][y_coord] == "~":
                    board[x_coord][y_coord] = "O" #Can't be captured
```

## DP - Longest Substring without Repeating Characters

```py
from functools import lru_cache

class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        @lru_cache(maxsize=None)
        def L(i): #subproblem
            if i == 0:
                return 1
            L_i_1 = L(i-1)
            start_index = i-L_i_1
            curr_substr = s[start_index:i]
            #print(i-1,curr_substr)
            if s[i] not in curr_substr:
                return L_i_1 + 1
            else:
                conflict_index = curr_substr.index(s[i]) + start_index
                return i - conflict_index
    
        highest = 0
        for i in range(len(s)):
            L_i = L(i)
            if L_i > highest:
                highest = L_i
        return highest
            
        
        
#Optimal subproblem:
#Let L(i) be length of longest substring of s[0,i] where substring ends at s[j].

#Relate:
# L(i) = 
#   L(i-1) + 1 if s[i] not in s[i-L(i-1),i-1]
#   else, if x is the index of conflicting character, then it's i-x
```
## Maximum Product Subarray

Uses the same DP technique we learn in class with the Max and Min problem

```py
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        @lru_cache(maxsize=None)
        def max_L(i):
            if i == 0:
                return nums[0]
            target = nums[i]
            if target > 0:
                return max(target,max_L(i-1) * target)
            else:
                return max(target,min_L(i-1) * target)
        @lru_cache(maxsize=None)
        def min_L(i):
            if i == 0:
                return nums[0]
            target = nums[i]
            if target > 0:
                return min(target,min_L(i-1) * target)
            else:
                return min(target,max_L(i-1) * target)
        
        highest = nums[0]
        for i in range(len(nums)):
            max_res = max_L(i)
            if max_res > highest:
                highest = max_res
        return highest
        
#Subproblems:
# Max(i) = maximum product of subarray ending at index i
# Min(i) = min     ""
# Max(i) = 
#       Max(i-1) * nums[i] if nums[i] > 0 
#       Min(i-1) * nums[i] if nums[i] < 0
#       else 0
#       Note: Need to consider when Max(i-1) or Min(i-1) is 0!
```
A much more elegant programmatic solution is this though (source: <https://leetcode.com/problems/maximum-product-subarray/discuss/2420264/Easiest-O(n)-Python-Soln-w-Explanation-or-Single-Pass-or>), which considers the fact that we can solve the problem iteratively such that at each new index we can solve both the max and min problem:

```py
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        ans=nums[0]
        maxprod,minprod=ans,ans
		
        for i in range(1,len(nums)):
            if nums[i]<0:
                maxprod,minprod=minprod,maxprod
                
            maxprod=max(nums[i],maxprod*nums[i])
            minprod=min(nums[i],minprod*nums[i])
            ans=max(ans,maxprod)
            
        return ans
```
This approach greatly reduces the memory footprint since we're not having to keep track of previous recursive results.


## Convert Sorted Array to BST

```py
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        #Step 1: Middle as root
        if len(nums) == 0:
            return None
        
        mid_i = len(nums) // 2
        
        root = TreeNode(nums[mid_i])

        #Step 2: Recursively sort the left and right side
        root.left = self.sortedArrayToBST(nums[:mid_i])
        root.right = self.sortedArrayToBST(nums[mid_i+1:])
        
        return root
```

## Inorder Tree Traversal

```py
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        out = []
        def traverse(curr:Optional[TreeNode]):
            if not curr:
                return
            
            traverse(curr.left)
            out.append(curr.val)
            traverse(curr.right)
            
        traverse(root)
        return out
```

## Adding two numbers together (represented by a linked list)

```py
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        #Initialize moving pointers
        node1 = l1
        node2 = l2
        head = None #Start of the linked list
        answer = None #End of the linked list
        carryover = False
        
        while node1 or node2 or carryover: 
            # Determine next digit value
            if not node1 and not node2:
                if not carryover:
                    break #Stop condition
                else:
                    new_val = 0 
            elif node1 and node2: 
                new_val = node1.val + node2.val
            elif node1 and not node2:
                new_val = node1.val
            else:
                new_val = node2.val
                
            if carryover: # Take into consideration carry-overs
                new_val += 1
                carryover = False #Reset carry 
            if new_val >= 10:
                carryover = True
                new_val -= 10
            
            new_digit = ListNode(new_val, None)
            if answer: #2nd digit and onwards
                answer.next = new_digit
                answer = new_digit
            else: 
                answer = ListNode(new_val, None)
                head = answer
            #Go to next digit
            if node1:
                node1 = node1.next
            if node2:
                node2 = node2.next
        return head
```

Note: Could have simplified syntax with `if else` structure on the same-line. Also when node1 or node2 is None, you can just default to their value as "0".

## Increasing Triplet Subsequence


```
Given an integer array nums, return true if there exists a triple of indices (i, j, k) such that i < j < k and nums[i] < nums[j] < nums[k]. If no such indices exists, return false.
```

Initial approach with DP:

```
class Solution:
    def increasingTriplet(self, nums: List[int]) -> bool:
        length = len(nums)
        
        @lru_cache(maxsize=None)
        def incSub(remaining:int, ref:int, index:int):
            #Whether or not an increasing subsequence of size `remaining`
            #exists in `nums` that starts at index `index` and has all of its
            #values larger than `ref`
            
            #Done: remaining is 0
            if remaining == 0:
                return True
        
            #Stop case: `index` exceeds length
            if index >= length:
                return False
            
            #Case 1: `index` is in subsequence
            if nums[index] > ref:
                # See whether we can satisfy remaining of subsequence
                if incSub(remaining-1, nums[index], index+1):
                    return True
            #Case 2: `index` not in subsequence
            if incSub(remaining, ref, index+1): #Go to the next index
                return True
            
            return False
        return incSub(3,float('-inf'),0)
```

I was hitting either the time limit or memory limit with this code. A much more efficient and simpler solution look like the following:

```py
class Solution:
    def increasingTriplet(self, nums: List[int]) -> bool:
        length = len(nums)
        
        i_th = float('inf')
        j_th = float('inf')
        
        for num in nums:
            if num < i_th:
                i_th = num
            elif num > i_th:
                j_th = min(j_th,num)
            
            if num > j_th:
                return True
        return False
```

How does this work?