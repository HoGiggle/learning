#!/usr/bin/python
# -*- coding: utf-8 -*-
import math
import random
import sys

class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class NumArray(object):

    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        length = len(nums)
        self.sums = [0] * (length + 1)
        for i in range(1, length + 1):
            self.sums[i] = self.sums[i - 1] + nums[i - 1]

    def sumRange(self, i, j):
        """
        :type i: int
        :type j: int
        :rtype: int
        """
        return self.sums[j + 1] - self.sums[i]


class Solution:
    def singleLink(self, visit, link, carry):
        while link is not None:
            add = link.val + carry
            carry = add / 10
            remain = add % 10
            link.val = remain
            visit.next = link
            visit = visit.next
            link = link.next
        return carry, visit

    def addTwoNumbers(self, l1, l2):
        carry = 0
        head = ListNode(0)
        visit = head
        while (l1 is not None) and (l2 is not None):
            # calculate
            add = l1.val + l2.val + carry
            carry = add / 10
            remain = add % 10
            # move
            l1.val = remain
            visit.next = l1
            visit = visit.next
            l1 = l1.next
            l2 = l2.next

        # single link move
        carry, visit = self.singleLink(visit, l1, carry)
        carry, visit = self.singleLink(visit, l2, carry)

        # final carry
        if carry > 0:
            final = ListNode(carry)
            visit.next = final
        return head.next

    def twoSum(self, nums, target):
        hash_map = {}
        for index, value in enumerate(nums, 1):
            hash_map[value] = index

        for index, value in enumerate(nums, 1):
            subValue = target - value
            if hash_map.has_key(subValue):
                if index != hash_map[subValue]:
                    return index, hash_map[subValue]

    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        if not s:
            return 0
        if len(s) <= 1:
            return len(s)
        locations = [-1 for i in range(256)]
        start = -1
        m = 0
        for i, v in enumerate(s):
            if (locations[ord(v)] > start):
                start = locations[ord(v)]
            m = max(m, i - start)
            locations[ord(v)] = i
        return m

    def divide(self, dividend, divisor):
        """
        :type dividend: int
        :type divisor: int
        :rtype: int
        """
        # special case
        MAX_INT = 2147483647
        absDividend = abs(dividend)
        absDivisor = abs(divisor)
        if absDividend < absDivisor:
            return 0
        if divisor == 0:
            return MAX_INT
        if (dividend == - MAX_INT - 1) and (divisor == -1):
            return MAX_INT

        # sign
        isNeg = False
        if dividend ^ divisor < 0:
            isNeg = True

        # bit operation
        res = 0
        i = 31
        while i >= 0:
            if (absDividend >> i) >= absDivisor:
                res += (1 << i)
                absDividend -= (absDivisor << i)
            i -= 1
        if isNeg:
            return ~res + 1
        return res

    def findPeakElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        left = 0
        right = len(nums) - 1
        while left < right:
            middle = (left + right) // 2
            if nums[middle] < nums[middle + 1]:
                left = middle + 1
            else:
                right = middle
        return left

    def maxSubArray(self, nums):
        """
        :param nums:
        :return: int
        """
        sum1, tmp = nums[0], 0
        for num in nums:
            tmp += num
            if tmp > sum1:
                sum1 = tmp
            if tmp < 0:
                tmp = 0
        return sum1

    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        res = {}
        halfLen = len(nums) // 2
        for num in nums:
            if num in res:
                res[num] = res[num] + 1
            else:
                res[num] = 1
            if res[num] > halfLen:
                return num
        return 0

    def middle(self, nums, start, end):
        """
        :param nums:
        :param start:
        :param end:
        :return: index
        """
        i, j, tmp = start, end, nums[start]
        while i < j:
            while (i < j) and (tmp >= nums[j]):
                j = j - 1
            nums[i] = nums[j]
            while (i < j) and (tmp <= nums[i]):
                i = i + 1
            nums[j] = nums[i]
        nums[i] = tmp
        return i

    def swap(self, nums, i, j):
        tmp = nums[i]
        nums[i] = nums[j]
        nums[j] = tmp

    def shuffle(self, nums):
        size = len(nums)
        for index, num in enumerate(nums):
            rand = random.randint(0, size - 1)
            self.swap(nums, index, rand)

    def findKthLargest(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        self.shuffle(nums)
        middle = self.middle(nums, 0, len(nums) - 1)
        local = middle + 1
        if local == k:
            return nums[middle]
        elif local < k:
            return self.findKthLargest(nums[local:], k - local)
        else:
            return self.findKthLargest(nums[:middle], k)

    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        if (not matrix) or (len(matrix) < 1) or (len(matrix[0]) < 1):
            return False
        m, n = len(matrix), len(matrix[0])
        if (target < matrix[0][0]) or (target > matrix[m - 1][n - 1]):
            return False

        row, col = 0, n - 1
        while (row < m) and (col >= 0):
            if matrix[row][col] == target:
                return True
            elif matrix[row][col] > target:
                col = col - 1
            else:
                row = row + 1
        return False

    def diffWaysToCompute(self, input):
        """
        :type input: str
        :rtype: List[int]
        """
        return [a + b if c == '+' else a - b if c == '-' else a * b for i, c in enumerate(input) if c < '0'
                for a in self.diffWaysToCompute(input[:i])
                for b in self.diffWaysToCompute(input[i + 1:])] or [int(input)]

    def countRangeSum(self, nums, lower, upper):
        """
        :type nums: List[int]
        :type lower: int
        :type upper: int
        :rtype: int
        """
        sums = [0] * (len(nums) + 1)
        for i in range(1, len(sums)):
            sums[i] = sums[i - 1] + nums[i - 1]
        return self.countMergeSort(sums, lower, upper, 0, len(sums))

    def countMergeSort(self, sums, lower, upper, start, end):
        if end - start <= 1:
            return 0
        mid = (start + end) // 2
        count = self.countMergeSort(sums, lower, upper, start, mid) \
                + self.countMergeSort(sums, lower, upper, mid, end)

        idx, cur, arr = 0, mid, [m for m in range(end - start)]
        for i in range(start, mid):
            j = mid
            while (j < end) and (sums[j] - sums[i] < lower):
                j += 1
            k = j
            while (k < end) and (sums[k] - sums[i] <= upper):
                k += 1
            while (cur < end) and (sums[cur] < sums[i]):
                arr[idx] = sums[cur]
                cur += 1
                idx += 1

            arr[idx] = sums[i]
            idx += 1
            count += k - j

        if cur != mid:
            for i in range(cur - start):
                sums[start + i] = arr[i]
        return count

    def divisorGame(self, N):
        """
        :type N: int
        :rtype: bool

        dp[n] = !(dp[n-i] && dp[n-n/i]), i为n的因子
        """
        if N == 1:
            return False
        result = [False] * (N + 1)
        for n in range(2, N + 1):
            sqrtN = int(math.sqrt(n)) + 1
            for i in range(1, sqrtN): #求因子
                if (n % i == 0) and ((not result[n - i]) or (i > 1 and not result[n - n / i])):
                    result[n] = True
                    break
        return result[N]

    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int

        profit[i] = prices[i] - minPrice[i-1]
        """
        # [7, 1, 5, 3, 6, 4]
        if len(prices) < 2:
            return 0
        minV, profit = prices[0], 0
        for i in range(1, len(prices)):
            if (prices[i] - minV) > profit:
                profit = prices[i] - minV
            if prices[i] < minV:
                minV = prices[i]
        return profit

    def minCostClimbingStairs(self, cost):
        """
        :type cost: List[int]
        :rtype: int

        dp[i] = min(dp[i-1], dp[i-2]) + cost[i]
        """
        if len(cost) == 2:
            return min(cost[0], cost[1])
        n_1, n_2 = cost[1], cost[0]
        for i in range(2, len(cost)):
            tmp = min(n_1, n_2) + cost[i]
            n_2, n_1 = n_1, tmp
        return min(n_1, n_2)

    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int

        dp[i] = dp[i-1] + dp[i-2]
        """
        if n <= 2:
            return n

        n_2, n_1 = 1, 2
        for i in range(2, n):
            tmp = n_2 + n_1
            n_2, n_1 = n_1, tmp
        return n_1

    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int

        dp[i]表示以index=i为结尾的最大连续子串
        dp[i] = max(0, dp[i-1]) + nums[i]
        """
        # [-2, 1, -3, 4, -1, 2, 1, -5, 4]
        middleMax, finalMax = nums[0], nums[0]
        for i in range(1, len(nums)):
            middleMax = nums[i] + max(0, middleMax)
            finalMax = max(middleMax, finalMax)
        return finalMax

    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int

        dp[i] = max(dp[i-2] + nums[i], dp[i-1])
        """
        # [2,7,9,3,1]
        pre, cur = 0, 0
        for num in nums:
            tmp = max(pre + num, cur)
            pre, cur = cur, tmp
        return cur

    def stoneGame(self, piles):
        # dp[i, j]表示i->j石堆, 先选的最大收益
        # dp[i, j] = max(左边选收益, 右边选收益)
        n = len(piles)
        dp = [[0] * n for i in range(n)]
        for i in range(n): dp[i][i] = piles[i]
        for d in range(1, n):  #d为剩余石堆长度
            for i in range(n - d): #先填充短距离的石堆收益
                dp[i][i + d] = max(piles[i] - dp[i + 1][i + d], piles[i + d] - dp[i][i + d - 1])
        return dp[0][-1] > 0

    def isSubsequence(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool

        长串、子串上分别记录游标, 长串游标后移, 如果两串字符相同, 子串游标后移, 如子串游标到达最后, 则为True
        """
        if len(s) > len(t):
            return False
        if len(s) == 0:
            return True

        sIndex, sLen, tLen = 0, len(s), len(t)
        for tIndex, tc in enumerate(t):
            if sIndex < sLen and tc == s[sIndex]:
                sIndex = sIndex + 1
                if sIndex == sLen:
                    return True
            elif (sLen - sIndex) > (tLen - tIndex):
                return False
        return False

    def stoneGameII(self, piles):
        """
        :type piles: List[int]
        :rtype: int
        """

    def countBits(self, num):
        """
        :type num: int
        :rtype: List[int]
        """
        """
          dp[i] = dp[i-1] + 1, i奇数; dp[i/2], i偶数
        """
        res = [0] * (num + 1)
        for i in range(1, num + 1):
            if i % 2 == 1:
                res[i] = res[i - 1] + 1
            else:
                res[i] = res[i / 2]
        return res

    def minFallingPathSum(self, A):
        """
        :type A: List[List[int]]
        :rtype: int

        dp[i, j] = min(dp[i-1, j-1], dp[i-1, j], dp[i-1, j+1]) + A[i, j]
        """
        length = len(A)
        dp = [[0]*length for _ in range(length)]
        for i in range(length):
            dp[0][i] = A[0][i]

        for i in range(1, length):
            for j in range(length):
                minV = dp[i-1][j]
                if (j - 1) >= 0:
                    minV = min(minV, dp[i-1][j-1])
                if (j + 1) < length:
                    minV = min(minV, dp[i-1][j+1])
                dp[i][j] = minV + A[i][j]

        return min(dp[length-1])
        # length = len(A)
        # dp, dp1 = [i for i in A[0]], [0]*length
        # for i in range(1, length):
        #     for j in range(length):
        #         minV = dp[j]
        #         if (j - 1) >= 0:
        #             minV = min(minV, dp[j-1])
        #         if (j + 1) < length:
        #             minV = min(minV, dp[j+1])
        #         dp1[j] = minV + A[i][j]
        #     dp = dp1[:]
        # return min(dp)

    def longestCommonSubsequence(self, text1, text2):
        """
        :type text1: str
        :type text2: str
        :rtype: int

        dp[i, j] = dp[i-1, j-1] + 1, if text1[i] == text2[j]
                 = max(dp[i, j-1], dp[i-1, j]), if text1[i] != text2[j]
        """
        len1, len2 = len(text1), len(text2)
        if len1 == 0 or len2 == 0:
            return 0
        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
        for i in range(len1):
            for j in range(len2):
                if text1[i] == text2[j]:
                    dp[i+1][j+1] = dp[i][j] + 1
                else:
                    dp[i+1][j+1] = max(dp[i+1][j], dp[i][j+1])
        return dp[len1][len2]

    def mctFromLeafValues(self, arr):
        """
        :type arr: List[int]
        :rtype: int

        dp[i, j] = dp[i, k] + dp[k + 1, j] + max(A[i, k]) * max(A[k + 1, j])
        时间O(n^3), 空间O(n^2)
        """
        """
        # 时间O(n), 空间O(n)。n个叶节点, 生成n-1非叶节点; 一个中间叶节点可以和前、后任一节点结合
        res, n = 0, len(A)
        stack = [float('inf')]
        for a in A:
            while stack[-1] <= a:
                mid = stack.pop()
                res += mid * min(stack[-1], a)
            stack.append(a)
        while len(stack) > 2:
            res += stack.pop() * stack[-1]
        return res
        """

        length = len(arr)
        maxV = [[0]*(length+1) for _ in range(length)]
        for i in range(length):
            for j in range(i + 1, length + 1):
                maxV[i][j] = max(maxV[i][j-1], arr[j-1])

        dp = [[0]*length for _ in range(length)]
        for d in range(1, length):
            for i in range(length - d):
                dp[i][i+d] = sys.maxint
                for k in range(i, i + d):
                    dp[i][i+d] = min(dp[i][i+d], dp[i][k] + dp[k+1][i+d] + maxV[i][k+1] * maxV[k+1][i+d+1])
        return dp[0][length-1]


    def countSubstrings(self, s):
        """
        :type s: str
        :rtype: int

        dp[i][j]表示起止index为i、j的子串是否为回文串
        dp[i][j] = dp[i-1][j-1], if s[i]
        """
        length = len(s)
        if length <= 1:
            return length

        dp = [[0]*length for _ in range(length)]
        for i in range(length):
            dp[i][i] = 1

        count = 0
        for d in range(1, length):
            for i in range(length - d):
                if s[i] == s[i+d]:
                    dp[i][i+d] = 1 if d == 1 else dp[i+1][i+d-1]
                    count += dp[i][i+d]
        return count

    def mincostTickets(self, days, costs):
        """
        :type days: List[int]
        :type costs: List[int]
        :rtype: int

        dp[i]表示第i天的最低支出
        dp[i] = min(dp[i-1] + costs[0], dp[i-7] + costs[1], dp[i-30] + costs[2])
        """
        length = days[-1] + 1
        dp = [-1] * length
        for i in days:
            dp[i] = 0
        dp[0] = 0

        for i in range(1, length):
            if dp[i] < 0:
                dp[i] = dp[i-1]
            else:
                dp[i] = min(dp[i-1] + costs[0], dp[max(0, i-7)] + costs[1], dp[max(0, i-30)] + costs[2])
        return dp[length - 1]

    def numberOfArithmeticSlices(self, A):
        """
        :type A: List[int]
        :rtype: int
        tips：
        1）dp[i][j], 判断slice[i, j]是否为等差数列, 如果slice[i, j]不是, 则slice[i, j+1]及之后都不是
        2）dp[i], 判断以i结束的数列是否为等差数列, 等差数列数量 = 数列长度 - 2
        """
        # # tip 1
        # length = len(A)
        # if length <= 2:
        #     return 0
        # count = 0
        # for i in range(length - 2):
        #     for j in range(i + 2, length):
        #         if A[j] - A[j-1] == A[j-1] - A[j-2]:
        #             count += 1
        #         else:
        #             break
        # return count

        # tip 2
        length = len(A)
        if length <= 2:
            return 0
        count, seq_len = 0, 2
        for i in range(2, length):
            if A[i] - A[i-1] == A[i-1] - A[i-2]:
                seq_len += 1
                count += seq_len - 2
            else:
                seq_len = 2
        return count

    def minimumDeleteSum(self, s1, s2):
        """
        :type s1: str
        :type s2: str
        :rtype: int
        """
        len1, len2 = len(s1) + 1, len(s2) + 1
        dp = [[0] * len2 for _ in range(len1)]
        for i in range(len1-1):
            dp[i+1][0] = dp[i][0] + ord(s1[i])
        for i in range(len2-1):
            dp[0][i+1] = dp[0][i] + ord(s2[i])
        for i in range(1, len1):
            for j in range(1, len2):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i - 1][j - 1] + ord(s1[i - 1]) + ord(s2[j - 1]), dp[i][j - 1] + ord(s2[j - 1]),
                                   dp[i - 1][j] + ord(s1[i - 1]))
        return dp[-1][-1]

    def findDisappearedNumbers(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        标记为负值说明位置有值
        """
        for i in xrange(len(nums)):
            index = abs(nums[i]) - 1
            if nums[index] > 0:
                nums[index] = -nums[index]
        return [i + 1 for i in xrange(len(nums)) if nums[i] > 0]

    def mergeTrees(self, t1, t2):
        """
        :type t1: TreeNode
        :type t2: TreeNode
        :rtype: TreeNode
        """
        if t1 is not None and t2 is not None:
            t1.val += t2.val
            t1.left = self.mergeTrees(t1.left, t2.left)
            t1.right = self.mergeTrees(t1.right, t2.right)
        return t1 if t1 is not None else t2


    def diameterOfBinaryTree(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        self.ans = 0

        def depth(p):
            if not p: return 0
            left, right = depth(p.left), depth(p.right)
            self.ans = max(self.ans, left + right)
            return 1 + max(left, right)

        depth(root)
        return self.ans


    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        def isEqual(l, r):
            """
            :type l: TreeNode
            :type r: TreeNode
            :rtype: bool
            """
            if l is None and r is None: return True
            if l is None or r is None: return False
            return (l.val == r.val) and isEqual(l.left, r.right) and isEqual(l.right, r.left)
        if root is None: return True
        return isEqual(root.left, root.right)

    def isPalindrome(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        slow, fast = head, head
        while fast is not None and fast.next is not None:
            slow = slow.next
            fast = fast.next.next

        if fast is not None: slow = slow.next # the length of list is odd

        # reverse [half, length]
        mid_head, mid_next = ListNode(-1), None
        while slow is not None:
            mid_head.next = slow
            slow = slow.next
            mid_head.next.next = mid_next
            mid_next = mid_head.next

        while mid_head.next is not None:
            if mid_head.next.val != head.val:
                return False
            mid_head = mid_head.next
            head = head.next
        return True












if __name__ == '__main__':
    s = Solution()
    # print s.findPeakElement([1, 2])
    # print(s.findKthLargest([3,2,1,5,6,4], 2))

    root = TreeNode(1)
    root.left = TreeNode(2)
    root.left.left = TreeNode(2)
    root.right = TreeNode(2)
    root.right.left = TreeNode(2)
    print s.isSymmetric(root)
