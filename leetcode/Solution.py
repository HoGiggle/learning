#!/usr/bin/python
# -*- coding: utf-8 -*-
import math
import random
import sys
import numpy as np
from collections import defaultdict,Counter


class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Node(object):
    def __init__(self, val, next, random):
        self.val = val
        self.next = next
        self.random = random


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


class MedianFinder(object):

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.size = 0
        self.data = [0]*1000
        self.count = [0]*1000

    def addNum(self, num):
        """
        :type num: int
        :rtype: None
        """
        self.data[num] += 1
        self.count[num] += 1
        self.size += 1




    def findMedian(self):
        """
        :rtype: float
        """


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

    def minCostClimbingStairs(self, cost):
        """
        :type cost: List[int]
        :rtype: int
        """
        lenth = len(cost)
        if lenth == 1:
            return cost[0]
        if lenth == 2:
            return cost[1]
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

    # def maxProfit(self, prices):
    #     """
    #     :type prices: List[int]
    #     :rtype: int
    #
    #     profit[i] = prices[i] - minPrice[i-1]
    #     """
    #     # [7, 1, 5, 3, 6, 4]
    #     if len(prices) < 2:
    #         return 0
    #     minV, profit = prices[0], 0
    #     for i in range(1, len(prices)):
    #         if (prices[i] - minV) > profit:
    #             profit = prices[i] - minV
    #         if prices[i] < minV:
    #             minV = prices[i]
    #     return profit

    # def maxProfit(self, prices):
    #     """
    #     :type prices: List[int]
    #     :rtype: int
    #     """
    #     buy, profit = sys.maxint, 0
    #     for price in prices:
    #         if price - buy > 0:
    #             profit += price - buy
    #         buy = price
    #     return profit
    #
    #     profit = 0
    #     for i in range(len(prices) - 1):
    #         profit += max(0, prices[i+1] - prices[i])
    #     return profit

    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        def helper(prices, s, e):
            if e - s <= 0:
                return 0
            if e - s == 1:
                return max(prices[e] - prices[s], 0)
            profit = 0
            for i in (s+1, e):
                left = helper(prices, s, i-1)
                right = helper(prices, i+1, e)
                if left + right > profit:
                    profit = left + right
            return profit
        return helper(prices, 0, len(prices)-1)



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

    def rob1(self, nums):
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


    def probabilityRecall_bin(self, data):
        """
        累计概率 + 二分查找
        :param data: List[List[String, Int]]
        :return:
        """
        # 累计概率
        sumdata = [0] * (len(data) + 1)
        sumdata[0] = 0
        for i in range(0, len(data)):
            sumdata[i + 1] = sumdata[i] + data[i][1]

        # 随机值
        rand = random.randint(1, sumdata[-1])
        start, end = 0, len(sumdata) - 1
        print "rand = %i" % rand
        while start < end:
            mid = (start + end) // 2
            if (rand <= sumdata[mid]) and (rand > sumdata[mid - 1]):
                return data[mid - 1][0]
            elif rand > sumdata[mid]:
                start = mid + 1
            else:
                end = mid - 1
        return data[start - 1][0]

    def probabilityRecall_counting(self, data):
        return

    def findUnsortedSubarray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # # tip1：两个方向, 头部小于后面最小值, 尾部大于前面最大值
        # if nums is None or len(nums) <= 1:
        #     return 0
        #
        # length = len(nums)
        # min_list, max_list = [nums[length-1]]*length, [nums[0]]*length
        # for i in xrange(1, length):
        #     max_list[i] = max(nums[i], max_list[i-1])
        #     min_list[length-i-1] = min(nums[length-i-1], min_list[length-i])
        #
        # count = 0
        # for i in xrange(length):
        #     if nums[i] > min_list[i]:
        #         break
        #     count += 1
        # for i in xrange(length):
        #     if nums[length-i-1] < max_list[length-i-1]:
        #         break
        #     count += 1
        # return length-count if count < length else 0

        # tip2：找到两个位置, 前面第一个大于后面的index, 后面第一个小于前面的index
        if nums is None or len(nums) <= 1:
            return 0

        length, start, end = len(nums), -1, -2
        max_v, min_v = nums[0], nums[length-1]
        for i in xrange(1, length):
            max_v = max(max_v, nums[i])
            min_v = min(min_v, nums[length-i-1])
            if nums[i] < max_v: end = i
            if nums[length-i-1] > min_v: start = length - i - 1
        return end - start + 1

    def pathSum(self, root, sum):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: int
        """
        def helper(p, s, ss, list):
            if p is None:
                return 0
            list1 = list[:]
            list1.append(p.val)

            if p.val == s:
                print list1
            t = 0
            if p.left is not None:
                t += helper(p.left, s - p.val, ss, list1)
                t += helper(p.left, ss, ss, [])

            if p.right is not None:
                t += helper(p.right, s - p.val,ss, list1)
                t += helper(p.right, ss, ss, [])
            return (1 if p.val == s else 0) + t
        helper(root, sum, sum, [])

        # if root is None:
        #     return 0
        # return (1 if root.val == sum else 0) \
        #        + self.pathSum(root.left, sum - root.val) + self.pathSum(root.right, sum - root.val) \
        #        + self.pathSum(root.left, sum) + self.pathSum(root.right, sum)

    def reconstructQueue(self, people):
        """
        :type people: List[List[int]]
        :rtype: List[List[int]]
        小值往前加不影响大值k
        """
        people.sort(key=lambda x: (-x[0], x[1]))
        que = []
        for p in people:
            que.insert(p[1], p)
        return que

    def dailyTemperatures(self, T):
        """
        :type T: List[int]
        :rtype: List[int]
        """
        # T = [73, 74, 75, 71, 69, 72, 76, 73]
        # [1, 1, 4, 2, 1, 1, 0, 0]

        # # 1. 暴力法
        # length = len(T)
        # res = [0] * length
        # for i in xrange(length-1):
        #     for j in xrange(i+1, length):
        #         if T[i] < T[j]:
        #             res[i] = j - i
        #             break
        # return res

        # 2. 栈实现
        res, stack, top = [0] * len(T), [0] * len(T), -1
        for i in xrange(len(T)):
            while top > -1 and (T[i] > T[stack[top]]):
                idx = stack[top]
                res[idx] = i - idx
                top -= 1
            top += 1
            stack[top] = i
        return res

    def topKFrequent(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        # 堆排序
        freq_dic = defaultdict(list)
        for key, freq in Counter(nums).items():
            freq_dic[freq].append(key)

        res = []
        for freq in range(len(nums), 0, -1):
            res.extend(freq_dic[freq])
            if len(res) >= k: return res[:k]
        return res[:k]


    def fastSort(self, nums):
        def middle(list, start, end):
            l, r, tmp = start, end, list[start]
            while r > l:
                while r > l and list[r] >= tmp: r -= 1
                list[l] = list[r]
                while r > l and list[l] <= tmp: l += 1
                list[r] = list[l]
            list[l] = tmp
            return l

        def sort(list, start, end):
            if start > end:
                return
            mid = middle(list, start, end)
            sort(list, start, mid - 1)
            sort(list, mid + 1, end)
            return list

        return sort(nums, 0, len(nums) - 1)

    def rangeSumBST(self, root, L, R):
        """
        :type root: TreeNode
        :type L: int
        :type R: int
        :rtype: int
        """
        def helper(p, l, r):
            if p is None: return 0
            if p.val <= r and p.val >= l:
                return helper(p.left, l, r) + helper(p.right, l, r) + p.val
            elif p.val < l:
                return helper(p.right, l, r)
            else:
                return helper(p.left, l, r)
        return helper(root, L, R)

    def inorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        def helper(p, list):
            if p is None: return
            helper(p.left, list)
            list.append(p.val)
            helper(p.right, list)
        res = []
        helper(root, res)
        return res

    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        def backtrack(nums, length, res, tmp_list, tmp_set):
            if len(tmp_list) == length:
                res.append(tmp_list[:])
            else:
                for n in nums:
                    if tmp_set.has_key(n): continue
                    tmp_list.append(n)
                    tmp_set[n] = None
                    backtrack(nums, length, res, tmp_list, tmp_set)
                    tmp_list.pop()
                    tmp_set.pop(n)

        res = []
        backtrack(nums, len(nums), res, [], {})
        return res

    def generateParenthesis(self, n):
        """
        :type n: int
        :rtype: List[str]
        """

    def findDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        def helper(nums, length, res, tmp_list, start):
            res.append(tmp_list[:])
            for i in xrange(start, length):
                tmp_list.append(nums[i])
                helper(nums, length, res, tmp_list, i+1)
                tmp_list.pop()
        res = []
        helper(nums, len(nums), res, [], 0)
        return res

    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        slow, fast, isCycle = head, head, False
        while slow is not None and fast is not None:
            slow = slow.next
            if fast.next is None: return None
            fast = fast.next.next
            if slow == fast:
                isCycle = True
                break

        if not isCycle: return None
        slow = head
        while slow != fast:
            slow = slow.next
            fast = fast.next
        return slow

    def findDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # # binary search, O(N * logN)
        # if len(nums) <= 1: return -1
        #
        # low, high = 0, len(nums)-1
        # while low < high:
        #     middle, count = (low + high)//2, 0
        #     for num in nums:
        #         if num <= middle:
        #             count += 1
        #     if count > middle:
        #         high = middle
        #     else:
        #         low = middle + 1
        # return low

        # slow fast pointer
        if len(nums) <= 1: return -1
        slow, fast = nums[0], nums[nums[0]]
        while slow != fast:
            slow = nums[slow]
            fast = nums[nums[fast]]

        slow = 0
        while slow != fast:
            slow = nums[slow]
            fast = nums[fast]
        return slow

    def rotate(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: None Do not return anything, modify matrix in-place instead.
        """
        def swap(matrix, r1, c1, r2, c2):
            tmp = matrix[r1][c1]
            matrix[r1][c1] = matrix[r2][c2]
            matrix[r2][c2] = tmp

        N = len(matrix)
        for row in range(N // 2):
            for col in range(row, N - row - 1):
                swap(matrix, row, col, col, N-row-1)
                swap(matrix, row, col, N-row-1, N-col-1)
                swap(matrix, row, col, N-col-1, row)

    def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """
        prime_list = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101]
        res_map = {}
        for str in strs:
            value = 1
            for i in range(len(str)):
                value *= prime_list[ord(str[i])-97]
            if res_map.has_key(value):
                res_map[value].append(str)
            else:
                res_map[value] = [str]

        res_list = []
        for str_list in res_map.values():
            res_list.append(str_list)
        return res_list

    def uniquePaths(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        # dp(m, n) = dp(m, n-1) + dp(m-1, n)
        dp = [[0]*n for _ in range(m)]
        for i in range(m):
            dp[i][0] = 1
        for j in range(n):
            dp[0][j] = 1

        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i][j-1] + dp[i-1][j]
        return dp[m-1][n-1]

    def rob(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """

        # if root is None:
        #     return 0
        # dp1, dp2 = root.val, 0
        # if root.left is not None:
        #     dp2 += self.rob(root.left)
        #     if root.left.left is not None:
        #         dp1 += self.rob(root.left.left)
        #     if root.left.right is not None:
        #         dp1 += self.rob(root.left.right)
        #
        # if root.right is not None:
        #     dp2 += self.rob(root.right)
        #     if root.right.left is not None:
        #         dp1 += self.rob(root.right.left)
        #     if root.right.right is not None:
        #         dp1 += self.rob(root.right.right)
        #
        # return max(dp1, dp2)

        # def helper(root, dic):
        #     if root is None: return 0
        #     if dic.has_key(root): return dic[root]
        #
        #     res = 0
        #     if root.left is not None:
        #         res += (helper(root.left.left, dic) + helper(root.left.right, dic))
        #     if root.right is not None:
        #         res += (helper(root.right.left, dic) + helper(root.right.right, dic))
        #
        #     res = max(root.val + res, helper(root.left, dic) + helper(root.right, dic))
        #     dic[root] = res
        #     return res
        # return helper(root, {})

        def helper(root):
            if root is None: return [0, 0]
            left = helper(root.left)
            right = helper(root.right)

            res = [0, 0]
            res[0] = root.val + left[1] + right[1]
            res[1] = max(left) + max(right)
            return res

        return max(helper(root))

    def minPathSum(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        if len(grid) == 0: return 0
        row, col = len(grid), len(grid[0])
        one_row = [0] * (col + 1)

        for i in range(1, col + 1):
            one_row[i] = grid[0][i-1] + one_row[i-1]

        for i in range(1, row):
            two_row = [sys.maxint] * (col + 1)
            for j in range(col):
                two_row[j+1] = min(two_row[j], one_row[j+1]) + grid[i][j]
            one_row = two_row[:]
        return two_row[col]

    def numTrees(self, n):
        """
        :type n: int
        :rtype: int
        """
        dp = [0] * (n + 1)
        dp[0] = dp[1] = 1
        for size in range(2, n + 1):
            for root in range(1, size + 1):
                dp[size] += dp[root-1] * dp[size-root]
        return dp[n]

    def decodeString(self, s):
        """
        :type s: str
        :rtype: str
        """
        # s = "3[a]2[bc]", return "aaabcbc".
        # s = "3[a2[c]]", return "accaccacc".
        # s = "2[abc]3[cd]ef", return "abcabccdcdcdef".
        num_st, num_top, chr_st, chr_top = [None]*len(s), -1, [None]*len(s), -1
        cur_num, res = 0, []
        for ch in s:
            if ch >= '0' and ch <= '9':
                cur_num = cur_num * 10 + int(ch)
            elif ch == '[':
                num_top += 1
                num_st[num_top], cur_num = cur_num, 0
                chr_top += 1
                chr_st[chr_top] = ch
            elif ch == ']':
                mid = []
                while chr_st[chr_top] != '[':
                    mid.insert(0, chr_st[chr_top])
                    chr_top -= 1
                chr_top -= 1

                repeat_str, mid = ''.join(mid), []
                for _ in range(num_st[num_top]):
                    mid.append(repeat_str)
                num_top -= 1

                if num_top == -1:
                    res.append(''.join(mid))
                else:
                    chr_top += 1
                    chr_st[chr_top] = ''.join(mid)
            else:
                if num_top == -1:
                    res.append(ch)
                else:
                    chr_top += 1
                    chr_st[chr_top] = ch

        return ''.join(res)

    def findTargetSumWays(self, nums, S):
        """
        :type nums: List[int]
        :type S: int
        :rtype: int
        """
        def helper(nums, s):
            dp = [0] * (s + 1)
            dp[0] = 1
            for n in nums:
                for i in range(s, n-1, -1):
                    dp[i] += dp[i-n]
            return dp[s]
        sum_all = sum(nums)
        return 0 if ((sum_all + S) & 1 == 1) else helper(nums, (S + sum_all) // 2)

    def hasPathSum(self, root, sum):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: bool
        """
        if root is None:
            return False

        if root.val == sum and root.left is None and root.right is None:
            return True
        return self.hasPathSum(root.left, sum - root.val) or self.hasPathSum(root.right, sum - root.val)

    def flatten(self, root):
        """
        :type root: TreeNode
        :rtype: None Do not return anything, modify root in-place instead.
        """
        def helper(root):
            if root is None:
                return None, None

            ls, le = helper(root.left) # left start, left end
            rs, re = helper(root.right) # right start, right end
            if ls is None:
                le = root
            if rs is None:
                re = le
            root.left, root.right, le.right = None, ls, rs

            return root, re
        helper(root)

    def getPi(self, err):
        x0= 3.0
        x1 = x0 + math.sin(x0)
        while abs(x1-x0) > err:
            x0 = x1
            x1 = x0 + math.sin(x0)
        return x1

    def kthNode(self, root, k):
        """
        :type root: TreeNode
        :return:
        """
        # def helper(root, k):
        #     if root is None:
        #         return
        #     helper(root.left, k)
        #     self.index += 1
        #     if self.index == k:
        #         self.node = root
        #         return
        #     helper(root.right, k)
        # self.index, self.node = 0, None
        # helper(root, k)
        # return self.node.val

        st, top = [None]*k, -1
        while top >= 0 or root is not None:
            if root is not None:
                top += 1
                st[top], root = root, root.left
            else:
                root, top = st[top], top - 1
                if k == 1: return root.val
                k, root = k - 1, root.right
        return None

    def reachNumber(self, target):
        """
        :type target: int
        :rtype: int
        """
        if target < 0: target = -target
        n = int(math.sqrt(2 * target))
        while ((n + 1) * n / 2 + target) & 1 == 1 or (n + 1) * n / 2 < target:
            n += 1
        return n

    def buildTree(self, preorder, inorder):
        """
        :type preorder: List[int]
        :type inorder: List[int]
        :rtype: TreeNode
        """
        # preorder = [3, 9, 20, 15, 7]
        # inorder = [9, 3, 15, 20, 7]
        if len(preorder) == 0 or len(preorder) != len(inorder):
            return None
        if len(preorder) == 1:
            return TreeNode(preorder[0])
        idx = inorder.index(preorder[0])
        root = TreeNode(preorder[0])
        root.left = self.buildTree(preorder[1:idx+1], inorder[:idx])
        root.right = self.buildTree(preorder[idx+1:], inorder[idx+1:])
        return root

    def sortColors(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        def swap(nums, i, j):
            tmp = nums[i]
            nums[i], nums[j] = nums[j], tmp

        st, end = 0, len(nums)-1
        while nums[st] == 0 and st < end: st += 1
        while nums[end] == 2 and st < end: end -= 1

        mid = st
        while mid <= end:
            while nums[mid] == 2:
                swap(nums, mid, end)
                end -= 1
            if nums[mid] == 0:
                swap(nums, mid, st)
                st += 1
            mid += 1

    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
        if len(digits) == 0: return digits

        dict = {}
        dict['2'] = ['a', 'b', 'c']
        dict['3'] = ['d', 'e', 'f']
        dict['4'] = ['g', 'h', 'i']
        dict['5'] = ['j', 'k', 'l']
        dict['6'] = ['m', 'n', 'o']
        dict['7'] = ['p', 'q', 'r', 's']
        dict['8'] = ['t', 'u', 'v']
        dict['9'] = ['w', 'x', 'y', 'z']

        cap = 1
        for ch in digits:
            cap *= len(dict[ch])

        res, head, tail = ['']*cap, 0, 1
        for num in digits:
            mid = tail
            while head != mid:
                for ch in dict[num]:
                    res[tail] = res[head] + ch
                    tail = (tail + 1) % cap
                head = (head + 1) % cap
        return res

    def numIslands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """
        # use DFS to find all around land


    def numSquares(self, n):
        """
        :type n: int
        :rtype: int
        """
        sqr = int(math.sqrt(n))
        double = sqr * sqr
        if double == n: return 1

        dp = [sys.maxint] * (n + 1)
        dp[0] = 0
        for i in range(1, n+1):
            sqr = int(math.sqrt(i))
            for j in range(sqr//2, sqr+1):
                dp[i] = min(dp[i], dp[i - j*j] + 1)

        return dp[n]

    def subarraySum(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        # 2、dp
        count, cur, res = {0: 1}, 0, 0
        for v in nums:
            cur += v
            res += count.get(cur - k, 0)
            count[cur] = count.get(cur, 0) + 1
        return res

    def canPartition(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        def helper(nums, s):
            dp = [0] * (s + 1)
            dp[0] = 1
            for n in nums:
                for i in range(s, n-1, -1):
                    dp[i] += dp[i-n]
                    if dp[s] > 0: return True
            return False
        sum_all = sum(nums)
        return False if (sum_all & 1 == 1) else helper(nums, sum_all // 2)

    def lengthOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 1、O(n^2)
        # if len(nums) == 0: return 0
        # dp = [1] * len(nums)
        # for i in range(0, len(nums)):
        #     for j in range(i-1, -1, -1):
        #         if nums[i] > nums[j]:
        #             dp[i] = max(dp[i], dp[j] + 1)
        # return max(dp)

        # 2、O(N * logN)
        if len(nums) == 0: return 0
        dp, size = [0] * len(nums), 0
        for n in nums:
            s, e = 0, size
            while s < e:
                mid = (s + e) // 2
                if dp[mid] < n:
                    s = mid + 1
                else:
                    e = mid
            dp[s], size = n, max(s + 1, size)
        return size

    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        if root in [None, p, q]: return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)

        if left is not None and right is not None: return root
        return left if left is not None else right

    def canFinish(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """
        # 判断有向图是否有环
        # 1、拓扑排序
        mat = [[0]*numCourses for _ in range(numCourses)]
        for l in prerequisites:
            mat[l[0]][l[1]] = 1

        n, stack, top, d_list = numCourses, [-1]*numCourses, -1, [0]*numCourses
        for i in range(numCourses):
            degree = sum(mat[i])
            d_list[i] = degree
            if degree == 0:
                top += 1
                stack[top] = i
                n -= 1

        while top >= 0:
            i = stack[top]
            top -= 1
            for j in range(numCourses):
                if mat[j][i] == 1:
                    mat[j][i], d_list[j] = 0, d_list[j]-1
                    if d_list[j] == 0:  #入度为零
                        top += 1
                        stack[top] = j
                        n -= 1
        return n == 0

    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: bool
        """
        def helper(s, dic, dp, st):
            if dp[st] is not None: return dp[st]
            if dic.has_key(s[st:]):
                dp[st] = True
                return True

            for i in range(st, len(s)-1):
                if dic.has_key(s[st:i+1]):
                    dp[i+1] = helper(s, dic, dp, i+1)
                    if dp[i+1]: return True
            return False

        dic, dp = {}, [None]*len(s)
        for word in wordDict:
            dic[word] = None
        return helper(s, dic, dp, 0)

    def searchRange(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        def helper(nums, st, end, target):
            if st > end or nums[st] > target or nums[end] < target:
                return -1, -1
            mid = (st + end) // 2
            if nums[mid] == target:
                s1, e1 = helper(nums, st, mid-1, target)
                s2, e2 = helper(nums, mid+1, end, target)
                s = mid if s1 == -1 else s1
                e = mid if e2 == -1 else e2
                return s, e
            elif nums[mid] > target:
                return helper(nums, st, mid-1, target)
            else:
                return helper(nums, mid+1, end, target)
        s, e = helper(nums, 0, len(nums)-1, target)
        return [s, e]

    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        def helper(nums, low, high, target):
            if low > high: return -1

            mid = (low + high) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] > target:
                if nums[mid] < nums[high] or nums[low] <= target:
                    return helper(nums, low, mid - 1, target)
                else:
                    return helper(nums, mid+1, high, target)
            else:
                if nums[mid] > nums[low] or nums[high] >= target:
                    return helper(nums, mid+1, high, target)
                else:
                    return helper(nums, low, mid - 1, target)
        return helper(nums, 0, len(nums)-1, target)

    def exist(self, board, word):
        """
        :type board: List[List[str]]
        :type word: str
        :rtype: bool
        """
        def helper(board, word, i, j, size):
            if size == len(word): return True
            if i < 0 or j < 0 or i == len(board) or j == len(board[0]): return False
            if board[i][j] != word[size]: return False

            tmp, board[i][j] = board[i][j], "*"
            size += 1
            res = helper(board, word, i-1, j, size) \
                  or helper(board, word, i+1, j, size) \
                  or helper(board, word, i, j-1, size) \
                  or helper(board, word, i, j+1, size)
            board[i][j] = tmp
            return res



        if len(board) == 0 or len(word) == 0: return False
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == word[0] and helper(board, word, i, j, 0):
                    return True
        return False

    def canJump(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        jump = 0
        for i in range(len(nums)):
            if jump < i:
                return False
            jump = max(jump, nums[i] + i)
        return jump >= (len(nums) - 1)

    def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
        dp = [amount+1] * (amount+1)
        dp[0] = 0
        for i in range(1, amount+1):
            for coin in coins:
                if (i - coin >= 0):
                    dp[i] = min(dp[i], dp[i-coin] + 1)
        return -1 if dp[amount] == (amount+1) else dp[amount]

    def copyRandomList(self, head):
        """
        :type head: Node
        :rtype: Node
        """
        if head is None: return head
        dic, node = {None:None}, head
        while node is not None:
            dic[node] = Node(node.val, None, None)
            node = node.next

        node = head
        while node is not None:
            dic[node].next = dic[node.next]
            dic[node].random = dic[node.random]
            node = node.next
        return dic[head]

    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        def helper(root, l):
            if root is None: return
            helper(root.left, l)
            l.append(root.val)
            helper(root.right, l)

        l = []
        helper(root, l)

        for i in range(len(l)-1):
            if l[i+1] <= l[i]:
                return False
        return True

    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        dp = [[0] * (len(word2)+1) for _ in range(len(word1)+1)]
        for i in range(1, len(word2)+1):
            dp[0][i] = i
        for i in range(1, len(word1)+1):
            dp[i][0] = i

        for i in range(1, len(word1)+1):
            for j in range(1, len(word2)+1):
                dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]) + 1
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
        return dp[len(word1)][len(word2)]

    def mergeKLists(self, lists):
        """
        :type lists: List[ListNode]
        :rtype: ListNode
        """
        min_v, max_v, n_map = sys.maxint, -sys.maxint, {}
        for head in lists:
            node = head
            while node is not None:
                if n_map.has_key(node.val):
                    n_map[node.val].append(node)
                else:
                    n_map[node.val] = [node]

                if node.val > max_v: max_v = node.val
                if node.val < min_v: min_v = node.val

                node = node.next

        head = ListNode(0)
        tmp = head
        for i in xrange(min_v, max_v+1):
            if n_map.has_key(i):
                for node in n_map[i]:
                    tmp.next = node
                    tmp = tmp.next
        return head.next

    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        if len(nums) < 3:
            return []
        nums.sort()
        res, m_map = [], {}
        for i, v in enumerate(nums[:-2]):
            if i >= 1 and v == nums[i - 1]:
                continue
            d = {}
            for x in nums[i + 1:]:
                if d.has_key(x):
                    if not m_map.has_key((v, -v - x, x)):
                        res.append([v, -v - x, x])
                        m_map[(v, -v - x, x)] = None
                else:
                    d[-v - x] = 1
        return res






if __name__ == '__main__':
    s = Solution()
    # print s.findPeakElement([1, 2])
    # print(s.findKthLargest([3,2,1,5,6,4], 2))
    # print s.canFinish(3, [[0,1],[0,2],[1,2]])
    ss = "1,3,5,8,10,11,13,14,15,17,19,20,22,23,24,25,26,29,30,31,33,35,36,39,40,41,42,43,44,46,53,54,55,59,60,64,65,130,132,133,137,143,144,146,148,149,181,182,210,214,215,218,219,220,223,226,267,268,274,275,276,277,280,281,283,284,285,286,288,289,295,297,298,299,300,301,302,303,304,305,311,312,313,314,315,316,317,318,319,329,330,331,10004,10005,10008,10010,10011,10012,10013,10014,10016,10017,10021,10024,10027,10082,10085,10093,10095,10097,10098,10112,10113,10139,10141,10142,10156,10158,10159,10160,10161,10164,10170,10171,10173,10179,10182,10193,10194,10195,10196,10197,10198,10199,10200,10201,10203,10204,10205,10206,10207,10211,10221,10225,10226,10227,10273,10274,10282,10283,10284"
    ss1 = "1,6,9,17,20,21,22,23,24,25,26,29,30,31,32,33,35,36,39,40,41,42,43,44,45,46,47,49,50,51,54,61,63,144,181,182,210,213,214,215,219,220,223,226,277,10002,10004,10005,10007,10008,10010,10011,10012,10013,10016,10017,10019,10021,10027,10028,10029,10030,10082,10083,10085,10086,10088,10089,10090,10091,10092,10093,10095,10097,10100,10101,10102,10106,10110,10112,10113,10114"

    ss_map = {}
    ss_split = []
    for slot in ss.split(","):
        ss_map[slot] = 1
        ss_split.append(int(slot))

    count = 0
    for slot in ss1.split(","):
        if ss_map.has_key(slot):
            count += 1
        else:
            ss_split.append(int(slot))
    print "common slot size = ", count
    print len(ss_split)
    ss_split.sort()
    print ss_split


    # job = ""
    # for ips in ip.split(","):
    #     job += ("sdnn_time_truncate_model-"+ips+"|")
    # print job