#!/usr/bin/python
# -*- coding: utf-8 -*-
import math
import random
import numpy as np


class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


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
        # while i < j:
        #     while (i < j) and (tmp <= nums[j]):
        #         j = j - 1
        #     nums[i] = nums[j]
        #     while (i < j) and (tmp >= nums[i]):
        #         i = i + 1
        #     nums[j] = nums[i]
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
        return [a+b if c == '+' else a-b if c == '-' else a*b for i, c in enumerate(input) if c < '0'
                for a in self.diffWaysToCompute(input[:i])
                for b in self.diffWaysToCompute(input[i+1:])] or [int(input)]

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

        one, sec = cost[0], cost[1]
        for i in range(2, len(cost)):
            tmp = sec
            sec = min(one, sec) + cost[i]
            one = tmp
        return sec

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


if __name__ == '__main__':
    s = Solution()
    # print s.findPeakElement([1, 2])
    # print(s.findKthLargest([3,2,1,5,6,4], 2))
    # print(s.minCostClimbingStairs([1, 100, 1, 1, 1, 100, 1, 1, 100, 1]))

    sq = [['召回', 1], ['排序', 2], ['投递', 3], ['haha', 4]]
    for i in range(20):
        print "result: %s" % s.probabilityRecall_bin(sq)