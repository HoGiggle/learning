#!/usr/bin/python
# -*- coding: utf-8 -*-

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

if __name__ == '__main__':
    s = Solution()
    print s.findPeakElement([1, 2])
