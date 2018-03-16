#!/usr/bin/python
# -*- coding: utf-8 -*-

class ElementOperator:
    def add(self, num1, num2):
        # 32bits integer max/min
        MAX = 0x7FFFFFFF
        MASK = 0xFFFFFFFF

        ans = num1
        while num2 != 0:
            ans = (num1 ^ num2) & MASK
            num2 = ((num1 & num2) << 1) & MASK
            num1 = ans
        return ans if ans <= MAX else ~(ans ^ MASK)

    def subtract(self, num1, num2):
        mid = self.add(~num2, 1)
        return self.add(num1, mid)

    def is_negative(self, num1, num2):
        return (num1 ^ num2) < 0

    def abs(self, num):
        if num >= 0:
            return num
        else:
            return self.add(~num, 1)

    def multiply(self, num1, num2):
        abs1 = self.abs(num1)
        abs2 = self.abs(num2)
        ans = 0
        while abs2 != 0:
            if abs2 & 1:
                ans = self.add(ans, abs1)
            abs2 = abs2 >> 1
            abs1 = abs1 << 1
        if self.is_negative(num1, num2):
            return self.add(~ans, 1)
        return ans



    def divide(self, num1, num2):
        # exception
        if num2 == 0:
            raise Exception("Divisor is zero.", num2)

        abs1 = self.abs(num1)
        abs2 = self.abs(num2)

        ans = 0
        i = 31
        while i >= 0:
            if (abs1 >> i) >= abs2:
                ans = self.add(ans, 1 << i)
                abs1 = self.subtract(abs1, abs2 << i)
            i = self.subtract(i, 1)
        if self.is_negative(num1, num2):
            return self.add(~ans, 1)
        return ans

if __name__ == '__main__':
    s = ElementOperator()
    print  s.subtract(5, -1)