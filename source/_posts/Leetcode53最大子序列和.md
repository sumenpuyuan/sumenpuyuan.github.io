---
title: Leetcode53最大子序列和
date: 2018-10-15 18:36:14
categories:
  - 算法
tags:
---

Input: [-2,1,-3,4,-1,2,1,-5,4],
Output: 6
Explanation: [4,-1,2,1] has the largest sum = 6.
```cpp
#include<iostream>
#include<limits.h>
#include<vector>
using namespace std;

class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int ans = 0, maxn = INT_MIN;
        int len = nums.size();
        for(int i = 0; i < len; i++){
            if(ans < 0) ans = 0;  //如果前面的和小0，那么重新开始求和
            ans += nums[i];
            maxn = max(maxn, ans);
        }
        return maxn;
    }
};
int main(){
    Solution s;
    int A[]={-2,1,-3,4,-1,2,1,-5,4};
    vector<int> B(A,A+9);
    cout<<s.maxSubArray(B);
}

```

