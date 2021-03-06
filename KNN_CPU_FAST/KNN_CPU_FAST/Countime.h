//
//  Countime.h
//  KNN_CPU_FAST
//
//  Created by 王方 on 18/9/21.
//  Copyright © 2018年 王方. All rights reserved.
//

#ifndef Countime_h
#define Countime_h

#if defined (__i386__)
static __inline__ unsigned long long GetCycleCount(void)
{
    unsigned long long int x;
    __asm__ volatile("rdtsc":"=A"(x));
    return x;
}
#elif defined (__x86_64__)
static __inline__ unsigned long long GetCycleCount(void)
{
    unsigned hi,lo;
    __asm__ volatile("rdtsc":"=a"(lo),"=d"(hi));
    return ((unsigned long long)lo)|(((unsigned long long)hi)<<32);
}
#endif


#endif /* Countime_h */
