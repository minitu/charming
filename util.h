#ifndef _UTIL_H_
#define _UTIL_H_

using clock_value_t = long long;

__device__ uint get_smid();
__device__ void sleep(clock_value_t sleep_cycles);

#endif // _UTIL_H_
