#ifndef _NVCHARM_H_
#define _NVCHARM_H_

struct DeviceCtx {
  int n_sms;
};

struct Proxy {
  int n_chares;
  int *mapping; // Chare -> SM mapping

  __device__ Proxy() : n_chares(-1), mapping(nullptr) {}
};

struct Chare {
  Proxy proxy;

  __device__ void create(int n_chares);
  __device__ void invoke(int ep, int idx);
};

#endif
