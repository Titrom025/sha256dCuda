#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include <openssl/sha.h>

#define WARP_SIZE                    32
#define NUM_WARPS_IN_BLOCK           4
#ifndef MAX_GPU_THREADS
#define MAX_GPU_THREADS 16
#endif
#define NUM_THREADS_IN_BLOCK         (NUM_WARPS_IN_BLOCK * WARP_SIZE)
#define LAUNCH_BOUNDS                __launch_bounds__(NUM_THREADS_IN_BLOCK,1)
#define DEVICE_FUNCTION_QUALIFIERS   __device__ __forceinline__

#define COMMA         ,
#define EMPTY

typedef unsigned int beu32;

#define W64(notLast,last)                        \
  W(0 ,0  ,0  ,0  ,0 , 0x428a2f98)   notLast     \
  W(1 ,0  ,0  ,0  ,0 , 0x71374491)   notLast     \
  W(2 ,0  ,0  ,0  ,0 , 0xb5c0fbcf)   notLast     \
  W(3 ,0  ,0  ,0  ,0 , 0xe9b5dba5)   notLast     \
  W(4 ,0  ,0  ,0  ,0 , 0x3956c25b)   notLast     \
  W(5 ,0  ,0  ,0  ,0 , 0x59f111f1)   notLast     \
  W(6 ,0  ,0  ,0  ,0 , 0x923f82a4)   notLast     \
  W(7 ,0  ,0  ,0  ,0 , 0xab1c5ed5)   notLast     \
  W(8 ,0  ,0  ,0  ,0 , 0xd807aa98)   notLast     \
  W(9 ,0  ,0  ,0  ,0 , 0x12835b01)   notLast     \
  W(10,0  ,0  ,0  ,0 , 0x243185be)   notLast     \
  W(11,0  ,0  ,0  ,0 , 0x550c7dc3)   notLast     \
  W(12,0  ,0  ,0  ,0 , 0x72be5d74)   notLast     \
  W(13,0  ,0  ,0  ,0 , 0x80deb1fe)   notLast     \
  W(14,0  ,0  ,0  ,0 , 0x9bdc06a7)   notLast     \
  W(15,0  ,0  ,0  ,0 , 0xc19bf174)   notLast     \
  W(16,0  ,1  ,9  ,14, 0xe49b69c1)   notLast     \
  W(17,1  ,2  ,10 ,15, 0xefbe4786)   notLast     \
  W(18,2  ,3  ,11 ,16, 0x0fc19dc6)   notLast     \
  W(19,3  ,4  ,12 ,17, 0x240ca1cc)   notLast     \
  W(20,4  ,5  ,13 ,18, 0x2de92c6f)   notLast     \
  W(21,5  ,6  ,14 ,19, 0x4a7484aa)   notLast     \
  W(22,6  ,7  ,15 ,20, 0x5cb0a9dc)   notLast     \
  W(23,7  ,8  ,16 ,21, 0x76f988da)   notLast     \
  W(24,8  ,9  ,17 ,22, 0x983e5152)   notLast     \
  W(25,9  ,10 ,18 ,23, 0xa831c66d)   notLast     \
  W(26,10 ,11 ,19 ,24, 0xb00327c8)   notLast     \
  W(27,11 ,12 ,20 ,25, 0xbf597fc7)   notLast     \
  W(28,12 ,13 ,21 ,26, 0xc6e00bf3)   notLast     \
  W(29,13 ,14 ,22 ,27, 0xd5a79147)   notLast     \
  W(30,14 ,15 ,23 ,28, 0x06ca6351)   notLast     \
  W(31,15 ,16 ,24 ,29, 0x14292967)   notLast     \
  W(32,16 ,17 ,25 ,30, 0x27b70a85)   notLast     \
  W(33,17 ,18 ,26 ,31, 0x2e1b2138)   notLast     \
  W(34,18 ,19 ,27 ,32, 0x4d2c6dfc)   notLast     \
  W(35,19 ,20 ,28 ,33, 0x53380d13)   notLast     \
  W(36,20 ,21 ,29 ,34, 0x650a7354)   notLast     \
  W(37,21 ,22 ,30 ,35, 0x766a0abb)   notLast     \
  W(38,22 ,23 ,31 ,36, 0x81c2c92e)   notLast     \
  W(39,23 ,24 ,32 ,37, 0x92722c85)   notLast     \
  W(40,24 ,25 ,33 ,38, 0xa2bfe8a1)   notLast     \
  W(41,25 ,26 ,34 ,39, 0xa81a664b)   notLast     \
  W(42,26 ,27 ,35 ,40, 0xc24b8b70)   notLast     \
  W(43,27 ,28 ,36 ,41, 0xc76c51a3)   notLast     \
  W(44,28 ,29 ,37 ,42, 0xd192e819)   notLast     \
  W(45,29 ,30 ,38 ,43, 0xd6990624)   notLast     \
  W(46,30 ,31 ,39 ,44, 0xf40e3585)   notLast     \
  W(47,31 ,32 ,40 ,45, 0x106aa070)   notLast     \
  W(48,32 ,33 ,41 ,46, 0x19a4c116)   notLast     \
  W(49,33 ,34 ,42 ,47, 0x1e376c08)   notLast     \
  W(50,34 ,35 ,43 ,48, 0x2748774c)   notLast     \
  W(51,35 ,36 ,44 ,49, 0x34b0bcb5)   notLast     \
  W(52,36 ,37 ,45 ,50, 0x391c0cb3)   notLast     \
  W(53,37 ,38 ,46 ,51, 0x4ed8aa4a)   notLast     \
  W(54,38 ,39 ,47 ,52, 0x5b9cca4f)   notLast     \
  W(55,39 ,40 ,48 ,53, 0x682e6ff3)   notLast     \
  W(56,40 ,41 ,49 ,54, 0x748f82ee)   notLast     \
  W(57,41 ,42 ,50 ,55, 0x78a5636f)   notLast     \
  W(58,42 ,43 ,51 ,56, 0x84c87814)   notLast     \
  W(59,43 ,44 ,52 ,57, 0x8cc70208)   notLast     \
  W(60,44 ,45 ,53 ,58, 0x90befffa)   notLast     \
  W(61,45 ,46 ,54 ,59, 0xa4506ceb)   notLast     \
  W(62,46 ,47 ,55 ,60, 0xbef9a3f7)   notLast     \
  W(63,47 ,48 ,56 ,61, 0xc67178f2)   last

#define H8(notLast,last)                        \
  H(0, a, 0x6a09e667)   notLast                 \
  H(1, b, 0xbb67ae85)   notLast                 \
  H(2, c, 0x3c6ef372)   notLast                 \
  H(3, d, 0xa54ff53a)   notLast                 \
  H(4, e, 0x510e527f)   notLast                 \
  H(5, f, 0x9b05688c)   notLast                 \
  H(6, g, 0x1f83d9ab)   notLast                 \
  H(7, h, 0x5be0cd19)   last

#define M8(notLast,last)                      \
  M(h, g)   notLast                           \
  M(g, f)   notLast                           \
  M(f, e)   notLast                           \
  M(e, d)   notLast                           \
  M(d, c)   notLast                           \
  M(c, b)   notLast                           \
  M(b, a)   notLast                           \
  M(a, t)   last

#define C16(notLast,last)                     \
  C(0 )   notLast                             \
  C(1 )   notLast                             \
  C(2 )   notLast                             \
  C(3 )   notLast                             \
  C(4 )   notLast                             \
  C(5 )   notLast                             \
  C(6 )   notLast                             \
  C(7 )   notLast                             \
  C(8 )   notLast                             \
  C(9 )   notLast                             \
  C(10)   notLast                             \
  C(11)   notLast                             \
  C(12)   notLast                             \
  C(13)   notLast                             \
  C(14)   notLast                             \
  C(15)   last

DEVICE_FUNCTION_QUALIFIERS
beu32 notand(beu32 a, const beu32 b)
{
#if __CUDA_ARCH__ >= 100
  beu32 d;
  asm("not.b32  %1, %1; \n\t"
      "and.b32  %0, %1, %2; \n\t"
      : "=r"(d), "+r"(a) : "r"(b));
  return d;
#else
  return ~a & b;
#endif
}

DEVICE_FUNCTION_QUALIFIERS
beu32 ror(const beu32 a, const unsigned int n)
{
#if __CUDA_ARCH__ >= 350
  beu32 d;
  asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(a), "r"(n));
  return d;
#else
  return (a >> n) | (a << (32 - n));
#endif
}

DEVICE_FUNCTION_QUALIFIERS
beu32 shr(const beu32 a, const unsigned int n)
{
#if __CUDA_ARCH__ >= 999
  beu32 d;
  asm("vshr.u32.u32.u32.clamp %0, %1, %2;" : "=r"(d) : "r"(a), "r"(n));
  return d;
#else
  return a >> n;
#endif
}

DEVICE_FUNCTION_QUALIFIERS
beu32 add3(const beu32 a, const beu32 b, const beu32 c)
{
#if __CUDA_ARCH__ >= 999
  beu32 d;
  asm("vadd.u32.u32.u32.add %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
#else
  return a + b + c;
#endif
}

DEVICE_FUNCTION_QUALIFIERS
void hmix(beu32* a,
          beu32* b,
          beu32* c,
          beu32* d,
          beu32* e,
          beu32* f,
          beu32* g,
          beu32* h,
          beu32* t)
{
#undef  M
#define M(alpha,mix) *alpha = *mix;

  M8(EMPTY,EMPTY);
}

#undef  C
#define C(i)              const beu32 c##i

#undef  H
#define H(i,alpha,magic)  const beu32 hin##i, beu32* hout##i

DEVICE_FUNCTION_QUALIFIERS
void sha256_chunk(C16(COMMA,EMPTY),
                  H8(COMMA,EMPTY))
{
  #undef  W
  #define W(i,m16,m15,m7,m2,magic)  beu32 w##i;

  W64(EMPTY,EMPTY);

  #undef  C
  #define C(i)  w##i = c##i;

  C16(EMPTY,EMPTY);

  #undef  W
  #define W(i,m16,m15,m7,m2,magic)                                \
  if (i >= 16) {                                                \
    w##i = w##m16 +                                             \
      add3(w##m7,                                               \
           (ror(w##m15, 7) ^ ror(w##m15,18) ^ shr(w##m15, 3)),  \
           (ror(w##m2, 17) ^ ror(w##m2, 19) ^ shr(w##m2, 10))); \
  }

  W64(EMPTY,EMPTY);

  #undef  H
  #define H(i,alpha,magic)  beu32 alpha = hin##i;

  H8(EMPTY,EMPTY);

  #undef  W
  #define W(i,m16,m15,m7,m2,magic)                        \
  {                                                     \
    beu32 t = add3(add3(h,w##i,magic),                  \
                   (ror(e,6) ^ ror(e,11) ^ ror(e,25)),  \
                   ((e & f) ^ notand(e,g)));            \
                                                        \
    d += t;                                             \
                                                        \
    t = add3(t,                                         \
             (ror(a,2) ^ ror(a,13) ^ ror(a,22)),        \
             ((a & (b ^ c)) ^ (b & c)));                \
                                                        \
    hmix(&a,&b,&c,&d,&e,&f,&g,&h,&t);                   \
  }

  W64(EMPTY,EMPTY);

  #undef  H
  #define H(i,alpha,magic)  *hout##i = hin##i + alpha;

  H8(EMPTY,EMPTY);
}

#undef  C
#define C(i)              const beu32 c##i

#undef  H
#define H(i,alpha,magic)  beu32* hout##i

DEVICE_FUNCTION_QUALIFIERS
void sha256_chunk0(C16(COMMA,EMPTY),H8(COMMA,EMPTY))
{
  #undef  C
  #define C(i)              c##i

  #undef  H
  #define H(i,alpha,magic)  magic,hout##i

  sha256_chunk(C16(COMMA,EMPTY),H8(COMMA,EMPTY));
}

struct vec8u {
  uint32_t v[8];
};

struct vec16u {
  uint32_t v[16];
};

struct ThreadData {
    vec8u state;
    uint32_t rdata[9];
};

struct MsgData {
  uint32_t pseed[3];
  ThreadData thrdata[MAX_GPU_THREADS];
  vec8u target;
};

struct DevHashResult {
  uint64_t nonce;
  uint32_t vcpu, found;
};

__constant__ MsgData c_msg;

extern "C" __global__ void sha256d_kernel(uint64_t start_nonce, DevHashResult *result) {
    uint32_t vcpu = blockIdx.y;
    ThreadData *thrdata = &c_msg.thrdata[vcpu];

    vec16u data;

    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t nonce = start_nonce + idx;

    uint32_t rdata6 = thrdata->rdata[0], rdata7 = thrdata->rdata[1], rdata8 = thrdata->rdata[2];
    uint64_t rdata = (((uint64_t)rdata6 << 56) | ((uint64_t)rdata7 << 24) | ((uint64_t)rdata8 >> 8)) + nonce;

    rdata6 = (uint32_t)(rdata >> 56) | (rdata6 & 0xFFFFFF00);
    rdata7 = (uint32_t)(rdata >> 24);

    uint32_t rdata10 = (uint32_t)(rdata << 8) | 0x80;
    rdata8 = (uint32_t)(rdata << 8) | (rdata8 & 0xFF);

    data.v[0x0] = rdata6;
    data.v[0x1] = rdata7;
    data.v[0x2] = rdata8;
    data.v[0x3] = c_msg.pseed[0];
    data.v[0x4] = c_msg.pseed[1];
    data.v[0x5] = c_msg.pseed[2];
    data.v[0x6] = thrdata->rdata[3];
    data.v[0x7] = thrdata->rdata[4];
    data.v[0x8] = thrdata->rdata[5];
    data.v[0x9] = thrdata->rdata[6];
    data.v[0xA] = thrdata->rdata[7];
    data.v[0xB] = thrdata->rdata[8];
    data.v[0xC] = rdata6;
    data.v[0xD] = rdata7;
    data.v[0xE] = rdata10;
    data.v[0xF] = 0x00000000;

    sha256_chunk0(data.v[0], data.v[1], data.v[2], data.v[3],
                  data.v[4], data.v[5], data.v[6], data.v[7], 
                  data.v[8], data.v[9], data.v[10], data.v[11],
                  data.v[12], data.v[13], data.v[14], data.v[15], 
                  &data.v[0], &data.v[1], &data.v[2], &data.v[3],
                  &data.v[4], &data.v[5], &data.v[6], &data.v[7]);
    
    data.v[0x8] = 0x00000000;
    data.v[0x9] = 0x00000000;
    data.v[0xA] = 0x00000000;
    data.v[0xB] = 0x00000000;
    data.v[0xC] = 0x00000000;
    data.v[0xD] = 0x00000000;
    data.v[0xE] = 0x00000000;
    data.v[0xF] = 0x000003d8;
    
    sha256_chunk0(data.v[0], data.v[1], data.v[2], data.v[3],
              data.v[4], data.v[5], data.v[6], data.v[7], 
              data.v[8], data.v[9], data.v[10], data.v[11],
              data.v[12], data.v[13], data.v[14], data.v[15], 
              &data.v[0], &data.v[1], &data.v[2], &data.v[3],
              &data.v[4], &data.v[5], &data.v[6], &data.v[7]);

    for (int i = 0; i < 8; i += 1) {
      if (data.v[i] > c_msg.target.v[i])
        return;

      if (data.v[i] < c_msg.target.v[i]) {
        if (atomicAdd(&result->found, 1) == 0) {
          result->nonce = nonce, result->vcpu = vcpu;
          printf("result: %p, result->vcpu: %d, result->nonce: %lu\n", result, result->vcpu, result->nonce);
        }

        return;
      }
    }

    // if ((idx + vcpu) % 10000000 == 0) {
    //   uint32_t old_value = atomicAdd(&result->found, 1);
    //   printf("old_value: %u, start_nonce: %lu, idx: %lu, vcpu: %lu\n", old_value, start_nonce, idx, vcpu);
    //   if (old_value == 0) {
    //       result->nonce = nonce;
    //       result->vcpu = vcpu;
    //       printf("result: %p, result->vcpu: %d, result->nonce: %lu, result->found: %u\n", result, result->vcpu, result->nonce, old_value);
    //   }
    // }
    return;
}
