#include <stdio.h>
#include <time.h>
#define N 320000
int test1() {
    int a[N], b[N], c[N], d[N];
    int mid = N/2;
    clock_t start = clock();
    for (int nl = 0; nl < 100000; nl++) {
        for (int i = 0; i < N; i++) {
            if (i < mid) {
                a[i] = a[i] + b[i]*c[i];
            } else {
                a[i] = a[i] + b[i]*d[i];
            }
        }
    }
    printf("test1: %f\n", 1.0*(clock() - start)/CLOCKS_PER_SEC);
    return a[N-1];
}

int test1p()
{
  int a[320000];
  int b[320000];
  int c[320000];
  int d[320000];
  int mid = 320000 / 2;
  clock_t start = clock();
  for (int nl = 0; nl < 100000; nl++)
  {
    for (int _t0 = 0; _t0 < 160000; _t0++)
    {
      a[_t0] = (b[_t0] * c[_t0]) + a[_t0];
    }

    for (int _t0 = 160000; _t0 < 320000; _t0++)
    {
      a[_t0] = (b[_t0] * d[_t0]) + a[_t0];
    }
  }

  printf("test1p: %f\n", (1.0 * (clock() - start)) / CLOCKS_PER_SEC);
  return a[320000 - 1];
}

int test2() {
    float a[N], b[N], c[N], d[N];
    int mid = N/2;
    float s;
    clock_t start = clock();
    for (int nl = 0; nl < 100000; nl++) {
        s = 0;
        for (int i = 0; i < N; i++) {
                s = s + 2;
                a[i] = a[i] + c[i]*s;
        }
    }
    printf("test2: %f\n", 1.0*(clock() - start)/CLOCKS_PER_SEC);
    return a[N-1];
}

int test2p()
{
  float a[320000];
  float b[320000];
  float c[320000];
  float d[320000];
  int mid = 320000 / 2;
  int s;
  clock_t start = clock();
  for (int nl = 0; nl < 100000; nl++)
  {
    for (int _t0 = 0; _t0 < 320000; _t0++)
    {
      a[_t0] = 2*(1 + _t0) * c[_t0] + a[_t0];
    }

    s = 640000;
  }

  printf("test2p: %f\n", (1.0 * (clock() - start)) / CLOCKS_PER_SEC);
  return a[320000 - 1];
}

int test2pp()
{
  float a[320000];
  float b[320000];
  float c[320000];
  float d[320000];
  int mid = 320000 / 2;
  int s;
  clock_t start = clock();
  for (int nl = 0; nl < 100000; nl++)
  {
    for (int _t0 = 0; _t0 < 320000; _t0++)
    {
        s = s + 2;
        a[_t0] = 2*(1 + _t0) * c[_t0] + a[_t0];
    }

  }

  printf("test2pp: %f\n", (1.0 * (clock() - start)) / CLOCKS_PER_SEC);
  return a[320000 - 1];
}

// int func1() {
//   float a[320000];
//   float b[320000];
//   float c[320000];
//   float d[320000];
//   clock_t start = clock();
//   for (int nl = 0; nl < 100000; nl++) {
//     for (int i = 0; i < N; i+=2) {
//       a[i] = b[i] + c[i];
//       a[i+1] = b[i+1] + d[i+1];
//     }
//   }
// 
//   printf("func1: %f\n", (1.0 * (clock() - start)) / CLOCKS_PER_SEC);
//   return a[N-1];
// }
// 
// int func2() {
//  float a[320000];
//   float b[320000];
//   float c[320000];
//   float d[320000];
//   clock_t start = clock();
//   for (int nl = 0; nl < 100000; nl++) {
// 	  #pragma clang loop vectorize(enable)
//     for (int i = 0; i < N/2; i++) {
//       // a[i] = b[i] + c[i];
//       a[2*i] = b[2*i] + c[2*i];
//     }
// 	  #pragma clang loop vectorize(enable)
//     for (int i = 0; i < N/2; i++) {
//       a[2*i+1] = b[2*i+1] + d[2*i+1];
//       // a[i+2] = b[i+2] + d[i+2];
//       // a[i+4] = b[i+4] + d[i+4];
//       // a[i+6] = b[i+6] + d[i+6];
//     }
//   }
//   printf("func2: %f\n", (1.0 * (clock() - start)) / CLOCKS_PER_SEC);
//   return a[N-1]; 
// 
// }

int main() {
    // func1();
    // func2();
    test1();
    test1p();
    test2();
    test2p();
    test2pp();
}