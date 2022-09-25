
#include <stdio.h>
#include <time.h>
int test1()
{
  int a[320000];
  int mid = 320000 / 2;
  clock_t start = clock();
  for (int nl = 0; nl < 100000; nl++)
  {
    for (int _t0 = 0; _t0 < 160000; _t0++)
    {
      a[_t0] = 1 + a[_t0];
    }

    for (int _t0 = 160000; _t0 < 320000; _t0++)
    {
      a[_t0] = 2 + a[_t0];
    }

  }

  printf("%f\n", (1.0 * (clock() - start)) / CLOCKS_PER_SEC);
  return a[320000 - 1];
}

int main()
{
  test1();
}


