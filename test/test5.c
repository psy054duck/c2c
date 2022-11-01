#include <stdio.h>
#include <time.h>
#define LEN N
#define LEN2 N
#define N 320000

int dummy(float cc[LEN2][LEN2]){
	// --  called in each loop to make all computations appear required
	return 0;
}
int test5() {
    float a[N], b[N], c[N], d[N], cc[N][N], bb[N][N];
    clock_t start = clock();
    for (int nl = 0; nl < 100000; nl++) {
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				cc[j][i] = a[j] + bb[j][i]*d[j];
			}
		}
		dummy(cc);
    }
    printf("test5: %f\n", 1.0*(clock() - start)/CLOCKS_PER_SEC);
    return cc[N-1][N-1];
}

int test5p()
{
  float a[320000];
  float b[320000];
  float c[320000];
  float d[320000];
  float cc[320000][320000];
  float bb[320000][320000];
  clock_t start = clock();
  for (int nl = 0; nl < 100000; nl++)
  {
    for (int _t0 = 0; _t0 < 320000; _t0++)
    {
      for (int _t1 = 0; _t1 < 320000; _t1++)
      {
        cc[_t0][_t1] = (bb[_t0][_t1] * d[_t0]) + a[_t0];
      }

    }

    dummy(cc);
  }

  printf("test5p: %f\n", (1.0 * (clock() - start)) / CLOCKS_PER_SEC);
  return cc[320000 - 1][N-1];
}

int main() {
  test5();
  test5p();
}