#include <stdio.h>
#include <time.h>
#define LEN N
#define N 320000
#define ntimes 1000


int dummy(float a[LEN], float b[LEN], float c[LEN], float d[LEN], float e[LEN]);
int s2244()
{


	clock_t start_t, end_t, clock_dif; double clock_dif_sec;

    float a[N], b[N], c[N], e[N], d[N];
	start_t = clock();

	for (int nl = 0; nl < ntimes; nl++) {
		for (int i = 0; i < N-1; i++) {
			a[i+1] = b[i] + e[i];
			a[i] = b[i] + c[i];
		}
		dummy(a, b, c, d, e);
	}
	return 0;
}