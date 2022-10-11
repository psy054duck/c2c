#include <stdio.h>
#include <time.h>
#define N 320000


int s2244()
{

//	node splitting
//	cycle with ture and anti dependency

	clock_t start_t, end_t, clock_dif; double clock_dif_sec;

    int a[N], b[N], c[N], e[N];
	init( "s244 ");
	start_t = clock();

	for (int nl = 0; nl < ntimes; nl++) {
		for (int i = 0; i < N-1; i++) {
			a[i+1] = b[i] + e[i];
			a[i] = b[i] + c[i];
		}
		dummy(a, b, c, d, e, aa, bb, cc, 0.);
	}
	return 0;
}