#include <stdio.h>
#include <time.h>
#define LEN N
#define LEN2 N
#define ntimes N
#define N 320000

// test
float a[N], b[N], c[N], d[N], e[N], cc[N][N], bb[N][N], aa[N][N];
int s2102()
{


	clock_t start_t, end_t, clock_dif; double clock_dif_sec;


	init( "s2102");
	start_t = clock();

	for (int nl = 0; nl < 100*(ntimes/LEN2); nl++) {
		for (int i = 0; i < LEN2; i++) {
			for (int j = 0; j < LEN2; j++) {
				aa[j][i] = 0;
			}
			aa[i][i] = 1;
		}
		dummy(a, b, c, d, e, aa, bb, cc, 0.);
	}
	end_t = clock(); clock_dif = end_t - start_t;
	clock_dif_sec = (clock_dif/1000000.0);
	printf("S2102\t %.2f \t\t", clock_dif_sec);
	check(11);
	return 0;
}