#include <stdio.h>
#include <time.h>
#define N 320000

int test5() {
    float a[N], b[N], c[N], d[N], cc[N][N], bb[N][N];
    clock_t start = clock();
    for (int nl = 0; nl < 100000; nl++) {
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				cc[j][i] = a[j] + bb[j][i]*d[j];
			}
		}
		dummy(a, b, c, d, e, aa, bb, cc, 0.);
    }
    printf("test2: %f\n", 1.0*(clock() - start)/CLOCKS_PER_SEC);
    return a[N-1];
}