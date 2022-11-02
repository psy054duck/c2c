#include <stdio.h>
#include <time.h>
#define LEN N
#define LEN2 N
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
		dummy(cc);
    }
    printf("test5: %f\n", 1.0*(clock() - start)/CLOCKS_PER_SEC);
    return cc[N-1][N-1];
}
