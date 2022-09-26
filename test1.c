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
    printf("%f\n", 1.0*(clock() - start)/CLOCKS_PER_SEC);
    return a[N-1];
}

int test2(int n1, int n3) {
    int aa[N][N], bb[N][N];
    for (int nl = 0; nl < 100; nl++) {
		for (int i = 0; i < N; ++i) {
			for (int j = 1; j < N; j++) {
				aa[j][i] = aa[j - 1][i] + bb[j][i];
			}
		}
	}
}

int main() {
    test1();
}