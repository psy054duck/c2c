#include <stdio.h>
#include <time.h>
#define N 320000
int test2() {
    int aa[N][N], bb[N][N];
		for (int i = 0; i < N; ++i) {
			for (int j = 1; j < N; j++) {
				aa[j][i] = aa[j - 1][i] + bb[j][i];
			}
		}
    return 0;
}

int main() {
    test2();
}