#include <stdio.h>
#include <time.h>
#define N 320000
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