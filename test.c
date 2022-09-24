#include <stdio.h>
#include <time.h>
#define N 320000
int test1() {
    int a[N];
    int mid = N/2;
    clock_t start = clock();
    for (int nl = 0; nl < 100000; nl++) {
        for (int i = 0; i < N; i++) {
            if (i < mid) {
                a[i] = a[i] + 1;
            } else {
                a[i] = a[i] + 2;
            }
        }
    }
    printf("%f\n", 1.0*(clock() - start)/CLOCKS_PER_SEC);
    return a[N-1];
}

int main() {
    test1();
}