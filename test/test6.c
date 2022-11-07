#include <stdio.h>
#include <time.h>
#define LEN N
#define LEN2 N
#define ntimes N
#define N 320000

float a[N], b[N], c[N], d[N], e[N], cc[N][N], bb[N][N], aa[N][N];


int s231()
{
	clock_t start_t, end_t, clock_dif; double clock_dif_sec;


	init( "s231 ");
	start_t = clock();

	for (int nl = 0; nl < 100*(ntimes/LEN2); nl++) {
		for (int i = 0; i < LEN2; ++i) {
			for (int j = 1; j < LEN2; j++) {
				aa[j][i] = aa[j - 1][i] + bb[j][i];
			}
		}
		dummy(a, b, c, d, e, aa, bb, cc, 0.);
	}
	end_t = clock(); clock_dif = end_t - start_t;
	clock_dif_sec = (double) (clock_dif/1000000.0);
	printf("S231\t %.2f \t\t", clock_dif_sec);;
	check(11);
	return 0;
}

int s112()
{


	clock_t start_t, end_t, clock_dif; double clock_dif_sec;


	start_t = clock();

	for (int nl = 0; nl < 3*ntimes; nl++) {
		for (int i = LEN - 2; i >= 0; i--) {
			a[i+1] = a[i] + b[i];
		}
		dummy(a, b, c, d, e, aa, bb, cc, 0.);
	}
	end_t = clock(); clock_dif = end_t - start_t;
	clock_dif_sec = (double) (clock_dif/1000000.0);
	printf("S112\t %.2f \t\t", clock_dif_sec);;
	check(1);
	return 0;
}
int s111()
{


	clock_t start_t, end_t, clock_dif; double clock_dif_sec;


	start_t = clock();

	for (int nl = 0; nl < 2*ntimes; nl++) {
		for (int i = 1; i < LEN; i += 2) {
			a[i] = a[i - 1] + b[i];
		}
		dummy(cc);
	}
	end_t = clock(); clock_dif = end_t - start_t;
	clock_dif_sec = (double) (clock_dif/1000000.0);
	printf("S111\t %.2f \t\t", clock_dif_sec);;
	check(1);
	return 0;
}

int s1111()
{

	clock_t start_t, end_t, clock_dif; double clock_dif_sec;

	start_t = clock();
	for (int nl = 0; nl < 2*ntimes; nl++) {
		for (int i = 0; i < LEN/2; i++) {
			a[2*i] = c[i] * b[i] + d[i] * b[i] + c[i] * c[i] + d[i] * b[i] + d[i] * c[i];
		}
		dummy(cc);
	}
	end_t = clock(); clock_dif = end_t - start_t;
	clock_dif_sec = (double) (clock_dif /1000000.0);
	
	printf("S1111\t %.2f \t\t ", clock_dif_sec);
	check(1);
	return 0;
}
int test1() {
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
        dummy(cc);
    }
    printf("test1: %f\n", 1.0*(clock() - start)/CLOCKS_PER_SEC);
    return a[N-1];
}

int test2() {
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

int s2244()
{


	clock_t start_t, end_t, clock_dif; double clock_dif_sec;

	start_t = clock();

	for (int nl = 0; nl < N; nl++) {
		for (int i = 0; i < N-1; i++) {
			a[i+1] = b[i] + e[i];
			a[i] = b[i] + c[i];
		}
		dummy(cc);
	}
	return 0;
}
