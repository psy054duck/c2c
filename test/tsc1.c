#include <time.h>
#define __attribute__(x)
#define __restrict__

#define LEN 32000
#define LEN2 256

#define ntimes 50000
#define lll LEN
float array[LEN2*LEN2] __attribute__((aligned(16)));

float x[LEN] __attribute__((aligned(16)));
float temp;
int temp_int;


__attribute__((aligned(16))) float a[LEN],b[LEN],c[LEN],d[LEN],e[LEN],
                                   aa[LEN2][LEN2],bb[LEN2][LEN2],cc[LEN2][LEN2],tt[LEN2][LEN2];

int s000()
{

//	linear dependence testing
//	no dependence - vectorizable

	clock_t start_t, end_t, clock_dif; double clock_dif_sec;


	init( "s000 ");
	start_t = clock();

	for (int nl = 0; nl < 2*ntimes; nl++) {
		for (int i = 0; i < lll; i++) {
//			a[i] = b[i] + c[i];
//			X[i] = (Y[i] * Z[i])+(U[i]*V[i]);
			X[i] = Y[i] + 1;
		}
		dummy((float*)X, (float*)Y, (float*)Z, (float*)U, (float*)V, aa, bb, cc, 0.);
	}
	end_t = clock(); clock_dif = end_t - start_t;
	clock_dif_sec = (double) (clock_dif/1000000.0);
	printf("S000\t %.2f \t\t", clock_dif_sec);;
	check(1);
	return 0;
}

// %1.1
int s111()
{

//	linear dependence testing
//	no dependence - vectorizable

	clock_t start_t, end_t, clock_dif; double clock_dif_sec;


	init( "s111 ");
	start_t = clock();

	for (int nl = 0; nl < 2*ntimes; nl++) {
//		#pragma vector always
		for (int i = 1; i < LEN; i += 2) {
			a[i] = a[i - 1] + b[i];
		}
		dummy(a, b, c, d, e, aa, bb, cc, 0.);
	}
	end_t = clock(); clock_dif = end_t - start_t;
	clock_dif_sec = (double) (clock_dif/1000000.0);
	printf("S111\t %.2f \t\t", clock_dif_sec);;
	check(1);
	return 0;
}

int s1111()
{

//	no dependence - vectorizable
//	jump in data access

	clock_t start_t, end_t, clock_dif; double clock_dif_sec;

	init("s111 ");
	start_t = clock();
	for (int nl = 0; nl < 2*ntimes; nl++) {
		for (int i = 0; i < LEN/2; i++) {
			a[2*i] = c[i] * b[i] + d[i] * b[i] + c[i] * c[i] + d[i] * b[i] + d[i] * c[i];
		}
		dummy(a, b, c, d, e, aa, bb, cc, 0.);
	}
	end_t = clock(); clock_dif = end_t - start_t;
	clock_dif_sec = (double) (clock_dif /1000000.0);
	
	printf("S1111\t %.2f \t\t ", clock_dif_sec);
	check(1);
	return 0;
}

// %1.1

int s112()
{

//	linear dependence testing
//	loop reversal

	clock_t start_t, end_t, clock_dif; double clock_dif_sec;


	init( "s112 ");
	start_t = clock();

	for (int nl = 0; nl < 3*ntimes; nl++) {
//		#pragma vector always
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


int s1112()
{

//	linear dependence testing
//	loop reversal

	clock_t start_t, end_t, clock_dif; double clock_dif_sec;
	

	init("s112 ");
	start_t = clock();

	for (int nl = 0; nl < ntimes*3; nl++) {
		for (int i = LEN - 1; i >= 0; i--) {
			a[i] = b[i] + (float) 1.;
		}
		dummy(a, b, c, d, e, aa, bb, cc, 0.);
	}
	end_t = clock(); clock_dif = end_t - start_t;
	clock_dif_sec = (double) (clock_dif /1000000.0);
	
	printf("S1112\t %.2f \t\t ", clock_dif_sec);
	check(1);
	return 0;
}

int s113()
{

//	linear dependence testing
//	a(i)=a(1) but no actual dependence cycle

	clock_t start_t, end_t, clock_dif; double clock_dif_sec;


	init( "s113 ");
	start_t = clock();

	for (int nl = 0; nl < 4*ntimes; nl++) {
		for (int i = 1; i < LEN; i++) {
			a[i] = a[0] + b[i];
		}
		dummy(a, b, c, d, e, aa, bb, cc, 0.);
	}
	end_t = clock(); clock_dif = end_t - start_t;
	clock_dif_sec = (double) (clock_dif/1000000.0);
	printf("S113\t %.2f \t\t", clock_dif_sec);;
	check(1);
	return 0;
}

int s1113()
{

//	linear dependence testing
//	one iteration dependency on a(LEN/2) but still vectorizable

	clock_t start_t, end_t, clock_dif; double clock_dif_sec;


	init( "s113 ");
	start_t = clock();

	for (int nl = 0; nl < 2*ntimes; nl++) {
		for (int i = 0; i < LEN; i++) {
			a[i] = a[LEN/2] + b[i];
		}
		dummy(a, b, c, d, e, aa, bb, cc, 0.);
	}
	end_t = clock(); clock_dif = end_t - start_t;
	clock_dif_sec = (double) (clock_dif/1000000.0);
	printf("S1113\t %.2f \t\t", clock_dif_sec);;
	check(1);
	return 0;
}

// %1.1

int s114()
{

//	linear dependence testing
//	transpose vectorization
//	Jump in data access - not vectorizable

	clock_t start_t, end_t, clock_dif; double clock_dif_sec;


	init( "s114 ");
	start_t = clock();

	for (int nl = 0; nl < 200*(ntimes/(LEN2)); nl++) {
		for (int i = 0; i < LEN2; i++) {
			for (int j = 0; j < i; j++) {
				aa[i][j] = aa[j][i] + bb[i][j];
			}
		}
		dummy(a, b, c, d, e, aa, bb, cc, 0.);
	}

	end_t = clock(); clock_dif = end_t - start_t;
	clock_dif_sec = (double) (clock_dif/1000000.0);
	printf("S114\t %.2f \t\t", clock_dif_sec);;
	check(11);
	return 0;
}

// %1.1

int s115()
{

//	linear dependence testing
//	triangular saxpy loop

	clock_t start_t, end_t, clock_dif; double clock_dif_sec;


	init( "s115 ");
	start_t = clock();

	for (int nl = 0; nl < 1000*(ntimes/LEN2); nl++) {
		for (int j = 0; j < LEN2; j++) {
			for (int i = j+1; i < LEN2; i++) {
				a[i] -= aa[j][i] * a[j];
			}
		}
		dummy(a, b, c, d, e, aa, bb, cc, 0.);
	}
	end_t = clock(); clock_dif = end_t - start_t;
	clock_dif_sec = (double) (clock_dif/1000000.0);
	printf("S115\t %.2f \t\t", clock_dif_sec);;
	check(1);
	return 0;
}

int s1115()
{

//	linear dependence testing
//	triangular saxpy loop

	clock_t start_t, end_t, clock_dif; double clock_dif_sec;


	init( "s115 ");
	start_t = clock();

	for (int nl = 0; nl < 100*(ntimes/LEN2); nl++) {
		for (int i = 0; i < LEN2; i++) {
			for (int j = 0; j < LEN2; j++) {
				aa[i][j] = aa[i][j]*cc[j][i] + bb[i][j];
			}
		}
		dummy(a, b, c, d, e, aa, bb, cc, 0.);
	}
	end_t = clock(); clock_dif = end_t - start_t;
	clock_dif_sec = (double) (clock_dif/1000000.0);
	printf("S1115\t %.2f \t\t", clock_dif_sec);;
	check(11);
	return 0;
}

// %1.1

int s116()
{

//	linear dependence testing

	clock_t start_t, end_t, clock_dif; double clock_dif_sec;


	init( "s116 ");
	start_t = clock();

	for (int nl = 0; nl < ntimes*10; nl++) {
		for (int i = 0; i < LEN - 5; i += 5) {
			a[i] = a[i + 1] * a[i];
			a[i + 1] = a[i + 2] * a[i + 1];
			a[i + 2] = a[i + 3] * a[i + 2];
			a[i + 3] = a[i + 4] * a[i + 3];
			a[i + 4] = a[i + 5] * a[i + 4];
		}
		dummy(a, b, c, d, e, aa, bb, cc, 0.);
	}
	end_t = clock(); clock_dif = end_t - start_t;
	clock_dif_sec = (double) (clock_dif/1000000.0);
	printf("S116\t %.2f \t\t", clock_dif_sec);;
	check(1);
	return 0;
}

// %1.1

int s118()
{

//	linear dependence testing
//	potential dot product recursion

	clock_t start_t, end_t, clock_dif; double clock_dif_sec;


	init( "s118 ");
	start_t = clock();

	for (int nl = 0; nl < 200*(ntimes/LEN2); nl++) {
		for (int i = 1; i < LEN2; i++) {
			for (int j = 0; j <= i - 1; j++) {
				a[i] += bb[j][i] * a[i-j-1];
			}
		}
		dummy(a, b, c, d, e, aa, bb, cc, 0.);
	}
	end_t = clock(); clock_dif = end_t - start_t;
	clock_dif_sec = (double) (clock_dif/1000000.0);
	printf("S118\t %.2f \t\t", clock_dif_sec);;
	check(1);
	return 0;
}

// %1.1

int s119()
{

//	linear dependence testing
//	no dependence - vectorizable

	clock_t start_t, end_t, clock_dif; double clock_dif_sec;
	

	init("s119 ");
	start_t = clock();

	for (int nl = 0; nl < 200*(ntimes/(LEN2)); nl++) {
		for (int i = 1; i < LEN2; i++) {
			for (int j = 1; j < LEN2; j++) {
				aa[i][j] = aa[i-1][j-1] + bb[i][j];
			}
		}
		dummy(a, b, c, d, e, aa, bb, cc, 0.);
	}
	end_t = clock(); clock_dif = end_t - start_t;
	clock_dif_sec = (double) (clock_dif /1000000.0);
	
	
	printf("S119\t %.2f \t\t ", clock_dif_sec);
	check(11);
	return 0;
}