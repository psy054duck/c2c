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