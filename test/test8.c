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

int s211()
{

//	statement reordering
//	statement reordering allows vectorization

	clock_t start_t, end_t, clock_dif; double clock_dif_sec;


	init( "s211 ");
	start_t = clock();

	for (int nl = 0; nl < ntimes; nl++) {
		for (int i = 1; i < LEN; i++) {
			a[i] = b[i] + 1;
		}
		dummy(a, b, c, d, e, aa, bb, cc, 0.);
	}
	end_t = clock(); clock_dif = end_t - start_t;
	clock_dif_sec = (double) (clock_dif/1000000.0);
	printf("S211\t %.2f \t\t", clock_dif_sec);;
	check(12);
	return 0;
}


// int s276()
// {
// 
// //	control flow
// //	if test using loop index
// 
// 	clock_t start_t, end_t, clock_dif; double clock_dif_sec;
// 
// 
// 	init( "s276 ");
// 	start_t = clock();
// 
// 	int mid = (LEN/2);
// 	for (int nl = 0; nl < 4*ntimes; nl++) {
// 		for (int i = 0; i < LEN; i++) {
// 			if (i+1 < mid) {
// 				a[i] += b[i] * c[i];
// 			} else {
// 				a[i] += b[i] * d[i];
// 			}
// 		}
// 		dummy(a, b, c, d, e, aa, bb, cc, 0.);
// 	}
// 	end_t = clock(); clock_dif = end_t - start_t;
// 	clock_dif_sec = (double) (clock_dif/1000000.0);
// 	printf("S276\t %.2f \t\t", clock_dif_sec);
// 	check(1);
// 	return 0;
// }