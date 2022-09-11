#define LEN 32000
#define LEN2 256

#define ntimes 50000
 float a[LEN],b[LEN],c[LEN],d[LEN],e[LEN],
                                   aa[LEN2][LEN2],bb[LEN2][LEN2],cc[LEN2][LEN2],tt[LEN2][LEN2];
int s276(int q)
{
	int mid = (LEN/2);
	for (int nl = 0; nl < 4*ntimes; nl++) {
		for (int i = 0; i < LEN; i++) {
			if (i+1 < mid) {
				a[i] += 1;
			} else {
				a[i] += 2;
			}
		}
	}
	return 0;
}