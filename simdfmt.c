#include <emmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>

uint8_t ss[5*16] __attribute__((aligned(16))) = {
    0, 2, 4, 6, 8, 9, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    2, 4, 6, 8, 9, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    4, 6, 8, 9, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    6, 8, 9, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    8, 9, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255
};

uint8_t ss1[5*16] __attribute__((aligned(16))) = {
    0, 2, 4, 6, 10, 11, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    2, 4, 6, 10, 11, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    4, 6, 10, 11, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    6, 10, 11, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    10, 11, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255
};

uint8_t ss2[5*16] __attribute__((aligned(16))) = {
    0, 2, 4, 6, 12, 13, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    2, 4, 6, 12, 13, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    4, 6, 12, 13, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    6, 12, 13, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    12, 13, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255
};

uint8_t ss3[5*16] __attribute__((aligned(16))) = {
    0, 2, 4, 6, 14, 15, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    2, 4, 6, 14, 15, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    4, 6, 14, 15, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    6, 14, 15, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    14, 15, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255
};

size_t fmt_u16_sse(char *buf, const uint16_t *xx)
{
    __m128i const10 = _mm_set1_epi16(10);
    __m128i const10inv = _mm_set1_epi16(0xcccd);
    __m128i const10inv2 = _mm_set1_epi16(0x199a);
    __m128i const100 = _mm_set1_epi16(100);
    __m128i const100inv = _mm_set1_epi16(0xa3d8);
    __m128i zero = _mm_set1_epi16('0');
    __m128i zero8 = _mm_set1_epi8('0');
    __m128i comma = _mm_set1_epi8(',');
    __m128i es = _mm_setr_epi8(0, 2, 4, 6, -1, -1, -1, -1, 8, 10, 12, 14, -1, -1, -1, -1);
    __m128i x = _mm_loadu_si128((__m128i *) xx);

    /* e = least significant digit (digit 0) */
    /* z = x / 10 */
    __m128i z = _mm_srli_epi16(_mm_mulhi_epu16(x, const10inv), 3);
    /* x = x - z * 10 = x % 10 */
    x = _mm_sub_epi16(x, _mm_mullo_epi16(z, const10));
    /* e = x + '0' */
    __m128i e = _mm_add_epi16(x, zero);

    /* Divide z by 100: x = z % 100, z = z / 100 */
    x = z;
    z = _mm_srli_epi16(_mm_mulhi_epu16(z, const100inv), 6);
    x = _mm_sub_epi16(x, _mm_mullo_epi16(z, const100));

    /* Divide x by 10: x = x % 10, y = x / 10 */
    __m128i y = _mm_mulhi_epu16(x, const10inv2);
    x = _mm_sub_epi16(x, _mm_mullo_epi16(y, const10));
    /* d = digit 1 */
    __m128i d = _mm_add_epi16(x, zero);
    /* c = digit 2 */
    __m128i c = _mm_add_epi16(y, zero);

    /* Divide z by 10: z = z % 10, y = z / 10 */
    y = _mm_mulhi_epu16(z, const10inv2);
    z = _mm_sub_epi16(z, _mm_mullo_epi16(y, const10));
    /* b = digit 1 */
    __m128i b = _mm_add_epi16(z, zero);
    /* a = digit 2 */
    __m128i a = _mm_add_epi16(y, zero);

    /* Gather least significant digits and interleave with commas */
    __m128i ee = _mm_shuffle_epi8(e, es);
    __m128i ec0 = _mm_unpacklo_epi8(ee, comma);
    __m128i ec1 = _mm_unpackhi_epi8(ee, comma);

    size_t n = 0;
    uint64_t zu;
    __m128i h;
    __m128i txt;
    uint64_t nz;

    __m128i f = _mm_unpacklo_epi16(a, b);
    __m128i g = _mm_unpacklo_epi16(c, d);

    /* Format xx[0] */
    /* Gather digits 4 - 1 */
    h = (__m128i)_mm_shuffle_ps((__m128)f, (__m128)g, 0x00);
    h = _mm_shuffle_epi32(h, 0x08);
    /* Combine with digit 0 + comma */
    h = (__m128i)_mm_shuffle_ps((__m128)h, (__m128)ec0, 0x44);
    /* Count number of leading zeros in digits 4 - 1 */
    nz = _mm_cmpistri(h, zero, _SIDD_UWORD_OPS | _SIDD_CMP_EQUAL_EACH | _SIDD_NEGATIVE_POLARITY | _SIDD_LEAST_SIGNIFICANT);
    /* Skip leading zeros, join with digit 0 and comma, store to buffer */
    txt = _mm_shuffle_epi8(h, *(__m128i *) &ss[16 * nz]);
    _mm_storeu_si128((__m128i *) &buf[n], txt);
    n += 6 - nz;

    /* Format xx[1] */
    h = (__m128i)_mm_shuffle_ps((__m128)f, (__m128)g, 0x55);
    h = _mm_shuffle_epi32(h, 0x08);
    h = (__m128i)_mm_shuffle_ps((__m128)h, (__m128)ec0, 0x44);
    nz = _mm_cmpistri(h, zero, _SIDD_UWORD_OPS | _SIDD_CMP_EQUAL_EACH | _SIDD_NEGATIVE_POLARITY | _SIDD_LEAST_SIGNIFICANT);
    txt =_mm_shuffle_epi8(h, *(__m128i *) &ss1[16 * nz]);
    _mm_storeu_si128((__m128i *) &buf[n], txt);
    n += 6 - nz;

    /* Format xx[2] */
    h = (__m128i)_mm_shuffle_ps((__m128)f, (__m128)g, 0xaa);
    h = _mm_shuffle_epi32(h, 0x08);
    h = (__m128i)_mm_shuffle_ps((__m128)h, (__m128)ec0, 0x44);
    nz = _mm_cmpistri(h, zero, _SIDD_UWORD_OPS | _SIDD_CMP_EQUAL_EACH | _SIDD_NEGATIVE_POLARITY | _SIDD_LEAST_SIGNIFICANT);
    txt =_mm_shuffle_epi8(h, *(__m128i *) &ss2[16 * nz]);
    _mm_storel_epi64((__m128i *) &buf[n], txt);
    n += 6 - nz;

    /* Format xx[3] */
    h = (__m128i)_mm_shuffle_ps((__m128)f, (__m128)g, 0xff);
    h = _mm_shuffle_epi32(h, 0x08);
    h = (__m128i)_mm_shuffle_ps((__m128)h, (__m128)ec0, 0x44);
    nz = _mm_cmpistri(h, zero, _SIDD_UWORD_OPS | _SIDD_CMP_EQUAL_EACH | _SIDD_NEGATIVE_POLARITY | _SIDD_LEAST_SIGNIFICANT);
    txt =_mm_shuffle_epi8(h, *(__m128i *) &ss3[16 * nz]);
    _mm_storel_epi64((__m128i *) &buf[n], txt);
    n += 6 - nz;

    f = _mm_unpackhi_epi16(a, b);
    g = _mm_unpackhi_epi16(c, d);

    /* Format xx[4] */
    h = (__m128i)_mm_shuffle_ps((__m128)f, (__m128)g, 0x00);
    h = _mm_shuffle_epi32(h, 0x08);
    h = (__m128i)_mm_shuffle_ps((__m128)h, (__m128)ec1, 0x44);
    nz = _mm_cmpistri(h, zero, _SIDD_UWORD_OPS | _SIDD_CMP_EQUAL_EACH | _SIDD_NEGATIVE_POLARITY | _SIDD_LEAST_SIGNIFICANT);
    txt =_mm_shuffle_epi8(h, *(__m128i *) &ss[16 * nz]);
    _mm_storel_epi64((__m128i *) &buf[n], txt);
    n += 6 - nz;

    /* Format xx[5] */
    h = (__m128i)_mm_shuffle_ps((__m128)f, (__m128)g, 0x55);
    h = _mm_shuffle_epi32(h, 0x08);
    h = (__m128i)_mm_shuffle_ps((__m128)h, (__m128)ec1, 0x44);
    nz = _mm_cmpistri(h, zero, _SIDD_UWORD_OPS | _SIDD_CMP_EQUAL_EACH | _SIDD_NEGATIVE_POLARITY | _SIDD_LEAST_SIGNIFICANT);
    txt =_mm_shuffle_epi8(h, *(__m128i *) &ss1[16 * nz]);
    _mm_storel_epi64((__m128i *) &buf[n], txt);
    n += 6 - nz;

    /* Format xx[6] */
    h = (__m128i)_mm_shuffle_ps((__m128)f, (__m128)g, 0xaa);
    h = _mm_shuffle_epi32(h, 0x08);
    h = (__m128i)_mm_shuffle_ps((__m128)h, (__m128)ec1, 0x44);
    nz = _mm_cmpistri(h, zero, _SIDD_UWORD_OPS | _SIDD_CMP_EQUAL_EACH | _SIDD_NEGATIVE_POLARITY | _SIDD_LEAST_SIGNIFICANT);
    txt =_mm_shuffle_epi8(h, *(__m128i *) &ss2[16 * nz]);
    _mm_storel_epi64((__m128i *) &buf[n], txt);
    n += 6 - nz;

    /* Format xx[7] */
    h = (__m128i)_mm_shuffle_ps((__m128)f, (__m128)g, 0xff);
    h = _mm_shuffle_epi32(h, 0x08);
    h = (__m128i)_mm_shuffle_ps((__m128)h, (__m128)ec1, 0x44);
    nz = _mm_cmpistri(h, zero, _SIDD_UWORD_OPS | _SIDD_CMP_EQUAL_EACH | _SIDD_NEGATIVE_POLARITY | _SIDD_LEAST_SIGNIFICANT);
    txt =_mm_shuffle_epi8(h, *(__m128i *) &ss3[16 * nz]);
    _mm_storel_epi64((__m128i *) &buf[n], txt);
    n += 6 - nz;

    buf[n] = 0;
    return n;
}

size_t fmt_u16_div10(char *buf, const uint16_t* xx)
{
    uint16_t x, y;
    int i, j;
    char xbuf[8];
    size_t n = 0;

    for (i = 0; i < 8; i++) {
        x = xx[i];
        for (j = 0; j < 5; j++) {
            y = x % 10;
            x /= 10;
            xbuf[j] = y;
        }
        for (j = 4; j > 0 && !xbuf[j]; j--);
        for (; j >= 0; j--)
            buf[n++] = xbuf[j] + '0';
        buf[n++] = ',';
    }
    buf[n] = 0;
    return n;
}

char *fmt_u16_div100_digits;

void fmt_u16_div100_init(void)
{
    fmt_u16_div100_digits = malloc(100 * 2);

    int i;
    for (i = 0; i < 100; i++) {
        sprintf(&fmt_u16_div100_digits[2 * i], "%02d", i);
    }
}

size_t fmt_u16_div100(char *buf, const uint16_t* xx)
{
    uint16_t x, y;
    int i, j;
    char xbuf[8];
    size_t n = 0;

    for (i = 0; i < 8; i++) {
        x = xx[i];
        y = x % 100;
        x /= 100;
        xbuf[1] = fmt_u16_div100_digits[2 * y];
        xbuf[0] = fmt_u16_div100_digits[2 * y + 1];
        y = x % 100;
        x /= 100;
        xbuf[3] = fmt_u16_div100_digits[2 * y];
        xbuf[2] = fmt_u16_div100_digits[2 * y + 1];
        xbuf[4] = x + '0';
        for (j = 4; j > 0 && xbuf[j] != '0'; j--);
        for (; j >= 0; j--)
            buf[n++] = xbuf[j];
        buf[n++] = ',';
    }
    buf[n] = 0;
    return n;
}

char *fmt_u16_table_digits;
size_t *fmt_u16_table_len;

void fmt_u16_table_init(void)
{
    fmt_u16_table_digits = malloc(65536 * 6);
    fmt_u16_table_len = malloc(65536 * sizeof(size_t));

    unsigned int i;
    for (i = 0; i < 65536; i++) {
        sprintf(&fmt_u16_table_digits[6 * i], "%05u,", i);
        fmt_u16_table_len[i] = i ? log10(i) + 1 : 2;
    }
}

size_t fmt_u16_table(char *buf, const uint16_t* xx)
{
    int i;
    size_t n = 0;
    size_t len;

    for (i = 0; i < 8; i++) {
        len = fmt_u16_table_len[xx[i]];
        memcpy(&buf[n], &fmt_u16_table_digits[6 * xx[i] + 6 - len], len);
        n += len;
    }
    buf[n] = 0;
    return n;
}

int main()
{

    //uint16_t xx[] = { 1, 12, 123, 1234, 12345, 23456, 34567, 45678 };
    uint16_t *xx;
    const int methods = 5;
    char *buf[methods];
    size_t len[methods];
    int ok[methods];
    int k;
    const int rep = 12500000;
    const size_t size = (size_t)rep * 8 * 6 + 1;
    struct timeval start, stop;
    double time;
    char *p;

    for (k = 0; k < methods; k++)
        buf[k] = malloc(size);

    xx = malloc(rep * 8 * sizeof(uint16_t));
    for (k = 0; k < rep * 8; k++)
        xx[k] = rand();

    printf("sprintf:");
    gettimeofday(&start, NULL);
    p = buf[0];
    for (k = 0; k < rep * 8; k++) {
        p += sprintf(p, "%hu,", xx[k]);
    }
    len[0] = p - buf[0];
    ok[0] = 1;
    gettimeofday(&stop, NULL);
    time = stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) / 1000000.0;
    printf("%zu %d %.2f %.2f\n", len[0], ok[0], time, rep * 8 / time);

    printf("div10: \t");
    gettimeofday(&start, NULL);
    p = buf[1];
    for (k = 0; k < rep; k++) {
        p += fmt_u16_div10(p, &xx[k * 8]);
    }
    len[1] = p - buf[1];
    ok[1] = (len[1] == len[0] && memcmp(buf[0], buf[1], len[0]) == 0);
    gettimeofday(&stop, NULL);
    time = stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) / 1000000.0;
    printf("%zu %d %.2f %.2f\n", len[1], ok[1], time, rep * 8 / time);

    printf("div100:\t");
    fmt_u16_div100_init();
    gettimeofday(&start, NULL);
    p = buf[2];
    for (k = 0; k < rep; k++) {
        p += fmt_u16_div100(p, &xx[k * 8]);
    }
    len[2] = p - buf[2];
    ok[2] = (len[2] == len[0] && memcmp(buf[0], buf[2], len[0]) == 0);
    gettimeofday(&stop, NULL);
    time = stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) / 1000000.0;
    printf("%zu %d %.2f %.2f\n", len[2], ok[2], time, rep * 8 / time);

    printf("table:\t");
    fmt_u16_table_init();
    gettimeofday(&start, NULL);
    p = buf[3];
    for (k = 0; k < rep; k++) {
        p += fmt_u16_table(p, &xx[k * 8]);
    }
    len[3] = p - buf[3];
    ok[3] = (len[3] == len[0] && memcmp(buf[0], buf[3], len[0]) == 0);
    gettimeofday(&stop, NULL);
    time = stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) / 1000000.0;
    printf("%zu %d %.2f %.2f\n", len[3], ok[3], time, rep * 8 / time);

    printf("sse:\t");
    gettimeofday(&start, NULL);
    p = buf[4];
    for (k = 0; k < rep; k++) {
        p += fmt_u16_sse(p, &xx[k * 8]);
    }
    len[4] = p - buf[4];
    ok[4] = (len[4] == len[0] && memcmp(buf[0], buf[4], len[0]) == 0);
    gettimeofday(&stop, NULL);
    time = stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) / 1000000.0;
    printf("%zu %d %.2f %.2f\n", len[4], ok[4], time, rep * 8 / time);

    free(xx);
    for (k = 0; k < methods; k++)
        free(buf[k]);
    return 0;
}
