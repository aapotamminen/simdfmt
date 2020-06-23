#include <emmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>

void printm128i(__m128i x)
{
    uint8_t xx[16];
    _mm_storeu_si128((__m128i *) xx, x);
    int i;
    for (i = 0; i < 16; i++)
        printf("%02x ", xx[i]);
    printf("\n");
}

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
    uint16_t x, y1, y2, y3, y4;
    int i, j;
    char xbuf[10];
    size_t n = 0;

    for (i = 0; i < 8; i++) {
        x = xx[i];
        y1 = x % 10;
        x /= 10;
        y2 = x % 10;
        x /= 10;
        y3 = x % 10;
        x /= 10;
        y4 = x % 10;
        x /= 10;
        xbuf[0] = x + '0';
        xbuf[1] = y4 + '0';
        xbuf[2] = y3 + '0';
        xbuf[3] = y2 + '0';
        xbuf[4] = y1 + '0';
        for (j = 0; j < 4 && xbuf[j] == '0'; j++);
        memcpy(&buf[n], &xbuf[j], 5);
        n += 5 - j;
        buf[n++] = ',';
    }
    buf[n] = 0;
    return n;
}

size_t fmt_u32_div10(char *buf, const uint32_t* xx)
{
    uint32_t x, y1, y2, y3, y4, y5, y6, y7, y8, y9;
    int i, j;
    char xbuf[10];
    size_t n = 0;

    for (i = 0; i < 8; i++) {
        x = xx[i];
        y1 = x % 10;
        x /= 10;
        y2 = x % 10;
        x /= 10;
        y3 = x % 10;
        x /= 10;
        y4 = x % 10;
        x /= 10;
        y5 = x % 10;
        x /= 10;
        y6 = x % 10;
        x /= 10;
        y7 = x % 10;
        x /= 10;
        y8 = x % 10;
        x /= 10;
        y9 = x % 10;
        x /= 10;
        xbuf[0] = x + '0';
        xbuf[1] = y9 + '0';
        xbuf[2] = y8 + '0';
        xbuf[3] = y7 + '0';
        xbuf[4] = y6 + '0';
        xbuf[5] = y5 + '0';
        xbuf[6] = y4 + '0';
        xbuf[7] = y3 + '0';
        xbuf[8] = y2 + '0';
        xbuf[9] = y1 + '0';
        for (j = 0; j < 9 && xbuf[j] == '0'; j++);
        memcpy(&buf[n], &xbuf[j], 10);
        n += 10 - j;
        buf[n++] = ',';
    }
    buf[n] = 0;
    return n;
}

char *fmt_div100_digits;

void fmt_div100_init(void)
{
    fmt_div100_digits = malloc(100 * 2);

    int i;
    for (i = 0; i < 100; i++) {
        sprintf(&fmt_div100_digits[2 * i], "%02d", i);
    }
}

size_t fmt_u16_div100(char *buf, const uint16_t* xx)
{
    uint16_t x, y1, y2;
    int i, j;
    char xbuf[8];
    size_t n = 0;

    for (i = 0; i < 8; i++) {
        x = xx[i];
        y1 = x % 100;
        x /= 100;
        y2 = x % 100;
        x /= 100;
        xbuf[0] = x + '0';
        memcpy(&xbuf[1], &fmt_div100_digits[2 * y2], 2);
        memcpy(&xbuf[3], &fmt_div100_digits[2 * y1], 2);
        for (j = 0; j < 4 && xbuf[j] == '0'; j++);
        memcpy(&buf[n], &xbuf[j], 5);
        n += 5 - j;
        buf[n++] = ',';
    }
    buf[n] = 0;
    return n;
}

size_t fmt_u32_div100(char *buf, const uint32_t* xx)
{
    uint32_t x, y1, y2, y3, y4;
    int i, j;
    char xbuf[10];
    size_t n = 0;

    for (i = 0; i < 8; i++) {
        x = xx[i];
        y1 = x % 100;
        x /= 100;
        y2 = x % 100;
        x /= 100;
        y3 = x % 100;
        x /= 100;
        y4 = x % 100;
        x /= 100;
        memcpy(&xbuf[0], &fmt_div100_digits[2 * x], 2);
        memcpy(&xbuf[2], &fmt_div100_digits[2 * y4], 2);
        memcpy(&xbuf[4], &fmt_div100_digits[2 * y3], 2);
        memcpy(&xbuf[6], &fmt_div100_digits[2 * y2], 2);
        memcpy(&xbuf[8], &fmt_div100_digits[2 * y1], 2);
        for (j = 0; j < 9 && xbuf[j] == '0'; j++);
        memcpy(&buf[n], &xbuf[j], 10);
        n += 10 - j;
        buf[n++] = ',';
    }
    buf[n] = 0;
    return n;
}

char *fmt_div1000_digits;

void fmt_div1000_init(void)
{
    fmt_div1000_digits = malloc(1000 * 4);

    int i;
    for (i = 0; i < 1000; i++) {
        sprintf(&fmt_div1000_digits[4 * i], "%03d ", i);
    }
}

size_t fmt_u16_div1000(char *buf, const uint16_t* xx)
{
    uint16_t x, y;
    int i, j;
    char xbuf[8];
    size_t n = 0;

    for (i = 0; i < 8; i++) {
        x = xx[i];
        y = x % 1000;
        x /= 1000;
        memcpy(&xbuf[0], &fmt_div1000_digits[4 * x + 1], 2);
        memcpy(&xbuf[2], &fmt_div1000_digits[4 * y], 3);
        for (j = 0; j < 4 && xbuf[j] == '0'; j++);
        memcpy(&buf[n], &xbuf[j], 5);
        n += 5 - j;
        buf[n++] = ',';
    }
    buf[n] = 0;
    return n;
}

size_t fmt_u32_div1000(char *buf, const uint32_t* xx)
{
    uint32_t x, y1, y2, y3;
    int i, j;
    char xbuf[16] __attribute__((aligned(4)));
    size_t n = 0;

    for (i = 0; i < 8; i++) {
        x = xx[i];
        y1 = x % 1000;
        x /= 1000;
        y2 = x % 1000;
        x /= 1000;
        y3 = x % 1000;
        x /= 1000;
        xbuf[0] = fmt_div1000_digits[4 * x + 2];
        memcpy(&xbuf[1], &fmt_div1000_digits[4 * y3], 4);
        memcpy(&xbuf[4], &fmt_div1000_digits[4 * y2], 4);
        memcpy(&xbuf[7], &fmt_div1000_digits[4 * y1], 4);
        for (j = 0; j < 9 && xbuf[j] == '0'; j++);
        memcpy(&buf[n], &xbuf[j], 10);
        n += 10 - j;
        buf[n++] = ',';
    }
    buf[n] = 0;
    return n;
}

size_t fmt_u64_div1000(char *buf, const uint64_t* xx)
{
    uint64_t x, y1, y2, y3, y4, y5, y6;
    int i, j;
    char xbuf[64] __attribute__((aligned(4)));
    size_t n = 0;

    for (i = 0; i < 8; i++) {
        x = xx[i];
        y1 = x % 1000;
        x /= 1000;
        y2 = x % 1000;
        x /= 1000;
        y3 = x % 1000;
        x /= 1000;
        y4 = x % 1000;
        x /= 1000;
        y5 = x % 1000;
        x /= 1000;
        y6 = x % 1000;
        x /= 1000;
        memcpy(&xbuf[0], &fmt_div1000_digits[4 * x + 1], 2);
        memcpy(&xbuf[2], &fmt_div1000_digits[4 * y6], 4);
        memcpy(&xbuf[5], &fmt_div1000_digits[4 * y5], 4);
        memcpy(&xbuf[8], &fmt_div1000_digits[4 * y4], 4);
        memcpy(&xbuf[11], &fmt_div1000_digits[4 * y3], 4);
        memcpy(&xbuf[14], &fmt_div1000_digits[4 * y2], 4);
        memcpy(&xbuf[17], &fmt_div1000_digits[4 * y1], 4);
        for (j = 0; j < 19 && xbuf[j] == '0'; j++);
        memcpy(&buf[n], &xbuf[j], 20);
        n += 20 - j;
        buf[n++] = ',';
    }
    buf[n] = 0;
    return n;
}

char *fmt_div10000_digits;

void fmt_div10000_init(void)
{
    fmt_div10000_digits = malloc(10000 * 4);

    int i;
    for (i = 0; i < 10000; i++) {
        sprintf(&fmt_div10000_digits[4 * i], "%04d", i);
    }
}

size_t fmt_u16_div10000(char *buf, const uint16_t* xx)
{
    uint16_t x, y;
    int i, j;
    char xbuf[8];
    size_t n = 0;

    for (i = 0; i < 8; i++) {
        x = xx[i];
        y = x % 10000;
        x /= 10000;
        xbuf[0] = fmt_div10000_digits[4 * x + 3];
        memcpy(&xbuf[1], &fmt_div10000_digits[4 * y], 4);
        for (j = 0; j < 4 && xbuf[j] == '0'; j++);
        memcpy(&buf[n], &xbuf[j], 5);
        n += 5 - j;
        buf[n++] = ',';
    }
    buf[n] = 0;
    return n;
}

size_t fmt_u32_div10000(char *buf, const uint32_t* xx)
{
    uint32_t x, y1, y2;
    int i, j;
    char xbuf[16] __attribute__((aligned(4)));
    size_t n = 0;

    for (i = 0; i < 8; i++) {
        x = xx[i];
        y1 = x % 10000;
        x /= 10000;
        y2 = x % 10000;
        x /= 10000;
        memcpy(&xbuf[0], &fmt_div10000_digits[4 * x + 2], 2);
        memcpy(&xbuf[2], &fmt_div10000_digits[4 * y2], 4);
        memcpy(&xbuf[6], &fmt_div10000_digits[4 * y1], 4);
        for (j = 0; j < 9 && xbuf[j] == '0'; j++);
        memcpy(&buf[n], &xbuf[j], 10);
        n += 10 - j;
        buf[n++] = ',';
    }
    buf[n] = 0;
    return n;
}

size_t fmt_u32_div10000_sse(char *buf, const uint32_t* xx)
{
    static const uint8_t sm[12 * 16] __attribute__((aligned(16))) = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, -1, -1, -1, -1,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, -1, -1, -1, -1, -1,
        2, 3, 4, 5, 6, 7, 8, 9, 10, 11, -1, -1, -1, -1, -1, -1,
        3, 4, 5, 6, 7, 8, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1,
        4, 5, 6, 7, 8, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1,
        5, 6, 7, 8, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        6, 7, 8, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        7, 8, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        8, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        9, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        10, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
    };

    uint32_t x, y1, y2;
    int i, j;
    size_t n = 0;

    for (i = 0; i < 8; i++) {
        x = xx[i];
        y1 = x % 10000;
        x /= 10000;
        y2 = x % 10000;
        x /= 10000;
        __m128i a = _mm_set1_epi8(0);
        a = _mm_insert_epi32(a, *(uint32_t *) &fmt_div10000_digits[4 * x], 0);
        a = _mm_insert_epi32(a, *(uint32_t *) &fmt_div10000_digits[4 * y2], 1);
        a = _mm_insert_epi32(a, *(uint32_t *) &fmt_div10000_digits[4 * y1], 2);
        j = __builtin_ctz(0x800 | ~_mm_movemask_epi8(_mm_cmpeq_epi8(a, _mm_set1_epi8('0'))));
        a = _mm_shuffle_epi8(a, *(__m128i *) &sm[16 * j]);
        _mm_storeu_si128((__m128i *) &buf[n], a);
        n += 12 - j;
        buf[n++] = ',';
    }
    buf[n] = 0;
    return n;
}

size_t fmt_u64_div10000(char *buf, const uint64_t* xx)
{
    uint64_t x, y1, y2, y3, y4;
    int i, j;
    char xbuf[64] __attribute__((aligned(4)));
    size_t n = 0;

    for (i = 0; i < 8; i++) {
        x = xx[i];
        y1 = x % 10000;
        x /= 10000;
        y2 = x % 10000;
        x /= 10000;
        y3 = x % 10000;
        x /= 10000;
        y4 = x % 10000;
        x /= 10000;
        memcpy(&xbuf[0], &fmt_div10000_digits[4 * x], 4);
        memcpy(&xbuf[4], &fmt_div10000_digits[4 * y4], 4);
        memcpy(&xbuf[8], &fmt_div10000_digits[4 * y3], 4);
        memcpy(&xbuf[12], &fmt_div10000_digits[4 * y2], 4);
        memcpy(&xbuf[16], &fmt_div10000_digits[4 * y1], 4);
        for (j = 0; j < 19 && xbuf[j] == '0'; j++);
        memcpy(&buf[n], &xbuf[j], 20);
        n += 20 - j;
        buf[n++] = ',';
    }
    buf[n] = 0;
    return n;
}

size_t fmt_u64_div10000_sse(char *buf, const uint64_t* xx)
{
    static const uint8_t sm[17 * 16] __attribute__((aligned(16))) = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -1,
        2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -1, -1,
        3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -1, -1, -1,
        4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -1, -1, -1, -1,
        5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -1, -1, -1, -1, -1,
        6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1,
        7, 8, 9, 10, 11, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1,
        8, 9, 10, 11, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1,
        9, 10, 11, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        10, 11, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        11, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
    };

    uint64_t x, y1, y2, y3, y4;
    int i, j;
    size_t n = 0;

    for (i = 0; i < 8; i++) {
        x = xx[i];
        y1 = x % 10000;
        x /= 10000;
        y2 = x % 10000;
        x /= 10000;
        y3 = x % 10000;
        x /= 10000;
        y4 = x % 10000;
        x /= 10000;
        __m128i a = _mm_set1_epi8(0);
        a = _mm_insert_epi32(a, *(uint32_t *) &fmt_div10000_digits[4 * x], 0);
        a = _mm_insert_epi32(a, *(uint32_t *) &fmt_div10000_digits[4 * y4], 1);
        a = _mm_insert_epi32(a, *(uint32_t *) &fmt_div10000_digits[4 * y3], 2);
        a = _mm_insert_epi32(a, *(uint32_t *) &fmt_div10000_digits[4 * y2], 3);
        j = __builtin_ctz(0x10000 | ~_mm_movemask_epi8(_mm_cmpeq_epi8(a, _mm_set1_epi8('0'))));
        a = _mm_shuffle_epi8(a, *(__m128i *) &sm[16 * j]);
        _mm_storeu_si128((__m128i *) &buf[n], a);
        n += 16 - j;
        if (j < 16) {
            memcpy(&buf[n], &fmt_div10000_digits[4 * y1], 4);
            n += 4;
        } else {
            a = _mm_set1_epi8(0);
            a = _mm_insert_epi32(a, *(uint32_t *) &fmt_div10000_digits[4 * y1], 0);
            j = __builtin_ctz(0x8 | ~_mm_movemask_epi8(_mm_cmpeq_epi8(a, _mm_set1_epi8('0'))));
            a = _mm_shuffle_epi8(a, *(__m128i *) &sm[16 * j]);
            _mm_storeu_si128((__m128i *) &buf[n], a);
            n += 4 - j;
        }
        buf[n++] = ',';
    }
    buf[n] = 0;
    return n;
}

char *fmt_div100000_digits;

void fmt_div100000_init(void)
{
    fmt_div100000_digits = malloc(100000 * 5);

    int i;
    for (i = 0; i < 100000; i++) {
        sprintf(&fmt_div100000_digits[5 * i], "%05d", i);
    }
}

size_t fmt_u32_div100000(char *buf, const uint32_t* xx)
{
    uint32_t x, y;
    int i, j;
    char xbuf[10];
    size_t n = 0;

    for (i = 0; i < 8; i++) {
        x = xx[i];
        y = x % 100000;
        x /= 100000;
        memcpy(&xbuf[0], &fmt_div100000_digits[5 * x], 5);
        memcpy(&xbuf[5], &fmt_div100000_digits[5 * y], 5);
        for (j = 0; j < 9 && xbuf[j] == '0'; j++);
        memcpy(&buf[n], &xbuf[j], 10);
        n += 10 - j;
        buf[n++] = ',';
    }
    buf[n] = 0;
    return n;
}

size_t fmt_u64_div100000(char *buf, const uint64_t* xx)
{
    uint64_t x, y1, y2, y3;
    int i, j;
    char xbuf[64];
    size_t n = 0;

    for (i = 0; i < 8; i++) {
        x = xx[i];
        y1 = x % 100000;
        x /= 100000;
        y2 = x % 100000;
        x /= 100000;
        y3 = x % 100000;
        x /= 100000;
        memcpy(&xbuf[0], &fmt_div100000_digits[5 * x], 5);
        memcpy(&xbuf[5], &fmt_div100000_digits[5 * y3], 5);
        memcpy(&xbuf[10], &fmt_div100000_digits[5 * y2], 5);
        memcpy(&xbuf[15], &fmt_div100000_digits[5 * y1], 5);
        for (j = 0; j < 19 && xbuf[j] == '0'; j++);
        memcpy(&buf[n], &xbuf[j], 20);
        n += 20 - j;
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
        fmt_u16_table_len[i] = sprintf(&fmt_u16_table_digits[6 * i], "%u,", i);
        sprintf(&fmt_u16_table_digits[6 * i], "%05u,", i);
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
    uint32_t *xu32;
    uint64_t *xu64;
    const int methods = 8;
    char *buf[methods];
    size_t len[methods];
    int ok[methods];
    int m;
    int k;
    const int rep = 12500000;
    const size_t size = (size_t)rep * 8 * 21 + 1;
    struct timeval start, stop;
    double time;
    char *p;

    for (k = 0; k < methods; k++)
        buf[k] = malloc(size);

    xx = malloc(rep * 8 * sizeof(uint16_t));
    for (k = 0; k < rep * 8; k++)
        xx[k] = rand();

    fmt_div100_init();
    fmt_div1000_init();
    fmt_div10000_init();
    fmt_u16_table_init();
    fmt_div100000_init();

    m = 0;
#if 0
    printf("%-16s", "u16 sprintf:");
    gettimeofday(&start, NULL);
    p = buf[m];
    for (k = 0; k < rep * 8; k++) {
        p += sprintf(p, "%hu,", xx[k]);
    }
    len[m] = p - buf[m];
    ok[m] = 1;
    gettimeofday(&stop, NULL);
    time = stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) / 1000000.0;
    printf("%zu %d %.2f %.2f\n", len[m], ok[m], time, rep * 8 / time);
#endif

    m++;
    printf("%-16s", "u16 d10:");
    gettimeofday(&start, NULL);
    p = buf[m];
    for (k = 0; k < rep; k++) {
        p += fmt_u16_div10(p, &xx[k * 8]);
    }
    len[m] = p - buf[m];
    ok[m] = (len[m] == len[0] && memcmp(buf[0], buf[m], len[0]) == 0);
    gettimeofday(&stop, NULL);
    time = stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) / 1000000.0;
    printf("%zu %d %.2f %.2f\n", len[m], ok[m], time, rep * 8 / time);

    m++;
    printf("%-16s", "u16 d100:");
    gettimeofday(&start, NULL);
    p = buf[m];
    for (k = 0; k < rep; k++) {
        p += fmt_u16_div100(p, &xx[k * 8]);
    }
    len[m] = p - buf[m];
    ok[m] = (len[m] == len[0] && memcmp(buf[0], buf[m], len[0]) == 0);
    gettimeofday(&stop, NULL);
    time = stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) / 1000000.0;
    printf("%zu %d %.2f %.2f\n", len[m], ok[m], time, rep * 8 / time);

    m++;
    printf("%-16s", "u16 d1000:");
    gettimeofday(&start, NULL);
    p = buf[m];
    for (k = 0; k < rep; k++) {
        p += fmt_u16_div1000(p, &xx[k * 8]);
    }
    len[m] = p - buf[m];
    ok[m] = (len[m] == len[0] && memcmp(buf[0], buf[m], len[0]) == 0);
    gettimeofday(&stop, NULL);
    time = stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) / 1000000.0;
    printf("%zu %d %.2f %.2f\n", len[m], ok[m], time, rep * 8 / time);

    m++;
    printf("%-16s", "u16 d10000:");
    gettimeofday(&start, NULL);
    p = buf[m];
    for (k = 0; k < rep; k++) {
        p += fmt_u16_div10000(p, &xx[k * 8]);
    }
    len[m] = p - buf[m];
    ok[m] = (len[m] == len[0] && memcmp(buf[0], buf[m], len[0]) == 0);
    gettimeofday(&stop, NULL);
    time = stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) / 1000000.0;
    printf("%zu %d %.2f %.2f\n", len[m], ok[m], time, rep * 8 / time);

    m++;
    printf("%-16s", "u16 table:");
    gettimeofday(&start, NULL);
    p = buf[m];
    for (k = 0; k < rep; k++) {
        p += fmt_u16_table(p, &xx[k * 8]);
    }
    len[m] = p - buf[m];
    ok[m] = (len[m] == len[0] && memcmp(buf[0], buf[m], len[0]) == 0);
    gettimeofday(&stop, NULL);
    time = stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) / 1000000.0;
    printf("%zu %d %.2f %.2f\n", len[m], ok[m], time, rep * 8 / time);

    m++;
    printf("%-16s", "u16 par sse:");
    gettimeofday(&start, NULL);
    p = buf[m];
    for (k = 0; k < rep; k++) {
        p += fmt_u16_sse(p, &xx[k * 8]);
    }
    len[m] = p - buf[m];
    ok[m] = (len[m] == len[0] && memcmp(buf[0], buf[m], len[0]) == 0);
    gettimeofday(&stop, NULL);
    time = stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) / 1000000.0;
    printf("%zu %d %.2f %.2f\n", len[m], ok[m], time, rep * 8 / time);

    xu32 = malloc(rep * 8 * sizeof(uint32_t));
    for (k = 0; k < rep * 8; k++)
        xu32[k] = rand();

    m = 0;
#if 0
    printf("%-16s", "u32 sprintf:");
    gettimeofday(&start, NULL);
    p = buf[m];
    for (k = 0; k < rep * 8; k++) {
        p += sprintf(p, "%u,", xu32[k]);
    }
    len[m] = p - buf[m];
    ok[m] = 1;
    gettimeofday(&stop, NULL);
    time = stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) / 1000000.0;
    printf("%zu %d %.2f %.2f\n", len[m], ok[m], time, rep * 8 / time);
#endif

    m++;
    printf("%-16s", "u32 d10:");
    gettimeofday(&start, NULL);
    p = buf[m];
    for (k = 0; k < rep; k++) {
        p += fmt_u32_div10(p, &xu32[k * 8]);
    }
    len[m] = p - buf[m];
    ok[m] = (len[m] == len[0] && memcmp(buf[0], buf[m], len[0]) == 0);
    gettimeofday(&stop, NULL);
    time = stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) / 1000000.0;
    printf("%zu %d %.2f %.2f\n", len[m], ok[m], time, rep * 8 / time);

    m++;
    printf("%-16s", "u32 d100:");
    gettimeofday(&start, NULL);
    p = buf[m];
    for (k = 0; k < rep; k++) {
        p += fmt_u32_div100(p, &xu32[k * 8]);
    }
    len[m] = p - buf[m];
    ok[m] = (len[m] == len[0] && memcmp(buf[0], buf[m], len[0]) == 0);
    gettimeofday(&stop, NULL);
    time = stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) / 1000000.0;
    printf("%zu %d %.2f %.2f\n", len[m], ok[m], time, rep * 8 / time);

    m++;
    printf("%-16s", "u32 d1000:");
    gettimeofday(&start, NULL);
    p = buf[m];
    for (k = 0; k < rep; k++) {
        p += fmt_u32_div1000(p, &xu32[k * 8]);
    }
    len[m] = p - buf[m];
    ok[m] = (len[m] == len[0] && memcmp(buf[0], buf[m], len[0]) == 0);
    gettimeofday(&stop, NULL);
    time = stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) / 1000000.0;
    printf("%zu %d %.2f %.2f\n", len[m], ok[m], time, rep * 8 / time);

    m++;
    printf("%-16s", "u32 d10000:");
    gettimeofday(&start, NULL);
    p = buf[m];
    for (k = 0; k < rep; k++) {
        p += fmt_u32_div10000(p, &xu32[k * 8]);
    }
    len[m] = p - buf[m];
    ok[m] = (len[m] == len[0] && memcmp(buf[0], buf[m], len[0]) == 0);
    gettimeofday(&stop, NULL);
    time = stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) / 1000000.0;
    printf("%zu %d %.2f %.2f\n", len[m], ok[m], time, rep * 8 / time);

    m++;
    printf("%-16s", "u32 d10000 sse:");
    gettimeofday(&start, NULL);
    p = buf[m];
    for (k = 0; k < rep; k++) {
        p += fmt_u32_div10000_sse(p, &xu32[k * 8]);
    }
    len[m] = p - buf[m];
    ok[m] = (len[m] == len[0] && memcmp(buf[0], buf[m], len[0]) == 0);
    gettimeofday(&stop, NULL);
    time = stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) / 1000000.0;
    printf("%zu %d %.2f %.2f\n", len[m], ok[m], time, rep * 8 / time);

    m++;
    printf("%-16s", "u32 d100000:");
    gettimeofday(&start, NULL);
    p = buf[m];
    for (k = 0; k < rep; k++) {
        p += fmt_u32_div100000(p, &xu32[k * 8]);
    }
    len[m] = p - buf[m];
    ok[m] = (len[m] == len[0] && memcmp(buf[0], buf[m], len[0]) == 0);
    gettimeofday(&stop, NULL);
    time = stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) / 1000000.0;
    printf("%zu %d %.2f %.2f\n", len[m], ok[m], time, rep * 8 / time);

    xu64 = malloc(rep * 8 * sizeof(uint64_t));
    for (k = 0; k < rep * 8; k++)
        xu64[k] = (((uint64_t)rand()) << 32) | rand();

    m = 0;
#if 0
    printf("%-16s", "u64 sprintf:");
    gettimeofday(&start, NULL);
    p = buf[m];
    for (k = 0; k < rep * 8; k++) {
        p += sprintf(p, "%lu,", xu64[k]);
    }
    len[m] = p - buf[m];
    ok[m] = 1;
    gettimeofday(&stop, NULL);
    time = stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) / 1000000.0;
    printf("%zu %d %.2f %.2f\n", len[m], ok[m], time, rep * 8 / time);
#endif

    m++;
    printf("%-16s", "u64 d1000:");
    gettimeofday(&start, NULL);
    p = buf[m];
    for (k = 0; k < rep; k++) {
        p += fmt_u64_div1000(p, &xu64[k * 8]);
    }
    len[m] = p - buf[m];
    ok[m] = (len[m] == len[0] && memcmp(buf[0], buf[m], len[0]) == 0);
    gettimeofday(&stop, NULL);
    time = stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) / 1000000.0;
    printf("%zu %d %.2f %.2f\n", len[m], ok[m], time, rep * 8 / time);

    m++;
    printf("%-16s", "u64 d10000:");
    gettimeofday(&start, NULL);
    p = buf[m];
    for (k = 0; k < rep; k++) {
        p += fmt_u64_div10000(p, &xu64[k * 8]);
    }
    len[m] = p - buf[m];
    ok[m] = (len[m] == len[0] && memcmp(buf[0], buf[m], len[0]) == 0);
    gettimeofday(&stop, NULL);
    time = stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) / 1000000.0;
    printf("%zu %d %.2f %.2f\n", len[m], ok[m], time, rep * 8 / time);

    m++;
    printf("%-16s", "u64 d10000 sse:");
    gettimeofday(&start, NULL);
    p = buf[m];
    for (k = 0; k < rep; k++) {
        p += fmt_u64_div10000_sse(p, &xu64[k * 8]);
    }
    len[m] = p - buf[m];
    ok[m] = (len[m] == len[0] && memcmp(buf[0], buf[m], len[0]) == 0);
    gettimeofday(&stop, NULL);
    time = stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) / 1000000.0;
    printf("%zu %d %.2f %.2f\n", len[m], ok[m], time, rep * 8 / time);

    m++;
    printf("%-16s", "u64 d100000:");
    gettimeofday(&start, NULL);
    p = buf[m];
    for (k = 0; k < rep; k++) {
        p += fmt_u64_div100000(p, &xu64[k * 8]);
    }
    len[m] = p - buf[m];
    ok[m] = (len[m] == len[0] && memcmp(buf[0], buf[m], len[0]) == 0);
    gettimeofday(&stop, NULL);
    time = stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) / 1000000.0;
    printf("%zu %d %.2f %.2f\n", len[m], ok[m], time, rep * 8 / time);

    free(xu64);
    free(xu32);
    free(xx);
    for (k = 0; k < methods; k++)
        free(buf[k]);
    return 0;
}
