#include "harness.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LIST 16
#define MAX_SELECTED 16

/* Defaults mirror configs/benchmark.json; scripts pass overrides as flags. */
static const int kMatrixSizes[] = {256, 512, 1024, 2048};
static const int kImageSizes[] = {512, 1024, 2048};
static const int kFilterSizes[] = {3, 5, 7};
static const int kSelftestSizes[] = {2, 3, 31};
static const int kSelftestFilterSizes[] = {1, 3, 5};

typedef enum { OP_MATRIX, OP_CONVOLUTION } op_kind;

typedef struct {
    int values[MAX_LIST];
    int count;
} int_list;

typedef struct {
    const char *command;
    const char *impls;
    int_list n, m, k;
    const char *filter;
    const char *image_path;
    const char *save_path;
    const char *csv_path;
    const char *member;
    int warmup;
    int runs;
    uint64_t seed;
    double atol;
    double rtol;
} options;

typedef struct {
    FILE *fp;
    const char *member;
} csv_writer;

typedef struct {
    op_kind op;
    const bench_impl *impl;
    int N, M, K;
    const char *filter;
    const float *A, *B;
    const uint32_t *image;
    const float *weights;
    float *out;
    size_t count;
    const double *ref;
} job;

/* ---------- small utilities ---------- */

static void set_list(int_list *list, const int *values, int count) {
    list->count = count;
    memcpy(list->values, values, (size_t)count * sizeof(int));
}

static int parse_list(const char *text, int_list *list) {
    list->count = 0;
    const char *p = text;
    while (*p) {
        char *end = NULL;
        long v = strtol(p, &end, 10);
        if (end == p || v <= 0 || v > 1000000 || list->count == MAX_LIST)
            return 0;
        list->values[list->count++] = (int)v;
        if (*end == ',')
            end++;
        else if (*end != '\0')
            return 0;
        p = end;
    }
    return list->count > 0;
}

static const char *status_name(int status) {
    switch (status) {
    case LAB6_OK:
        return "ok";
    case LAB6_INVALID_ARGUMENT:
        return "invalid argument";
    case LAB6_RUNTIME_ERROR:
        return "runtime error";
    case LAB6_NOT_IMPLEMENTED:
        return "not implemented";
    default:
        return "unknown status";
    }
}

static const char *op_name(op_kind op) {
    return op == OP_MATRIX ? "matrix" : "convolution";
}

static int supports(const bench_impl *impl, op_kind op) {
    return op == OP_MATRIX ? impl->matmul != NULL : impl->convolve != NULL;
}

static void *xmalloc(size_t bytes) {
    void *p = malloc(bytes ? bytes : 1);
    if (!p)
        fprintf(stderr, "out of memory (%zu bytes)\n", bytes);
    return p;
}

/* splitmix64: deterministic and identical on every platform. */
static uint64_t rng_next(uint64_t *state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static float rng_unit(uint64_t *state) {
    return (float)(rng_next(state) >> 40) * (1.0f / 16777216.0f);
}

static void fill_unit(float *v, size_t count, uint64_t *state) {
    for (size_t i = 0; i < count; i++)
        v[i] = rng_unit(state);
}

static void fill_pixels(uint32_t *v, size_t count, uint64_t *state) {
    for (size_t i = 0; i < count; i++)
        v[i] = (uint32_t)(rng_next(state) % 256u);
}

static void fill_nan(float *v, size_t count) {
    for (size_t i = 0; i < count; i++)
        v[i] = NAN;
}

/* ---------- filters and double-precision references ---------- */

/*
 * mean: 1/K^2 everywhere. edge: -1 everywhere, K^2-1 at the center.
 * sharpen: 2*identity - mean. All defined for any odd K.
 */
static int make_filter(const char *name, int K, float *f) {
    const int count = K * K;
    const int center = count / 2;
    if (strcmp(name, "mean") == 0) {
        for (int i = 0; i < count; i++)
            f[i] = 1.0f / (float)count;
    } else if (strcmp(name, "edge") == 0) {
        for (int i = 0; i < count; i++)
            f[i] = -1.0f;
        f[center] = (float)(count - 1);
    } else if (strcmp(name, "sharpen") == 0) {
        for (int i = 0; i < count; i++)
            f[i] = -1.0f / (float)count;
        f[center] += 2.0f;
    } else {
        return 0;
    }
    return 1;
}

static void ref_matmul(const float *A, const float *B, double *R, int N) {
    const size_t n = (size_t)N;
    for (size_t i = 0; i < n; i++) {
        double *row = R + i * n;
        for (size_t j = 0; j < n; j++)
            row[j] = 0.0;
        for (size_t k = 0; k < n; k++) {
            const double a = A[i * n + k];
            const float *b = B + k * n;
            for (size_t j = 0; j < n; j++)
                row[j] += a * b[j];
        }
    }
}

static void ref_convolve(const uint32_t *image, const float *f, double *R,
                         int M, int K) {
    const int r = K / 2;
    for (int y = 0; y < M; y++) {
        for (int x = 0; x < M; x++) {
            double sum = 0.0;
            for (int i = 0; i < K; i++) {
                const int iy = y + r - i;
                if (iy < 0 || iy >= M)
                    continue;
                for (int j = 0; j < K; j++) {
                    const int ix = x + r - j;
                    if (ix >= 0 && ix < M)
                        sum += (double)f[i * K + j] *
                               (double)image[(size_t)iy * M + ix];
                }
            }
            R[(size_t)y * M + x] = sum;
        }
    }
}

static int compare(const float *got, const double *ref, size_t count,
                   double atol, double rtol, double *max_abs) {
    int ok = 1;
    *max_abs = 0.0;
    for (size_t i = 0; i < count; i++) {
        if (!isfinite(got[i])) {
            *max_abs = INFINITY;
            return 0;
        }
        const double err = fabs((double)got[i] - ref[i]);
        if (err > *max_abs)
            *max_abs = err;
        if (err > atol + rtol * fabs(ref[i]))
            ok = 0;
    }
    return ok;
}

/* ---------- CSV ---------- */

static int csv_open(csv_writer *w, const char *path, const char *member) {
    w->fp = NULL;
    w->member = member ? member : "";
    if (!path)
        return 1;
    w->fp = fopen(path, "a");
    if (!w->fp) {
        perror(path);
        return 0;
    }
    fseek(w->fp, 0, SEEK_END);
    if (ftell(w->fp) == 0)
        fprintf(w->fp, "member,operation,implementation,N,M,K,filter,"
                       "timing_scope,run,time_ms,correct,max_abs_error\n");
    return 1;
}

static void csv_int(FILE *fp, int v) {
    if (v > 0)
        fprintf(fp, "%d", v);
    fputc(',', fp);
}

static void csv_row(csv_writer *w, const job *j, const char *scope, int run,
                    double ms, int correct, double max_abs) {
    if (!w->fp)
        return;
    fprintf(w->fp, "%s,%s,%s,", w->member, op_name(j->op), j->impl->name);
    csv_int(w->fp, j->N);
    csv_int(w->fp, j->M);
    csv_int(w->fp, j->K);
    fprintf(w->fp, "%s,%s,%d,%.6f,%s,%.6g\n", j->filter ? j->filter : "", scope,
            run, ms, correct ? "true" : "false", max_abs);
}

/* ---------- running one configuration ---------- */

static int call_job(const job *j, lab6_timings *t) {
    if (j->op == OP_MATRIX)
        return j->impl->matmul(j->A, j->B, j->out, j->N, t);
    return j->impl->convolve(j->image, j->weights, j->out, j->M, j->K, t);
}

static void describe(const job *j, char *buf, size_t size) {
    if (j->op == OP_MATRIX)
        snprintf(buf, size, "N=%d", j->N);
    else
        snprintf(buf, size, "M=%d K=%d filter=%s", j->M, j->K,
                 j->filter ? j->filter : "custom");
}

static int measure(const job *j, const options *o, csv_writer *csv) {
    char label[96];
    describe(j, label, sizeof label);
    lab6_timings t;

    for (int w = 0; w < o->warmup; w++) {
        fill_nan(j->out, j->count);
        int status = call_job(j, &t);
        if (status != LAB6_OK) {
            printf("FAIL %-9s %s %s: %s\n", j->impl->name, op_name(j->op),
                   label, status_name(status));
            return 0;
        }
    }

    double sum_compute = 0.0, sum_e2e = 0.0, worst = 0.0;
    int all_ok = 1;
    for (int run = 1; run <= o->runs; run++) {
        fill_nan(j->out, j->count);
        int status = call_job(j, &t);
        if (status != LAB6_OK) {
            printf("FAIL %-9s %s %s run %d: %s\n", j->impl->name,
                   op_name(j->op), label, run, status_name(status));
            return 0;
        }
        double max_abs = 0.0;
        int ok = compare(j->out, j->ref, j->count, o->atol, o->rtol, &max_abs);
        csv_row(csv, j, "compute", run, t.compute_ms, ok, max_abs);
        csv_row(csv, j, "end_to_end", run, t.end_to_end_ms, ok, max_abs);
        sum_compute += t.compute_ms;
        sum_e2e += t.end_to_end_ms;
        if (max_abs > worst)
            worst = max_abs;
        all_ok = all_ok && ok;
    }
    if (csv->fp)
        fflush(csv->fp);
    printf("%s %-9s %-28s compute %10.3f ms  end_to_end %10.3f ms  "
           "max_abs_err %.3g\n",
           all_ok ? "ok  " : "FAIL", j->impl->name, label,
           sum_compute / o->runs, sum_e2e / o->runs, worst);
    return all_ok;
}

/* ---------- commands ---------- */

static int select_impls(const bench_impl *impls, int impl_count,
                        const char *list, op_kind op, int any_op,
                        const bench_impl **out, int *out_count) {
    *out_count = 0;
    if (!list || strcmp(list, "all") == 0) {
        for (int i = 0; i < impl_count && *out_count < MAX_SELECTED; i++)
            if (any_op || supports(&impls[i], op))
                out[(*out_count)++] = &impls[i];
        return *out_count > 0;
    }
    const char *p = list;
    while (*p) {
        size_t len = strcspn(p, ",");
        const bench_impl *found = NULL;
        for (int i = 0; i < impl_count; i++)
            if (strlen(impls[i].name) == len &&
                strncmp(impls[i].name, p, len) == 0)
                found = &impls[i];
        if (!found || (!any_op && !supports(found, op)) ||
            *out_count == MAX_SELECTED) {
            fprintf(stderr,
                    "unknown or unsupported implementation '%.*s'; "
                    "available:",
                    (int)len, p);
            for (int i = 0; i < impl_count; i++)
                if (any_op || supports(&impls[i], op))
                    fprintf(stderr, " %s", impls[i].name);
            fputc('\n', stderr);
            return 0;
        }
        out[(*out_count)++] = found;
        p += len;
        if (*p == ',')
            p++;
    }
    return *out_count > 0;
}

static int run_matrix(const options *o, const bench_impl **sel, int nsel,
                      csv_writer *csv) {
    int all_ok = 1;
    for (int s = 0; s < o->n.count; s++) {
        const int N = o->n.values[s];
        const size_t count = (size_t)N * (size_t)N;
        float *A = xmalloc(count * sizeof(float));
        float *B = xmalloc(count * sizeof(float));
        float *C = xmalloc(count * sizeof(float));
        double *ref = xmalloc(count * sizeof(double));
        if (!A || !B || !C || !ref) {
            free(A), free(B), free(C), free(ref);
            return 0;
        }
        uint64_t state = o->seed;
        fill_unit(A, count, &state);
        fill_unit(B, count, &state);
        ref_matmul(A, B, ref, N);

        for (int i = 0; i < nsel; i++) {
            job j = {OP_MATRIX, sel[i], N,    0, 0,     NULL, A,
                     B,         NULL,   NULL, C, count, ref};
            all_ok &= measure(&j, o, csv);
        }
        free(A), free(B), free(C), free(ref);
    }
    return all_ok;
}

static uint32_t *load_image(const char *path, int *M_out) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        perror(path);
        return NULL;
    }
    fseek(fp, 0, SEEK_END);
    long bytes = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    size_t count = bytes > 0 ? (size_t)bytes / sizeof(uint32_t) : 0;
    int M = 0;
    while ((size_t)(M + 1) * (size_t)(M + 1) <= count)
        M++;
    if (bytes <= 0 || (size_t)bytes % sizeof(uint32_t) != 0 ||
        (size_t)M * (size_t)M != count) {
        fprintf(stderr, "%s: expected a square uint32 image\n", path);
        fclose(fp);
        return NULL;
    }
    uint32_t *image = xmalloc(count * sizeof(uint32_t));
    if (image && fread(image, sizeof(uint32_t), count, fp) != count) {
        fprintf(stderr, "%s: short read\n", path);
        free(image);
        image = NULL;
    }
    fclose(fp);
    for (size_t i = 0; image && i < count; i++) {
        if (image[i] > 255u) {
            fprintf(stderr, "%s: pixel %zu out of range [0,255]\n", path, i);
            free(image);
            image = NULL;
        }
    }
    *M_out = M;
    return image;
}

static int save_output(const char *path, const float *out, size_t count) {
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        perror(path);
        return 0;
    }
    int ok = fwrite(out, sizeof(float), count, fp) == count;
    ok = (fclose(fp) == 0) && ok;
    if (!ok)
        fprintf(stderr, "%s: write failed\n", path);
    return ok;
}

static int run_conv(const options *o, const bench_impl **sel, int nsel,
                    csv_writer *csv) {
    int_list sizes = o->m;
    uint32_t *loaded = NULL;
    if (o->image_path) {
        int M = 0;
        loaded = load_image(o->image_path, &M);
        if (!loaded)
            return 0;
        sizes.count = 1;
        sizes.values[0] = M;
    }
    if (o->save_path && (nsel != 1 || sizes.count != 1 || o->k.count != 1)) {
        fprintf(stderr, "--save needs exactly one implementation, image size "
                        "and filter size\n");
        free(loaded);
        return 0;
    }

    int all_ok = 1;
    for (int s = 0; s < sizes.count; s++) {
        const int M = sizes.values[s];
        const size_t count = (size_t)M * (size_t)M;
        uint32_t *image = loaded ? loaded : xmalloc(count * sizeof(uint32_t));
        float *out = xmalloc(count * sizeof(float));
        double *ref = xmalloc(count * sizeof(double));
        if (!image || !out || !ref) {
            if (image != loaded)
                free(image);
            free(loaded), free(out), free(ref);
            return 0;
        }
        if (!loaded) {
            uint64_t state = o->seed;
            fill_pixels(image, count, &state);
        }
        for (int f = 0; f < o->k.count; f++) {
            const int K = o->k.values[f];
            float weights[49 * 49];
            if (K % 2 == 0 || K > 49 || !make_filter(o->filter, K, weights)) {
                fprintf(stderr, "invalid filter %s with K=%d (odd K <= 49)\n",
                        o->filter, K);
                all_ok = 0;
                continue;
            }
            ref_convolve(image, weights, ref, M, K);
            for (int i = 0; i < nsel; i++) {
                job j = {OP_CONVOLUTION,
                         sel[i],
                         0,
                         M,
                         K,
                         o->filter,
                         NULL,
                         NULL,
                         image,
                         weights,
                         out,
                         count,
                         ref};
                int ok = measure(&j, o, csv);
                all_ok &= ok;
                if (ok && o->save_path)
                    all_ok &= save_output(o->save_path, out, count);
            }
        }
        if (image != loaded)
            free(image);
        free(out), free(ref);
    }
    free(loaded);
    return all_ok;
}

/* Returns 1 when got matches ref, printing one PASS/FAIL line. */
static int check_case(const char *impl, const char *what, int status,
                      const float *got, const double *ref, size_t count,
                      const options *o) {
    if (status != LAB6_OK) {
        printf("FAIL %-9s %-36s %s\n", impl, what, status_name(status));
        return 0;
    }
    double max_abs = 0.0;
    int ok = compare(got, ref, count, o->atol, o->rtol, &max_abs);
    printf("%s %-9s %-36s max_abs_err %.3g\n", ok ? "PASS" : "FAIL", impl, what,
           max_abs);
    return ok;
}

static int timings_valid(const lab6_timings *t) {
    return isfinite(t->compute_ms) && isfinite(t->end_to_end_ms) &&
           t->compute_ms >= 0.0 && t->end_to_end_ms >= t->compute_ms;
}

static int selftest_matmul(const bench_impl *impl, const options *o) {
    /* Non-symmetric so transposed or swapped operands are caught. */
    const float A[] = {1, 2, 3, 4};
    const float B[] = {5, 6, 7, 8};
    const double expected[] = {19, 22, 43, 50};
    float C[4];
    fill_nan(C, 4);
    int status = impl->matmul(A, B, C, 2, NULL);
    int ok = check_case(impl->name, "matmul hand 2x2 (timings=NULL)", status, C,
                        expected, 4, o);
    if (status == LAB6_NOT_IMPLEMENTED)
        return 0;

    for (size_t s = 0; s < sizeof kSelftestSizes / sizeof(int); s++) {
        const int N = kSelftestSizes[s];
        const size_t count = (size_t)N * N;
        float *a = xmalloc(count * sizeof(float));
        float *b = xmalloc(count * sizeof(float));
        float *c = xmalloc(count * sizeof(float));
        double *ref = xmalloc(count * sizeof(double));
        if (!a || !b || !c || !ref) {
            free(a), free(b), free(c), free(ref);
            return 0;
        }
        uint64_t state = o->seed + (uint64_t)N;
        fill_unit(a, count, &state);
        fill_unit(b, count, &state);
        fill_nan(c, count);
        ref_matmul(a, b, ref, N);
        lab6_timings t;
        status = impl->matmul(a, b, c, N, &t);
        char what[64];
        snprintf(what, sizeof what, "matmul random N=%d", N);
        int case_ok = check_case(impl->name, what, status, c, ref, count, o);
        if (case_ok && !timings_valid(&t)) {
            printf("FAIL %-9s %-36s bad timings %.3f / %.3f ms\n", impl->name,
                   what, t.compute_ms, t.end_to_end_ms);
            case_ok = 0;
        }
        ok &= case_ok;
        free(a), free(b), free(c), free(ref);
    }

    float dummy = 0.0f;
    status = impl->matmul(&dummy, &dummy, &dummy, 0, NULL);
    if (status != LAB6_INVALID_ARGUMENT) {
        printf("FAIL %-9s %-36s returned %s\n", impl->name,
               "matmul rejects N=0", status_name(status));
        ok = 0;
    }
    return ok;
}

static int selftest_convolve(const bench_impl *impl, const options *o) {
    /* Only the top-left tap is set: true convolution reads image[y+1][x+1],
     * cross-correlation would read image[y-1][x-1]. */
    const uint32_t image[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    const float filter[] = {1, 0, 0, 0, 0, 0, 0, 0, 0};
    const double expected[] = {5, 6, 0, 8, 9, 0, 0, 0, 0};
    float out[9];
    fill_nan(out, 9);
    int status = impl->convolve(image, filter, out, 3, 3, NULL);
    int ok = check_case(impl->name, "conv hand asymmetric (timings=NULL)",
                        status, out, expected, 9, o);
    if (status == LAB6_NOT_IMPLEMENTED)
        return 0;

    for (size_t s = 0; s < sizeof kSelftestSizes / sizeof(int); s++) {
        for (size_t f = 0; f < sizeof kSelftestFilterSizes / sizeof(int); f++) {
            const int M = kSelftestSizes[s];
            const int K = kSelftestFilterSizes[f];
            const size_t count = (size_t)M * M;
            uint32_t *img = xmalloc(count * sizeof(uint32_t));
            float *res = xmalloc(count * sizeof(float));
            double *ref = xmalloc(count * sizeof(double));
            float weights[25];
            if (!img || !res || !ref) {
                free(img), free(res), free(ref);
                return 0;
            }
            uint64_t state = o->seed + (uint64_t)(M * 64 + K);
            fill_pixels(img, count, &state);
            for (int i = 0; i < K * K; i++)
                weights[i] = 2.0f * rng_unit(&state) - 1.0f;
            fill_nan(res, count);
            ref_convolve(img, weights, ref, M, K);
            lab6_timings t;
            status = impl->convolve(img, weights, res, M, K, &t);
            char what[64];
            snprintf(what, sizeof what, "conv random M=%d K=%d", M, K);
            int case_ok =
                check_case(impl->name, what, status, res, ref, count, o);
            if (case_ok && !timings_valid(&t)) {
                printf("FAIL %-9s %-36s bad timings %.3f / %.3f ms\n",
                       impl->name, what, t.compute_ms, t.end_to_end_ms);
                case_ok = 0;
            }
            ok &= case_ok;
            free(img), free(res), free(ref);
        }
    }

    float dummy = 0.0f;
    uint32_t pixel = 0;
    status = impl->convolve(&pixel, &dummy, &dummy, 1, 2, NULL);
    if (status != LAB6_INVALID_ARGUMENT) {
        printf("FAIL %-9s %-36s returned %s\n", impl->name,
               "conv rejects even K", status_name(status));
        ok = 0;
    }
    return ok;
}

static int run_selftest(const options *o, const bench_impl **sel, int nsel) {
    const float A[] = {1, 2, 3, 4};
    const float B[] = {5, 6, 7, 8};
    const double mat_expected[] = {19, 22, 43, 50};
    const uint32_t image[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    const float filter[] = {1, 0, 0, 0, 0, 0, 0, 0, 0};
    const double conv_expected[] = {5, 6, 0, 8, 9, 0, 0, 0, 0};
    double r[9];
    float rf[9];
    int ok = 1;

    ref_matmul(A, B, r, 2);
    for (int i = 0; i < 4; i++)
        rf[i] = (float)r[i];
    ok &= check_case("reference", "matmul hand 2x2", LAB6_OK, rf, mat_expected,
                     4, o);
    ref_convolve(image, filter, r, 3, 3);
    for (int i = 0; i < 9; i++)
        rf[i] = (float)r[i];
    ok &= check_case("reference", "conv hand asymmetric", LAB6_OK, rf,
                     conv_expected, 9, o);

    for (int i = 0; i < nsel; i++) {
        if (sel[i]->matmul)
            ok &= selftest_matmul(sel[i], o);
        if (sel[i]->convolve)
            ok &= selftest_convolve(sel[i], o);
    }
    printf("selftest: %s\n", ok ? "all passed" : "FAILED");
    return ok;
}

/* ---------- entry point ---------- */

static void usage(const char *program, const bench_impl *impls, int count) {
    fprintf(stderr,
            "usage: %s <selftest|matrix|conv> [options]\n"
            "  --impl LIST     comma-separated names or 'all' (default all)\n"
            "  --n LIST        matrix sizes (default 256,512,1024,2048)\n"
            "  --m LIST        image sizes (default 512,1024,2048)\n"
            "  --k LIST        filter sizes (default 3,5,7)\n"
            "  --filter NAME   mean|edge|sharpen (default mean)\n"
            "  --image PATH    raw uint32 square image; default random\n"
            "  --save PATH     write float32 output (one impl/M/K only)\n"
            "  --csv PATH      append measurements (header if file empty)\n"
            "  --member NAME   member column (default $LAB6_MEMBER)\n"
            "  --warmup N --runs N --seed N --atol X --rtol X\n"
            "implementations:",
            program);
    for (int i = 0; i < count; i++)
        fprintf(stderr, " %s", impls[i].name);
    fputc('\n', stderr);
}

static int parse_args(int argc, char **argv, options *o) {
    memset(o, 0, sizeof *o);
    set_list(&o->n, kMatrixSizes, (int)(sizeof kMatrixSizes / sizeof(int)));
    set_list(&o->m, kImageSizes, (int)(sizeof kImageSizes / sizeof(int)));
    set_list(&o->k, kFilterSizes, (int)(sizeof kFilterSizes / sizeof(int)));
    o->filter = "mean";
    o->member = getenv("LAB6_MEMBER");
    o->warmup = 1;
    o->runs = 3;
    o->seed = 542;
    o->atol = 1e-3;
    o->rtol = 1e-4;

    if (argc < 2 || argv[1][0] == '-')
        return 0;
    o->command = argv[1];
    for (int i = 2; i < argc; i += 2) {
        const char *flag = argv[i];
        const char *value = i + 1 < argc ? argv[i + 1] : NULL;
        if (!value) {
            fprintf(stderr, "missing value for %s\n", flag);
            return 0;
        }
        int ok = 1;
        if (strcmp(flag, "--impl") == 0)
            o->impls = value;
        else if (strcmp(flag, "--n") == 0)
            ok = parse_list(value, &o->n);
        else if (strcmp(flag, "--m") == 0)
            ok = parse_list(value, &o->m);
        else if (strcmp(flag, "--k") == 0)
            ok = parse_list(value, &o->k);
        else if (strcmp(flag, "--filter") == 0)
            o->filter = value;
        else if (strcmp(flag, "--image") == 0)
            o->image_path = value;
        else if (strcmp(flag, "--save") == 0)
            o->save_path = value;
        else if (strcmp(flag, "--csv") == 0)
            o->csv_path = value;
        else if (strcmp(flag, "--member") == 0)
            o->member = value;
        else if (strcmp(flag, "--warmup") == 0)
            ok = (o->warmup = atoi(value)) >= 0;
        else if (strcmp(flag, "--runs") == 0)
            ok = (o->runs = atoi(value)) > 0;
        else if (strcmp(flag, "--seed") == 0)
            o->seed = strtoull(value, NULL, 10);
        else if (strcmp(flag, "--atol") == 0)
            ok = (o->atol = atof(value)) >= 0.0;
        else if (strcmp(flag, "--rtol") == 0)
            ok = (o->rtol = atof(value)) >= 0.0;
        else {
            fprintf(stderr, "unknown option %s\n", flag);
            return 0;
        }
        if (!ok) {
            fprintf(stderr, "invalid value for %s: %s\n", flag, value);
            return 0;
        }
    }
    return 1;
}

int bench_main(int argc, char **argv, const char *program,
               const bench_impl *impls, int impl_count) {
    options o;
    if (!parse_args(argc, argv, &o)) {
        usage(program, impls, impl_count);
        return 2;
    }

    int is_selftest = strcmp(o.command, "selftest") == 0;
    op_kind op = OP_MATRIX;
    if (strcmp(o.command, "conv") == 0)
        op = OP_CONVOLUTION;
    else if (!is_selftest && strcmp(o.command, "matrix") != 0) {
        usage(program, impls, impl_count);
        return 2;
    }

    const bench_impl *sel[MAX_SELECTED];
    int nsel = 0;
    if (!select_impls(impls, impl_count, o.impls, op, is_selftest, sel, &nsel))
        return 2;

    if (is_selftest)
        return run_selftest(&o, sel, nsel) ? 0 : 1;

    csv_writer csv;
    if (!csv_open(&csv, o.csv_path, o.member))
        return 1;
    int ok = op == OP_MATRIX ? run_matrix(&o, sel, nsel, &csv)
                             : run_conv(&o, sel, nsel, &csv);
    if (csv.fp)
        fclose(csv.fp);
    return ok ? 0 : 1;
}
