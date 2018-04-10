
/* TODO: Need to write back into pann every time? Or find some other workaround for PositionSet. */
/* Make sure USE_L1 in patch.h is set to 1. */
/* OpenMP shared data structures (heap) may be slow when multithreaded due to things like new/delete that may not be properly parallelized (hash_set<int> was slow due to some global problem like this). */

#include "knn.h"
#include <math.h>
#include "simpatch.h"
#include <algorithm>
#ifndef UNIX_MODE
#include <hash_set>
#endif

#define USE_GRAPHCUTS       1

#ifdef UNIX_MODE
#undef USE_GRAPHCUTS
#define USE_GRAPHCUTS 0
#endif

/* kNN has actually been implemented when searching over all rotations+scales, but it is not currently exposed by the MATLAB wrapper. */
#define TRANSLATE_ONLY      1

#define KNN_MEDOID          0
#define USE_ERF             0
#define OVERLAPPING_PATCHES 0
#define VOTE_SUM            1
#define PROP_DIST2          0
#define ENRICH_DURING       0
#define P_BEST_ONLY         0
#define RS_BEST_ONLY        0
#define P_RAND_ONLY         0
#define RS_RAND_ONLY        0
#define RS_RAND1TOK         0
#define P_RAND1TOK          0
// RS_RAND1TOK has a bug -- should take the minimum k elements, it currently just takes k first from the max-heap

/* Hardcodes hash table size, slightly faster for small k-NN. */
#define SMALLK              1

#ifndef UNIX_MODE
using namespace stdext;
#else
#include <set>
#define hash_set set
#endif

#define HISTO_ANGLES 32
#define M_PI 3.1415926535897932384626433832795

#define accurate_timer() 0.0

BITMAP *greyscale16(BITMAP *a) {
  if (bitmap_color_depth(a) != 32) { fprintf(stderr, "Input color depth for greyscale not 32\n"); exit(1); }
  if (sizeof(unsigned short) != 2) { fprintf(stderr, "Size of unsigned short != 2\n"); exit(1); }
  BITMAP *ans = create_bitmap_ex(16, a->w, a->h);
  for (int y = 0; y < a->h; y++) {
    int *in_row = (int *) a->line[y];
    unsigned short *out_row = (unsigned short *) ans->line[y];
    for (int x = 0; x < a->w; x++) {
      int c = in_row[x];
      int r = (c&255), g = ((c>>8)&255), b = (c>>16);
      out_row[x] = (((2*30)*r+(2*59)*g+(2*11)*b+1)/200)*257;
    }
  }
  return ans;
}

BITMAP *greyscale(BITMAP *a) {
  if (bitmap_color_depth(a) != 32) { fprintf(stderr, "Input color depth for greyscale not 32\n"); exit(1); }
  BITMAP *ans = create_bitmap_ex(8, a->w, a->h);
  for (int y = 0; y < a->h; y++) {
    int *in_row = (int *) a->line[y];
    unsigned char *out_row = (unsigned char *) ans->line[y];
    for (int x = 0; x < a->w; x++) {
      int c = in_row[x];
      int r = (c&255), g = ((c>>8)&255), b = (c>>16);
      out_row[x] = (((2*30)*r+(2*59)*g+(2*11)*b+1)/200);
    }
  }
  return ans;
}

BITMAP *greyscale16_to_color(BITMAP *a) {
  if (bitmap_color_depth(a) != 16) { fprintf(stderr, "Bitmap color depth for greyscale_to_color not 16\n"); exit(1); }
  if (sizeof(unsigned short) != 2) { fprintf(stderr, "Size of unsigned short != 2\n"); exit(1); }
  BITMAP *ans = create_bitmap_ex(32, a->w, a->h);
  for (int y = 0; y < a->h; y++) {
    int *out_row = (int *) ans->line[y];
    unsigned short *in_row = (unsigned short *) a->line[y];
    for (int x = 0; x < a->w; x++) {
      int c = in_row[x]>>8;
      out_row[x] = c|(c<<8)|(c<<16);
    }
  }
  return ans;
}


BITMAP *greyscale_to_color(BITMAP *a) {
  if (bitmap_color_depth(a) != 8) { fprintf(stderr, "Bitmap color depth for greyscale_to_color not 8\n"); exit(1); }
  BITMAP *ans = create_bitmap_ex(32, a->w, a->h);
  for (int y = 0; y < a->h; y++) {
    int *out_row = (int *) ans->line[y];
    unsigned char *in_row = (unsigned char *) a->line[y];
    for (int x = 0; x < a->w; x++) {
      int c = in_row[x];
      out_row[x] = c|(c<<8)|(c<<16);
    }
  }
  return ans;
}

#define FILTER_SHIFT 16

BITMAP *horiz_gaussian_blur(BITMAP *a, double sigma) {
  fprintf(stderr, "horiz_gaussian_blur unimplemented\n"); exit(1);
#if 0
  if (bitmap_color_depth(a) != 16) { fprintf(stderr, "Bitmap color depth for gaussian_blur not 16\n"); exit(1); }
  if (sizeof(unsigned short) != 2) { fprintf(stderr, "Size of unsigned short != 2\n"); exit(1); }
  if (sigma < 0) { sigma = 0; }
  int n = int(6*sigma+1);
  if ((n&1)==0) { n++; }
  int n2 = n>>1;
  double *ff = new double[n];
  unsigned *f = new unsigned[n];
  double scale = 0.0;
  for (int i = 0; i < n; i++) {
    int di = i-(n/2);
    double vi = exp(-di*di/(2*sigma*sigma));
    ff[i] = vi;
    scale += vi;
  }
  scale = (1<<FILTER_SHIFT)/scale;
  //unsigned fi_sum = 0;
  for (int i = 0; i < n; i++) {
    f[i] = int(ff[i]*scale);
    //fi_sum += f[i];
  }
  //printf("fi_sum = %d (%d)\n", fi_sum, 1<<FILTER_SHIFT);
  unsigned *f2 = &f[n2];
  //printf("%d %d %f\n", n, n2, sigma);
  delete[] ff;
  BITMAP *ans = create_bitmap_ex(16, a->w, a->h);
//  int maxv = 0;
  int n2_min = MIN(n2, ans->w);
  int xlast = ans->w-n2-1;
  if (xlast < 0) { xlast = 0; }
  if (xlast < n2_min) { xlast = n2_min; }
  if (xlast > ans->w) { fprintf(stderr, "xlast too large (%d %d)\n", xlast, ans->w); exit(1); }
  #pragma omp parallel for schedule(static,1)
  for (int y = 0; y < ans->h; y++) {
    unsigned short *arow = (unsigned short *) a->line[y];
    unsigned short *ans_row = (unsigned short *) ans->line[y];
    /*for (int x = 0; x < ans->w; x++) {
      ans_row[x] = arow[x];
    }*/
    /*
    for (int x = 0; x < ans->w; x++) {
      int v = 0;
      for (int dx = -n2; dx <= n2; dx++) {
        int fi = f[dx+n2];
        int xp = x+dx;
        if (xp < 0) { xp = 0; }
        else if (xp >= ans->w) { xp = ans->w-1; }
        v += arow[xp]*fi;
      }
      ans_row[x] = (v+(1<<(FILTER_SHIFT-1)))>>FILTER_SHIFT;
      //if (ans_row[x] > maxv) { maxv = ans_row[x]; }
    }*/
    
    for (int x = 0; x < n2_min; x++) {
      //if (x < 0 || x >= ans->w) { fprintf(stderr, "out of bounds a: %d (%d)\n", x, ans->w); exit(1); }
      unsigned v = 0;
      for (int dx = -n2; dx <= n2; dx++) {
        unsigned fi = f2[dx]; //f[dx+n2];
        int xp = x+dx;
        if (xp < 0) { xp = 0; }
        else if (xp >= ans->w) { xp = ans->w-1; }
        v += arow[xp]*fi;
      }
      ans_row[x] = (v+(1<<(FILTER_SHIFT-1)))>>FILTER_SHIFT;
    }
    for (int x = n2_min; x < xlast; x++) {
      //if (x < 0 || x >= ans->w) { fprintf(stderr, "out of bounds b: %d (%d)\n", x, ans->w); exit(1); }
      unsigned v = 0;
      for (int dx = -n2; dx <= n2; dx++) {
        //if (xp < 0) { fprintf(stderr, "bad < b\n"); exit(1); xp = 0; }
        //else if (xp >= ans->w) { fprintf(stderr, "bad > b\n"); exit(1); xp = ans->w-1; }
        v += arow[x+dx]*f2[dx];
        //v += arow[x+dx+1]*f2[dx+1];
      }
      //v += arow[x+n2]*f2[n2];
      ans_row[x] = (v+(1<<(FILTER_SHIFT-1)))>>FILTER_SHIFT;
    }
    for (int x = xlast; x < ans->w; x++) {
      //if (x < 0 || x >= ans->w) { fprintf(stderr, "out of bounds c: %d (%d)\n", x, ans->w); exit(1); }
      unsigned v = 0;
      for (int dx = -n2; dx <= n2; dx++) {
        unsigned fi = f2[dx]; //f[dx+n2];
        int xp = x+dx;
        if (xp < 0) { xp = 0; }
        else if (xp >= ans->w) { xp = ans->w-1; }
        v += arow[xp]*fi;
      }
      ans_row[x] = (v+(1<<(FILTER_SHIFT-1)))>>FILTER_SHIFT;
      //if (ans_row[x] > maxv) { maxv = ans_row[x]; }
    }
  }
  //printf("%d\n", maxv);
  delete[] f;
  return ans;
#endif
  return NULL;
}

BITMAP *transpose16(BITMAP *bmp) {
  if (bitmap_color_depth(bmp) != 16) { fprintf(stderr, "Bitmap color depth for transpose16 not 16\n"); exit(1); }
  if (sizeof(unsigned short) != 2) { fprintf(stderr, "Size of unsigned short != 2\n"); exit(1); }
  BITMAP *ans = create_bitmap_ex(16, bmp->h, bmp->w);
  for (int y = 0; y < bmp->h; y++) {
    unsigned short *row = (unsigned short *) bmp->line[y];
    for (int x = 0; x < bmp->w; x++) {
      //int c = _getpixel32(bmp, x, y);
      int c = row[x];
      ((unsigned short *) ans->line[x])[y] = c;
      //_putpixel32(ans, y, x, c);
    }  
  }
  return ans;
}

BITMAP *gaussian_blur16(BITMAP *a, double sigma) {
  BITMAP *temp = horiz_gaussian_blur(a, sigma);
  BITMAP *tempT = transpose16(temp);
  destroy_bitmap(temp);
  BITMAP *ans = horiz_gaussian_blur(tempT, sigma);
  destroy_bitmap(tempT);
  BITMAP *ansT = transpose16(ans);
  destroy_bitmap(ans);
  return ansT;
}

BITMAP *color_gaussian_blur(BITMAP *a, double sigma, int aconstraint_alpha) {
  if (bitmap_color_depth(a) != 32) { fprintf(stderr, "color_gaussian_blur: input is not 32-bit\n"); exit(1); }
  if (sizeof(unsigned short) != 2) { fprintf(stderr, "sizeof(unsigned short) != 2\n"); exit(1); }
  BITMAP *r = create_bitmap_ex(16, a->w, a->h);
  BITMAP *g = create_bitmap_ex(16, a->w, a->h);
  BITMAP *b = create_bitmap_ex(16, a->w, a->h);
  BITMAP *alpha = aconstraint_alpha ? create_bitmap_ex(16, a->w, a->h): NULL;
  for (int y = 0; y < a->h; y++) {
    for (int x = 0; x < a->w; x++) {
      int c = ((int *) a->line[y])[x];
      ((unsigned short *) r->line[y])[x] = (c&255)<<8;
      ((unsigned short *) g->line[y])[x] = ((c>>8)&255)<<8;
      ((unsigned short *) b->line[y])[x] = ((c>>16)&255)<<8;
      if (aconstraint_alpha) {
        ((unsigned short *) alpha->line[y])[x] = (c>>24)<<8;
      }
    }
  }
  BITMAP *rp = gaussian_blur16(r, sigma);
  BITMAP *gp = gaussian_blur16(g, sigma);
  BITMAP *bp = gaussian_blur16(b, sigma);
  BITMAP *alphap = aconstraint_alpha ? gaussian_blur16(alpha, sigma): NULL;
  BITMAP *ans = create_bitmap(a->w, a->h);
  for (int y = 0; y < a->h; y++) {
    for (int x = 0; x < a->w; x++) {
      ((int *) ans->line[y])[x] = 
      (((unsigned short *) rp->line[y])[x]>>8) |
      ((((unsigned short *) gp->line[y])[x]>>8)<<8) |
      ((((unsigned short *) bp->line[y])[x]>>8)<<16) |
      (aconstraint_alpha ? ((((unsigned short *) alphap->line[y])[x]>>8)<<24): 0);
    }
  }
  destroy_bitmap(r);
  destroy_bitmap(g);
  destroy_bitmap(b);
  destroy_bitmap(alpha);
  destroy_bitmap(rp);
  destroy_bitmap(gp);
  destroy_bitmap(bp);
  destroy_bitmap(alphap);
  return ans;
}

BITMAP *gaussian_deriv_angle(BITMAP *a, double sigma, BITMAP **gx_out, BITMAP **gy_out) {
  fprintf(stderr, "gaussian_deriv_angle unimplemented\n"); exit(1);
#if 0
  BITMAP *b = gaussian_blur16(a, sigma);
  if (bitmap_color_depth(b) != 16) { fprintf(stderr, "result of gaussian_blur16 is not 16 bit\n"); exit(1); }
  BITMAP *ans = create_bitmap(a->w, a->h);
  if (a->w < 3 || a->h < 3) { fprintf(stderr, "a too small (%dx%d < 3x3)\n", a->w, a->h); exit(1); }
  if (gx_out) {
    *gx_out = create_bitmap(a->w, a->h);
  }
  if (gy_out) {
    *gy_out = create_bitmap(a->w, a->h);
  }
  for (int y = 0; y < ans->h; y++) {
    int *ans_row = (int *) ans->line[y];
    int yc = y;
    if (yc < 1) { yc = 1; }
    else if (yc >= ans->h-1) { yc = ans->h-2; }
    unsigned short *brow1 = (unsigned short *) b->line[yc-1];
    unsigned short *brow2 = (unsigned short *) b->line[yc];
    unsigned short *brow3 = (unsigned short *) b->line[yc+1];
    for (int x = 0; x < ans->w; x++) {
      int xc = x;
      if (xc < 1) { xc = 1; }
      else if (xc >= ans->w-1) { xc = ans->w-2; }
      int dx = brow2[xc+1]-brow2[xc-1];
      int dy = brow3[xc]-brow1[xc];
      if (gx_out) {
        ((unsigned *) ((*gx_out)->line[y]))[x] = (unsigned) dx;
        ((unsigned *) ((*gy_out)->line[y]))[x] = (unsigned) dy;
      }
      dx <<= 8;
      dy <<= 8;
      fixed angle = -fixatan2(dy, dx);
      if (angle < 0) { angle += 256*65536; }
      ans_row[x] = angle>>(24-ANGLE_SHIFT);
    }
  }
  destroy_bitmap(b);
  return ans;
#endif
  return NULL;
}

PRINCIPAL_ANGLE *create_principal_angle(Params *p, BITMAP *bmp) {
  fprintf(stderr, "create_principal_angle unimplemented\n"); exit(1);
#if 0
  init_xform_tables();
  if (bitmap_color_depth(bmp) != 32) { fprintf(stderr, "Bitmap color depth != 32 for get_principal_angle\n"); exit(1); }
  BITMAP *bmp16 = greyscale16(bmp);
  PRINCIPAL_ANGLE *ans = new PRINCIPAL_ANGLE();
  for (int i = 0; i < N_PRINCIPAL_ANGLE; i++) {
    int mid = int((i+0.5)*NUM_SCALES/N_PRINCIPAL_ANGLE);
    double sigma = (xform_scale_table[mid]*(1.0/65536.0))*(p->patch_w/2.0)*0.85;
    fprintf(stderr, "%d, %d, %f, sigma=%f\n", i, mid, (xform_scale_table[mid]*(1.0/65536.0)), sigma);
    ans->angle[i] = gaussian_deriv_angle(bmp16, sigma);
  }
  destroy_bitmap(bmp16);
  return ans;
#endif
  return NULL;
}

void destroy_principal_angle(PRINCIPAL_ANGLE *b) {
  fprintf(stderr, "destroy_principal_angle unimplemented\n"); exit(1);
#if 0
  if (!b) { return; }
  for (int i = 0; i < N_PRINCIPAL_ANGLE; i++) {
    destroy_bitmap(b->angle[i]);
  }
  delete b;
#endif
}

/* Get principal angle given upper left corner (x0, y0) of patch and scale. */
int get_principal_angle(Params *p, PRINCIPAL_ANGLE *b, int x0, int y0, int scale) {
  int h = p->patch_w/2;
  int xc = x0+h, yc = y0+h;
  int i = scale>>(SCALE_SHIFT-N_PRINCIPAL_ANGLE_SHIFT);
  if ((unsigned) i >= (unsigned) N_PRINCIPAL_ANGLE) { fprintf(stderr, "i out of range in get_principal_angle (%d)\n", i); exit(1); }
  BITMAP *bmp = b->angle[i];
  if ((unsigned) xc >= (unsigned) bmp->w || (unsigned) yc >= (unsigned) bmp->h) { fprintf(stderr, "x, y out of bounds in get_principal_angle (%d, %d, %dx%d)\n", xc, yc, bmp->w, bmp->h); exit(1); }
  return ((int *) bmp->line[yc])[xc];
}

double knn_avg_dist(Params *p, VBMP *annd) {
  double ans = 0;
  int n = 0;
  if (annd->n != p->knn) { fprintf(stderr, "annd->n (%d) != p->knn (%d)\n", annd->n, p->knn); exit(1); }
  for (int y = 0; y < annd->h-p->patch_w+1; y++) {
    //int *row = (int *) annd->line[y];
    for (int x = 0; x < annd->w-p->patch_w+1; x++) {
      int *pd = annd->get(x, y);
      for (int i = 0; i < p->knn; i++) {
        ans += USE_L1 ? pd[i]: sqrt(double(pd[i]));
        n++;
      }
    }
  }
  return ans / n;
}

template<class T>
class triple { public:
  T a, b, c;
  triple(const T a_, const T b_, const T c_) :a(a_), b(b_), c(c_) { }
  int operator < (const triple &b) const {
    return a < b.a;
  }
};

template<class T>
class knn_pair { public:
  T a, b;
  knn_pair(const T a_, const T b_) :a(a_), b(b_) { }
  int operator < (const knn_pair &b) const {
    return a < b.a;
  }
  int operator == (const knn_pair &b) const {
    return a == b.a;
  }
};

#if TRANSLATE_ONLY
#define qtype knn_pair
#else
#define qtype triple
#endif

//#define NHASH 256
#if SMALLK
#define NHASH 256
#define NHASH_M1 (NHASH-1)
#else
#define NHASH nhash
#define NHASH_M1 nhash_m1
#endif
#define HASH_FUNC(xv, yv) (((xv)+(yv)*3829)&(NHASH_M1))
//#define KNOWN_EMPTY -1
//#define CHECK_COORDS if (xv < 0 || yv < 0) { fprintf(stderr, "xv < 0 or yv < 0\n"); exit(1); }
#define CHECK_COORDS

class PositionLink { public:
  int v;
  PositionLink *next;
};

int roundup_pow2(int x) {
  int ans = 1;
  for (;;) {
    if (ans >= x) { return ans; }
    ans = ans*2;
  }
}

/* Hash table of fixed size for positions, much faster (10x or more) than STL if few/no collisions. Max size is NHASH. */
class PositionSet { public:
  //int hash[NHASH];
#if SMALLK
  PositionLink *item[NHASH];
  PositionLink item_data[NHASH];
#else
  PositionLink **item;
  PositionLink *item_data;
  int nhash;
  int nhash_m1;
#endif
  PositionLink *free_item;
  //int known[NHASH]; // an item known to be at the given bucket, or KNOWN_EMPTY if no item known
  int *pnn;
  vector<qtype<int> > *q;
  PositionSet() {
  }
  /*
  void check_equals(PositionSet *other) {
    for (int i = 0; i < NHASH; i++) {
      if (hash[i] != other->hash[i]) { fprintf(stderr, "not equal at %d: %d %d\n", i, hash[i], other->hash[i]); exit(1); }
    }
  }
  */
  void init(int *pnn_, vector<qtype<int> > *q_, int nhash_) {
#if !SMALLK
    nhash = roundup_pow2(nhash_)*4;
    nhash_m1 = nhash-1;
    item_data = new PositionLink[NHASH];
    item = new PositionLink *[NHASH];
#endif
    //for (int i = 0; i < NHASH; i++) { hash[i] = 0; }
    for (int i = 0; i < NHASH; i++) { item[i] = NULL; }
    for (int i = 0; i < NHASH-1; i++) { item_data[i].next = &item_data[i+1]; }
    item_data[NHASH-1].next = NULL;
    free_item = &item_data[0];
    //for (int i = 0; i < NHASH; i++) { known[i] = KNOWN_EMPTY; }
    pnn = pnn_;
    q = q_;
  }
  PositionSet(int *pnn_, vector<qtype<int> > *q_, int nhash_) {
    init(pnn_, q_, nhash_);
/*
    for (int i = 0; i < NHASH; i++) { hash[i] = 0; }
    //for (int i = 0; i < NHASH; i++) { known[i] = KNOWN_EMPTY; }
    pnn = pnn_;
    q = q_;*/
  }
#if !SMALLK
  ~PositionSet() {
    delete[] item_data;
    delete[] item;
  }
#endif
  int contains(int xv, int yv, int i) {
    CHECK_COORDS
    int h = HASH_FUNC(xv, yv); //(xv+yv*3829)&(NHASH-1);
    //int pnni = ;
    PositionLink *current = item[h];
    while (current) {
      if (current->v == XY_TO_INT(xv, yv)) { return 1; }
      current = current->next;
    }
    return 0;
    /*
    if (!hash[h]) {
      return 0;
    } else {
      //if (XY_TO_INT(xv, yv) == known[h]) { return 1; }
      int pnni = XY_TO_INT(xv, yv);
      for (int j = 0; j < i; j++) {
        //if (pnn[j] == pnni) { return 1; }
        if ((*q)[j].b == pnni) { return 1; }
      }
      return 0;
    }
    */
  }
  int contains_noqueue(int xv, int yv, int i) {
    return contains(xv, yv, i);
    /*
    CHECK_COORDS
    int h = HASH_FUNC(xv, yv); //(xv+yv*3829)&(NHASH-1);
    if (!hash[h]) {
      return 0;
    } else {
      //if (XY_TO_INT(xv, yv) == known[h]) { return 1; }
      int pnni = XY_TO_INT(xv, yv);
      for (int j = 0; j < i; j++) {
        if (pnn[j] == pnni) { return 1; }
        //if ((*q)[j].b == pnni) { return 1; }
      }
      return 0;
    }*/
  }

  /* If element not in set, insert it and return 1.  If element in set, do nothing and return 0. */
  int try_insert(int xv, int yv, int i) {
    CHECK_COORDS
    //fprintf(stderr, "try_insert(%d, %d, %d)\n", xv, yv, i);
    int h = HASH_FUNC(xv, yv); //(xv+yv*3829)&(NHASH-1);
    int pnni = XY_TO_INT(xv, yv);
    PositionLink *current = item[h];
    while (current) {
      if (current->v == pnni) { return 0; }
      current = current->next;
    }
    PositionLink *add = free_item;
    free_item = free_item->next;
    if (!free_item) { fprintf(stderr, "hash table full\n"); exit(1); }
    add->v = pnni;
    add->next = item[h];
    item[h] = add;
    return 1;
    /*
    if (!hash[h]) {
      hash[h] = 1;
      //known[h] = XY_TO_INT(xv, yv);
      return 1;
    } else {
      int pnni = XY_TO_INT(xv, yv);
      for (int j = 0; j < i; j++) {
        if (pnn[j] == pnni) { return 0; }
      }
      hash[h]++;
      return 1;
    }
    */
  }
  /* Insert an element that's not in the set. */
  void insert_nonexistent(int xv, int yv, int verbose=1) {
    CHECK_COORDS
    int h = HASH_FUNC(xv, yv); //(xv+yv*3829)&(NHASH-1);
    int pnni = XY_TO_INT(xv, yv);
    PositionLink *add = free_item;
    free_item = free_item->next;
    if (!free_item) { fprintf(stderr, "hash table full\n"); exit(1); }
    add->v = pnni;
    add->next = item[h];
    item[h] = add;
    //if (verbose) { fprintf(stderr, "insert_nonexistent(%d, %d), h=%d\n", xv, yv, h); }
    //hash[h]++;
    //known[h] = XY_TO_INT(xv, yv);
  }
  /* Remove an element that's in the set. */
  void remove(int xv, int yv) {
    CHECK_COORDS
    int h = HASH_FUNC(xv, yv);
    //fprintf(stderr, "remove(%d, %d), h=%d\n", xv, yv, h);
    //if (hash[h] <= 0) { fprintf(stderr, "remove with nonexistent element: %d %d\n", xv, yv); exit(1); }
    //hash[h]--;
    int pnni = XY_TO_INT(xv, yv);
    PositionLink **prev = &item[h];
    PositionLink *current = item[h];
    while (current) {
      if (current->v == pnni) {
        *prev = current->next;
        current->next = free_item;
        free_item = current;
        return;
      }
      prev = &current->next;
      current = current->next;
    }
    fprintf(stderr, "remove with nonexistent element: %d %d\n", xv, yv); exit(1);
    //return 0;
    //known[h] = KNOWN_EMPTY;
  }
  void check(int i, int id) {
    return;
    /*
    int hash2[NHASH] = { 0 };
    for (int j = 0; j < i; j++) {
      int h = HASH_FUNC(INT_TO_X(pnn[j]), INT_TO_Y(pnn[j]));
      hash2[h]++;
    }
    for (int j = 0; j < NHASH; j++) {
      if (hash[j] != hash2[j]) { fprintf(stderr, "hash counts differ at j=%d (%d, %d)\n", j, hash[j], hash2[j]); exit(1); }
    }
    for (int j = 0; j < i; j++) {
      for (int k = 0; k < j; k++) {
        if (pnn[j] == pnn[k]) { fprintf(stderr, "duplicate element: %d %d (positions %d %d of %d), id=%d\n", INT_TO_X(pnn[j]), INT_TO_Y(pnn[j]), j, k, i, id); exit(1); }
      }
    }
    */
  }
};

#undef NHASH
#undef NHASH_M1
/*#define NHASH 512
#define NHASH_M1 (NHASH-1)*/
#define NHASH nhash
#define NHASH_M1 nhash_m1

class LargeKPositionSet { public:
  //int hash[NHASH];
/*#if SMALLK
  PositionLink *item[NHASH];
  PositionLink item_data[NHASH];
#else*/
  PositionLink **item;
  PositionLink *item_data;
  int nhash;
  int nhash_m1;
//#endif
  PositionLink *free_item;
  //int known[NHASH]; // an item known to be at the given bucket, or KNOWN_EMPTY if no item known
  int *pnn;
  vector<qtype<int> > *q;
  LargeKPositionSet() {
  }
  /*
  void check_equals(LargeKPositionSet *other) {
    for (int i = 0; i < NHASH; i++) {
      if (hash[i] != other->hash[i]) { fprintf(stderr, "not equal at %d: %d %d\n", i, hash[i], other->hash[i]); exit(1); }
    }
  }
  */
  void init(int *pnn_, vector<qtype<int> > *q_, int nhash_) {
//#if !SMALLK
    nhash = roundup_pow2(nhash_)*4;
    nhash_m1 = nhash-1;
    item_data = new PositionLink[NHASH];
    item = new PositionLink *[NHASH];
//#endif
    //for (int i = 0; i < NHASH; i++) { hash[i] = 0; }
    for (int i = 0; i < NHASH; i++) { item[i] = NULL; }
    for (int i = 0; i < NHASH-1; i++) { item_data[i].next = &item_data[i+1]; }
    item_data[NHASH-1].next = NULL;
    free_item = &item_data[0];
    //for (int i = 0; i < NHASH; i++) { known[i] = KNOWN_EMPTY; }
    pnn = pnn_;
    q = q_;
  }
  LargeKPositionSet(int *pnn_, vector<qtype<int> > *q_, int nhash_) {
    init(pnn_, q_, nhash_);
/*
    for (int i = 0; i < NHASH; i++) { hash[i] = 0; }
    //for (int i = 0; i < NHASH; i++) { known[i] = KNOWN_EMPTY; }
    pnn = pnn_;
    q = q_;*/
  }
//#if !SMALLK
  ~LargeKPositionSet() {
    delete[] item_data;
    delete[] item;
  }
//#endif
  int contains(int xv, int yv, int i) {
    CHECK_COORDS
    int h = HASH_FUNC(xv, yv); //(xv+yv*3829)&(NHASH-1);
    //int pnni = ;
    PositionLink *current = item[h];
    while (current) {
      if (current->v == XY_TO_INT(xv, yv)) { return 1; }
      current = current->next;
    }
    return 0;
    /*
    if (!hash[h]) {
      return 0;
    } else {
      //if (XY_TO_INT(xv, yv) == known[h]) { return 1; }
      int pnni = XY_TO_INT(xv, yv);
      for (int j = 0; j < i; j++) {
        //if (pnn[j] == pnni) { return 1; }
        if ((*q)[j].b == pnni) { return 1; }
      }
      return 0;
    }
    */
  }
  int contains_noqueue(int xv, int yv, int i) {
    return contains(xv, yv, i);
    /*
    CHECK_COORDS
    int h = HASH_FUNC(xv, yv); //(xv+yv*3829)&(NHASH-1);
    if (!hash[h]) {
      return 0;
    } else {
      //if (XY_TO_INT(xv, yv) == known[h]) { return 1; }
      int pnni = XY_TO_INT(xv, yv);
      for (int j = 0; j < i; j++) {
        if (pnn[j] == pnni) { return 1; }
        //if ((*q)[j].b == pnni) { return 1; }
      }
      return 0;
    }*/
  }

  /* If element not in set, insert it and return 1.  If element in set, do nothing and return 0. */
  int try_insert(int xv, int yv, int i) {
    CHECK_COORDS
    //fprintf(stderr, "try_insert(%d, %d, %d)\n", xv, yv, i);
    int h = HASH_FUNC(xv, yv); //(xv+yv*3829)&(NHASH-1);
    int pnni = XY_TO_INT(xv, yv);
    PositionLink *current = item[h];
    while (current) {
      if (current->v == pnni) { return 0; }
      current = current->next;
    }
    PositionLink *add = free_item;
    free_item = free_item->next;
    if (!free_item) { fprintf(stderr, "hash table full\n"); exit(1); }
    add->v = pnni;
    add->next = item[h];
    item[h] = add;
    return 1;
    /*
    if (!hash[h]) {
      hash[h] = 1;
      //known[h] = XY_TO_INT(xv, yv);
      return 1;
    } else {
      int pnni = XY_TO_INT(xv, yv);
      for (int j = 0; j < i; j++) {
        if (pnn[j] == pnni) { return 0; }
      }
      hash[h]++;
      return 1;
    }
    */
  }
  /* Insert an element that's not in the set. */
  void insert_nonexistent(int xv, int yv, int verbose=1) {
    CHECK_COORDS
    int h = HASH_FUNC(xv, yv); //(xv+yv*3829)&(NHASH-1);
    int pnni = XY_TO_INT(xv, yv);
    PositionLink *add = free_item;
    free_item = free_item->next;
    if (!free_item) { fprintf(stderr, "hash table full\n"); exit(1); }
    add->v = pnni;
    add->next = item[h];
    item[h] = add;
    //if (verbose) { fprintf(stderr, "insert_nonexistent(%d, %d), h=%d\n", xv, yv, h); }
    //hash[h]++;
    //known[h] = XY_TO_INT(xv, yv);
  }
  /* Remove an element that's in the set. */
  void remove(int xv, int yv) {
    CHECK_COORDS
    int h = HASH_FUNC(xv, yv);
    //fprintf(stderr, "remove(%d, %d), h=%d\n", xv, yv, h);
    //if (hash[h] <= 0) { fprintf(stderr, "remove with nonexistent element: %d %d\n", xv, yv); exit(1); }
    //hash[h]--;
    int pnni = XY_TO_INT(xv, yv);
    PositionLink **prev = &item[h];
    PositionLink *current = item[h];
    while (current) {
      if (current->v == pnni) {
        *prev = current->next;
        current->next = free_item;
        free_item = current;
        return;
      }
      prev = &current->next;
      current = current->next;
    }
    fprintf(stderr, "remove with nonexistent element: %d %d\n", xv, yv); exit(1);
    //return 0;
    //known[h] = KNOWN_EMPTY;
  }
  void check(int i, int id) {
    return;
    /*
    int hash2[NHASH] = { 0 };
    for (int j = 0; j < i; j++) {
      int h = HASH_FUNC(INT_TO_X(pnn[j]), INT_TO_Y(pnn[j]));
      hash2[h]++;
    }
    for (int j = 0; j < NHASH; j++) {
      if (hash[j] != hash2[j]) { fprintf(stderr, "hash counts differ at j=%d (%d, %d)\n", j, hash[j], hash2[j]); exit(1); }
    }
    for (int j = 0; j < i; j++) {
      for (int k = 0; k < j; k++) {
        if (pnn[j] == pnn[k]) { fprintf(stderr, "duplicate element: %d %d (positions %d %d of %d), id=%d\n", INT_TO_X(pnn[j]), INT_TO_Y(pnn[j]), j, k, i, id); exit(1); }
      }
    }
    */
  }
};

#include <set>

void insert_vbmp(VBMP *bmp, int i, BITMAP *a) {
  if (a->w != bmp->w || a->h != bmp->h) { fprintf(stderr, "insert_vbmp: sizes differ (%dx%d) (%dx%d)\n", a->w, a->h, bmp->w, bmp->h); exit(1); }
  if ((unsigned) i >= (unsigned) bmp->n) { fprintf(stderr, "i out of range in insert_vbmp %d of %d\n", i, bmp->n); exit(1); }
  for (int y = 0; y < bmp->h; y++) {
    for (int x = 0; x < bmp->w; x++) {
      bmp->get(x, y)[i] = ((int *) a->line[y])[x];
    }
  }
}

BITMAP *extract_vbmp(VBMP *bmp, int i) {
  if ((unsigned) i >= (unsigned) bmp->n) { fprintf(stderr, "extract_vbmp index out of bounds\n"); exit(1); }
  BITMAP *ans = create_bitmap(bmp->w, bmp->h);
  for (int y = 0; y < bmp->h; y++) {
    for (int x = 0; x < bmp->w; x++) {
      int c = bmp->get(x, y)[i];
      ((int *) ans->line[y])[x] = c;
    }
  }
  return ans;
}

VBMP *knn_init_nn(Params *p, BITMAP *a, BITMAP *b, VBMP *&ann_sim, PRINCIPAL_ANGLE *pa) {
  init_xform_tables();
  VBMP *ann = new VBMP(a->w, a->h, p->knn);
#if TRANSLATE_ONLY
  ann_sim = NULL;
#else
  ann_sim = new VBMP(a->w, a->h, p->knn);
#endif
  //int h = p->patch_w/2;
  int bew = BEW, beh = BEH;
  int USE_PA = p->knn_algo == KNN_ALGO_PRINANGLE;
  if (USE_PA && !pa) { fprintf(stderr, "USE_PA=1 and pa is NULL\n"); exit(1); }
  if (p->knn_algo == KNN_ALGO_TOP1NN) {
#if !TRANSLATE_ONLY
    fprintf(stderr, "Unimplemented for rotation/scale\n"); exit(1);
#endif
    #pragma omp parallel for schedule(static,8)
    for (int y = 0; y < AEH; y++) {
      int seed = rand()^(y<<11);
      for (int x = 0; x < AEW; x++) {
        int *pnn = ann->get(x, y);

        seed = RANDI(seed);
        int xv = seed%bew;
        seed = RANDI(seed);
        int yv = seed%beh;
        *pnn = XY_TO_INT(xv, yv);
      }
    }
  } else {
    #pragma omp parallel for schedule(static,8)
    for (int y = 0; y < AEH; y++) {
      int seed = rand()^(y<<11);
      //int *ann_row = (int *) ann->line[y];
      //int *sim_row = (int *) ann_sim->line[y];
      for (int x = 0; x < AEW; x++) {
        int angle0 = 0;
        if (USE_PA) {
          angle0 = get_principal_angle(p, pa, x, y, SCALE_UNITY);
        }
        int *pnn = ann->get(x, y);
  #if !TRANSLATE_ONLY
        int *psim = ann_sim->get(x, y);
  #endif
        //set<int> chosen;
        //int hash[NHASH] = { 0 };
        PositionSet chosen(pnn, NULL, p->knn);
        for (int i = 0; i < p->knn; i++) {
          for (;;) {
            seed = RANDI(seed);
            int xv = seed%bew;
            seed = RANDI(seed);
            int yv = seed%beh;
            pnn[i] = XY_TO_INT(xv, yv);
            /*int h = (xv+yv*3829)&(NHASH-1);
            if (!hash[h]) {
              hash[h]++;
              break;
            } else {
              int done = 1;
              for (int j = 0; j < i; j++) {
                if (pnn[j] == pnn[i]) { done = 0; break; }
              }
              if (done) { break; }
            }
            */
            if (chosen.try_insert(xv, yv, i)) { break; }
            //break; // FIXME
            //if (chosen.count(pnn[i]) == 0) { break; }
          }
          //fprintf(stderr, "first check\n");
          //chosen.check(i+1, 0);
          //chosen.insert(pnn[i]);
#if !TRANSLATE_ONLY
          int xpp = INT_TO_X(pnn[i]), ypp = INT_TO_Y(pnn[i]);
          int tpp = 0;
          int spp = p->allow_scale ? (rand()%NUM_SCALES): SCALE_UNITY;
          //fprintf(stderr, "p->allow_scale: %d\n", p->allow_scale);
          if (p->allow_rotate) {
            tpp = (rand()%NUM_ANGLES);
            if (USE_PA && rand() < int(RAND_MAX*p->prob_prinangle)) {
              tpp = (get_principal_angle(p, pa, xpp, ypp, spp)-angle0)&(NUM_ANGLES-1);
            }
          }
          psim[i] = XY_TO_INT(spp, tpp);
#endif
          //check_offset(p, b, x, y, xp, yp);
        }
      }
    }
  }
  return ann;
}

double t_linear = 0;

template<int PATCH_W, int D_KNOWN>
void knn_attempt_n(vector<qtype<int> > &q, int *adata, BITMAP *b, XFORM bpos, int bx, int by, int bs, int bt, Params *p, int dval_known, PositionSet &pos0) {
  //int h = PATCH_W/2;
  if (q.size() != p->knn) { fprintf(stderr, "q size is wrong (%d, %d)\n", q.size(), p->knn); exit(1); }
  if ((unsigned) (bx) < (unsigned) (b->w-PATCH_W+1) &&
      (unsigned) (by) < (unsigned) (b->h-PATCH_W+1)) {
    int pos = XY_TO_INT(bx, by);
    //double start_t = accurate_timer();
    #if 0
    for (int i = 0; i < p->knn; i++) {
      if (pos == q[i].b /* abs(bx-INT_TO_X(q[i].b)) <= 1 && abs(by-INT_TO_Y(q[i].b)) <= 1*/) {
        //if (index >= 0) { fprintf(stderr, "index found twice\n"); exit(1); };
        index = i;
      }
    }
    #endif
#if TRANSLATE_ONLY
    if (pos0.contains(bx, by, p->knn)) { return; }
#else
    int index = pos0.contains(bx, by, p->knn) ? 0: -1;
    if (index >= 0) {
      for (int i = 0; i < p->knn; i++) {
        if (pos == q[i].b /* abs(bx-INT_TO_X(q[i].b)) <= 1 && abs(by-INT_TO_Y(q[i].b)) <= 1*/) {
          //if (index >= 0) { fprintf(stderr, "index found twice\n"); exit(1); };
          index = i;
          break;
        }
      }
    }
#endif
  //return;
    //t_linear += accurate_timer()-start_t;
    int err = q[0].a;
    /*for (int i = 1; i < p->knn; i++) {
      if (q[i].a > err) { fprintf(stderr, "first element is not largest\n"); exit(1); }
    }*/
    //if (!is_heap(q.begin(), q.end())) { fprintf(stderr, "not a heap\n"); exit(1); }
    //XFORM bpos = get_xform(p, bx, by, bs, bt);
    int current;
    if (D_KNOWN) {
      current = dval_known;
    } else {
#if TRANSLATE_ONLY
      current = fast_patch_dist<PATCH_W, 0>(adata, b, bpos.x0>>16, bpos.y0>>16, err, p);
#else
      current = sim_fast_patch_dist<PATCH_W, 1>(adata, b, bpos, err);
#endif
    }
    if (current < err) {
#if !TRANSLATE_ONLY
      if (index >= 0) {
        //printf("knn_attempt, improve %d\n", index);
        if (current < q[index].a) {
          //start_t = accurate_timer();
          q[index] = qtype<int>(current, XY_TO_INT(bx, by), XY_TO_INT(bs, bt));
          q.push_back(qtype<int>(0, 0, 0)); // Bug in make_heap()?  Requires one extra element
          make_heap(&q[0], &q[p->knn]);
          q.pop_back();
          //if (q.size() != p->knn) { fprintf(stderr, "q size is wrong after replacing element (%d, %d)\n", q.size(), p->knn); exit(1); }
          //t_linear += accurate_timer()-start_t;
        }
      } else {
        //return;
#endif
#if !TRANSLATE_ONLY
        q.push_back(qtype<int>(-1,-1,-1));  // Bug in pop_heap()/push_heap()?  Requires one extra element
#else
        q.push_back(qtype<int>(-1,-1));  // Bug in pop_heap()/push_heap()?  Requires one extra element
#endif
        //printf("knn positionset remove\n");
        //printf("knn_attempt, remove %d %d\n", INT_TO_X(q[0].b), INT_TO_Y(q[0].b));
        //for (int ij = 0; ij < p->knn; ij++) {
        //  printf("  heap: %d %d\n", INT_TO_X(q[ij].b), INT_TO_Y(q[ij].b));
        //}
        pos0.remove(INT_TO_X(q[0].b), INT_TO_Y(q[0].b));
        pop_heap(&q[0], &q[p->knn]);
        pos0.insert_nonexistent(bx, by);
#if !TRANSLATE_ONLY
        q[p->knn-1] = qtype<int>(current, XY_TO_INT(bx, by), XY_TO_INT(bs, bt));
#else
        q[p->knn-1] = qtype<int>(current, XY_TO_INT(bx, by));
#endif
        //for (int ij = 0; ij < p->knn; ij++) {
        //  printf("  heap after remove: %d %d\n", INT_TO_X(q[ij].b), INT_TO_Y(q[ij].b));
        //}
        //printf("inserting %d %d\n", bx, by);
        push_heap(&q[0], &q[p->knn]);
//        if (q[q.size()-1].a != -1 || q[q.size()-1].b != -1 || q[q.size()-1].c != -1) { fprintf(stderr, "last element isn't (-1,-1,-1) sentinel\n"); exit(1); }
        q.pop_back();
        //if (q.size() != p->knn) { fprintf(stderr, "q size is wrong after push-pop (%d, %d)\n", q.size(), p->knn); exit(1); }
#if !TRANSLATE_ONLY
      }
#endif
      /*xbest = bx;
      ybest = by;
      sbest = bs;
      tbest = bt;*/
    }
  }
}

template<int PATCH_W>
VBMP *knn_init_dist_n(Params *p, BITMAP *a, BITMAP *b, VBMP *ann, VBMP *ann_sim) {
  init_xform_tables();
  VBMP *ans = new VBMP(a->w, a->h, p->knn);
  clear_to_color(ans, INT_MAX);
#if TRANSLATE_ONLY
  if (ann_sim) { fprintf(stderr, "in mode TRANSLATE_ONLY, expected ann_sim to be NULL\n"); exit(1); }
#endif
  if (p->knn_algo == KNN_ALGO_TOP1NN) {
#if !TRANSLATE_ONLY
    fprintf(stderr, "Unimplemented for rotation/scale\n"); exit(1);
#endif

    #pragma omp parallel for schedule(static,4)
    for (int y = 0; y < AEH; y++) {
      for (int x = 0; x < AEW; x++) {
        int *p_ann = ann->get(x, y);
        int *p_ans = ans->get(x, y);
        int anni = *p_ann;
        int bx = INT_TO_X(anni), by = INT_TO_Y(anni);
        int d = patch_dist_ab<PATCH_W, 0, 0>(p, a, x, y, b, bx, by, INT_MAX, NULL);
        *p_ans = d;
      }
    }
  } else {
    #pragma omp parallel for schedule(static,4)
    for (int y = 0; y < AEH; y++) {
      vector<qtype<int> > v;
      v.reserve(p->knn+1);
      //int *row = (int *) ans->line[y];
      for (int x = 0; x < AEW; x++) {
        int *p_ann = ann->get(x, y);
  #if !TRANSLATE_ONLY
        int *p_ann_sim = ann_sim->get(x, y);
  #endif
        int *p_ans = ans->get(x, y);
        v.clear();
        for (int i = 0; i < p->knn; i++) {
          //int bx, by, bs, bt;
          //getnn(ann, x, y, bx, by);
          //getnn(ann_sim, x, y, bs, bt);
          int anni = p_ann[i];
          int bx = INT_TO_X(anni), by = INT_TO_Y(anni);
  #if !TRANSLATE_ONLY
          int ann_simi = p_ann_sim[i];
          int bs = INT_TO_X(ann_simi), bt = INT_TO_Y(ann_simi);
          int d = sim_patch_dist_ab<PATCH_W, 0>(p, a, x, y, b, bx, by, bs, bt, INT_MAX);
          v.push_back(qtype<int>(d, anni, ann_simi));
  #else
          int d = patch_dist_ab<PATCH_W, 0, 0>(p, a, x, y, b, bx, by, INT_MAX, NULL);
          v.push_back(qtype<int>(d, anni));
  #endif
        }
        if (p->knn_algo == KNN_ALGO_HEAP) {
  #if !TRANSLATE_ONLY
          v.push_back(qtype<int>(0, 0, 0)); // Bug in make_heap()?  Requires one extra element
  #else
          v.push_back(qtype<int>(0, 0)); // Bug in make_heap()?  Requires one extra element
  #endif
          //fprintf(stderr, "before make_heap %d\n", v.size()); fflush(stderr);
          make_heap(&v[0], &v[p->knn]);
          v.pop_back();
        }
        if (v.size() != p->knn) { fprintf(stderr, "v size != knn (%d, %d)\n", v.size(), p->knn); exit(1); }
        //fprintf(stderr, "after make_heap\n"); fflush(stderr);
        for (int i = 0; i < p->knn; i++) {
          qtype<int> current = v[i];
          p_ans[i] = current.a;
          p_ann[i] = current.b;
  #if !TRANSLATE_ONLY
          p_ann_sim[i] = current.c;
  #endif
        }
      }
    }
  }
  return ans;
}

VBMP *knn_init_dist(Params *p, BITMAP *a, BITMAP *b, VBMP *ann, VBMP *ann_sim) {
  if      (p->patch_w == 1 ) { return knn_init_dist_n<1>(p, a, b, ann, ann_sim); }
  else if (p->patch_w == 2 ) { return knn_init_dist_n<2>(p, a, b, ann, ann_sim); }
  else if (p->patch_w == 3 ) { return knn_init_dist_n<3>(p, a, b, ann, ann_sim); }
  else if (p->patch_w == 4 ) { return knn_init_dist_n<4>(p, a, b, ann, ann_sim); }
  else if (p->patch_w == 5 ) { return knn_init_dist_n<5>(p, a, b, ann, ann_sim); }
  else if (p->patch_w == 6 ) { return knn_init_dist_n<6>(p, a, b, ann, ann_sim); }
  else if (p->patch_w == 7 ) { return knn_init_dist_n<7>(p, a, b, ann, ann_sim); }
  else if (p->patch_w == 8 ) { return knn_init_dist_n<8>(p, a, b, ann, ann_sim); }
  else if (p->patch_w == 9 ) { return knn_init_dist_n<9>(p, a, b, ann, ann_sim); }
  else if (p->patch_w == 10) { return knn_init_dist_n<10>(p, a, b, ann, ann_sim); }
  else if (p->patch_w == 11) { return knn_init_dist_n<11>(p, a, b, ann, ann_sim); }
  else if (p->patch_w == 12) { return knn_init_dist_n<12>(p, a, b, ann, ann_sim); }
  else if (p->patch_w == 13) { return knn_init_dist_n<13>(p, a, b, ann, ann_sim); }
  else if (p->patch_w == 14) { return knn_init_dist_n<14>(p, a, b, ann, ann_sim); }
  else if (p->patch_w == 15) { return knn_init_dist_n<15>(p, a, b, ann, ann_sim); }
  else if (p->patch_w == 16) { return knn_init_dist_n<16>(p, a, b, ann, ann_sim); }
  else if (p->patch_w == 17) { return knn_init_dist_n<17>(p, a, b, ann, ann_sim); }
  else if (p->patch_w == 18) { return knn_init_dist_n<18>(p, a, b, ann, ann_sim); }
  else if (p->patch_w == 19) { return knn_init_dist_n<19>(p, a, b, ann, ann_sim); }
  else if (p->patch_w == 20) { return knn_init_dist_n<20>(p, a, b, ann, ann_sim); }
  else if (p->patch_w == 21) { return knn_init_dist_n<21>(p, a, b, ann, ann_sim); }
  else if (p->patch_w == 22) { return knn_init_dist_n<22>(p, a, b, ann, ann_sim); }
  else if (p->patch_w == 23) { return knn_init_dist_n<23>(p, a, b, ann, ann_sim); }
  else if (p->patch_w == 24) { return knn_init_dist_n<24>(p, a, b, ann, ann_sim); }
  else if (p->patch_w == 25) { return knn_init_dist_n<25>(p, a, b, ann, ann_sim); }
  else if (p->patch_w == 26) { return knn_init_dist_n<26>(p, a, b, ann, ann_sim); }
  else if (p->patch_w == 27) { return knn_init_dist_n<27>(p, a, b, ann, ann_sim); }
  else if (p->patch_w == 28) { return knn_init_dist_n<28>(p, a, b, ann, ann_sim); }
  else if (p->patch_w == 29) { return knn_init_dist_n<29>(p, a, b, ann, ann_sim); }
  else if (p->patch_w == 30) { return knn_init_dist_n<30>(p, a, b, ann, ann_sim); }
  else if (p->patch_w == 31) { return knn_init_dist_n<31>(p, a, b, ann, ann_sim); }
  else if (p->patch_w == 32) { return knn_init_dist_n<32>(p, a, b, ann, ann_sim); }
  else { fprintf(stderr, "Patch width unsupported: %d\n", p->patch_w); exit(1); }
}

int save_dist_count = 0;
BITMAP *dprev = NULL;

void normalize_bitmap(BITMAP *ans) {
  int dmax = 0;
  for (int y = 0; y < ans->h; y++) {
    int *row = (int *) ans->line[y];
    for (int x = 0; x < ans->w; x++) {
      if (row[x] > dmax) { dmax = row[x]; }
    }
  }
  if (dmax == 0) { dmax = 1; }
  for (int y = 0; y < ans->h; y++) {
    int *row = (int *) ans->line[y];
    for (int x = 0; x < ans->w; x++) {
      int c = row[x]*255/dmax;
      row[x] = (c)|(c<<8)|(c<<16);
    }
  }  
}

void save_dist(Params *p, VBMP *annd, const char *suffix) {
  fprintf(stderr, "save_dist unimplemented\n"); exit(1);
#if 0
  BITMAP *ans = create_bitmap(annd->w, annd->h);
  for (int y = 0; y < ans->h; y++) {
    int *row = (int *) ans->line[y];
    for (int x = 0; x < ans->w; x++) {
      int *p_annd = annd->get(x, y);
      int d = 0;
      for (int i = 0; i < p->knn; i++) {
        d += p_annd[i];
      }
      row[x] = d;
    }
  }
  char buf[256];
  sprintf(buf, "dist%02d%s.bmp", save_dist_count, suffix);
  save_bitmap(buf, ans, NULL);
  if (dprev) {
    BITMAP *delta = create_bitmap(annd->w, annd->h);
    for (int y = 0; y < ans->h; y++) {
      int *arow = (int *) dprev->line[y];
      int *brow = (int *) ans->line[y];
      int *crow = (int *) delta->line[y];
      for (int x = 0; x < ans->w; x++) {
        crow[x] = arow[x]-brow[x];
        if (crow[x] < 0) { fprintf(stderr, "Distance increased\n"); exit(1); }
      }
    }
    normalize_bitmap(delta);
    char buf2[256];
    sprintf(buf2, "delta%02d%s.bmp", save_dist_count, suffix);
    save_bitmap(buf2, delta, NULL);

    destroy_bitmap(delta);
  }
  //destroy_bitmap(ans);
  save_dist_count++;
  destroy_bitmap(dprev);
  dprev = ans;
#endif
}

VBMP *copy_vbmp(VBMP *a) {
  if (!a) { fprintf(stderr, "copy_vbmp received NULL\n"); exit(1); }
  VBMP *ans = new VBMP(a->w, a->h, a->n);
  for (int y = 0; y < a->h; y++) {
    for (int x = 0; x < a->w; x++) {
      int *src  = a->get(x, y);
      int *dest = ans->get(x, y);
      for (int i = 0; i < a->n; i++) {
        dest[i] = src[i];
      }
    }
  }
  return ans;
}

int contains_pair(Params *p, int *p_ann, int *p_annd, knn_pair<int> x) {
  for (int i = 0; i < p->knn; i++) {
    if (p_ann[i] == x.b) {
      if (p_annd[i] != x.a) { fprintf(stderr, "distances disagree in contains_pair: %d %d %d\n", p_ann[i], p_annd[i], x.a); exit(1); }
      return 1;
    }
  }
  return 0;
}

void check_change_knn(Params *p, BITMAP *a, BITMAP *b) {
  if (p->knn == 1) { fprintf(stderr, "Error: check_change_knn requires -knn argument >= 2\n"); exit(1); }
  VBMP *sim_ignore = NULL;

  VBMP *ann = knn_init_nn(p, a, b, sim_ignore);
  VBMP *annd = knn_init_dist(p, a, b, ann, NULL);
  knn_check(p, a, b, ann, NULL, annd);
  knn(p, a, b, ann, sim_ignore, annd);
  knn_check(p, a, b, ann, NULL, annd);
  
  int k1 = p->knn/2;
  Params p1(*p);
  VBMP *ann1 = copy_vbmp(ann);
  VBMP *annd1 = copy_vbmp(annd);
  change_knn(&p1, a, b, ann1, sim_ignore, annd1, k1);

  int k2 = p->knn*2;
  Params p2(*p);
  VBMP *ann2 = copy_vbmp(ann);
  VBMP *annd2 = copy_vbmp(annd);
  change_knn(&p2, a, b, ann2, sim_ignore, annd2, k2);

  printf("a: %dx%d, b: %dx%d, ann1: %dx%d, annd1: %dx%d, ann2: %dx%d, annd2: %dx%d\n", a->w, a->h, b->w, b->h, ann1->w, ann1->h, annd1->w, annd1->h, ann2->w, ann2->h, annd2->w, annd2->h);
  printf("%d %d %d %d %d %d\n", ann1->n, ann2->n, annd1->n, annd2->n, ann->n, annd->n);

  printf("knn_check ann1:\n");
  knn_check(&p1, a, b, ann1, NULL, annd1);
  printf("OK\n");
  printf("knn_check ann2:\n");
  knn_check(&p2, a, b, ann2, NULL, annd2);
  printf("OK\n");

  int n_not_special = 0;
  int n_special = 0;
  for (int y = 0; y < AEH; y++) {
    for (int x = 0; x < AEW; x++) {
      vector<knn_pair<int> > L;
      int *p_ann = ann->get(x, y);
      int *p_annd = annd->get(x, y);
      for (int i = 0; i < p->knn; i++) {
        L.push_back(knn_pair<int>(p_annd[i], p_ann[i]));
      }
      sort(L.begin(), L.end());
      //knn_pair<int> worst = L[L.size()-1];
      //knn_pair<int> best = L[0];
      //if (best.a > worst.a) { fprintf(stderr, "Best distance (%d) >= worst distance (%d)\n", best.a, worst.a); exit(1); }
      int *p_ann1 = ann1->get(x, y);
      int *p_annd1 = annd1->get(x, y);
      int *p_ann2 = ann2->get(x, y);
      int *p_annd2 = annd2->get(x, y);
      //if (contains_pair(p, p_ann1, p_annd1, worst)) { fprintf(stderr, "k1 contains worst %d %d\n", x, y); exit(1); }
      //if (!contains_pair(p, p_ann1, p_annd1, best)) { fprintf(stderr, "k1 does not contain best %d %d\n", x, y); exit(1); }
      for (int i = 0; i < p->knn; i++) {
        int should_contain = i < k1;

        int currentd = L[i].a;
        int prevd = i-1>=0 ? L[i-1].a: -1;
        int nextd = i+1<((int) L.size()) ? L[i+1].a: -1;
        int special = (currentd == nextd || currentd == prevd);
        if (!special) { n_not_special++; }
        if (special) { n_special++; }

        if (contains_pair(&p1, p_ann1, p_annd1, L[i]) != should_contain) {
          if (!special) {
            fprintf(stderr, "k1 containment (%d %d) wrong at ith (%d) %d %d (d=%d, next d=%d, prev d=%d)\n", contains_pair(p, p_ann1, p_annd1, L[i]), should_contain, i, x, y, currentd, nextd, prevd);
            exit(1);
          }
        }
      }
      for (int i = 0; i < p->knn; i++) {
        if (!contains_pair(&p2, p_ann2, p_annd2, L[i])) { fprintf(stderr, "k2 does not contain ith (%d) %d %d\n", i, x, y); exit(1); }
      }
    }
  }
  printf("n not special: %d\n", n_not_special);
  printf("n special: %d\n", n_special);
  printf("pixels*knn: %d\n", a->w*a->h*p->knn);
  fprintf(stderr, "check_change_knn: OK\n");
}

void change_knn(Params *p, BITMAP *a, BITMAP *b, VBMP *&ann0, VBMP *&ann_sim0, VBMP *&annd0, int kp, PRINCIPAL_ANGLE *pa) {
  if (kp == p->knn) { return; }
  VBMP *ann = new VBMP(ann0->w, ann0->h, kp);
  VBMP *annd = new VBMP(ann0->w, ann0->h, kp);
#if TRANSLATE_ONLY
  if (ann_sim0) { fprintf(stderr, "TRANSLATE_ONLY, expected ann_sim0 to be NULL\n"); exit(1); }
#else
  if (!ann_sim0) { fprintf(stderr, "!TRANSLATE_ONLY, expected ann_sim0 to be non-NULL\n"); exit(1); }
  VBMP *ann_sim = new VBMP(ann0->w, ann0->h, kp);
#endif
  int kmin = MIN(p->knn, kp);
  if (kp >= p->knn) {
    for (int y = 0; y < ann0->h; y++) {
      for (int x = 0; x < ann0->w; x++) {
        int *p_ann = ann0->get(x, y);
        int *p_annd = annd0->get(x, y);
#if !TRANSLATE_ONLY
        int *p_ann_sim = ann_sim0->get(x, y);
#endif
        int *q_ann = ann->get(x, y);
        int *q_annd = annd->get(x, y);
#if !TRANSLATE_ONLY
        int *q_ann_sim = ann_sim->get(x, y);
#endif
        for (int i = 0; i < kmin; i++) {
          q_ann[i] = p_ann[i];
          q_annd[i] = p_annd[i];
#if !TRANSLATE_ONLY
          q_ann_sim[i] = p_ann_sim[i];
#endif
        }
      }
    }
  } else {
    for (int y = 0; y < ann0->h; y++) {
      vector<qtype<int> > v;
      v.reserve(p->knn);
      for (int x = 0; x < ann0->w; x++) {
        int *p_ann = ann0->get(x, y);
        int *p_annd = annd0->get(x, y);
#if !TRANSLATE_ONLY
        int *p_ann_sim = ann_sim0->get(x, y);
#endif

        v.clear();
        for (int i = 0; i < p->knn; i++) {
#if !TRANSLATE_ONLY
          v.push_back(qtype<int>(-p_annd[i], p_ann[i], p_ann_sim[i]));
#else
          v.push_back(qtype<int>(-p_annd[i], p_ann[i]));
#endif
        }
        int numv = p->knn;
#if !TRANSLATE_ONLY
        v.push_back(qtype<int>(1,-1,-1));  // Bug in pop_heap()/push_heap()?  Requires one extra element
#else
        v.push_back(qtype<int>(1,-1));  // Bug in pop_heap()/push_heap()?  Requires one extra element
#endif
        make_heap(&v[0], &v[p->knn]);
        int *q_ann = ann->get(x, y);
        int *q_annd = annd->get(x, y);
#if !TRANSLATE_ONLY
        int *q_ann_sim = ann_sim->get(x, y);
#endif
        for (int i = 0; i < kmin; i++) {
          //pos0.remove(INT_TO_X(q[0].b), INT_TO_Y(q[0].b));
          //q.pop_back();

          q_ann[i] = v[0].b;
          q_annd[i] = -v[0].a;
          //if (x == 0 && y == 0) { printf("Element %d is: %d %d (d=%d)\n", i, INT_TO_X(q_ann[i]), INT_TO_Y(q_ann[i]), q_annd[i]); }
#if !TRANSLATE_ONLY
          q_ann_sim[i] = v[0].c;
#endif
          if (i < kmin - 1) { pop_heap(&v[0], &v[numv]); numv--; }
        }
      }
    }
  }
  
  if (kp > p->knn) {
    Params pcopy(*p);
    pcopy.knn = kp-p->knn;
    VBMP *nn_sim = NULL;
    VBMP *nn = knn_init_nn(&pcopy, a, b, nn_sim, pa);
    int bew = BEW, beh = BEH;

    #pragma omp parallel for schedule(static,8)
    for (int y = 0; y < AEH; y++) {
      int seed = rand()^(y<<11);
      for (int x = 0; x < AEW; x++) {
        int *p_ann = nn->get(x, y);
        int *q_ann = ann->get(x, y);

        PositionSet chosen(q_ann, NULL, p->knn);
        for (int i = 0; i < p->knn; i++) {
          chosen.insert_nonexistent(INT_TO_X(q_ann[i]), INT_TO_Y(q_ann[i]));
        }
        for (int i = p->knn; i < kp; i++) {
          for (;;) {
            seed = RANDI(seed);
            int xv = seed%bew;
            seed = RANDI(seed);
            int yv = seed%beh;
            p_ann[i-p->knn] = q_ann[i] = XY_TO_INT(xv, yv);
            if (chosen.try_insert(xv, yv, i)) { break; }
          }
        }
      }
    }
/*
    VBMP *nnd = knn_init_dist(&pcopy, a, b, nn, nn_sim);

    #pragma omp parallel for schedule(static,8)
    for (int y = 0; y < ann0->h; y++) {
      for (int x = 0; x < ann0->w; x++) {
        int *p_ann = nn->get(x, y);
        int *p_annd = nnd->get(x, y);
#if !TRANSLATE_ONLY
        int *p_ann_sim = nn_sim->get(x, y);
#endif
        int *q_ann = ann->get(x, y)+p->knn;
        int *q_annd = annd->get(x, y)+p->knn;
#if !TRANSLATE_ONLY
        int *q_ann_sim = ann_sim->get(x, y)+p->knn;
#endif
        for (int i = 0; i < pcopy.knn; i++) {
          q_ann[i] = p_ann[i];
          q_annd[i] = p_annd[i];
#if !TRANSLATE_ONLY
          q_ann_sim[i] = p_ann_sim[i];
#endif
        }
      }
    }
    */
    delete annd;
    Params pnew(*p);
    pnew.knn = kp;
#if TRANSLATE_ONLY
    VBMP *ann_sim = NULL;
#endif
    annd = knn_init_dist(&pnew, a, b, ann, ann_sim);
  }
  
  delete ann0;
  delete annd0;
#if !TRANSLATE_ONLY
  delete ann_sim0;
  ann_sim0 = ann_sim;
#endif
  ann0 = ann;
  annd0 = annd;
  
  p->knn = kp;
}

void get_best(Params *p, int *&ann, int *annd, int *&ann_sim) {
  int *ann_best = &ann[0], *annd_best = &annd[0];
#if !TRANSLATE_ONLY
  int *ann_sim_best = &ann_sim[0];
#endif
  for (int i = 1; i < p->knn; i++) {
    if (annd[i] < *annd_best) {
      ann_best = &ann[i];
      annd_best = &annd[i];
#if !TRANSLATE_ONLY
      ann_sim_best = &ann_sim[i];
#endif
    }
  }
  ann = ann_best;
  //annd = annd_best;
#if !TRANSLATE_ONLY
  ann_sim = ann_sim_best;
#endif
}

template<int PATCH_W, int USE_PA>
void knn_n(Params *p, BITMAP *a, BITMAP *b,
            VBMP *ann, VBMP *ann_sim, VBMP *annd,
            RegionMasks *amask=NULL, BITMAP *bmask=NULL,
            int level=0, int em_iter=0, RecomposeParams *rp=NULL, int offset_iter=0, int update_type=0, int cache_b=0,
            RegionMasks *region_masks=NULL, int tiles=-1, PRINCIPAL_ANGLE *pa=NULL, int save_first=1) {
  init_xform_tables();
  if (tiles < 0) { tiles = p->cores; }
#if TRANSLATE_ONLY
  if (ann_sim) { fprintf(stderr, "in mode TRANSLATE_ONLY, expected ann_sim to be NULL\n"); exit(1); }
#endif
  if (USE_PA && !pa) { fprintf(stderr, "knn_n: USE_PA=1 but no pa argument\n"); exit(1); }
  
  printf("in knn_nn_n, tiles=%d, rs_max=%d\n", tiles, p->rs_max);
  Box box = get_abox(p, a, amask);
//  poll_server();
  int nn_iter = 0;
  for (; nn_iter < p->nn_iters; nn_iter++) {
#if SAVE_DIST
    if (save_first || nn_iter != 0) {
      save_dist(p, annd, "nn");
    }
#endif
    unsigned int iter_seed = rand();
//    if (p->update) { p->update(level, em_iter, nn_iter, p, rp, a, b, NULL, NULL, NULL, NULL, bmask, NULL, region_masks, amask, update_type, p->q); }

    #pragma omp parallel num_threads(tiles)
    {
#if SYNC_WRITEBACK
      int *ann_writeback = new int[a->w*p->knn];
      int *annd_writeback = new int[a->w*p->knn];
#if !TRANSLATE_ONLY
#error Not implemented for case TRANSLATE_ONLY=0 and SYNC_WRITEBACK=1
#endif
#endif
#if USE_OPENMP
      int ithread = omp_get_thread_num();
#else
      int ithread = 0;
#endif
      int xmin = box.xmin, xmax = box.xmax;
      int ymin = box.ymin + (box.ymax-box.ymin)*ithread/tiles;
      int ymax = box.ymin + (box.ymax-box.ymin)*(ithread+1)/tiles;

      int ystart = ymin, yfinal = ymax, ychange=1;
      int xstart = xmin, xfinal = xmax, xchange=1;
      if ((nn_iter + offset_iter) % 2 == 1) {
        xstart = xmax-1; xfinal = xmin-1; xchange=-1;
        ystart = ymax-1; yfinal = ymin-1; ychange=-1;
      }
      int dx = -xchange, dy = -ychange;

      int bew = b->w-PATCH_W, beh = b->h-PATCH_W;
      int max_mag = MAX(b->w, b->h);
      int rs_ipart = int(p->rs_iters);
      double rs_fpart = p->rs_iters - rs_ipart;
      int rs_max = p->rs_max;
      if (rs_max > max_mag) { rs_max = max_mag; }

      vector<qtype<int> > v;
      v.reserve(p->knn);
      for (int i = 0; i < p->knn; i++) {
#if !TRANSLATE_ONLY
        v.push_back(qtype<int>(0, 0, 0));
#else
        v.push_back(qtype<int>(0, 0));
#endif
      }
#if (RS_RAND1TOK||P_RAND1TOK)
      vector<qtype<int> > vcopy;
      vcopy.reserve(p->knn);
#endif
#if RS_RAND1TOK
#define VARRAY vcopy
#else
#define VARRAY v
#endif
      
      int adata[PATCH_W*PATCH_W];
      for (int y = ystart; y != yfinal; y += ychange) {
        //int *annd_row = (int *) annd->line[y];
        for (int x = xstart; x != xfinal; x += xchange) {
          int angle0;
          if (USE_PA) {
            angle0 = get_principal_angle(p, pa, x, y, SCALE_UNITY);
          }
          for (int dy0 = 0; dy0 < PATCH_W; dy0++) {
            int *drow = ((int *) a->line[y+dy0])+x;
            int *adata_row = adata+(dy0*PATCH_W);
            for (int dx0 = 0; dx0 < PATCH_W; dx0++) {
              adata_row[dx0] = drow[dx0];
            }
          }
          
          /*int xbest, ybest, sbest, tbest;
          getnn(ann, x, y, xbest, ybest);
          getnn(ann_sim, x, y, sbest, tbest);
          check_offset(p, b, x, y, xbest, ybest);

          int err = annd_row[x];
          if (err == 0) { continue; }*/
          int *p_ann = ann->get(x, y);
          int *p_annd = annd->get(x, y);
          PositionSet pos(p_ann, &v, p->knn);
#if !TRANSLATE_ONLY
          int *p_ann_sim = ann_sim->get(x, y);
#endif
//          v.clear();
          for (int i = 0; i < p->knn; i++) {
#if !TRANSLATE_ONLY
            v[i] = qtype<int>(p_annd[i], p_ann[i], p_ann_sim[i]);
#else
            v[i] = qtype<int>(p_annd[i], p_ann[i]);
#endif
            pos.insert_nonexistent(INT_TO_X(p_ann[i]), INT_TO_Y(p_ann[i]));
            /*if (!pos.try_insert(INT_TO_X(p_ann[i]), INT_TO_Y(p_ann[i]), i)) {
              fprintf(stderr, "could not insert element to PositionSet: %d %d\n", INT_TO_X(p_ann[i]), INT_TO_Y(p_ann[i])); exit(1);
            }*/
          }

#if ENRICH_DURING
          for (int j = 0; j < p->knn; j++) {
            int xp = INT_TO_X(p_ann[j]), yp = INT_TO_Y(p_ann[j]);
            int *q_ann = ann->get(xp, yp);
            int *q_annd = annd->get(xp, yp);
    #if !TRANSLATE_ONLY
            int *q_ann_sim = ann_sim_temp->get(x, y);
    #endif
            for (int k = 0; k < p->knn; k++) {
              int xpp = INT_TO_X(q_ann[k]), ypp = INT_TO_Y(q_ann[k]);
    #if !TRANSLATE_ONLY
              int spp = INT_TO_X(q_ann_sim[k]), tpp = INT_TO_Y(q_ann_sim[k]);
              /*if (USE_PA) {
                tpp = (get_principal_angle(p, pa, xpp, ypp, spp)-angle0)&(NUM_ANGLES-1);
              }*/
              //xpp -= dx;
              XFORM bpos = get_xform(p, xpp, ypp, spp, tpp);
              xpp -= (bpos.dxdu*dx)>>16;
              ypp -= (bpos.dydu*dx)>>16;
              bpos = get_xform(p, xpp, ypp, spp, tpp);
    #else
              XFORM bpos;
              bpos.x0 = xpp<<16;
              bpos.y0 = ypp<<16;
              int spp = 0, tpp = 0;
    #endif
              knn_attempt_n<PATCH_W, 0>(v, adata, b, bpos, xpp, ypp, spp, tpp, p, 0, pos);
            }
          }
#endif

          unsigned int seed = (x | (y<<11)) ^ iter_seed;

          /* Propagate */
          if (p->do_propagate) {
            /* Propagate x */
            if ((unsigned) (x+dx) < (unsigned) (ann->w-PATCH_W)) {
              int *q_ann = ann->get(x+dx, y);
#if !TRANSLATE_ONLY
              int *q_ann_sim = ann_sim->get(x+dx, y);
#else
              int *q_ann_sim = NULL;
#endif
#if P_BEST_ONLY
              get_best(p, q_ann, annd->get(x+dx, y), q_ann_sim);
#endif
              int istart = 0;
#if P_RAND_ONLY
              seed = RANDI(seed);
              istart = seed%p->knn;
#endif
              //printf("p istart: %d, n: %d\n", istart, istart+(P_BEST_ONLY ? 1: p->knn));
              int n_prop = ((P_BEST_ONLY||P_RAND_ONLY) ? 1: p->knn);
#if P_RAND1TOK
              int *q_annd = annd->get(x+dx, y);
              seed = RANDI(seed);
              n_prop = 1+(seed%p->knn);
              vcopy.clear();
              for (int i = 0; i < p->knn; i++) {
#if !TRANSLATE_ONLY
                vcopy.push_back(qtype<int>(q_annd[i], q_ann[i], q_ann_sim[i]));
#else
                vcopy.push_back(qtype<int>(q_annd[i], q_ann[i]));
#endif
              }
              sort(vcopy.begin(), vcopy.end());
#endif
              for (int i = istart; i < istart+n_prop; i++) {
#if !P_RAND1TOK
                int xpp = INT_TO_X(q_ann[i]), ypp = INT_TO_Y(q_ann[i]);
#else
                int xpp = INT_TO_X(vcopy[i].b), ypp = INT_TO_Y(vcopy[i].b);
#endif
#if !TRANSLATE_ONLY
#if !P_RAND1TOK
                int spp = INT_TO_X(q_ann_sim[i]), tpp = INT_TO_Y(q_ann_sim[i]);
#else
                int spp = INT_TO_X(vcopy[i].c), tpp = INT_TO_Y(vcopy[i].c);
#endif
                /*if (USE_PA) {
                  tpp = (get_principal_angle(p, pa, xpp, ypp, spp)-angle0)&(NUM_ANGLES-1);
                }*/
                //xpp -= dx;
                XFORM bpos = get_xform(p, xpp, ypp, spp, tpp);
                xpp -= (bpos.dxdu*dx)>>16;
                ypp -= (bpos.dydu*dx)>>16;
                bpos = get_xform(p, xpp, ypp, spp, tpp);  // FIXME: Is this xform even right?
#else
                xpp -= dx;
                XFORM bpos;
                bpos.x0 = xpp<<16;
                bpos.y0 = ypp<<16;
                int spp = 0, tpp = 0;
#endif
                //printf("attempt %d\n", i);
                knn_attempt_n<PATCH_W, 0>(v, adata, b, bpos, xpp, ypp, spp, tpp, p, 0, pos);

                //PositionSet pos1(p_ann, &v);
                //printf("elements in queue:\n");
                //for (int ij = 0; ij < p->knn; ij++) {
                //  printf("%d %d\n", INT_TO_X(v[ij].b), INT_TO_Y(v[ij].b));
                //  pos1.insert_nonexistent(INT_TO_X(v[ij].b), INT_TO_Y(v[ij].b), 0);
                //}
                //pos1.check_equals(&pos);
                //
                //if (i == 2) { break; }
              }
            }
            //continue;

            /* Propagate y */
            if ((unsigned) (y+dy) < (unsigned) (ann->h-PATCH_W)) {
              int *q_ann = ann->get(x, y+dy);
#if !TRANSLATE_ONLY
              int *q_ann_sim = ann_sim->get(x, y+dy);
#else
              int *q_ann_sim = NULL;
#endif
#if P_BEST_ONLY
              get_best(p, q_ann, annd->get(x, y+dy), q_ann_sim);
#endif
              int istart = 0;
#if P_RAND_ONLY
              seed = RANDI(seed);
              istart = seed%p->knn;
#endif

              int n_prop = ((P_BEST_ONLY||P_RAND_ONLY) ? 1: p->knn);
#if P_RAND1TOK
              int *q_annd = annd->get(x, y+dy);
              seed = RANDI(seed);
              n_prop = 1+(seed%p->knn);
              vcopy.clear();
              for (int i = 0; i < p->knn; i++) {
#if !TRANSLATE_ONLY
                vcopy.push_back(qtype<int>(q_annd[i], q_ann[i], q_ann_sim[i]));
#else
                vcopy.push_back(qtype<int>(q_annd[i], q_ann[i]));
#endif
              }
              sort(vcopy.begin(), vcopy.end());
#endif

              for (int i = istart; i < istart+n_prop; i++) {
              //printf("p istart: %d, n: %d\n", istart, istart+(P_BEST_ONLY ? 1: p->knn));
#if !P_RAND1TOK
                int xpp = INT_TO_X(q_ann[i]), ypp = INT_TO_Y(q_ann[i]);
#else
                int xpp = INT_TO_X(vcopy[i].b), ypp = INT_TO_Y(vcopy[i].b);
#endif
#if !TRANSLATE_ONLY
#if !P_RAND1TOK
                int spp = INT_TO_X(q_ann_sim[i]), tpp = INT_TO_Y(q_ann_sim[i]);
#else
                int spp = INT_TO_X(vcopy[i].c), tpp = INT_TO_Y(vcopy[i].c);
#endif
                /*if (USE_PA) {
                  tpp = (get_principal_angle(p, pa, xpp, ypp, spp)-angle0)&(NUM_ANGLES-1);
                }*/
                XFORM bpos = get_xform(p, xpp, ypp, spp, tpp);
                xpp -= (bpos.dxdv*dy)>>16;
                ypp -= (bpos.dydv*dy)>>16;
                bpos = get_xform(p, xpp, ypp, spp, tpp);
#else
                ypp -= dy;
                XFORM bpos;
                bpos.x0 = xpp<<16;
                bpos.y0 = ypp<<16;
                int spp = 0, tpp = 0;
#endif
                knn_attempt_n<PATCH_W, 0>(v, adata, b, bpos, xpp, ypp, spp, tpp, p, 0, pos);
              }
            }

#if PROP_DIST2
            /* Propagate x, distance 2 */
            if ((unsigned) (x+dx*2) < (unsigned) (ann->w-PATCH_W)) {
              int *q_ann = ann->get(x+dx*2, y);
#if !TRANSLATE_ONLY
              int *q_ann_sim = ann_sim->get(x+dx*2, y);
#endif
              for (int i = 0; i < (P_BEST_ONLY ? 1: p->knn); i++) {
                int xpp = INT_TO_X(q_ann[i]), ypp = INT_TO_Y(q_ann[i]);
#if !TRANSLATE_ONLY
                int spp = INT_TO_X(q_ann_sim[i]), tpp = INT_TO_Y(q_ann_sim[i]);
                /*if (USE_PA) {
                  tpp = (get_principal_angle(p, pa, xpp, ypp, spp)-angle0)&(NUM_ANGLES-1);
                }*/
                //xpp -= dx;
                XFORM bpos = get_xform(p, xpp, ypp, spp, tpp);
                xpp -= (bpos.dxdu*dx*2)>>16;
                ypp -= (bpos.dydu*dx*2)>>16;
                bpos = get_xform(p, xpp, ypp, spp, tpp);
#else
                XFORM bpos;
                xpp -= dx*2;
                bpos.x0 = xpp<<16;
                bpos.y0 = ypp<<16;
                int spp = 0, tpp = 0;
#endif
                knn_attempt_n<PATCH_W, 0>(v, adata, b, bpos, xpp, ypp, spp, tpp, p, 0, pos);
              }
            }

            /* Propagate y, distance 2 */
            if ((unsigned) (y+dy*2) < (unsigned) (ann->h-PATCH_W)) {
              int *q_ann = ann->get(x, y+dy*2);
#if !TRANSLATE_ONLY
              int *q_ann_sim = ann_sim->get(x, y+dy*2);
#endif
              for (int i = 0; i < (P_BEST_ONLY ? 1: p->knn); i++) {
                int xpp = INT_TO_X(q_ann[i]), ypp = INT_TO_Y(q_ann[i]);
#if !TRANSLATE_ONLY
                int spp = INT_TO_X(q_ann_sim[i]), tpp = INT_TO_Y(q_ann_sim[i]);
                /*if (USE_PA) {
                  tpp = (get_principal_angle(p, pa, xpp, ypp, spp)-angle0)&(NUM_ANGLES-1);
                }*/
                XFORM bpos = get_xform(p, xpp, ypp, spp, tpp);
                xpp -= (bpos.dxdv*dy*2)>>16;
                ypp -= (bpos.dydv*dy*2)>>16;
                bpos = get_xform(p, xpp, ypp, spp, tpp);
#else
                ypp -= dy*2;
                XFORM bpos;
                bpos.x0 = xpp<<16;
                bpos.y0 = ypp<<16;
                int spp = 0, tpp = 0;
#endif
                knn_attempt_n<PATCH_W, 0>(v, adata, b, bpos, xpp, ypp, spp, tpp, p, 0, pos);
              }
            }
#endif
          }

//#if 0
          /* Random search */
          seed = RANDI(seed);
          int rs_iters = 1-((seed&65535)*(1.0/(65536-1))) < rs_fpart ? rs_ipart + 1: rs_ipart;
  //        int rs_iters = 1-random() < rs_fpart ? rs_ipart + 1: rs_ipart;
          //fprintf(stderr, "%d rs iters\n", rs_iters);

          int rs_max_curr = rs_max;

/*
  #if !TRANSLATE_ONLY
          v.push_back(qtype<int>(0, 0, 0)); // Bug in make_heap()?  Requires one extra element
  #else
          v.push_back(qtype<int>(0, 0)); // Bug in make_heap()?  Requires one extra element
  #endif
          //fprintf(stderr, "before make_heap %d\n", v.size()); fflush(stderr);
          sort_heap(&v[0], &v[p->knn]);
          v.pop_back();
*/

          int h = p->patch_w/2;
          int ymin_clamp = h, xmin_clamp = h;
          int ymax_clamp = BEH+h, xmax_clamp = BEW+h;

          int nchosen = (RS_BEST_ONLY||RS_RAND_ONLY) ? 1: p->knn;

#if RS_RAND1TOK
          seed = RANDI(seed);
          nchosen = 1 + (seed % p->knn);
          vcopy = v;
          sort(vcopy.begin(), vcopy.end());
#endif

          for (int mag = rs_max_curr; mag >= p->rs_min; mag = int(mag*p->rs_ratio)) {
#if !TRANSLATE_ONLY
            int smag = NUM_SCALES*mag/rs_max_curr;
            int tmag = (NUM_SCALES == NUM_ANGLES) ? smag: (NUM_ANGLES*mag/rs_max_curr);
            tmag *= 6;
            smag *= 6;
#endif
            //int pfrac = int(65536.0/p->knn+0.5);
            for (int rs_iter = 0; rs_iter < rs_iters; rs_iter++) {
              int nstart = 0;
#if RS_RAND_ONLY
              seed = RANDI(seed);
              nstart = seed%p->knn;
#endif
#if RS_BEST_ONLY
              int *p_ann_copy = p_ann, *p_annd_copy = p_annd;
#if !TRANSLATE_ONLY
              int *p_ann_sim_copy = p_ann_sim;
#else
              int *p_ann_sim_copy = NULL;
#endif
              get_best(p, p_ann_copy, p_annd_copy, p_ann_sim_copy);
              nstart = p_ann_copy - p_ann;
#endif
              //printf("rs nstart: %d, n: %d\n", nstart, nstart+nchosen);
              for (int i = nstart; i < nstart+nchosen; i++) {
                //seed = RANDI(seed);
                //if ((seed&65535) > pfrac) { continue; }
                int xbest = INT_TO_X(VARRAY[i].b), ybest = INT_TO_Y(VARRAY[i].b);
                int xmin = xbest-mag, xmax = xbest+mag;
                int ymin = ybest-mag, ymax = ybest+mag;
                xmax++;
                ymax++;
                /*if (xmin < xmin_clamp) { xmin = xmin_clamp; }
                if (ymin < ymin_clamp) { ymin = ymin_clamp; }
                if (xmax > xmax_clamp) { xmax = xmax_clamp; }
                if (ymax > ymax_clamp) { ymax = ymax_clamp; }*/
                if (xmin < 0) { xmin = 0; }
                if (ymin < 0) { ymin = 0; }
                if (xmax > bew) { xmax = bew; }
                if (ymax > beh) { ymax = beh; }

#if !TRANSLATE_ONLY
                int sbest = INT_TO_X(VARRAY[i].c), tbest = INT_TO_Y(VARRAY[i].c);
                int smin = sbest-smag, smax = sbest+smag+1;
                int tmin = tbest-tmag, tmax = tbest+tmag+1;
                if (smin < 0) { smin = 0; }
                if (smax > NUM_SCALES) { smax = NUM_SCALES; }
#endif
                //fprintf(stderr, "RS: xbest: %d, ybest: %d, err: %d, mag: %d, bew: %d, beh: %d, smag: %d, tmag: %d, xmin: %d, xmax: %d, ymin: %d, ymax: %d, smin: %d, smax: %d, tmin: %d, tmax: %d\n", xbest, ybest, err, mag, bew, beh, smag, tmag, xmin, xmax, ymin, ymax, smin, smax, tmin, tmax); fflush(stderr);

                seed = RANDI(seed);
                int xpp = xmin+seed%(xmax-xmin);
                seed = RANDI(seed);
                int ypp = ymin+seed%(ymax-ymin);
#if !TRANSLATE_ONLY
                seed = RANDI(seed);
                int spp = p->allow_scale ? (smin+seed%(smax-smin)): SCALE_UNITY;
                seed = RANDI(seed);
                int tpp = p->allow_rotate ? (tmin+seed%(tmax-tmin)): 0;
                seed = RANDI(seed);
                if (USE_PA && (seed&65535) < int(65536*p->prob_prinangle)) {
                  tpp = (get_principal_angle(p, pa, xpp, ypp, spp)-angle0)&(NUM_ANGLES-1);
                }
                XFORM bpos = get_xform(p, xpp, ypp, spp, tpp); // FIXME: Is this xform even right?
#else
                XFORM bpos;
                bpos.x0 = xpp<<16;
                bpos.y0 = ypp<<16;
                int spp = 0, tpp = 0;
#endif
                //knn_attempt_n<PATCH_W>(err, xbest, ybest, sbest, tbest, adata, b, bpos, xpp, ypp, spp, tpp, p);
                knn_attempt_n<PATCH_W, 0>(v, adata, b, bpos, xpp, ypp, spp, tpp, p, 0, pos);

                /*
                for (int ii = 0; ii < p->knn; ii++) {
                  p_annd[ii] = v[ii].a;
                  p_ann[ii] = v[ii].b;
      #if !TRANSLATE_ONLY
                  p_ann_sim[ii] = v[ii].c;
      #endif
                }
                //fprintf(stderr, "second check\n");
                pos.check(p->knn, 4);*/


                //check_offset(p, b, x, y, xbest, ybest);
                //attempt_n<PATCH_W, SPARSE, IS_MASK, IS_WINDOW>(err, xbest, ybest, adata, b, xpp, ypp, bmask, region_masks, src_mask, p);
              }
            }
          }
//#endif

          /*((int *) ann->line[y])[x] = XY_TO_INT(xbest, ybest);
          ((int *) ann_sim->line[y])[x] = XY_TO_INT(sbest, tbest);
          ((int *) annd->line[y])[x] = err;*/
#if SYNC_WRITEBACK
          if (y+ychange != yfinal) {     
#endif
          for (int i = 0; i < p->knn; i++) {
            p_annd[i] = v[i].a;
            p_ann[i] = v[i].b;
#if !TRANSLATE_ONLY
            p_ann_sim[i] = v[i].c;
#endif
          }
#if SYNC_WRITEBACK
          } else {
            for (int kval = 0; kval < p->knn; kval++) {
              ann_writeback[x*p->knn+kval] = v[kval].b; //XY_TO_INT(xbest, ybest);
              annd_writeback[x*p->knn+kval] = v[kval].a; //err;
            }
          }
#endif
          //fprintf(stderr, "second check\n");
          //pos.check(p->knn, 1);
        } // x
      } // y

#if SYNC_WRITEBACK
      #pragma omp barrier
      int ywrite = yfinal-ychange;
      if (ymin < ymax && (unsigned) ywrite < (unsigned) AEH) {
        //int *ann_line = (int *) ann->line[ywrite];
        //int *annd_line = (int *) annd->line[ywrite];
        for (int x = xmin; x < xmax; x++) {
          int *r_ann = ann->get(x, ywrite);
          int *r_annd = annd->get(x, ywrite);
          for (int kval = 0; kval < p->knn; kval++) {
            r_ann[kval] = ann_writeback[x*p->knn+kval];
            r_annd[kval] = annd_writeback[x*p->knn+kval];
          }
        }
      }
      delete[] ann_writeback;
      delete[] annd_writeback;
#endif
    } // parallel
    fprintf(stderr, "done with %d iters\n", nn_iter);
#if SAVE_DIST
    if (nn_iter == p->nn_iters-1) {
      save_dist(p, annd, "nn");
    }
#endif
  } // nn_iter
  printf("done knn_n, %d iters, rs_max=%d\n", nn_iter, p->rs_max);
}

template<int PATCH_W>
void knn_n_top1nn(Params *p, BITMAP *a, BITMAP *b,
            VBMP *ann0, VBMP *ann_sim0, VBMP *annd0,
            RegionMasks *amask=NULL, BITMAP *bmask=NULL,
            int level=0, int em_iter=0, RecomposeParams *rp=NULL, int offset_iter=0, int update_type=0, int cache_b=0,
            RegionMasks *region_masks=NULL, int tiles=-1, PRINCIPAL_ANGLE *pa=NULL, int save_first=1) {
  if (ann_sim0) { fprintf(stderr, "Unimplemented for rotation+scale\n"); exit(1); }
  BITMAP *ann = create_bitmap(ann0->w, ann0->h);
  BITMAP *annd = create_bitmap(annd0->w, annd0->h);
  for (int y = 0; y < ann->h; y++) {
    int *ann_row = (int *) ann->line[y];
    int *annd_row = (int *) annd->line[y];
    for (int x = 0; x < ann->w; x++) {
      int *p_ann = ann0->get(x, y);
      int *p_annd = annd0->get(x, y);
      ann_row[x] = *p_ann;
      annd_row[x] = *p_annd;
    }
  }

  int max_mag = MAX(b->w, b->h);
  int rs_max = p->rs_max;
  if (rs_max > max_mag) { rs_max = max_mag; }
  int nper = 2;
  int rs_max_curr = rs_max;
  for (int mag = rs_max_curr; mag >= p->rs_min; mag = int(mag*p->rs_ratio)) {
    nper++;
  }
  int nper0 = nper;
  nper *= p->nn_iters;
  if (p->knn > nper) { fprintf(stderr, "p->knn (%d) > nper=(samples/iter)*(iters) (%d)\n", p->knn, nper); exit(1); }

  long long nsize = sizeof(knn_pair<int>)*ann->w*ann->h*nper;
  if (nsize > (long long) INT_MAX) { fprintf(stderr, "nsize (%ll) > INT_MAX (%d)\n", nsize, INT_MAX); exit(1); }
  knn_pair<int> *tries = (knn_pair<int> *) malloc(int(nsize));
//  printf("malloc %d bytes, %d size of knn_pair, %p pointer\n", int(nsize), sizeof(knn_pair<int>), tries);
  if (!tries) { fprintf(stderr, "malloc failed\n"); exit(1); }

  if (tiles < 0) { tiles = p->cores; }
  printf("in knn_n_top1nn, tiles=%d, nper=%d, nper0=%d\n", tiles, nper, nper0);
  Box box = get_abox(p, a, amask);
//  poll_server();
  int nn_iter = 0;
  for (; nn_iter < p->nn_iters; nn_iter++) {
    unsigned int iter_seed = rand();
//    if (p->update) { p->update(level, em_iter, nn_iter, p, rp, a, b, ann, annd, NULL, NULL, bmask, NULL, region_masks, amask, update_type, p->q); }

    #pragma omp parallel num_threads(tiles)
    {
#if USE_OPENMP
      int ithread = omp_get_thread_num();
#else
      int ithread = 0;
#endif
      int xmin = box.xmin, xmax = box.xmax;
      int ymin = box.ymin + (box.ymax-box.ymin)*ithread/tiles;
      int ymax = box.ymin + (box.ymax-box.ymin)*(ithread+1)/tiles;

      int ystart = ymin, yfinal = ymax, ychange=1;
      int xstart = xmin, xfinal = xmax, xchange=1;
      if ((nn_iter + offset_iter) % 2 == 1) {
        xstart = xmax-1; xfinal = xmin-1; xchange=-1;
        ystart = ymax-1; yfinal = ymin-1; ychange=-1;
      }
      int dx = -xchange, dy = -ychange;

      int bew = b->w-PATCH_W, beh = b->h-PATCH_W;
      int rs_ipart = int(p->rs_iters);
      double rs_fpart = p->rs_iters - rs_ipart;

      ALLOC_ADATA
      for (int y = ystart; y != yfinal; y += ychange) {
        int *annd_row = (int *) annd->line[y];
        for (int x = xstart; x != xfinal; x += xchange) {
          knn_pair<int> *p_tries = &tries[(y*ann->w+x)*nper+nn_iter*nper0];
          knn_pair<int> *p_tries0 = p_tries;
          for (int dy0 = 0; dy0 < PATCH_W; dy0++) {
            int *drow = ((int *) a->line[y+dy0])+x;
            int *adata_row = adata+(dy0*PATCH_W);
            for (int dx0 = 0; dx0 < PATCH_W; dx0++) {
              adata_row[dx0] = drow[dx0];
            }
          }
          
          int xbest, ybest;
          getnn(ann, x, y, xbest, ybest);
          int err = annd_row[x];
//          if (err == 0) { continue; }

          /* Propagate */
          int did_Px = 0, did_Py = 0;
          if (p->do_propagate) {
            /* Propagate x */
            if ((unsigned) (x+dx) < (unsigned) (ann->w-PATCH_W)) {
              int xpp, ypp;
              getnn(ann, x+dx, y, xpp, ypp);
              xpp -= dx;

              if ((xpp != xbest || ypp != ybest) &&
                  (unsigned) xpp < (unsigned) (b->w-PATCH_W+1)) {
                did_Px = 1;
                int err0 = ((int *) annd->line[y])[x+dx];

                int xa = dx, xb = 0;
                if (dx > 0) { xa = 0; xb = dx; }
                int partial = 0;
                for (int yi = 0; yi < PATCH_W; yi++) {
                  int c1 = ((int *) a->line[y+yi])[x+xa];
                  int c2 = ((int *) b->line[ypp+yi])[xpp+xa];
                  int c3 = ((int *) a->line[y+yi])[x+xb+PATCH_W-1];
                  int c4 = ((int *) b->line[ypp+yi])[xpp+xb+PATCH_W-1];
                  int dr12 = (c1&255)-(c2&255);
                  int dg12 = ((c1>>8)&255)-((c2>>8)&255);
                  int db12 = (c1>>16)-(c2>>16);
                  int dr34 = (c3&255)-(c4&255);
                  int dg34 = ((c3>>8)&255)-((c4>>8)&255);
                  int db34 = (c3>>16)-(c4>>16);
                  partial += DELTA_TERM_RGB(dr34, dg34, db34) -
                             DELTA_TERM_RGB(dr12, dg12, db12);
                             // dr34*dr34+dg34*dg34+db34*db34
                             //-dr12*dr12-dg12*dg12-db12*db12;
                }
                err0 += (dx < 0) ? partial: -partial;
                p_tries[0].a = err0;
                p_tries[0].b = XY_TO_INT(xpp, ypp);
                if (err0 < err) {
                  err = err0;
                  xbest = xpp;
                  ybest = ypp;
                }
              }
            }

            /* Propagate y */
            if ((unsigned) (y+dy) < (unsigned) (ann->h-PATCH_W)) {
              int xpp, ypp;
              getnn(ann, x, y+dy, xpp, ypp);
              ypp -= dy;

              if ((xpp != xbest || ypp != ybest) &&
                  (unsigned) ypp < (unsigned) (b->h-PATCH_W+1)) {
                did_Py = 1;
                int err0 = ((int *) annd->line[y+dy])[x];

                int ya = dy, yb = 0;
                if (dy > 0) { ya = 0; yb = dy; }
                int partial = 0;
                int *c1row = &((int *) a->line[y+ya])[x];
                int *c2row = &((int *) b->line[ypp+ya])[xpp];
                int *c3row = &((int *) a->line[y+yb+PATCH_W-1])[x];
                int *c4row = &((int *) b->line[ypp+yb+PATCH_W-1])[xpp];
                for (int xi = 0; xi < PATCH_W; xi++) {
                  int c1 = c1row[xi];
                  int c2 = c2row[xi];
                  int c3 = c3row[xi];
                  int c4 = c4row[xi];
                  int dr12 = (c1&255)-(c2&255);
                  int dg12 = ((c1>>8)&255)-((c2>>8)&255);
                  int db12 = (c1>>16)-(c2>>16);
                  int dr34 = (c3&255)-(c4&255);
                  int dg34 = ((c3>>8)&255)-((c4>>8)&255);
                  int db34 = (c3>>16)-(c4>>16);
                  partial += DELTA_TERM_RGB(dr34, dg34, db34) -
                             DELTA_TERM_RGB(dr12, dg12, db12);
                             // dr34*dr34+dg34*dg34+db34*db34
                             //-dr12*dr12-dg12*dg12-db12*db12;
                }
                err0 += (dy < 0) ? partial: -partial;
                p_tries[1].a = err0;
                p_tries[1].b = XY_TO_INT(xpp, ypp);
                if (err0 < err) {
                  err = err0;
                  xbest = xpp;
                  ybest = ypp;
                }
              }
            }
          }

          if (!did_Px) {
            p_tries[0].a = INT_MAX;
            p_tries[0].b = 0;
          }
          if (!did_Py) {
            p_tries[1].a = INT_MAX;
            p_tries[1].b = 0;
          }

          p_tries += 2;

          /* Random search */
          unsigned int seed = (x | (y<<11)) ^ iter_seed;
          seed = RANDI(seed);
          //int rs_iters = 1-(seed*(1.0/(RAND_MAX-1))) < rs_fpart ? rs_ipart + 1: rs_ipart;
          int rs_iters = 1; //1-((seed&65535)*(1.0/(65536-1))) < rs_fpart ? rs_ipart + 1: rs_ipart;
  //        int rs_iters = 1-random() < rs_fpart ? rs_ipart + 1: rs_ipart;

          for (int mag = rs_max_curr; mag >= p->rs_min; mag = int(mag*p->rs_ratio)) {
            for (int rs_iter = 0; rs_iter < rs_iters; rs_iter++) {
              int xmin = xbest-mag, xmax = xbest+mag;
              int ymin = ybest-mag, ymax = ybest+mag;
              xmax++;
              ymax++;
              if (xmin < 0) { xmin = 0; }
              if (ymin < 0) { ymin = 0; }
              if (xmax > bew) { xmax = bew; }
              if (ymax > beh) { ymax = beh; }

              seed = RANDI(seed);
              int xpp = xmin+seed%(xmax-xmin);
              seed = RANDI(seed);
              int ypp = ymin+seed%(ymax-ymin);
              //int xpp = xmin+rand()%(xmax-xmin);
              //int ypp = ymin+rand()%(ymax-ymin);
              if ((unsigned) xpp < (unsigned) (b->w-PATCH_W+1) &&
                  (unsigned) ypp < (unsigned) (b->h-PATCH_W+1)) {
                int current = fast_patch_nobranch<PATCH_W, 0>(adata, b, xpp, ypp, p);
                p_tries->a = current;
                p_tries->b = XY_TO_INT(xpp, ypp);
                if (current < err) {
                  err = current;
                  xbest = xpp;
                  ybest = ypp;
                }
              } else {
                p_tries->a = INT_MAX;
                p_tries->b = 0;
              }
              p_tries++;
            }
          }
          
          int ntry = p_tries - p_tries0;
          if (ntry != nper0) { fprintf(stderr, "ntry (%d) != nper0 (%d)\n", ntry, nper0); exit(1); }

          ((int *) ann->line[y])[x] = XY_TO_INT(xbest, ybest);
          ((int *) annd->line[y])[x] = err;
        } // x
      } // y
    } // parallel
  } // nn_iter
  printf("done with %d iters, nper=%d\n", nn_iter, nper);

  #pragma omp parallel for schedule(static,8)
  for (int y = 0; y < AEH; y++) {
    //vector<knn_pair<int> > v;
    //v.reserve(p->knn+1);
    for (int x = 0; x < AEW; x++) {
      //printf("%d, %d (nper=%d, tries=%p)\n", x, y, nper, tries);
      knn_pair<int> *tp = &tries[(y*ann->w+x)*nper];
      //for (int i = 0; i < nper; i++) { tp[i].a = rand(); tp[i].b = rand(); }
      //partial_sort(tp, tp+p->knn, tp+nper);
      // FIXME: Could do a partial_sort() of an adaptive number of elements, if unique() gives too few then double the number
      sort(tp, tp+nper);
      int nunique = unique(tp, tp+nper)-tp;
      int *p_ann = ann0->get(x, y);
      int *p_annd = annd0->get(x, y);
      if (nunique < p->knn) { //fprintf(stderr, "nunique (%d) < p->knn (%d) at %d,%d\n", nunique, p->knn, x, y); exit(1); }
        vector<knn_pair<int> > L;
        for (int i = 0; i < nper; i++) {
          L.push_back(tp[i]);
        }
        for (int i = 0; i < p->knn; i++) {
          L.push_back(knn_pair<int>(p_annd[i], p_ann[i]));
        }
        sort(L.begin(), L.end());
        int nunique = unique(L.begin(), L.end())-L.begin();
        if (nunique < p->knn || nunique > (int) L.size()) { fprintf(stderr, "nunique value not correct: %d < p->knn or above L.size()=%d\n", nunique, p->knn, L.size()); }

        for (int i = 0; i < p->knn; i++) {
          p_ann[i] = L[p->knn-1-i].b;
          p_annd[i] = L[p->knn-1-i].a;
        }
      } else {
        for (int i = 0; i < p->knn; i++) {
          p_ann[i] = tp[p->knn-1-i].b;
          p_annd[i] = tp[p->knn-1-i].a;
        }
      }
      /*v.clear();
      for (int i = 0; i < p->knn; i++) {
        //p_ann[i] = tp[i].b;
        //p_annd[i] = tp[i].a;
        v.push_back(knn_pair<int>(tp[i].a, tp[i].b));
      }
      //FIXME: Could really transform list into heap in place, without doing any comparisons

      v.push_back(qtype<int>(0, 0)); // Bug in make_heap()?  Requires one extra element
      make_heap(&v[0], &v[p->knn]);
      v.pop_back();
      if (v.size() != p->knn) { fprintf(stderr, "v size != knn (%d, %d)\n", v.size(), p->knn); exit(1); }

      for (int i = 0; i < p->knn; i++) {
        qtype<int> current = v[i];
        p_annd[i] = current.a;
        p_ann[i] = current.b;
      }*/
    }
  }
  
  destroy_bitmap(ann);
  destroy_bitmap(annd);
  free((void *) tries);
  printf("done knn_n_cputiled, %d iters, rs_max=%d\n", nn_iter, p->rs_max);
}

/*
void sort_knn(Params *p, BITMAP *a, VBMP *ann, VBMP *ann_sim, VBMP *annd) {
  knn_pair<int> *L = (knn_pair<int> *) malloc(sizeof(knn_pair<int>)*p->knn);
  for (int y = 0; y < ann->h; y++) {
    for (int x = 0; x < ann->w; x++) {
      int *p_ann = ann->get(x, y);
      int *p_annd = annd->get(x, y);
      for (int i = 0; i < p->knn; i++) {
        L[i].a = p_annd[i];
        L[i].b = p_ann[i];
      }
      sort(L, L+p->knn); // FIXME: Could use partial sort
      for (int i = 0; i < p->knn; i++) {
        p_ann[i] = L[i].b; //L[(p->knn-1-i)].b;
        p_annd[i] = L[i].a; //L[(p->knn-1-i)].a;    // If n < p->knn then elements are repeated
      }
    }
  }
  delete[] L;
}
*/

template<int PATCH_W>
void knn_n_window(Params *p, BITMAP *a, BITMAP *b,
            VBMP *ann, VBMP *ann_sim, VBMP *annd,
            RegionMasks *amask=NULL, BITMAP *bmask=NULL,
            int level=0, int em_iter=0, RecomposeParams *rp=NULL, int offset_iter=0, int update_type=0, int cache_b=0,
            RegionMasks *region_masks=NULL, int tiles=-1, PRINCIPAL_ANGLE *pa=NULL, int save_first=1) {
  printf("in knn_n_window, window_w=%d, window_h=%d\n", p->window_w, p->window_h);
#if !TRANSLATE_ONLY
  printf("Warning: using knn_n_window with rotation+scale enabled.  Was that what you wanted?\n");
#endif
  int w = p->window_w, h = p->window_h;
  #pragma omp parallel for schedule(static, 8)
  for (int y = 0; y < AEH; y++) {
    knn_pair<int> *L = (knn_pair<int> *) malloc(sizeof(knn_pair<int>)*w*h);
    ALLOC_ADATA
    for (int x = 0; x < AEW; x++) {
      for (int dy0 = 0; dy0 < PATCH_W; dy0++) {
        int *drow = ((int *) a->line[y+dy0])+x;
        int *adata_row = adata+(dy0*PATCH_W);
        for (int dx0 = 0; dx0 < PATCH_W; dx0++) {
          adata_row[dx0] = drow[dx0];
        }
      }
      int n = 0;
      int xchange = 0, ychange = 0;
      if (y-h/2 < 0) { ychange = abs(y-h/2); }
      else if (y-h/2+h >= AEH) { ychange = -(y-h/2+h-(AEH-1)); }
      if (x-w/2 < 0) { xchange = abs(x-w/2); }
      else if (x-w/2+w >= AEW) { xchange = -(x-w/2+w-(AEW-1)); }

      for (int yi = 0; yi < h; yi++) {
        int dy = yi-h/2;
        int yp = y+dy+ychange;
        if ((unsigned) yp >= (unsigned) AEH) { continue; }
        for (int xi = 0; xi < w; xi++) {
          int dx = xi-w/2;
          int xp = x+dx+xchange;
          if ((unsigned) xp >= (unsigned) AEW) { continue; }
          int d = fast_patch_nobranch<PATCH_W, 0>(adata, a, xp, yp, p);
          L[n].a = d;
          L[n].b = XY_TO_INT(xp, yp);
          n++;
        }
      }
      sort(L, L+n); // FIXME: Could use partial sort
      int *p_ann = ann->get(x, y);
      int *p_annd = annd->get(x, y);
      if (n < p->knn) { fprintf(stderr, "n (%d) < p->knn (%d) at %d, %d\n", n, p->knn, x, y); exit(1); }
      if (n > p->knn) { fprintf(stderr, "n (%d) > p->knn (%d) at %d, %d\n", n, p->knn, x, y); exit(1); }
      for (int i = 0; i < p->knn; i++) {
        p_ann[i] = L[(p->knn-1-i)].b;
        p_annd[i] = L[(p->knn-1-i)].a;    // If n < p->knn then elements are repeated
#if !TRANSLATE_ONLY
        ann_sim->get(x, y)[i] = XY_TO_INT(SCALE_UNITY, 0);
#endif
      }
    }
    free((void *) L);
  }
  printf("done knn_n_window, window_w=%d, window_h=%d\n", p->window_w, p->window_h);
}

template<int PATCH_W>
void avoid_attempt_n(int &err, int &xbest, int &ybest, int &sbest, int &tbest, int *adata, BITMAP *b, XFORM bpos, int bx, int by, int bs, int bt, Params *p, PositionSet *avoid, int nn_i) {
  if (!avoid->contains_noqueue(bx, by, nn_i)) {
#if !TRANSLATE_ONLY
    sim_attempt_n<PATCH_W>(err, xbest, ybest, sbest, tbest, adata, b, bpos, bx, by, bs, bt, p);
#else
    attempt_n<PATCH_W, 0, 0>(err, xbest, ybest, adata, b, bx, by, NULL, NULL, 0, p);
#endif
  }
}

template<int PATCH_W>
void knn_n_avoid(Params *p, BITMAP *a, BITMAP *b,
            VBMP *ann, VBMP *ann_sim, VBMP *annd,
            RegionMasks *amask=NULL, BITMAP *bmask=NULL,
            int level=0, int em_iter=0, RecomposeParams *rp=NULL, int offset_iter=0, int update_type=0, int cache_b=0,
            RegionMasks *region_masks=NULL, int tiles=-1) {
  init_xform_tables();
  if (tiles < 0) { tiles = p->cores; }
#if TRANSLATE_ONLY
  if (ann_sim) { fprintf(stderr, "in mode TRANSLATE_ONLY, expected ann_sim to be NULL\n"); exit(1); }
#endif

  printf("in knn_n_avoid, tiles=%d, rs_max=%d\n", tiles, p->rs_max);
  Box box = get_abox(p, a, amask);
//  poll_server();
  PositionSet *avoid = new PositionSet[a->w*a->h];
  for (int y = 0; y < a->h; y++) {
    for (int x = 0; x < a->w; x++) {
      int *p_ann = ann->get(x, y);
      avoid[y*a->w+x].init(p_ann, NULL, p->knn);
    }
  }
  // FIXME: Use PositionSet instead of hash_set<int>
  // FIXME: Remove error checks after debugging
  
  for (int nn_i = 0; nn_i < p->knn; nn_i++) {
  for (int nn_iter = 0; nn_iter < p->nn_iters; nn_iter++) {
    //fprintf(stderr, "begin nn_i=%d, nn_iter=%d\n", nn_i, nn_iter);
    unsigned int iter_seed = rand();
//    if (p->update) { p->update(level, em_iter, nn_iter, p, rp, a, b, NULL, NULL, NULL, NULL, bmask, NULL, region_masks, amask, update_type, p->q); }

    #pragma omp parallel num_threads(tiles)
    {
#if USE_OPENMP
      int ithread = omp_get_thread_num();
#else
      int ithread = 0;
#endif
      int xmin = box.xmin, xmax = box.xmax;
      int ymin = box.ymin + (box.ymax-box.ymin)*ithread/tiles;
      int ymax = box.ymin + (box.ymax-box.ymin)*(ithread+1)/tiles;

      int ystart = ymin, yfinal = ymax, ychange=1;
      int xstart = xmin, xfinal = xmax, xchange=1;
      if ((nn_iter + offset_iter) % 2 == 1) {
        xstart = xmax-1; xfinal = xmin-1; xchange=-1;
        ystart = ymax-1; yfinal = ymin-1; ychange=-1;
      }
      int dx = -xchange, dy = -ychange;

      int bew = b->w-PATCH_W, beh = b->h-PATCH_W;
      int max_mag = MAX(b->w, b->h);
      int rs_ipart = int(p->rs_iters);
      double rs_fpart = p->rs_iters - rs_ipart;
      int rs_max = p->rs_max;
      if (rs_max > max_mag) { rs_max = max_mag; }

      //vector<qtype<int> > v;
      //v.reserve(p->knn);
      int adata[PATCH_W*PATCH_W];
      for (int y = ystart; y != yfinal; y += ychange) {
        //int *annd_row = (int *) annd->line[y];
        for (int x = xstart; x != xfinal; x += xchange) {
          unsigned int seed = (x | (y<<11)) ^ iter_seed;

          PositionSet *avoid_current = &avoid[y*a->w+x];
          for (int dy0 = 0; dy0 < PATCH_W; dy0++) {
            int *drow = ((int *) a->line[y+dy0])+x;
            int *adata_row = adata+(dy0*PATCH_W);
            for (int dx0 = 0; dx0 < PATCH_W; dx0++) {
              adata_row[dx0] = drow[dx0];
            }
          }
          
          //int xbest, ybest, sbest, tbest;
          //getnn(ann, x, y, xbest, ybest);
          //getnn(ann_sim, x, y, sbest, tbest);
          int ann_v = ann->get(x, y)[nn_i];
          int xbest = INT_TO_X(ann_v), ybest = INT_TO_Y(ann_v);
#if !TRANSLATE_ONLY
          int ann_sim_v = ann_sim->get(x, y)[nn_i];
          int sbest = INT_TO_X(ann_sim_v), tbest = INT_TO_Y(ann_sim_v);
#else
          int sbest = SCALE_UNITY, tbest = 0;
#endif
//          check_offset(p, b, x, y, xbest, ybest);
          int err = annd->get(x, y)[nn_i]; //annd_row[x];
          while (avoid_current->contains_noqueue(xbest, ybest, nn_i) != 0) {
            seed = RANDI(seed);
            xbest = seed%BEW;
            seed = RANDI(seed);
            ybest = seed%BEH;
            if (avoid_current->contains_noqueue(xbest, ybest, nn_i) == 0) {
#if !TRANSLATE_ONLY
              err = sim_patch_dist_ab<PATCH_W, 0>(p, a, x, y, b, xbest, ybest, sbest, tbest, INT_MAX);
#else
              err = patch_dist_ab<PATCH_W, 0, 0>(p, a, x, y, b, xbest, ybest, INT_MAX, NULL);
#endif
              break;
            }
          }

          //if (avoid_current->count(XY_TO_INT(xbest, ybest)) != 0) { fprintf(stderr, "xbest, ybest already in hash map (originally)\n"); exit(1); }

          /* Propagate */
          if (p->do_propagate) {
            /* Propagate x */
            if ((unsigned) (x+dx) < (unsigned) (ann->w-PATCH_W)) {
              ann_v = ann->get(x+dx, y)[nn_i];
              int xpp = INT_TO_X(ann_v), ypp = INT_TO_Y(ann_v);
#if !TRANSLATE_ONLY
              ann_sim_v = ann_sim->get(x+dx, y)[nn_i];
              int spp = INT_TO_X(ann_sim_v), tpp = INT_TO_Y(ann_sim_v);
              XFORM bpos = get_xform(p, xpp, ypp, spp, tpp);
              xpp -= (bpos.dxdu*dx)>>16;
              ypp -= (bpos.dydu*dx)>>16;
              bpos = get_xform(p, xpp, ypp, spp, tpp);
#else
              XFORM bpos;
              xpp -= dx;
              int spp = SCALE_UNITY, tpp = 0;
#endif
              avoid_attempt_n<PATCH_W>(err, xbest, ybest, sbest, tbest, adata, b, bpos, xpp, ypp, spp, tpp, p, avoid_current, nn_i);
            }

            /* Propagate y */
            if ((unsigned) (y+dy) < (unsigned) (ann->h-PATCH_W)) {
              ann_v = ann->get(x, y+dy)[nn_i];
              int xpp = INT_TO_X(ann_v), ypp = INT_TO_Y(ann_v);
#if !TRANSLATE_ONLY
              ann_sim_v = ann_sim->get(x, y+dy)[nn_i];
              int spp = INT_TO_X(ann_sim_v), tpp = INT_TO_Y(ann_sim_v);
              XFORM bpos = get_xform(p, xpp, ypp, spp, tpp);
              xpp -= (bpos.dxdv*dy)>>16;
              ypp -= (bpos.dydv*dy)>>16;
              bpos = get_xform(p, xpp, ypp, spp, tpp);
#else
              XFORM bpos;
              ypp -= dy;
              int spp = SCALE_UNITY, tpp = 0;
#endif
              avoid_attempt_n<PATCH_W>(err, xbest, ybest, sbest, tbest, adata, b, bpos, xpp, ypp, spp, tpp, p, avoid_current, nn_i);
            }
          }

          //if (avoid_current->count(XY_TO_INT(xbest, ybest)) != 0) { fprintf(stderr, "xbest, ybest already in hash map (after propagate)\n"); exit(1); }

          /* Random search */
          seed = RANDI(seed);
          int rs_iters = 1-((seed&65535)*(1.0/(65536-1))) < rs_fpart ? rs_ipart + 1: rs_ipart;
  //        int rs_iters = 1-random() < rs_fpart ? rs_ipart + 1: rs_ipart;
          //fprintf(stderr, "%d rs iters\n", rs_iters);

          int rs_max_curr = rs_max;

          int h = p->patch_w/2;
          int ymin_clamp = h, xmin_clamp = h;
          int ymax_clamp = BEH+h, xmax_clamp = BEW+h;

          for (int mag = rs_max_curr; mag >= p->rs_min; mag = int(mag*p->rs_ratio)) {
            int smag = NUM_SCALES*mag/rs_max_curr;
            int tmag = (NUM_SCALES == NUM_ANGLES) ? smag: (NUM_ANGLES*mag/rs_max_curr);  // FIXME: This should be divided by 2
            for (int rs_iter = 0; rs_iter < rs_iters; rs_iter++) {
//              int xbest = INT_TO_X(v[i].b), ybest = INT_TO_Y(v[i].b);
              int xmin = xbest-mag, xmax = xbest+mag;
              int ymin = ybest-mag, ymax = ybest+mag;
              xmax++;
              ymax++;
              /*if (xmin < xmin_clamp) { xmin = xmin_clamp; }
              if (ymin < ymin_clamp) { ymin = ymin_clamp; }
              if (xmax > xmax_clamp) { xmax = xmax_clamp; }
              if (ymax > ymax_clamp) { ymax = ymax_clamp; }*/
              if (xmin < 0) { xmin = 0; }
              if (ymin < 0) { ymin = 0; }
              if (xmax > bew) { xmax = bew; }
              if (ymax > beh) { ymax = beh; }

              seed = RANDI(seed);
              int xpp = xmin+seed%(xmax-xmin);
              seed = RANDI(seed);
              int ypp = ymin+seed%(ymax-ymin);
#if !TRANSLATE_ONLY
              //int sbest = INT_TO_X(v[i].c), tbest = INT_TO_Y(v[i].c);
              int smin = sbest-smag, smax = sbest+smag+1;
              int tmin = tbest-tmag, tmax = tbest+tmag+1;
              if (smin < 0) { smin = 0; }
              if (smax > NUM_SCALES) { smax = NUM_SCALES; }

              seed = RANDI(seed);
              int spp = p->allow_scale ? (smin+seed%(smax-smin)): SCALE_UNITY;
              seed = RANDI(seed);
              int tpp = p->allow_rotate ? (tmin+seed%(tmax-tmin)): 0;
              XFORM bpos = get_xform(p, xpp, ypp, spp, tpp);
#else
              XFORM bpos;
              int spp = SCALE_UNITY, tpp = 0;
#endif
              avoid_attempt_n<PATCH_W>(err, xbest, ybest, sbest, tbest, adata, b, bpos, xpp, ypp, spp, tpp, p, avoid_current, nn_i);
              //knn_attempt_n<PATCH_W>(v, adata, b, bpos, xpp, ypp, spp, tpp, p);
              //check_offset(p, b, x, y, xbest, ybest);
              //avoid_attempt_n<PATCH_W, SPARSE, IS_MASK, IS_WINDOW>(err, xbest, ybest, adata, b, xpp, ypp, bmask, region_masks, src_mask, p);
            }
          }
//#endif

          //if (avoid_current->count(XY_TO_INT(xbest, ybest)) != 0) { fprintf(stderr, "xbest, ybest already in hash map (after iter)\n"); exit(1); }
          //avoid[y*a->w+x].insert(XY_TO_INT(xbest, ybest));

          ann->get(x, y)[nn_i] = XY_TO_INT(xbest, ybest);
#if !TRANSLATE_ONLY
          ann_sim->get(x, y)[nn_i] = XY_TO_INT(sbest, tbest);
#endif
          annd->get(x, y)[nn_i] = err; //annd_row[x];
          /*((int *) ann->line[y])[x] = XY_TO_INT(xbest, ybest);
          ((int *) ann_sim->line[y])[x] = XY_TO_INT(sbest, tbest);
          ((int *) annd->line[y])[x] = err;*/
        } // x
      } // y
    } // parallel
    //fprintf(stderr, "done with %d iters\n", nn_iter);
  } // nn_iter
  
  //fprintf(stderr, "nn_i=%d\n", nn_i);
  for (int y = box.ymin; y < box.ymax; y++) {
    for (int x = box.xmin; x < box.xmax; x++) {
      PositionSet *avoid_current = &avoid[y*a->w+x];
      //if (avoid_current->size() != 0 && nn_i == 0) { fprintf(stderr, "avoid size nonzero on first iter\n"); exit(1); }
      int ann_v = ann->get(x, y)[nn_i];
      //if (avoid_current->count(ann_v)) { fprintf(stderr, "xbest, ybest already in hash map (on insert, nn_i=%d)\n", nn_i); exit(1); }
      //avoid_current->insert(ann_v);
      //if (avoid_current->contains_noqueue(INT_TO_X(ann_v), INT_TO_Y(ann_v), nn_i)) { fprintf(stderr, "xbest, ybest already in hash map (on insert, nn_i=%d)\n", nn_i); exit(1); }
      avoid_current->insert_nonexistent(INT_TO_X(ann_v), INT_TO_Y(ann_v));
    }
  }
  
  } // nn_i
  printf("done knn_n_avoid, rs_max=%d\n", p->rs_max);
  delete[] avoid;
}

template<int PATCH_W>
void knn_min_pair_n(Params *p, BITMAP *a, BITMAP *b, VBMP *ann, VBMP *annd, VBMP *ann_sim, VBMP *ann_temp, VBMP *annd_temp, VBMP *ann_sim_temp) {
  if (ann->w != a->w || ann->h != a->h) { fprintf(stderr, "ann size (%dx%d) differs from a size (%dx%d)\n", ann->w, ann->h, a->w, a->h); exit(1); }
  if (annd->w != a->w || annd->h != a->h) { fprintf(stderr, "annd size (%dx%d) differs from a size (%dx%d)\n", annd->w, annd->h, a->w, a->h); exit(1); }
  if (ann_temp->w != a->w || ann_temp->h != a->h) { fprintf(stderr, "ann_temp size (%dx%d) differs from a size (%dx%d)\n", ann_temp->w, ann_temp->h, a->w, a->h); exit(1); }
  if (annd_temp->w != a->w || annd_temp->h != a->h) { fprintf(stderr, "annd_temp size (%dx%d) differs from a size (%dx%d)\n", annd_temp->w, annd_temp->h, a->w, a->h); exit(1); }
#if !TRANSLATE_ONLY
  if (ann_sim->w != a->w || ann_sim->h != a->h) { fprintf(stderr, "ann_sim size (%dx%d) differs from a size (%dx%d)\n", ann_sim->w, ann_sim->h, a->w, a->h); exit(1); }
  if (ann_sim_temp->w != a->w || ann_sim_temp->h != a->h) { fprintf(stderr, "ann_sim_temp size (%dx%d) differs from a size (%dx%d)\n", ann_sim_temp->w, ann_sim_temp->h, a->w, a->h); exit(1); }
#endif

  vector<qtype<int> > v;
  v.reserve(p->knn+1);

  Box box = get_abox(p, a, NULL);
  XFORM bpos = get_xform(p, 0, 0, SCALE_UNITY, 0);
  for (int y = box.ymin; y < box.ymax; y++) {
    //int *annr = (int *) ann->line[y];
    //int *anndr = (int *) annd->line[y];
    //int *ann_tempr = (int *) ann_temp->line[y];
    //int *annd_tempr = (int *) annd_temp->line[y];
    for (int x = box.xmin; x < box.xmax; x++) {
      //if (annd_tempr[x] < anndr[x]) {
      //  anndr[x] = annd_tempr[x];
      //  annr[x] = ann_tempr[x];
      //}

      int *p_ann = ann->get(x, y);
      int *p_annd = annd->get(x, y);
#if !TRANSLATE_ONLY
      int *p_ann_sim = ann_sim->get(x, y);
#endif

      v.clear();
      for (int i = 0; i < p->knn; i++) {
#if !TRANSLATE_ONLY
        v.push_back(qtype<int>(p_annd[i], p_ann[i], p_ann_sim[i]));
#else
        v.push_back(qtype<int>(p_annd[i], p_ann[i]));
#endif
      }

      int *q_ann = ann_temp->get(x, y);
      int *q_annd = annd_temp->get(x, y);
#if !TRANSLATE_ONLY
      int *q_ann_sim = ann_sim_temp->get(x, y);
#endif
      PositionSet pos(p_ann, &v, p->knn);   // FIXME: Unimplemented
      for (int i = 0; i < p->knn; i++) {
        pos.insert_nonexistent(INT_TO_X(p_ann[i]), INT_TO_Y(p_ann[i]));
      }
      for (int i = 0; i < p->knn; i++) {
        int xpp = INT_TO_X(q_ann[i]), ypp = INT_TO_Y(q_ann[i]);
#if !TRANSLATE_ONLY
        int spp = INT_TO_X(q_ann_sim[i]), tpp = INT_TO_Y(q_ann_sim[i]);
#else
        int spp = SCALE_UNITY, tpp = 0;
#endif
        knn_attempt_n<PATCH_W, 1>(v, NULL, b, bpos, xpp, ypp, spp, tpp, p, q_annd[i], pos);
      }
      for (int i = 0; i < p->knn; i++) {
        p_annd[i] = v[i].a;
        p_ann[i] = v[i].b;
#if !TRANSLATE_ONLY
        p_ann_sim[i] = v[i].c;
#endif
      }
    }
  }
}

void knn_min_pair(Params *p, BITMAP *a, BITMAP *b, VBMP *ann0, VBMP *annd0, VBMP *ann_sim0, VBMP *ann_temp, VBMP *annd_temp, VBMP *ann_sim_temp) {
  if      (p->patch_w == 1 ) { return knn_min_pair_n<1 >(p, a, b, ann0, annd0, ann_sim0, ann_temp, annd_temp, ann_sim_temp); }
  else if (p->patch_w == 2 ) { return knn_min_pair_n<2 >(p, a, b, ann0, annd0, ann_sim0, ann_temp, annd_temp, ann_sim_temp); }
  else if (p->patch_w == 3 ) { return knn_min_pair_n<3 >(p, a, b, ann0, annd0, ann_sim0, ann_temp, annd_temp, ann_sim_temp); }
  else if (p->patch_w == 4 ) { return knn_min_pair_n<4 >(p, a, b, ann0, annd0, ann_sim0, ann_temp, annd_temp, ann_sim_temp); }
  else if (p->patch_w == 5 ) { return knn_min_pair_n<5 >(p, a, b, ann0, annd0, ann_sim0, ann_temp, annd_temp, ann_sim_temp); }
  else if (p->patch_w == 6 ) { return knn_min_pair_n<6 >(p, a, b, ann0, annd0, ann_sim0, ann_temp, annd_temp, ann_sim_temp); }
  else if (p->patch_w == 7 ) { return knn_min_pair_n<7 >(p, a, b, ann0, annd0, ann_sim0, ann_temp, annd_temp, ann_sim_temp); }
  else if (p->patch_w == 8 ) { return knn_min_pair_n<8 >(p, a, b, ann0, annd0, ann_sim0, ann_temp, annd_temp, ann_sim_temp); }
  else if (p->patch_w == 9 ) { return knn_min_pair_n<9 >(p, a, b, ann0, annd0, ann_sim0, ann_temp, annd_temp, ann_sim_temp); }
  else if (p->patch_w == 10) { return knn_min_pair_n<10>(p, a, b, ann0, annd0, ann_sim0, ann_temp, annd_temp, ann_sim_temp); }
  else if (p->patch_w == 11) { return knn_min_pair_n<11>(p, a, b, ann0, annd0, ann_sim0, ann_temp, annd_temp, ann_sim_temp); }
  else if (p->patch_w == 12) { return knn_min_pair_n<12>(p, a, b, ann0, annd0, ann_sim0, ann_temp, annd_temp, ann_sim_temp); }
  else if (p->patch_w == 13) { return knn_min_pair_n<13>(p, a, b, ann0, annd0, ann_sim0, ann_temp, annd_temp, ann_sim_temp); }
  else if (p->patch_w == 14) { return knn_min_pair_n<14>(p, a, b, ann0, annd0, ann_sim0, ann_temp, annd_temp, ann_sim_temp); }
  else if (p->patch_w == 15) { return knn_min_pair_n<15>(p, a, b, ann0, annd0, ann_sim0, ann_temp, annd_temp, ann_sim_temp); }
  else if (p->patch_w == 16) { return knn_min_pair_n<16>(p, a, b, ann0, annd0, ann_sim0, ann_temp, annd_temp, ann_sim_temp); }
  else if (p->patch_w == 17) { return knn_min_pair_n<17>(p, a, b, ann0, annd0, ann_sim0, ann_temp, annd_temp, ann_sim_temp); }
  else if (p->patch_w == 18) { return knn_min_pair_n<18>(p, a, b, ann0, annd0, ann_sim0, ann_temp, annd_temp, ann_sim_temp); }
  else if (p->patch_w == 19) { return knn_min_pair_n<19>(p, a, b, ann0, annd0, ann_sim0, ann_temp, annd_temp, ann_sim_temp); }
  else if (p->patch_w == 20) { return knn_min_pair_n<20>(p, a, b, ann0, annd0, ann_sim0, ann_temp, annd_temp, ann_sim_temp); }
  else if (p->patch_w == 21) { return knn_min_pair_n<21>(p, a, b, ann0, annd0, ann_sim0, ann_temp, annd_temp, ann_sim_temp); }
  else if (p->patch_w == 22) { return knn_min_pair_n<22>(p, a, b, ann0, annd0, ann_sim0, ann_temp, annd_temp, ann_sim_temp); }
  else if (p->patch_w == 23) { return knn_min_pair_n<23>(p, a, b, ann0, annd0, ann_sim0, ann_temp, annd_temp, ann_sim_temp); }
  else if (p->patch_w == 24) { return knn_min_pair_n<24>(p, a, b, ann0, annd0, ann_sim0, ann_temp, annd_temp, ann_sim_temp); }
  else if (p->patch_w == 25) { return knn_min_pair_n<25>(p, a, b, ann0, annd0, ann_sim0, ann_temp, annd_temp, ann_sim_temp); }
  else if (p->patch_w == 26) { return knn_min_pair_n<26>(p, a, b, ann0, annd0, ann_sim0, ann_temp, annd_temp, ann_sim_temp); }
  else if (p->patch_w == 27) { return knn_min_pair_n<27>(p, a, b, ann0, annd0, ann_sim0, ann_temp, annd_temp, ann_sim_temp); }
  else if (p->patch_w == 28) { return knn_min_pair_n<28>(p, a, b, ann0, annd0, ann_sim0, ann_temp, annd_temp, ann_sim_temp); }
  else if (p->patch_w == 29) { return knn_min_pair_n<29>(p, a, b, ann0, annd0, ann_sim0, ann_temp, annd_temp, ann_sim_temp); }
  else if (p->patch_w == 30) { return knn_min_pair_n<30>(p, a, b, ann0, annd0, ann_sim0, ann_temp, annd_temp, ann_sim_temp); }
  else if (p->patch_w == 31) { return knn_min_pair_n<31>(p, a, b, ann0, annd0, ann_sim0, ann_temp, annd_temp, ann_sim_temp); }
  else if (p->patch_w == 32) { return knn_min_pair_n<32>(p, a, b, ann0, annd0, ann_sim0, ann_temp, annd_temp, ann_sim_temp); }
  else { fprintf(stderr, "Patch size unsupported for knn_min_pair: %d\n", p->patch_w); exit(1); }
}

void knn_oneway(Params *p, BITMAP *a, BITMAP *b,
         VBMP *ann, VBMP *ann_sim, VBMP *annd,
         RegionMasks *amask, BITMAP *bmask,
         int level, int em_iter, RecomposeParams *rp, int offset_iter, int update_type, int cache_b,
         RegionMasks *region_masks, int tiles, PRINCIPAL_ANGLE *pa, int save_first) {
#if TRANSLATE_ONLY
  if (ann_sim) { fprintf(stderr, "ann_sim non-NULL, but TRANSLATE_ONLY=1\n"); exit(1); }
#endif
  VBMP *ann0 = ann, *annd0 = annd, *ann_sim0 = ann_sim;
  VBMP *ann_temp = NULL, *annd_temp = NULL, *ann_sim_temp = NULL;
  //PRINCIPAL_ANGLE *pa = NULL;
  //if (p->knn_algo == KNN_ALGO_PRINANGLE) {
  //  if (a != b) { fprintf(stderr, "principal angle unimplemented for a and b different bitmaps\n"); exit(1); }
  //  pa = create_principal_angle(p, a);
  //}
  for (int restart = 0; restart < p->restarts; restart++) {
    if (restart > 0) {
      ann = ann_temp = knn_init_nn(p, a, b, ann_sim);
#if TRANSLATE_ONLY
      if (ann_sim) { fprintf(stderr, "restarts>0, ann_sim non-NULL, but TRANSLATE_ONLY=1\n"); exit(1); }
#endif
      ann_sim_temp = ann_sim;  //init_nn(p, a, b, bmask, region_masks, amask, 1, ann_window, awinsize);
      annd = annd_temp = knn_init_dist(p, a, b, ann, ann_sim); //init_dist(p, a, b, ann_temp, bmask, region_masks, amask);
    }

    if (p->knn_algo == KNN_ALGO_HEAP) {
      if      (p->patch_w == 1) { knn_n<1,0>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 2) { knn_n<2,0>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 3) { knn_n<3,0>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 4) { knn_n<4,0>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 5) { knn_n<5,0>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 6) { knn_n<6,0>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 7) { knn_n<7,0>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 8) { knn_n<8,0>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 9) { knn_n<9,0>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 10) { knn_n<10,0>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 11) { knn_n<11,0>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 12) { knn_n<12,0>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 13) { knn_n<13,0>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 14) { knn_n<14,0>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 15) { knn_n<15,0>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 16) { knn_n<16,0>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 17) { knn_n<17,0>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 18) { knn_n<18,0>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 19) { knn_n<19,0>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 20) { knn_n<20,0>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 21) { knn_n<21,0>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 22) { knn_n<22,0>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 23) { knn_n<23,0>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 24) { knn_n<24,0>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 25) { knn_n<25,0>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 26) { knn_n<26,0>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 27) { knn_n<27,0>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 28) { knn_n<28,0>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 29) { knn_n<29,0>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 30) { knn_n<30,0>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 31) { knn_n<31,0>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 32) { knn_n<32,0>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else { fprintf(stderr, "Patch size unsupported for knn_oneway: %d\n", p->patch_w); exit(1); }
    } else if (p->knn_algo == KNN_ALGO_PRINANGLE) {
      if      (p->patch_w == 1) { knn_n<1,1>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 2) { knn_n<2,1>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 3) { knn_n<3,1>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 4) { knn_n<4,1>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 5) { knn_n<5,1>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 6) { knn_n<6,1>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 7) { knn_n<7,1>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 8) { knn_n<8,1>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 9) { knn_n<9,1>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 10) { knn_n<10,1>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 11) { knn_n<11,1>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 12) { knn_n<12,1>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 13) { knn_n<13,1>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 14) { knn_n<14,1>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 15) { knn_n<15,1>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 16) { knn_n<16,1>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 17) { knn_n<17,1>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 18) { knn_n<18,1>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 19) { knn_n<19,1>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 20) { knn_n<20,1>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 21) { knn_n<21,1>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 22) { knn_n<22,1>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 23) { knn_n<23,1>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 24) { knn_n<24,1>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 25) { knn_n<25,1>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 26) { knn_n<26,1>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 27) { knn_n<27,1>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 28) { knn_n<28,1>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 29) { knn_n<29,1>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 30) { knn_n<30,1>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 31) { knn_n<31,1>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 32) { knn_n<32,1>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else { fprintf(stderr, "Patch size unsupported for knn_oneway: %d\n", p->patch_w); exit(1); }
    } else if (p->knn_algo == KNN_ALGO_AVOID) {
      if      (p->patch_w == 1) { knn_n_avoid<1>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles); }
      else if (p->patch_w == 2) { knn_n_avoid<2>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles); }
      else if (p->patch_w == 3) { knn_n_avoid<3>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles); }
      else if (p->patch_w == 4) { knn_n_avoid<4>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles); }
      else if (p->patch_w == 5) { knn_n_avoid<5>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles); }
      else if (p->patch_w == 6) { knn_n_avoid<6>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles); }
      else if (p->patch_w == 7) { knn_n_avoid<7>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles); }
      else if (p->patch_w == 8) { knn_n_avoid<8>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles); }
      else if (p->patch_w == 9) { knn_n_avoid<9>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles); }
      else if (p->patch_w == 10) { knn_n_avoid<10>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles); }
      else if (p->patch_w == 11) { knn_n_avoid<11>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles); }
      else if (p->patch_w == 12) { knn_n_avoid<12>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles); }
      else if (p->patch_w == 13) { knn_n_avoid<13>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles); }
      else if (p->patch_w == 14) { knn_n_avoid<14>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles); }
      else if (p->patch_w == 15) { knn_n_avoid<15>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles); }
      else if (p->patch_w == 16) { knn_n_avoid<16>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles); }
      else if (p->patch_w == 17) { knn_n_avoid<17>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles); }
      else if (p->patch_w == 18) { knn_n_avoid<18>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles); }
      else if (p->patch_w == 19) { knn_n_avoid<19>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles); }
      else if (p->patch_w == 20) { knn_n_avoid<20>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles); }
      else if (p->patch_w == 21) { knn_n_avoid<21>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles); }
      else if (p->patch_w == 22) { knn_n_avoid<22>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles); }
      else if (p->patch_w == 23) { knn_n_avoid<23>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles); }
      else if (p->patch_w == 24) { knn_n_avoid<24>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles); }
      else if (p->patch_w == 25) { knn_n_avoid<25>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles); }
      else if (p->patch_w == 26) { knn_n_avoid<26>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles); }
      else if (p->patch_w == 27) { knn_n_avoid<27>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles); }
      else if (p->patch_w == 28) { knn_n_avoid<28>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles); }
      else if (p->patch_w == 29) { knn_n_avoid<29>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles); }
      else if (p->patch_w == 30) { knn_n_avoid<30>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles); }
      else if (p->patch_w == 31) { knn_n_avoid<31>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles); }
      else if (p->patch_w == 32) { knn_n_avoid<32>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles); }
      else { fprintf(stderr, "Patch size unsupported for knn_oneway: %d\n", p->patch_w); exit(1); }
    } else if (p->knn_algo == KNN_ALGO_TOP1NN) {
      if      (p->patch_w == 1) { knn_n_top1nn<1>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 2) { knn_n_top1nn<2>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 3) { knn_n_top1nn<3>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 4) { knn_n_top1nn<4>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 5) { knn_n_top1nn<5>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 6) { knn_n_top1nn<6>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 7) { knn_n_top1nn<7>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 8) { knn_n_top1nn<8>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 9) { knn_n_top1nn<9>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 10) { knn_n_top1nn<10>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 11) { knn_n_top1nn<11>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 12) { knn_n_top1nn<12>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 13) { knn_n_top1nn<13>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 14) { knn_n_top1nn<14>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 15) { knn_n_top1nn<15>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 16) { knn_n_top1nn<16>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 17) { knn_n_top1nn<17>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 18) { knn_n_top1nn<18>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 19) { knn_n_top1nn<19>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 20) { knn_n_top1nn<20>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 21) { knn_n_top1nn<21>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 22) { knn_n_top1nn<22>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 23) { knn_n_top1nn<23>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 24) { knn_n_top1nn<24>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 25) { knn_n_top1nn<25>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 26) { knn_n_top1nn<26>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 27) { knn_n_top1nn<27>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 28) { knn_n_top1nn<28>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 29) { knn_n_top1nn<29>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 30) { knn_n_top1nn<30>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 31) { knn_n_top1nn<31>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 32) { knn_n_top1nn<32>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else { fprintf(stderr, "Patch size unsupported for knn_oneway: %d\n", p->patch_w); exit(1); }
    } else if (p->knn_algo == KNN_ALGO_WINDOW) {
      if      (p->patch_w == 1) { knn_n_window<1>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 2) { knn_n_window<2>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 3) { knn_n_window<3>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 4) { knn_n_window<4>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 5) { knn_n_window<5>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 6) { knn_n_window<6>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 7) { knn_n_window<7>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 8) { knn_n_window<8>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 9) { knn_n_window<9>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 10) { knn_n_window<10>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 11) { knn_n_window<11>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 12) { knn_n_window<12>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 13) { knn_n_window<13>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 14) { knn_n_window<14>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 15) { knn_n_window<15>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 16) { knn_n_window<16>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 17) { knn_n_window<17>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 18) { knn_n_window<18>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 19) { knn_n_window<19>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 20) { knn_n_window<20>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 21) { knn_n_window<21>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 22) { knn_n_window<22>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 23) { knn_n_window<23>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 24) { knn_n_window<24>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 25) { knn_n_window<25>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 26) { knn_n_window<26>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 27) { knn_n_window<27>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 28) { knn_n_window<28>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 29) { knn_n_window<29>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 30) { knn_n_window<30>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 31) { knn_n_window<31>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else if (p->patch_w == 32) { knn_n_window<32>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first); }
      else { fprintf(stderr, "Patch size unsupported for knn_oneway: %d\n", p->patch_w); exit(1); }
    } else if (p->knn_algo == KNN_ALGO_KDTREE || p->knn_algo == KNN_ALGO_FLANN) {
      fprintf(stderr, "kd-tree not implemented in mex code\n"); exit(1);
//      tree_nn(p, a, b, NULL, NULL, NULL, NULL, 0, NULL, 0, 0, INT_MAX, INT_MAX, 0, 100, ann, annd);
    }
    if (ann_temp) {
      knn_min_pair(p, a, b, ann0, annd0, ann_sim0, ann_temp, annd_temp, ann_sim_temp);
      delete ann_temp; //destroy_bitmap(ann_temp);
      delete annd_temp; //destroy_bitmap(annd_temp);
      delete ann_sim_temp; //destroy_bitmap(ann_sim_temp);
      ann_temp = annd_temp = ann_sim_temp = NULL;
    }
  }
//  destroy_principal_angle(pa);
}

int change_k_to = 0;
int forward_enrich_hops = 2;
int inverse_enrich_hops = 1;

void knn(Params *p, BITMAP *a, BITMAP *b,
         VBMP *&ann, VBMP *&ann_sim, VBMP *&annd,
         RegionMasks *amask, BITMAP *bmask,
         int level, int em_iter, RecomposeParams *rp, int offset_iter, int update_type, int cache_b,
         RegionMasks *region_masks, int tiles, PRINCIPAL_ANGLE *pa, int save_first) {

  if (p->knn_algo == KNN_ALGO_CHANGEK) {
    if (change_k_to <= 1) { fprintf(stderr, "in KNN_ALGO_CHANGEK, but need -changek (change_k_to) specified\n"); exit(1); }
    fprintf(stderr, "KNN_ALGO_CHANGEK, first part, k=%d\n", p->knn);
    Params pcopy(*p);
    pcopy.knn_algo = KNN_ALGO_HEAP;
    pcopy.nn_iters = p->nn_iters/2;
    knn(&pcopy, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first);
    fprintf(stderr, "KNN_ALGO_CHANGEK, second part, changing k to %d\n", change_k_to);
    change_knn(&pcopy, a, b, ann, ann_sim, annd, change_k_to, pa);
    fprintf(stderr, "After resizing arrays, k=%d\n", pcopy.knn);
    pcopy.nn_iters = p->nn_iters-pcopy.nn_iters;
    knn(&pcopy, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles, pa, save_first);
    p->knn = pcopy.knn;
    fprintf(stderr, "Done KNN_ALGO_CHANGEK, setting p->knn=%d\n", p->knn);
    return;
  }
  int enrich_iters = p->enrich_iters;
  if (enrich_iters == 0) { enrich_iters = 1; }
  double enrich_T = 0;
  for (int iter = 0; iter < enrich_iters; iter++) {
    knn_oneway(p, a, b, ann, ann_sim, annd, NULL, NULL, 0, 0, NULL, 0, 0, 0, NULL, -1, pa, iter == 0);

    double enrich_start_t = accurate_timer();
    if (p->enrich_iters != 0) {
      for (int enrich_i = 0; enrich_i < p->enrich_times; enrich_i++) {
        if (a != b) { fprintf(stderr, "implementation assumes a == b\n"); exit(1); }
        if (p->do_inverse_enrich) {
          if (inverse_enrich_hops == 1) {
            knn_inverse_enrich(p, a, a, ann, ann_sim, annd, pa);
          } else if (inverse_enrich_hops == 2) {
            knn_inverse_enrich2(p, a, a, ann, ann_sim, annd, pa);
          } else {
            fprintf(stderr, "inverse enrich hops: %d unsupported\n", inverse_enrich_hops); exit(1);
          }
        }
        //char buf[256];
        //sprintf(buf, "dist%d.bmp", nn_iter+1);
  #if SAVE_DIST
        save_dist(p, annd, "inverse_enrich");          //knn_inverse_enrich2(p, a, a, ann, ann_sim, annd, pa);
  #endif
        if (p->do_enrich) {
          if (forward_enrich_hops == 2) {
            knn_enrich(p, a, a, ann, ann_sim, annd, pa);
          } else if (forward_enrich_hops == 3) {
            knn_enrich3(p, a, a, ann, ann_sim, annd, pa);
          } else if (forward_enrich_hops == 4) {
            knn_enrich4(p, a, a, ann, ann_sim, annd, pa);
          } else {
            fprintf(stderr, "forward enrich hops: %d unsupported\n", forward_enrich_hops); exit(1);
          }
        }
  #if SAVE_DIST
        save_dist(p, annd, "enrich");
  #endif
        //knn_enrich3(p, a, a, ann, ann_sim, annd, pa);
        //knn_inverse_enrich(p, a, a, ann, ann_sim, annd, pa);
        //knn_enrich(p, a, a, ann, ann_sim, annd, pa);
        //knn_enrich(p, a, a, ann, ann_sim, annd, pa);
      }
    }
    enrich_T += accurate_timer() - enrich_start_t;
  }
  printf("enrich time: %f secs\n", enrich_T);
}

#if USE_ERF
#include "erf.cpp"
#endif

int norm_channel(Params *p, const vector<pair<int, int> > &L, int verbose, int cnoise) {
  fprintf(stderr, "norm_channel disabled\n"); exit(1);
#if 0
  if (L.size() == 0) { return 255; /*fprintf(stderr, "Cannot normalize 0 length vector\n"); exit(1);*/ }
#if KNN_MEDOID
  vector<int> values;
  vector<double> weights;
#else
  double ans = 0, wans = 0;
  double noise_sigma = 300*2;
#endif
  for (int i = 0; i < (int) L.size(); i++) {
    int d = L[i].first;
    int v = INT_TO_X(L[i].second);
    int ip_full = INT_TO_Y(L[i].second);
    int ip = ip_full&255;
    int wi = (ip_full>>8);
    double w = wi; //exp(-d*d/(2.0*p->nlm_sigma*p->nlm_sigma));
    //if (i == 0 && d != 0) { fprintf(stderr, "distance for element 0 is nonzero: %d\n", d); exit(1); }
    if (p->nlm_weight == WEIGHT_GAUSSIAN) {
      if (ip == 0 && (unsigned) (i+1) < (unsigned) L.size()) {
        d = L[i+1].first; //*0.925; //*0.95;
      }
      //w *= exp(-d*d*d/(2.0*p->nlm_sigma*p->nlm_sigma*p->nlm_sigma));
#if USE_GAUSSIAN
#define GAUSSIAN_TERM (gaussian_kernel_sum)
#else
#define GAUSSIAN_TERM 1
#endif
#if USE_L1
      w *= exp(-d*d/(p->nlm_sigma*GAUSSIAN_TERM));
#else
      w *= exp(-d/(p->nlm_sigma*GAUSSIAN_TERM));
#endif
      //w *= exp(-d/(2.0*p->nlm_sigma*p->nlm_sigma));
      //w = exp(-d/(2.0*p->nlm_sigma));
    }
#if !KNN_MEDOID
    ans += v*w;
    wans += w;
#else
    values.push_back(v);
    weights.push_back(w);
#endif
    if (verbose) { fprintf(stderr, "%d %d %d %f\n", i, ip, d, w); }
  }
  //printf("  %d\n", values.size());
  if (verbose) { fprintf(stderr, "\n"); }
  //if (wans == 0) { fprintf(stderr, "Cannot normalize vector with weights adding to zero\n"); exit(1); }
  //if (wans == 0) { return 255; } // FIXME
#if !KNN_MEDOID
  int result = int(ans/wans+0.5);
#else
  int ans = 0;
  double medoid_ans = 1e100;
  //for (int i = 0; i < (int) MIN(values.size(), 2); i++) {
  for (int iter = 0; iter < 20; iter++) {
    int i = rand()%values.size();
    double dmed = 0;
    for (int j = 0; j < (int) values.size(); j++) {
      dmed += abs(values[i]-values[j])*weights[j];
    }
    if (dmed < medoid_ans) {
      medoid_ans = dmed;
      ans = values[i];
    }
  }
  int result = ans;
#endif
#if USE_ERF
  double diff = abs(result - cnoise);
  double P_result = 1-erf(diff / (noise_sigma*sqrt(2.0)));
  //if (abs(result - cnoise) > noise_sigma) {
  //  
  //}
  return P_result * result + (1-P_result) * cnoise;
#else
  return result;
#endif
#endif
}

#if VOTE_SUM
BITMAP *knn_norm_image(Params *p, double *accum, int w, int h, BITMAP *b) {
#else
BITMAP *knn_norm_image(Params *p, vector<pair<int, int> > *accum, int w, int h, BITMAP *b) {
#endif
  BITMAP *ans = create_bitmap(w, h);
  #pragma omp parallel for schedule(dynamic, 1)
  for (int x = 0; x < w; x++) {
    for (int y = 0; y < h; y++) {
      int c = ((int *) (b->line[y]))[x];
      int *row = (int *) ans->line[y];
#if VOTE_SUM
      double *prow = &accum[4*(y*w+x)];
      int r, g, b;
      if (prow[3] != 0) {
        double scale = 1.0/prow[3];
        r = int(prow[0]*scale+0.5);
        g = int(prow[1]*scale+0.5);
        b = int(prow[2]*scale+0.5);
      } else {
        r = g = b = 255;
      }
#else
      vector<pair<int, int> > *prow = &accum[3*(y*w+x)];
      int is_verbose = (x==10 && y == 5);
      int r = norm_channel(p, prow[0], is_verbose, c&255);
      int g = norm_channel(p, prow[1], is_verbose, (c>>8)&255);
      int b = norm_channel(p, prow[2], is_verbose, (c>>16));
#endif
      row[x] = (r)|(g<<8)|(b<<16);
    }
  }
  return ans;
}

static int write_weights = 0;

template<int PATCH_W, class ACCUM>
BITMAP *knn_vote_n(Params *p, BITMAP *b,
                 VBMP *ann, VBMP *ann_sim, VBMP *annd, VBMP *bnn, VBMP *bnn_sim,
                 BITMAP *bmask, BITMAP *bweight,
                 double coherence_weight, double complete_weight,
                 RegionMasks *amask, BITMAP *aweight, BITMAP *ainit, RegionMasks *region_masks, BITMAP *aconstraint, int mask_self_only, KNNWeightFunc *weight_func, double **accum_out) {
  FILE *fout = NULL;
  if (write_weights) {
    fout = fopen("weight_plot.txt", "wt");
  }

#if TRANSLATE_ONLY
  if (ann_sim) { fprintf(stderr, "in mode TRANSLATE_ONLY, expected ann_sim to be NULL\n"); exit(1); }
#endif

  init_xform_tables();
#if VOTE_SUM
  double *accum = new double[4*ann->w*ann->h];
  memset((void *) accum, 0, sizeof(double)*4*ann->w*ann->h);
  if (accum_out) {
    *accum_out = accum;
  }
#else
  vector<pair<int, int> > *accum = new vector<pair<int, int> >[3*ann->w*ann->h];
  if (accum_out) { fprintf(stderr, "accum_out not implemented if VOTE_SUM=0\n"); exit(1); }
#endif
  int do_clear = 0;
  if (!VOTE_SUM) { do_clear = 1; fprintf(stderr, "not VOTE_SUM not implemented\n"); exit(1); }
  int y_last_clear = 0;
  BITMAP *ans = do_clear ? create_bitmap(ann->w, ann->h): NULL;
/*  if (accum_out) {
    accum_out = new double[ann->w*ann->h*4];
  }*/
#if ((!OVERLAPPING_PATCHES)&&VOTE_SUM)
#pragma omp parallel for schedule(static, 8)
#endif
  for (int y = 0; y < ann->h-p->patch_w+1; y++) {
    vector<qtype<int> > v;
    v.reserve(p->knn);
    //printf("%d\n", y);
    for (int x = 0; x < ann->w-p->patch_w+1; x++) {
      int *p_ann = ann->get(x, y);
#if !TRANSLATE_ONLY
      int *p_ann_sim = ann_sim->get(x, y);
#endif
      int *p_annd = annd->get(x, y);
      v.clear();
      for (int i = 0; i < p->knn; i++) {
#if !TRANSLATE_ONLY
        v.push_back(qtype<int>(p_annd[i], p_ann[i], p_ann_sim[i]));
#else
        v.push_back(qtype<int>(p_annd[i], p_ann[i]));
#endif
      }
      sort(v.begin(), v.end());
      for (int i = 0; i < p->knn; i++) {
        p_annd[i] = v[i].a;
        p_ann[i] = v[i].b;
#if !TRANSLATE_ONLY
        p_ann_sim[i] = v[i].c;
#endif
      }
      double write_weights_sum = 0, write_weights_denom = 0;
      
      for (int i = 0; i < p->knn; i++) {
        //int xp, yp, sp, tp;
        //getnn(ann, x, y, xp, yp);
        //getnn(ann_sim, x, y, sp, tp);
        int xp = INT_TO_X(p_ann[i]);
        int yp = INT_TO_Y(p_ann[i]);
        int d = p_annd[i];
        /*if (d == 0) {
          int dmin = INT_MAX;
          for (int j = 0; j < p->knn; j++) {
            if (p_annd[j] < dmin && p_annd[j] != 0) { dmin = p_annd[j]; }
          }
          d = dmin;
        }*/
#if !TRANSLATE_ONLY
        int sp = INT_TO_X(p_ann_sim[i]);
        int tp = INT_TO_Y(p_ann_sim[i]);
        XFORM bpos = get_xform(p, xp, yp, sp, tp);
        int bx_row = bpos.x0, by_row = bpos.y0;
#endif
        int dy_min = 0, dy_max = PATCH_W, dx_min = 0, dx_max = PATCH_W;
        int is_border = (y == 0) || (x == 0) || (x == ann->w-p->patch_w) || (y == ann->h-p->patch_w);
#if !OVERLAPPING_PATCHES
        if (!is_border) {
          dy_min = PATCH_W/2; dy_max = PATCH_W/2+1;
          dx_min = PATCH_W/2; dx_max = PATCH_W/2+1;

          //bx += ;
          //by += ;
#if !TRANSLATE_ONLY
          bx_row += bpos.dxdv*dx_min + bpos.dxdu*dx_min;
          by_row += bpos.dydv*dy_min + bpos.dydu*dy_min;
#endif
        }
#endif
        for (int dy = dy_min; dy < dy_max; dy++) {
#if !TRANSLATE_ONLY
          int bx = bx_row, by = by_row;
#else
          if (yp >= BEH || xp >= BEW) { fprintf(stderr, "coords out of range\n"); exit(1); }
          int *src_row = ((int *) b->line[yp+dy])+xp;
#endif
#if VOTE_SUM
          double *prow = &accum[4*((y+dy)*ann->w+x)];
#else
          vector<pair<int, int> > *prow = &accum[3*((y+dy)*ann->w+x)];
#endif
          for (int dx = dx_min; dx < dx_max; dx++) {
#if !OVERLAPPING_PATCHES
            if ((dx == PATCH_W/2 && dy == PATCH_W/2) || (y == 0 && dy < PATCH_W/2) || (x == 0 && dx < PATCH_W/2) || (x == ann->w-p->patch_w && dx > PATCH_W/2) || (y == ann->h-p->patch_w && dy > PATCH_W/2)) {
            //if (!is_border || ((y == 0 && dy < PATCH_W/2) || (x == 0 && dx < PATCH_W/2) || (x == ann->w-p->patch_w && dx > PATCH_W/2) || (y == ann->h-p->patch_w && dy > PATCH_W/2))) {
#endif
#if VOTE_SUM
              double *ptr = &prow[4*dx];
#else
              vector<pair<int, int> > *ptr = &prow[3*dx];
#endif
#if !TRANSLATE_ONLY
              int rv, gv, bv;
              getpixel_bilin(b, bx, by, rv, gv, bv);
#else
              int cv = src_row[dx];
              int rv = cv&255, gv = (cv>>8)&255, bv = (cv>>16);
#endif
              //int wi = simnn_weight[dy][dx];
#if VOTE_SUM
              int dw = d;
              if (i == 0 && i < p->knn-1) {
                dw = p_annd[i+1];
              }
#if 0
              double ww;
              if (weight_func) {
                double dactual = (USE_L1 ? dw: sqrt(double(dw)));
                int is_center = i == 0 && i < p->knn-1;
                ww = weight_func->weight(dactual, is_center);
                if (write_weights) {
                  write_weights_sum += ww;
                  write_weights_denom += 1;
                }
              } else {
#if USE_L1
                ww = exp(-dw*dw/(p->nlm_sigma*GAUSSIAN_TERM));
#else
                ww = exp(-dw/(p->nlm_sigma*GAUSSIAN_TERM));
#endif
              }
#endif
              int ww = 1;
              ptr[0] += rv*ww;
              ptr[1] += gv*ww;
              ptr[2] += bv*ww;
              ptr[3] += ww;
#else
              if (weight_func) { fprintf(stderr, "weight func not implemented when VOTE_SUM=0\n"); exit(1); }
              int wi = 1;
              ptr[0].push_back(pair<int, int>(d, XY_TO_INT(rv, i|(wi<<8))));
              ptr[1].push_back(pair<int, int>(d, XY_TO_INT(gv, i|(wi<<8))));
              ptr[2].push_back(pair<int, int>(d, XY_TO_INT(bv, i|(wi<<8))));
#endif
              
#if !OVERLAPPING_PATCHES
            }
#endif
#if !TRANSLATE_ONLY
            bx += bpos.dxdu;
            by += bpos.dydu;
#endif
          }
#if !TRANSLATE_ONLY
          bx_row += bpos.dxdv;
          by_row += bpos.dydv;
#endif
        }
      }

      if (write_weights) {
        fprintf(fout, "%f ", write_weights_sum*1.0/write_weights_denom);
      }
    }
    if (write_weights) {
      fprintf(fout, "\n");
    }

#if 0
    if (do_clear) {
      int y_current_clear = y-p->patch_w*2;
      if (y == ann->h-p->patch_w) { y_current_clear = ann->h; }
      if (y_current_clear > y_last_clear) {
        BITMAP *subb = create_sub_bitmap(b, 0, y_last_clear, b->w, y_current_clear-y_last_clear);
#if VOTE_SUM
        BITMAP *subans = knn_norm_image(p, &accum[4*(y_last_clear*ann->w)], ann->w, (y_current_clear-y_last_clear), subb);
#else
        BITMAP *subans = knn_norm_image(p, &accum[3*(y_last_clear*ann->w)], ann->w, (y_current_clear-y_last_clear), subb);
#endif
        destroy_bitmap(subb);
        blit(subans, ans, 0, 0, 0, y_last_clear, subans->w, subans->h);
        destroy_bitmap(subans);
        
#if !VOTE_SUM
        for (int yp = y_last_clear; yp < y_current_clear; yp++) {
          for (int xp = 0; xp < ann->w-p->patch_w+1; xp++) {
            vector<pair<int, int> > *prow = &accum[3*(yp*ann->w+xp)];
            /*prow[0].clear();
            prow[1].clear();
            prow[2].clear();
            prow[0].resize(0);
            prow[1].resize(0);
            prow[2].resize(0);*/
            vector<pair<int, int> >().swap(prow[0]);
            vector<pair<int, int> >().swap(prow[1]);
            vector<pair<int, int> >().swap(prow[2]);
          }
        }
#endif
        y_last_clear = y_current_clear;
      }
    }
#endif
  }
  if (!do_clear) {
    ans = knn_norm_image(p, accum, ann->w, ann->h, b);
  }
  if (!accum_out) {
    delete[] accum;
  }
#if 0
  ACCUM *accum = new ACCUM[4*ann->w*ann->h];
  int nacc = 4*ann->w*ann->h;
  for (int i = 0; i < nacc; i++) { accum[i] = 0; }
  //memset((void *) accum, 0, sizeof(ACCUM)*4*ann->w*ann->h);

  for (int y = 0; y < ann->h-p->patch_w+1; y++) {
    for (int x = 0; x < ann->w-p->patch_w+1; x++) {
      int xp, yp, sp, tp;
      getnn(ann, x, y, xp, yp);
      getnn(ann_sim, x, y, sp, tp);
      XFORM bpos = get_xform(p, xp, yp, sp, tp);
      int bx_row = bpos.x0, by_row = bpos.y0;
      for (int dy = 0; dy < PATCH_W; dy++) {
        int bx = bx_row, by = by_row;
        ACCUM *prow = &accum[4*((y+dy)*ann->w+x)];
        for (int dx = 0; dx < PATCH_W; dx++) {
          ACCUM *ptr = &prow[4*dx];
          int rv, gv, bv;
          getpixel_bilin(b, bx, by, rv, gv, bv);
          ptr[0] += rv;
          ptr[1] += gv;
          ptr[2] += bv;
          ptr[3]++;
          bx += bpos.dxdu;
          by += bpos.dydu;
        }
        bx_row += bpos.dxdv;
        by_row += bpos.dydv;
      }
    }
  }
  
  BITMAP *ans = norm_image(accum, ann->w, ann->h);
  delete[] accum;
  return ans;
#endif
  //fprintf(stderr, "knn_vote unimplemented\n"); exit(1);
  if (write_weights) {
    fclose(fout);
  }

  return ans;
}

BITMAP *knn_vote(Params *p, BITMAP *b,
                 VBMP *ann, VBMP *ann_sim, VBMP *annd, VBMP *bnn, VBMP *bnn_sim,
                 BITMAP *bmask, BITMAP *bweight,
                 double coherence_weight, double complete_weight,
                 RegionMasks *amask, BITMAP *aweight, BITMAP *ainit, RegionMasks *region_masks, BITMAP *aconstraint, int mask_self_only, KNNWeightFunc *weight_func, double **accum_out) {
  if      (p->patch_w == 1) { return knn_vote_n<1,int>(p, b, ann, ann_sim, annd, bnn, bnn_sim, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only, weight_func, accum_out); }
  else if (p->patch_w == 2) { return knn_vote_n<2,int>(p, b, ann, ann_sim, annd, bnn, bnn_sim, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only, weight_func, accum_out); }
  else if (p->patch_w == 3) { return knn_vote_n<3,int>(p, b, ann, ann_sim, annd, bnn, bnn_sim, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only, weight_func, accum_out); }
  else if (p->patch_w == 4) { return knn_vote_n<4,int>(p, b, ann, ann_sim, annd, bnn, bnn_sim, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only, weight_func, accum_out); }
  else if (p->patch_w == 5) { return knn_vote_n<5,int>(p, b, ann, ann_sim, annd, bnn, bnn_sim, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only, weight_func, accum_out); }
  else if (p->patch_w == 6) { return knn_vote_n<6,int>(p, b, ann, ann_sim, annd, bnn, bnn_sim, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only, weight_func, accum_out); }
  else if (p->patch_w == 7) { return knn_vote_n<7,int>(p, b, ann, ann_sim, annd, bnn, bnn_sim, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only, weight_func, accum_out); }
  else if (p->patch_w == 8) { return knn_vote_n<8,int>(p, b, ann, ann_sim, annd, bnn, bnn_sim, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only, weight_func, accum_out); }
  else if (p->patch_w == 9) { return knn_vote_n<9,int>(p, b, ann, ann_sim, annd, bnn, bnn_sim, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only, weight_func, accum_out); }
  else if (p->patch_w == 10) { return knn_vote_n<10,int>(p, b, ann, ann_sim, annd, bnn, bnn_sim, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only, weight_func, accum_out); }
  else if (p->patch_w == 11) { return knn_vote_n<11,int>(p, b, ann, ann_sim, annd, bnn, bnn_sim, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only, weight_func, accum_out); }
  else if (p->patch_w == 12) { return knn_vote_n<12,int>(p, b, ann, ann_sim, annd, bnn, bnn_sim, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only, weight_func, accum_out); }
  else if (p->patch_w == 13) { return knn_vote_n<13,int>(p, b, ann, ann_sim, annd, bnn, bnn_sim, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only, weight_func, accum_out); }
  else if (p->patch_w == 14) { return knn_vote_n<14,int>(p, b, ann, ann_sim, annd, bnn, bnn_sim, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only, weight_func, accum_out); }
  else if (p->patch_w == 15) { return knn_vote_n<15,int>(p, b, ann, ann_sim, annd, bnn, bnn_sim, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only, weight_func, accum_out); }
  else if (p->patch_w == 16) { return knn_vote_n<16,int>(p, b, ann, ann_sim, annd, bnn, bnn_sim, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only, weight_func, accum_out); }
  else if (p->patch_w == 17) { return knn_vote_n<17,int>(p, b, ann, ann_sim, annd, bnn, bnn_sim, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only, weight_func, accum_out); }
  else if (p->patch_w == 18) { return knn_vote_n<18,int>(p, b, ann, ann_sim, annd, bnn, bnn_sim, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only, weight_func, accum_out); }
  else if (p->patch_w == 19) { return knn_vote_n<19,int>(p, b, ann, ann_sim, annd, bnn, bnn_sim, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only, weight_func, accum_out); }
  else if (p->patch_w == 20) { return knn_vote_n<20,int>(p, b, ann, ann_sim, annd, bnn, bnn_sim, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only, weight_func, accum_out); }
  else if (p->patch_w == 21) { return knn_vote_n<21,int>(p, b, ann, ann_sim, annd, bnn, bnn_sim, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only, weight_func, accum_out); }
  else if (p->patch_w == 22) { return knn_vote_n<22,int>(p, b, ann, ann_sim, annd, bnn, bnn_sim, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only, weight_func, accum_out); }
  else if (p->patch_w == 23) { return knn_vote_n<23,int>(p, b, ann, ann_sim, annd, bnn, bnn_sim, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only, weight_func, accum_out); }
  else if (p->patch_w == 24) { return knn_vote_n<24,int>(p, b, ann, ann_sim, annd, bnn, bnn_sim, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only, weight_func, accum_out); }
  else if (p->patch_w == 25) { return knn_vote_n<25,int>(p, b, ann, ann_sim, annd, bnn, bnn_sim, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only, weight_func, accum_out); }
  else if (p->patch_w == 26) { return knn_vote_n<26,int>(p, b, ann, ann_sim, annd, bnn, bnn_sim, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only, weight_func, accum_out); }
  else if (p->patch_w == 27) { return knn_vote_n<27,int>(p, b, ann, ann_sim, annd, bnn, bnn_sim, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only, weight_func, accum_out); }
  else if (p->patch_w == 28) { return knn_vote_n<28,int>(p, b, ann, ann_sim, annd, bnn, bnn_sim, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only, weight_func, accum_out); }
  else if (p->patch_w == 29) { return knn_vote_n<29,int>(p, b, ann, ann_sim, annd, bnn, bnn_sim, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only, weight_func, accum_out); }
  else if (p->patch_w == 30) { return knn_vote_n<30,int>(p, b, ann, ann_sim, annd, bnn, bnn_sim, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only, weight_func, accum_out); }
  else if (p->patch_w == 31) { return knn_vote_n<31,int>(p, b, ann, ann_sim, annd, bnn, bnn_sim, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only, weight_func, accum_out); }
  else if (p->patch_w == 32) { return knn_vote_n<32,int>(p, b, ann, ann_sim, annd, bnn, bnn_sim, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only, weight_func, accum_out); }
  else { fprintf(stderr, "Patch size unsupported for knn_vote: %d\n", p->patch_w); exit(1); }
}

#define ENRICH_HASHSET 1

KNNSolverWeightFunc::KNNSolverWeightFunc(double x[3]) {
  param[0] = x[0];
  param[1] = x[1];
  param[2] = x[2];
}

double KNNSolverWeightFunc::weight(double d, int is_center) {
  if (is_center) { /*return 0;*/ return exp(param[2]); }
  return exp(d*d*param[0]+d*param[1]);
}

//double psnr(BITMAP *a, BITMAP *b);

class KNNSolver: public ObjectiveFunc { public:
  Params *p;
  BITMAP *b;
  VBMP *ann;
  VBMP *ann_sim;
  VBMP *annd;
  int n;
  BITMAP *aorig;
  KNNSolver(Params *p_, BITMAP *b_, VBMP *ann_, VBMP *ann_sim_, VBMP *annd_, int n_, BITMAP *aorig_) {
    p = p_;
    b = b_;
    ann = ann_;
    ann_sim = ann_sim_;
    annd = annd_;
    n = n_;
    aorig = aorig_;
  }
  virtual double f(double x[]) {
    //BITMAP *a = knn_vote(p, a, ann, ann_sim, annd, NULL, NULL, 0, 0, 0.5, 0.5, weight_func);
    KNNSolverWeightFunc weight_func(x);
    BITMAP *a = knn_vote(p, b, ann, ann_sim, annd, NULL, NULL, NULL, NULL, COHERENCE_WEIGHT, COMPLETE_WEIGHT, NULL, NULL, NULL, NULL, NULL, 0, &weight_func);
    if (a->w != aorig->w || a->h != aorig->h) { fprintf(stderr, "Difference in dimensions\n"); exit(1); }
    /*
    long long ans = 0;
//#pragma omp parallel for schedule(static, 8)
    for (int y = 0; y < a->h; y++) {
      long long subans = 0;
      int *row1 = (int *) a->line[y];
      int *row2 = (int *) aorig->line[y];
      for (int x = 0; x < a->w; x++) {
        int c1 = row1[x];
        int c2 = row2[x];
        int dr = (c1&255)-(c2&255);
        int dg = ((c1>>8)&255)-((c2>>8)&255);
        int db = (c1>>16)-(c2>>16);
        subans += dr*dr+dg*dg+db*db;
      }
//#pragma omp atomic
      ans += subans;
    }
    return ans;
    */
    double ans = 0.0; //-psnr(a, aorig);
    destroy_bitmap(a);
    return ans;
  }
};

double patsearch(ObjectiveFunc *f, double *x, double *ap, int n, int iters) {
  double f0 = f->f(x);
  for (int i = 0; i < iters; i++) {
    printf("patsearch iter %d, f0=%f\n", i, f0);
    for (int j = 0; j < n; j++) {
      double xorig = x[j];
      x[j] += ap[j];
      double fp = f->f(x);
      if (fp < f0) {
        f0 = fp;
        ap[j] *= 2;
        continue;
      }
      x[j] = xorig - ap[j];
      fp = f->f(x);
      if (fp < f0) {
        f0 = fp;
        ap[j] *= 2;
        continue;
      }
      ap[j] *= 0.5;
      x[j] = xorig;
    }
  }
  return f0;
}

BITMAP *knn_vote_solve(Params *p, BITMAP *b,
                 VBMP *ann, VBMP *ann_sim, VBMP *annd, int n, BITMAP *aorig, double weight_out[3]) {
  KNNSolver f(p, b, ann, ann_sim, annd, n, aorig);
  double x[3] = { 0, 0, 0 }; //{ 0, 0, 0 };
  double ap0[3] = { 1e-6, 1e-4, 1 };
  double ap[3] = { ap0[0], ap0[1], ap0[2] };
  double fval = patsearch(&f, x, ap, 3, 50);
  printf("fval: %f at %e, %e, %e\n", fval, x[0], x[1], x[2]);
  double ap_prime[3] = { ap0[0], ap0[1], ap0[2] };
  fval = patsearch(&f, x, ap_prime, 3, 20);
  printf("fval: %f at %e, %e, %e\n", fval, x[0], x[1], x[2]);
  //printf("stopping\n");
  //exit(1);
  KNNSolverWeightFunc weight_func(x);
  //weight_func.param[2] = -1000;
  write_weights = 1;
  BITMAP *a = knn_vote(p, b, ann, ann_sim, annd, NULL, NULL, NULL, NULL, COHERENCE_WEIGHT, COMPLETE_WEIGHT, NULL, NULL, NULL, NULL, NULL, 0, &weight_func);
  write_weights = 0;
  weight_out[0] = x[0];
  weight_out[1] = x[1];
  weight_out[2] = x[2];
  return a;
}

int enrich_ok = 0;

template<int PATCH_W>
void knn_enrich_n(Params *p, BITMAP *a, BITMAP *b, VBMP *ann, VBMP *ann_sim, VBMP *annd, PRINCIPAL_ANGLE *pa) {
  double enrich_start_t = accurate_timer();
  if (a != b) { fprintf(stderr, "knn_enrich unimplemented for a != b\n"); exit(1); }
#if !TRANSLATE_ONLY
  fprintf(stderr, "knn_enrich unimplemented for rotation+scale mode\n"); exit(1);
#endif
  fprintf(stderr, "knn_enrich_n, O(k)=%d\n", enrich_ok);
  
  Box box = get_abox(p, a, NULL);
  #pragma omp parallel for schedule(dynamic, 4)
  for (int y = box.ymin; y < box.ymax; y++) {
    int adata[PATCH_W*PATCH_W];
    vector<qtype<int> > v;
    v.reserve(p->knn+1);
#if ENRICH_HASHSET
    //int *annL = new int[p->knn*(p->knn+1)];
#endif
    for (int x = box.xmin; x < box.xmax; x++) {
      //tried.clear();
      for (int dy0 = 0; dy0 < PATCH_W; dy0++) {
        int *drow = ((int *) a->line[y+dy0])+x;
        int *adata_row = adata+(dy0*PATCH_W);
        for (int dx0 = 0; dx0 < PATCH_W; dx0++) {
          adata_row[dx0] = drow[dx0];
        }
      }

      int *p_ann = ann->get(x, y);
      int *p_annd = annd->get(x, y);
#if !TRANSLATE_ONLY
      int *p_ann_sim = ann_sim->get(x, y);
#endif

      PositionSet pos(p_ann, &v, p->knn);
#if ENRICH_HASHSET
      LargeKPositionSet tried(NULL, NULL, enrich_ok ? p->knn*2: p->knn*p->knn+p->knn);
      int annN = 0;
#endif
      v.clear();
      for (int i = 0; i < p->knn; i++) {
#if !TRANSLATE_ONLY
        v.push_back(qtype<int>(p_annd[i], p_ann[i], p_ann_sim[i]));
#else
        v.push_back(qtype<int>(p_annd[i], p_ann[i]));
#endif
        pos.insert_nonexistent(INT_TO_X(p_ann[i]), INT_TO_Y(p_ann[i]));
#if ENRICH_HASHSET
        tried.insert_nonexistent(INT_TO_X(p_ann[i]), INT_TO_Y(p_ann[i]));
        //annL[i] = p_ann[i];
#endif
      }
#if ENRICH_HASHSET
      annN = p->knn;
#endif
      int nk0 = p->knn;
      if (enrich_ok) { nk0 = 1; }
      //int ncand = int(sqrt(double(p->knn)));
      for (int j0 = 0; j0 < /*ncand*/ p->knn; j0++) {
        int j = j0;
        if (enrich_ok) { j = rand()%p->knn; }
        int xp = INT_TO_X(p_ann[j]), yp = INT_TO_Y(p_ann[j]);
        int *q_ann = ann->get(xp, yp);
        int *q_annd = annd->get(xp, yp);
#if !TRANSLATE_ONLY
        int *q_ann_sim = ann_sim->get(xp, yp);
#endif
        for (int k0 = 0; k0 < /*ncand*/ nk0; k0++) {
          int k = k0;
          if (enrich_ok) { k = rand()%p->knn; }
          int xpp = INT_TO_X(q_ann[k]), ypp = INT_TO_Y(q_ann[k]);
#if ENRICH_HASHSET
          int inserted = tried.try_insert(xpp, ypp, annN);
          if (!inserted) { continue; }
          //annL[annN++] = XY_TO_INT(xpp, ypp);
#endif
#if !TRANSLATE_ONLY
          int spp = INT_TO_X(q_ann_sim[k]), tpp = INT_TO_Y(q_ann_sim[k]);
          /*if (USE_PA) {
            tpp = (get_principal_angle(p, pa, xpp, ypp, spp)-angle0)&(NUM_ANGLES-1);
          }*/
          //xpp -= dx;
          XFORM bpos = get_xform(p, xpp, ypp, spp, tpp);
//          xpp -= (bpos.dxdu*dx)>>16;
//          ypp -= (bpos.dydu*dx)>>16;
          bpos = get_xform(p, xpp, ypp, spp, tpp);
#else
          XFORM bpos;
          bpos.x0 = xpp<<16;
          bpos.y0 = ypp<<16;
          int spp = 0, tpp = 0;
#endif
          knn_attempt_n<PATCH_W, 0>(v, adata, b, bpos, xpp, ypp, spp, tpp, p, 0, pos);
        }
      }

      for (int i = 0; i < p->knn; i++) {
        p_annd[i] = v[i].a;
        p_ann[i] = v[i].b;
#if !TRANSLATE_ONLY
        p_ann_sim[i] = v[i].c;
#endif
      }
    }
#if ENRICH_HASHSET
    //delete[] annL;
#endif
  }
  fprintf(stderr, "enrich time: %f secs\n", accurate_timer()-enrich_start_t);
}

void knn_enrich(Params *p, BITMAP *a, BITMAP *b, VBMP *ann, VBMP *ann_sim, VBMP *annd, PRINCIPAL_ANGLE *pa) {
  if      (p->patch_w == 1 ) { return knn_enrich_n<1 >(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 2 ) { return knn_enrich_n<2 >(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 3 ) { return knn_enrich_n<3 >(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 4 ) { return knn_enrich_n<4 >(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 5 ) { return knn_enrich_n<5 >(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 6 ) { return knn_enrich_n<6 >(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 7 ) { return knn_enrich_n<7 >(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 8 ) { return knn_enrich_n<8 >(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 9 ) { return knn_enrich_n<9 >(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 10) { return knn_enrich_n<10>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 11) { return knn_enrich_n<11>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 12) { return knn_enrich_n<12>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 13) { return knn_enrich_n<13>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 14) { return knn_enrich_n<14>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 15) { return knn_enrich_n<15>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 16) { return knn_enrich_n<16>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 17) { return knn_enrich_n<17>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 18) { return knn_enrich_n<18>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 19) { return knn_enrich_n<19>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 20) { return knn_enrich_n<20>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 21) { return knn_enrich_n<21>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 22) { return knn_enrich_n<22>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 23) { return knn_enrich_n<23>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 24) { return knn_enrich_n<24>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 25) { return knn_enrich_n<25>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 26) { return knn_enrich_n<26>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 27) { return knn_enrich_n<27>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 28) { return knn_enrich_n<28>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 29) { return knn_enrich_n<29>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 30) { return knn_enrich_n<30>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 31) { return knn_enrich_n<31>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 32) { return knn_enrich_n<32>(p, a, b, ann, ann_sim, annd, pa); }
  else { fprintf(stderr, "Patch width unsupported: %d\n", p->patch_w); exit(1); }
}


template<int PATCH_W>
void knn_enrich3_n(Params *p, BITMAP *a, BITMAP *b, VBMP *ann, VBMP *ann_sim, VBMP *annd, PRINCIPAL_ANGLE *pa) {
  if (a != b) { fprintf(stderr, "knn_enrich unimplemented for a != b\n"); exit(1); }
#if !TRANSLATE_ONLY
  fprintf(stderr, "knn_enrich unimplemented for rotation+scale mode\n"); exit(1);
#endif
  fprintf(stderr, "enrich3, O(k)=%d\n", enrich_ok);
  
  Box box = get_abox(p, a, NULL);
  #pragma omp parallel for schedule(dynamic, 4)
  for (int y = box.ymin; y < box.ymax; y++) {
    int adata[PATCH_W*PATCH_W];
    vector<qtype<int> > v;
    v.reserve(p->knn+1);
#if ENRICH_HASHSET
    int *annL = new int[p->knn*p->knn*(p->knn+1)];
#endif
    for (int x = box.xmin; x < box.xmax; x++) {
      for (int dy0 = 0; dy0 < PATCH_W; dy0++) {
        int *drow = ((int *) a->line[y+dy0])+x;
        int *adata_row = adata+(dy0*PATCH_W);
        for (int dx0 = 0; dx0 < PATCH_W; dx0++) {
          adata_row[dx0] = drow[dx0];
        }
      }

      int *p_ann = ann->get(x, y);
      int *p_annd = annd->get(x, y);
#if !TRANSLATE_ONLY
      int *p_ann_sim = ann_sim->get(x, y);
#endif

      PositionSet pos(p_ann, &v, p->knn);
#if ENRICH_HASHSET
      LargeKPositionSet tried(annL, NULL, enrich_ok ? p->knn*2: p->knn*p->knn*p->knn);
      int annN = 0;
#endif
      v.clear();
      for (int i = 0; i < p->knn; i++) {
#if !TRANSLATE_ONLY
        v.push_back(qtype<int>(p_annd[i], p_ann[i], p_ann_sim[i]));
#else
        v.push_back(qtype<int>(p_annd[i], p_ann[i]));
#endif
        pos.insert_nonexistent(INT_TO_X(p_ann[i]), INT_TO_Y(p_ann[i]));
#if ENRICH_HASHSET
        tried.insert_nonexistent(INT_TO_X(p_ann[i]), INT_TO_Y(p_ann[i]));
        annL[i] = p_ann[i];
#endif
      }
#if ENRICH_HASHSET
      annN = p->knn;
#endif

      int n_inner = p->knn;
      if (enrich_ok) { n_inner = 1; }
      for (int j0 = 0; j0 < p->knn; j0++) {
        int j = j0;
        if (enrich_ok) { j = rand()%p->knn; }
        int xp = INT_TO_X(p_ann[j]), yp = INT_TO_Y(p_ann[j]);
        int *m_ann = ann->get(xp, yp);
        int *m_annd = annd->get(xp, yp);
#if !TRANSLATE_ONLY
        int *m_ann_sim = ann_sim->get(xp, yp);
#endif
        for (int m0 = 0; m0 < n_inner; m0++) {
          int m = m0;
          if (enrich_ok) { m = rand()%p->knn; }
          int xm = INT_TO_X(m_ann[m]), ym = INT_TO_Y(m_ann[m]);
          int *q_ann = ann->get(xm, ym);
          int *q_annd = annd->get(xm, ym);
#if !TRANSLATE_ONLY
          int *q_ann_sim = ann_sim->get(xm, ym);
#endif
        for (int k0 = 0; k0 < n_inner; k0++) {
          int k = k0;
          if (enrich_ok) { k = rand()%p->knn; }
          int xpp = INT_TO_X(q_ann[k]), ypp = INT_TO_Y(q_ann[k]);
#if ENRICH_HASHSET
          int inserted = tried.try_insert(xpp, ypp, annN);
          if (!inserted) { continue; }
          annL[annN++] = XY_TO_INT(xpp, ypp);
#endif
#if !TRANSLATE_ONLY
          int spp = INT_TO_X(q_ann_sim[k]), tpp = INT_TO_Y(q_ann_sim[k]);
          /*if (USE_PA) {
            tpp = (get_principal_angle(p, pa, xpp, ypp, spp)-angle0)&(NUM_ANGLES-1);
          }*/
          //xpp -= dx;
          XFORM bpos = get_xform(p, xpp, ypp, spp, tpp);
//          xpp -= (bpos.dxdu*dx)>>16;
//          ypp -= (bpos.dydu*dx)>>16;
          bpos = get_xform(p, xpp, ypp, spp, tpp);
#else
          XFORM bpos;
          bpos.x0 = xpp<<16;
          bpos.y0 = ypp<<16;
          int spp = 0, tpp = 0;
#endif
          knn_attempt_n<PATCH_W, 0>(v, adata, b, bpos, xpp, ypp, spp, tpp, p, 0, pos);
        }
        }
      }

      for (int i = 0; i < p->knn; i++) {
        p_annd[i] = v[i].a;
        p_ann[i] = v[i].b;
#if !TRANSLATE_ONLY
        p_ann_sim[i] = v[i].c;
#endif
      }
    }
#if ENRICH_HASHSET
    delete[] annL;
#endif
  }
}

void knn_enrich3(Params *p, BITMAP *a, BITMAP *b, VBMP *ann, VBMP *ann_sim, VBMP *annd, PRINCIPAL_ANGLE *pa) {
  if      (p->patch_w == 1 ) { return knn_enrich3_n<1 >(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 2 ) { return knn_enrich3_n<2 >(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 3 ) { return knn_enrich3_n<3 >(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 4 ) { return knn_enrich3_n<4 >(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 5 ) { return knn_enrich3_n<5 >(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 6 ) { return knn_enrich3_n<6 >(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 7 ) { return knn_enrich3_n<7 >(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 8 ) { return knn_enrich3_n<8 >(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 9 ) { return knn_enrich3_n<9 >(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 10) { return knn_enrich3_n<10>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 11) { return knn_enrich3_n<11>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 12) { return knn_enrich3_n<12>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 13) { return knn_enrich3_n<13>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 14) { return knn_enrich3_n<14>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 15) { return knn_enrich3_n<15>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 16) { return knn_enrich3_n<16>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 17) { return knn_enrich3_n<17>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 18) { return knn_enrich3_n<18>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 19) { return knn_enrich3_n<19>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 20) { return knn_enrich3_n<20>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 21) { return knn_enrich3_n<21>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 22) { return knn_enrich3_n<22>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 23) { return knn_enrich3_n<23>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 24) { return knn_enrich3_n<24>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 25) { return knn_enrich3_n<25>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 26) { return knn_enrich3_n<26>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 27) { return knn_enrich3_n<27>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 28) { return knn_enrich3_n<28>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 29) { return knn_enrich3_n<29>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 30) { return knn_enrich3_n<30>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 31) { return knn_enrich3_n<31>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 32) { return knn_enrich3_n<32>(p, a, b, ann, ann_sim, annd, pa); }
  else { fprintf(stderr, "Patch width unsupported: %d\n", p->patch_w); exit(1); }
}

template<int PATCH_W>
void knn_enrich4_n(Params *p, BITMAP *a, BITMAP *b, VBMP *ann, VBMP *ann_sim, VBMP *annd, PRINCIPAL_ANGLE *pa) {
  if (a != b) { fprintf(stderr, "knn_enrich unimplemented for a != b\n"); exit(1); }
#if !TRANSLATE_ONLY
  fprintf(stderr, "knn_enrich unimplemented for rotation+scale mode\n"); exit(1);
#endif
  fprintf(stderr, "enrich4, O(k)=%d\n", enrich_ok);
  
  Box box = get_abox(p, a, NULL);
  #pragma omp parallel for schedule(dynamic, 4)
  for (int y = box.ymin; y < box.ymax; y++) {
    int adata[PATCH_W*PATCH_W];
    vector<qtype<int> > v;
    v.reserve(p->knn+1);
#if ENRICH_HASHSET
    int *annL = new int[p->knn*p->knn*(p->knn+1)];
#endif
    for (int x = box.xmin; x < box.xmax; x++) {
      for (int dy0 = 0; dy0 < PATCH_W; dy0++) {
        int *drow = ((int *) a->line[y+dy0])+x;
        int *adata_row = adata+(dy0*PATCH_W);
        for (int dx0 = 0; dx0 < PATCH_W; dx0++) {
          adata_row[dx0] = drow[dx0];
        }
      }

      int *p_ann = ann->get(x, y);
      int *p_annd = annd->get(x, y);
#if !TRANSLATE_ONLY
      int *p_ann_sim = ann_sim->get(x, y);
#endif

      PositionSet pos(p_ann, &v, p->knn);
#if ENRICH_HASHSET
      LargeKPositionSet tried(annL, NULL, enrich_ok ? p->knn*2: p->knn*p->knn*p->knn*p->knn);
      int annN = 0;
#endif
      v.clear();
      for (int i = 0; i < p->knn; i++) {
#if !TRANSLATE_ONLY
        v.push_back(qtype<int>(p_annd[i], p_ann[i], p_ann_sim[i]));
#else
        v.push_back(qtype<int>(p_annd[i], p_ann[i]));
#endif
        pos.insert_nonexistent(INT_TO_X(p_ann[i]), INT_TO_Y(p_ann[i]));
#if ENRICH_HASHSET
        tried.insert_nonexistent(INT_TO_X(p_ann[i]), INT_TO_Y(p_ann[i]));
        annL[i] = p_ann[i];
#endif
      }
#if ENRICH_HASHSET
      annN = p->knn;
#endif

      int n_inner = p->knn;
      if (enrich_ok) { n_inner = 1; }
      for (int j0 = 0; j0 < p->knn; j0++) {
        int j = j0;
        if (enrich_ok) { j = rand()%p->knn; }
        int xp = INT_TO_X(p_ann[j]), yp = INT_TO_Y(p_ann[j]);
        int *m_ann = ann->get(xp, yp);
        int *m_annd = annd->get(xp, yp);
#if !TRANSLATE_ONLY
        int *m_ann_sim = ann_sim->get(xp, yp);
#endif
        for (int m0 = 0; m0 < n_inner; m0++) {
          int m = m0;
          if (enrich_ok) { m = rand()%p->knn; }
          int xm = INT_TO_X(m_ann[m]), ym = INT_TO_Y(m_ann[m]);
          int *q_ann = ann->get(xm, ym);
          int *q_annd = annd->get(xm, ym);
#if !TRANSLATE_ONLY
          int *q_ann_sim = ann_sim->get(xm, ym);
#endif
        for (int z0 = 0; z0 < n_inner; z0++) {
          int z = z0;
          if (enrich_ok) { z = rand()%p->knn; }
          int xz = INT_TO_X(q_ann[z]), yz = INT_TO_Y(q_ann[z]);
          int *z_ann = ann->get(xz, yz);
          int *z_annd = annd->get(xz, yz);
#if !TRANSLATE_ONLY
          int *z_ann_sim = ann_sim->get(xz, yz);
#endif
        for (int k0 = 0; k0 < n_inner; k0++) {
          int k = k0;
          if (enrich_ok) { k = rand()%p->knn; }
          int xpp = INT_TO_X(z_ann[k]), ypp = INT_TO_Y(z_ann[k]);
#if ENRICH_HASHSET
          int inserted = tried.try_insert(xpp, ypp, annN);
          if (!inserted) { continue; }
          annL[annN++] = XY_TO_INT(xpp, ypp);
#endif
#if !TRANSLATE_ONLY
          int spp = INT_TO_X(z_ann_sim[k]), tpp = INT_TO_Y(z_ann_sim[k]);
          /*if (USE_PA) {
            tpp = (get_principal_angle(p, pa, xpp, ypp, spp)-angle0)&(NUM_ANGLES-1);
          }*/
          //xpp -= dx;
          XFORM bpos = get_xform(p, xpp, ypp, spp, tpp);
//          xpp -= (bpos.dxdu*dx)>>16;
//          ypp -= (bpos.dydu*dx)>>16;
          bpos = get_xform(p, xpp, ypp, spp, tpp);
#else
          XFORM bpos;
          bpos.x0 = xpp<<16;
          bpos.y0 = ypp<<16;
          int spp = 0, tpp = 0;
#endif
          knn_attempt_n<PATCH_W, 0>(v, adata, b, bpos, xpp, ypp, spp, tpp, p, 0, pos);
        }
        }
        }
      }

      for (int i = 0; i < p->knn; i++) {
        p_annd[i] = v[i].a;
        p_ann[i] = v[i].b;
#if !TRANSLATE_ONLY
        p_ann_sim[i] = v[i].c;
#endif
      }
    }
#if ENRICH_HASHSET
    delete[] annL;
#endif
  }
}

void knn_enrich4(Params *p, BITMAP *a, BITMAP *b, VBMP *ann, VBMP *ann_sim, VBMP *annd, PRINCIPAL_ANGLE *pa) {
  if      (p->patch_w == 1 ) { return knn_enrich4_n<1 >(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 2 ) { return knn_enrich4_n<2 >(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 3 ) { return knn_enrich4_n<3 >(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 4 ) { return knn_enrich4_n<4 >(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 5 ) { return knn_enrich4_n<5 >(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 6 ) { return knn_enrich4_n<6 >(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 7 ) { return knn_enrich4_n<7 >(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 8 ) { return knn_enrich4_n<8 >(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 9 ) { return knn_enrich4_n<9 >(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 10) { return knn_enrich4_n<10>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 11) { return knn_enrich4_n<11>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 12) { return knn_enrich4_n<12>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 13) { return knn_enrich4_n<13>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 14) { return knn_enrich4_n<14>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 15) { return knn_enrich4_n<15>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 16) { return knn_enrich4_n<16>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 17) { return knn_enrich4_n<17>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 18) { return knn_enrich4_n<18>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 19) { return knn_enrich4_n<19>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 20) { return knn_enrich4_n<20>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 21) { return knn_enrich4_n<21>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 22) { return knn_enrich4_n<22>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 23) { return knn_enrich4_n<23>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 24) { return knn_enrich4_n<24>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 25) { return knn_enrich4_n<25>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 26) { return knn_enrich4_n<26>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 27) { return knn_enrich4_n<27>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 28) { return knn_enrich4_n<28>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 29) { return knn_enrich4_n<29>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 30) { return knn_enrich4_n<30>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 31) { return knn_enrich4_n<31>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 32) { return knn_enrich4_n<32>(p, a, b, ann, ann_sim, annd, pa); }
  else { fprintf(stderr, "Patch width unsupported: %d\n", p->patch_w); exit(1); }
}

class Link { public:
  int pos;
  int dval;
  Link *next;
};

template<int PATCH_W>
void knn_inverse_enrich_n(Params *p, BITMAP *a, BITMAP *b, VBMP *ann, VBMP *ann_sim, VBMP *annd, PRINCIPAL_ANGLE *pa) {
  if (a != b) { fprintf(stderr, "knn_inverse_enrich unimplemented for a != b\n"); exit(1); }
  double start_t = accurate_timer();
#if !TRANSLATE_ONLY
  fprintf(stderr, "knn_inverse_enrich unimplemented for rotation+scale mode\n"); exit(1);
#endif
  Link *links = new Link[p->knn*a->w*a->h];
  Link **start = new Link *[a->w*a->h];
  Link *links0 = links;
  for (int i = 0; i < a->w*a->h; i++) { start[i] = NULL; }

  printf("build LL\n");
  Box box = get_abox(p, a, NULL);
  //#pragma omp parallel for schedule(dynamic, 4)
  for (int y = box.ymin; y < box.ymax; y++) {
//    int adata[PATCH_W*PATCH_W];
    vector<qtype<int> > v;
    v.reserve(p->knn+1);
/*
#if ENRICH_HASHSET
    int *annL = new int[p->knn*(p->knn+1)];
#endif
*/
    for (int x = box.xmin; x < box.xmax; x++) {
      int isrc = XY_TO_INT(x, y);
      int *p_ann = ann->get(x, y);
      int *p_annd = annd->get(x, y);
      for (int j = 0; j < p->knn; j++) {
        int xp = INT_TO_X(p_ann[j]), yp = INT_TO_Y(p_ann[j]);
        int idest = yp*a->w+xp;
        links->pos = isrc;
        links->next = start[idest];
        links->dval = p_annd[j];
        start[idest] = links;
        links++;
      }
    }
  }

  printf("done LL (%f secs)\n", accurate_timer()-start_t);
  #pragma omp parallel for schedule(dynamic, 4)
  for (int y = box.ymin; y < box.ymax; y++) {
    int adata[PATCH_W*PATCH_W];
    vector<qtype<int> > v;
    v.reserve(p->knn+1);
/*
#if ENRICH_HASHSET
    int *annL = new int[p->knn*(p->knn+1)];
#endif
*/
    for (int x = box.xmin; x < box.xmax; x++) {
      for (int dy0 = 0; dy0 < PATCH_W; dy0++) {
        int *drow = ((int *) a->line[y+dy0])+x;
        int *adata_row = adata+(dy0*PATCH_W);
        for (int dx0 = 0; dx0 < PATCH_W; dx0++) {
          adata_row[dx0] = drow[dx0];
        }
      }

      int *p_ann = ann->get(x, y);
      int *p_annd = annd->get(x, y);
#if !TRANSLATE_ONLY
      int *p_ann_sim = ann_sim->get(x, y);
#endif

      PositionSet pos(p_ann, &v, p->knn);
/*
#if ENRICH_HASHSET
      PositionSet tried(annL, NULL);
      int annN = 0;
#endif
*/
      v.clear();
      for (int i = 0; i < p->knn; i++) {
#if !TRANSLATE_ONLY
        v.push_back(qtype<int>(p_annd[i], p_ann[i], p_ann_sim[i]));
#else
        v.push_back(qtype<int>(p_annd[i], p_ann[i]));
#endif
        pos.insert_nonexistent(INT_TO_X(p_ann[i]), INT_TO_Y(p_ann[i]));
/*
#if ENRICH_HASHSET
        tried.insert_nonexistent(INT_TO_X(p_ann[i]), INT_TO_Y(p_ann[i]));
        annL[i] = p_ann[i];
#endif
*/
      }
/*
#if ENRICH_HASHSET
      annN = p->knn;
#endif
*/

      Link *current = start[y*a->w+x];
      while (current) {
        int xpp = INT_TO_X(current->pos), ypp = INT_TO_Y(current->pos);
/*
#if ENRICH_HASHSET
        int inserted = tried.try_insert(xpp, ypp, annN);
        if (!inserted) { current = current->next; continue; }
        annL[annN++] = XY_TO_INT(xpp, ypp);
#endif
*/
#if !TRANSLATE_ONLY
        fprintf(stderr, "Rotation mode not implemented\n"); exit(1);
        int spp = SCALE_UNITY, tpp = 0;
        //int spp = INT_TO_X(q_ann_sim[k]), tpp = INT_TO_Y(q_ann_sim[k]);
        /*if (USE_PA) {
          tpp = (get_principal_angle(p, pa, xpp, ypp, spp)-angle0)&(NUM_ANGLES-1);
        }*/
        //xpp -= dx;
        XFORM bpos = get_xform(p, xpp, ypp, spp, tpp);
        //xpp -= (bpos.dxdu*dx)>>16;
        //ypp -= (bpos.dydu*dx)>>16;
        bpos = get_xform(p, xpp, ypp, spp, tpp);
#else
        XFORM bpos;
        bpos.x0 = xpp<<16;
        bpos.y0 = ypp<<16;
        int spp = 0, tpp = 0;
#endif
        knn_attempt_n<PATCH_W, 1>(v, adata, b, bpos, xpp, ypp, spp, tpp, p, current->dval, pos);
        current = current->next;
      }

      for (int i = 0; i < p->knn; i++) {
        p_annd[i] = v[i].a;
        p_ann[i] = v[i].b;
#if !TRANSLATE_ONLY
        p_ann_sim[i] = v[i].c;
#endif
      }
    }
/*
#if ENRICH_HASHSET
    delete[] annL;
#endif
*/
  }
  delete[] links0;
  delete[] start;
  fprintf(stderr, "inverse_enrich time: %f secs\n", accurate_timer()-start_t);
  //knn_check(p, a, b, ann, ann_sim, annd);
}

void knn_inverse_enrich(Params *p, BITMAP *a, BITMAP *b, VBMP *ann, VBMP *ann_sim, VBMP *annd, PRINCIPAL_ANGLE *pa) {
  if      (p->patch_w == 1 ) { return knn_inverse_enrich_n<1 >(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 2 ) { return knn_inverse_enrich_n<2 >(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 3 ) { return knn_inverse_enrich_n<3 >(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 4 ) { return knn_inverse_enrich_n<4 >(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 5 ) { return knn_inverse_enrich_n<5 >(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 6 ) { return knn_inverse_enrich_n<6 >(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 7 ) { return knn_inverse_enrich_n<7 >(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 8 ) { return knn_inverse_enrich_n<8 >(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 9 ) { return knn_inverse_enrich_n<9 >(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 10) { return knn_inverse_enrich_n<10>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 11) { return knn_inverse_enrich_n<11>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 12) { return knn_inverse_enrich_n<12>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 13) { return knn_inverse_enrich_n<13>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 14) { return knn_inverse_enrich_n<14>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 15) { return knn_inverse_enrich_n<15>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 16) { return knn_inverse_enrich_n<16>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 17) { return knn_inverse_enrich_n<17>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 18) { return knn_inverse_enrich_n<18>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 19) { return knn_inverse_enrich_n<19>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 20) { return knn_inverse_enrich_n<20>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 21) { return knn_inverse_enrich_n<21>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 22) { return knn_inverse_enrich_n<22>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 23) { return knn_inverse_enrich_n<23>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 24) { return knn_inverse_enrich_n<24>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 25) { return knn_inverse_enrich_n<25>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 26) { return knn_inverse_enrich_n<26>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 27) { return knn_inverse_enrich_n<27>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 28) { return knn_inverse_enrich_n<28>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 29) { return knn_inverse_enrich_n<29>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 30) { return knn_inverse_enrich_n<30>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 31) { return knn_inverse_enrich_n<31>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 32) { return knn_inverse_enrich_n<32>(p, a, b, ann, ann_sim, annd, pa); }
  else { fprintf(stderr, "Patch width unsupported: %d\n", p->patch_w); exit(1); }
}

class LinkNoDist { public:
  int pos;
  LinkNoDist *next;
};

template<int PATCH_W>
void knn_inverse_enrich2_n(Params *p, BITMAP *a, BITMAP *b, VBMP *ann, VBMP *ann_sim, VBMP *annd, PRINCIPAL_ANGLE *pa) {
  if (a != b) { fprintf(stderr, "knn_inverse_enrich2 unimplemented for a != b\n"); exit(1); }
#if !TRANSLATE_ONLY
  fprintf(stderr, "knn_inverse_enrich2 unimplemented for rotation+scale mode\n"); exit(1);
#endif
  fprintf(stderr, "inverse_enrich2\n");
  LinkNoDist *links = new LinkNoDist[p->knn*(p->knn+1)*a->w*a->h];
  LinkNoDist **start = new LinkNoDist *[a->w*a->h];
  LinkNoDist *links0 = links;
  for (int i = 0; i < a->w*a->h; i++) { start[i] = NULL; }

  printf("build LL\n");
  Box box = get_abox(p, a, NULL);
  //#pragma omp parallel for schedule(dynamic, 4)
  for (int y = box.ymin; y < box.ymax; y++) {
//    int adata[PATCH_W*PATCH_W];
    vector<qtype<int> > v;
    v.reserve(p->knn+1);
#if ENRICH_HASHSET
    int *annL = new int[p->knn*(p->knn+1)];
#endif
    for (int x = box.xmin; x < box.xmax; x++) {
      int isrc = XY_TO_INT(x, y);
      int *p_ann = ann->get(x, y);
      //int *p_annd = annd->get(x, y);
      for (int j = 0; j < p->knn; j++) {
        int xp = INT_TO_X(p_ann[j]), yp = INT_TO_Y(p_ann[j]);
        int *q_ann = ann->get(xp, yp);
        //int *q_annd = annd->get(xp, yp);
        {
          int idest = yp*a->w+xp;
          links->pos = isrc;
          links->next = start[idest];
          start[idest] = links;
          links++;
        }
        for (int k = 0; k < p->knn; k++) {
          int xpp = INT_TO_X(q_ann[k]), ypp = INT_TO_Y(q_ann[k]);
          int idest = ypp*a->w+xpp;
          links->pos = isrc;
          links->next = start[idest];
          start[idest] = links;
          links++;
        }
      }
    }
  }

  printf("done LL\n");
  #pragma omp parallel for schedule(dynamic, 4)
  for (int y = box.ymin; y < box.ymax; y++) {
    int adata[PATCH_W*PATCH_W];
    vector<qtype<int> > v;
    v.reserve(p->knn+1);
/*
#if ENRICH_HASHSET
    int *annL = new int[p->knn*(p->knn+1)];
#endif
*/
    for (int x = box.xmin; x < box.xmax; x++) {
      for (int dy0 = 0; dy0 < PATCH_W; dy0++) {
        int *drow = ((int *) a->line[y+dy0])+x;
        int *adata_row = adata+(dy0*PATCH_W);
        for (int dx0 = 0; dx0 < PATCH_W; dx0++) {
          adata_row[dx0] = drow[dx0];
        }
      }

      int *p_ann = ann->get(x, y);
      int *p_annd = annd->get(x, y);
#if !TRANSLATE_ONLY
      int *p_ann_sim = ann_sim->get(x, y);
#endif

      PositionSet pos(p_ann, &v, p->knn);
/*
#if ENRICH_HASHSET
      PositionSet tried(annL, NULL);
      int annN = 0;
#endif
*/
      v.clear();
      for (int i = 0; i < p->knn; i++) {
#if !TRANSLATE_ONLY
        v.push_back(qtype<int>(p_annd[i], p_ann[i], p_ann_sim[i]));
#else
        v.push_back(qtype<int>(p_annd[i], p_ann[i]));
#endif
        pos.insert_nonexistent(INT_TO_X(p_ann[i]), INT_TO_Y(p_ann[i]));
/*
#if ENRICH_HASHSET
        tried.insert_nonexistent(INT_TO_X(p_ann[i]), INT_TO_Y(p_ann[i]));
        annL[i] = p_ann[i];
#endif
*/
      }
/*
#if ENRICH_HASHSET
      annN = p->knn;
#endif
*/

      LinkNoDist *current = start[y*a->w+x];
      while (current) {
        int xpp = INT_TO_X(current->pos), ypp = INT_TO_Y(current->pos);
/*
#if ENRICH_HASHSET
        int inserted = tried.try_insert(xpp, ypp, annN);
        if (!inserted) { current = current->next; continue; }
        annL[annN++] = XY_TO_INT(xpp, ypp);
#endif
*/
#if !TRANSLATE_ONLY
        fprintf(stderr, "Rotation mode not implemented\n"); exit(1);
        int spp = SCALE_UNITY, tpp = 0; //INT_TO_X(q_ann_sim[k]), tpp = INT_TO_Y(q_ann_sim[k]);
        /*if (USE_PA) {
          tpp = (get_principal_angle(p, pa, xpp, ypp, spp)-angle0)&(NUM_ANGLES-1);
        }*/
        //xpp -= dx;
        XFORM bpos = get_xform(p, xpp, ypp, spp, tpp);
        //xpp -= (bpos.dxdu*dx)>>16;
        //ypp -= (bpos.dydu*dx)>>16;
        bpos = get_xform(p, xpp, ypp, spp, tpp);
#else
        XFORM bpos;
        bpos.x0 = xpp<<16;
        bpos.y0 = ypp<<16;
        int spp = 0, tpp = 0;
#endif
        knn_attempt_n<PATCH_W, 0>(v, adata, b, bpos, xpp, ypp, spp, tpp, p, 0, pos);
        current = current->next;
      }

      for (int i = 0; i < p->knn; i++) {
        p_annd[i] = v[i].a;
        p_ann[i] = v[i].b;
#if !TRANSLATE_ONLY
        p_ann_sim[i] = v[i].c;
#endif
      }
    }
/*
#if ENRICH_HASHSET
    delete[] annL;
#endif
*/
  }
  delete[] links0;
  delete[] start;
}

void knn_inverse_enrich2(Params *p, BITMAP *a, BITMAP *b, VBMP *ann, VBMP *ann_sim, VBMP *annd, PRINCIPAL_ANGLE *pa) {
  if      (p->patch_w == 1 ) { return knn_inverse_enrich2_n<1 >(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 2 ) { return knn_inverse_enrich2_n<2 >(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 3 ) { return knn_inverse_enrich2_n<3 >(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 4 ) { return knn_inverse_enrich2_n<4 >(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 5 ) { return knn_inverse_enrich2_n<5 >(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 6 ) { return knn_inverse_enrich2_n<6 >(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 7 ) { return knn_inverse_enrich2_n<7 >(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 8 ) { return knn_inverse_enrich2_n<8 >(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 9 ) { return knn_inverse_enrich2_n<9 >(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 10) { return knn_inverse_enrich2_n<10>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 11) { return knn_inverse_enrich2_n<11>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 12) { return knn_inverse_enrich2_n<12>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 13) { return knn_inverse_enrich2_n<13>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 14) { return knn_inverse_enrich2_n<14>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 15) { return knn_inverse_enrich2_n<15>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 16) { return knn_inverse_enrich2_n<16>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 17) { return knn_inverse_enrich2_n<17>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 18) { return knn_inverse_enrich2_n<18>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 19) { return knn_inverse_enrich2_n<19>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 20) { return knn_inverse_enrich2_n<20>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 21) { return knn_inverse_enrich2_n<21>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 22) { return knn_inverse_enrich2_n<22>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 23) { return knn_inverse_enrich2_n<23>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 24) { return knn_inverse_enrich2_n<24>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 25) { return knn_inverse_enrich2_n<25>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 26) { return knn_inverse_enrich2_n<26>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 27) { return knn_inverse_enrich2_n<27>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 28) { return knn_inverse_enrich2_n<28>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 29) { return knn_inverse_enrich2_n<29>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 30) { return knn_inverse_enrich2_n<30>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 31) { return knn_inverse_enrich2_n<31>(p, a, b, ann, ann_sim, annd, pa); }
  else if (p->patch_w == 32) { return knn_inverse_enrich2_n<32>(p, a, b, ann, ann_sim, annd, pa); }
  else { fprintf(stderr, "Patch width unsupported: %d\n", p->patch_w); exit(1); }
}

template<int PATCH_W>
void knn_check_n(Params *p, BITMAP *a, BITMAP *b, VBMP *ann, VBMP *ann_sim, VBMP *annd, int check_duplicates) {
  //if (a != b) { fprintf(stderr, "knn_check not implemented for a != b\n"); exit(1); }
  int ncheck = 0;
  #pragma omp parallel for schedule(static, 8)
  for (int y = 0; y < AEH; y++) {
    int adata[PATCH_W*PATCH_W];
    int ncheck_row = 0;
    for (int x = 0; x < AEW; x++) {
      for (int dy0 = 0; dy0 < PATCH_W; dy0++) {
        int *drow = ((int *) a->line[y+dy0])+x;
        int *adata_row = adata+(dy0*PATCH_W);
        for (int dx0 = 0; dx0 < PATCH_W; dx0++) {
          adata_row[dx0] = drow[dx0];
        }
      }

      int *p_ann = ann->get(x, y);
      int *p_annd = annd->get(x, y);
#if !TRANSLATE_ONLY
      int *p_ann_sim = ann_sim->get(x, y);
#endif
      hash_set<int> pos;
      for (int i = 0; i < p->knn; i++) {
        if (check_duplicates && pos.count(p_ann[i])) { fprintf(stderr, "duplicate element at %d,%d: %d,%d (d=%d)\n", x, y, INT_TO_X(p_ann[i]), INT_TO_Y(p_ann[i]), p_annd[i]); exit(1); }
        pos.insert(p_ann[i]);
        ncheck_row++;
#if TRANSLATE_ONLY
        int d = fast_patch_dist<PATCH_W, 0>(adata, b, INT_TO_X(p_ann[i]), INT_TO_Y(p_ann[i]), INT_MAX, p);
#else
        int xpp = INT_TO_X(p_ann[i]), ypp = INT_TO_Y(p_ann[i]);
        int spp = INT_TO_X(p_ann_sim[i]), tpp = INT_TO_Y(p_ann_sim[i]);
        XFORM bpos = get_xform(p, xpp, ypp, spp, tpp);
        int d = sim_fast_patch_dist<PATCH_W, 1>(adata, b, bpos, INT_MAX);
#endif
        if (d != p_annd[i]) { fprintf(stderr, "distances not equal at %d,%d (%d): %d, %d\n", x, y, i, d, p_annd[i]); exit(1); }
      }
    }
#pragma omp atomic
    ncheck += ncheck_row;
  }
  fprintf(stderr, "knn_check OK, %d checked\n", ncheck);
}

void knn_check(Params *p, BITMAP *a, BITMAP *b, VBMP *ann, VBMP *ann_sim, VBMP *annd, int check_duplicates) {
  if      (p->patch_w == 1 ) { return knn_check_n<1 >(p, a, b, ann, ann_sim, annd, check_duplicates); }
  else if (p->patch_w == 2 ) { return knn_check_n<2 >(p, a, b, ann, ann_sim, annd, check_duplicates); }
  else if (p->patch_w == 3 ) { return knn_check_n<3 >(p, a, b, ann, ann_sim, annd, check_duplicates); }
  else if (p->patch_w == 4 ) { return knn_check_n<4 >(p, a, b, ann, ann_sim, annd, check_duplicates); }
  else if (p->patch_w == 5 ) { return knn_check_n<5 >(p, a, b, ann, ann_sim, annd, check_duplicates); }
  else if (p->patch_w == 6 ) { return knn_check_n<6 >(p, a, b, ann, ann_sim, annd, check_duplicates); }
  else if (p->patch_w == 7 ) { return knn_check_n<7 >(p, a, b, ann, ann_sim, annd, check_duplicates); }
  else if (p->patch_w == 8 ) { return knn_check_n<8 >(p, a, b, ann, ann_sim, annd, check_duplicates); }
  else if (p->patch_w == 9 ) { return knn_check_n<9 >(p, a, b, ann, ann_sim, annd, check_duplicates); }
  else if (p->patch_w == 10) { return knn_check_n<10>(p, a, b, ann, ann_sim, annd, check_duplicates); }
  else if (p->patch_w == 11) { return knn_check_n<11>(p, a, b, ann, ann_sim, annd, check_duplicates); }
  else if (p->patch_w == 12) { return knn_check_n<12>(p, a, b, ann, ann_sim, annd, check_duplicates); }
  else if (p->patch_w == 13) { return knn_check_n<13>(p, a, b, ann, ann_sim, annd, check_duplicates); }
  else if (p->patch_w == 14) { return knn_check_n<14>(p, a, b, ann, ann_sim, annd, check_duplicates); }
  else if (p->patch_w == 15) { return knn_check_n<15>(p, a, b, ann, ann_sim, annd, check_duplicates); }
  else if (p->patch_w == 16) { return knn_check_n<16>(p, a, b, ann, ann_sim, annd, check_duplicates); }
  else if (p->patch_w == 17) { return knn_check_n<17>(p, a, b, ann, ann_sim, annd, check_duplicates); }
  else if (p->patch_w == 18) { return knn_check_n<18>(p, a, b, ann, ann_sim, annd, check_duplicates); }
  else if (p->patch_w == 19) { return knn_check_n<19>(p, a, b, ann, ann_sim, annd, check_duplicates); }
  else if (p->patch_w == 20) { return knn_check_n<20>(p, a, b, ann, ann_sim, annd, check_duplicates); }
  else if (p->patch_w == 21) { return knn_check_n<21>(p, a, b, ann, ann_sim, annd, check_duplicates); }
  else if (p->patch_w == 22) { return knn_check_n<22>(p, a, b, ann, ann_sim, annd, check_duplicates); }
  else if (p->patch_w == 23) { return knn_check_n<23>(p, a, b, ann, ann_sim, annd, check_duplicates); }
  else if (p->patch_w == 24) { return knn_check_n<24>(p, a, b, ann, ann_sim, annd, check_duplicates); }
  else if (p->patch_w == 25) { return knn_check_n<25>(p, a, b, ann, ann_sim, annd, check_duplicates); }
  else if (p->patch_w == 26) { return knn_check_n<26>(p, a, b, ann, ann_sim, annd, check_duplicates); }
  else if (p->patch_w == 27) { return knn_check_n<27>(p, a, b, ann, ann_sim, annd, check_duplicates); }
  else if (p->patch_w == 28) { return knn_check_n<28>(p, a, b, ann, ann_sim, annd, check_duplicates); }
  else if (p->patch_w == 29) { return knn_check_n<29>(p, a, b, ann, ann_sim, annd, check_duplicates); }
  else if (p->patch_w == 30) { return knn_check_n<30>(p, a, b, ann, ann_sim, annd, check_duplicates); }
  else if (p->patch_w == 31) { return knn_check_n<31>(p, a, b, ann, ann_sim, annd, check_duplicates); }
  else if (p->patch_w == 32) { return knn_check_n<32>(p, a, b, ann, ann_sim, annd, check_duplicates); }
  else { fprintf(stderr, "Patch width unsupported: %d\n", p->patch_w); exit(1); }
}

extern BITMAP *buffer;
extern BITMAP *mouse;

void sort_knn(Params *p, VBMP *ann, VBMP *ann_sim, VBMP *annd) {
#if TRANSLATE_ONLY
  if (ann_sim) { fprintf(stderr, "sort_knn: in mode TRANSLATE_ONLY, expected ann_sim to be NULL\n"); exit(1); }
#endif
#if !TRANSLATE_ONLY
  if (!ann_sim) { fprintf(stderr, "sort_knn: in mode !TRANSLATE_ONLY, expected ann_sim to be given\n"); exit(1); }
#endif

  vector<qtype<int> > v;
  int aew = (ann->w-p->patch_w+1);
  int aeh = (ann->h-p->patch_w+1);
  for (int y = 0; y < aeh; y++) {
    for (int x = 0; x < aew; x++) {
      v.clear();
      int *p_ann = ann->get(x, y);
#if !TRANSLATE_ONLY
      int *p_ann_sim = ann_sim->get(x, y);
#endif
      int *p_annd = annd->get(x, y);

      for (int i = 0; i < p->knn; i++) {
#if !TRANSLATE_ONLY
        v.push_back(qtype<int>(p_annd[i], p_ann[i], p_ann_sim[i]));
#else
        v.push_back(qtype<int>(p_annd[i], p_ann[i]));
#endif
      }
      sort(v.begin(), v.end());

      for (int i = 0; i < p->knn; i++) {
        p_ann[i] = v[i].b;
        p_annd[i] = v[i].a;
#if !TRANSLATE_ONLY
        p_ann_sim[i] = v[i].c;
#endif
      }
    }
  }
}

BITMAP *sample_xform(Params *p, BITMAP *a, XFORM bpos) {
  BITMAP *ans = create_bitmap(p->patch_w, p->patch_w);

  int bx_row = bpos.x0, by_row = bpos.y0;
  for (int y = 0; y < p->patch_w; y++) {
    int bx = bx_row, by = by_row;
    for (int x = 0; x < p->patch_w; x++) {
      int r2, g2, b2;
      getpixel_bilin(a, bx, by, r2, g2, b2);
      ((int *) ans->line[y])[x] = (r2)|(g2<<8)|(b2<<16);

      bx += bpos.dxdu;
      by += bpos.dydu;
    }
    bx_row += bpos.dxdv;
    by_row += bpos.dydv;
  }

  return ans;
}

void knn_vis(Params *p, BITMAP *a, VBMP *ann0, VBMP *ann_sim0, VBMP *annd0, int is_bitmap, BITMAP *vote, BITMAP *orig, BITMAP *vote_uniform) {
  fprintf(stderr, "knn_vis disabled\n"); exit(1);
#if 0
  VBMP *ann = copy_vbmp(ann0);
  VBMP *annd = copy_vbmp(annd0);
#if TRANSLATE_ONLY
  if (ann_sim0) { fprintf(stderr, "in mode TRANSLATE_ONLY, expected ann_sim to be NULL\n"); exit(1); }
  VBMP *ann_sim = ann_sim0;
#else
  VBMP *ann_sim = copy_vbmp(ann_sim0);
#endif
  sort_knn(p, ann, ann_sim, annd);
  int left = 0, top = 0;
  
  double t = accurate_timer();
  if (!buffer) { fprintf(stderr, "No offscreen buffer\n"); exit(1); }
  if (!mouse) { fprintf(stderr, "No mouse cursor loaded\n"); exit(1); }
  int mag = 4;
  text_mode(-1);
  int mz = mouse_z;
  
  BITMAP *buades = NULL; //load_image("flowers_denoise_buades7.bmp");
  
  while (!key[KEY_ESC]) {
   // printf("in knn_vis\n");
    int last_z = mz;
    mz = mouse_z;
    int dz = mz - last_z;
    mag += dz;
    if (mag < 1) { mag = 1; }
    if (mag > 10) { mag = 10; }
    
    double last_time = t;
    t = accurate_timer();
    double dt = t-last_time;
    int s = int(dt*400+0.5);
    if (key[KEY_RIGHT]) { left += s; }
    if (key[KEY_DOWN]) { top += s; }
    if (key[KEY_LEFT]) { left -= s; }
    if (key[KEY_UP]) { top -= s; }
    if (key[KEY_R]) { left = top = 0; }
    
    clear(buffer);
    //blit(a, buffer, 0, 0, 0, 0, a->w, a->h);
    int dovis = 1;
    if (key[KEY_O] && orig) {
      stretch_blit(orig, buffer, 0, 0, a->w, a->h, -left, -top, orig->w*mag, orig->h*mag);
    } else if (key[KEY_B]) {
      dovis = 0;
      if (buades) {
        stretch_blit(buades, buffer, 0, 0, a->w, a->h, -left, -top, buades->w*mag, buades->h*mag);
      }
    } else if (key[KEY_SPACE] && vote) {
      stretch_blit(vote, buffer, 0, 0, a->w, a->h, -left, -top, vote->w*mag, vote->h*mag);
    } else if (key[KEY_U] && vote_uniform) {
      printf("show vote_uniform\n");
      stretch_blit(vote_uniform, buffer, 0, 0, a->w, a->h, -left, -top, vote_uniform->w*mag, vote_uniform->h*mag);
    } else {
      stretch_blit(a, buffer, 0, 0, a->w, a->h, -left, -top, a->w*mag, a->h*mag);
    }
    int mx = (mouse_x+left)/mag, my = (mouse_y+top)/mag;
    rect(buffer, mx*mag-left, my*mag-top, (mx+p->patch_w-1)*mag-left, (my+p->patch_w-1)*mag-top, makecol(0, 0, 255));

    if ((unsigned) mx < (unsigned) AEW && (unsigned) my < (unsigned) AEH) {
      int *p_ann, *p_annd;
#if !TRANSLATE_ONLY
      int *p_ann_sim;
#endif
      if (!is_bitmap) {
        p_ann = ann->get(mx, my);
#if !TRANSLATE_ONLY
        p_ann_sim = ann_sim->get(mx, my);
#endif
        p_annd = annd->get(mx, my);
      } else {
        p_ann     = ((int *) ((BITMAP *) ann    )->line[my])+mx;
#if !TRANSLATE_ONLY
        p_ann_sim = ((int *) ((BITMAP *) ann_sim)->line[my])+mx;
#endif
        p_annd    = ((int *) ((BITMAP *) annd   )->line[my])+mx;
        if (p->knn != 1) { fprintf(stderr, "knn != 1 and is_bitmap is one\n"); exit(1); }
      }

      int patch_upsample = 5;
      int ybelow = buffer->h-p->patch_w*patch_upsample-30;
      
      for (int i = 0; i < p->knn; i++) {
        int xp = INT_TO_X(p_ann[i]);
        int yp = INT_TO_Y(p_ann[i]);
        int d = p_annd[i];
#if !TRANSLATE_ONLY
        int sp = INT_TO_X(p_ann_sim[i]);
        int tp = INT_TO_Y(p_ann_sim[i]);
        XFORM bpos = get_xform(p, xp, yp, sp, tp);
#else
        XFORM bpos = get_xform(p, xp, yp, SCALE_UNITY, 0);
#endif
        int rx[4], ry[4];
        int pw = p->patch_w-1;
        rx[0] = (bpos.x0*mag)>>16; ry[0] = (bpos.y0*mag)>>16;
        rx[1] = ((bpos.x0+bpos.dxdu*pw)*mag)>>16; ry[1] = ((bpos.y0+bpos.dydu*pw)*mag)>>16;
        rx[2] = ((bpos.x0+bpos.dxdu*pw+bpos.dxdv*pw)*mag)>>16; ry[2] = ((bpos.y0+bpos.dydu*pw+bpos.dydv*pw)*mag)>>16;
        rx[3] = ((bpos.x0+bpos.dxdv*pw)*mag)>>16; ry[3] = ((bpos.y0+bpos.dydv*pw)*mag)>>16;
        /*int pts[8] = { rx[0], ry[0],
                       rx[1], ry[1],
                       rx[2], ry[2],
                       rx[3], ry[3] };*/
        //polygon(buffer, 4, pts, makecol(0, 255, 0));
        if (dovis) {
          for (int j = 0; j < 4; j++) {
            line(buffer, rx[j]-left, ry[j]-top, rx[(j+1)%4]-left, ry[(j+1)%4]-top, makecol(0, 255, 0));
          }
          for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
              textprintf_ex(buffer, font, rx[0]-left+dx, ry[0]-top+dy, makecol(0, 0, 0), -1, "%d", i);
            }
          }
          textprintf_ex(buffer, font, rx[0]-left, ry[0]-top, makecol(255, 255, 255), -1, "%d", i);
          line(buffer, (mx+p->patch_w/2)*mag-left, (my+p->patch_w/2)*mag-top, (rx[0]+rx[1]+rx[2]+rx[3])/4-left, (ry[0]+ry[1]+ry[2]+ry[3])/4-top, makecol(255, 255, 255));

          int xbelow = (p->patch_w*patch_upsample+10)*i;
          //XFORM bpos = get_xform(p, xpp, ypp, spp, tpp);
          BITMAP *sampled = sample_xform(p, a, bpos);
          stretch_blit(sampled, buffer, 0, 0, sampled->w, sampled->h, xbelow, ybelow, sampled->w*patch_upsample, sampled->h*patch_upsample);
          destroy_bitmap(sampled);
          textprintf_ex(buffer, font, xbelow, ybelow-10, makecol(255,255,255), -1, "%d", i);
          rect(buffer, xbelow, ybelow, xbelow+p->patch_w*patch_upsample, ybelow+p->patch_w*patch_upsample, makecol(255, 255, 255));
          textprintf_ex(buffer, font, xbelow, ybelow+p->patch_w*patch_upsample+10, makecol(255,255,255), -1, "%d", int(sqrt(double(d))+0.5));
        }
      }
    }

    draw_sprite(buffer, mouse, mx*mag-left, my*mag-top);
    blit(buffer, screen, 0, 0, 0, 0, buffer->w, buffer->h);
  }
  
  delete ann;
  delete annd;
#if !TRANSLATE_ONLY
  delete ann_sim;
#endif
#endif
}

BITMAP *visnn(Params *p, BITMAP *ann, int is_rel, BITMAP *b);

#define XYREL_TO_INT(x, y) ((x+16384)|((y+16384)<<16))
#define INT_TO_XREL(p) (((p)&65535)-16384)
#define INT_TO_YREL(p) (((p)>>16)-16384)

#include <map>
#include <algorithm>
#define for_each(type, it, L) for (type::iterator (it) = (L).begin(); (it) != (L).end(); ++(it))

#define SMOOTH_DOWNSAMPLE 64

int div_rounddown(int x, int y) {
  if (x >= 0) { return x / y; }
  return -((-x+y-1)/y);
}

void map_offset(int &dx, int &dy) {
  //dx /= SMOOTH_DOWNSAMPLE;
  //dy /= SMOOTH_DOWNSAMPLE;
  dx = div_rounddown(dx+SMOOTH_DOWNSAMPLE/2, SMOOTH_DOWNSAMPLE);
  dy = div_rounddown(dy+SMOOTH_DOWNSAMPLE/2, SMOOTH_DOWNSAMPLE);
  dx *= SMOOTH_DOWNSAMPLE;
  dy *= SMOOTH_DOWNSAMPLE;
}

double data_energy(int k, double d2) {
  return 0; //k;
}

double smooth_energy(int x1, int y1, int x2, int y2) {
  int dx = x2-x1;
  int dy = y2-y1;
  int d2 = dx*dx+dy*dy;
  double d = sqrt(double(d2));
  if (d < 3) {
    return d;
  } else {
    return 50;
  }
  // return d
  //return d2 == 0 ? 0: 50;
}

double smooth_energy_for(Params *p, BITMAP *a, VBMP *ann, int *ans, int x1, int y1, int x2, int y2) {
  if ((unsigned) x1 >= (unsigned) AEW || (unsigned) y1 >= (unsigned) AEH) { return 0; }
  if ((unsigned) x2 >= (unsigned) AEW || (unsigned) y2 >= (unsigned) AEH) { return 0; }
  int *p_ann1 = ann->get(x1, y1);
  int *p_ann2 = ann->get(x2, y2);
  int label1 = ans[y1*AEW+x1];
  int label2 = ans[y2*AEW+x2];
  if ((unsigned) label1 >= (unsigned) p->knn) { fprintf(stderr, "label out of bounds: %d\n", label1); exit(1); }
  if ((unsigned) label2 >= (unsigned) p->knn) { fprintf(stderr, "label out of bounds: %d\n", label2); exit(1); }

  int idx1 = p_ann1[label1];
  int idx2 = p_ann2[label2];
  int xp1 = INT_TO_X(idx1)-x1, yp1 = INT_TO_Y(idx1)-y1;
  int xp2 = INT_TO_X(idx2)-x2, yp2 = INT_TO_Y(idx2)-y2;
  return smooth_energy(xp1, yp1, xp2, yp2);
}

double data_energy_for(Params *p, BITMAP *a, int *ans, int i, VBMP *annd) {
  int *p_annd = annd->get(i%AEW, i/AEW);
  return data_energy(ans[i], p_annd[ans[i]]);
}

/* Simple greedy pixel-changing algorithm. */
BITMAP *knn_smooth2(Params *p, BITMAP *a, BITMAP *b, VBMP *ann, VBMP *ann_sim, VBMP *annd, BITMAP *&annd_out) {
  int *ans = new int[AEW*AEH];
  for (int i = 0; i < AEW*AEH; i++) { ans[i] = rand()%p->knn; }
  double ans_f = 0;
  for (int i = 0; i < AEW*AEH; i++) {
    //int *p_annd = annd->get(i%AEW, i/AEW);
    //ans_f += data_energy(ans[i], p_annd[ans[i]]);
    ans_f += data_energy_for(p, a, ans, i, annd);
  }
  double data_f = ans_f;
  for (int i = 0; i < AEW*AEH; i++) {
    int x = i%AEW, y = i/AEW;
    if (x < AEW-1) {
      ans_f += smooth_energy_for(p, a, ann, ans, x, y, x+1, y);
    }
    if (y < AEH-1) {
      ans_f += smooth_energy_for(p, a, ann, ans, x, y, x, y+1);
    }
  }
  printf("starting energy: %f (data: %f)\n", ans_f, data_f);

  for (int iter = 0; iter < 32; iter++) {
    int istart = 0, iend = AEW*AEH, di = 1;
    if (iter % 2 == 1) { istart = AEW*AEH-1; iend = -1; di = -1; }
    for (int i = istart; i != iend; i += di) {
      int x = i%AEW, y = i/AEW;
      int label_prev = ans[i];
      for (int j = 0; j < p->knn; j++) {
        if (j == label_prev) { continue; }
        double ans_prev = ans_f;
        int label_current = ans[i];
        ans_f -= data_energy_for(p, a, ans, i, annd);
        ans_f -= smooth_energy_for(p, a, ann, ans, x, y, x+1, y);
        ans_f -= smooth_energy_for(p, a, ann, ans, x, y, x, y+1);
        ans_f -= smooth_energy_for(p, a, ann, ans, x-1, y, x, y);
        ans_f -= smooth_energy_for(p, a, ann, ans, x, y-1, x, y);
        ans[i] = j;
        ans_f += data_energy_for(p, a, ans, i, annd);
        ans_f += smooth_energy_for(p, a, ann, ans, x, y, x+1, y);
        ans_f += smooth_energy_for(p, a, ann, ans, x, y, x, y+1);
        ans_f += smooth_energy_for(p, a, ann, ans, x-1, y, x, y);
        ans_f += smooth_energy_for(p, a, ann, ans, x, y-1, x, y);
        if (ans_f >= ans_prev) {
          ans[i] = label_current;
          ans_f = ans_prev;
        }
      }
    }
    printf("energy after %d iters: %f\n", iter, ans_f);
  }

  BITMAP *ann_out = create_bitmap(a->w, a->h);
  clear(ann_out);
  annd_out = copy_image(ann_out);
  for (int i = 0; i < AEW*AEH; i++) {
	  int lbl = ans[i];
	  int x = i%(AEW), y = i/(AEW);
    int *p_ann = ann->get(x, y);
    int *p_annd = annd->get(x, y);
    if ((unsigned) lbl >= (unsigned) p->knn) { fprintf(stderr, "lbl out of range (%d %d)\n", lbl, p->knn); exit(1); }
	  ((int *) ann_out->line[y])[x] = p_ann[lbl];
	  ((int *) annd_out->line[y])[x] = p_annd[lbl];
  }
  return ann_out;
}

VBMP *bitmap_to_vbmp(BITMAP *bmp) {
  VBMP *ans = new VBMP(bmp->w, bmp->h, 1);
  insert_vbmp(ans, 0, bmp);
  destroy_bitmap(bmp);
  return ans;
}


BITMAP *visd(Params *p, BITMAP *annd);

void smooth_coherent(Params *p, BITMAP *a, BITMAP *b, BITMAP *ann, BITMAP *annd, BITMAP *&ann_out, BITMAP *&annd_out, int nsmooth) {
  ann_out = copy_image(ann);
  //annd_out = copy_image(annd);
  printf("smooth_coherent, nsmooth=%d\n", nsmooth);
  /*
  int n2 = nsmooth*2+1;
  int *dgrid = new int[n2*n2];
  int xmin = nsmooth;
  int ymin = nsmooth;
  int xmax = AEW-nsmooth;
  int ymax = AEH-nsmooth;
  for (int dy = -nsmooth; dy <= nsmooth; dy++) {
    for (int dx = -nsmooth; dx <= nsmooth; dx++) {
      if ((unsigned) (xmin+dx) >= (unsigned) AEW || (unsigned) (ymin+dy) >= (unsigned) AEH) { fprintf("out of bounds %d %d of %d %d\n", xmin+dx, ymin+dy, AEW, AEH); exit(1); }
      int off0 = ((int *) ann->line[ymin+dy])[xmin+dx];
      int x0 = INT_TO_X(off0)
    }
  }*/

  #pragma omp parallel for schedule(static,16)
  for (int y = 0; y < AEH; y++) {
    for (int x = 0; x < AEW; x++) {
      int dbest = INT_MAX;
      int xbest = 0, ybest = 0;
      for (int dy = -nsmooth; dy <= nsmooth; dy++) {
        if ((unsigned) (y+dy) >= (unsigned) AEH) { continue; }
        for (int dx = -nsmooth; dx <= nsmooth; dx++) {
          if ((unsigned) (x+dx) >= (unsigned) AEW) { continue; }
          int off0 = ((int *) ann->line[y+dy])[x+dx];
          int x0 = INT_TO_X(off0)-(x+dx), y0 = INT_TO_Y(off0)-(y+dy);
          
          int d = 0;
          for (int dy1 = -nsmooth; dy1 <= nsmooth; dy1++) {
            int ysrc = y+dy1;
            int ydest = ysrc+y0;
            if ((unsigned) ysrc >= (unsigned) AEH || (unsigned) ydest >= (unsigned) BEH) { d = INT_MAX; break; }
            for (int dx1 = -nsmooth; dx1 <= nsmooth; dx1++) {
              int xsrc = x+dx1;
              int xdest = xsrc+x0;
              if ((unsigned) xsrc >= (unsigned) AEW || (unsigned) xdest >= (unsigned) BEW) { d = INT_MAX; break; }

              d += patch_dist(p, a, xsrc, ysrc, b, xdest, ydest);
            }
          }

          if (d < dbest) {
            dbest = d;
            xbest = x+x0;
            ybest = y+y0;
          }
        }
      }

      ((int *) ann_out->line[y])[x] = XY_TO_INT(xbest, ybest);
    }
  }
  printf("done smooth_coherent\n");
  annd_out = init_dist(p, a, b, ann);
  //delete[] dgrid;
}

void combine_knn(Params *p1, Params *p2, BITMAP *a, BITMAP *b, VBMP *ann1, VBMP *ann_sim1, VBMP *annd1, VBMP *ann2, VBMP *ann_sim2, VBMP *annd2, VBMP *&ann, VBMP *&ann_sim, VBMP *&annd) {
  int k = p1->knn + p2->knn;
  ann = new VBMP(a->w, a->h, k);
  annd = new VBMP(a->w, a->h, k);
#if TRANSLATE_ONLY
  ann_sim = NULL;
#else
  ann_sim = new VBMP(a->w, a->h, k);
#endif
  for (int y = 0; y < a->h-p1->patch_w+1; y++) {
    for (int x = 0; x < a->w-p1->patch_w+1; x++) {
      for (int i = 0; i < p1->knn; i++) {
        ann->get(x, y)[i] = ann1->get(x, y)[i];
        annd->get(x, y)[i] = annd1->get(x, y)[i];
#if !TRANSLATE_ONLY
        ann_sim->get(x, y)[i] = ann_sim1->get(x, y)[i];
#endif
      }
      for (int i = 0; i < p2->knn; i++) {
        ann->get(x, y)[i+p1->knn] = ann2->get(x, y)[i];
        annd->get(x, y)[i+p1->knn] = annd2->get(x, y)[i];
#if !TRANSLATE_ONLY
        ann_sim->get(x, y)[i+p1->knn] = ann_sim2->get(x, y)[i];
#endif
      }
    }
  }
}
