
/* Code for 3-channel input image. Warning: weight BITMAPs always encode 32-bit floats, not ints. */

#ifndef _nn_h
#define _nn_h

#include "allegro_emu.h"
#include <stdio.h>
#include <vector>
#include <limits.h>
#include <string.h>
#include <stdlib.h>

/* Set this to 0 if your compiler does have OpenMP support. */
#define USE_OPENMP             1

#if USE_OPENMP
#include <omp.h>
#endif

using namespace std;

/* --------------------------------------------------------------------
   Defines
   -------------------------------------------------------------------- */

#define IS_VERBOSE             0
#define SYNC_WRITEBACK         1
#define MAX_PATCH_W            32

#define AEW (a->w - p->patch_w + 1)
#define AEH (a->h - p->patch_w + 1)
#define BEW (b->w - p->patch_w + 1)
#define BEH (b->h - p->patch_w + 1)

#ifndef MAX
//template<class T> inline T MAX(T a, T b) { a > b ? a : b; }
//template<class T> inline T MIN(T a, T b) { a < b ? a : b; } 
#define MAX(a, b) ((a)>(b)?(a):(b))
#define MIN(a, b) ((a)<(b)?(a):(b))
#endif

/* --------------------------------------------------------------------
   Defaults
   -------------------------------------------------------------------- */

#define COHERENCE_WEIGHT        0.5
#define COMPLETE_WEIGHT         0.5

#define XY_TO_INT(x, y) (((y)<<12)|(x))
#define INT_TO_X(v) ((v)&((1<<12)-1))
#define INT_TO_Y(v) ((v)>>12)
#define XY_TO_INT_SHIFT 12

/* --------------------------------------------------------------------
   Randomized NN algorithm
   -------------------------------------------------------------------- */

#if !IS_VERBOSE
#ifdef printf
#undef printf
#endif
#define printf (void)
#define fflush (void)
#endif

class RecomposeParams;
class Params;
class RegionMasks;

#define ALGO_CPU             0
#define ALGO_GPUCPU          6
#define ALGO_FULLRAND        7
#define ALGO_CPUTILED        8

#define KNN_ALGO_HEAP        0
#define KNN_ALGO_AVOID       1
#define KNN_ALGO_PRINANGLE   2
#define KNN_ALGO_TOP1NN      3
#define KNN_ALGO_WINDOW      4
#define KNN_ALGO_KDTREE      5
#define KNN_ALGO_CHANGEK     6
#define KNN_ALGO_FLANN       7

#define VOTE_MEAN            0
#define VOTE_SUM             1

class Params { public:
  int algo;              /* Algorithm to use, one of ALGO_*. */
  
  /* Randomized NN algorithm parameters. */
  int patch_w;           /* Width and height of square patch. */
  int vec_len;           /* Length of vector if using vectorized NN algorithms (vecnn.h), for non-square patches and feature descriptors. */
  int nn_iters;          /* Iters of randomized NN algorithm. */
  int rs_max;            /* Maximum width for RS. */
  int rs_min;            /* Minimum width for RS. */
  double rs_ratio;       /* Ratio (< 1) of successive RS sizes. */
  double rs_iters;       /* RS iters per pixel.  1 => search all scales once. */
  int do_propagate;      /* Do propagation. */
  int gpu_prop;          /* Maximum propagation distance for GPU algo. */
  int xmin, ymin;        /* Min update region coord, or -1 for no box. */
  int xmax, ymax;        /* Max update region coord, or -1 for no box. */
  int resample_seamcarv; /* Resample via seam carving. */
  int vote_algo;         /* Vote algorithm, one of VOTE_*. */
  int prefer_coherent;   /* Prefer coherent regions, bool, default false. */
  int allow_coherent;    /* This must be enabled for the previous flag to take effect. */
  int cores;             /* If > 1, use OpenMP. */
  int window_w;          /* Constraint search window width. */
  int window_h;          /* Constraint search window height. */
  int weight_r;          /* Multiplicative weights for R, G, B in distance computation. */
  int weight_g;
  int weight_b;
  
  /* Tree NN algorithm parameters. */
  int pca_dim;           /* Tree PCA dim, INT_MAX for no PCA. */
  double pca_var;        /* Fraction of total variance, e.g. 0.95 for 95%, negative to not use this param. */
  float eps;             /* Tree epsilon. */
  int kcoherence_k;      /* k for kcoherence. */
  int kcoherence_iters;  /* Iters of kcoherence "propagation", 2 iters is Lefebre '95. */
  int kcoherence_neighbors; /* k-coherence neighbors, 4 or 8. */

  int knn;
  int knn_algo;
  int restarts;
  int enrich_iters;
  int enrich_times;
  int do_inverse_enrich;
  int do_enrich;

  /* Defaults. */
  Params()
    :algo(ALGO_CPU),
     patch_w(7),
     vec_len(0),
     nn_iters(5),
     rs_max(INT_MAX),
     rs_min(1),
     rs_ratio(0.5),
     rs_iters(1),
     do_propagate(1),
     gpu_prop(8),
     xmin(-1), ymin(-1),
     xmax(-1), ymax(-1),
     resample_seamcarv(0),
     vote_algo(VOTE_MEAN),
     pca_dim(25),
     pca_var(-1),
     eps(2),
     prefer_coherent(0),
     allow_coherent(0),
     cores(2),
     window_w(INT_MAX),
     window_h(INT_MAX),
     weight_r(1),
     weight_g(1),
     weight_b(1),
     kcoherence_k(2),
     kcoherence_iters(2),
     kcoherence_neighbors(8),
     knn(0),
     knn_algo(KNN_ALGO_HEAP),
     restarts(1),
     enrich_iters(0),
     enrich_times(1),
     do_inverse_enrich(1),
     do_enrich(1)
     { }
};

void init_params(Params *p);
void init_openmp(Params *p);
void srand2(unsigned seed);

BITMAP *init_nn(Params *p, BITMAP *a, BITMAP *b, BITMAP *bmask=NULL, RegionMasks *region_masks=NULL, RegionMasks *amask=NULL, int trim_patch=1, BITMAP *ann_window=NULL, BITMAP *awinsize=NULL);
BITMAP *init_dist(Params *p, BITMAP *a, BITMAP *b, BITMAP *ann, BITMAP *bmask=NULL, RegionMasks *region_masks=NULL, RegionMasks *amask=NULL);

void nn(Params *p, BITMAP *a, BITMAP *b,
        BITMAP *ann, BITMAP *annd,
        RegionMasks *amask=NULL, BITMAP *bmask=NULL,
        int level=0, int em_iter=0, RecomposeParams *rp=NULL, int offset_iter=0, int update_type=0, int cache_b=0,
        RegionMasks *region_masks=NULL, int tiles=-1, BITMAP *ann_window=NULL, BITMAP *awinsize=NULL);

class Box { 
public:
  int xmin, ymin, xmax, ymax;
};

class RegionMasks { 
public:
  BITMAP *bmp;
  Box box[256];
  RegionMasks(Params *p, BITMAP *region_masks, int full=0, BITMAP *bmask=NULL);
};

void destroy_region_masks(RegionMasks *m);

/* --------------------------------------------------------------------
   Utility functions
   -------------------------------------------------------------------- */

/* PRNG without global variable, pass nonzero seed as argument. */
#define RANDI(u) (18000 * ((u) & 65535) + ((u) >> 16))

BITMAP *norm_image(double *accum, int w, int h);
BITMAP *norm_image(int *accum, int w, int h);

int is_window(Params *p);

int window_constraint(Params *p, BITMAP *a, BITMAP *b, int ax, int ay, int bx, int by, BITMAP *ann_window=NULL, BITMAP *awinsize=NULL);

Box get_abox(Params *p, BITMAP *a, RegionMasks *amask, int trim_patch=1);

BITMAP *copy_image(BITMAP *a);

BITMAP *vote(Params *p, BITMAP *b,
             BITMAP *ann, BITMAP *bnn=NULL,
             BITMAP *bmask=NULL, BITMAP *bweight=NULL,
             double coherence_weight=COHERENCE_WEIGHT, double complete_weight=COMPLETE_WEIGHT,
             RegionMasks *amask=NULL, BITMAP *aweight=NULL, BITMAP *ainit=NULL, RegionMasks *region_masks=NULL, BITMAP *aconstraint=NULL, int mask_self_only=0);

int patch_dist(Params *p, BITMAP *a, int ax, int ay,
               BITMAP *b, int bx, int by, int maxval=INT_MAX, RegionMasks *region_masks=NULL);

/* Clip votes to valid rectangle, return number of votes clipped. */
int clip_nn(Params *p, BITMAP *ann, BITMAP *b);

void check_colors(BITMAP *bmp);
void check_dists(Params *p, BITMAP *a, BITMAP *b, BITMAP *ann, BITMAP *annd, int max_dist=3, RegionMasks *amask=NULL);
void check_nn(Params *p, BITMAP *ann, BITMAP *b, BITMAP *bmask=NULL, RegionMasks *amask=NULL, RegionMasks *region_masks=NULL);

void getnn(BITMAP *ann, int x, int y, int &xp, int &yp);

void minnn(Params *p, BITMAP *a, BITMAP *b, BITMAP *ann, BITMAP *annd, BITMAP *ann_prev, BITMAP *bmask, int level, int em_iter, RecomposeParams *rp=NULL, RegionMasks *region_masks=NULL, RegionMasks *amask=NULL, int ntiles=-1);


class VECBITMAP_ARB { 
public:
  vector<unsigned> *data;
  int w, h;
  vector<unsigned> *get(int x, int y);
};

VECBITMAP_ARB *create_vecbitmap_arb(int w, int h);
void destroy_vecbitmap_arb(VECBITMAP_ARB *bmp);
void save_vecbitmap_arb(const char *filename, VECBITMAP_ARB *bmp);
void sort_vecbitmap_arb(VECBITMAP_ARB *bmp);

/* Parameters for high-level recomposition. */
class RecomposeParams { public:
  int minnn_optp_nn_iters;    /* Optimized params: NN iters for previous offsets. */
  int minnn_optp_rs_max;      /* Optimized params: Max RS for previous offsets. */

  RecomposeParams()
    :minnn_optp_nn_iters(2),
     minnn_optp_rs_max(1) {}
};

#endif
