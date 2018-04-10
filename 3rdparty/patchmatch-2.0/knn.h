
/* PatchMatch finding k-NN, with similarity transform of patches (rotation+scale). */

#ifndef _knn_h
#define _knn_h

#include "nn.h"
#include "simnn.h"
#include "vecnn.h"

#define SAVE_DIST           0

#define VBMP VECBITMAP<int>

BITMAP *greyscale(BITMAP *a);
BITMAP *greyscale16(BITMAP *a);
BITMAP *greyscale_to_color(BITMAP *a);
BITMAP *gaussian_blur16(BITMAP *a, double sigma);
BITMAP *greyscale16_to_color(BITMAP *a);
BITMAP *gaussian_deriv_angle(BITMAP *a, double sigma, BITMAP **dx_out=NULL, BITMAP **dy_out=NULL);

BITMAP *color_gaussian_blur(BITMAP *a, double sigma, int aconstraint_alpha);

BITMAP *extract_vbmp(VBMP *bmp, int i);
void insert_vbmp(VBMP *bmp, int i, BITMAP *a);
//void sort_knn(Params *p, BITMAP *a, VBMP *ann, VBMP *ann_sim, VBMP *annd);
VBMP *copy_vbmp(VBMP *a);

#define N_PRINCIPAL_ANGLE_SHIFT 8
#define N_PRINCIPAL_ANGLE (1<<N_PRINCIPAL_ANGLE_SHIFT)

class PRINCIPAL_ANGLE { public:
  BITMAP *angle[N_PRINCIPAL_ANGLE];
};

PRINCIPAL_ANGLE *create_principal_angle(Params *p, BITMAP *bmp);
void destroy_principal_angle(PRINCIPAL_ANGLE *b);
int get_principal_angle(Params *p, PRINCIPAL_ANGLE *b, int x0, int y0, int scale);

VBMP *knn_init_nn(Params *p, BITMAP *a, BITMAP *b, VBMP *&ann_sim, PRINCIPAL_ANGLE *pa=NULL);
VBMP *knn_init_dist(Params *p, BITMAP *a, BITMAP *b, VBMP *ann, VBMP *ann_sim);

void knn(Params *p, BITMAP *a, BITMAP *b,
         VBMP *&ann, VBMP *&ann_sim, VBMP *&annd,
         RegionMasks *amask=NULL, BITMAP *bmask=NULL,
         int level=0, int em_iter=0, RecomposeParams *rp=NULL, int offset_iter=0, int update_type=0, int cache_b=0,
         RegionMasks *region_masks=NULL, int tiles=-1, PRINCIPAL_ANGLE *pa=NULL, int save_first=0);

class KNNWeightFunc { public:
  virtual double weight(double d, int is_center) = 0;
};

class KNNSolverWeightFunc: public KNNWeightFunc { public:
  double param[3];
  KNNSolverWeightFunc(double x[3]);
  virtual double weight(double d, int is_center);
};

class ObjectiveFunc { public:
  virtual double f(double x[]) = 0;
};

double patsearch(ObjectiveFunc *f, double *x, double *ap, int n, int iters);

BITMAP *knn_vote(Params *p, BITMAP *b,
                 VBMP *ann, VBMP *ann_sim, VBMP *annd, VBMP *bnn=NULL, VBMP *bnn_sim=NULL,
                 BITMAP *bmask=NULL, BITMAP *bweight=NULL,
                 double coherence_weight=COHERENCE_WEIGHT, double complete_weight=COMPLETE_WEIGHT,
                 RegionMasks *amask=NULL, BITMAP *aweight=NULL, BITMAP *ainit=NULL, RegionMasks *region_masks=NULL, BITMAP *aconstraint=NULL, int mask_self_only=0, KNNWeightFunc *weight_func=NULL, double **accum_out=NULL);

BITMAP *knn_vote_solve(Params *p, BITMAP *b,
                 VBMP *ann, VBMP *ann_sim, VBMP *annd, int n, BITMAP *aorig, double weight_out[3]);

void knn_vis(Params *p, BITMAP *a, VBMP *ann, VBMP *ann_sim, VBMP *annd, int is_bitmap=0, BITMAP *vote=NULL, BITMAP *orig=NULL, BITMAP *vote_uniform=NULL);

void knn_dual_vis(Params *p, BITMAP *a, BITMAP *b, VBMP *ann, VBMP *ann_sim, VBMP *annd, int is_bitmap=0, BITMAP *vote=NULL, BITMAP *orig=NULL);

double knn_avg_dist(Params *p, VBMP *annd);

void knn_enrich(Params *p, BITMAP *a, BITMAP *b, VBMP *ann, VBMP *ann_sim, VBMP *annd, PRINCIPAL_ANGLE *pa=NULL);

void knn_enrich3(Params *p, BITMAP *a, BITMAP *b, VBMP *ann, VBMP *ann_sim, VBMP *annd, PRINCIPAL_ANGLE *pa=NULL);

void knn_enrich4(Params *p, BITMAP *a, BITMAP *b, VBMP *ann, VBMP *ann_sim, VBMP *annd, PRINCIPAL_ANGLE *pa=NULL);

void knn_inverse_enrich(Params *p, BITMAP *a, BITMAP *b, VBMP *ann, VBMP *ann_sim, VBMP *annd, PRINCIPAL_ANGLE *pa=NULL);

void knn_inverse_enrich2(Params *p, BITMAP *a, BITMAP *b, VBMP *ann, VBMP *ann_sim, VBMP *annd, PRINCIPAL_ANGLE *pa=NULL);

void knn_check(Params *p, BITMAP *a, BITMAP *b, VBMP *ann, VBMP *ann_sim, VBMP *annd, int check_duplicates=1);

void save_dist(Params *p, VBMP *annd, const char *suffix);

/* Also changes p->knn to kp. */
void change_knn(Params *p, BITMAP *a, BITMAP *b, VBMP *&ann, VBMP *&ann_sim, VBMP *&annd, int kp, PRINCIPAL_ANGLE *pa=NULL);

void combine_knn(Params *p1, Params *p2, BITMAP *a, BITMAP *b, VBMP *ann1, VBMP *ann_sim1, VBMP *annd1, VBMP *ann2, VBMP *ann_sim2, VBMP *annd2, VBMP *&ann, VBMP *&ann_sim, VBMP *&annd);

void check_change_knn(Params *p, BITMAP *a, BITMAP *b);

void sort_knn(Params *p, VBMP *ann, VBMP *ann_sim, VBMP *annd);

#endif
