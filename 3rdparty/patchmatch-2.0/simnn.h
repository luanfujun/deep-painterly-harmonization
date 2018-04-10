
/* PatchMatch with similarity transform of patches (rotation+scale). */

#ifndef _simnn_h
#define _simnn_h

#include "nn.h"

/*
#define ANGLE_SHIFT XY_TO_INT_SHIFT
#define NUM_ANGLES (1<<ANGLE_SHIFT)
#define SCALE_SHIFT XY_TO_INT_SHIFT
#define NUM_SCALES (1<<SCALE_SHIFT)
#define SCALE_MIN 1.0
#define SCALE_MAX 1.0
*/

#define ANGLE_SHIFT XY_TO_INT_SHIFT
#define NUM_ANGLES (1<<ANGLE_SHIFT)
#define SCALE_SHIFT XY_TO_INT_SHIFT
#define NUM_SCALES (1<<SCALE_SHIFT)
//#define SCALE_MIN 0.5
//#define SCALE_MAX 2.0
#define SCALE_UNITY (NUM_SCALES/2)      /* exp(log(SCALE_MIN)+(log(SCALE_MAX)-log(SCALE_MIN))*SCALE_UNITY*1.0/NUM_SCALES) should be exactly 1.0. */

class XFORM { 
public:
  int x0, y0, dxdu, dydu, dxdv, dydv;   /* Coords left shifted by 16. */
};

void init_xform_tables(double SCALE_MIN=0.5, double SCALE_MAX=2.0, int force_init=0);
XFORM get_xform(Params *p, int x, int y, int scale, int theta);  /* x and y not left shifted. */

void getpixel_bilin(BITMAP *bimg, int bx, int by, int &r, int &g, int &b);  /* Coords left shifted by 16. */

BITMAP *sim_init_nn(Params *p, BITMAP *a, BITMAP *b, BITMAP *&ann_sim);
BITMAP *sim_init_dist(Params *p, BITMAP *a, BITMAP *b, BITMAP *ann, BITMAP *ann_sim);

void sim_nn(Params *p, BITMAP *a, BITMAP *b,
            BITMAP *ann, BITMAP *ann_sim, BITMAP *annd,
            RegionMasks *amask=NULL, BITMAP *bmask=NULL,
            int level=0, int em_iter=0, RecomposeParams *rp=NULL, int offset_iter=0, int update_type=0, int cache_b=0,
            RegionMasks *region_masks=NULL, int tiles=-1);

BITMAP *sim_vote(Params *p, BITMAP *b,
						BITMAP *ann, BITMAP *ann_sim, BITMAP *bnn=NULL, BITMAP *bnn_sim=NULL,
						BITMAP *bmask=NULL, BITMAP *bweight=NULL,
						double coherence_weight=COHERENCE_WEIGHT, double complete_weight=COMPLETE_WEIGHT,
						RegionMasks *amask=NULL, BITMAP *aweight=NULL, BITMAP *ainit=NULL, RegionMasks *region_masks=NULL, BITMAP *aconstraint=NULL, int mask_self_only=0);

#endif
