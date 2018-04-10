
/* Warning: weight BITMAPs always encode 32-bit floats, not ints. */

#include "nn.h"
#include <deque>
#include <vector>
#include <algorithm>
#include <math.h>
#include <float.h>

#include "patch.h"

static char AdobePatentID_P876E1[] = "AdobePatentID=\"P876E1\""; // AdobePatentID="P876E1"
static char AdobePatentID_P962[] = "AdobePatentID=\"P962\""; // AdobePatentID="P962"

#define rand rand2
#undef RAND_MAX
#define RAND_MAX 4294967295U
/* A small PRNG by George Marsaglia, faster than rand(), from:
   http://www.math.uni-bielefeld.de/~sillke/ALGORITHMS/random/marsaglia-c */
unsigned int rand2_u = 2282506733U, rand2_v = 1591164231U;
inline unsigned int rand2() {
  rand2_v = 36969 * (rand2_v & 65535) + (rand2_v >> 16);
  rand2_u = 18000 * (rand2_u & 65535) + (rand2_u >> 16);
  return (rand2_v << 16) + (rand2_u & 65535);
}
void srand2(unsigned seed) {
  rand2_u = seed;
  rand2_v = ~seed;
  if (!rand2_u) { rand2_u++; }
  if (!rand2_v) { rand2_v++; }
  for (int i = 0; i < 10; i++) { rand2(); }
  
  rand2_u = rand2();
  rand2_v = rand2()^seed;
  if (!rand2_u) { rand2_u++; }
  if (!rand2_v) { rand2_v++; }
}
unsigned int randi(unsigned int u) {
  u = 18000 * (u & 65535) + (u >> 16);
  u = 18000 * (u & 65535) + (u >> 16);
  u = 18000 * (u & 65535) + (u >> 16);
  u = 18000 * (u & 65535) + (u >> 16);
  u = 18000 * (u & 65535) + (u >> 16);
  return u;
}

#define random() (rand()*(1.0/(RAND_MAX-1)))
#define randomi(u) (randi(u)*(1.0/(RAND_MAX-1)))
#undef PATCH_W

using namespace std;

/* ----------------------------------------------------------------
   Code dependent on patch size
   ---------------------------------------------------------------- */

int last_cores = 0;

void init_params(Params *p) {
  init_openmp(p);
}

void init_openmp(Params *p) {
  if (p->cores != last_cores) {
    last_cores = p->cores;
#if USE_OPENMP
    omp_set_num_threads(p->cores);
    omp_set_nested(1);
    omp_set_dynamic(0);
#endif
  }
}

BITMAP *norm_image(double *accum, int w, int h) {
  BITMAP *ans = create_bitmap(w, h);
#pragma omp parallel for schedule(static, 16)
  for (int y = 0; y < h; y++) {
    int *row = (int *) ans->line[y];
    double *prow = &accum[4*(y*w)];
    for (int x = 0; x < w; x++) {
      double *p = &prow[4*x];
      double s = p[3] ? 1.0 / p[3]: 1.0;
      row[x] = int(p[0]*s+0.5)|(int(p[1]*s+0.5)<<8)|(int(p[2]*s+0.5)<<16);  /* Changed: round() instead of floor. */
    }
  }
  return ans;
}

BITMAP *norm_image(int *accum, int w, int h) {
  BITMAP *ans = create_bitmap(w, h);
#pragma omp parallel for schedule(static, 16)
  for (int y = 0; y < h; y++) {
    int *row = (int *) ans->line[y];
    int *prow = &accum[4*(y*w)];
    for (int x = 0; x < w; x++) {
      int *p = &prow[4*x];
      int c = p[3] ? p[3]: 1;
      int c2 = c>>1;             /* Changed: round() instead of floor. */
      row[x] = int((p[0]+c2)/c)|(int((p[1]+c2)/c)<<8)|(int((p[2]+c2)/c)<<16);
    }
  }
  return ans;
}

BITMAP *norm_image_init(int *accum, int w, int h, BITMAP *ainit) {
  BITMAP *ans = create_bitmap(w, h);
#pragma omp parallel for schedule(static, 16)
  for (int y = 0; y < h; y++) {
    int *row = (int *) ans->line[y];
    int *a0row = (int *) ainit->line[y];
    int *prow = &accum[4*(y*w)];
    for (int x = 0; x < w; x++) {
      int *p = &prow[4*x];
      int c = p[3] ? p[3]: 1;
      int c2 = c>>1;
      row[x] = p[3] ? int((p[0]+c2)/c)|(int((p[1]+c2)/c)<<8)|(int((p[2]+c2)/c)<<16) : a0row[x];
    }
  }
  return ans;
}

// fill in non-voted pixel using value of corresponding pixel in ainit 
BITMAP *norm_image_init(double *accum, int w, int h, BITMAP *ainit) {
  BITMAP *ans = create_bitmap(w, h);
#pragma omp parallel for schedule(static, 16)
  for (int y = 0; y < h; y++) {
    int *row = (int *) ans->line[y];
    int *a0row = (int *) ainit->line[y];
    double *prow = &accum[4*(y*w)];
    for (int x = 0; x < w; x++) {
      double *p = &prow[4*x];
      double s = p[3] ? 1.0 / p[3]: 1.0;
      row[x] = p[3] ? int(p[0]*s+0.5)|(int(p[1]*s+0.5)<<8)|(int(p[2]*s+0.5)<<16) : a0row[x];
    }
  }
  return ans;
}

template<int PATCH_W, int IS_MASK, int IS_WINDOW>
BITMAP *init_dist_n(Params *p, BITMAP *a, BITMAP *b, BITMAP *ann, BITMAP *bmask, RegionMasks *region_masks, RegionMasks *amask) {
  BITMAP *ans = create_bitmap(a->w, a->h);
  clear_to_color(ans, INT_MAX);
  if (region_masks) { // XC?, why the second line?
    if (region_masks->bmp->w != a->w || region_masks->bmp->h != a->h) { fprintf(stderr, "region_masks (%dx%d) size != a (%dx%d) size\n", region_masks->bmp->w, region_masks->bmp->h, a->w, a->h); exit(1); }
    if (region_masks->bmp->w != b->w || region_masks->bmp->h != b->h) { fprintf(stderr, "region_masks (%dx%d) size != b (%dx%d) size\n", region_masks->bmp->w, region_masks->bmp->h, b->w, b->h); exit(1); }
  }

  Box box = get_abox(p, a, amask);
#pragma omp parallel for schedule(static, 4)
  for (int y = box.ymin; y < box.ymax; y++) {
    int adata[PATCH_W*PATCH_W];
    int *row = (int *) ans->line[y];
    int *arow = amask ? (int *) amask->bmp->line[y]: NULL;
    for (int x = box.xmin; x < box.xmax; x++) {
      if (IS_MASK && amask && arow[x]) { continue; }
      int xp, yp;
      getnn(ann, x, y, xp, yp);

      if (IS_MASK && region_masks && ((int *) region_masks->bmp->line[y])[x] != ((int *) region_masks->bmp->line[yp])[xp]) {
        row[x] = INT_MAX; continue;
      }
      
      for (int dy = 0; dy < PATCH_W; dy++) { // copy a patch from a to adata
        int *drow = ((int *) a->line[y+dy])+x;
        int *adata_row = adata+(dy*PATCH_W);
        for (int dx = 0; dx < PATCH_W; dx++) {
          adata_row[dx] = drow[dx];
        }
      }

      if (IS_MASK && bmask && ((int *) bmask->line[yp])[xp]) { row[x] = INT_MAX; continue; }
      row[x] = fast_patch_nobranch<PATCH_W, IS_WINDOW>(adata, b, xp, yp, p);
      //if (x == 1 && y == 1) { printf("1, 1 => %d, %d (%d)\n", xp, yp, row[x]); }
    }
  }
  return ans;
}

template<int PATCH_W>
BITMAP *vote_n(Params *p, BITMAP *b,
							 BITMAP *ann, BITMAP *bnn,
							 BITMAP *bmask, BITMAP *bweight,
							 double coherence_weight, double complete_weight,         
							 RegionMasks *amask, BITMAP *aweight, BITMAP *ainit, RegionMasks *region_masks, BITMAP *aconstraint, int mask_self_only) 
{
 
	printf("vote, mask_self_only=%d\n", mask_self_only);

	int sz = ann->w * ann->h; sz = sz << 2; // 4 * w * h
  double *accum = new double[sz]; // accumulator for RGB and total weights
  memset((void *) accum, 0, sizeof(double) * sz);
  
  double wa = 1, wb = 1;
  if (bnn) {
    wa = coherence_weight / ((ann->h-PATCH_W+1) * (ann->w-PATCH_W+1));
    wb = complete_weight  / ((bnn->h-PATCH_W+1) * (bnn->w-PATCH_W+1));
  }

  Box box = get_abox(p, ann, amask);
  
  /* Coherence */
  printf("vote_n, amask: %p, bmask: %p, aweight: %p, bweight: %p, mask_self_only: %d\n", amask, bmask, aweight, bweight, mask_self_only);
  for (int ay = box.ymin; ay < box.ymax; ay++) {
    int *amask_row = amask ? (int *) amask->bmp->line[ay]: NULL;
    for (int ax = box.xmin; ax < box.xmax; ax++) {
	    if (amask && amask_row[ax]) { continue; }
      int bx, by;
      getnn(ann, ax, ay, bx, by);
	  // Note: currently an output patch CANNOT vote to an input patch outside the mask 'bmask' (this seems reasonable)
      if (!mask_self_only && bmask && ((int *) bmask->line[by])[bx]) { continue; } 

      double w = wa;
      if (aweight) { w *= ((float *) aweight->line[ay+PATCH_W/2])[ax+PATCH_W/2]; /*printf("aweight: %f\n", w);*/ } // weight is centered, unlike NN fields
      if (!mask_self_only && bweight) { w *= ((float *) bweight->line[by+PATCH_W/2])[bx+PATCH_W/2]; }
      //if (w < 1e-30) { w = 1e-20; }
      for (int dy = 0; dy < PATCH_W; dy++) {
        int *brow = ((int *) b->line[by+dy]) + bx;
        double *prow = &accum[4*((ay+dy)*ann->w+ax)];
        for (int dx = 0; dx < PATCH_W; dx++) {
          int c = brow[dx];
          double *p = &prow[4*dx];
          p[0] += (c&255)*w;
          p[1] += ((c>>8)&255)*w;
          p[2] += (c>>16)*w;
          p[3] += w;
        }
      }
    }
  }

  /* Completeness */
  if (bnn) {
    printf("Using completeness\n");
    for (int by = 0; by < bnn->h-PATCH_W+1; by++) {
      for (int bx = 0; bx < bnn->w-PATCH_W+1; bx++) {
        if (bmask && ((int *) bmask->line[by])[bx]) { continue; }
        int ax, ay;
        getnn(bnn, bx, by, ax, ay);
		// Note: currently an input patch CAN vote to an output patch outside the mask 'amask' (not sure this is good)
		//if (amask && ((int *) amask->line[ay])[ax]) { continue; } 

        double w = wb;
        if (bweight) { w *= ((float *) bweight->line[by+PATCH_W/2])[bx+PATCH_W/2]; }
        for (int dy = 0; dy < PATCH_W; dy++) {
          int *brow = ((int *) b->line[by+dy]) + bx;
          double *prow = &accum[4*((ay+dy)*ann->w+ax)];
          for (int dx = 0; dx < PATCH_W; dx++) {
            int c = brow[dx];
            double *p = &prow[4*dx];
            p[0] += (c&255)*w;
            p[1] += ((c>>8)&255)*w;
            p[2] += (c>>16)*w;
            p[3] += w;
          }
        }
      }
    }
  }
  
  //BITMAP *ans = norm_image(accum, ann->w, ann->h);
  BITMAP *ans = NULL;
  printf("ainit: %p\n", ainit);
  ans = ainit ? norm_image_init(accum, ann->w, ann->h, ainit) : norm_image(accum, ann->w, ann->h);

  delete[] accum;
  
  return ans;
}

/* 2-core parallel vote().  Gives slightly different result, due to addition not being associative (round off error). */
template<int PATCH_W>
BITMAP *vote_n_openmp(Params *p, BITMAP *b,
											BITMAP *ann, BITMAP *bnn,
                      BITMAP *bmask, BITMAP *bweight,
                      double coherence_weight, double complete_weight,
                      RegionMasks *amask, BITMAP *aweight, BITMAP *ainit, RegionMasks *region_masks, BITMAP *aconstraint, int mask_self_only) {
  init_openmp(p);
  //return vote_n<PATCH_W>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint);
  
	int sz = ann->w*ann->h; sz = sz << 2;
  double *accum1 = new double[sz];
  memset((void *) accum1, 0, sizeof(double) * sz);
  double *accum2 = new double[sz];
  memset((void *) accum2, 0, sizeof(double) * sz);

  double wa = 1, wb = 1;
  if (bnn) {
    wa = coherence_weight / ((ann->h-PATCH_W+1) * (ann->w-PATCH_W+1));
    wb = complete_weight  / ((bnn->h-PATCH_W+1) * (bnn->w-PATCH_W+1));
  }

  Box box = get_abox(p, ann, amask);
  
  printf("vote_n_openmp, amask: %p, bmask: %p, aweight: %p, bweight: %p, mask_self_only: %d\n", amask, bmask, aweight, bweight, mask_self_only);
//#pragma omp parallel for schedule(static, 16)
#pragma omp parallel
  {
#if USE_OPENMP
    int nthread = omp_get_thread_num();
#else
    int nthread = 0;
#endif
    if (nthread == 0) {
      /* Coherence */
      for (int ay = box.ymin; ay < box.ymax; ay++) {
        int *amask_row = amask ? (int *) amask->bmp->line[ay]: NULL;
        for (int ax = box.xmin; ax < box.xmax; ax++) {
	        if (amask && amask_row[ax]) { continue; }
          int bx, by;
          getnn(ann, ax, ay, bx, by);
	      // Note: currently an output patch CANNOT vote to an input patch outside the mask 'bmask' (this seems reasonable)

          if (!mask_self_only && bmask && ((int *) bmask->line[by])[bx]) { continue; } 

          double w = wa;
          if (aweight) { w *= ((float *) aweight->line[ay+PATCH_W/2])[ax+PATCH_W/2]; /*printf("aweight: %f\n", w);*/ }
          if (!mask_self_only && bweight) { w *= ((float *) bweight->line[by+PATCH_W/2])[bx+PATCH_W/2]; }
          //if (w < 1e-30) { w = 1e-20; }
          for (int dy = 0; dy < PATCH_W; dy++) {
            int *brow = ((int *) b->line[by+dy]) + bx;
            double *prow = &accum1[4*((ay+dy)*ann->w+ax)];
            for (int dx = 0; dx < PATCH_W; dx++) {
              int c = brow[dx];
              double *p = &prow[4*dx];
              p[0] += (c&255)*w;
              p[1] += ((c>>8)&255)*w;
              p[2] += (c>>16)*w;
              p[3] += w;
            }
          }
        }
      }
    } 
		else if (nthread == 1) {
      /* Completeness */
      if (bnn) {
        printf("Using completeness\n");
        for (int by = 0; by < bnn->h-PATCH_W+1; by++) {
          for (int bx = 0; bx < bnn->w-PATCH_W+1; bx++) {
            if (bmask && ((int *) bmask->line[by])[bx]) { continue; }
            int ax, ay;
            getnn(bnn, bx, by, ax, ay);
		    // Note: currently an input patch CAN vote to an output patch outside the mask 'amask' (not sure this is good)
		    //if (amask && ((int *) amask->line[ay])[ax]) { continue; } 

            double w = wb;
            if (bweight) { w *= ((float *) bweight->line[by+PATCH_W/2])[bx+PATCH_W/2]; }
            for (int dy = 0; dy < PATCH_W; dy++) {
              int *brow = ((int *) b->line[by+dy]) + bx;
              double *prow = &accum2[4*((ay+dy)*ann->w+ax)];
              for (int dx = 0; dx < PATCH_W; dx++) {
                int c = brow[dx];
                double *p = &prow[4*dx];
                p[0] += (c&255)*w;
                p[1] += ((c>>8)&255)*w;
                p[2] += (c>>16)*w;
                p[3] += w;
              }
            }
          }
        }
      }
    }
  }

#pragma omp parallel for schedule(static, 16)
  for (int ay = box.ymin; ay < box.ymax+p->patch_w-1; ay++) {
    int *amask_row = amask ? (int *) amask->bmp->line[ay]: NULL;
    double *arow1 = &accum1[4*(ay*ann->w)];
    double *arow2 = &accum2[4*(ay*ann->w)];
    int xfinal = (box.xmax+p->patch_w-1)*4;
    for (int ax = box.xmin*4; ax < xfinal; ax++) {
      arow1[ax] += arow2[ax];
    }
  }
  
  //BITMAP *ans = norm_image(accum, ann->w, ann->h);
  BITMAP *ans = NULL;
  printf("ainit: %p\n", ainit);
  ans = ainit ? norm_image_init(accum1, ann->w, ann->h, ainit) : norm_image(accum1, ann->w, ann->h);

  delete[] accum1;
  delete[] accum2;
  
  return ans;
}

class VoteLink { 
public:
  int pos;
  VoteLink *next;
};

/* Fully parallel vote(), used when algorithm is ALGO_CPUTILED. */
template<int PATCH_W, class WTYPE, int IS_SIMPLE, int ALLOW_AMASK>
BITMAP *vote_n_cputiled(Params *p, BITMAP *b,
                        BITMAP *ann, BITMAP *bnn,
                        BITMAP *bmask, BITMAP *bweight,
                        double coherence_weight, double complete_weight,
                        RegionMasks *amask, BITMAP *aweight, BITMAP *ainit, RegionMasks *region_masks, BITMAP *aconstraint, int mask_self_only) 
{
  int tiles = p->cores;
  printf("vote_n_cputiled, IS_SIMPLE=%d, ALLOW_AMASK=%d, mask_self_only=%d\n", IS_SIMPLE, ALLOW_AMASK, mask_self_only);

	int sz = ann->w * ann->h; sz = sz << 2; // 4 * w * h
  WTYPE *accum = new WTYPE[sz];
  memset((void *) accum, 0, sizeof(WTYPE) * sz);

  double wa = 1, wb = 1; // wa: coherence_weight, wb: complete_weight
  if (bnn) {
    wa = coherence_weight / ((ann->h-PATCH_W+1) * (ann->w-PATCH_W+1));
    wb = complete_weight  / ((bnn->h-PATCH_W+1) * (bnn->w-PATCH_W+1));
  }

  Box box = get_abox(p, ann, amask);
  
  /* Completeness */
  VoteLink *links = NULL;
  VoteLink **start = NULL;
  if (bnn) {
    printf("Using completeness\n");
    int beh = bnn->h-PATCH_W+1, bew = bnn->w-PATCH_W+1;
    links = new VoteLink[bew*beh];
    start = new VoteLink *[ann->w*ann->h];
    memset((void *) start, 0, sizeof(VoteLink*)*ann->w*ann->h);
    VoteLink *linksp = links;
    for (int by = 0; by < bnn->h-PATCH_W+1; by++) {
      for (int bx = 0; bx < bnn->w-PATCH_W+1; bx++) {
        if (bmask && ((int *) bmask->line[by])[bx]) { continue; }
        int ax, ay;
        getnn(bnn, bx, by, ax, ay);
        
        VoteLink **startp = &start[ay*ann->w+ax];
        linksp->next = *startp;
        linksp->pos = XY_TO_INT(bx, by);
        *startp = linksp;
        linksp++;
      }
    }
  }

  /* Coherence.  In theory there could be contention issues on the overlapping region between threads, but in practice this doesn't happen. */
  printf("amask: %p, bmask: %p, aweight: %p, bweight: %p\n", amask, bmask, aweight, bweight);
  #pragma omp parallel num_threads(tiles)
  {
#if USE_OPENMP
    int ithread = omp_get_thread_num();
#else
    int ithread = 0;
#endif
    //int xmin = box.xmin, ymin = box.ymin, xmax = box.xmax, ymax = box.ymax;
    int ymin = box.ymin + (box.ymax-box.ymin)*ithread/tiles;
    int ymax = box.ymin + (box.ymax-box.ymin)*(ithread+1)/tiles;

    for (int ay = ymin/*+ystep*/; ay < ymax; ay++ /*+=p->patch_w*/) {
      int *amask_row = amask ? (int *) amask->bmp->line[ay]: NULL;
      for (int ax = box.xmin; ax < box.xmax; ax++) {
	      if ((!IS_SIMPLE || ALLOW_AMASK) && amask && amask_row[ax]) { continue; }
        int bx, by;
        getnn(ann, ax, ay, bx, by);
	    // Note: currently an output patch CANNOT vote to an input patch outside the mask 'bmask' (this seems reasonable)
        if (!IS_SIMPLE && !mask_self_only && bmask && ((int *) bmask->line[by])[bx]) { continue; } 

        WTYPE w = IS_SIMPLE ? 1: wa;
        if (!IS_SIMPLE && aweight) { w *= ((float *) aweight->line[ay+PATCH_W/2])[ax+PATCH_W/2]; /*printf("aweight: %f\n", w);*/ }
        if (!IS_SIMPLE && !mask_self_only && bweight) { w *= ((float *) bweight->line[by+PATCH_W/2])[bx+PATCH_W/2]; }
        //if (w < 1e-30) { w = 1e-20; }
        for (int dy = 0; dy < PATCH_W; dy++) {
          int *brow = ((int *) b->line[by+dy]) + bx;
          WTYPE *prow = &accum[4*((ay+dy)*ann->w+ax)];
          for (int dx = 0; dx < PATCH_W; dx++) {
            int c = brow[dx];
            WTYPE *p = &prow[4*dx];
            p[0] += (c&255)*w;
            p[1] += ((c>>8)&255)*w;
            p[2] += (c>>16)*w;
            p[3] += w;
          }
        }
        
        if (!IS_SIMPLE && bnn) {
          /* Completeness */
          VoteLink *current = start[ay*ann->w+ax];
          while (current) {
            int bx = INT_TO_X(current->pos), by = INT_TO_Y(current->pos);

            WTYPE w = wb;
            if (bweight) { w *= ((float *) bweight->line[by+PATCH_W/2])[bx+PATCH_W/2]; }
            for (int dy = 0; dy < PATCH_W; dy++) {
              int *brow = ((int *) b->line[by+dy]) + bx;
              WTYPE *prow = &accum[4*((ay+dy)*ann->w+ax)];
              for (int dx = 0; dx < PATCH_W; dx++) {
                int c = brow[dx];
                WTYPE *p = &prow[4*dx];
                p[0] += (c&255)*w;
                p[1] += ((c>>8)&255)*w;
                p[2] += (c>>16)*w;
                p[3] += w;
              }
            }

            current = current->next;
          }
        }
      }
    }
  }

  //BITMAP *ans = norm_image(accum, ann->w, ann->h);
  BITMAP *ans = NULL;
  printf("ainit: %p\n", ainit);
  ans = ainit ? norm_image_init(accum, ann->w, ann->h, ainit) : norm_image(accum, ann->w, ann->h);

  delete[] accum;
  delete[] links;
  delete[] start;
  
  return ans;
}

int window_constraint(Params *p, BITMAP *a, BITMAP *b, int ax, int ay, int bx, int by, BITMAP *ann_window, BITMAP *awinsize) {
  int bxp,byp,win_w,win_h;
  if (ann_window) {
    // local windows around prior field
    getnn(ann_window, ax, ay, bxp, byp);
  } else {
    // local windows around linear coordinate transform of a to b
    bxp = ax * b->w / a->w;
    byp = ay * b->h / a->h;
  }
  int dx = bx-bxp;
  int dy = by-byp;
  if (awinsize) {
    getnn(awinsize, ax, ay, win_w, win_h);
  } else {
    win_w = p->window_w;
    win_h = p->window_h;
  }
  //mexPrintf("ax=%d bxp=%d dx=%d p->win_w=%d win_w=%d\n", ax,bxp,dx,p->window_w,win_w);
  return (abs(dx)<<1) <= win_w && (abs(dy)<<1) <= win_h;
}

/* IS_WINDOW means that window_w, window_h search window constraints are used, and weight_r,g,b weights for distance computation are used. */
// annd stores previous ann error
template<int PATCH_W, int IS_MASK, int IS_WINDOW>
void nn_n(Params *p, BITMAP *a, BITMAP *b,
          BITMAP *ann, BITMAP *annd,
          RegionMasks *amask, BITMAP *bmask,
          int level, int em_iter, RecomposeParams *rp, int offset_iter, int update_type, RegionMasks *region_masks, int tiles,
					BITMAP *ann_window, BITMAP *awinsize) 
{

  printf("in nn_n, masks are: %p %p %p, tiles=%d\n", amask, bmask, region_masks, tiles);
  Box box = get_abox(p, a, amask);
  int nn_iter = 0;

  int xmin = box.xmin, ymin = box.ymin, xmax = box.xmax, ymax = box.ymax;
  for (; nn_iter < p->nn_iters; nn_iter++) {
    unsigned int iter_seed = rand();

    int ystart = ymin, yfinal = ymax, ychange=1; // from up-left to bottom-right
    int xstart = xmin, xfinal = xmax, xchange=1;
    if ((nn_iter + offset_iter) % 2 == 1) {
      xstart = xmax-1; xfinal = xmin-1; xchange=-1; // from bottom-right to up-left
      ystart = ymax-1; yfinal = ymin-1; ychange=-1;
    }
    int dx = -xchange, dy = -ychange;

    int bew = b->w-PATCH_W, beh = b->h-PATCH_W;
    int max_mag = max(b->w, b->h);
    int rs_ipart = int(p->rs_iters);
    double rs_fpart = p->rs_iters - rs_ipart;
    int rs_max = p->rs_max;
    if (rs_max > max_mag) { rs_max = max_mag; }

    int adata[PATCH_W*PATCH_W];
    for (int y = ystart; y != yfinal; y += ychange) {
      int *annd_row = (int *) annd->line[y];
      int *amask_row = IS_MASK ? (amask ? (int *) amask->bmp->line[y]: NULL): NULL;
      for (int x = xstart; x != xfinal; x += xchange) {
        if (IS_MASK && amask && amask_row[x]) { continue; }

        for (int dy0 = 0; dy0 < PATCH_W; dy0++) { // copy a patch from a
          int *drow = ((int *) a->line[y+dy0])+x;
          int *adata_row = adata+(dy0*PATCH_W);
          for (int dx0 = 0; dx0 < PATCH_W; dx0++) {
            adata_row[dx0] = drow[dx0];
          }
        }
        
        int src_mask = IS_MASK ? (region_masks ? ((int *) region_masks->bmp->line[y])[x]: 0): 0;
        
        int xbest, ybest;
        getnn(ann, x, y, xbest, ybest);
        int err = annd_row[x];
        if (err == 0) { continue; }

        /* Propagate */
        if (p->do_propagate) {
          if (!IS_WINDOW) {
            /* Propagate x */
            if ((unsigned) (x+dx) < (unsigned) (ann->w-PATCH_W)) {
              int xpp, ypp;
              getnn(ann, x+dx, y, xpp, ypp);
              xpp -= dx;

              if ((xpp != xbest || ypp != ybest) &&
                  (unsigned) xpp < (unsigned) (b->w-PATCH_W+1) &&
                  (!IS_MASK ||
                    ((!region_masks || ((int *) region_masks->bmp->line[ypp])[xpp] == src_mask) &&
                     (!bmask || !((int *) bmask->line[ypp])[xpp]) &&
                     (!amask || !((int *) amask->bmp->line[y])[x+dx]))
                   )) 
							{
                // faster way to calculate error with known error( Neighbor(ax,ay), MatchInB(Neighbor(ax,ay)) )
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
                  partial +=  dr34*dr34+dg34*dg34+db34*db34
                             -dr12*dr12-dg12*dg12-db12*db12;
                }
                err0 += (dx < 0) ? partial: -partial; 
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
                  (unsigned) ypp < (unsigned) (b->h-PATCH_W+1) &&
                  (!IS_MASK || 
                    ((!region_masks || ((int *) region_masks->bmp->line[ypp])[xpp] == src_mask) &&
                     (!bmask || !((int *) bmask->line[ypp])[xpp]) &&
                     (!amask || !((int *) amask->bmp->line[y+dy])[x]))
                  )) {
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
                  partial +=  dr34*dr34+dg34*dg34+db34*db34
                             -dr12*dr12-dg12*dg12-db12*db12;
                }
                err0 += (dy < 0) ? partial: -partial;
                if (err0 < err) {
                  err = err0;
                  xbest = xpp;
                  ybest = ypp;
                }
              }
            }
          } 
					else {
            /* Propagate x */
            if ((unsigned) (x+dx) < (unsigned) (ann->w-PATCH_W)) {
              int xpp, ypp;
              getnn(ann, x+dx, y, xpp, ypp);
              xpp -= dx;

              if (!IS_WINDOW || window_constraint(p, a, b, x, y, xpp, ypp, ann_window, awinsize)) {
                attempt_n<PATCH_W, IS_MASK, IS_WINDOW>(err, xbest, ybest, adata, b, xpp, ypp, bmask, region_masks, src_mask, p);
              }
            }

            /* Propagate y */
            if ((unsigned) (y+dy) < (unsigned) (ann->h-PATCH_W)) {
              int xpp, ypp;
              getnn(ann, x, y+dy, xpp, ypp);
              ypp -= dy;

              if (!IS_WINDOW || window_constraint(p, a, b, x, y, xpp, ypp, ann_window, awinsize)) {
                attempt_n<PATCH_W, IS_MASK, IS_WINDOW>(err, xbest, ybest, adata, b, xpp, ypp, bmask, region_masks, src_mask, p);
              }
            }
          }
        }

        /* Random search */
        unsigned int seed = (x | (y<<11)) ^ iter_seed;
        seed = RANDI(seed);
        int rs_iters = 1-(seed*(1.0/(RAND_MAX-1))) < rs_fpart ? rs_ipart + 1: rs_ipart;

        int rs_max_curr = rs_max;
        for (int mag = rs_max_curr; mag >= p->rs_min; mag = int(mag*p->rs_ratio) /*mag > 1 ? 1: 0*/) {
          for (int rs_iter = 0; rs_iter < rs_iters; rs_iter++) {
            int xmin = max(xbest-mag,0), xmax = min(xbest+mag+1,bew);
            int ymin = max(ybest-mag,0), ymax = min(ybest+mag+1,beh);
            seed = RANDI(seed);
            int xpp = xmin+seed%(xmax-xmin);
            seed = RANDI(seed);
            int ypp = ymin+seed%(ymax-ymin);
            if (!IS_WINDOW || window_constraint(p, a, b, x, y, xpp, ypp, ann_window, awinsize)) {
              attempt_n<PATCH_W, IS_MASK, IS_WINDOW>(err, xbest, ybest, adata, b, xpp, ypp, bmask, region_masks, src_mask, p);
            }
          }
        }
        
        ((int *) ann->line[y])[x] = XY_TO_INT(xbest, ybest);
        ((int *) annd->line[y])[x] = err;
      }
    }
  }
  printf("done nn_n, did %d iters, rs_max=%d\n", nn_iter, p->rs_max);
}

/* GPU algorithm implemented on CPU. */
template<int PATCH_W, int IS_MASK, int IS_WINDOW>
void nn_n_gpucpu(Params *p, BITMAP *a, BITMAP *b,
          BITMAP *ann, BITMAP *annd,
          RegionMasks *amask, BITMAP *bmask,
          int level, int em_iter, RecomposeParams *rp, int offset_iter, int update_type, RegionMasks *region_masks, int tiles) {
  if (IS_WINDOW) { fprintf(stderr, "window_w,h, weight_r,g,b not supported for algorithm GPUCPU\n"); exit(1); }
  printf("in nn_n_gpucpu, masks are: %p %p %p\n", amask, bmask, region_masks);
  Box box = get_abox(p, a, amask);
  BITMAP *ann0 = ann, *annd0 = annd;
  BITMAP *ann_out = create_bitmap(ann->w, ann->h);
  BITMAP *annd_out = create_bitmap(annd->w, annd->h);
  BITMAP *ann_out0 = ann_out, *annd_out0 = annd_out;
  BITMAP *last_iter = NULL;
  for (int nn_iter = 0; nn_iter < p->nn_iters; nn_iter++) {
    int allow_rs = nn_iter < p->nn_iters - 1;

    int bew = b->w-PATCH_W, beh = b->h-PATCH_W;
    int max_mag = max(b->w, b->h);
    int rs_ipart = int(p->rs_iters);
    double rs_fpart = p->rs_iters - rs_ipart;
    int rs_max = p->rs_max;
    if (rs_max > max_mag) { rs_max = max_mag; }

    for (int jump = p->gpu_prop; jump >= 1; jump /= 2) {
      int last = (jump == 1 && nn_iter == p->nn_iters - 1);
      //printf("%d %d last: %d\n", nn_iter, jump, last);
      unsigned int iter_seed = rand();
      #pragma omp parallel for schedule(dynamic, 10)
      for (int y = box.ymin; y < box.ymax; y++) {
        int adata[PATCH_W*PATCH_W];
        int *annd_row = (int *) annd->line[y];
        int *amask_row = IS_MASK ? (amask ? (int *) amask->bmp->line[y]: NULL): NULL;
        for (int x = box.xmin; x < box.xmax; x++) {
          if (IS_MASK && amask && amask_row[x]) { continue; }

          for (int dy0 = 0; dy0 < PATCH_W; dy0++) {
            int *drow = ((int *) a->line[y+dy0])+x;
            int *adata_row = adata+(dy0*PATCH_W);
            for (int dx0 = 0; dx0 < PATCH_W; dx0++) {
              adata_row[dx0] = drow[dx0];
            }
          }
          
          int src_mask = IS_MASK ? (region_masks ? ((int *) region_masks->bmp->line[y])[x]: 0): 0;
          
          int xbest, ybest;
          getnn(ann, x, y, xbest, ybest);
          int xbest0 = xbest, ybest0 = ybest;
          int err = annd_row[x];
          if (err == 0) {
            continue;
          }

          /* Propagate +x */
          if ((unsigned) (x+jump) < (unsigned) (ann->w-PATCH_W)) {
            int idx = ((int *) ann->line[y])[x+jump];
            if (!last_iter || ((int *) last_iter->line[y])[x+jump] != idx) {
              int xpp = INT_TO_X(idx), ypp = INT_TO_Y(idx);
              xpp -= jump;

              attempt_n<PATCH_W, IS_MASK, IS_WINDOW>(err, xbest, ybest, adata, b, xpp, ypp, bmask, region_masks, src_mask, p);
            }
          }

          /* Propagate +y */
          if ((unsigned) (y+jump) < (unsigned) (ann->h-PATCH_W)) {
            int idx = ((int *) ann->line[y+jump])[x];
            if (!last_iter || ((int *) last_iter->line[y+jump])[x] != idx) {
              int xpp = INT_TO_X(idx), ypp = INT_TO_Y(idx);
              ypp -= jump;

              attempt_n<PATCH_W, IS_MASK, IS_WINDOW>(err, xbest, ybest, adata, b, xpp, ypp, bmask, region_masks, src_mask, p);
            }
          }

          /* Propagate -x */
          if (x >= jump) {
            int idx = ((int *) ann->line[y])[x-jump];
            if (!last_iter || ((int *) last_iter->line[y])[x-jump] != idx) {
              int xpp = INT_TO_X(idx), ypp = INT_TO_Y(idx);
              xpp += jump;

              attempt_n<PATCH_W, IS_MASK, IS_WINDOW>(err, xbest, ybest, adata, b, xpp, ypp, bmask, region_masks, src_mask, p);
            }
          }

          /* Propagate -y */
          if (y >= jump) {
            int idx = ((int *) ann->line[y-jump])[x];
            if (!last_iter || ((int *) last_iter->line[y-jump])[x] != idx) {
              int xpp = INT_TO_X(idx), ypp = INT_TO_Y(idx);
              ypp += jump;

              attempt_n<PATCH_W, IS_MASK, IS_WINDOW>(err, xbest, ybest, adata, b, xpp, ypp, bmask, region_masks, src_mask, p);
            }
          }
        
          if (jump == 1 && allow_rs) {
            unsigned int seed = (x | (y<<11)) ^ iter_seed;
            seed = RANDI(seed);
            /* Random search */
            int rs_iters = 1-(seed*(1.0/(RAND_MAX-1))) < rs_fpart ? rs_ipart + 1: rs_ipart;

            for (int mag = rs_max; mag >= p->rs_min; mag = int(mag*p->rs_ratio)) {
              for (int rs_iter = 0; rs_iter < rs_iters; rs_iter++) {
                int xmin = max(xbest-mag,0), xmax = min(xbest+mag+1,bew);
                int ymin = max(ybest-mag,0), ymax = min(ybest+mag+1,beh);
//                unsigned int seed = seed0 ^ (mag << 22);
                seed = RANDI(seed);
                int xpp = xmin+seed%(xmax-xmin);
                seed = RANDI(seed);
                int ypp = ymin+seed%(ymax-ymin);
                attempt_n<PATCH_W, IS_MASK, IS_WINDOW>(err, xbest, ybest, adata, b, xpp, ypp, bmask, region_masks, src_mask, p);
              }
            }
          }

          ((int *) ann_out->line[y])[x] = XY_TO_INT(xbest, ybest);
          ((int *) annd_out->line[y])[x] = err;
        }
      } /* Loop over y. */
      swap(ann, ann_out);
      swap(annd, annd_out);
      /* At this point, ann and annd have just been written to. */
    } /* Loop over jump. */
    if (last_iter) { destroy_bitmap(last_iter); }
    if (nn_iter < p->nn_iters - 1) { last_iter = copy_image(ann_out); } else { last_iter = NULL; }
  }
  if (last_iter) { destroy_bitmap(last_iter); }
  //check_nn(p, ann, b);
  //check_dists(p, a, b, ann, annd, 0);
  if (ann != ann0) {
    blit(ann, ann0, 0, 0, 0, 0, ann->w, ann->h);
    blit(annd, annd0, 0, 0, 0, 0, annd->w, annd->h);
  }
  destroy_bitmap(ann_out0);
  destroy_bitmap(annd_out0);
  //check_nn(p, ann0, b);
  //check_dists(p, a, b, ann0, annd0, 0);
}

/* for multi-core */
template<int PATCH_W, int IS_MASK, int IS_WINDOW>
void nn_n_cputiled(Params *p, BITMAP *a, BITMAP *b,
          BITMAP *ann, BITMAP *annd,
          RegionMasks *amask, BITMAP *bmask,
          int level, int em_iter, RecomposeParams *rp, int offset_iter, int update_type, RegionMasks *region_masks, int tiles,		  
					BITMAP *ann_window, BITMAP *awinsize) 
{

  if (tiles < 0) { tiles = p->cores; }
  printf("in nn_n_cputiled, masks are: %p %p %p, tiles=%d, rs_max=%d\n", amask, bmask, region_masks, tiles, p->rs_max);
  Box box = get_abox(p, a, amask);
  int nn_iter = 0;
  for (; nn_iter < p->nn_iters; nn_iter++) {
    unsigned int iter_seed = rand();

    #pragma omp parallel num_threads(tiles)
    {
#if SYNC_WRITEBACK
      int *ann_writeback = new int[a->w];
      int *annd_writeback = new int[a->w];
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
      int max_mag = max(b->w, b->h);
      int rs_ipart = int(p->rs_iters);
      double rs_fpart = p->rs_iters - rs_ipart;
      int rs_max = p->rs_max;
      if (rs_max > max_mag) { rs_max = max_mag; }

      int adata[PATCH_W*PATCH_W];
      for (int y = ystart; y != yfinal; y += ychange) {
        int *annd_row = (int *) annd->line[y];
        int *amask_row = IS_MASK ? (amask ? (int *) amask->bmp->line[y]: NULL): NULL;
        for (int x = xstart; x != xfinal; x += xchange) {
          if (IS_MASK && amask && amask_row[x]) { continue; }

          for (int dy0 = 0; dy0 < PATCH_W; dy0++) {
            int *drow = ((int *) a->line[y+dy0])+x;
            int *adata_row = adata+(dy0*PATCH_W);
            for (int dx0 = 0; dx0 < PATCH_W; dx0++) {
              adata_row[dx0] = drow[dx0];
            }
          }
          
          int src_mask = IS_MASK ? (region_masks ? ((int *) region_masks->bmp->line[y])[x]: 0): 0;
          
          int xbest, ybest;
          getnn(ann, x, y, xbest, ybest);
          int err = annd_row[x];
          if (err == 0) {
#if SYNC_WRITEBACK
            if (y+ychange == yfinal) {
              ann_writeback[x] = XY_TO_INT(xbest, ybest);
              annd_writeback[x] = err;
            }
#endif
            continue;
          }

          /* Propagate */
          if (p->do_propagate) {
            if (!IS_WINDOW) {
              /* Propagate x */
              if ((unsigned) (x+dx) < (unsigned) (ann->w-PATCH_W)) {
                int xpp, ypp;
                getnn(ann, x+dx, y, xpp, ypp);
                xpp -= dx;
                if ((xpp != xbest || ypp != ybest) &&
                    (unsigned) xpp < (unsigned) (b->w-PATCH_W+1) &&
                    (!IS_MASK ||
                      ((!region_masks || ((int *) region_masks->bmp->line[ypp])[xpp] == src_mask) &&
                       (!bmask || !((int *) bmask->line[ypp])[xpp]) &&
                       (!amask || !((int *) amask->bmp->line[y])[x+dx]))
                     )) 
								{
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
                    partial +=  dr34*dr34+dg34*dg34+db34*db34
                               -dr12*dr12-dg12*dg12-db12*db12;
                  }
                  err0 += (dx < 0) ? partial: -partial;
                  if (err0 < err) {
                    err = err0;
                    xbest = xpp;
                    ybest = ypp;
                  }
                }
              } // end of propagate x

              /* Propagate y */
              if ((unsigned) (y+dy) < (unsigned) (ann->h-PATCH_W)) {
                int xpp, ypp;
                getnn(ann, x, y+dy, xpp, ypp);
                ypp -= dy;
                if ((xpp != xbest || ypp != ybest) &&
                    (unsigned) ypp < (unsigned) (b->h-PATCH_W+1) &&
                    (!IS_MASK || 
                      ((!region_masks || ((int *) region_masks->bmp->line[ypp])[xpp] == src_mask) &&
                       (!bmask || !((int *) bmask->line[ypp])[xpp]) &&
                       (!amask || !((int *) amask->bmp->line[y+dy])[x]))
                    )) 
								{
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
                    partial +=  dr34*dr34+dg34*dg34+db34*db34
                               -dr12*dr12-dg12*dg12-db12*db12;
                  }
                  err0 += (dy < 0) ? partial: -partial;
                  if (err0 < err) {
                    err = err0;
                    xbest = xpp;
                    ybest = ypp;
                  }
                }
              }
            } // end IS_WINDOW = false
						else { // IS_WINDOW = true
              
							/* Propagate x */
              if ((unsigned) (x+dx) < (unsigned) (ann->w-PATCH_W)) {
                int xpp, ypp;
                getnn(ann, x+dx, y, xpp, ypp);
                xpp -= dx;

                if (!IS_WINDOW || window_constraint(p, a, b, x, y, xpp, ypp, ann_window, awinsize)) {
                  attempt_n<PATCH_W, IS_MASK, IS_WINDOW>(err, xbest, ybest, adata, b, xpp, ypp, bmask, region_masks, src_mask, p);
                }
              }

              /* Propagate y */
              if ((unsigned) (y+dy) < (unsigned) (ann->h-PATCH_W)) {
                int xpp, ypp;
                getnn(ann, x, y+dy, xpp, ypp);
                ypp -= dy;

                if (!IS_WINDOW || window_constraint(p, a, b, x, y, xpp, ypp, ann_window, awinsize)) {
                  attempt_n<PATCH_W, IS_MASK, IS_WINDOW>(err, xbest, ybest, adata, b, xpp, ypp, bmask, region_masks, src_mask, p);
                }
              }
            }
          }

          /* Random search */
          unsigned int seed = (x | (y<<11)) ^ iter_seed;
          seed = RANDI(seed);
          int rs_iters = 1-(seed*(1.0/(RAND_MAX-1))) < rs_fpart ? rs_ipart + 1: rs_ipart;
  //        int rs_iters = 1-random() < rs_fpart ? rs_ipart + 1: rs_ipart;

          int rs_max_curr = rs_max;
          for (int mag = rs_max_curr; mag >= p->rs_min; mag = int(mag*p->rs_ratio)) {
            for (int rs_iter = 0; rs_iter < rs_iters; rs_iter++) {
              int xmin = max(xbest-mag,0), xmax = min(xbest+mag+1, bew);
              int ymin = max(ybest-mag,0), ymax = min(ybest+mag+1, beh);              
              seed = RANDI(seed);
              int xpp = xmin+seed%(xmax-xmin);
              seed = RANDI(seed);
              int ypp = ymin+seed%(ymax-ymin);
              //int xpp = xmin+rand()%(xmax-xmin);
              //int ypp = ymin+rand()%(ymax-ymin);
              if (!IS_WINDOW || window_constraint(p, a, b, x, y, xpp, ypp, ann_window, awinsize)) {
                attempt_n<PATCH_W, IS_MASK, IS_WINDOW>(err, xbest, ybest, adata, b, xpp, ypp, bmask, region_masks, src_mask, p);
              }
            }
          }

#if SYNC_WRITEBACK
          if (y+ychange != yfinal) {     
#endif
          ((int *) ann->line[y])[x] = XY_TO_INT(xbest, ybest);
          ((int *) annd->line[y])[x] = err;
#if SYNC_WRITEBACK
          } else {
            ann_writeback[x] = XY_TO_INT(xbest, ybest);
            annd_writeback[x] = err;
          }
#endif

        } // x
      } // y

#if SYNC_WRITEBACK
      #pragma omp barrier
      int ywrite = yfinal-ychange;
      if (ymin < ymax && (unsigned) ywrite < (unsigned) AEH) {
        int *ann_line = (int *) ann->line[ywrite];
        int *annd_line = (int *) annd->line[ywrite];
        for (int x = xmin; x < xmax; x++) {
          ann_line[x] = ann_writeback[x];
          annd_line[x] = annd_writeback[x];
        }
      }
      delete[] ann_writeback;
      delete[] annd_writeback;
#endif
    } // parallel
  } // nn_iter
  printf("done nn_n_cputiled, %d iters, rs_max=%d\n", nn_iter, p->rs_max);
}

/* only propogation, no random search */
template<int PATCH_W>
void nn_n_proponly(Params *p, BITMAP *a, BITMAP *b,
          BITMAP *ann, BITMAP *annd,
          RegionMasks *amask, BITMAP *bmask,
          int level, int em_iter, RecomposeParams *rp, int offset_iter, int update_type, RegionMasks *region_masks, int tiles) {
  if (tiles < 0) { tiles = p->cores; }
  printf("in nn_n_proponly, masks are: %p %p %p, tiles=%d\n", amask, bmask, region_masks, tiles);
  Box box = get_abox(p, a, amask);
  for (int nn_iter = 0; nn_iter < p->nn_iters; nn_iter++) {
    unsigned int iter_seed = rand();

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

      for (int y = ystart; y != yfinal; y += ychange) {
        int *annd_row = (int *) annd->line[y];
        for (int x = xstart; x != xfinal; x += xchange) {
          int xbest, ybest;
          getnn(ann, x, y, xbest, ybest);
          int err = annd_row[x];
          if (err == 0) { continue; }

          /* Propagate x */
          if ((unsigned) (x+dx) < (unsigned) (ann->w-PATCH_W)) { // with the trick of using unsigned, pixels on the boundary won't be checked 
            int xpp, ypp;
            getnn(ann, x+dx, y, xpp, ypp);
            xpp -= dx;

            if ((xpp != xbest || ypp != ybest) &&
                (unsigned) xpp < (unsigned) (b->w-PATCH_W+1)) {
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
                partial +=  dr34*dr34+dg34*dg34+db34*db34
                           -dr12*dr12-dg12*dg12-db12*db12;
              }
              err0 += (dx < 0) ? partial: -partial;
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
                partial +=  dr34*dr34+dg34*dg34+db34*db34
                           -dr12*dr12-dg12*dg12-db12*db12;
              }
              err0 += (dy < 0) ? partial: -partial;
              if (err0 < err) {
                err = err0;
                xbest = xpp;
                ybest = ypp;
              }
            }
          }

          ((int *) ann->line[y])[x] = XY_TO_INT(xbest, ybest);
          ((int *) annd->line[y])[x] = err;
        } // x
      } // y
    } // parallel
  } // nn_iter
  printf("done nn_n_proponly\n");
}

#define NTABLE 2048

double COS_TABLE[NTABLE];
double SIN_TABLE[NTABLE];
double RAD_TABLE[NTABLE];

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif

int gauss_table_init = 0;
void init_gauss_table() {
  if (gauss_table_init) { return; }
  gauss_table_init = 1;
  for (int i = 0; i < NTABLE; i++) {
    double r = i*1.0/NTABLE;
    COS_TABLE[i] = cos(r*2.0*M_PI);
    SIN_TABLE[i] = sin(r*2.0*M_PI);
    RAD_TABLE[i] = sqrt(-2.0 * log(1.0 - r));
  }
}

double gauss1d(double mu, double sigma) {
  /* (From Python docs)
     When x and y are two variables from [0, 1), uniformly distributed, then
       cos(2*pi*x)*sqrt(-2*log(1-y))
       sin(2*pi*x)*sqrt(-2*log(1-y))
     are two *independent* variables with normal dist. (mu = 0, sigma = 1). */
  return mu + COS_TABLE[rand()&(NTABLE-1)] * RAD_TABLE[rand()&(NTABLE-1)] * sigma;
}

void sample_gaussian(int &dx, int &dy, double sigma) {
  for (;;) {
    int itheta = rand()&(NTABLE-1);
    double r = gauss1d(0, sigma);
    dx = (int) floor(COS_TABLE[itheta]*r+0.5);
    dy = (int) floor(SIN_TABLE[itheta]*r+0.5);
    if (dx != 0 || dy != 0) { break; }
  }
}

/* Fully randomized algorithm. */
template<int PATCH_W, int IS_MASK, int IS_WINDOW>
void nn_n_fullrand(Params *p, BITMAP *a, BITMAP *b,
          BITMAP *ann, BITMAP *annd,
          RegionMasks *amask, BITMAP *bmask,
          int level, int em_iter, RecomposeParams *rp, int offset_iter, int update_type, RegionMasks *region_masks, int tiles) {
  if (IS_WINDOW) { fprintf(stderr, "window_w,h, weight_r,g,b not supported for algorithm FULLRAND\n"); exit(1); }
  printf("in nn_n_fullrand, masks are: %p %p %p, tiles=%d\n", amask, bmask, region_masks, tiles);
  init_gauss_table();

  Box box = get_abox(p, a, amask);
  int xmin = box.xmin, ymin = box.ymin, xmax = box.xmax, ymax = box.ymax;
  int npixels = (ymax-ymin)*(xmax-xmin)*p->nn_iters;

  int bew = b->w-PATCH_W, beh = b->h-PATCH_W;
  int max_mag = max(b->w, b->h);
  int rs_ipart = int(p->rs_iters);
  double rs_fpart = p->rs_iters - rs_ipart;
  int rs_max = p->rs_max;
  if (rs_max > max_mag) { rs_max = max_mag; }

  double P_prop = 0.56;
  double sigma_prop = 6.3;
  double sigma_rs = 4.5;
  
  vector<int> rs_list;
  for (int mag = rs_max; mag >= p->rs_min; mag = int(mag*p->rs_ratio)) {
    rs_list.push_back(mag);
  }
  
  #pragma omp parallel for schedule(static, 128)
  for (int ipixel = 0; ipixel < npixels; ipixel++) {
    int adata[PATCH_W*PATCH_W];
    int seed = ((ipixel<<2)^(ipixel<<17))+1;
    int x = xmin+rand()%(xmax-xmin), y = ymin+rand()%(ymax-ymin);
    int *amask_row = IS_MASK ? (amask ? (int *) amask->bmp->line[y]: NULL): NULL;
    if (IS_MASK && amask && amask_row[x]) { continue; }

    for (int dy0 = 0; dy0 < PATCH_W; dy0++) {
      int *drow = ((int *) a->line[y+dy0])+x;
      int *adata_row = adata+(dy0*PATCH_W);
      for (int dx0 = 0; dx0 < PATCH_W; dx0++) {
        adata_row[dx0] = drow[dx0];
      }
    }
    
    int src_mask = IS_MASK ? (region_masks ? ((int *) region_masks->bmp->line[y])[x]: 0): 0;
    
    int xbest, ybest, err;
    #pragma omp critical
    {
      getnn(ann, x, y, xbest, ybest);
      err = ((int *) annd->line[y])[x];
    }
    if (err == 0) { continue; }

    if (random() < P_prop) {
      /* Propagation */
      int dx, dy;
      sample_gaussian(dx, dy, sigma_prop);
      if ((unsigned) (x+dx) < (unsigned) (ann->w-PATCH_W) && (unsigned) (y+dy) < (unsigned) (ann->h-PATCH_W)) {
        int xpp, ypp;
        #pragma omp critical
        {
          getnn(ann, x+dx, y+dy, xpp, ypp);
        }
        xpp -= dx;
        ypp -= dy;

        attempt_n<PATCH_W, IS_MASK, IS_WINDOW>(err, xbest, ybest, adata, b, xpp, ypp, bmask, region_masks, src_mask, p);
      }
    } else {
      /* Random search */
      int dx, dy;
      sample_gaussian(dx, dy, sigma_rs);
      int xpp = xbest+dx, ypp = ybest+dy;
      attempt_n<PATCH_W, IS_MASK, IS_WINDOW>(err, xbest, ybest, adata, b, xpp, ypp, bmask, region_masks, src_mask, p);
    }
    
    #pragma omp critical
    {
      ((int *) ann->line[y])[x] = XY_TO_INT(xbest, ybest);
      ((int *) annd->line[y])[x] = err;
    }
  }
  printf("done nn_n_fullrand\n");
}

/* ----------------------------------------------------------------
   Ordinary code
   ---------------------------------------------------------------- */

RegionMasks::RegionMasks(Params *p, BITMAP *region_masks, int full, BITMAP *bmask) {
  bmp = region_masks;
  if (!bmp) { fprintf(stderr, "Region mask is NULL\n"); exit(1); }
  for (int i = 0; i < 256; i++) {
    box[i].xmin = box[i].ymin = INT_MAX;
    box[i].xmax = box[i].ymax = -INT_MAX;
  }
  if (full) {
    box[0].xmin = box[0].ymin = 0;
    box[0].xmax = region_masks->w - p->patch_w + 1;
    box[0].ymax = region_masks->h - p->patch_w + 1;
  } 
	else {
    for (int y = 0; y <= region_masks->h - p->patch_w; y++) {
      int *row = (int *) region_masks->line[y];
      for (int x = 0; x <= region_masks->w - p->patch_w; x++) {
        int i = row[x]&255;
        //if (bmask && ((int *) bmask->line[y])[x]) { continue; }
        //if ((unsigned) i > (unsigned) 255) { fprintf(stderr, "Error: region mask index not a uint8 (%d at %d,%d).\n", i, x, y); exit(1); }
        if (x < box[i].xmin) { box[i].xmin = x; }
        if (x >= box[i].xmax) { box[i].xmax = x+1; }
        if (y < box[i].ymin) { box[i].ymin = y; }
        if (y >= box[i].ymax) { box[i].ymax = y+1; }
      }
    }
  }
  for (int i = 0; i < 256; i++) {
    if (box[i].xmin != INT_MAX) {
      printf("%d => %d %d %d %d\n", i, box[i].xmin, box[i].ymin, box[i].xmax, box[i].ymax);
    }
  }
}

void destroy_region_masks(RegionMasks *m) {
  if (!m) { return; }
  destroy_bitmap(m->bmp);
  delete m;
}

// returns the half-open range [xmin, xmax)
Box get_abox(Params *p, BITMAP *a, RegionMasks *amask, int trim_patch) {
  if (!amask) {
    Box ans;
    ans.xmin = ans.ymin = 0;
		ans.xmax = trim_patch ? (a->w - p->patch_w + 1) : a->w; 
		ans.ymax = trim_patch ? (a->h - p->patch_w + 1) : a->h;
    return ans;
  }
  Box ans = amask->box[0];
  //save_bitmap("amask.bmp", amask->bmp, NULL);
  if (ans.xmin < 0 || ans.ymin < 0 || ans.xmax > a->w-p->patch_w+1 || ans.ymax > a->h-p->patch_w+1) { fprintf(stderr, "box out of range %d %d %d %d (%d %d %d %d)\n", ans.xmin, ans.ymin, ans.xmax, ans.ymax, 0, 0, a->w-p->patch_w+1, a->h-p->patch_w+1); exit(1); }
  if (ans.xmin >= ans.xmax || ans.ymin >= ans.ymax) { ans.xmin = ans.ymin = 0; ans.xmax = ans.ymax = 1; } //fprintf(stderr, "box size 0 (%d %d %d %d)\n", ans.xmin, ans.ymin, ans.xmax, ans.ymax); exit(1); }
  // FIXME: Instead, set box size to 1 at (0,0) if it has size zero
  printf("get_abox (%dx%d) => %d %d %d %d\n", a->w, a->h, ans.xmin, ans.ymin, ans.xmax, ans.ymax);
  return ans;
}

#define MAX_NN_GUESS_ITERS 32

/* same as the meaning of IS_WINDOW */
int is_window(Params *p) {
  return p->window_w < INT_MAX || p->window_h < INT_MAX || p->weight_r != 1 || p->weight_g != 1 || p->weight_b != 1;
}

/* TODO: We can use a BITMAP * for region_masks now, no need to calculate the bboxes in RegionMasks constructor. */
/* initialize the NN field */
BITMAP *init_nn(Params *p, BITMAP *a, BITMAP *b, BITMAP *bmask, RegionMasks *region_masks, RegionMasks *amask, int trim_patch, BITMAP *ann_window, BITMAP *awinsize) {
  BITMAP *bmp = create_bitmap(a->w, a->h);
  clear(bmp);
	int ew = trim_patch ? (b->w - p->patch_w + 1) : b->w;
	int eh = trim_patch ? (b->h - p->patch_w + 1) : b->h;
  printf("init_nn: ew=%d, eh=%d\n", ew, eh);

  Box box = get_abox(p, a, amask);

  if (region_masks) {
    if (a->w != b->w || a->h != b->h || a->w != region_masks->bmp->w || a->h != region_masks->bmp->h) { 
			fprintf(stderr, "Size differs in init_nn, with region masks (%dx%d, %dx%d, %dx%d)\n", a->w, a->h, b->w, b->h, region_masks->bmp->w, region_masks->bmp->h); 
			exit(1); 
		}
  }

  if (is_window(p)) {
    /* Use window and mask for initial guess constraints. */
    vector<int> sample[256];
    if (!region_masks) { sample[0].reserve(ew*eh); }
    int id0 = 0;
    for (int y = 0; y < eh; y++) {
      int *bmask_row = bmask ? (int *) bmask->line[y]: NULL;
      int *rmask_row = region_masks ? (int *) region_masks->bmp->line[y]: NULL;
      for (int x = 0; x < ew; x++) {
        if (bmask_row && bmask_row[x]) { continue; }
        id0 = rmask_row ? rmask_row[x]: 0;
        sample[id0].push_back(XY_TO_INT(x, y));
      }
    }

    int warned = 0;
    int max_iters = 64; // try at most max_iters times
    for (int y = box.ymin; y < box.ymax; y++) {
      int *row = (int *) bmp->line[y];
      int *arow = amask ? (int *) amask->bmp->line[y]: NULL;
      int *rmask_row = region_masks ? (int *) region_masks->bmp->line[y]: NULL;
      for (int x = box.xmin; x < box.xmax; x++) {
        if (amask && arow[x]) { continue; }
        int id = rmask_row ? rmask_row[x]: 0;
        if (sample[id].size() == 0) {
          if (!warned) { warned = 1; fprintf(stderr, "Warning: No matching index for color index %d in window\n", id); }
          id = id0;
          if (sample[id].size() == 0) { row[x] = 0; continue; }
        }
				int idx = -1, iter = 0;
				int xdest = -1, ydest = -1;
        for (; iter < max_iters; iter++) {
          idx = rand() % sample[id].size();
          xdest = INT_TO_X(sample[id][idx]), ydest = INT_TO_Y(sample[id][idx]);
          if (window_constraint(p, a, b, x, y, xdest, ydest, ann_window, awinsize)) { break; }
        }
				if (iter>=max_iters) {
					if (ann_window) {
					// adopt the prior field
						getnn(ann_window, x, y, xdest, ydest); 
					} 
					else {
					// adopt identity field (linear transform if different dimensions)
						xdest = x * b->w / a->w;
						ydest = y * b->h / a->h;
					}
					row[x] = XY_TO_INT(xdest, ydest);
					continue;
				}
        if (idx < 0) {
          if (!warned) { warned = 1; fprintf(stderr, "Warning: No matching index for color index %d in window\n", id); }
          id = id0;
          if (sample[id].size() == 0) { row[x] = 0; continue; }
        }
        row[x] = sample[id][idx];
      }
    }
  } 
	else if (!bmask && !region_masks) {
    //fprintf(stderr, "init_nn openmp\n");
    #pragma omp parallel for schedule(static, 8)
    for (int y = box.ymin; y < box.ymax; y++) {
      unsigned int seed = rand();
      int *row = (int *) bmp->line[y];
      int *arow = amask ? (int *) amask->bmp->line[y]: NULL;
      for (int x = box.xmin; x < box.xmax; x++) {
        if (amask && arow[x]) { continue; }
        int xp = 0, yp = 0;
        //xp = rand()%ew;
        //yp = rand()%eh;
        xp = seed % ew;
        seed = RANDI(seed);
        yp = seed % eh;
        seed = RANDI(seed);
        //if (iter >= MAX_NN_GUESS_ITERS) { fprintf(stderr, "Warning: too many iters at %d,%d\n", x, y); }
        //if (x == 1 && y == 1) { printf("1, 1 => %d %d\n", xp, yp); }
        row[x] = XY_TO_INT(xp, yp);
      }
    }
  } 
	else {
    vector<int> sample[256];
    if (!region_masks) { sample[0].reserve(ew*eh); }
    int id0 = 0;
    for (int y = 0; y < eh; y++) {
      int *bmask_row = bmask ? (int *) bmask->line[y]: NULL;
      int *rmask_row = region_masks ? (int *) region_masks->bmp->line[y]: NULL;
      for (int x = 0; x < ew; x++) {
        if (bmask_row && bmask_row[x]) { continue; }
        id0 = rmask_row ? rmask_row[x]: 0;
        sample[id0].push_back(XY_TO_INT(x, y));
      }
    }

    int warned = 0;
    for (int y = box.ymin; y < box.ymax; y++) {
      int *row = (int *) bmp->line[y];
      int *arow = amask ? (int *) amask->bmp->line[y]: NULL;
      int *rmask_row = region_masks ? (int *) region_masks->bmp->line[y]: NULL;
      for (int x = box.xmin; x < box.xmax; x++) {
        if (amask && arow[x]) { continue; }
        int id = rmask_row ? rmask_row[x]: 0;
        if (sample[id].size() == 0) {
          if (!warned) { warned = 1; fprintf(stderr, "Warning: No matching index for color index %d\n", id); }
          id = id0;
          if (sample[id].size() == 0) { row[x] = 0; continue; }
        }
        int idx = rand() % sample[id].size();
        row[x] = sample[id][idx];
      }
    }
  }

//  printf("done init_nn\n");

  return bmp;
}

BITMAP *init_dist(Params *p, BITMAP *a, BITMAP *b, BITMAP *ann, BITMAP *bmask, RegionMasks *region_masks, RegionMasks *amask) {
  BITMAP *ans = NULL;

  if (is_window(p)) {
    if      (p->patch_w == 1) { ans = init_dist_n<1,1,1>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 2) { ans = init_dist_n<2,1,1>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 3) { ans = init_dist_n<3,1,1>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 4) { ans = init_dist_n<4,1,1>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 5) { ans = init_dist_n<5,1,1>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 6) { ans = init_dist_n<6,1,1>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 7) { ans = init_dist_n<7,1,1>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 8) { ans = init_dist_n<8,1,1>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 9) { ans = init_dist_n<9,1,1>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 10) { ans = init_dist_n<10,1,1>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 11) { ans = init_dist_n<11,1,1>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 12) { ans = init_dist_n<12,1,1>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 13) { ans = init_dist_n<13,1,1>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 14) { ans = init_dist_n<14,1,1>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 15) { ans = init_dist_n<15,1,1>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 16) { ans = init_dist_n<16,1,1>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 17) { ans = init_dist_n<17,1,1>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 18) { ans = init_dist_n<18,1,1>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 19) { ans = init_dist_n<19,1,1>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 20) { ans = init_dist_n<20,1,1>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 21) { ans = init_dist_n<21,1,1>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 22) { ans = init_dist_n<22,1,1>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 23) { ans = init_dist_n<23,1,1>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 24) { ans = init_dist_n<24,1,1>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 25) { ans = init_dist_n<25,1,1>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 26) { ans = init_dist_n<26,1,1>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 27) { ans = init_dist_n<27,1,1>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 28) { ans = init_dist_n<28,1,1>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 29) { ans = init_dist_n<29,1,1>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 30) { ans = init_dist_n<30,1,1>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 31) { ans = init_dist_n<31,1,1>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 32) { ans = init_dist_n<32,1,1>(p, a, b, ann, bmask, region_masks, amask); }
    else { fprintf(stderr, "Patch size unsupported: %d\n", p->patch_w); exit(1); }
  } 
	else if (amask || bmask || region_masks) {
    //printf("init_dist is masked\n");
    if      (p->patch_w == 1) { ans = init_dist_n<1,1,0>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 2) { ans = init_dist_n<2,1,0>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 3) { ans = init_dist_n<3,1,0>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 4) { ans = init_dist_n<4,1,0>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 5) { ans = init_dist_n<5,1,0>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 6) { ans = init_dist_n<6,1,0>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 7) { ans = init_dist_n<7,1,0>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 8) { ans = init_dist_n<8,1,0>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 9) { ans = init_dist_n<9,1,0>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 10) { ans = init_dist_n<10,1,0>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 11) { ans = init_dist_n<11,1,0>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 12) { ans = init_dist_n<12,1,0>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 13) { ans = init_dist_n<13,1,0>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 14) { ans = init_dist_n<14,1,0>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 15) { ans = init_dist_n<15,1,0>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 16) { ans = init_dist_n<16,1,0>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 17) { ans = init_dist_n<17,1,0>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 18) { ans = init_dist_n<18,1,0>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 19) { ans = init_dist_n<19,1,0>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 20) { ans = init_dist_n<20,1,0>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 21) { ans = init_dist_n<21,1,0>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 22) { ans = init_dist_n<22,1,0>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 23) { ans = init_dist_n<23,1,0>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 24) { ans = init_dist_n<24,1,0>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 25) { ans = init_dist_n<25,1,0>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 26) { ans = init_dist_n<26,1,0>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 27) { ans = init_dist_n<27,1,0>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 28) { ans = init_dist_n<28,1,0>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 29) { ans = init_dist_n<29,1,0>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 30) { ans = init_dist_n<30,1,0>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 31) { ans = init_dist_n<31,1,0>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 32) { ans = init_dist_n<32,1,0>(p, a, b, ann, bmask, region_masks, amask); }
		else { fprintf(stderr, "Patch size unsupported: %d\n", p->patch_w); exit(1); }
  } 
	else {
    //printf("init_dist is unmasked\n");
    if      (p->patch_w == 1) { ans = init_dist_n<1,0,0>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 2) { ans = init_dist_n<2,0,0>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 3) { ans = init_dist_n<3,0,0>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 4) { ans = init_dist_n<4,0,0>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 5) { ans = init_dist_n<5,0,0>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 6) { ans = init_dist_n<6,0,0>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 7) { ans = init_dist_n<7,0,0>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 8) { ans = init_dist_n<8,0,0>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 9) { ans = init_dist_n<9,0,0>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 10) { ans = init_dist_n<10,0,0>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 11) { ans = init_dist_n<11,0,0>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 12) { ans = init_dist_n<12,0,0>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 13) { ans = init_dist_n<13,0,0>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 14) { ans = init_dist_n<14,0,0>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 15) { ans = init_dist_n<15,0,0>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 16) { ans = init_dist_n<16,0,0>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 17) { ans = init_dist_n<17,0,0>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 18) { ans = init_dist_n<18,0,0>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 19) { ans = init_dist_n<19,0,0>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 20) { ans = init_dist_n<20,0,0>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 21) { ans = init_dist_n<21,0,0>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 22) { ans = init_dist_n<22,0,0>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 23) { ans = init_dist_n<23,0,0>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 24) { ans = init_dist_n<24,0,0>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 25) { ans = init_dist_n<25,0,0>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 26) { ans = init_dist_n<26,0,0>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 27) { ans = init_dist_n<27,0,0>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 28) { ans = init_dist_n<28,0,0>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 29) { ans = init_dist_n<29,0,0>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 30) { ans = init_dist_n<30,0,0>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 31) { ans = init_dist_n<31,0,0>(p, a, b, ann, bmask, region_masks, amask); }
    else if (p->patch_w == 32) { ans = init_dist_n<32,0,0>(p, a, b, ann, bmask, region_masks, amask); }
		else { fprintf(stderr, "Patch size unsupported: %d\n", p->patch_w); exit(1); }
  }
  return ans;
}

/* Has defaults for params at and after amask. */
void nn(Params *p, BITMAP *a, BITMAP *b,
        BITMAP *ann, BITMAP *annd,
        RegionMasks *amask, BITMAP *bmask,
        int level, int em_iter, RecomposeParams *rp, int offset_iter, int update_type, int cache_b,
        RegionMasks *region_masks, int tiles, BITMAP *ann_window, BITMAP *awinsize) {
  int algo = p->algo;
  if (algo == ALGO_CPU) {
    if (is_window(p)) {
      printf("Running nn, using windowed and masked\n");
      if      (p->patch_w == 1) { nn_n<1,1,1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 2) { nn_n<2,1,1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 3) { nn_n<3,1,1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 4) { nn_n<4,1,1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 5) { nn_n<5,1,1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 6) { nn_n<6,1,1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 7) { nn_n<7,1,1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 8) { nn_n<8,1,1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 9) { nn_n<9,1,1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 10) { nn_n<10,1,1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 11) { nn_n<11,1,1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 12) { nn_n<12,1,1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 13) { nn_n<13,1,1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 14) { nn_n<14,1,1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 15) { nn_n<15,1,1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 16) { nn_n<16,1,1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
			else if (p->patch_w == 17) { nn_n<17,1,1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 18) { nn_n<18,1,1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 19) { nn_n<19,1,1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 20) { nn_n<20,1,1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 21) { nn_n<21,1,1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 22) { nn_n<22,1,1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 23) { nn_n<23,1,1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 24) { nn_n<24,1,1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 25) { nn_n<25,1,1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 26) { nn_n<26,1,1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 27) { nn_n<27,1,1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 28) { nn_n<28,1,1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 29) { nn_n<29,1,1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 30) { nn_n<30,1,1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 31) { nn_n<31,1,1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 32) { nn_n<32,1,1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else { fprintf(stderr, "Patch size unsupported: %d\n", p->patch_w); exit(1); }
    } 
		else if (bmask == NULL && amask == NULL && region_masks == NULL) {
      printf("Running nn, using unmasked\n");
      if      (p->patch_w == 1) { nn_n<1,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 2) { nn_n<2,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 3) { nn_n<3,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 4) { nn_n<4,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 5) { nn_n<5,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 6) { nn_n<6,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 7) { nn_n<7,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 8) { nn_n<8,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 9) { nn_n<9,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 10) { nn_n<10,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 11) { nn_n<11,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 12) { nn_n<12,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 13) { nn_n<13,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 14) { nn_n<14,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 15) { nn_n<15,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 16) { nn_n<16,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 17) { nn_n<17,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 18) { nn_n<18,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 19) { nn_n<19,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 20) { nn_n<20,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 21) { nn_n<21,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 22) { nn_n<22,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 23) { nn_n<23,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 24) { nn_n<24,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 25) { nn_n<25,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 26) { nn_n<26,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 27) { nn_n<27,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 28) { nn_n<28,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 29) { nn_n<29,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 30) { nn_n<30,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 31) { nn_n<31,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 32) { nn_n<32,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else { fprintf(stderr, "Patch size unsupported: %d\n", p->patch_w); exit(1); }
    } 
		else {
      printf("Running nn, using masked\n");
      if      (p->patch_w == 1) { nn_n<1,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 2) { nn_n<2,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 3) { nn_n<3,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 4) { nn_n<4,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 5) { nn_n<5,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 6) { nn_n<6,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 7) { nn_n<7,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 8) { nn_n<8,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 9) { nn_n<9,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 10) { nn_n<10,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 11) { nn_n<11,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 12) { nn_n<12,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 13) { nn_n<13,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 14) { nn_n<14,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 15) { nn_n<15,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 16) { nn_n<16,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 17) { nn_n<17,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 18) { nn_n<18,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 19) { nn_n<19,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 20) { nn_n<20,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 21) { nn_n<21,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 22) { nn_n<22,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 23) { nn_n<23,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 24) { nn_n<24,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 25) { nn_n<25,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 26) { nn_n<26,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 27) { nn_n<27,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 28) { nn_n<28,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 29) { nn_n<29,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 30) { nn_n<30,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 31) { nn_n<31,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 32) { nn_n<32,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else { fprintf(stderr, "Patch size unsupported: %d\n", p->patch_w); exit(1); }
    }
  } 
	else if (algo == ALGO_CPUTILED) {
    if (is_window(p)) {
      printf("Running nn cputiled, using windowed and masked\n");
      if      (p->patch_w == 1) { nn_n_cputiled<1,1,1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 2) { nn_n_cputiled<2,1,1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 3) { nn_n_cputiled<3,1,1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 4) { nn_n_cputiled<4,1,1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 5) { nn_n_cputiled<5,1,1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 6) { nn_n_cputiled<6,1,1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 7) { nn_n_cputiled<7,1,1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 8) { nn_n_cputiled<8,1,1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 9) { nn_n_cputiled<9,1,1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 10) { nn_n_cputiled<10,1,1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 11) { nn_n_cputiled<11,1,1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 12) { nn_n_cputiled<12,1,1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 13) { nn_n_cputiled<13,1,1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 14) { nn_n_cputiled<14,1,1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 15) { nn_n_cputiled<15,1,1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 16) { nn_n_cputiled<16,1,1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 17) { nn_n_cputiled<17,1,1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 18) { nn_n_cputiled<18,1,1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 19) { nn_n_cputiled<19,1,1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 20) { nn_n_cputiled<20,1,1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 21) { nn_n_cputiled<21,1,1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 22) { nn_n_cputiled<22,1,1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 23) { nn_n_cputiled<23,1,1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 24) { nn_n_cputiled<24,1,1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 25) { nn_n_cputiled<25,1,1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 26) { nn_n_cputiled<26,1,1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 27) { nn_n_cputiled<27,1,1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 28) { nn_n_cputiled<28,1,1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 29) { nn_n_cputiled<29,1,1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 30) { nn_n_cputiled<30,1,1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 31) { nn_n_cputiled<31,1,1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 32) { nn_n_cputiled<32,1,1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else { fprintf(stderr, "Patch size unsupported: %d\n", p->patch_w); exit(1); }
    } 
		else if (bmask == NULL && amask == NULL && region_masks == NULL) {
      //printf("Using unmasked\n");
      if (p->rs_max == 0) {
        printf("Running nn cputiled, using propagation only\n");
        if      (p->patch_w == 1) { nn_n_proponly<1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
        else if (p->patch_w == 2) { nn_n_proponly<2>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
        else if (p->patch_w == 3) { nn_n_proponly<3>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
        else if (p->patch_w == 4) { nn_n_proponly<4>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
        else if (p->patch_w == 5) { nn_n_proponly<5>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
        else if (p->patch_w == 6) { nn_n_proponly<6>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
        else if (p->patch_w == 7) { nn_n_proponly<7>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
        else if (p->patch_w == 8) { nn_n_proponly<8>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
        else if (p->patch_w == 9) { nn_n_proponly<9>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
        else if (p->patch_w == 10) { nn_n_proponly<10>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
        else if (p->patch_w == 11) { nn_n_proponly<11>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
        else if (p->patch_w == 12) { nn_n_proponly<12>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
        else if (p->patch_w == 13) { nn_n_proponly<13>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
        else if (p->patch_w == 14) { nn_n_proponly<14>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
        else if (p->patch_w == 15) { nn_n_proponly<15>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
        else if (p->patch_w == 16) { nn_n_proponly<16>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
        else if (p->patch_w == 17) { nn_n_proponly<17>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
        else if (p->patch_w == 18) { nn_n_proponly<18>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
        else if (p->patch_w == 19) { nn_n_proponly<19>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
        else if (p->patch_w == 20) { nn_n_proponly<20>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
        else if (p->patch_w == 21) { nn_n_proponly<21>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
        else if (p->patch_w == 22) { nn_n_proponly<22>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
        else if (p->patch_w == 23) { nn_n_proponly<23>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
        else if (p->patch_w == 24) { nn_n_proponly<24>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
        else if (p->patch_w == 25) { nn_n_proponly<25>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
        else if (p->patch_w == 26) { nn_n_proponly<26>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
        else if (p->patch_w == 27) { nn_n_proponly<27>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
        else if (p->patch_w == 28) { nn_n_proponly<28>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
        else if (p->patch_w == 29) { nn_n_proponly<29>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
        else if (p->patch_w == 30) { nn_n_proponly<30>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
        else if (p->patch_w == 31) { nn_n_proponly<31>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
        else if (p->patch_w == 32) { nn_n_proponly<32>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
        else { fprintf(stderr, "Patch size unsupported: %d\n", p->patch_w); exit(1); }
      } 
			else {
        printf("Running nn cputiled, no windows or masks\n");
        if      (p->patch_w == 1) { nn_n_cputiled<1,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
        else if (p->patch_w == 2) { nn_n_cputiled<2,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
        else if (p->patch_w == 3) { nn_n_cputiled<3,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
        else if (p->patch_w == 4) { nn_n_cputiled<4,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
        else if (p->patch_w == 5) { nn_n_cputiled<5,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
        else if (p->patch_w == 6) { nn_n_cputiled<6,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
        else if (p->patch_w == 7) { nn_n_cputiled<7,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
        else if (p->patch_w == 8) { nn_n_cputiled<8,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
        else if (p->patch_w == 9) { nn_n_cputiled<9,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
        else if (p->patch_w == 10) { nn_n_cputiled<10,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
        else if (p->patch_w == 11) { nn_n_cputiled<11,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
        else if (p->patch_w == 12) { nn_n_cputiled<12,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
        else if (p->patch_w == 13) { nn_n_cputiled<13,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
        else if (p->patch_w == 14) { nn_n_cputiled<14,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
        else if (p->patch_w == 15) { nn_n_cputiled<15,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
        else if (p->patch_w == 16) { nn_n_cputiled<16,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
        else if (p->patch_w == 17) { nn_n_cputiled<17,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
        else if (p->patch_w == 18) { nn_n_cputiled<18,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
        else if (p->patch_w == 19) { nn_n_cputiled<19,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
        else if (p->patch_w == 20) { nn_n_cputiled<20,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
        else if (p->patch_w == 21) { nn_n_cputiled<21,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
        else if (p->patch_w == 22) { nn_n_cputiled<22,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
        else if (p->patch_w == 23) { nn_n_cputiled<23,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
        else if (p->patch_w == 24) { nn_n_cputiled<24,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
        else if (p->patch_w == 25) { nn_n_cputiled<25,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
        else if (p->patch_w == 26) { nn_n_cputiled<26,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
        else if (p->patch_w == 27) { nn_n_cputiled<27,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
        else if (p->patch_w == 28) { nn_n_cputiled<28,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
        else if (p->patch_w == 29) { nn_n_cputiled<29,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
        else if (p->patch_w == 30) { nn_n_cputiled<30,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
        else if (p->patch_w == 31) { nn_n_cputiled<31,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
        else if (p->patch_w == 32) { nn_n_cputiled<32,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
        else { fprintf(stderr, "Patch size unsupported: %d\n", p->patch_w); exit(1); }
      }
    } 
		else {
      printf("Running nn cputiled, using masks\n");
      if      (p->patch_w == 1) { nn_n_cputiled<1,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 2) { nn_n_cputiled<2,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 3) { nn_n_cputiled<3,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 4) { nn_n_cputiled<4,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 5) { nn_n_cputiled<5,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 6) { nn_n_cputiled<6,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 7) { nn_n_cputiled<7,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 8) { nn_n_cputiled<8,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 9) { nn_n_cputiled<9,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 10) { nn_n_cputiled<10,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 11) { nn_n_cputiled<11,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 12) { nn_n_cputiled<12,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 13) { nn_n_cputiled<13,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 14) { nn_n_cputiled<14,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 15) { nn_n_cputiled<15,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 16) { nn_n_cputiled<16,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 17) { nn_n_cputiled<17,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 18) { nn_n_cputiled<18,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 19) { nn_n_cputiled<19,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 20) { nn_n_cputiled<20,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 21) { nn_n_cputiled<21,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 22) { nn_n_cputiled<22,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 23) { nn_n_cputiled<23,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 24) { nn_n_cputiled<24,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 25) { nn_n_cputiled<25,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 26) { nn_n_cputiled<26,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 27) { nn_n_cputiled<27,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 28) { nn_n_cputiled<28,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 29) { nn_n_cputiled<29,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 30) { nn_n_cputiled<30,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 31) { nn_n_cputiled<31,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else if (p->patch_w == 32) { nn_n_cputiled<32,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles, ann_window, awinsize); }
      else { fprintf(stderr, "Patch size unsupported: %d\n", p->patch_w); exit(1); }
    }
  } 
	else if (algo == ALGO_GPUCPU) {
    if (bmask == NULL && amask == NULL && region_masks == NULL) {
      if      (p->patch_w == 1) { nn_n_gpucpu<1,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 2) { nn_n_gpucpu<2,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 3) { nn_n_gpucpu<3,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 4) { nn_n_gpucpu<4,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 5) { nn_n_gpucpu<5,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 6) { nn_n_gpucpu<6,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 7) { nn_n_gpucpu<7,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 8) { nn_n_gpucpu<8,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 9) { nn_n_gpucpu<9,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 10) { nn_n_gpucpu<10,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 11) { nn_n_gpucpu<11,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 12) { nn_n_gpucpu<12,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 13) { nn_n_gpucpu<13,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 14) { nn_n_gpucpu<14,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 15) { nn_n_gpucpu<15,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 16) { nn_n_gpucpu<16,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 17) { nn_n_gpucpu<17,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 18) { nn_n_gpucpu<18,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 19) { nn_n_gpucpu<19,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 20) { nn_n_gpucpu<20,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 21) { nn_n_gpucpu<21,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 22) { nn_n_gpucpu<22,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 23) { nn_n_gpucpu<23,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 24) { nn_n_gpucpu<24,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 25) { nn_n_gpucpu<25,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 26) { nn_n_gpucpu<26,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 27) { nn_n_gpucpu<27,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 28) { nn_n_gpucpu<28,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 29) { nn_n_gpucpu<29,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 30) { nn_n_gpucpu<30,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 31) { nn_n_gpucpu<31,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 32) { nn_n_gpucpu<32,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else { fprintf(stderr, "Patch size unsupported: %d\n", p->patch_w); exit(1); }
    } 
		else {
      if      (p->patch_w == 1) { nn_n_gpucpu<1,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 2) { nn_n_gpucpu<2,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 3) { nn_n_gpucpu<3,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 4) { nn_n_gpucpu<4,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 5) { nn_n_gpucpu<5,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 6) { nn_n_gpucpu<6,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 7) { nn_n_gpucpu<7,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 8) { nn_n_gpucpu<8,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 9) { nn_n_gpucpu<9,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 10) { nn_n_gpucpu<10,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 11) { nn_n_gpucpu<11,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 12) { nn_n_gpucpu<12,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 13) { nn_n_gpucpu<13,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 14) { nn_n_gpucpu<14,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 15) { nn_n_gpucpu<15,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 16) { nn_n_gpucpu<16,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 17) { nn_n_gpucpu<17,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 18) { nn_n_gpucpu<18,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 19) { nn_n_gpucpu<19,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 20) { nn_n_gpucpu<20,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 21) { nn_n_gpucpu<21,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 22) { nn_n_gpucpu<22,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 23) { nn_n_gpucpu<23,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 24) { nn_n_gpucpu<24,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 25) { nn_n_gpucpu<25,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 26) { nn_n_gpucpu<26,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 27) { nn_n_gpucpu<27,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 28) { nn_n_gpucpu<28,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 29) { nn_n_gpucpu<29,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 30) { nn_n_gpucpu<30,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 31) { nn_n_gpucpu<31,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 32) { nn_n_gpucpu<32,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else { fprintf(stderr, "Patch size unsupported: %d\n", p->patch_w); exit(1); }
    }
  } 
	else if (p->algo == ALGO_FULLRAND) {
    if (bmask == NULL && amask == NULL && region_masks == NULL) {
      if      (p->patch_w == 1) { nn_n_fullrand<1,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 2) { nn_n_fullrand<2,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 3) { nn_n_fullrand<3,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 4) { nn_n_fullrand<4,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 5) { nn_n_fullrand<5,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 6) { nn_n_fullrand<6,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 7) { nn_n_fullrand<7,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 8) { nn_n_fullrand<8,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 9) { nn_n_fullrand<9,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 10) { nn_n_fullrand<10,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 11) { nn_n_fullrand<11,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 12) { nn_n_fullrand<12,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 13) { nn_n_fullrand<13,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 14) { nn_n_fullrand<14,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 15) { nn_n_fullrand<15,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 16) { nn_n_fullrand<16,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 17) { nn_n_fullrand<17,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 18) { nn_n_fullrand<18,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 19) { nn_n_fullrand<19,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 20) { nn_n_fullrand<20,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 21) { nn_n_fullrand<21,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 22) { nn_n_fullrand<22,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 23) { nn_n_fullrand<23,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 24) { nn_n_fullrand<24,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 25) { nn_n_fullrand<25,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 26) { nn_n_fullrand<26,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 27) { nn_n_fullrand<27,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 28) { nn_n_fullrand<28,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 29) { nn_n_fullrand<29,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 30) { nn_n_fullrand<30,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 31) { nn_n_fullrand<31,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 32) { nn_n_fullrand<32,0,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else { fprintf(stderr, "Patch size unsupported: %d\n", p->patch_w); exit(1); }
    } 
		else {
      if      (p->patch_w == 1) { nn_n_fullrand<1,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 2) { nn_n_fullrand<2,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 3) { nn_n_fullrand<3,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 4) { nn_n_fullrand<4,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 5) { nn_n_fullrand<5,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 6) { nn_n_fullrand<6,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 7) { nn_n_fullrand<7,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 8) { nn_n_fullrand<8,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 9) { nn_n_fullrand<9,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 10) { nn_n_fullrand<10,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 11) { nn_n_fullrand<11,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 12) { nn_n_fullrand<12,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 13) { nn_n_fullrand<13,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 14) { nn_n_fullrand<14,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 15) { nn_n_fullrand<15,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 16) { nn_n_fullrand<16,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 17) { nn_n_fullrand<17,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 18) { nn_n_fullrand<18,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 19) { nn_n_fullrand<19,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 20) { nn_n_fullrand<20,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 21) { nn_n_fullrand<21,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 22) { nn_n_fullrand<22,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 23) { nn_n_fullrand<23,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 24) { nn_n_fullrand<24,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 25) { nn_n_fullrand<25,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 26) { nn_n_fullrand<26,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 27) { nn_n_fullrand<27,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 28) { nn_n_fullrand<28,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 29) { nn_n_fullrand<29,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 30) { nn_n_fullrand<30,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 31) { nn_n_fullrand<31,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else if (p->patch_w == 32) { nn_n_fullrand<32,1,0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, region_masks, tiles); }
      else { fprintf(stderr, "Patch size unsupported: %d\n", p->patch_w); exit(1); }
    }
  } 
	else {
    fprintf(stderr, "Unknown algorithm %d\n", algo); exit(1);
  }
}

/* Has defaults for params at and after bnn. */
BITMAP *vote(Params *p, BITMAP *b,
          BITMAP *ann, BITMAP *bnn,
          BITMAP *bmask, BITMAP *bweight,
          double coherence_weight, double complete_weight,
          RegionMasks *amask, BITMAP *aweight, BITMAP *ainit, RegionMasks *region_masks, BITMAP *aconstraint, int mask_self_only) {
  BITMAP *ans = NULL;
  if (p->vote_algo == VOTE_MEAN) {
    if (p->algo == ALGO_CPUTILED) {
      printf("voting mean, fully parallel, %d cores\n", p->cores);
      if (bnn == NULL && bmask == NULL && bweight == NULL && amask == NULL && aweight == NULL && region_masks == NULL) {
        if      (p->patch_w == 1) { ans = vote_n_cputiled<1,int,1,0>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 2) { ans = vote_n_cputiled<2,int,1,0>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 3) { ans = vote_n_cputiled<3,int,1,0>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 4) { ans = vote_n_cputiled<4,int,1,0>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 5) { ans = vote_n_cputiled<5,int,1,0>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 6) { ans = vote_n_cputiled<6,int,1,0>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 7) { ans = vote_n_cputiled<7,int,1,0>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 8) { ans = vote_n_cputiled<8,int,1,0>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 9) { ans = vote_n_cputiled<9,int,1,0>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 10) { ans = vote_n_cputiled<10,int,1,0>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 11) { ans = vote_n_cputiled<11,int,1,0>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 12) { ans = vote_n_cputiled<12,int,1,0>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 13) { ans = vote_n_cputiled<13,int,1,0>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 14) { ans = vote_n_cputiled<14,int,1,0>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 15) { ans = vote_n_cputiled<15,int,1,0>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 16) { ans = vote_n_cputiled<16,int,1,0>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 17) { ans = vote_n_cputiled<17,int,1,0>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 18) { ans = vote_n_cputiled<18,int,1,0>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 19) { ans = vote_n_cputiled<19,int,1,0>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 20) { ans = vote_n_cputiled<20,int,1,0>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 21) { ans = vote_n_cputiled<21,int,1,0>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 22) { ans = vote_n_cputiled<22,int,1,0>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 23) { ans = vote_n_cputiled<23,int,1,0>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 24) { ans = vote_n_cputiled<24,int,1,0>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 25) { ans = vote_n_cputiled<25,int,1,0>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 26) { ans = vote_n_cputiled<26,int,1,0>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 27) { ans = vote_n_cputiled<27,int,1,0>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 28) { ans = vote_n_cputiled<28,int,1,0>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 29) { ans = vote_n_cputiled<29,int,1,0>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 30) { ans = vote_n_cputiled<30,int,1,0>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 31) { ans = vote_n_cputiled<31,int,1,0>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 32) { ans = vote_n_cputiled<32,int,1,0>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else { fprintf(stderr, "Patch size unsupported: %d\n", p->patch_w); exit(1); }
      } else if (bnn == NULL && aweight == NULL && region_masks == NULL) {
        if      (p->patch_w == 1) { ans = vote_n_cputiled<1,int,0,1>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 2) { ans = vote_n_cputiled<2,int,0,1>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 3) { ans = vote_n_cputiled<3,int,0,1>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 4) { ans = vote_n_cputiled<4,int,0,1>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 5) { ans = vote_n_cputiled<5,int,0,1>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 6) { ans = vote_n_cputiled<6,int,0,1>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 7) { ans = vote_n_cputiled<7,int,0,1>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 8) { ans = vote_n_cputiled<8,int,0,1>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 9) { ans = vote_n_cputiled<9,int,0,1>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 10) { ans = vote_n_cputiled<10,int,0,1>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 11) { ans = vote_n_cputiled<11,int,0,1>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 12) { ans = vote_n_cputiled<12,int,0,1>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 13) { ans = vote_n_cputiled<13,int,0,1>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 14) { ans = vote_n_cputiled<14,int,0,1>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 15) { ans = vote_n_cputiled<15,int,0,1>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 16) { ans = vote_n_cputiled<16,int,0,1>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 17) { ans = vote_n_cputiled<17,int,0,1>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 18) { ans = vote_n_cputiled<18,int,0,1>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 19) { ans = vote_n_cputiled<19,int,0,1>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 20) { ans = vote_n_cputiled<20,int,0,1>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 21) { ans = vote_n_cputiled<21,int,0,1>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 22) { ans = vote_n_cputiled<22,int,0,1>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 23) { ans = vote_n_cputiled<23,int,0,1>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 24) { ans = vote_n_cputiled<24,int,0,1>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 25) { ans = vote_n_cputiled<25,int,0,1>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 26) { ans = vote_n_cputiled<26,int,0,1>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 27) { ans = vote_n_cputiled<27,int,0,1>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 28) { ans = vote_n_cputiled<28,int,0,1>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 29) { ans = vote_n_cputiled<29,int,0,1>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 30) { ans = vote_n_cputiled<30,int,0,1>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 31) { ans = vote_n_cputiled<31,int,0,1>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 32) { ans = vote_n_cputiled<32,int,0,1>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else { fprintf(stderr, "Patch size unsupported: %d\n", p->patch_w); exit(1); }
      } else {
        if      (p->patch_w == 1) { ans = vote_n_cputiled<1,double,0,0>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 2) { ans = vote_n_cputiled<2,double,0,0>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 3) { ans = vote_n_cputiled<3,double,0,0>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 4) { ans = vote_n_cputiled<4,double,0,0>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 5) { ans = vote_n_cputiled<5,double,0,0>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 6) { ans = vote_n_cputiled<6,double,0,0>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 7) { ans = vote_n_cputiled<7,double,0,0>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 8) { ans = vote_n_cputiled<8,double,0,0>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 9) { ans = vote_n_cputiled<9,double,0,0>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 10) { ans = vote_n_cputiled<10,double,0,0>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 11) { ans = vote_n_cputiled<11,double,0,0>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 12) { ans = vote_n_cputiled<12,double,0,0>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 13) { ans = vote_n_cputiled<13,double,0,0>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 14) { ans = vote_n_cputiled<14,double,0,0>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 15) { ans = vote_n_cputiled<15,double,0,0>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 16) { ans = vote_n_cputiled<16,double,0,0>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 17) { ans = vote_n_cputiled<17,double,0,0>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 18) { ans = vote_n_cputiled<18,double,0,0>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 19) { ans = vote_n_cputiled<19,double,0,0>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 20) { ans = vote_n_cputiled<20,double,0,0>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 21) { ans = vote_n_cputiled<21,double,0,0>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 22) { ans = vote_n_cputiled<22,double,0,0>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 23) { ans = vote_n_cputiled<23,double,0,0>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 24) { ans = vote_n_cputiled<24,double,0,0>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 25) { ans = vote_n_cputiled<25,double,0,0>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 26) { ans = vote_n_cputiled<26,double,0,0>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 27) { ans = vote_n_cputiled<27,double,0,0>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 28) { ans = vote_n_cputiled<28,double,0,0>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 29) { ans = vote_n_cputiled<29,double,0,0>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 30) { ans = vote_n_cputiled<30,double,0,0>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 31) { ans = vote_n_cputiled<31,double,0,0>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 32) { ans = vote_n_cputiled<32,double,0,0>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else { fprintf(stderr, "Patch size unsupported: %d\n", p->patch_w); exit(1); }
      }
    } else {
      if (p->cores == 1) {
        printf("voting mean, 1 core\n");
        if      (p->patch_w == 1) { ans = vote_n<1>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 2) { ans = vote_n<2>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 3) { ans = vote_n<3>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 4) { ans = vote_n<4>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 5) { ans = vote_n<5>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 6) { ans = vote_n<6>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 7) { ans = vote_n<7>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 8) { ans = vote_n<8>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 9) { ans = vote_n<9>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 10) { ans = vote_n<10>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 11) { ans = vote_n<11>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 12) { ans = vote_n<12>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 13) { ans = vote_n<13>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 14) { ans = vote_n<14>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 15) { ans = vote_n<15>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 16) { ans = vote_n<16>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 17) { ans = vote_n<17>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 18) { ans = vote_n<18>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 19) { ans = vote_n<19>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 20) { ans = vote_n<20>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 21) { ans = vote_n<21>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 22) { ans = vote_n<22>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 23) { ans = vote_n<23>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 24) { ans = vote_n<24>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 25) { ans = vote_n<25>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 26) { ans = vote_n<26>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 27) { ans = vote_n<27>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 28) { ans = vote_n<28>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 29) { ans = vote_n<29>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 30) { ans = vote_n<30>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 31) { ans = vote_n<31>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 32) { ans = vote_n<32>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else { fprintf(stderr, "Patch size unsupported: %d\n", p->patch_w); exit(1); }
      } else {
        printf("voting mean, 1 or 2 parallel, %d cores\n", p->cores);
        if      (p->patch_w == 1) { ans = vote_n_openmp<1>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 2) { ans = vote_n_openmp<2>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 3) { ans = vote_n_openmp<3>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 4) { ans = vote_n_openmp<4>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 5) { ans = vote_n_openmp<5>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 6) { ans = vote_n_openmp<6>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 7) { ans = vote_n_openmp<7>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 8) { ans = vote_n_openmp<8>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 9) { ans = vote_n_openmp<9>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 10) { ans = vote_n_openmp<10>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 11) { ans = vote_n_openmp<11>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 12) { ans = vote_n_openmp<12>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 13) { ans = vote_n_openmp<13>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 14) { ans = vote_n_openmp<14>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 15) { ans = vote_n_openmp<15>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 16) { ans = vote_n_openmp<16>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 17) { ans = vote_n_openmp<17>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 18) { ans = vote_n_openmp<18>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 19) { ans = vote_n_openmp<19>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 20) { ans = vote_n_openmp<20>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 21) { ans = vote_n_openmp<21>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 22) { ans = vote_n_openmp<22>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 23) { ans = vote_n_openmp<23>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 24) { ans = vote_n_openmp<24>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 25) { ans = vote_n_openmp<25>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 26) { ans = vote_n_openmp<26>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 27) { ans = vote_n_openmp<27>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 28) { ans = vote_n_openmp<28>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 29) { ans = vote_n_openmp<29>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 30) { ans = vote_n_openmp<30>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 31) { ans = vote_n_openmp<31>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else if (p->patch_w == 32) { ans = vote_n_openmp<32>(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
        else { fprintf(stderr, "Patch size unsupported: %d\n", p->patch_w); exit(1); }
      }
    }
  } else {
    fprintf(stderr, "Unknown voting algorithm: %d\n", p->vote_algo); exit(1);
  }

  return ans;
}

int patch_dist(Params *p, BITMAP *a, int ax, int ay,
               BITMAP *b, int bx, int by, int maxval, RegionMasks *region_masks) {
  if ((unsigned) ax >= (unsigned) (a->w - p->patch_w + 1) ||
      (unsigned) ay >= (unsigned) (a->h - p->patch_w + 1)) { fprintf(stderr, "a coord out of bounds: %d, %d (%dx%d)\n", ax, ay, a->w, a->h); exit(1); }
  if ((unsigned) bx >= (unsigned) (b->w - p->patch_w + 1) ||
      (unsigned) by >= (unsigned) (b->h - p->patch_w + 1)) { fprintf(stderr, "b coord out of bounds: %d, %d (%dx%d)\n", bx, by, b->w, b->h); exit(1); }
  if (region_masks && ((int *) region_masks->bmp->line[ay])[ax] != ((int *) region_masks->bmp->line[by])[bx]) { return INT_MAX; }
  
  if (is_window(p)) {
    int ans = 0;
    for (int dy = 0; dy < p->patch_w; dy++) {
      int *row1 = ((int *) a->line[ay+dy])+ax;
      int *row2 = ((int *) b->line[by+dy])+bx;
      for (int dx = 0; dx < p->patch_w; dx++) {
        //printf("patch_w: %d\n", p->patch_w); fflush(stdout);
        int c1 = row1[dx];
        int c2 = row2[dx];
        int dr = (c1&255)-(c2&255);
        int dg = ((c1>>8)&255)-((c2>>8)&255);
        int db = (c1>>16)-(c2>>16);
        dr *= p->weight_r;
        dg *= p->weight_g;
        db *= p->weight_b;
        ans += dr*dr+dg*dg+db*db;
        //printf("end pixel\n"); fflush(stdout);
        if (ans > maxval) { return ans; }
      }
    }
    return ans;
  } 
	else {
    int ans = 0;
    for (int dy = 0; dy < p->patch_w; dy++) {
      int *row1 = ((int *) a->line[ay+dy])+ax;
      int *row2 = ((int *) b->line[by+dy])+bx;
      for (int dx = 0; dx < p->patch_w; dx++) {
        //printf("patch_w: %d\n", p->patch_w); fflush(stdout);
        int c1 = row1[dx];
        int c2 = row2[dx];
        int dr = (c1&255)-(c2&255);
        int dg = ((c1>>8)&255)-((c2>>8)&255);
        int db = (c1>>16)-(c2>>16);
        ans += dr*dr+dg*dg+db*db;
        //printf("end pixel\n"); fflush(stdout);
        if (ans > maxval) { return ans; }
      }
    }
    return ans;
  }
}

void check_colors(BITMAP *bmp) {
  for (int y = 0; y < bmp->h; y++) {
    for (int x = 0; x < bmp->w; x++) {
      int c = _getpixel32(bmp, x, y);
      if (c >> 24) { fprintf(stderr, "alpha component found at (%d, %d)\n", x, y); exit(1); }
    }
  }
}

int clip_nn(Params *p, BITMAP *ann, BITMAP *b) {
  int ans = 0;
  for (int y = 0; y < ann->h-p->patch_w+1; y++) {
    for (int x = 0; x < ann->w-p->patch_w+1; x++) {
      int xp, yp;
      getnn(ann, x, y, xp, yp);
      if ((unsigned) xp >= (unsigned) (b->w-p->patch_w+1) ||
          (unsigned) yp >= (unsigned) (b->h-p->patch_w+1)) {
        ans++;
        if (xp < 0) { xp = 0; }
        if (yp < 0) { yp = 0; }
        if (xp > b->w-p->patch_w) { xp = b->w-p->patch_w; }
        if (yp > b->h-p->patch_w) { yp = b->h-p->patch_w; }
      }
      _putpixel32(ann, x, y, XY_TO_INT(xp, yp));
    }
  }
  return ans;
}

void check_nn(Params *p, BITMAP *ann, BITMAP *b, BITMAP *bmask, RegionMasks *amask, RegionMasks *region_masks) {
  Box box = get_abox(p, ann, amask);
  for (int y = box.ymin; y < box.ymax; y++) {
    for (int x = box.xmin; x < box.xmax; x++) {
      if (amask && ((int *) amask->bmp->line[y])[x]) { continue; }
      int xp, yp;
      getnn(ann, x, y, xp, yp);
      if (bmask && ((int *) bmask->line[yp])[xp]) { fprintf(stderr, "NN mapping %d,%d => %d,%d maps is b masked\n", x, y, xp, yp); exit(1); }
      if ((unsigned) xp >= (unsigned) (b->w-p->patch_w+1) ||
          (unsigned) yp >= (unsigned) (b->h-p->patch_w+1)) {
        fprintf(stderr, "Bad NN mapping: (%d, %d) => (%d, %d) ann: %dx%d, b: %dx%d\n", x, y, xp, yp, ann->w, ann->h, b->w, b->h); exit(1);
      }
      if (region_masks && ((int *) region_masks->bmp->line[y])[x] != ((int *) region_masks->bmp->line[yp])[xp]) {
        fprintf(stderr, "NN mapping %d,%d => %d,%d maps is region masked\n", x, y, xp, yp); exit(1);
      }
    }
  }
}

void check_dists(Params *p, BITMAP *a, BITMAP *b, BITMAP *ann, BITMAP *annd, int max_dist, RegionMasks *amask) {
  check_colors(a);
  check_colors(b);
  Box box = get_abox(p, a, amask);
  int max_err = 0, min_val = INT_MAX, min_valb = INT_MAX;
  for (int y = box.ymin; y < box.ymax; y++) {
    for (int x = box.xmin; x < box.xmax; x++) {
      if (amask && ((int *) amask->bmp->line[y])[x]) { continue; }
      int d1 = ((int *) annd->line[y])[x];
      int xp, yp;
      getnn(ann, x, y, xp, yp);
      int d2 = patch_dist(p, a, x, y, b, xp, yp);
      if (abs(d1 - d2)>max_err) {
        max_err = abs(d1 - d2);
        if (d1 < min_val) { min_val = d1; min_valb = d2; }
      }
      if (abs(d1 - d2)>max_dist) {
        printf("Distances not equal at (%d, %d) => (%d, %d): %d should be %d\n", x, y, xp, yp, d1, d2);
        for (int dy = -1; dy <= 1; dy++) {
          for (int dx = -1; dx <= 1; dx++) {
            if ((unsigned) (xp+dx) < (unsigned) (b->w-p->patch_w+1) &&
                (unsigned) (yp+dy) < (unsigned) (b->h-p->patch_w+1)) {
              printf("%08d ", patch_dist(p, a, x, y, b, xp+dx, yp+dy));    
            }
            printf("\n");
          }
        }
        exit(1);
      }
    }
  }
  if (max_err > 0) 
		printf("check_dists: max_err=%d, min_val=%d should be %d\n", max_err, min_val, min_valb);
}

BITMAP *copy_image(BITMAP *a) {
  if (!a) { fprintf(stderr, "copy_image: Argument is NULL\n"); exit(1); }
  BITMAP *ans = create_bitmap(a->w, a->h);
  blit(a, ans, 0, 0, 0, 0, a->w, a->h);
  return ans;
}

void getnn(BITMAP *ann, int x, int y, int &xp, int &yp) {
  int dest = ((int *) ann->line[y])[x];
  xp = INT_TO_X(dest);
  yp = INT_TO_Y(dest);
}

template<int PATCH_W, int IS_WINDOW, int HAS_MASKS>
void minnn_n(Params *p, BITMAP *a, BITMAP *b, BITMAP *ann, BITMAP *annd, BITMAP *ann_prev, BITMAP *bmask, int level, int em_iter, RecomposeParams *rp, RegionMasks *region_masks, RegionMasks *amask, int ntiles) {
  if (ntiles < 0) { ntiles = p->cores; }
  printf("minnn: %d %d %d %d, tiles=%d\n", ann->w, ann->h, ann_prev->w, ann_prev->h, ntiles);
  if (!rp) { fprintf(stderr, "minnn: rp is NULL\n"); exit(1); }
  Box box = get_abox(p, a, amask);

  #pragma omp parallel for schedule(static,4) num_threads(ntiles)
  for (int y = box.ymin; y < box.ymax; y++) {
    int *amask_row = amask ? (int *) amask->bmp->line[y]: NULL;
    int *annd_row = (int *) annd->line[y];
    for (int x = box.xmin; x < box.xmax; x++) {
      if (HAS_MASKS && amask && amask_row[x]) { continue; }
      int dcurrent = annd_row[x];
      int xp, yp;
      getnn(ann_prev, x, y, xp, yp);
      if ((unsigned) xp >= (unsigned) (b->w-p->patch_w+1) ||
          (unsigned) yp >= (unsigned) (b->h-p->patch_w+1)) { continue; }
      if (HAS_MASKS && bmask && ((int *) bmask->line[yp])[xp]) { continue; }
      
			int dprev = patch_dist_ab<PATCH_W, IS_WINDOW, HAS_MASKS>(p, a, x, y, b, xp, yp, dcurrent, region_masks);
      
      if (dprev < dcurrent) {
        _putpixel32(ann, x, y, XY_TO_INT(xp, yp));
        _putpixel32(annd, x, y, dprev);
      }
    }
  }
  Params pcopy(*p);
  pcopy.nn_iters = rp->minnn_optp_nn_iters;
  pcopy.rs_max = rp->minnn_optp_rs_max;
  
  nn(&pcopy, a, b, ann, annd, amask, bmask, level, em_iter, rp, 0, 0, 1, region_masks, ntiles);
}

void minnn(Params *p, BITMAP *a, BITMAP *b, BITMAP *ann, BITMAP *annd, BITMAP *ann_prev, BITMAP *bmask, int level, int em_iter, RecomposeParams *rp, RegionMasks *region_masks, RegionMasks *amask, int ntiles) {
  if (is_window(p)) {
    if      (p->patch_w == 1 ) { return minnn_n<1 , 1, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
    else if (p->patch_w == 2 ) { return minnn_n<2 , 1, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
    else if (p->patch_w == 3 ) { return minnn_n<3 , 1, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
    else if (p->patch_w == 4 ) { return minnn_n<4 , 1, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
    else if (p->patch_w == 5 ) { return minnn_n<5 , 1, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
    else if (p->patch_w == 6 ) { return minnn_n<6 , 1, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
    else if (p->patch_w == 7 ) { return minnn_n<7 , 1, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
    else if (p->patch_w == 8 ) { return minnn_n<8 , 1, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
    else if (p->patch_w == 9 ) { return minnn_n<9 , 1, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
    else if (p->patch_w == 10) { return minnn_n<10, 1, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
    else if (p->patch_w == 11) { return minnn_n<11, 1, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
    else if (p->patch_w == 12) { return minnn_n<12, 1, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
    else if (p->patch_w == 13) { return minnn_n<13, 1, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
    else if (p->patch_w == 14) { return minnn_n<14, 1, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
    else if (p->patch_w == 15) { return minnn_n<15, 1, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
    else if (p->patch_w == 16) { return minnn_n<16, 1, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 17) { return minnn_n<17, 1, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 18) { return minnn_n<18, 1, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 19) { return minnn_n<19, 1, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 20) { return minnn_n<20, 1, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 21) { return minnn_n<21, 1, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 22) { return minnn_n<22, 1, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 23) { return minnn_n<23, 1, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 24) { return minnn_n<24, 1, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 25) { return minnn_n<25, 1, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 26) { return minnn_n<26, 1, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 27) { return minnn_n<27, 1, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 28) { return minnn_n<28, 1, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 29) { return minnn_n<29, 1, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 30) { return minnn_n<30, 1, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 31) { return minnn_n<31, 1, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 32) { return minnn_n<32, 1, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else { fprintf(stderr, "Patch size unsupported: %d\n", p->patch_w); exit(1); }
  } 
	else if (bmask || region_masks || amask) {
    if      (p->patch_w == 1 ) { return minnn_n<1 , 0, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
    else if (p->patch_w == 2 ) { return minnn_n<2 , 0, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
    else if (p->patch_w == 3 ) { return minnn_n<3 , 0, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
    else if (p->patch_w == 4 ) { return minnn_n<4 , 0, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
    else if (p->patch_w == 5 ) { return minnn_n<5 , 0, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
    else if (p->patch_w == 6 ) { return minnn_n<6 , 0, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
    else if (p->patch_w == 7 ) { return minnn_n<7 , 0, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
    else if (p->patch_w == 8 ) { return minnn_n<8 , 0, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
    else if (p->patch_w == 9 ) { return minnn_n<9 , 0, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
    else if (p->patch_w == 10) { return minnn_n<10, 0, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
    else if (p->patch_w == 11) { return minnn_n<11, 0, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
    else if (p->patch_w == 12) { return minnn_n<12, 0, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
    else if (p->patch_w == 13) { return minnn_n<13, 0, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
    else if (p->patch_w == 14) { return minnn_n<14, 0, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
    else if (p->patch_w == 15) { return minnn_n<15, 0, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
    else if (p->patch_w == 16) { return minnn_n<16, 0, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 17) { return minnn_n<17, 0, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 18) { return minnn_n<18, 0, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 19) { return minnn_n<19, 0, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 20) { return minnn_n<20, 0, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 21) { return minnn_n<21, 0, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 22) { return minnn_n<22, 0, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 23) { return minnn_n<23, 0, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 24) { return minnn_n<24, 0, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 25) { return minnn_n<25, 0, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 26) { return minnn_n<26, 0, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 27) { return minnn_n<27, 0, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 28) { return minnn_n<28, 0, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 29) { return minnn_n<29, 0, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 30) { return minnn_n<30, 0, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 31) { return minnn_n<31, 0, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 32) { return minnn_n<32, 0, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else { fprintf(stderr, "Patch size unsupported: %d\n", p->patch_w); exit(1); }
  } 
	else {
    if      (p->patch_w == 1 ) { return minnn_n<1 , 0, 0>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
    else if (p->patch_w == 2 ) { return minnn_n<2 , 0, 0>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
    else if (p->patch_w == 3 ) { return minnn_n<3 , 0, 0>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
    else if (p->patch_w == 4 ) { return minnn_n<4 , 0, 0>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
    else if (p->patch_w == 5 ) { return minnn_n<5 , 0, 0>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
    else if (p->patch_w == 6 ) { return minnn_n<6 , 0, 0>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
    else if (p->patch_w == 7 ) { return minnn_n<7 , 0, 0>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
    else if (p->patch_w == 8 ) { return minnn_n<8 , 0, 0>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
    else if (p->patch_w == 9 ) { return minnn_n<9 , 0, 0>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
    else if (p->patch_w == 10) { return minnn_n<10, 0, 0>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
    else if (p->patch_w == 11) { return minnn_n<11, 0, 0>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
    else if (p->patch_w == 12) { return minnn_n<12, 0, 0>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
    else if (p->patch_w == 13) { return minnn_n<13, 0, 0>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
    else if (p->patch_w == 14) { return minnn_n<14, 0, 0>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
    else if (p->patch_w == 15) { return minnn_n<15, 0, 0>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
    else if (p->patch_w == 16) { return minnn_n<16, 0, 0>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 17) { return minnn_n<17, 0, 0>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 18) { return minnn_n<18, 0, 0>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 19) { return minnn_n<19, 0, 0>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 20) { return minnn_n<20, 0, 0>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 21) { return minnn_n<21, 0, 0>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 22) { return minnn_n<22, 0, 0>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 23) { return minnn_n<23, 0, 0>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 24) { return minnn_n<24, 0, 0>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 25) { return minnn_n<25, 0, 0>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 26) { return minnn_n<26, 0, 0>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 27) { return minnn_n<27, 0, 0>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 28) { return minnn_n<28, 0, 0>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 29) { return minnn_n<29, 0, 0>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 30) { return minnn_n<30, 0, 0>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 31) { return minnn_n<31, 0, 0>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 32) { return minnn_n<32, 0, 0>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else { fprintf(stderr, "Patch size unsupported: %d\n", p->patch_w); exit(1); }
  }
}

/* unused code */

vector<unsigned> *VECBITMAP_ARB::get(int x, int y) {
  return &data[y*w+x];
}

VECBITMAP_ARB *create_vecbitmap_arb(int w, int h) {
  VECBITMAP_ARB *ans = new VECBITMAP_ARB();
  ans->w = w;
  ans->h = h;
  ans->data = new vector<unsigned>[w*h];
  return ans;
}

void destroy_vecbitmap_arb(VECBITMAP_ARB *bmp) {
  delete[] bmp->data;
  delete bmp;
}

void sort_vecbitmap_arb(VECBITMAP_ARB *bmp) {
  for (int y = 0; y < bmp->h; y++) {
    for (int x = 0; x < bmp->w; x++) {
      vector<unsigned> *p = bmp->get(x, y);
      sort(p->begin(), p->end());
    }
  }
}

inline void write_unsigned(FILE *f, unsigned x) {
  fputc(x&255, f);
  fputc((x>>8)&255, f);
  fputc((x>>16)&255, f);
  fputc((x>>24), f);
}

void save_vecbitmap_arb(const char *filename, VECBITMAP_ARB *bmp) {
  FILE *f = fopen(filename, "wb");
  int nelem = 2;
  for (int y = 0; y < bmp->h; y++) {
    for (int x = 0; x < bmp->w; x++) {
      vector<unsigned> *p = bmp->get(x, y);
      nelem++;
      nelem += p->size();
    }
  }
  fprintf(stderr, "number of elements: %d\n", nelem);
  write_unsigned(f, nelem);
  write_unsigned(f, bmp->w);
  write_unsigned(f, bmp->h);
  for (int y = 0; y < bmp->h; y++) {
    for (int x = 0; x < bmp->w; x++) {
      vector<unsigned> *p = bmp->get(x, y);
      write_unsigned(f, p->size());
      for (int i = 0; i < (int) p->size(); i++) {
        write_unsigned(f, (*p)[i]);
      }
    }
  }
  fclose(f);
}
