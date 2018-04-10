
#ifndef _nnvec_h
#define _nnvec_h

#include "allegro_emu.h"
#include <float.h>
#include "nn.h"

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))
/* -------------------------------------------------------------------------
   Randomized NN algorithm, on vector input (patches, SIFT descriptors, etc)
   -------------------------------------------------------------------------
   
   Descriptors are vectors of type T and dimension n.
   Note that ACCUM must be a signed type, and must be large enough to hold the distance between any two descriptors. */

// data is represented row by row (different from matlab)
// w * h * n: (x, y, z) -> x + y * w + z * h * w
template<class T>
class VECBITMAP { 
public:
  T *data;
  int w, h, n;
  T *get(int x, int y) { return &data[(y*w+x)*n]; }   /* Get patch (x, y). */
  T *line_n1(int y) { return &data[y*w]; }            /* Get line y assuming n=1. */
  VECBITMAP() { }
  VECBITMAP(int w_, int h_, int n_) { w = w_; h = h_; n = n_; data = new T[w*h*n]; }
  ~VECBITMAP() { delete[] data; }
};

template<class T>
void clear_to_color(VECBITMAP<T> *bmp, T c) {
  int n = bmp->w*bmp->h*bmp->n;
  for (int i = 0; i < n; i++) {
    bmp->data[i] = c;
  }
}

template<class T>
T get_maxval() { fprintf(stderr, "get_maxval of unsupported template type\n"); exit(1); }


template<> int get_maxval();
template<> float get_maxval();
template<> double get_maxval();
template<> long long get_maxval();

#include "vecpatch.h"

template<class T>
inline BITMAP *wrap_vecbitmap(VECBITMAP<T> *a) {
  return new BITMAP(a->w, a->h); // no memory allocated for per pixel data
}

// nothing special here, just re-use init_nn to assign a random NN field
template<class T>
BITMAP *vec_init_nn(Params *p, VECBITMAP<T> *a, VECBITMAP<T> *b, BITMAP *bmask=NULL, RegionMasks *region_masks=NULL, RegionMasks *amask=NULL) {
  BITMAP *awrap = wrap_vecbitmap(a);
  BITMAP *bwrap = wrap_vecbitmap(b);
  BITMAP *ans = init_nn(p, awrap, bwrap, bmask, region_masks, amask, 0);
  delete awrap;
  delete bwrap;
  return ans;
}

template<class T>
Box get_abox_vec(Params *p, VECBITMAP<T> *a, RegionMasks *amask, int trim_patch=1) {
  BITMAP *aw = wrap_vecbitmap(a);
  Box ans = get_abox(p, aw, amask, trim_patch);
  delete aw;
  return ans;
}

template<class T, class ACCUM, int IS_MASK, int IS_WINDOW>
VECBITMAP<ACCUM> *vec_init_dist_n(Params *p, VECBITMAP<T> *a, VECBITMAP<T> *b, BITMAP *ann, BITMAP *bmask=NULL, RegionMasks *region_masks=NULL, RegionMasks *amask=NULL) {
  VECBITMAP<ACCUM> *ans = new VECBITMAP<ACCUM>(a->w, a->h, 1); // the third dimension only has one element
  ACCUM maxval = get_maxval<ACCUM>();
  for (int y = 0; y < ans->h; y++) {
    ACCUM *row = ans->line_n1(y);
    for (int x = 0; x < ans->w; x++) {
      row[x] = maxval;
    }
  }
  if (region_masks) {
    if (region_masks->bmp->w != a->w || region_masks->bmp->h != a->h) { fprintf(stderr, "region_masks (%dx%d) size != a (%dx%d) size\n", region_masks->bmp->w, region_masks->bmp->h, a->w, a->h); exit(1); }
    if (region_masks->bmp->w != b->w || region_masks->bmp->h != b->h) { fprintf(stderr, "region_masks (%dx%d) size != b (%dx%d) size\n", region_masks->bmp->w, region_masks->bmp->h, b->w, b->h); exit(1); }
  }

  Box box = get_abox_vec(p, a, amask, 0);
#pragma omp parallel for schedule(static, 4)
  for (int y = box.ymin; y < box.ymax; y++) {
    ACCUM *row = (ACCUM *) ans->line_n1(y);
    int *arow = amask ? (int *) amask->bmp->line[y]: NULL;
    for (int x = box.xmin; x < box.xmax; x++) {
      if (IS_MASK && amask && arow[x]) { continue; }
      int xp, yp;
      getnn(ann, x, y, xp, yp);

      if (IS_MASK && region_masks && ((int *) region_masks->bmp->line[y])[x] != ((int *) region_masks->bmp->line[yp])[xp]) {
        row[x] = maxval; continue;
      }

// <!-- XC, change here to handle a real patch!, refer to init_nn_dist 
      T *apatch = a->get(x, y);

      if (IS_MASK && bmask && ((int *) bmask->line[yp])[xp]) { row[x] = maxval; continue; }
      T *bpatch = b->get(xp, yp);

      row[x] = vec_fast_patch_nobranch<T, ACCUM, IS_WINDOW>(apatch, bpatch, p);
// XC --> 
      //if (x == 1 && y == 1) { printf("1, 1 => %d, %d (%d)\n", xp, yp, row[x]); }
    }
  }
  return ans;
}

template<class T, class ACCUM>
VECBITMAP<ACCUM> *vec_init_dist(Params *p, VECBITMAP<T> *a, VECBITMAP<T> *b, BITMAP *ann, BITMAP *bmask=NULL, RegionMasks *region_masks=NULL, RegionMasks *amask=NULL) {
	VECBITMAP<ACCUM> *ans = NULL;
	if (is_window(p)) 
		ans = vec_init_dist_n<T, ACCUM, 1, 1>(p, a, b, ann, bmask, region_masks, amask);
	else if (amask || bmask || region_masks) 
    ans = vec_init_dist_n<T, ACCUM, 1, 0>(p, a, b, ann, bmask, region_masks, amask);
	else 
    ans = vec_init_dist_n<T, ACCUM, 0, 0>(p, a, b, ann, bmask, region_masks, amask);
	return ans;
}

template<class T>
int window_constraint_wrap(Params *p, VECBITMAP<T> *a, VECBITMAP<T> *b, int ax, int ay, int bx, int by) {
  BITMAP aw, bw;
  aw.w = a->w; aw.h = a->h;
  bw.w = b->w; bw.h = b->h;
  return window_constraint(p, &aw, &bw, ax, ay, bx, by);
}

// <!-- XC, this is the part to change!!!
// XC -->
template<class T, class ACCUM, int IS_MASK, int IS_WINDOW>
void vec_nn_n(Params *p, VECBITMAP<T> *a, VECBITMAP<T> *b,
            BITMAP *ann, VECBITMAP<ACCUM> *annd,
            RegionMasks *amask=NULL, BITMAP *bmask=NULL,
            int level=0, int em_iter=0, RecomposeParams *rp=NULL, int offset_iter=0, int update_type=0, int cache_b=0,
            RegionMasks *region_masks=NULL, int tiles=-1) {
  if (tiles < 0) { tiles = p->cores; }
  printf("in vec_nn_n, masks are: %p %p %p, tiles=%d, rs_max=%d\n", amask, bmask, region_masks, tiles, p->rs_max);
  Box box = get_abox_vec(p, a, amask, 0);
  int nn_iter = 0;
  for (; nn_iter < p->nn_iters; nn_iter++) {
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

			// from top left
      int xstart = xmin, xfinal = xmax, xchange=1;
      int ystart = ymin, yfinal = ymax, ychange=1;
      if ((nn_iter + offset_iter) % 2 == 1) { 
        xstart = xmax-1; xfinal = xmin-1; xchange=-1;
        ystart = ymax-1; yfinal = ymin-1; ychange=-1;
      }
      int dx = -xchange, dy = -ychange;

      int bew = b->w, beh = b->h;
      int max_mag = MAX(b->w, b->h);
      int rs_ipart = int(p->rs_iters);
      double rs_fpart = p->rs_iters - rs_ipart;
      int rs_max = p->rs_max;
      if (rs_max > max_mag) { rs_max = max_mag; }

      for (int y = ystart; y != yfinal; y += ychange) {
        ACCUM *annd_row = annd->line_n1(y);
        int *amask_row = IS_MASK ? (amask ? (int *) amask->bmp->line[y]: NULL): NULL;
        for (int x = xstart; x != xfinal; x += xchange) {
          if (IS_MASK && amask && amask_row[x]) { continue; }

          T *apatch = a->get(x, y);
          
          int src_mask = IS_MASK ? (region_masks ? ((int *) region_masks->bmp->line[y])[x]: 0): 0;

          int xbest, ybest;
          getnn(ann, x, y, xbest, ybest);
          ACCUM err = annd_row[x];
          if (!err) { continue; }

          /* Propagate */
          if (p->do_propagate) {
            /* Propagate x */
            if ((unsigned) (x+dx) < (unsigned) (ann->w)) {
              int xpp, ypp;
              getnn(ann, x+dx, y, xpp, ypp);
              xpp -= dx;

              if (!IS_WINDOW || window_constraint_wrap(p, a, b, x, y, xpp, ypp)) {
                vec_attempt_n<T, ACCUM, IS_MASK, IS_WINDOW>(err, xbest, ybest, apatch, b, xpp, ypp, bmask, region_masks, src_mask, p);
              }
            }

            /* Propagate y */
            if ((unsigned) (y+dy) < (unsigned) (ann->h)) {
              int xpp, ypp;
              getnn(ann, x, y+dy, xpp, ypp);
              ypp -= dy;

              if (!IS_WINDOW || window_constraint_wrap(p, a, b, x, y, xpp, ypp)) {
                vec_attempt_n<T, ACCUM, IS_MASK, IS_WINDOW>(err, xbest, ybest, apatch, b, xpp, ypp, bmask, region_masks, src_mask, p);
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
              int xmin = MAX(xbest-mag,0), xmax = MIN(xbest+mag+1,bew);
              int ymin = MAX(ybest-mag,0), ymax = MIN(ybest+mag+1,beh);
              
              seed = RANDI(seed);
              int xpp = xmin+seed%(xmax-xmin);
              seed = RANDI(seed);
              int ypp = ymin+seed%(ymax-ymin);
              if (!IS_WINDOW || window_constraint_wrap(p, a, b, x, y, xpp, ypp)) {
                vec_attempt_n<T, ACCUM, IS_MASK, IS_WINDOW>(err, xbest, ybest, apatch, b, xpp, ypp, bmask, region_masks, src_mask, p);
              }
            }
          }
          
          ((int *) ann->line[y])[x] = XY_TO_INT(xbest, ybest);
          annd_row[x] = err;
        } // x
      } // y
    } // parallel
  } // nn_iter
  printf("done vec_nn_n, %d iters, rs_max=%d\n", nn_iter, p->rs_max);
}

// <!-- XC, might need to consider the last two parameter as nn(...)
/* 
void nn(Params *p, BITMAP *a, BITMAP *b,
        BITMAP *ann, BITMAP *annd,
        RegionMasks *amask=NULL, BITMAP *bmask=NULL,
        int level=0, int em_iter=0, RecomposeParams *rp=NULL, int offset_iter=0, int update_type=0, int cache_b=0,
        RegionMasks *region_masks=NULL, int tiles=-1, BITMAP *ann_window=NULL, BITMAP *awinsize=NULL);
*/
// XC -->

template<class T, class ACCUM>
void vec_nn(Params *p, VECBITMAP<T> *a, VECBITMAP<T> *b,
        BITMAP *ann, VECBITMAP<ACCUM> *annd,
        RegionMasks *amask=NULL, BITMAP *bmask=NULL,
        int level=0, int em_iter=0, RecomposeParams *rp=NULL, int offset_iter=0, int update_type=0, int cache_b=0,
        RegionMasks *region_masks=NULL, int tiles=-1) 
{
  if (p->algo == ALGO_CPU || p->algo == ALGO_CPUTILED) {
    if (is_window(p)) {
      printf("Running vec_nn (cputiled), using windowed and masked\n");
      vec_nn_n<T, ACCUM, 1, 1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
    } else if (bmask == NULL && amask == NULL && region_masks == NULL) {
      printf("Running vec_nn (cputiled), using unmasked\n");
      vec_nn_n<T, ACCUM, 0, 0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
    } else {
      printf("Running vec_nn (cputiled), using masked\n");
      vec_nn_n<T, ACCUM, 1, 0>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
    }
  } else {
    fprintf(stderr, "vec_nn: algorithm %d unsupported\n", p->algo); exit(1);
  }
}

template<class T, class ACCUM, int IS_WINDOW, int HAS_MASKS>
void vec_minnn_n(Params *p, VECBITMAP<T> *a, VECBITMAP<T> *b, BITMAP *ann, VECBITMAP<ACCUM> *annd, BITMAP *ann_prev, BITMAP *bmask, int level, int em_iter, RecomposeParams *rp, RegionMasks *region_masks, RegionMasks *amask, int ntiles) {
  if (ntiles < 0) { ntiles = p->cores; }
  printf("vec_minnn: %d %d %d %d, tiles=%d\n", ann->w, ann->h, ann_prev->w, ann_prev->h, ntiles);
  if (!rp) { fprintf(stderr, "vec_minnn_n: rp is NULL\n"); exit(1); }
//  double start_t = accurate_timer();
  Box box = get_abox_vec(p, a, amask, 0);

  #pragma omp parallel for schedule(static,4) num_threads(ntiles)
  for (int y = box.ymin; y < box.ymax; y++) {
    int *amask_row = amask ? (int *) amask->bmp->line[y]: NULL;
    ACCUM *annd_row = (ACCUM *) annd->line_n1(y);
    for (int x = box.xmin; x < box.xmax; x++) {
      if (HAS_MASKS && amask && amask_row[x]) { continue; }

      ACCUM dcurrent = annd_row[x];
      int xp, yp;
      getnn(ann_prev, x, y, xp, yp);
      if ((unsigned) xp >= (unsigned) (b->w) ||
          (unsigned) yp >= (unsigned) (b->h)) { continue; }
      if (HAS_MASKS && bmask && ((int *) bmask->line[yp])[xp]) { continue; }

// <!-- XC, some newer patch distance for discriptor mode has to be used here (add code to vecpatch.h)
      //int dprev = patch_dist(p, a, x, y, b, xp, yp, dcurrent, region_masks);
      ACCUM dprev = vec_patch_dist_ab<T, ACCUM, IS_WINDOW, HAS_MASKS>(p, a, x, y, b, xp, yp, dcurrent, region_masks);
// XC -->

      if (dprev < dcurrent) {
        _putpixel32(ann, x, y, XY_TO_INT(xp, yp));
        annd_row[x] = dprev;
      }
    }
  }
//  nn_time += accurate_timer() - start_t;
  Params pcopy(*p);
  pcopy.nn_iters = rp->minnn_optp_nn_iters;
  pcopy.rs_max = rp->minnn_optp_rs_max;
  
  vec_nn<T, ACCUM>(&pcopy, a, b, ann, annd, amask, bmask, level, em_iter, rp, 0, 0, 1, region_masks, ntiles);
}

template<class T, class ACCUM>
void vec_minnn(Params *p, VECBITMAP<T> *a, VECBITMAP<T> *b, BITMAP *ann, VECBITMAP<ACCUM> *annd, BITMAP *ann_prev, BITMAP *bmask, int level, int em_iter, RecomposeParams *rp, RegionMasks *region_masks, RegionMasks *amask, int ntiles) {
  if (is_window(p))
    return vec_minnn_n<T, ACCUM, 1, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles);
  else if (bmask || region_masks || amask) 
    return vec_minnn_n<T, ACCUM, 0, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles);
  else 
    return vec_minnn_n<T, ACCUM, 0, 0>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles);
}


// --------------------------------
// XC version of discriptor mode 
// --------------------------------

template<class T>
BITMAP *XCvec_init_nn(Params *p, VECBITMAP<T> *a, VECBITMAP<T> *b, BITMAP *bmask=NULL, RegionMasks *region_masks=NULL, RegionMasks *amask=NULL) {
  BITMAP *awrap = wrap_vecbitmap(a);
  BITMAP *bwrap = wrap_vecbitmap(b);
  BITMAP *ans = init_nn(p, awrap, bwrap, bmask, region_masks, amask, 1); // trim patch!!!
  delete awrap;
  delete bwrap;
  return ans;
}


template<class T, class ACCUM, int IS_MASK, int IS_WINDOW, int PATCH_W>
VECBITMAP<ACCUM> *XCvec_init_dist_n(Params *p, VECBITMAP<T> *a, VECBITMAP<T> *b, BITMAP *ann, BITMAP *bmask=NULL, RegionMasks *region_masks=NULL, RegionMasks *amask=NULL) \
{
	VECBITMAP<ACCUM> *ans = new VECBITMAP<ACCUM>(a->w, a->h, 1); // the third dimension only has one element
  // set all distances to a large number
	ACCUM maxval = get_maxval<ACCUM>();
  for (int y = 0; y < ans->h; y++) {
    ACCUM *row = ans->line_n1(y);
    for (int x = 0; x < ans->w; x++) {
      row[x] = maxval;
    }
  }
  if (region_masks) {
    if (region_masks->bmp->w != a->w || region_masks->bmp->h != a->h) { fprintf(stderr, "region_masks (%dx%d) size != a (%dx%d) size\n", region_masks->bmp->w, region_masks->bmp->h, a->w, a->h); exit(1); }
    if (region_masks->bmp->w != b->w || region_masks->bmp->h != b->h) { fprintf(stderr, "region_masks (%dx%d) size != b (%dx%d) size\n", region_masks->bmp->w, region_masks->bmp->h, b->w, b->h); exit(1); }
  }

  Box box = get_abox_vec(p, a, amask, 1); // need to trim patch!!!
	#pragma omp parallel for schedule(static, 4)
  for (int y = box.ymin; y < box.ymax; y++) {
    T *adata[PATCH_W][PATCH_W]; // aggregate a patch of vectors
		ACCUM *row = (ACCUM *) ans->line_n1(y);
    int *arow = amask ? (int *) amask->bmp->line[y]: NULL;
    for (int x = box.xmin; x < box.xmax; x++) {
      if (IS_MASK && amask && arow[x]) { continue; }
      int xp, yp;
      getnn(ann, x, y, xp, yp);

      if (IS_MASK && region_masks && ((int *) region_masks->bmp->line[y])[x] != ((int *) region_masks->bmp->line[yp])[xp]) { 
				continue; 
			}

			for (int dy = 0; dy < PATCH_W; dy++) { 
        for (int dx = 0; dx < PATCH_W; dx++) {
					adata[dy][dx] = a->get(x+dx, y+dy);
        }
      } 

      if (IS_MASK && bmask && ((int *) bmask->line[yp])[xp]) { continue; }
      row[x] = XCvec_fast_patch_nobranch<T, ACCUM, IS_WINDOW, PATCH_W>(adata, b, xp, yp, p);
			//if (x == 1 && y == 1) { printf("1, 1 => %d, %d (%d)\n", xp, yp, row[x]); }
		}
  }
  return ans;
}

template<class T, class ACCUM>
VECBITMAP<ACCUM> *XCvec_init_dist(Params *p, VECBITMAP<T> *a, VECBITMAP<T> *b, BITMAP *ann, BITMAP *bmask=NULL, RegionMasks *region_masks=NULL, RegionMasks *amask=NULL) {
	VECBITMAP<ACCUM> *ans = NULL;
	if (is_window(p)) { 
		if			(p->patch_w == 1) { ans = XCvec_init_dist_n<T, ACCUM, 1, 1, 1>(p, a, b, ann, bmask, region_masks, amask); }
		else if (p->patch_w == 2) { ans = XCvec_init_dist_n<T, ACCUM, 1, 1, 2>(p, a, b, ann, bmask, region_masks, amask); }
		else if (p->patch_w == 3) { ans = XCvec_init_dist_n<T, ACCUM, 1, 1, 3>(p, a, b, ann, bmask, region_masks, amask); }
		else if (p->patch_w == 4) { ans = XCvec_init_dist_n<T, ACCUM, 1, 1, 4>(p, a, b, ann, bmask, region_masks, amask); }
		else if (p->patch_w == 5) { ans = XCvec_init_dist_n<T, ACCUM, 1, 1, 5>(p, a, b, ann, bmask, region_masks, amask); }
		else if (p->patch_w == 6) { ans = XCvec_init_dist_n<T, ACCUM, 1, 1, 6>(p, a, b, ann, bmask, region_masks, amask); }
		else if (p->patch_w == 7) { ans = XCvec_init_dist_n<T, ACCUM, 1, 1, 7>(p, a, b, ann, bmask, region_masks, amask); }
		else if (p->patch_w == 8) { ans = XCvec_init_dist_n<T, ACCUM, 1, 1, 8>(p, a, b, ann, bmask, region_masks, amask); }
		else if (p->patch_w == 9) { ans = XCvec_init_dist_n<T, ACCUM, 1, 1, 9>(p, a, b, ann, bmask, region_masks, amask); }
		else if (p->patch_w == 10) { ans = XCvec_init_dist_n<T, ACCUM, 1, 1, 10>(p, a, b, ann, bmask, region_masks, amask); }
		else if (p->patch_w == 11) { ans = XCvec_init_dist_n<T, ACCUM, 1, 1, 11>(p, a, b, ann, bmask, region_masks, amask); }
		else if (p->patch_w == 12) { ans = XCvec_init_dist_n<T, ACCUM, 1, 1, 12>(p, a, b, ann, bmask, region_masks, amask); }
		else if (p->patch_w == 13) { ans = XCvec_init_dist_n<T, ACCUM, 1, 1, 13>(p, a, b, ann, bmask, region_masks, amask); }
		else if (p->patch_w == 14) { ans = XCvec_init_dist_n<T, ACCUM, 1, 1, 14>(p, a, b, ann, bmask, region_masks, amask); }
		else if (p->patch_w == 15) { ans = XCvec_init_dist_n<T, ACCUM, 1, 1, 15>(p, a, b, ann, bmask, region_masks, amask); }
		else if (p->patch_w == 16) { ans = XCvec_init_dist_n<T, ACCUM, 1, 1, 16>(p, a, b, ann, bmask, region_masks, amask); }
		else if (p->patch_w == 17) { ans = XCvec_init_dist_n<T, ACCUM, 1, 1, 17>(p, a, b, ann, bmask, region_masks, amask); }
		else if (p->patch_w == 18) { ans = XCvec_init_dist_n<T, ACCUM, 1, 1, 18>(p, a, b, ann, bmask, region_masks, amask); }
		else if (p->patch_w == 19) { ans = XCvec_init_dist_n<T, ACCUM, 1, 1, 19>(p, a, b, ann, bmask, region_masks, amask); }
		else if (p->patch_w == 20) { ans = XCvec_init_dist_n<T, ACCUM, 1, 1, 20>(p, a, b, ann, bmask, region_masks, amask); }
		else if (p->patch_w == 21) { ans = XCvec_init_dist_n<T, ACCUM, 1, 1, 21>(p, a, b, ann, bmask, region_masks, amask); }
		else if (p->patch_w == 22) { ans = XCvec_init_dist_n<T, ACCUM, 1, 1, 22>(p, a, b, ann, bmask, region_masks, amask); }
		else if (p->patch_w == 23) { ans = XCvec_init_dist_n<T, ACCUM, 1, 1, 23>(p, a, b, ann, bmask, region_masks, amask); }
		else if (p->patch_w == 24) { ans = XCvec_init_dist_n<T, ACCUM, 1, 1, 24>(p, a, b, ann, bmask, region_masks, amask); }
		else if (p->patch_w == 25) { ans = XCvec_init_dist_n<T, ACCUM, 1, 1, 25>(p, a, b, ann, bmask, region_masks, amask); }
		else if (p->patch_w == 26) { ans = XCvec_init_dist_n<T, ACCUM, 1, 1, 26>(p, a, b, ann, bmask, region_masks, amask); }
		else if (p->patch_w == 27) { ans = XCvec_init_dist_n<T, ACCUM, 1, 1, 27>(p, a, b, ann, bmask, region_masks, amask); }
		else if (p->patch_w == 28) { ans = XCvec_init_dist_n<T, ACCUM, 1, 1, 28>(p, a, b, ann, bmask, region_masks, amask); }
		else if (p->patch_w == 29) { ans = XCvec_init_dist_n<T, ACCUM, 1, 1, 29>(p, a, b, ann, bmask, region_masks, amask); }
		else if (p->patch_w == 30) { ans = XCvec_init_dist_n<T, ACCUM, 1, 1, 30>(p, a, b, ann, bmask, region_masks, amask); }
		else if (p->patch_w == 31) { ans = XCvec_init_dist_n<T, ACCUM, 1, 1, 31>(p, a, b, ann, bmask, region_masks, amask); }
		else if (p->patch_w == 32) { ans = XCvec_init_dist_n<T, ACCUM, 1, 1, 32>(p, a, b, ann, bmask, region_masks, amask); }
		else { fprintf(stderr, "Patch size unsupported: %d\n", p->patch_w); exit(1); }
	}
	else if (amask || bmask || region_masks) {
		if			(p->patch_w == 1) { ans = XCvec_init_dist_n<T, ACCUM, 1, 0, 1>(p, a, b, ann, bmask, region_masks, amask); }
		else if	(p->patch_w == 2) { ans = XCvec_init_dist_n<T, ACCUM, 1, 0, 2>(p, a, b, ann, bmask, region_masks, amask); }
		else if	(p->patch_w == 3) { ans = XCvec_init_dist_n<T, ACCUM, 1, 0, 3>(p, a, b, ann, bmask, region_masks, amask); }
		else if	(p->patch_w == 4) { ans = XCvec_init_dist_n<T, ACCUM, 1, 0, 4>(p, a, b, ann, bmask, region_masks, amask); }
		else if	(p->patch_w == 5) { ans = XCvec_init_dist_n<T, ACCUM, 1, 0, 5>(p, a, b, ann, bmask, region_masks, amask); }
		else if	(p->patch_w == 6) { ans = XCvec_init_dist_n<T, ACCUM, 1, 0, 6>(p, a, b, ann, bmask, region_masks, amask); }
		else if	(p->patch_w == 7) { ans = XCvec_init_dist_n<T, ACCUM, 1, 0, 7>(p, a, b, ann, bmask, region_masks, amask); }
		else if	(p->patch_w == 8) { ans = XCvec_init_dist_n<T, ACCUM, 1, 0, 8>(p, a, b, ann, bmask, region_masks, amask); }
		else if	(p->patch_w == 9) { ans = XCvec_init_dist_n<T, ACCUM, 1, 0, 9>(p, a, b, ann, bmask, region_masks, amask); }
		else if	(p->patch_w == 10) { ans = XCvec_init_dist_n<T, ACCUM, 1, 0, 10>(p, a, b, ann, bmask, region_masks, amask); }
		else if	(p->patch_w == 11) { ans = XCvec_init_dist_n<T, ACCUM, 1, 0, 11>(p, a, b, ann, bmask, region_masks, amask); }
		else if	(p->patch_w == 12) { ans = XCvec_init_dist_n<T, ACCUM, 1, 0, 12>(p, a, b, ann, bmask, region_masks, amask); }
		else if	(p->patch_w == 13) { ans = XCvec_init_dist_n<T, ACCUM, 1, 0, 13>(p, a, b, ann, bmask, region_masks, amask); }
		else if	(p->patch_w == 14) { ans = XCvec_init_dist_n<T, ACCUM, 1, 0, 14>(p, a, b, ann, bmask, region_masks, amask); }
		else if	(p->patch_w == 15) { ans = XCvec_init_dist_n<T, ACCUM, 1, 0, 15>(p, a, b, ann, bmask, region_masks, amask); }
		else if	(p->patch_w == 16) { ans = XCvec_init_dist_n<T, ACCUM, 1, 0, 16>(p, a, b, ann, bmask, region_masks, amask); }
		else if	(p->patch_w == 17) { ans = XCvec_init_dist_n<T, ACCUM, 1, 0, 17>(p, a, b, ann, bmask, region_masks, amask); }
		else if	(p->patch_w == 18) { ans = XCvec_init_dist_n<T, ACCUM, 1, 0, 18>(p, a, b, ann, bmask, region_masks, amask); }
		else if	(p->patch_w == 19) { ans = XCvec_init_dist_n<T, ACCUM, 1, 0, 19>(p, a, b, ann, bmask, region_masks, amask); }
		else if	(p->patch_w == 20) { ans = XCvec_init_dist_n<T, ACCUM, 1, 0, 20>(p, a, b, ann, bmask, region_masks, amask); }
		else if	(p->patch_w == 21) { ans = XCvec_init_dist_n<T, ACCUM, 1, 0, 21>(p, a, b, ann, bmask, region_masks, amask); }
		else if	(p->patch_w == 22) { ans = XCvec_init_dist_n<T, ACCUM, 1, 0, 22>(p, a, b, ann, bmask, region_masks, amask); }
		else if	(p->patch_w == 23) { ans = XCvec_init_dist_n<T, ACCUM, 1, 0, 23>(p, a, b, ann, bmask, region_masks, amask); }
		else if	(p->patch_w == 24) { ans = XCvec_init_dist_n<T, ACCUM, 1, 0, 24>(p, a, b, ann, bmask, region_masks, amask); }
		else if	(p->patch_w == 25) { ans = XCvec_init_dist_n<T, ACCUM, 1, 0, 25>(p, a, b, ann, bmask, region_masks, amask); }
		else if	(p->patch_w == 26) { ans = XCvec_init_dist_n<T, ACCUM, 1, 0, 26>(p, a, b, ann, bmask, region_masks, amask); }
		else if	(p->patch_w == 27) { ans = XCvec_init_dist_n<T, ACCUM, 1, 0, 27>(p, a, b, ann, bmask, region_masks, amask); }
		else if	(p->patch_w == 28) { ans = XCvec_init_dist_n<T, ACCUM, 1, 0, 28>(p, a, b, ann, bmask, region_masks, amask); }
		else if	(p->patch_w == 29) { ans = XCvec_init_dist_n<T, ACCUM, 1, 0, 29>(p, a, b, ann, bmask, region_masks, amask); }
		else if	(p->patch_w == 30) { ans = XCvec_init_dist_n<T, ACCUM, 1, 0, 30>(p, a, b, ann, bmask, region_masks, amask); }
		else if	(p->patch_w == 31) { ans = XCvec_init_dist_n<T, ACCUM, 1, 0, 31>(p, a, b, ann, bmask, region_masks, amask); }
		else if	(p->patch_w == 32) { ans = XCvec_init_dist_n<T, ACCUM, 1, 0, 32>(p, a, b, ann, bmask, region_masks, amask); }
		else { fprintf(stderr, "Patch size unsupported: %d\n", p->patch_w); exit(1); }
	}
	else { 
		if			(p->patch_w == 1) { ans = XCvec_init_dist_n<T, ACCUM, 0, 0, 1>(p, a, b, ann, bmask, region_masks, amask); }
		else if	(p->patch_w == 2) { ans = XCvec_init_dist_n<T, ACCUM, 0, 0, 2>(p, a, b, ann, bmask, region_masks, amask); }
		else if	(p->patch_w == 3) { ans = XCvec_init_dist_n<T, ACCUM, 0, 0, 3>(p, a, b, ann, bmask, region_masks, amask); }
		else if	(p->patch_w == 4) { ans = XCvec_init_dist_n<T, ACCUM, 0, 0, 4>(p, a, b, ann, bmask, region_masks, amask); }
		else if	(p->patch_w == 5) { ans = XCvec_init_dist_n<T, ACCUM, 0, 0, 5>(p, a, b, ann, bmask, region_masks, amask); }
		else if	(p->patch_w == 6) { ans = XCvec_init_dist_n<T, ACCUM, 0, 0, 6>(p, a, b, ann, bmask, region_masks, amask); }
		else if	(p->patch_w == 7) { ans = XCvec_init_dist_n<T, ACCUM, 0, 0, 7>(p, a, b, ann, bmask, region_masks, amask); }
		else if	(p->patch_w == 8) { ans = XCvec_init_dist_n<T, ACCUM, 0, 0, 8>(p, a, b, ann, bmask, region_masks, amask); }
		else if	(p->patch_w == 9) { ans = XCvec_init_dist_n<T, ACCUM, 0, 0, 9>(p, a, b, ann, bmask, region_masks, amask); }
		else if	(p->patch_w == 10) { ans = XCvec_init_dist_n<T, ACCUM, 0, 0, 10>(p, a, b, ann, bmask, region_masks, amask); }
		else if	(p->patch_w == 11) { ans = XCvec_init_dist_n<T, ACCUM, 0, 0, 11>(p, a, b, ann, bmask, region_masks, amask); }
		else if	(p->patch_w == 12) { ans = XCvec_init_dist_n<T, ACCUM, 0, 0, 12>(p, a, b, ann, bmask, region_masks, amask); }
		else if	(p->patch_w == 13) { ans = XCvec_init_dist_n<T, ACCUM, 0, 0, 13>(p, a, b, ann, bmask, region_masks, amask); }
		else if	(p->patch_w == 14) { ans = XCvec_init_dist_n<T, ACCUM, 0, 0, 14>(p, a, b, ann, bmask, region_masks, amask); }
		else if	(p->patch_w == 15) { ans = XCvec_init_dist_n<T, ACCUM, 0, 0, 15>(p, a, b, ann, bmask, region_masks, amask); }
		else if	(p->patch_w == 16) { ans = XCvec_init_dist_n<T, ACCUM, 0, 0, 16>(p, a, b, ann, bmask, region_masks, amask); }
		else if	(p->patch_w == 17) { ans = XCvec_init_dist_n<T, ACCUM, 0, 0, 17>(p, a, b, ann, bmask, region_masks, amask); }
		else if	(p->patch_w == 18) { ans = XCvec_init_dist_n<T, ACCUM, 0, 0, 18>(p, a, b, ann, bmask, region_masks, amask); }
		else if	(p->patch_w == 19) { ans = XCvec_init_dist_n<T, ACCUM, 0, 0, 19>(p, a, b, ann, bmask, region_masks, amask); }
		else if	(p->patch_w == 20) { ans = XCvec_init_dist_n<T, ACCUM, 0, 0, 20>(p, a, b, ann, bmask, region_masks, amask); }
		else if	(p->patch_w == 21) { ans = XCvec_init_dist_n<T, ACCUM, 0, 0, 21>(p, a, b, ann, bmask, region_masks, amask); }
		else if	(p->patch_w == 22) { ans = XCvec_init_dist_n<T, ACCUM, 0, 0, 22>(p, a, b, ann, bmask, region_masks, amask); }
		else if	(p->patch_w == 23) { ans = XCvec_init_dist_n<T, ACCUM, 0, 0, 23>(p, a, b, ann, bmask, region_masks, amask); }
		else if	(p->patch_w == 24) { ans = XCvec_init_dist_n<T, ACCUM, 0, 0, 24>(p, a, b, ann, bmask, region_masks, amask); }
		else if	(p->patch_w == 25) { ans = XCvec_init_dist_n<T, ACCUM, 0, 0, 25>(p, a, b, ann, bmask, region_masks, amask); }
		else if	(p->patch_w == 26) { ans = XCvec_init_dist_n<T, ACCUM, 0, 0, 26>(p, a, b, ann, bmask, region_masks, amask); }
		else if	(p->patch_w == 27) { ans = XCvec_init_dist_n<T, ACCUM, 0, 0, 27>(p, a, b, ann, bmask, region_masks, amask); }
		else if	(p->patch_w == 28) { ans = XCvec_init_dist_n<T, ACCUM, 0, 0, 28>(p, a, b, ann, bmask, region_masks, amask); }
		else if	(p->patch_w == 29) { ans = XCvec_init_dist_n<T, ACCUM, 0, 0, 29>(p, a, b, ann, bmask, region_masks, amask); }
		else if	(p->patch_w == 30) { ans = XCvec_init_dist_n<T, ACCUM, 0, 0, 30>(p, a, b, ann, bmask, region_masks, amask); }
		else if	(p->patch_w == 31) { ans = XCvec_init_dist_n<T, ACCUM, 0, 0, 31>(p, a, b, ann, bmask, region_masks, amask); }
		else if	(p->patch_w == 32) { ans = XCvec_init_dist_n<T, ACCUM, 0, 0, 32>(p, a, b, ann, bmask, region_masks, amask); }
		else { fprintf(stderr, "Patch size unsupported: %d\n", p->patch_w); exit(1); }	
	}
	return ans;
}

// similar to nn_n_cputiled
template<class T, class ACCUM, int IS_MASK, int IS_WINDOW, int PATCH_W>
void XCvec_nn_n(Params *p, VECBITMAP<T> *a, VECBITMAP<T> *b,
            BITMAP *ann, VECBITMAP<ACCUM> *annd,
            RegionMasks *amask=NULL, BITMAP *bmask=NULL,
            int level=0, int em_iter=0, RecomposeParams *rp=NULL, int offset_iter=0, int update_type=0, int cache_b=0,
            RegionMasks *region_masks=NULL, int tiles=-1) 
{
  if (tiles < 0) { tiles = p->cores; }
  printf("in vec_nn_n, masks are: %p %p %p, tiles=%d, rs_max=%d\n", amask, bmask, region_masks, tiles, p->rs_max);
  Box box = get_abox_vec(p, a, amask, 1); // !!! need to trim patch
  int nn_iter = 0;
  for (; nn_iter < p->nn_iters; nn_iter++) {
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

      int ystart = ymin, yfinal = ymax, ychange=1; // from up-left to bottom-right
			int xstart = xmin, xfinal = xmax, xchange=1; 
      if ((nn_iter + offset_iter) % 2 == 1) { 
        ystart = ymax-1; yfinal = ymin-1; ychange=-1; // from bottom-right to up-left
        xstart = xmax-1; xfinal = xmin-1; xchange=-1;
      }
      int dx = -xchange, dy = -ychange;

      int bew = b->w-PATCH_W, beh = b->h-PATCH_W;
      int max_mag = MAX(b->w, b->h);
      int rs_ipart = int(p->rs_iters);
      double rs_fpart = p->rs_iters - rs_ipart;
      int rs_max = p->rs_max;
      if (rs_max > max_mag) { rs_max = max_mag; }

			T* adata[PATCH_W][PATCH_W];
      for (int y = ystart; y != yfinal; y += ychange) {
        ACCUM *annd_row = annd->line_n1(y);
        int *amask_row = IS_MASK ? (amask ? (int *) amask->bmp->line[y]: NULL): NULL;
        for (int x = xstart; x != xfinal; x += xchange) {
          if (IS_MASK && amask && amask_row[x]) { continue; }

          for (int dy0 = 0; dy0 < PATCH_W; dy0++) { // copy a patch from a
						for (int dx0 = 0; dx0 < PATCH_W; dx0++) {
							adata[dy0][dx0] = a->get(x+dx0, y+dy0);
						}
					}
          
          int src_mask = IS_MASK ? (region_masks ? ((int *) region_masks->bmp->line[y])[x]: 0): 0;

          int xbest, ybest;
          getnn(ann, x, y, xbest, ybest);
          ACCUM err = annd_row[x];
          if (err == 0) { continue; }

          /* Propagate */
          if (p->do_propagate) {
						if(!IS_WINDOW) {
							
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
                  ACCUM err0 = annd_row[x+dx];

                  int xa = dx, xb = 0;
                  if (dx > 0) { xa = 0; xb = dx; }
                  ACCUM partial = 0;
                  for (int yi = 0; yi < PATCH_W; yi++) {
										T* c1 = a->get(x+xa, y+yi);
										T* c2 = b->get(xpp+xa, ypp+yi);
										T* c3 = a->get(x+xb+PATCH_W-1, y+yi);
										T* c4 = b->get(xpp+xb+PATCH_W-1, ypp+yi);
										for (int i = 0; i < p->vec_len; i ++) {
											ACCUM di12 = ((ACCUM)c1[i]) - ((ACCUM)c2[i]);
											ACCUM di34 = ((ACCUM)c3[i]) - ((ACCUM)c4[i]);
											partial += (di34*di34 - di12*di12); 
										}
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
                  ACCUM err0 = annd->line_n1(y+dy)[x];

                  int ya = dy, yb = 0;
                  if (dy > 0) { ya = 0; yb = dy; }
                  ACCUM partial = 0;
  								for (int xi = 0; xi < PATCH_W; xi++) {
										T* c1 = a->get(x+xi, y+ya);
										T* c2 = b->get(xpp+xi, ypp+ya);
										T* c3 = a->get(x+xi, y+yb+PATCH_W-1);
										T* c4 = b->get(xpp+xi, ypp+yb+PATCH_W-1);
										for (int i = 0; i < p->vec_len; i ++) {
											ACCUM di12 = ((ACCUM)c1[i]) - ((ACCUM)c2[i]);
											ACCUM di34 = ((ACCUM)c3[i]) - ((ACCUM)c4[i]);
											partial += di34*di34 - di12*di12; 
										}
	                }
                  err0 += (dy < 0) ? partial: -partial;
                  if (err0 < err) {
                    err = err0;
                    xbest = xpp;
                    ybest = ypp;
                  }
                }
							} // end of progagate y
						
						} // end of IS_WINDOW = false
						else { // IS_WINDOW = true
							
							/* Propagate x */
							if ((unsigned) (x+dx) < (unsigned) (ann->w-PATCH_W)) {
								int xpp, ypp;
								getnn(ann, x+dx, y, xpp, ypp);
								xpp -= dx;

								if (!IS_WINDOW || window_constraint_wrap(p, a, b, x, y, xpp, ypp)) {
									XCvec_attempt_n<T, ACCUM, IS_MASK, IS_WINDOW, PATCH_W>(err, xbest, ybest, adata, b, xpp, ypp, bmask, region_masks, src_mask, p);
								}
							}

							/* Propagate y */
							if ((unsigned) (y+dy) < (unsigned) (ann->h-PATCH_W)) {
								int xpp, ypp;
								getnn(ann, x, y+dy, xpp, ypp);
								ypp -= dy;

								if (!IS_WINDOW || window_constraint_wrap(p, a, b, x, y, xpp, ypp)) {
									XCvec_attempt_n<T, ACCUM, IS_MASK, IS_WINDOW, PATCH_W>(err, xbest, ybest, adata, b, xpp, ypp, bmask, region_masks, src_mask, p);
								}
							}
						} // end of IS_WINDOW = true
					} // end of do_propagation

          /* Random search */
          unsigned int seed = (x | (y<<11)) ^ iter_seed;
          seed = RANDI(seed);
          int rs_iters = 1-(seed*(1.0/(RAND_MAX-1))) < rs_fpart ? rs_ipart + 1: rs_ipart;
  //        int rs_iters = 1-random() < rs_fpart ? rs_ipart + 1: rs_ipart;

          int rs_max_curr = rs_max;
          for (int mag = rs_max_curr; mag >= p->rs_min; mag = int(mag*p->rs_ratio)) {
            for (int rs_iter = 0; rs_iter < rs_iters; rs_iter++) {
              int xmin = MAX(xbest-mag,0), xmax = MIN(xbest+mag+1,bew);
              int ymin = MAX(ybest-mag,0), ymax = MIN(ybest+mag+1,beh);
              seed = RANDI(seed);
              int xpp = xmin+seed%(xmax-xmin);
              seed = RANDI(seed);
              int ypp = ymin+seed%(ymax-ymin);
              if (!IS_WINDOW || window_constraint_wrap(p, a, b, x, y, xpp, ypp)) {
                XCvec_attempt_n<T, ACCUM, IS_MASK, IS_WINDOW, PATCH_W>(err, xbest, ybest, adata, b, xpp, ypp, bmask, region_masks, src_mask, p);
              }
            }
          }
          
          ((int *) ann->line[y])[x] = XY_TO_INT(xbest, ybest);
          annd_row[x] = err;
        } // x
      } // y
    } // parallel
  } // nn_iter
  printf("done vec_nn_n, %d iters, rs_max=%d\n", nn_iter, p->rs_max);
}


template<class T, class ACCUM>
void XCvec_nn(Params *p, VECBITMAP<T> *a, VECBITMAP<T> *b,
        BITMAP *ann, VECBITMAP<ACCUM> *annd,
        RegionMasks *amask=NULL, BITMAP *bmask=NULL,
        int level=0, int em_iter=0, RecomposeParams *rp=NULL, int offset_iter=0, int update_type=0, int cache_b=0,
        RegionMasks *region_masks=NULL, int tiles=-1) 
{
  if (p->algo == ALGO_CPU || p->algo == ALGO_CPUTILED) {
    if (is_window(p)) {
      printf("Running vec_nn (cputiled), using windowed and masked\n");
      if			(p->patch_w == 1) XCvec_nn_n<T, ACCUM, 1, 1, 1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if (p->patch_w == 2) XCvec_nn_n<T, ACCUM, 1, 1, 2>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if (p->patch_w == 3) XCvec_nn_n<T, ACCUM, 1, 1, 3>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if (p->patch_w == 4) XCvec_nn_n<T, ACCUM, 1, 1, 4>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if (p->patch_w == 5) XCvec_nn_n<T, ACCUM, 1, 1, 5>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if (p->patch_w == 6) XCvec_nn_n<T, ACCUM, 1, 1, 6>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if (p->patch_w == 7) XCvec_nn_n<T, ACCUM, 1, 1, 7>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if (p->patch_w == 8) XCvec_nn_n<T, ACCUM, 1, 1, 8>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if (p->patch_w == 9) XCvec_nn_n<T, ACCUM, 1, 1, 9>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if (p->patch_w == 10) XCvec_nn_n<T, ACCUM, 1, 1, 10>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if (p->patch_w == 11) XCvec_nn_n<T, ACCUM, 1, 1, 11>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if (p->patch_w == 12) XCvec_nn_n<T, ACCUM, 1, 1, 12>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if (p->patch_w == 13) XCvec_nn_n<T, ACCUM, 1, 1, 13>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if (p->patch_w == 14) XCvec_nn_n<T, ACCUM, 1, 1, 14>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if (p->patch_w == 15) XCvec_nn_n<T, ACCUM, 1, 1, 15>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if (p->patch_w == 16) XCvec_nn_n<T, ACCUM, 1, 1, 16>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if (p->patch_w == 17) XCvec_nn_n<T, ACCUM, 1, 1, 17>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if (p->patch_w == 18) XCvec_nn_n<T, ACCUM, 1, 1, 18>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if (p->patch_w == 19) XCvec_nn_n<T, ACCUM, 1, 1, 19>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if (p->patch_w == 20) XCvec_nn_n<T, ACCUM, 1, 1, 20>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if (p->patch_w == 21) XCvec_nn_n<T, ACCUM, 1, 1, 21>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if (p->patch_w == 22) XCvec_nn_n<T, ACCUM, 1, 1, 22>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if (p->patch_w == 23) XCvec_nn_n<T, ACCUM, 1, 1, 23>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if (p->patch_w == 24) XCvec_nn_n<T, ACCUM, 1, 1, 24>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if (p->patch_w == 25) XCvec_nn_n<T, ACCUM, 1, 1, 25>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if (p->patch_w == 26) XCvec_nn_n<T, ACCUM, 1, 1, 26>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if (p->patch_w == 27) XCvec_nn_n<T, ACCUM, 1, 1, 27>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if (p->patch_w == 28) XCvec_nn_n<T, ACCUM, 1, 1, 28>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if (p->patch_w == 29) XCvec_nn_n<T, ACCUM, 1, 1, 29>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if (p->patch_w == 30) XCvec_nn_n<T, ACCUM, 1, 1, 30>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if (p->patch_w == 31) XCvec_nn_n<T, ACCUM, 1, 1, 31>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if (p->patch_w == 32) XCvec_nn_n<T, ACCUM, 1, 1, 32>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else { fprintf(stderr, "Patch size unsupported: %d\n", p->patch_w); exit(1); }	
		}
		else if (bmask == NULL && amask == NULL && region_masks == NULL) {
      printf("Running vec_nn (cputiled), using unmasked\n");
			if			(p->patch_w == 1) XCvec_nn_n<T, ACCUM, 0, 0, 1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if (p->patch_w == 2) XCvec_nn_n<T, ACCUM, 0, 0, 2>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if (p->patch_w == 3) XCvec_nn_n<T, ACCUM, 0, 0, 3>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if (p->patch_w == 4) XCvec_nn_n<T, ACCUM, 0, 0, 4>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if (p->patch_w == 5) XCvec_nn_n<T, ACCUM, 0, 0, 5>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if (p->patch_w == 6) XCvec_nn_n<T, ACCUM, 0, 0, 6>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if (p->patch_w == 7) XCvec_nn_n<T, ACCUM, 0, 0, 7>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if (p->patch_w == 8) XCvec_nn_n<T, ACCUM, 0, 0, 8>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if (p->patch_w == 9) XCvec_nn_n<T, ACCUM, 0, 0, 9>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if (p->patch_w == 10) XCvec_nn_n<T, ACCUM, 0, 0, 10>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if (p->patch_w == 11) XCvec_nn_n<T, ACCUM, 0, 0, 11>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if (p->patch_w == 12) XCvec_nn_n<T, ACCUM, 0, 0, 12>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if (p->patch_w == 13) XCvec_nn_n<T, ACCUM, 0, 0, 13>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if (p->patch_w == 14) XCvec_nn_n<T, ACCUM, 0, 0, 14>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if (p->patch_w == 15) XCvec_nn_n<T, ACCUM, 0, 0, 15>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if (p->patch_w == 16) XCvec_nn_n<T, ACCUM, 0, 0, 16>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if (p->patch_w == 17) XCvec_nn_n<T, ACCUM, 0, 0, 17>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if (p->patch_w == 18) XCvec_nn_n<T, ACCUM, 0, 0, 18>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if (p->patch_w == 19) XCvec_nn_n<T, ACCUM, 0, 0, 19>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if (p->patch_w == 20) XCvec_nn_n<T, ACCUM, 0, 0, 20>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if (p->patch_w == 21) XCvec_nn_n<T, ACCUM, 0, 0, 21>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if (p->patch_w == 22) XCvec_nn_n<T, ACCUM, 0, 0, 22>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if (p->patch_w == 23) XCvec_nn_n<T, ACCUM, 0, 0, 23>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if (p->patch_w == 24) XCvec_nn_n<T, ACCUM, 0, 0, 24>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if (p->patch_w == 25) XCvec_nn_n<T, ACCUM, 0, 0, 25>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if (p->patch_w == 26) XCvec_nn_n<T, ACCUM, 0, 0, 26>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if (p->patch_w == 27) XCvec_nn_n<T, ACCUM, 0, 0, 27>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if (p->patch_w == 28) XCvec_nn_n<T, ACCUM, 0, 0, 28>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if (p->patch_w == 29) XCvec_nn_n<T, ACCUM, 0, 0, 29>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if (p->patch_w == 30) XCvec_nn_n<T, ACCUM, 0, 0, 30>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if (p->patch_w == 31) XCvec_nn_n<T, ACCUM, 0, 0, 31>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if (p->patch_w == 32) XCvec_nn_n<T, ACCUM, 0, 0, 32>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else { fprintf(stderr, "Patch size unsupported: %d\n", p->patch_w); exit(1); }	
    } 
		else {
      printf("Running vec_nn (cputiled), using masked\n");
      if			(p->patch_w == 1) XCvec_nn_n<T, ACCUM, 1, 0, 1>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if	(p->patch_w == 2) XCvec_nn_n<T, ACCUM, 1, 0, 2>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if	(p->patch_w == 3) XCvec_nn_n<T, ACCUM, 1, 0, 3>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if	(p->patch_w == 4) XCvec_nn_n<T, ACCUM, 1, 0, 4>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if	(p->patch_w == 5) XCvec_nn_n<T, ACCUM, 1, 0, 5>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if	(p->patch_w == 6) XCvec_nn_n<T, ACCUM, 1, 0, 6>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if	(p->patch_w == 7) XCvec_nn_n<T, ACCUM, 1, 0, 7>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if	(p->patch_w == 8) XCvec_nn_n<T, ACCUM, 1, 0, 8>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if	(p->patch_w == 9) XCvec_nn_n<T, ACCUM, 1, 0, 9>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if	(p->patch_w == 10) XCvec_nn_n<T, ACCUM, 1, 0, 10>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if	(p->patch_w == 11) XCvec_nn_n<T, ACCUM, 1, 0, 11>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if	(p->patch_w == 12) XCvec_nn_n<T, ACCUM, 1, 0, 12>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if	(p->patch_w == 13) XCvec_nn_n<T, ACCUM, 1, 0, 13>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if	(p->patch_w == 14) XCvec_nn_n<T, ACCUM, 1, 0, 14>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if	(p->patch_w == 15) XCvec_nn_n<T, ACCUM, 1, 0, 15>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if	(p->patch_w == 16) XCvec_nn_n<T, ACCUM, 1, 0, 16>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if	(p->patch_w == 17) XCvec_nn_n<T, ACCUM, 1, 0, 17>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if	(p->patch_w == 18) XCvec_nn_n<T, ACCUM, 1, 0, 18>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if	(p->patch_w == 19) XCvec_nn_n<T, ACCUM, 1, 0, 19>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if	(p->patch_w == 20) XCvec_nn_n<T, ACCUM, 1, 0, 20>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if	(p->patch_w == 21) XCvec_nn_n<T, ACCUM, 1, 0, 21>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if	(p->patch_w == 22) XCvec_nn_n<T, ACCUM, 1, 0, 22>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if	(p->patch_w == 23) XCvec_nn_n<T, ACCUM, 1, 0, 23>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if	(p->patch_w == 24) XCvec_nn_n<T, ACCUM, 1, 0, 24>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if	(p->patch_w == 25) XCvec_nn_n<T, ACCUM, 1, 0, 25>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if	(p->patch_w == 26) XCvec_nn_n<T, ACCUM, 1, 0, 26>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if	(p->patch_w == 27) XCvec_nn_n<T, ACCUM, 1, 0, 27>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if	(p->patch_w == 28) XCvec_nn_n<T, ACCUM, 1, 0, 28>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if	(p->patch_w == 29) XCvec_nn_n<T, ACCUM, 1, 0, 29>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if	(p->patch_w == 30) XCvec_nn_n<T, ACCUM, 1, 0, 30>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if	(p->patch_w == 31) XCvec_nn_n<T, ACCUM, 1, 0, 31>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else if	(p->patch_w == 32) XCvec_nn_n<T, ACCUM, 1, 0, 32>(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
			else { fprintf(stderr, "Patch size unsupported: %d\n", p->patch_w); exit(1); }	
    }
  } 
	else {
    fprintf(stderr, "vec_nn: algorithm %d unsupported\n", p->algo); exit(1);
  }
}


template<class T, class ACCUM, int IS_WINDOW, int HAS_MASKS, int PATCH_W>
void XCvec_minnn_n(Params *p, VECBITMAP<T> *a, VECBITMAP<T> *b, BITMAP *ann, VECBITMAP<ACCUM> *annd, BITMAP *ann_prev, BITMAP *bmask, int level, int em_iter, RecomposeParams *rp, RegionMasks *region_masks, RegionMasks *amask, int ntiles) {
  if (ntiles < 0) { ntiles = p->cores; }
  printf("vec_minnn: %d %d %d %d, tiles=%d\n", ann->w, ann->h, ann_prev->w, ann_prev->h, ntiles);
  if (!rp) { fprintf(stderr, "vec_minnn_n: rp is NULL\n"); exit(1); }
//  double start_t = accurate_timer();
  Box box = get_abox_vec(p, a, amask, 1); // trim patch !!!

  #pragma omp parallel for schedule(static,4) num_threads(ntiles)
  for (int y = box.ymin; y < box.ymax; y++) {
    int *amask_row = amask ? (int *) amask->bmp->line[y]: NULL;
    ACCUM *annd_row = (ACCUM *) annd->line_n1(y);
    for (int x = box.xmin; x < box.xmax; x++) {
      if (HAS_MASKS && amask && amask_row[x]) { continue; }
      ACCUM dcurrent = annd_row[x];
      int xp, yp;
      getnn(ann_prev, x, y, xp, yp);
      if ((unsigned) xp >= (unsigned) (b->w-p->patch_w+1) ||
          (unsigned) yp >= (unsigned) (b->h-p->patch_w+1)) { continue; }
      if (HAS_MASKS && bmask && ((int *) bmask->line[yp])[xp]) { continue; }

      ACCUM dprev = XCvec_patch_dist_ab<T, ACCUM, IS_WINDOW, HAS_MASKS, PATCH_W>(p, a, x, y, b, xp, yp, dcurrent, region_masks);

      if (dprev < dcurrent) {
        _putpixel32(ann, x, y, XY_TO_INT(xp, yp));
        annd_row[x] = dprev;
      }
    }
  }
//  nn_time += accurate_timer() - start_t;
  Params pcopy(*p);
  pcopy.nn_iters = rp->minnn_optp_nn_iters;
  pcopy.rs_max = rp->minnn_optp_rs_max;
  
  XCvec_nn<T, ACCUM>(&pcopy, a, b, ann, annd, amask, bmask, level, em_iter, rp, 0, 0, 1, region_masks, ntiles);
}



template<class T, class ACCUM>
void XCvec_minnn(Params *p, VECBITMAP<T> *a, VECBITMAP<T> *b, BITMAP *ann, VECBITMAP<ACCUM> *annd, BITMAP *ann_prev, BITMAP *bmask, int level, int em_iter, RecomposeParams *rp, RegionMasks *region_masks, RegionMasks *amask, int ntiles) {
	if (is_window(p)) {
		if			(p->patch_w == 1) { return XCvec_minnn_n<T, ACCUM, 1, 1, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 2) { return XCvec_minnn_n<T, ACCUM, 1, 1, 2>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 3) { return XCvec_minnn_n<T, ACCUM, 1, 1, 3>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 4) { return XCvec_minnn_n<T, ACCUM, 1, 1, 4>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 5) { return XCvec_minnn_n<T, ACCUM, 1, 1, 5>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 6) { return XCvec_minnn_n<T, ACCUM, 1, 1, 6>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 7) { return XCvec_minnn_n<T, ACCUM, 1, 1, 7>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 8) { return XCvec_minnn_n<T, ACCUM, 1, 1, 8>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 9) { return XCvec_minnn_n<T, ACCUM, 1, 1, 9>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 10) { return XCvec_minnn_n<T, ACCUM, 1, 1, 10>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 11) { return XCvec_minnn_n<T, ACCUM, 1, 1, 11>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 12) { return XCvec_minnn_n<T, ACCUM, 1, 1, 12>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 13) { return XCvec_minnn_n<T, ACCUM, 1, 1, 13>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 14) { return XCvec_minnn_n<T, ACCUM, 1, 1, 14>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 15) { return XCvec_minnn_n<T, ACCUM, 1, 1, 15>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 16) { return XCvec_minnn_n<T, ACCUM, 1, 1, 16>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 17) { return XCvec_minnn_n<T, ACCUM, 1, 1, 17>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 18) { return XCvec_minnn_n<T, ACCUM, 1, 1, 18>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 19) { return XCvec_minnn_n<T, ACCUM, 1, 1, 19>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 20) { return XCvec_minnn_n<T, ACCUM, 1, 1, 20>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 21) { return XCvec_minnn_n<T, ACCUM, 1, 1, 21>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 22) { return XCvec_minnn_n<T, ACCUM, 1, 1, 22>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 23) { return XCvec_minnn_n<T, ACCUM, 1, 1, 23>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 24) { return XCvec_minnn_n<T, ACCUM, 1, 1, 24>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 25) { return XCvec_minnn_n<T, ACCUM, 1, 1, 25>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 26) { return XCvec_minnn_n<T, ACCUM, 1, 1, 26>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 27) { return XCvec_minnn_n<T, ACCUM, 1, 1, 27>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 28) { return XCvec_minnn_n<T, ACCUM, 1, 1, 28>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 29) { return XCvec_minnn_n<T, ACCUM, 1, 1, 29>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 30) { return XCvec_minnn_n<T, ACCUM, 1, 1, 30>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 31) { return XCvec_minnn_n<T, ACCUM, 1, 1, 31>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 32) { return XCvec_minnn_n<T, ACCUM, 1, 1, 32>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else { fprintf(stderr, "Patch size unsupported: %d\n", p->patch_w); exit(1); }	
	}
	else if (bmask || region_masks || amask) {
		if			(p->patch_w == 1) { return XCvec_minnn_n<T, ACCUM, 0, 1, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 2) { return XCvec_minnn_n<T, ACCUM, 0, 1, 2>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 3) { return XCvec_minnn_n<T, ACCUM, 0, 1, 3>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 4) { return XCvec_minnn_n<T, ACCUM, 0, 1, 4>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 5) { return XCvec_minnn_n<T, ACCUM, 0, 1, 5>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 6) { return XCvec_minnn_n<T, ACCUM, 0, 1, 6>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 7) { return XCvec_minnn_n<T, ACCUM, 0, 1, 7>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 8) { return XCvec_minnn_n<T, ACCUM, 0, 1, 8>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 9) { return XCvec_minnn_n<T, ACCUM, 0, 1, 9>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 10) { return XCvec_minnn_n<T, ACCUM, 0, 1, 10>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 11) { return XCvec_minnn_n<T, ACCUM, 0, 1, 11>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 12) { return XCvec_minnn_n<T, ACCUM, 0, 1, 12>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 13) { return XCvec_minnn_n<T, ACCUM, 0, 1, 13>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 14) { return XCvec_minnn_n<T, ACCUM, 0, 1, 14>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 15) { return XCvec_minnn_n<T, ACCUM, 0, 1, 15>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 16) { return XCvec_minnn_n<T, ACCUM, 0, 1, 16>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 17) { return XCvec_minnn_n<T, ACCUM, 0, 1, 17>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 18) { return XCvec_minnn_n<T, ACCUM, 0, 1, 18>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 19) { return XCvec_minnn_n<T, ACCUM, 0, 1, 19>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 20) { return XCvec_minnn_n<T, ACCUM, 0, 1, 20>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 21) { return XCvec_minnn_n<T, ACCUM, 0, 1, 21>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 22) { return XCvec_minnn_n<T, ACCUM, 0, 1, 22>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 23) { return XCvec_minnn_n<T, ACCUM, 0, 1, 23>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 24) { return XCvec_minnn_n<T, ACCUM, 0, 1, 24>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 25) { return XCvec_minnn_n<T, ACCUM, 0, 1, 25>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 26) { return XCvec_minnn_n<T, ACCUM, 0, 1, 26>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 27) { return XCvec_minnn_n<T, ACCUM, 0, 1, 27>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 28) { return XCvec_minnn_n<T, ACCUM, 0, 1, 28>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 29) { return XCvec_minnn_n<T, ACCUM, 0, 1, 29>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 30) { return XCvec_minnn_n<T, ACCUM, 0, 1, 30>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 31) { return XCvec_minnn_n<T, ACCUM, 0, 1, 31>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 32) { return XCvec_minnn_n<T, ACCUM, 0, 1, 32>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else { fprintf(stderr, "Patch size unsupported: %d\n", p->patch_w); exit(1); }	
	}
	else {
		if			(p->patch_w == 1) { return XCvec_minnn_n<T, ACCUM, 0, 0, 1>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 2) { return XCvec_minnn_n<T, ACCUM, 0, 0, 2>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 3) { return XCvec_minnn_n<T, ACCUM, 0, 0, 3>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 4) { return XCvec_minnn_n<T, ACCUM, 0, 0, 4>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 5) { return XCvec_minnn_n<T, ACCUM, 0, 0, 5>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 6) { return XCvec_minnn_n<T, ACCUM, 0, 0, 6>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 7) { return XCvec_minnn_n<T, ACCUM, 0, 0, 7>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 8) { return XCvec_minnn_n<T, ACCUM, 0, 0, 8>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 9) { return XCvec_minnn_n<T, ACCUM, 0, 0, 9>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 10) { return XCvec_minnn_n<T, ACCUM, 0, 0, 10>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 11) { return XCvec_minnn_n<T, ACCUM, 0, 0, 11>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 12) { return XCvec_minnn_n<T, ACCUM, 0, 0, 12>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 13) { return XCvec_minnn_n<T, ACCUM, 0, 0, 13>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 14) { return XCvec_minnn_n<T, ACCUM, 0, 0, 14>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 15) { return XCvec_minnn_n<T, ACCUM, 0, 0, 15>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 16) { return XCvec_minnn_n<T, ACCUM, 0, 0, 16>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 17) { return XCvec_minnn_n<T, ACCUM, 0, 0, 17>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 18) { return XCvec_minnn_n<T, ACCUM, 0, 0, 18>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 19) { return XCvec_minnn_n<T, ACCUM, 0, 0, 19>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 20) { return XCvec_minnn_n<T, ACCUM, 0, 0, 20>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 21) { return XCvec_minnn_n<T, ACCUM, 0, 0, 21>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 22) { return XCvec_minnn_n<T, ACCUM, 0, 0, 22>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 23) { return XCvec_minnn_n<T, ACCUM, 0, 0, 23>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 24) { return XCvec_minnn_n<T, ACCUM, 0, 0, 24>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 25) { return XCvec_minnn_n<T, ACCUM, 0, 0, 25>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 26) { return XCvec_minnn_n<T, ACCUM, 0, 0, 26>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 27) { return XCvec_minnn_n<T, ACCUM, 0, 0, 27>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 28) { return XCvec_minnn_n<T, ACCUM, 0, 0, 28>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 29) { return XCvec_minnn_n<T, ACCUM, 0, 0, 29>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 30) { return XCvec_minnn_n<T, ACCUM, 0, 0, 30>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 31) { return XCvec_minnn_n<T, ACCUM, 0, 0, 31>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else if (p->patch_w == 32) { return XCvec_minnn_n<T, ACCUM, 0, 0, 32>(p, a, b, ann, annd, ann_prev, bmask, level, em_iter, rp, region_masks, amask, ntiles); }
		else { fprintf(stderr, "Patch size unsupported: %d\n", p->patch_w); exit(1); }	
	}
}



// ------------------------------
// Similarity stuff hereafter
// ------------------------------

BITMAP *vecbitmap_to_bitmap(VECBITMAP<int> *a);

#define VEC_MODE_PATCH 0
#define VEC_MODE_DESC  1
#define VEC_MODE_SIM   2

BITMAP *vecwrap_init_nn(int vec_mode, Params *p, BITMAP *a, BITMAP *b, BITMAP *bmask=NULL, RegionMasks *region_masks=NULL, RegionMasks *amask=NULL, BITMAP **ann_sim=NULL);
BITMAP *vecwrap_init_dist(int vec_mode, Params *p, BITMAP *a, BITMAP *b, BITMAP *ann, BITMAP *bmask=NULL, RegionMasks *region_masks=NULL, RegionMasks *amask=NULL, BITMAP *ann_sim=NULL);
void vecwrap_nn(int vec_mode, Params *p, BITMAP *a, BITMAP *b,
        BITMAP *ann, BITMAP *annd,
        RegionMasks *amask=NULL, BITMAP *bmask=NULL,
        int level=0, int em_iter=0, RecomposeParams *rp=NULL, int offset_iter=0, int update_type=0, int cache_b=0,
        RegionMasks *region_masks=NULL, int tiles=-1, BITMAP *ann_sim=NULL);
				BITMAP *vecwrap_vote(int vec_mode, Params *p, BITMAP *b,
				BITMAP *ann, BITMAP *ann_sim=NULL, BITMAP *bnn=NULL,
				BITMAP *bmask=NULL, BITMAP *bweight=NULL,
				double coherence_weight=COHERENCE_WEIGHT, double complete_weight=COMPLETE_WEIGHT,
				RegionMasks *amask=NULL, BITMAP *aweight=NULL, BITMAP *ainit=NULL, RegionMasks *region_masks=NULL, BITMAP *aconstraint=NULL, int mask_self_only=0);

#endif
