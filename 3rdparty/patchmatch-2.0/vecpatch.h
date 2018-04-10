
/* Patch distance templates for vector input. */

#ifndef _vecpatch_h
#define _vecpatch_h

#include "vecnn.h"

template<class T, class ACCUM, int IS_WINDOW>
ACCUM vec_fast_patch_dist(T *apatch, T *bpatch, ACCUM maxval, Params *p) {
  ACCUM ans = 0;
  for (int i = 0; i < p->vec_len; i++) {
    ACCUM d = ((ACCUM) apatch[i])-((ACCUM) bpatch[i]);
    ans += d*d;
    if (ans > maxval) { return ans; }
  }
  return ans;
}

template<class T, class ACCUM, int IS_MASK, int IS_WINDOW>
void vec_attempt_n(ACCUM &err, int &xbest, int &ybest, T *apatch, VECBITMAP<T> *b, int bx, int by, BITMAP *bmask, RegionMasks *region_masks, int src_mask, Params *p) {
  if ((bx != xbest || by != ybest) &&
      (unsigned) bx < (unsigned) (b->w) &&
      (unsigned) by < (unsigned) (b->h)) 
	{
    if (IS_MASK && region_masks && src_mask != ((int *) region_masks->bmp->line[by])[bx]) { return; }
    if (IS_MASK && bmask && ((int *) bmask->line[by])[bx]) { return; }
    ACCUM current = vec_fast_patch_dist<T, ACCUM, IS_WINDOW>(apatch, b->get(bx, by), err, p);
    if (current < err) {
      err = current;
      xbest = bx;
      ybest = by;
    }
  }
}

template<class T, class ACCUM, int IS_WINDOW>
ACCUM vec_fast_patch_nobranch(T *apatch, T *bpatch, Params *p) {
  ACCUM ans = 0;
  for (int i = 0; i < p->vec_len; i++) {
    ACCUM d = ((ACCUM) apatch[i])-((ACCUM) bpatch[i]);
    ans += d*d;
  }
  return ans;
}

template<class T, class ACCUM, int IS_WINDOW, int HAS_MASKS>
ACCUM vec_patch_dist_ab(Params *p, VECBITMAP<T> *a, int ax, int ay, VECBITMAP<T> *b, int bx, int by, ACCUM maxval, RegionMasks *region_masks) {
  if (HAS_MASKS && region_masks && ((int *) region_masks->bmp->line[ay])[ax] != ((int *) region_masks->bmp->line[by])[bx]) 
		return get_maxval<ACCUM>();
  return vec_fast_patch_nobranch<T, ACCUM, IS_WINDOW>(a->get(ax, ay), b->get(bx, by), p);
}


// -----------------------------
// XC version
// -----------------------------

template<class T, class ACCUM, int IS_WINDOW, int PATCH_W>
ACCUM XCvec_fast_patch_nobranch(T *adata[PATCH_W][PATCH_W], VECBITMAP<T> *b, int bx, int by, Params *p) 
{
  ACCUM ans = 0;

	// aggregate a patch of vectors for b, assume we have trimed patches to possible choices
	T *bdata[PATCH_W][PATCH_W]; 
	for (int dy = 0; dy < PATCH_W; dy++) { 
    for (int dx = 0; dx < PATCH_W; dx++) {
			bdata[dy][dx] = b->get(bx+dx, by+dy);
    }
  } 

	// patch distance
	for (int dy = 0; dy < PATCH_W; dy++) {
		for (int dx = 0; dx < PATCH_W; dx++) {
			T *apatch = adata[dy][dx];
			T *bpatch = bdata[dy][dx];
			for(int i = 0; i < p->vec_len; i ++) {
				ACCUM d = ((ACCUM) apatch[i])-((ACCUM) bpatch[i]);
				ans += d*d;
			}
		}
	}
	
	return ans;
}


template<class T, class ACCUM, int IS_WINDOW, int PATCH_W>
ACCUM XCvec_fast_patch_dist(T *adata[PATCH_W][PATCH_W], VECBITMAP<T> *b, int bx, int by, ACCUM maxval, Params *p) {
  ACCUM ans = 0;

	// aggregate a patch of vectors for b, assume we have trimed patches to possible choices
	T *bdata[PATCH_W][PATCH_W]; 
	for (int dy = 0; dy < PATCH_W; dy++) { 
    for (int dx = 0; dx < PATCH_W; dx++) {
			bdata[dy][dx] = b->get(bx+dx, by+dy);
    }
  } 

	// patch distance
	for (int dy = 0; dy < PATCH_W; dy++) {
		for (int dx = 0; dx < PATCH_W; dx++) {
			T *apatch = adata[dy][dx];
			T *bpatch = bdata[dy][dx];
			for(int i = 0; i < p->vec_len; i ++) {
				ACCUM d = ((ACCUM) apatch[i])-((ACCUM) bpatch[i]);
				ans += d*d;
				if (ans > maxval) { return ans; }
			}
		}
	}
	
	return ans;
}


template<class T, class ACCUM, int IS_MASK, int IS_WINDOW, int PATCH_W>
void XCvec_attempt_n(ACCUM &err, int &xbest, int &ybest, T *adata[PATCH_W][PATCH_W], VECBITMAP<T> *b, int bx, int by, BITMAP *bmask, RegionMasks *region_masks, int src_mask, Params *p) {
  if ((bx != xbest || by != ybest) &&
      (unsigned) bx < (unsigned) (b->w-PATCH_W+1) &&
      (unsigned) by < (unsigned) (b->h-PATCH_W+1)) 
	{
    if (IS_MASK && region_masks && src_mask != ((int *) region_masks->bmp->line[by])[bx]) { return; }
    if (IS_MASK && bmask && ((int *) bmask->line[by])[bx]) { return; }
    ACCUM current = XCvec_fast_patch_dist<T, ACCUM, IS_WINDOW, PATCH_W>(adata, b, bx, by, err, p);
    if (current < err) {
      err = current;
      xbest = bx;
      ybest = by;
    }
  }
}


template<class T, class ACCUM, int IS_WINDOW, int HAS_MASKS, int PATCH_W>
ACCUM XCvec_patch_dist_ab(Params *p, VECBITMAP<T> *a, int ax, int ay, VECBITMAP<T> *b, int bx, int by, ACCUM maxval, RegionMasks *region_masks) {
  if (HAS_MASKS && region_masks && ((int *) region_masks->bmp->line[ay])[ax] != ((int *) region_masks->bmp->line[by])[bx]) 
		return get_maxval<ACCUM>();

	// aggregate a patch of vectors for a, assume we have trimed patches to possible choices
	T *adata[PATCH_W][PATCH_W]; 
	for (int dy = 0; dy < PATCH_W; dy++) { 
    for (int dx = 0; dx < PATCH_W; dx++) {
			adata[dy][dx] = a->get(ax+dx, ay+dy);
    }
  } 

  return XCvec_fast_patch_nobranch<T, ACCUM, IS_WINDOW, PATCH_W>(adata, b, bx, by, p);
}


#endif
