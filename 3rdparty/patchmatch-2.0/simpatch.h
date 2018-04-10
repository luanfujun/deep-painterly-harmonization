
/* Patch distance templates for similarity transform (rotation+scale). */

#ifndef _simpatch_h
#define _simpatch_h

#include "patch.h"
#include "simnn.h"

// Turn one of these on. The fastest and most accurate way to sample is actually to nearest neighbor sample from a bilinearly upsampled image, but that's not implemented here.
#define SAMPLE_NN             1
#define SAMPLE_BILINEAR_EXACT 0

template<int TPATCH_W, int DO_BRANCH>
int sim_fast_patch_dist(int *adata, BITMAP *b, XFORM bpos, int maxval) {
  /* Do bounds checking outside the inner loop */
  int ul_x = bpos.x0;
  int ul_y = bpos.y0;
  int ur_x = bpos.x0+bpos.dxdu*(TPATCH_W-1);
  int ur_y = bpos.y0+bpos.dydu*(TPATCH_W-1);
  int ll_x = bpos.x0+bpos.dxdv*(TPATCH_W-1);
  int ll_y = bpos.y0+bpos.dydv*(TPATCH_W-1);
  int lr_x = ll_x+bpos.dxdu*(TPATCH_W-1);
  int lr_y = ll_y+bpos.dydu*(TPATCH_W-1);
  int bw16 = (b->w-1)<<16, bh16 = (b->h-1)<<16;
  if ((unsigned) ul_x >= (unsigned) bw16 ||
      (unsigned) ul_y >= (unsigned) bh16 ||
      (unsigned) ur_x >= (unsigned) bw16 ||
      (unsigned) ur_y >= (unsigned) bh16 ||
      (unsigned) ll_x >= (unsigned) bw16 ||
      (unsigned) ll_y >= (unsigned) bh16 ||
      (unsigned) lr_x >= (unsigned) bw16 ||
      (unsigned) lr_y >= (unsigned) bh16) { return INT_MAX-4096; }

  int ans = 0;
  int bx_row = bpos.x0, by_row = bpos.y0;
#if SAMPLE_NN
  bx_row += 32768;
  by_row += 32768;
#endif
  for (int dy = 0; dy < TPATCH_W; dy++) {
    int bx = bx_row, by = by_row;
    for (int dx = 0; dx < TPATCH_W; dx++) {
      unsigned int c1 = adata[dx];
      int r2, g2, b2;
#if SAMPLE_BILINEAR_EXACT
//      getpixel_bilin(b, bx, by, r2, g2, b2);   // A slower and safer method is to do bounds checking in the inner loop
      int bxi = bx>>16, byi = by>>16;
      int bxf = bx&((1<<16)-1), byf = by&((1<<16)-1);

      int *row1 = ((int *) b->line[byi])+bxi;
      int *row2 = ((int *) b->line[byi+1])+bxi;
      int cul = row1[0], cur = row1[1];
      int cll = row2[0], clr = row2[1];

      int rul = cul&255, rur = cur&255;
      int rll = cll&255, rlr = clr&255;

      int gul = (cul>>8)&255, gur = (cur>>8)&255;
      int gll = (cll>>8)&255, glr = (clr>>8)&255;

      int bul = cul>>16, bur = cur>>16;
      int bll = cll>>16, blr = clr>>16;

      int rt = rul+(((rur-rul)*bxf)>>16);
      int rb = rll+(((rlr-rll)*bxf)>>16);
      r2 = rt+(((rb-rt)*byf)>>16);

      int gt = gul+(((gur-gul)*bxf)>>16);
      int gb = gll+(((glr-gll)*bxf)>>16);
      g2 = gt+(((gb-gt)*byf)>>16);

      int bt = bul+(((bur-bul)*bxf)>>16);
      int bb = bll+(((blr-bll)*bxf)>>16);
      b2 = bt+(((bb-bt)*byf)>>16);
#else
      int c2 = ((int *) b->line[(by)>>16])[(bx)>>16];
      r2 = (c2&255);
      g2 = (c2>>8)&255;
      b2 = (c2>>16);
#endif
      int dr = (c1&255)-r2;
      int dg = ((c1>>8)&255)-g2;
      int db = (c1>>16)-b2;
      ans += DELTA_TERM; //dr*dr+dg*dg+db*db;
      if (DO_BRANCH && ans > maxval) { return ans; }
      bx += bpos.dxdu;
      by += bpos.dydu;
    }
    adata += TPATCH_W;
    bx_row += bpos.dxdv;
    by_row += bpos.dydv;
  }
  return ans;
}

template<int PATCH_W>
void sim_attempt_n(int &err, int &xbest, int &ybest, int &sbest, int &tbest, int *adata, BITMAP *b, XFORM bpos, int bx, int by, int bs, int bt, Params *p) {
  //int h = PATCH_W/2;
  if ((bx != xbest || by != ybest || bs != sbest || bt != tbest) &&
      (unsigned) (bx) < (unsigned) (b->w-PATCH_W+1) &&
      (unsigned) (by) < (unsigned) (b->h-PATCH_W+1)) 
	{
    //XFORM bpos = get_xform(p, bx, by, bs, bt);
    int current = sim_fast_patch_dist<PATCH_W, 1>(adata, b, bpos, err);
    if (current < err) {
      err = current;
      xbest = bx;
      ybest = by;
      sbest = bs;
      tbest = bt;
    }
  }
}

template<int TPATCH_W, int DO_BRANCH>
int sim_patch_dist_ab(Params *p, BITMAP *a, int ax, int ay, BITMAP *b, int bx, int by, int bs, int bt, int maxval) {
  int adata[TPATCH_W*TPATCH_W];
  int *ptr = adata;
  for (int dy = 0; dy < TPATCH_W; dy++) {
    int *row = ((int *) a->line[ay+dy])+ax;
    for (int dx = 0; dx < TPATCH_W; dx++) {
      *ptr++ = *row++;
    }
  }
  XFORM bpos = get_xform(p, bx, by, bs, bt);
  return sim_fast_patch_dist<TPATCH_W, DO_BRANCH>(adata, b, bpos, maxval);
}

#endif
