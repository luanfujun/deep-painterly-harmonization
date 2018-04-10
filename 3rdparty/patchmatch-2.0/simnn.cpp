
#include "simnn.h"
#include "simpatch.h"
#include <math.h>

void getpixel_bilin(BITMAP *bimg, int bx, int by, int &r, int &g, int &b) {
  int bxi = bx>>16, byi = by>>16;
  int bxf = bx&((1<<16)-1), byf = by&((1<<16)-1);
  if (bxi < 0) { bxi = 0; bxf = 0; }
  else if (bxi >= bimg->w-1) { bxi = bimg->w-2; bxf = 1; }
  if (byi < 0) { byi = 0; byf = 0; }
  else if (byi >= bimg->h-1) { byi = bimg->h-2; byf = 1; }

  int *row1 = ((int *) bimg->line[byi])+bxi;
  int *row2 = ((int *) bimg->line[byi+1])+bxi;
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
  r = rt+(((rb-rt)*byf)>>16);

  int gt = gul+(((gur-gul)*bxf)>>16);
  int gb = gll+(((glr-gll)*bxf)>>16);
  g = gt+(((gb-gt)*byf)>>16);

  int bt = bul+(((bur-bul)*bxf)>>16);
  int bb = bll+(((blr-bll)*bxf)>>16);
  b = bt+(((bb-bt)*byf)>>16);
}

int xform_cos_table[NUM_ANGLES];
int xform_sin_table[NUM_ANGLES];
int xform_scale_table[NUM_SCALES];

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif

int xform_tables_init = 0;
void init_xform_tables(double SCALE_MIN, double SCALE_MAX, int force_init) {
  if (xform_tables_init && !force_init) { return; }
  xform_tables_init = 1;
  for (int i = 0; i < NUM_ANGLES; i++) {
    double theta = i*(2.0*M_PI/NUM_ANGLES);
    xform_cos_table[i] = int(cos(theta)*65536.0);
    xform_sin_table[i] = int(sin(theta)*65536.0);
  }
  double a = log(SCALE_MIN), b = log(SCALE_MAX);
  for (int i = 0; i < NUM_SCALES; i++) {
    double scale = exp(a+(b-a)*i*1.0/NUM_SCALES);
    xform_scale_table[i] = int(scale*65536.0);
  }
}

XFORM get_xform(Params *p, int x, int y, int scale, int theta) {
  int c = xform_cos_table[theta&(NUM_ANGLES-1)];
  int s = xform_sin_table[theta&(NUM_ANGLES-1)];
  if ((unsigned) scale >= (unsigned) NUM_SCALES) { fprintf(stderr, "scale out of range: %d (%d)\n", scale, NUM_SCALES); exit(1); }
  int scalef = xform_scale_table[scale];
  c = fixmul(c, scalef);
  s = fixmul(s, scalef);
  XFORM ans;
  ans.dxdu = c;
  ans.dydu = -s;
  ans.dxdv = s;
  ans.dydv = c;
  int h = p->patch_w/2;
  int xc = x+h, yc = y+h;
  ans.x0 = (xc<<16)-ans.dxdu*h-ans.dxdv*h;
  ans.y0 = (yc<<16)-ans.dydu*h-ans.dydv*h;
  return ans;
}

void check_offset(Params *p, BITMAP *b, int x, int y, int xp, int yp) {
  //int h = p->patch_w/2;
  //if ((unsigned) (xp-h) >= (unsigned) BEW ||
  //    (unsigned) (yp-h) >= (unsigned) BEH) { fprintf(stderr, "offset (%d, %d) => (%d, %d) out of range b: %dx%d\n", x, y, xp, yp, b->w, b->h); exit(1); }
  if ((unsigned) xp >= (unsigned) BEW ||
      (unsigned) yp >= (unsigned) BEH) { fprintf(stderr, "offset (%d, %d) => (%d, %d) out of range b: %dx%d\n", x, y, xp, yp, b->w, b->h); exit(1); }
}

BITMAP *sim_init_nn(Params *p, BITMAP *a, BITMAP *b, BITMAP *&ann_sim) {
  init_xform_tables();
  BITMAP *ann = init_nn(p, a, b);
  ann_sim = create_bitmap(ann->w, ann->h);
  //int h = p->patch_w/2;
  for (int y = 0; y < AEH; y++) {
    //int *ann_row = (int *) ann->line[y];
    int *row = (int *) ann_sim->line[y];
    for (int x = 0; x < AEW; x++) {
      //int xp, yp;
      //getnn(ann, x, y, xp, yp);
      //xp += h;
      //yp += h;
      //ann_row[x] = XY_TO_INT(xp, yp);
      row[x] = XY_TO_INT(rand()%NUM_SCALES, rand()%NUM_ANGLES);
      //check_offset(p, b, x, y, xp, yp);
    }
  }
  return ann;
}

template<int PATCH_W>
BITMAP *sim_init_dist_n(Params *p, BITMAP *a, BITMAP *b, BITMAP *ann, BITMAP *ann_sim) {
  init_xform_tables();
  BITMAP *ans = create_bitmap(a->w, a->h);
  clear_to_color(ans, INT_MAX);
  for (int y = 0; y < AEH; y++) {
    int *row = (int *) ans->line[y];
    for (int x = 0; x < AEW; x++) {
      int bx, by, bs, bt;
      getnn(ann, x, y, bx, by);
      getnn(ann_sim, x, y, bs, bt);
      row[x] = sim_patch_dist_ab<PATCH_W, 0>(p, a, x, y, b, bx, by, bs, bt, INT_MAX);
    }
  }
  return ans;
}

BITMAP *sim_init_dist(Params *p, BITMAP *a, BITMAP *b, BITMAP *ann, BITMAP *ann_sim) {
  if      (p->patch_w == 1 ) { return sim_init_dist_n<1>(p, a, b, ann, ann_sim); }
  else if (p->patch_w == 2 ) { return sim_init_dist_n<2>(p, a, b, ann, ann_sim); }
  else if (p->patch_w == 3 ) { return sim_init_dist_n<3>(p, a, b, ann, ann_sim); }
  else if (p->patch_w == 4 ) { return sim_init_dist_n<4>(p, a, b, ann, ann_sim); }
  else if (p->patch_w == 5 ) { return sim_init_dist_n<5>(p, a, b, ann, ann_sim); }
  else if (p->patch_w == 6 ) { return sim_init_dist_n<6>(p, a, b, ann, ann_sim); }
  else if (p->patch_w == 7 ) { return sim_init_dist_n<7>(p, a, b, ann, ann_sim); }
  else if (p->patch_w == 8 ) { return sim_init_dist_n<8>(p, a, b, ann, ann_sim); }
  else if (p->patch_w == 9 ) { return sim_init_dist_n<9>(p, a, b, ann, ann_sim); }
  else if (p->patch_w == 10) { return sim_init_dist_n<10>(p, a, b, ann, ann_sim); }
  else if (p->patch_w == 11) { return sim_init_dist_n<11>(p, a, b, ann, ann_sim); }
  else if (p->patch_w == 12) { return sim_init_dist_n<12>(p, a, b, ann, ann_sim); }
  else if (p->patch_w == 13) { return sim_init_dist_n<13>(p, a, b, ann, ann_sim); }
  else if (p->patch_w == 14) { return sim_init_dist_n<14>(p, a, b, ann, ann_sim); }
  else if (p->patch_w == 15) { return sim_init_dist_n<15>(p, a, b, ann, ann_sim); }
  else if (p->patch_w == 16) { return sim_init_dist_n<16>(p, a, b, ann, ann_sim); }
  else if (p->patch_w == 17) { return sim_init_dist_n<17>(p, a, b, ann, ann_sim); }
  else if (p->patch_w == 18) { return sim_init_dist_n<18>(p, a, b, ann, ann_sim); }
  else if (p->patch_w == 19) { return sim_init_dist_n<19>(p, a, b, ann, ann_sim); }
  else if (p->patch_w == 20) { return sim_init_dist_n<20>(p, a, b, ann, ann_sim); }
  else if (p->patch_w == 21) { return sim_init_dist_n<21>(p, a, b, ann, ann_sim); }
  else if (p->patch_w == 22) { return sim_init_dist_n<22>(p, a, b, ann, ann_sim); }
  else if (p->patch_w == 23) { return sim_init_dist_n<23>(p, a, b, ann, ann_sim); }
  else if (p->patch_w == 24) { return sim_init_dist_n<24>(p, a, b, ann, ann_sim); }
  else if (p->patch_w == 25) { return sim_init_dist_n<25>(p, a, b, ann, ann_sim); }
  else if (p->patch_w == 26) { return sim_init_dist_n<26>(p, a, b, ann, ann_sim); }
  else if (p->patch_w == 27) { return sim_init_dist_n<27>(p, a, b, ann, ann_sim); }
  else if (p->patch_w == 28) { return sim_init_dist_n<28>(p, a, b, ann, ann_sim); }
  else if (p->patch_w == 29) { return sim_init_dist_n<29>(p, a, b, ann, ann_sim); }
  else if (p->patch_w == 30) { return sim_init_dist_n<30>(p, a, b, ann, ann_sim); }
  else if (p->patch_w == 31) { return sim_init_dist_n<31>(p, a, b, ann, ann_sim); }
  else if (p->patch_w == 32) { return sim_init_dist_n<32>(p, a, b, ann, ann_sim); }
  else { fprintf(stderr, "Patch width unsupported: %d\n", p->patch_w); exit(1); }
}

template<int PATCH_W>
void sim_nn_n(Params *p, BITMAP *a, BITMAP *b,
            BITMAP *ann, BITMAP *ann_sim, BITMAP *annd,
            RegionMasks *amask=NULL, BITMAP *bmask=NULL,
            int level=0, int em_iter=0, RecomposeParams *rp=NULL, int offset_iter=0, int update_type=0, int cache_b=0,
            RegionMasks *region_masks=NULL, int tiles=-1) {
  init_xform_tables();
  if (tiles < 0) { tiles = p->cores; }
  printf("in sim_nn_n, tiles=%d, rs_max=%d\n", tiles, p->rs_max);
  Box box = get_abox(p, a, amask);
  int nn_iter = 0;
  for (; nn_iter < p->nn_iters; nn_iter++) {
    unsigned int iter_seed = rand();

    #pragma omp parallel num_threads(tiles)
    {
#if SYNC_WRITEBACK
      int *ann_writeback = new int[a->w];
      int *annd_writeback = new int[a->w];
      int *ann_sim_writeback = new int[a->w];
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

      int adata[PATCH_W*PATCH_W];
      for (int y = ystart; y != yfinal; y += ychange) {
        int *annd_row = (int *) annd->line[y];
        for (int x = xstart; x != xfinal; x += xchange) {
          for (int dy0 = 0; dy0 < PATCH_W; dy0++) {
            int *drow = ((int *) a->line[y+dy0])+x;
            int *adata_row = adata+(dy0*PATCH_W);
            for (int dx0 = 0; dx0 < PATCH_W; dx0++) {
              adata_row[dx0] = drow[dx0];
            }
          }
          
          int xbest, ybest, sbest, tbest;
          getnn(ann, x, y, xbest, ybest);
          getnn(ann_sim, x, y, sbest, tbest);
          check_offset(p, b, x, y, xbest, ybest);

          int err = annd_row[x];
          if (err == 0) {
#if SYNC_WRITEBACK
            if (y+ychange == yfinal) {
              ann_writeback[x] = XY_TO_INT(xbest, ybest);
              ann_sim_writeback[x] = XY_TO_INT(sbest, tbest);
              annd_writeback[x] = err;
            }
#endif
            continue;
          }

          /* Propagate */
          if (p->do_propagate) {
            /* Propagate x */
            if ((unsigned) (x+dx) < (unsigned) (ann->w-PATCH_W)) {
              int xpp, ypp, spp, tpp;
              getnn(ann, x+dx, y, xpp, ypp);
              getnn(ann_sim, x+dx, y, spp, tpp);
              //xpp -= dx;
              XFORM bpos = get_xform(p, xpp, ypp, spp, tpp);
              xpp -= (bpos.dxdu*dx)>>16;
              ypp -= (bpos.dydu*dx)>>16;
              bpos = get_xform(p, xpp, ypp, spp, tpp);

              sim_attempt_n<PATCH_W>(err, xbest, ybest, sbest, tbest, adata, b, bpos, xpp, ypp, spp, tpp, p);
              check_offset(p, b, x, y, xbest, ybest);
            }

            /* Propagate y */
            if ((unsigned) (y+dy) < (unsigned) (ann->h-PATCH_W)) {
              int xpp, ypp, spp, tpp;
              getnn(ann, x, y+dy, xpp, ypp);
              getnn(ann_sim, x, y+dy, spp, tpp);
              //xpp -= dx;
              XFORM bpos = get_xform(p, xpp, ypp, spp, tpp);
              xpp -= (bpos.dxdv*dy)>>16;
              ypp -= (bpos.dydv*dy)>>16;
              bpos = get_xform(p, xpp, ypp, spp, tpp);

              sim_attempt_n<PATCH_W>(err, xbest, ybest, sbest, tbest, adata, b, bpos, xpp, ypp, spp, tpp, p);
              check_offset(p, b, x, y, xbest, ybest);
            }
          }

          /* Random search */
          unsigned int seed = (x | (y<<11)) ^ iter_seed;
          seed = RANDI(seed);
          int rs_iters = 1-(seed*(1.0/(RAND_MAX-1))) < rs_fpart ? rs_ipart + 1: rs_ipart;
  //        int rs_iters = 1-random() < rs_fpart ? rs_ipart + 1: rs_ipart;

          int rs_max_curr = rs_max;

          int h = p->patch_w/2;
          int ymin_clamp = h, xmin_clamp = h;
          int ymax_clamp = BEH+h, xmax_clamp = BEW+h;

          for (int mag = rs_max_curr; mag >= p->rs_min; mag = int(mag*p->rs_ratio)) {
            int smag = NUM_SCALES*mag/rs_max_curr;
            int tmag = (NUM_SCALES == NUM_ANGLES) ? smag: (NUM_ANGLES*mag/rs_max_curr);  // FIXME: This should be divided by 2
            for (int rs_iter = 0; rs_iter < rs_iters; rs_iter++) {
              int xmin = MAX(xbest-mag,0), xmax = MIN(xbest+mag+1,bew);
              int ymin = MAX(ybest-mag,0), ymax = MIN(ybest+mag+1,beh);
              /*if (xmin < xmin_clamp) { xmin = xmin_clamp; }
              if (ymin < ymin_clamp) { ymin = ymin_clamp; }
              if (xmax > xmax_clamp) { xmax = xmax_clamp; }
              if (ymax > ymax_clamp) { ymax = ymax_clamp; }*/

              int smin = sbest-smag, smax = sbest+smag+1;
              int tmin = tbest-tmag, tmax = tbest+tmag+1;
              if (smin < 0) { smin = 0; }
              if (smax > NUM_SCALES) { smax = NUM_SCALES; }
              //fprintf(stderr, "RS: xbest: %d, ybest: %d, err: %d, mag: %d, bew: %d, beh: %d, smag: %d, tmag: %d, xmin: %d, xmax: %d, ymin: %d, ymax: %d, smin: %d, smax: %d, tmin: %d, tmax: %d\n", xbest, ybest, err, mag, bew, beh, smag, tmag, xmin, xmax, ymin, ymax, smin, smax, tmin, tmax); fflush(stderr);

              seed = RANDI(seed);
              int xpp = xmin+seed%(xmax-xmin);
              seed = RANDI(seed);
              int ypp = ymin+seed%(ymax-ymin);
              seed = RANDI(seed);
              int spp = smin+seed%(smax-smin);
              seed = RANDI(seed);
              int tpp = tmin+seed%(tmax-tmin);

              XFORM bpos = get_xform(p, xpp, ypp, spp, tpp);
              sim_attempt_n<PATCH_W>(err, xbest, ybest, sbest, tbest, adata, b, bpos, xpp, ypp, spp, tpp, p);
              check_offset(p, b, x, y, xbest, ybest);
            }
          }

#if SYNC_WRITEBACK
          if (y+ychange != yfinal) {     
#endif
          ((int *) ann->line[y])[x] = XY_TO_INT(xbest, ybest);
          ((int *) ann_sim->line[y])[x] = XY_TO_INT(sbest, tbest);
          ((int *) annd->line[y])[x] = err;
#if SYNC_WRITEBACK
          } else {
            ann_writeback[x] = XY_TO_INT(xbest, ybest);
            annd_writeback[x] = err;
            ann_sim_writeback[x] = XY_TO_INT(sbest, tbest);
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
        int *ann_sim_line = (int *) ann_sim->line[ywrite];
        for (int x = xmin; x < xmax; x++) {
          ann_line[x] = ann_writeback[x];
          annd_line[x] = annd_writeback[x];
          ann_sim_line[x] = ann_sim_writeback[x];
        }
      }
      delete[] ann_writeback;
      delete[] annd_writeback;
      delete[] ann_sim_writeback;
#endif
    } // parallel
    fprintf(stderr, "done with %d iters\n", nn_iter);
  } // nn_iter
  printf("done sim_nn_n, %d iters, rs_max=%d\n", nn_iter, p->rs_max);
}

void sim_nn(Params *p, BITMAP *a, BITMAP *b,
            BITMAP *ann, BITMAP *ann_sim, BITMAP *annd,
            RegionMasks *amask, BITMAP *bmask,
            int level, int em_iter, RecomposeParams *rp, int offset_iter, int update_type, int cache_b,
            RegionMasks *region_masks, int tiles) {
  if      (p->patch_w == 1) { sim_nn_n<1>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles); }
  else if (p->patch_w == 2) { sim_nn_n<2>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles); }
  else if (p->patch_w == 3) { sim_nn_n<3>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles); }
  else if (p->patch_w == 4) { sim_nn_n<4>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles); }
  else if (p->patch_w == 5) { sim_nn_n<5>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles); }
  else if (p->patch_w == 6) { sim_nn_n<6>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles); }
  else if (p->patch_w == 7) { sim_nn_n<7>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles); }
  else if (p->patch_w == 8) { sim_nn_n<8>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles); }
  else if (p->patch_w == 9) { sim_nn_n<9>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles); }
  else if (p->patch_w == 10) { sim_nn_n<10>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles); }
  else if (p->patch_w == 11) { sim_nn_n<11>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles); }
  else if (p->patch_w == 12) { sim_nn_n<12>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles); }
  else if (p->patch_w == 13) { sim_nn_n<13>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles); }
  else if (p->patch_w == 14) { sim_nn_n<14>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles); }
  else if (p->patch_w == 15) { sim_nn_n<15>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles); }
  else if (p->patch_w == 16) { sim_nn_n<16>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles); }
  else if (p->patch_w == 17) { sim_nn_n<17>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles); }
  else if (p->patch_w == 18) { sim_nn_n<18>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles); }
  else if (p->patch_w == 19) { sim_nn_n<19>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles); }
  else if (p->patch_w == 20) { sim_nn_n<20>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles); }
  else if (p->patch_w == 21) { sim_nn_n<21>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles); }
  else if (p->patch_w == 22) { sim_nn_n<22>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles); }
  else if (p->patch_w == 23) { sim_nn_n<23>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles); }
  else if (p->patch_w == 24) { sim_nn_n<24>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles); }
  else if (p->patch_w == 25) { sim_nn_n<25>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles); }
  else if (p->patch_w == 26) { sim_nn_n<26>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles); }
  else if (p->patch_w == 27) { sim_nn_n<27>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles); }
  else if (p->patch_w == 28) { sim_nn_n<28>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles); }
  else if (p->patch_w == 29) { sim_nn_n<29>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles); }
  else if (p->patch_w == 30) { sim_nn_n<30>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles); }
  else if (p->patch_w == 31) { sim_nn_n<31>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles); }
  else if (p->patch_w == 32) { sim_nn_n<32>(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles); }
  else { fprintf(stderr, "Patch size unsupported: %d\n", p->patch_w); exit(1); }
}

template<int PATCH_W, class ACCUM>
BITMAP *sim_vote_n(Params *p, BITMAP *b,
                 BITMAP *ann, BITMAP *ann_sim, BITMAP *bnn, BITMAP *bnn_sim,
                 BITMAP *bmask, BITMAP *bweight,
                 double coherence_weight, double complete_weight,
                 RegionMasks *amask, BITMAP *aweight, BITMAP *ainit, RegionMasks *region_masks, BITMAP *aconstraint, int mask_self_only) 
{
  init_xform_tables();
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
}

BITMAP *sim_vote(Params *p, BITMAP *b,
                 BITMAP *ann, BITMAP *ann_sim, BITMAP *bnn, BITMAP *bnn_sim,
                 BITMAP *bmask, BITMAP *bweight,
                 double coherence_weight, double complete_weight,
                 RegionMasks *amask, BITMAP *aweight, BITMAP *ainit, RegionMasks *region_masks, BITMAP *aconstraint, int mask_self_only) 
{
  if      (p->patch_w == 1) { return sim_vote_n<1,int>(p, b, ann, ann_sim, bnn, bnn_sim, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
  else if (p->patch_w == 2) { return sim_vote_n<2,int>(p, b, ann, ann_sim, bnn, bnn_sim, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
  else if (p->patch_w == 3) { return sim_vote_n<3,int>(p, b, ann, ann_sim, bnn, bnn_sim, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
  else if (p->patch_w == 4) { return sim_vote_n<4,int>(p, b, ann, ann_sim, bnn, bnn_sim, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
  else if (p->patch_w == 5) { return sim_vote_n<5,int>(p, b, ann, ann_sim, bnn, bnn_sim, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
  else if (p->patch_w == 6) { return sim_vote_n<6,int>(p, b, ann, ann_sim, bnn, bnn_sim, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
  else if (p->patch_w == 7) { return sim_vote_n<7,int>(p, b, ann, ann_sim, bnn, bnn_sim, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
  else if (p->patch_w == 8) { return sim_vote_n<8,int>(p, b, ann, ann_sim, bnn, bnn_sim, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
  else if (p->patch_w == 9) { return sim_vote_n<9,int>(p, b, ann, ann_sim, bnn, bnn_sim, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
  else if (p->patch_w == 10) { return sim_vote_n<10,int>(p, b, ann, ann_sim, bnn, bnn_sim, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
  else if (p->patch_w == 11) { return sim_vote_n<11,int>(p, b, ann, ann_sim, bnn, bnn_sim, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
  else if (p->patch_w == 12) { return sim_vote_n<12,int>(p, b, ann, ann_sim, bnn, bnn_sim, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
  else if (p->patch_w == 13) { return sim_vote_n<13,int>(p, b, ann, ann_sim, bnn, bnn_sim, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
  else if (p->patch_w == 14) { return sim_vote_n<14,int>(p, b, ann, ann_sim, bnn, bnn_sim, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
  else if (p->patch_w == 15) { return sim_vote_n<15,int>(p, b, ann, ann_sim, bnn, bnn_sim, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
  else if (p->patch_w == 16) { return sim_vote_n<16,int>(p, b, ann, ann_sim, bnn, bnn_sim, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
  else if (p->patch_w == 17) { return sim_vote_n<17,int>(p, b, ann, ann_sim, bnn, bnn_sim, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
  else if (p->patch_w == 18) { return sim_vote_n<18,int>(p, b, ann, ann_sim, bnn, bnn_sim, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
  else if (p->patch_w == 19) { return sim_vote_n<19,int>(p, b, ann, ann_sim, bnn, bnn_sim, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
  else if (p->patch_w == 20) { return sim_vote_n<20,int>(p, b, ann, ann_sim, bnn, bnn_sim, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
  else if (p->patch_w == 21) { return sim_vote_n<21,int>(p, b, ann, ann_sim, bnn, bnn_sim, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
  else if (p->patch_w == 22) { return sim_vote_n<22,int>(p, b, ann, ann_sim, bnn, bnn_sim, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
  else if (p->patch_w == 23) { return sim_vote_n<23,int>(p, b, ann, ann_sim, bnn, bnn_sim, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
  else if (p->patch_w == 24) { return sim_vote_n<24,int>(p, b, ann, ann_sim, bnn, bnn_sim, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
  else if (p->patch_w == 25) { return sim_vote_n<25,int>(p, b, ann, ann_sim, bnn, bnn_sim, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
  else if (p->patch_w == 26) { return sim_vote_n<26,int>(p, b, ann, ann_sim, bnn, bnn_sim, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
  else if (p->patch_w == 27) { return sim_vote_n<27,int>(p, b, ann, ann_sim, bnn, bnn_sim, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
  else if (p->patch_w == 28) { return sim_vote_n<28,int>(p, b, ann, ann_sim, bnn, bnn_sim, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
  else if (p->patch_w == 29) { return sim_vote_n<29,int>(p, b, ann, ann_sim, bnn, bnn_sim, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
  else if (p->patch_w == 30) { return sim_vote_n<30,int>(p, b, ann, ann_sim, bnn, bnn_sim, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
  else if (p->patch_w == 31) { return sim_vote_n<31,int>(p, b, ann, ann_sim, bnn, bnn_sim, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
  else if (p->patch_w == 32) { return sim_vote_n<32,int>(p, b, ann, ann_sim, bnn, bnn_sim, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only); }
  else { fprintf(stderr, "Patch size unsupported: %d\n", p->patch_w); exit(1); }
}
