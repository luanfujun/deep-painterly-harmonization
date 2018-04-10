
#include "vecnn.h"
#include "simnn.h"

template<> int get_maxval() { return INT_MAX; }
template<> float get_maxval() { return FLT_MAX; }
template<> double get_maxval() { return DBL_MAX; }
template<> long long get_maxval() { return LLONG_MAX; }

VECBITMAP<unsigned char> *bitmap_to_patches(Params *p, BITMAP *a) {
  //int n = -1, d = -1;
  //unsigned char *data = im2patches_ub(a, p->patch_w, n, d);
  //VECBITMAP<unsigned char> *ans = new VECBITMAP<unsigned char>();
  //ans->w = a->w;
  //ans->h = a->h;
  //ans->n = d;
  //ans->data = data;
  if (p->vec_len != p->patch_w*p->patch_w*3) { fprintf(stderr, "vec_len (%d) != 3*patch_w**2 (%d)\n", p->vec_len, p->patch_w*p->patch_w*3); exit(1); }
  VECBITMAP<unsigned char> *ans = new VECBITMAP<unsigned char>(a->w, a->h, p->vec_len);
  unsigned char *ptr = ans->data;
  for (int y = 0; y < a->h; y++) {
    for (int x = 0; x < a->w; x++) {
      for (int dy = 0; dy < p->patch_w; dy++) {
        for (int dx = 0; dx < p->patch_w; dx++) {
          int xp = x+dx, yp = y+dy;
          if (xp >= a->w) { xp = a->w - 1; }
          if (yp >= a->h) { yp = a->h - 1; }
          int c = _getpixel32(a, xp, yp);
          *ptr++ = getr32(c);
          *ptr++ = getg32(c);
          *ptr++ = getb32(c);
        }
      }
    }
  }
  int npatches = ptr - ans->data;
  int n = a->w * a->h * p->vec_len;
  if (n != npatches) { fprintf(stderr, "n != npatches (%d != %d)\n", n, npatches); exit(1); }

  return ans;
}

BITMAP *vecbitmap_to_bitmap(VECBITMAP<int> *a) {
  if (a->n != 1) { fprintf(stderr, "vecbitmap n != 1 (%d)\n", a->n); exit(1); }
	BITMAP *ans = create_bitmap(a->w, a->h);
  for (int y = 0; y < a->h; y++) {
    int *a_row = a->line_n1(y);
    int *ans_row = (int *) ans->line[y];
    for (int x = 0; x < a->w; x++) {
      ans_row[x] = a_row[x];
		}
  }
  return ans;
}

VECBITMAP<int> *bitmap_to_vecbitmap(BITMAP *a) {
  VECBITMAP<int> *ans = new VECBITMAP<int>(a->w, a->h, 1);
  for (int y = 0; y < a->h; y++) {
    int *a_row = (int *) a->line[y];
    int *ans_row = ans->line_n1(y);
    for (int x = 0; x < a->w; x++) {
      ans_row[x] = a_row[x];
    }
  }
  return ans;
}

BITMAP *vecwrap_init_nn(int vec_mode, Params *p, BITMAP *a, BITMAP *b, BITMAP *bmask, RegionMasks *region_masks, RegionMasks *amask, BITMAP **ann_sim) {
  if (vec_mode == VEC_MODE_PATCH) {
    return init_nn(p, a, b, bmask, region_masks, amask);
  } 
	else if (vec_mode == VEC_MODE_DESC) {
    VECBITMAP<unsigned char> *av = bitmap_to_patches(p, a);
    VECBITMAP<unsigned char> *bv = bitmap_to_patches(p, b);
    BITMAP *ans = vec_init_nn<unsigned char>(p, av, bv, bmask, region_masks, amask);
    delete av;
    delete bv;
    return ans;
  } 
	else if (vec_mode == VEC_MODE_SIM) {
    if (!ann_sim) { fprintf(stderr, "vecwrap_init_nn: expected argument ann_sim\n"); exit(1); }
    return sim_init_nn(p, a, b, *ann_sim);
  } 
	else {
    fprintf(stderr, "vecwrap_init_nn: unknown mode %d\n", vec_mode); exit(1);
  }
}

BITMAP *vecwrap_init_dist(int vec_mode, Params *p, BITMAP *a, BITMAP *b, BITMAP *ann, BITMAP *bmask, RegionMasks *region_masks, RegionMasks *amask, BITMAP *ann_sim) {
  if (vec_mode == VEC_MODE_PATCH) {
    return init_dist(p, a, b, ann, bmask, region_masks, amask);
  } 
	else if (vec_mode == VEC_MODE_DESC) {
    VECBITMAP<unsigned char> *av = bitmap_to_patches(p, a);
    VECBITMAP<unsigned char> *bv = bitmap_to_patches(p, b);

    VECBITMAP<int> *ans = vec_init_dist<unsigned char, int>(p, av, bv, ann, bmask, region_masks, amask);

    delete av;
    delete bv;
    BITMAP *ansd = vecbitmap_to_bitmap(ans);
    delete ans;
    return ansd;
  } 
	else if (vec_mode == VEC_MODE_SIM) {
    if (!ann_sim) { fprintf(stderr, "vecwrap_init_dist: expected argument ann_sim\n"); exit(1); }
    return sim_init_dist(p, a, b, ann, ann_sim);
  } 
	else {
    fprintf(stderr, "vecwrap_init_dist: unknown mode %d\n", vec_mode); exit(1);
  }
}

void vecwrap_nn(int vec_mode, Params *p, BITMAP *a, BITMAP *b,
        BITMAP *ann, BITMAP *annd,
        RegionMasks *amask, BITMAP *bmask,
        int level, int em_iter, RecomposeParams *rp, int offset_iter, int update_type, int cache_b,
        RegionMasks *region_masks, int tiles, BITMAP *ann_sim) 
{
  if (vec_mode == VEC_MODE_PATCH) {
    return nn(p, a, b, ann, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
  } 
  else if (vec_mode == VEC_MODE_DESC) {
    VECBITMAP<unsigned char> *av = bitmap_to_patches(p, a);
    VECBITMAP<unsigned char> *bv = bitmap_to_patches(p, b);
    VECBITMAP<int> *anndv = bitmap_to_vecbitmap(annd);

    vec_nn<unsigned char, int>(p, av, bv, ann, anndv, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);

    BITMAP *anndp = vecbitmap_to_bitmap(anndv);
    if (anndp->w != annd->w || anndp->h != annd->h) { fprintf(stderr, "Sizes differ in vecwrap_nn: %dx%d, %dx%d\n", anndp->w, anndp->h, annd->w, annd->h); exit(1); }
    blit(anndp, annd, 0, 0, 0, 0, anndp->w, anndp->h);

    delete av;
    delete bv;
    delete anndv;
    destroy_bitmap(anndp);
  } 
  else if (vec_mode == VEC_MODE_SIM) {
    if (!ann_sim) { fprintf(stderr, "vecwrap_nn: expected argument ann_sim\n"); exit(1); }
    return sim_nn(p, a, b, ann, ann_sim, annd, amask, bmask, level, em_iter, rp, offset_iter, update_type, cache_b, region_masks, tiles);
  } 
  else {
    fprintf(stderr, "vecwrap_nn: unknown mode %d\n", vec_mode); exit(1);
  }
}

BITMAP *vecwrap_vote(int vec_mode, Params *p, BITMAP *b,
         BITMAP *ann, BITMAP *ann_sim, BITMAP *bnn,
         BITMAP *bmask, BITMAP *bweight,
         double coherence_weight, double complete_weight,
         RegionMasks *amask, BITMAP *aweight, BITMAP *ainit, RegionMasks *region_masks, BITMAP *aconstraint, int mask_self_only) 
{
  if (vec_mode == VEC_MODE_PATCH || vec_mode == VEC_MODE_DESC) {
    return vote(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only);
  }
	else if (vec_mode == VEC_MODE_SIM) {
    if (!ann_sim) { fprintf(stderr, "vecwrap_vote: expected argument ann_sim\n"); exit(1); }
    return sim_vote(p, b, ann, ann_sim, bnn, NULL, bmask, bweight, coherence_weight, complete_weight, amask, aweight, ainit, region_masks, aconstraint, mask_self_only);
  } 
	else {
    fprintf(stderr, "vecwrap_vote: unknown mode %d\n", vec_mode); exit(1);
  }
}
