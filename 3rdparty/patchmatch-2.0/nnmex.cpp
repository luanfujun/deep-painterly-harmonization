
// MATLAB interface, for PatchMatch + Generalized PatchMatch.

#include "allegro_emu.h"
#include "mex.h"
#include "nn.h"
//#include "matrix.h"
#include "simnn.h"
#include "mexutil.h"
#include "knn.h"

static char AdobePatentID_P876E1[] = "AdobePatentID=\"P876E1\""; // AdobePatentID="P876E1"
static char AdobePatentID_P962[] = "AdobePatentID=\"P962\""; // AdobePatentID="P962"

void init_params(Params *p);

#define MODE_IMAGE  0
#define MODE_VECB   1
#define MODE_VECF   2

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif

int is_float(const mxArray *A) {
  return mxIsDouble(A) || mxIsSingle(A);
}

extern int xform_scale_table[NUM_SCALES];

/*void logf(const char *s) {
  FILE *f = fopen("log.txt", "a");
  fprintf(f, s);
  fclose(f);
}*/

void mexFunction(int nout, mxArray *pout[], int nin, const mxArray *pin[]) {
  FILE *f = fopen("log.txt", "wt");
  fclose(f);

  if (nin < 2) { mexErrMsgTxt("nnmex called with < 2 input arguments"); }

  const mxArray *A = pin[0], *B = pin[1];
  const mxArray *ANN_PREV = NULL, *ANN_WINDOW = NULL, *AWINSIZE = NULL;
  int aw = -1, ah = -1, bw = -1, bh = -1;
  BITMAP *a = NULL, *b = NULL, *ann_prev = NULL, *ann_window = NULL, *awinsize = NULL;
  VECBITMAP<unsigned char> *ab = NULL, *bb = NULL;
  VECBITMAP<float> *af = NULL, *bf = NULL;

  if (mxGetNumberOfDimensions(A) != 3 || mxGetNumberOfDimensions(B) != 3) { mexErrMsgTxt("dims != 3"); }
  if (mxGetDimensions(A)[2] != mxGetDimensions(B)[2]) { mexErrMsgTxt("3rd dimension not same size"); }

  int mode = MODE_IMAGE;
  if (mxGetDimensions(A)[2] != 3) { // a discriptor rather than rgb
    if (mxIsUint8(A) && mxIsUint8(B)) { mode = MODE_VECB; } 
    else if (is_float(A) && is_float(B)) { mode = MODE_VECF; } 
    else { mexErrMsgTxt("input not uint8, single, or double"); }
  }

  Params *p = new Params();
  RecomposeParams *rp = new RecomposeParams();
  BITMAP *borig = NULL;
  if (mode == MODE_IMAGE) {
    a = convert_bitmap(A);
    b = convert_bitmap(B);
	borig = b;
    aw = a->w; ah = a->h;
    bw = b->w; bh = b->h;
  } 
  else if (mode == MODE_VECB) {
    ab = convert_vecbitmap<unsigned char>(A);
    bb = convert_vecbitmap<unsigned char>(B);
    if (ab->n != bb->n) { mexErrMsgTxt("3rd dimension differs"); }
    aw = ab->w; ah = ab->h;
    bw = bb->w; bh = bb->h;
    p->vec_len = ab->n;
  } 
  else if (mode == MODE_VECF) {
    af = convert_vecbitmap<float>(A);
    bf = convert_vecbitmap<float>(B);
    if (af->n != bf->n) { mexErrMsgTxt("3rd dimension differs"); }
    aw = af->w; ah = af->h;
    bw = bf->w; bh = bf->h;
    p->vec_len = af->n;
  }

  double *win_size = NULL;
  BITMAP *amask = NULL, *bmask = NULL;

  double scalemin = 0.5, scalemax = 2.0;  // The product of these must be one.
  /* parse parameters */
  int i = 2;
  int sim_mode = 0;
  int knn_chosen = -1;
  p->algo = ALGO_CPU;
  int enrich_mode = 0;
  if (nin > i && !mxIsEmpty(pin[i])) {
    if (mxStringEquals(pin[i], "cpu")) { p->algo = ALGO_CPU; }
    else if (mxStringEquals(pin[i], "gpucpu")) { p->algo = ALGO_GPUCPU; }
    else if (mxStringEquals(pin[i], "cputiled")) { p->algo = ALGO_CPUTILED; }
    else if (mxStringEquals(pin[i], "rotscale")) { sim_mode = 1; }
    else if (mxStringEquals(pin[i], "enrich")) { p->algo = ALGO_CPUTILED; enrich_mode = 1; }
    else { mexErrMsgTxt("Unknown algorithm"); }
  } i++;
  if (nin > i && !mxIsEmpty(pin[i])) { p->patch_w = int(mxGetScalar(pin[i])); } i++;
  if (nin > i && !mxIsEmpty(pin[i])) { p->nn_iters = int(mxGetScalar(pin[i])); } i++;
  if (nin > i && !mxIsEmpty(pin[i])) { p->rs_max = int(mxGetScalar(pin[i])); } i++;
  if (nin > i && !mxIsEmpty(pin[i])) { p->rs_min = int(mxGetScalar(pin[i])); } i++;
  if (nin > i && !mxIsEmpty(pin[i])) { p->rs_ratio = mxGetScalar(pin[i]); } i++;
  if (nin > i && !mxIsEmpty(pin[i])) { p->rs_iters = mxGetScalar(pin[i]); } i++;
  if (nin > i && !mxIsEmpty(pin[i])) { p->cores = int(mxGetScalar(pin[i])); } i++;
  if (nin > i && !mxIsEmpty(pin[i])) { bmask = convert_bitmap(pin[i]); } i++; // XC+
  if (nin > i && !mxIsEmpty(pin[i])) { 
    if (!mxIsDouble(pin[i])) { mexErrMsgTxt("\nwin_size should be of type double."); }
    win_size = (double*)mxGetData(pin[i]);
    if (mxGetNumberOfElements(pin[i])==1) { p->window_h = p->window_w = int(win_size[0]); }
    else if (mxGetNumberOfElements(pin[i])==2) { p->window_h = int(win_size[0]); p->window_w = int(win_size[1]); }
    else { mexErrMsgTxt("\nwin_size should be a scalar for square window or [h w] for a rectangular one."); }
  } i++;
  /* continue parsing parameters */
  // [ann_prev=NULL], [ann_window=NULL], [awinsize=NULL], 
  if (nin > i && !mxIsEmpty(pin[i])) { 
    ANN_PREV = pin[i];
    int clip_count = 0;
    ann_prev = convert_field(p, ANN_PREV, bw, bh, clip_count);       // Bug fixed by Connelly
  } i++;
  if (nin > i && !mxIsEmpty(pin[i])) { 
    ANN_WINDOW = pin[i];
    int clip_count = 0;
    ann_window = convert_field(p, ANN_WINDOW, bw, bh, clip_count);      
  } i++;
  if (nin > i && !mxIsEmpty(pin[i])) { 
    AWINSIZE = pin[i];
    awinsize = convert_winsize_field(p, AWINSIZE, aw, ah);  
    if (p->window_w==INT_MAX||p->window_h==INT_MAX) { p->window_w = -1; p->window_h = -1; }
  } i++;
  if (nin > i && !mxIsEmpty(pin[i])) { 
    knn_chosen = int(mxGetScalar(pin[i]));
    if (knn_chosen == 1) { knn_chosen = -1; }
    if (knn_chosen <= 0) { mexErrMsgTxt("\nknn is less than zero"); }
  } i++;
  if (nin > i && !mxIsEmpty(pin[i])) { 
    scalemax = mxGetScalar(pin[i]);
    if (scalemax <= 0) { mexErrMsgTxt("\nscalerange is less than zero"); }
    scalemin = 1.0/scalemax;
    if (scalemax < scalemin) {
      double temp = scalemax;
      scalemax = scalemin;
      scalemin = temp;
    }
  } i++;

  if (ann_window&&!awinsize&&!win_size) {
    mexErrMsgTxt("\nUsing ann_window - either awinsize or win_size should be defined.\n");
  }

  if (enrich_mode) {
	int nn_iters = p->nn_iters;
    p->enrich_iters = nn_iters/2;
	p->nn_iters = 2;
	if (A != B) { mexErrMsgTxt("\nOur implementation of enrichment requires that image A = image B.\n"); }
	if (mode == MODE_IMAGE) {
	  b = a;
	} else {
	  mexErrMsgTxt("\nEnrichment only implemented for 3 channel uint8 inputs.\n");
	}
  }

  init_params(p);
  if (sim_mode) {
    init_xform_tables(scalemin, scalemax, 1);
  }
  
  RegionMasks *amaskm = amask ? new RegionMasks(p, amask): NULL;

  BITMAP *ann = NULL; // NN field
  BITMAP *annd_final = NULL; // NN patch distance field
  BITMAP *ann_sim_final = NULL;

  VBMP *vann_sim = NULL;
  VBMP *vann = NULL;
  VBMP *vannd = NULL;

  if (mode == MODE_IMAGE) {
    // input as RGB image
    if (!a || !b) { mexErrMsgTxt("internal error: no a or b image"); }
    if (knn_chosen > 1) {
      p->knn = knn_chosen;
      if (sim_mode) { mexErrMsgTxt("rotating+scaling patches not implemented with knn (actually it is implemented it is not exposed by the wrapper)"); }
      PRINCIPAL_ANGLE *pa = NULL;
      vann_sim = NULL;
      vann = knn_init_nn(p, a, b, vann_sim, pa);
      vannd = knn_init_dist(p, a, b, vann, vann_sim);
      knn(p, a, b, vann, vann_sim, vannd, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, pa);
//      sort_knn(p, vann, vann_sim, vannd);
    } else if (sim_mode) {
      BITMAP *ann_sim = NULL;
      ann = sim_init_nn(p, a, b, ann_sim);
      BITMAP *annd = sim_init_dist(p, a, b, ann, ann_sim);
      sim_nn(p, a, b, ann, ann_sim, annd);
      if (ann_prev) { mexErrMsgTxt("when searching over rotations+scales, previous guess is not supported"); }
      annd_final = annd;
      ann_sim_final = ann_sim;
    } else {
      ann = init_nn(p, a, b, bmask, NULL, amaskm, 1, ann_window, awinsize);
      BITMAP *annd = init_dist(p, a, b, ann, bmask, NULL, amaskm);
      nn(p, a, b, ann, annd, amaskm, bmask, 0, 0, rp, 0, 0, 0, NULL, p->cores, ann_window, awinsize); 
      if (ann_prev) minnn(p, a, b, ann, annd, ann_prev, bmask, 0, 0, rp, NULL, amaskm, p->cores);  
      annd_final = annd;
    }
  } 
/*
  else if (mode == MODE_VECB) {
//    mexPrintf("mode vecb %dx%dx%d, %dx%dx%d\n", ab->w, ab->h, ab->n, bb->w, bb->h, bb->n);
//    mexPrintf("  %d %d %d %d\n", ab->get(0,0)[0], ab->get(1,0)[0], ab->get(0,1)[0], ab->get(0,0)[1]);
    if (!ab || !bb) { mexErrMsgTxt("internal error: no a or b image"); }
    ann = vec_init_nn<unsigned char>(p, ab, bb, bmask, NULL, amaskm);
    VECBITMAP<int> *annd = vec_init_dist<unsigned char, int>(p, ab, bb, ann, bmask, NULL, amaskm);
//    mexPrintf("  %d %d %d %p %p\n", annd->get(0,0)[0], annd->get(1,0)[0], annd->get(0,1)[0], amaskm, bmask);
    vec_nn<unsigned char, int>(p, ab, bb, ann, annd, amaskm, bmask, 0, 0, rp, 0, 0, 0, NULL, p->cores); 
    if (ann_prev) vec_minnn<unsigned char, int>(p, ab, bb, ann, annd, ann_prev, bmask, 0, 0, rp, NULL, amaskm, p->cores);  
    annd_final = vecbitmap_to_bitmap(annd);
    delete annd;
  } 
  else if (mode == MODE_VECF) {
//    mexPrintf("mode vecf %dx%dx%d, %dx%dx%d\n", af->w, af->h, af->n, bf->w, bf->h, bf->n);
//    mexPrintf("  %f %f %f %f\n", af->get(0,0)[0], af->get(1,0)[0], af->get(0,1)[0], af->get(0,0)[1]);
    if (!af || !bf) { mexErrMsgTxt("internal error: no a or b image"); }
    ann = vec_init_nn<float>(p, af, bf, bmask, NULL, amaskm);
    VECBITMAP<float> *annd = vec_init_dist<float, float>(p, af, bf, ann, bmask, NULL, amaskm);
    vec_nn<float, float>(p, af, bf, ann, annd, amaskm, bmask, 0, 0, rp, 0, 0, 0, NULL, p->cores); 
    if (ann_prev) vec_minnn<float, float>(p, af, bf, ann, annd, ann_prev, bmask, 0, 0, rp, NULL, amaskm, p->cores);  
    annd_final = create_bitmap(annd->w, annd->h);
    clear(annd_final);
    delete annd;
  }
*/
  else if(mode == MODE_VECB) {
    if (sim_mode) { mexErrMsgTxt("internal error: rotation+scales not implemented with descriptor mode"); }
    if (knn_chosen > 1) { mexErrMsgTxt("internal error: kNN not implemented with descriptor mode"); }
//    mexPrintf("mode vecb_xc %dx%dx%d, %dx%dx%d\n", ab->w, ab->h, ab->n, bb->w, bb->h, bb->n);
    // input as uint8 discriptors per pixel
    if (!ab || !bb) { mexErrMsgTxt("internal error: no a or b image"); }
    ann = XCvec_init_nn<unsigned char>(p, ab, bb, bmask, NULL, amaskm);
    VECBITMAP<int> *annd = XCvec_init_dist<unsigned char, int>(p, ab, bb, ann, bmask, NULL, amaskm);
    XCvec_nn<unsigned char, int>(p, ab, bb, ann, annd, amaskm, bmask, 0, 0, rp, 0, 0, 0, NULL, p->cores); 
    if (ann_prev) XCvec_minnn<unsigned char, int>(p, ab, bb, ann, annd, ann_prev, bmask, 0, 0, rp, NULL, amaskm, p->cores);  
    annd_final = vecbitmap_to_bitmap(annd);
    delete annd;
  } else if(mode == MODE_VECF) {
    if (sim_mode) { mexErrMsgTxt("internal error: rotation+scales not implemented with descriptor mode"); }
    if (knn_chosen > 1) { mexErrMsgTxt("internal error: kNN not implemented with descriptor mode"); }
    // input as float/double discriptors per pixel
    if (!af || !bf) { mexErrMsgTxt("internal error: no a or b image"); }
    ann = XCvec_init_nn<float>(p, af, bf, bmask, NULL, amaskm);
    VECBITMAP<float> *annd = XCvec_init_dist<float, float>(p, af, bf, ann, bmask, NULL, amaskm);
    XCvec_nn<float, float>(p, af, bf, ann, annd, amaskm, bmask, 0, 0, rp, 0, 0, 0, NULL, p->cores); 
    if (ann_prev) XCvec_minnn<float, float>(p, af, bf, ann, annd, ann_prev, bmask, 0, 0, rp, NULL, amaskm, p->cores);  
    annd_final = create_bitmap(annd->w, annd->h);
    clear(annd_final);
    delete annd;	
  }

  destroy_region_masks(amaskm);

  // output ann: x | y | patch_distance
  if(nout >= 1) {
    mxArray *ans = NULL;
    if (knn_chosen > 1) {
      if (sim_mode) { mexErrMsgTxt("rotating+scaling patches return value not implemented with knn"); }
      mwSize dims[4] = { ah, aw, 3, knn_chosen };
      ans = mxCreateNumericArray(4, dims, mxINT32_CLASS, mxREAL);
      int *data = (int *) mxGetData(ans);
      for (int kval = 0; kval < knn_chosen; kval++) {
        int *xchan = &data[aw*ah*3*kval+0];
        int *ychan = &data[aw*ah*3*kval+aw*ah];
        int *dchan = &data[aw*ah*3*kval+2*aw*ah];
        for (int y = 0; y < ah; y++) {
//          int *ann_row = (int *) ann->line[y];
//          int *annd_row = (int *) annd_final->line[y];
          for (int x = 0; x < aw; x++) {
//            int pp = ann_row[x];
            int pp = vann->get(x, y)[kval];
            int pos = y + x * ah;
            xchan[pos] = INT_TO_X(pp);
            ychan[pos] = INT_TO_Y(pp);
            dchan[pos] = vannd->get(x, y)[kval];
          }
        }
      }
    } else if (ann_sim_final) {
      mwSize dims[3] = { ah, aw, 5 };
      ans = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);
      float *data = (float *) mxGetData(ans);
      float *xchan = &data[0];
      float *ychan = &data[aw*ah];
      float *dchan = &data[2*aw*ah];
      float *tchan = &data[3*aw*ah];
      float *schan = &data[4*aw*ah];
      double angle_scale = 2.0*M_PI/NUM_ANGLES;
      for (int y = 0; y < ah; y++) {
        int *ann_row = (int *) ann->line[y];
        int *annd_row = (int *) annd_final->line[y];
        int *ann_sim_row = ann_sim_final ? (int *) ann_sim_final->line[y]: NULL;
        for (int x = 0; x < aw; x++) {
          int pp = ann_row[x];
          int pos = y + x * ah;
          xchan[pos] = INT_TO_X(pp);
          ychan[pos] = INT_TO_Y(pp);
          dchan[pos] = annd_row[x];
          if (ann_sim_final) {
            int v = ann_sim_row[x];
            int tval = INT_TO_Y(v)&(NUM_ANGLES-1);
            int sval = INT_TO_X(v);

            tchan[pos] = tval*angle_scale;
            schan[pos] = xform_scale_table[sval]*(1.0/65536.0);
          }
        }
      }
    } else {
      mwSize dims[3] = { ah, aw, 3 };
      ans = mxCreateNumericArray(3, dims, mxINT32_CLASS, mxREAL);
      int *data = (int *) mxGetData(ans);
      int *xchan = &data[0];
      int *ychan = &data[aw*ah];
      int *dchan = &data[2*aw*ah];
      for (int y = 0; y < ah; y++) {
        int *ann_row = (int *) ann->line[y];
        int *annd_row = (int *) annd_final->line[y];
        for (int x = 0; x < aw; x++) {
          int pp = ann_row[x];
          int pos = y + x * ah;
          xchan[pos] = INT_TO_X(pp);
          ychan[pos] = INT_TO_Y(pp);
          dchan[pos] = annd_row[x];
        }
      }
    }
    pout[0] = ans;
  }

  // clean up
  delete vann;
  delete vann_sim;
  delete vannd;
  delete p;
  delete rp;
  destroy_bitmap(a);
  destroy_bitmap(borig);
  delete ab;
  delete bb;
  delete af;
  delete bf;
  destroy_bitmap(ann);
  destroy_bitmap(annd_final);
  destroy_bitmap(ann_sim_final);
  if (ann_prev) destroy_bitmap(ann_prev);
  if (ann_window) destroy_bitmap(ann_window);
  if (awinsize) destroy_bitmap(awinsize);
}
