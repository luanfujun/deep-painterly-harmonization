
// MATLAB interface, for vote().

#include "allegro_emu.h"
#include "mex.h"
#include "nn.h"
//#include "matrix.h"
#include "mexutil.h"

void mexFunction(int nout, mxArray *pout[], int nin, const mxArray *pin[]) {
  if (nin < 2) { mexErrMsgTxt("votemex called with < 2 input arguments"); }
  const mxArray *B = pin[0], *Ann = pin[1], *Bnn = NULL;

  BITMAP *b = convert_bitmap(B);

  Params *p = new Params();
  // [bnn=[]], [algo='cpu'], [patch_w=7], [bmask=[]], [bweight=[]], [coherence_weight=1], [complete_weight=1], [amask=[]], [aweight=[]]
  BITMAP *bmask = NULL, *bweight = NULL, *amask = NULL, *aweight = NULL, *ainit = NULL;
  double coherence_weight = 1, complete_weight = 1;
  int i = 2;
  if (nin > i && !mxIsEmpty(pin[i])) { Bnn = pin[i]; } i++;
  if (nin > i && !mxIsEmpty(pin[i])) {
    if			(mxStringEquals(pin[i], "cpu")) { p->algo = ALGO_CPU; }
    else if (mxStringEquals(pin[i], "gpucpu")) { p->algo = ALGO_GPUCPU; }
    else if (mxStringEquals(pin[i], "cputiled")) { p->algo = ALGO_CPUTILED; }
    else { mexErrMsgTxt("Unknown algorithm"); }
  } i++;
  if (nin > i && !mxIsEmpty(pin[i])) { p->patch_w = int(mxGetScalar(pin[i])); } i++;
  if (nin > i && !mxIsEmpty(pin[i])) { bmask = convert_bitmap(pin[i]); } i++;
  if (nin > i && !mxIsEmpty(pin[i])) { bweight = convert_bitmapf(pin[i]); } i++;
  if (nin > i && !mxIsEmpty(pin[i])) { coherence_weight = mxGetScalar(pin[i]); } i++;
  if (nin > i && !mxIsEmpty(pin[i])) { complete_weight = mxGetScalar(pin[i]); } i++;
  if (nin > i && !mxIsEmpty(pin[i])) { amask = convert_bitmap(pin[i]); } i++;
  if (nin > i && !mxIsEmpty(pin[i])) { aweight = convert_bitmapf(pin[i]); } i++;
  if (nin > i && !mxIsEmpty(pin[i])) { ainit = convert_bitmap(pin[i]); } i++;

  int aclip = 0, bclip = 0;
  BITMAP *ann = convert_field(p, Ann, b->w, b->h, aclip);
  BITMAP *bnn = Bnn ? convert_field(p, Bnn, ann->w, ann->h, bclip): NULL;

  int nclip = aclip + bclip;
  //if (nclip) printf("Warning: clipped %d votes (%d a -> b, %d b -> a)\n", nclip, aclip, bclip);

  RegionMasks *amaskm = amask ? new RegionMasks(p, amask): NULL;
  //BITMAP *a = vote(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amaskm, aweight, ainit);
  // David Jacobs -- Added mask_self_only as true for multiple.
  BITMAP *a = vote(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amaskm, aweight, ainit, NULL, NULL, 1);
	
  destroy_region_masks(amaskm);

	if(nout >= 1) {
		mxArray *ans = bitmap_to_array(a);
		pout[0] = ans;
  } 

  delete p;
  destroy_bitmap(a);
  destroy_bitmap(ainit);
  destroy_bitmap(b);
  destroy_bitmap(ann);
  destroy_bitmap(bnn);
  destroy_bitmap(bmask);
  destroy_bitmap(bweight);
//  destroy_bitmap(amask);
  destroy_bitmap(aweight);
}
