
#ifndef _mexutil_h
#define _mexutil_h

#include "allegro_emu.h"
#include "mex.h"
#include "nn.h"
#include "vecnn.h"

BITMAP *convert_bitmap(const mxArray *A);
BITMAP *convert_bitmapf(const mxArray *A);
BITMAP *convert_field(Params *p, const mxArray *A, int bw, int bh, int &nclip, int trim_patch=1);
BITMAP *convert_winsize_field(Params *p, const mxArray *A, int w, int h);
mxArray *bitmap_to_array(BITMAP *a);
int mxStringEquals(const mxArray *A, const char *s);

template<class T> 
inline void clip_value(T *p, T low, T high) { 
	if(*p < low) *p = low; 
	else if(*p > high) *p = high; 
}

// from MATLAB's column major to C++'s row major
template<class T>
VECBITMAP<T> *convert_vecbitmap(const mxArray *A) {
  if (mxGetNumberOfDimensions(A) != 3) { mexErrMsgTxt("dims != 3"); }

  int h = mxGetDimensions(A)[0];
  int w = mxGetDimensions(A)[1];
  int n = mxGetDimensions(A)[2];

  VECBITMAP<T> *ans = new VECBITMAP<T>(w, h, n);

  int offset = w*h;
  if (mxIsUint8(A)) {
    unsigned char *data = (unsigned char *) mxGetData(A);
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        T *patch = ans->get(x, y);
        unsigned char *pos0 = &data[y+x*h];
        for (int i = 0; i < n; i++) {
          patch[i] = pos0[offset*i];
        }
      }
    }
  } 
	else if (mxIsSingle(A)) {
    float *data = (float *) mxGetData(A);
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        T *patch = ans->get(x, y);
        float *pos0 = &data[y+x*h];
        for (int i = 0; i < n; i++) {
          patch[i] = pos0[offset*i];
        }
      }
    }
  } 
	else if (mxIsDouble(A)) {
    double *data = (double *) mxGetData(A);
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        T *patch = ans->get(x, y);
        double *pos0 = &data[y+x*h];
        for (int i = 0; i < n; i++) {
          patch[i] = pos0[offset*i];
        }
      }
    }
  } 
	else {
    mexErrMsgTxt("bitmap not uint8, single, or double");
  }
  return ans;
}

#endif
