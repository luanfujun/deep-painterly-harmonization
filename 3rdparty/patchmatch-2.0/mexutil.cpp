
#include "mexutil.h"
#include "nn.h"
#include <string.h>

// stack RGB to a single 32bit integer
BITMAP *convert_bitmap(const mxArray *A) {
  if (mxGetNumberOfDimensions(A) != 3) { mexErrMsgTxt("dims != 3"); }
  if (mxGetDimensions(A)[2] != 3) { mexErrMsgTxt("color channels != 3"); }

	// matlab, h * w * n: (y, x, i) -> y + x * h + i * w * h
  int h = mxGetDimensions(A)[0];
  int w = mxGetDimensions(A)[1];
	int offset = h * w;
	int offset2 = offset << 1; // h * w * 2

  if (mxIsUint8(A)) {
    unsigned char *data = (unsigned char *) mxGetData(A);
    BITMAP *ans = create_bitmap(w, h);
		for (int y = 0; y < h; y++) {
      int *row = (int *) ans->line[y];
			for (int x = 0; x < w; x++) {
				unsigned char *base = data + y + x * h;
				int r = base[0];
				int g = base[offset];
				int b = base[offset2];
				clip_value<int>(&r, 0, 255);
				clip_value<int>(&g, 0, 255);
				clip_value<int>(&b, 0, 255);
        row[x] = r|(g<<8)|(b<<16);
      }
    }
    return ans;
  } 
	else if (mxIsSingle(A)) {
    float *data = (float *) mxGetData(A);
    BITMAP *ans = create_bitmap(w, h);
    for (int y = 0; y < h; y++) {
      int *row = (int *) ans->line[y];
      for (int x = 0; x < w; x++) {
				float *base = data + y + x * h;
        float r = base[0];
        float g = base[offset];
        float b = base[offset2];
				clip_value<float>(&r, 0, 1);
				clip_value<float>(&g, 0, 1);
				clip_value<float>(&b, 0, 1);
        row[x] = int(r*255)|(int(g*255)<<8)|(int(b*255)<<16);
      }
    }
    return ans;
  } 
	else if (mxIsDouble(A)) {
    double *data = (double *) mxGetData(A);
    BITMAP *ans = create_bitmap(w, h);
    for (int y = 0; y < h; y++) {
      int *row = (int *) ans->line[y];
      for (int x = 0; x < w; x++) {
				double *base = data + y + x * h;
        double r = base[0];
        double g = base[offset];
        double b = base[offset2];
        clip_value<double>(&r, 0, 1);
				clip_value<double>(&g, 0, 1);
				clip_value<double>(&b, 0, 1);
				row[x] = int(r*255)|(int(g*255)<<8)|(int(b*255)<<16);
      }
    }
    return ans;
  } 
	else {
    mexErrMsgTxt("bitmap not uint8, single, or double");
		return NULL;
  }
}

BITMAP *convert_bitmapf(const mxArray *A) {
  if (!(mxGetNumberOfDimensions(A) == 2 ||
       (mxGetNumberOfDimensions(A) == 3 && mxGetDimensions(A)[2] == 3))) { mexErrMsgTxt("float bitmap doesn't have 2 dims or 3 dims with 3 channels"); }

  int h = mxGetDimensions(A)[0];
  int w = mxGetDimensions(A)[1];

  if (mxIsUint8(A)) { // treated as 32bit float
    unsigned char *data = (unsigned char *) mxGetData(A);
    BITMAP *ans = create_bitmap(w, h);
    for (int y = 0; y < h; y++) {
      float *row = (float *) ans->line[y];
      for (int x = 0; x < w; x++) {
        row[x] = data[y+x*h];
      }
    }
    return ans;
  } 
	else if (mxIsDouble(A)) {
    double *data = (double *) mxGetData(A);
    BITMAP *ans = create_bitmap(w, h);
    for (int y = 0; y < h; y++) {
      float *row = (float *) ans->line[y];
      for (int x = 0; x < w; x++) {
        row[x] = data[y+x*h];
      }
    }
    return ans;
  } 
	else {
    mexErrMsgTxt("float bitmap not uint8 or double");
		return NULL;
  }
}

// stack (x, y) to an integer by concatenate "yx" (assume y and x has at most 12 bit, or 4096 values)
BITMAP *convert_field(Params *p, const mxArray *A, int bw, int bh, int &nclip, int trim_patch) {
  nclip = 0;
  int h = mxGetDimensions(A)[0];
  int w = mxGetDimensions(A)[1];
	int offset = h * w;
	int ndims = mxGetNumberOfDimensions(A);
  if (ndims != 3) {
    char buf[256];
    if (ndims == 1) {
      sprintf(buf, "field dims != 3 (%d 1d array where nn field expected)", h);
    } else if (ndims == 2) {
      sprintf(buf, "field dims != 3 (%dx%d 2d array where nn field expected)", h, w);
    } else {
      sprintf(buf, "field dims != 3 (%d dimension array where nn field expected)", ndims);
    }
    mexErrMsgTxt(buf);
  }
  if (mxGetDimensions(A)[2] != 3) { mexErrMsgTxt("field channels != 3"); }
  if (!mxIsInt32(A)) { mexErrMsgTxt("field is not int32"); }

  int bew = trim_patch ? (bw - p->patch_w + 1): bw;
  int beh = trim_patch ? (bh - p->patch_w + 1): bh;

  unsigned int *data = (unsigned int *) mxGetData(A);
  BITMAP *ann = create_bitmap(w, h);
  //annd = create_bitmap(w, h);
  for (int y = 0; y < h; y++) {
    int *ann_row = (int *) ann->line[y];
    //int *annd_row = (int *) annd->line[y];
    for (int x = 0; x < w; x++) {
			unsigned int *base = data + y + x*h; 
      int xp = base[0];
      int yp = base[offset];
      if ((unsigned) xp >= (unsigned) bew ||
          (unsigned) yp >= (unsigned) beh) {
        nclip++;
				clip_value<int>(&xp, 0, bew-1);
				clip_value<int>(&yp, 0, beh-1);
      }

      //int dp = data[(y+x*h)+2*w*h];
      ann_row[x] = XY_TO_INT(xp, yp);
      //annd_row[x] = dp;
    }
  }
  return ann;
}

BITMAP *convert_winsize_field(Params *p, const mxArray *A, int w, int h) {
  int ndims = mxGetNumberOfDimensions(A);
  if (!( (ndims == 2)||((ndims == 3)&&(mxGetDimensions(A)[2] == 2))&&(w==mxGetDimensions(A)[1])&&(h==mxGetDimensions(A)[0]) )) {
		char buf[256];
	//sprintf(buf, "ndims=%d, mxGetDim(A)=[%d %d %d], [2 w h]=[2 %d %d] \n", ndims,mxGetDimensions(A)[2],mxGetDimensions(A)[1],mxGetDimensions(A)[0],h,w);
		sprintf(buf, "winsize field dims=%d (a [%dx%d] or [%dx%dx2] field of window size is expected)", ndims, h, w, h, w);
    mexErrMsgTxt(buf);
  }
  if (!mxIsInt32(A)) { mexErrMsgTxt("winsize field is not int32"); }

  unsigned int *data = (unsigned int *) mxGetData(A);
	int offset = h * w;
  BITMAP *awsz = create_bitmap(w, h);
  char buf[256];
  if (ndims == 2) {
		for (int y = 0; y < h; y++) {
			int *awsz_row = (int *) awsz->line[y];
			for (int x = 0; x < w; x++) {
				int wp = data[y+x*h];
				awsz_row[x] = XY_TO_INT(wp, wp);
			}
		}
  } 
	else if ((ndims == 3)&&(mxGetDimensions(A)[2] == 2)) {
		for (int y = 0; y < h; y++) {
			int *awsz_row = (int *) awsz->line[y];
			for (int x = 0; x < w; x++) {
				unsigned int *base = data + y + x * h;
				int wp = base[0];
				int hp = base[offset];
				awsz_row[x] = XY_TO_INT(wp, hp);
			}
    }
  }
  return awsz;
}

mxArray *bitmap_to_array(BITMAP *a) {
  mwSize dims[3] = { a->h, a->w, 3 };
  mxArray *ans = mxCreateNumericArray(3, dims, mxUINT8_CLASS, mxREAL);
  unsigned char *data = (unsigned char *) mxGetData(ans);
	int offset = a->w * a->h;
	int offset2 = offset << 1;
  unsigned char *rchan = &data[0];
  unsigned char *gchan = &data[offset];
  unsigned char *bchan = &data[offset2];
  for (int y = 0; y < a->h; y++) {
    int *row = (int *) a->line[y];
    for (int x = 0; x < a->w; x++) {
      int c = row[x];
			int pos = y + x * a->h;
      rchan[pos] = c&255;
      gchan[pos] = (c>>8)&255;
      bchan[pos] = (c>>16);
    }
  }
  return ans;
}

int mxStringEquals(const mxArray *A, const char *s) {
  char buf[256];
  if (!mxIsChar(A)) { return 0; }
  if (mxGetString(A, buf, 255)) { return 0; }
  return strcmp(s, buf) == 0;
}
