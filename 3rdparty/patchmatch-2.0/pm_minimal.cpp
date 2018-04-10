
/* -------------------------------------------------------------------------
  Minimal (unoptimized) example of PatchMatch. Requires that ImageMagick be installed.

  To improve generality you can:
   - Use whichever distance function you want in dist(), e.g. compare SIFT descriptors computed densely.
   - Search over a larger search space, such as rotating+scaling patches (see MATLAB mex for examples of both)
  
  To improve speed you can:
   - Turn on optimizations (/Ox /Oi /Oy /fp:fast or -O6 -s -ffast-math -fomit-frame-pointer -fstrength-reduce -msse2 -funroll-loops)
   - Use the MATLAB mex which is already tuned for speed
   - Use multiple cores, tiling the input. See our publication "The Generalized PatchMatch Correspondence Algorithm"
   - Tune the distance computation: manually unroll loops for each patch size, use SSE instructions (see readme)
   - Precompute random search samples (to avoid using rand, and mod)
   - Move to the GPU
  -------------------------------------------------------------------------- */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef MAX
#define MAX(a, b) ((a)>(b)?(a):(b))
#define MIN(a, b) ((a)<(b)?(a):(b))
#endif

/* -------------------------------------------------------------------------
   BITMAP: Minimal image class
   ------------------------------------------------------------------------- */

class BITMAP { public:
  int w, h;
  int *data;
  BITMAP(int w_, int h_) :w(w_), h(h_) { data = new int[w*h]; }
  ~BITMAP() { delete[] data; }
  int *operator[](int y) { return &data[y*w]; }
};

void check_im() {
  if (system("identify > null.txt") != 0) {
    fprintf(stderr, "ImageMagick must be installed, and 'convert' and 'identify' must be in the path\n"); exit(1);
  }
}

BITMAP *load_bitmap(const char *filename) {
  check_im();
  char rawname[256], txtname[256];
  strcpy(rawname, filename);
  strcpy(txtname, filename);
  if (!strstr(rawname, ".")) { fprintf(stderr, "Error reading image '%s': no extension found\n", filename); exit(1); }
  sprintf(strstr(rawname, "."), ".raw");
  sprintf(strstr(txtname, "."), ".txt");
  char buf[256];
  sprintf(buf, "convert %s rgba:%s", filename, rawname);
  if (system(buf) != 0) { fprintf(stderr, "Error reading image '%s': ImageMagick convert gave an error\n", filename); exit(1); }
  sprintf(buf, "identify -format \"%%w %%h\" %s > %s", filename, txtname);
  if (system(buf) != 0) { fprintf(stderr, "Error reading image '%s': ImageMagick identify gave an error\n", filename); exit(1); }
  FILE *f = fopen(txtname, "rt");
  if (!f) { fprintf(stderr, "Error reading image '%s': could not read output of ImageMagick identify\n", filename); exit(1); }
  int w = 0, h = 0;
  if (fscanf(f, "%d %d", &w, &h) != 2) { fprintf(stderr, "Error reading image '%s': could not get size from ImageMagick identify\n", filename); exit(1); }
  fclose(f);
  f = fopen(rawname, "rb");
  BITMAP *ans = new BITMAP(w, h);
  unsigned char *p = (unsigned char *) ans->data;
  for (int i = 0; i < w*h*4; i++) {
    int ch = fgetc(f);
    if (ch == EOF) { fprintf(stderr, "Error reading image '%s': raw file is smaller than expected size %dx%dx4\n", filename, w, h, 4); exit(1); }
    *p++ = ch;
  }
  fclose(f);
  return ans;
}

void save_bitmap(BITMAP *bmp, const char *filename) {
  check_im();
  char rawname[256];
  strcpy(rawname, filename);
  if (!strstr(rawname, ".")) { fprintf(stderr, "Error writing image '%s': no extension found\n", filename); exit(1); }
  sprintf(strstr(rawname, "."), ".raw");
  char buf[256];
  FILE *f = fopen(rawname, "wb");
  if (!f) { fprintf(stderr, "Error writing image '%s': could not open raw temporary file\n", filename); exit(1); }
  unsigned char *p = (unsigned char *) bmp->data;
  for (int i = 0; i < bmp->w*bmp->h*4; i++) {
    fputc(*p++, f);
  }
  fclose(f);
  sprintf(buf, "convert -size %dx%d -depth 8 rgba:%s %s", bmp->w, bmp->h, rawname, filename);
  if (system(buf) != 0) { fprintf(stderr, "Error writing image '%s': ImageMagick convert gave an error\n", filename); exit(1); }
}

/* -------------------------------------------------------------------------
   PatchMatch, using L2 distance between upright patches that translate only
   ------------------------------------------------------------------------- */

int patch_w  = 7;
int pm_iters = 5;
int rs_max   = INT_MAX;

#define XY_TO_INT(x, y) (((y)<<12)|(x))
#define INT_TO_X(v) ((v)&((1<<12)-1))
#define INT_TO_Y(v) ((v)>>12)

/* Measure distance between 2 patches with upper left corners (ax, ay) and (bx, by), terminating early if we exceed a cutoff distance.
   You could implement your own descriptor here. */
int dist(BITMAP *a, BITMAP *b, int ax, int ay, int bx, int by, int cutoff=INT_MAX) {
  int ans = 0;
  for (int dy = 0; dy < patch_w; dy++) {
    int *arow = &(*a)[ay+dy][ax];
    int *brow = &(*b)[by+dy][bx];
    for (int dx = 0; dx < patch_w; dx++) {
      int ac = arow[dx];
      int bc = brow[dx];
      int dr = (ac&255)-(bc&255);
      int dg = ((ac>>8)&255)-((bc>>8)&255);
      int db = (ac>>16)-(bc>>16);
      ans += dr*dr + dg*dg + db*db;
    }
    if (ans >= cutoff) { return cutoff; }
  }
  return ans;
}

void improve_guess(BITMAP *a, BITMAP *b, int ax, int ay, int &xbest, int &ybest, int &dbest, int bx, int by) {
  int d = dist(a, b, ax, ay, bx, by, dbest);
  if (d < dbest) {
    dbest = d;
    xbest = bx;
    ybest = by;
  }
}

/* Match image a to image b, returning the nearest neighbor field mapping a => b coords, stored in an RGB 24-bit image as (by<<12)|bx. */
void patchmatch(BITMAP *a, BITMAP *b, BITMAP *&ann, BITMAP *&annd) {
  /* Initialize with random nearest neighbor field (NNF). */
  ann = new BITMAP(a->w, a->h);
  annd = new BITMAP(a->w, a->h);
  int aew = a->w - patch_w+1, aeh = a->h - patch_w + 1;       /* Effective width and height (possible upper left corners of patches). */
  int bew = b->w - patch_w+1, beh = b->h - patch_w + 1;
  memset(ann->data, 0, sizeof(int)*a->w*a->h);
  memset(annd->data, 0, sizeof(int)*a->w*a->h);
  for (int ay = 0; ay < aeh; ay++) {
    for (int ax = 0; ax < aew; ax++) {
      int bx = rand()%bew;
      int by = rand()%beh;
      (*ann)[ay][ax] = XY_TO_INT(bx, by);
      (*annd)[ay][ax] = dist(a, b, ax, ay, bx, by);
    }
  }
  for (int iter = 0; iter < pm_iters; iter++) {
    /* In each iteration, improve the NNF, by looping in scanline or reverse-scanline order. */
    int ystart = 0, yend = aeh, ychange = 1;
    int xstart = 0, xend = aew, xchange = 1;
    if (iter % 2 == 1) {
      xstart = xend-1; xend = -1; xchange = -1;
      ystart = yend-1; yend = -1; ychange = -1;
    }
    for (int ay = ystart; ay != yend; ay += ychange) {
      for (int ax = xstart; ax != xend; ax += xchange) { 
        /* Current (best) guess. */
        int v = (*ann)[ay][ax];
        int xbest = INT_TO_X(v), ybest = INT_TO_Y(v);
        int dbest = (*annd)[ay][ax];

        /* Propagation: Improve current guess by trying instead correspondences from left and above (below and right on odd iterations). */
        if ((unsigned) (ax - xchange) < (unsigned) aew) {
          int vp = (*ann)[ay][ax-xchange];
          int xp = INT_TO_X(vp) + xchange, yp = INT_TO_Y(vp);
          if ((unsigned) xp < (unsigned) bew) {
            improve_guess(a, b, ax, ay, xbest, ybest, dbest, xp, yp);
          }
        }

        if ((unsigned) (ay - ychange) < (unsigned) aeh) {
          int vp = (*ann)[ay-ychange][ax];
          int xp = INT_TO_X(vp), yp = INT_TO_Y(vp) + ychange;
          if ((unsigned) yp < (unsigned) beh) {
            improve_guess(a, b, ax, ay, xbest, ybest, dbest, xp, yp);
          }
        }

        /* Random search: Improve current guess by searching in boxes of exponentially decreasing size around the current best guess. */
        int rs_start = rs_max;
        if (rs_start > MAX(b->w, b->h)) { rs_start = MAX(b->w, b->h); }
        for (int mag = rs_start; mag >= 1; mag /= 2) {
          /* Sampling window */
          int xmin = MAX(xbest-mag, 0), xmax = MIN(xbest+mag+1,bew);
          int ymin = MAX(ybest-mag, 0), ymax = MIN(ybest+mag+1,beh);
          int xp = xmin+rand()%(xmax-xmin);
          int yp = ymin+rand()%(ymax-ymin);
          improve_guess(a, b, ax, ay, xbest, ybest, dbest, xp, yp);
        }

        (*ann)[ay][ax] = XY_TO_INT(xbest, ybest);
        (*annd)[ay][ax] = dbest;
      }
    }
  }
}

int main(int argc, char *argv[]) {
  argc--;
  argv++;
  if (argc != 4) { fprintf(stderr, "pm_minimal a b ann annd\n"
                                   "Given input images a, b outputs nearest neighbor field 'ann' mapping a => b coords, and the squared L2 distance 'annd'\n"
                                   "These are stored as RGB 24-bit images, with a 24-bit int at every pixel. For the NNF we store (by<<12)|bx."); exit(1); }
  printf("Loading input images\n");
  BITMAP *a = load_bitmap(argv[0]);
  BITMAP *b = load_bitmap(argv[1]);
  BITMAP *ann = NULL, *annd = NULL;
  printf("Running PatchMatch\n");
  patchmatch(a, b, ann, annd);
  printf("Saving output images\n");
  save_bitmap(ann, argv[2]);
  save_bitmap(annd, argv[3]);
  return 0;
}
