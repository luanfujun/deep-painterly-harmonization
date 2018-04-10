
#include "allegro_emu.h"
#include <stdlib.h>
#include <stdio.h>

BITMAP *create_bitmap(int w, int h) {
  BITMAP *ans = new BITMAP(w, h);
  ans->data = new unsigned char[4*w*h]; // always 32bit
  ans->line = new unsigned char*[h];
  for (int y = 0; y < h; y++) 
    ans->line[y] = &ans->data[y*4*w];
  return ans;
}

void blit(BITMAP *a, BITMAP *b, int ax, int ay, int bx, int by, int w, int h) {
  if (ax != 0 || ay != 0 || bx != 0 || by != 0) { fprintf(stderr, "blit not fully implemented, nonzero coords not supported\n"); exit(1); }
  for (int y = 0; y < h; y++) {
    int *arow = (int *) a->line[y];
    int *brow = (int *) b->line[y];
    for (int x = 0; x < w; x++) {
      brow[x] = arow[x];
    }
  }
}

void destroy_bitmap(BITMAP *bmp) {
	if (bmp) {
		delete[] bmp->line;
		delete[] bmp->data;
		delete bmp;
	}
}

fixed fixmul(fixed a0, fixed b0) {
  long long a = a0;
  long long b = b0;
  return (int) ((a*b)>>16);
}

void clear(BITMAP *bmp) {
  clear_to_color(bmp, 0);
}

void clear_to_color(BITMAP *bmp, int c) {
  for (int y = 0; y < bmp->h; y++) {
    int *row = (int *) bmp->line[y];
    for (int x = 0; x < bmp->w; x++) {
      row[x] = c;
    }
  }
}

int bitmap_color_depth(BITMAP *bmp) {
  return 32;
}

BITMAP *create_bitmap_ex(int depth, int w, int h) {
  if (depth != 32) { fprintf(stderr, "depth not supported for create_bitmap_ex: %d\n", depth); exit(1); }
  return create_bitmap(w, h);
}
