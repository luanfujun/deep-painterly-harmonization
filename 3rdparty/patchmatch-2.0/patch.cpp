
#include "patch.h"

static char AdobePatentID_P876E1[] = "AdobePatentID=\"P876E1\"";

template<>
int fast_patch_dist<1, 0>(int *adata, BITMAP *b, int bx, int by, int maxval, Params *p) {
  //if (IS_MASK && bmask && ((int *) bmask->line[by])[bx]) { return INT_MAX; }
  int *row2 = ((int *) b->line[by])+bx;
  {
    unsigned int c1 = adata[0];
    unsigned int c2 = row2[0];
    int dr = (c1&255)-(c2&255);
    int dg = ((c1>>8)&255)-((c2>>8)&255);
    int db = (c1>>16)-(c2>>16);
    return dr*dr+dg*dg+db*db;
  }
}

template<>
int fast_patch_dist<2, 0>(int *adata, BITMAP *b, int bx, int by, int maxval, Params *p) {
  //if (bmask && ((int *) bmask->line[by])[bx]) { return INT_MAX; }
  int ans = 0;
  for (int dy = 0; dy < 2; dy++) {
    int *row2 = ((int *) b->line[by+dy])+bx;
    {
      unsigned int c1 = adata[0];
      unsigned int c2 = row2[0];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
      if (ans > maxval) { return ans; }
    }
    {
      unsigned int c1 = adata[1];
      unsigned int c2 = row2[1];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
      if (ans > maxval) { return ans; }
    }
    adata += 2;
  }
  return ans;
}

template<>
int fast_patch_dist<3, 0>(int *adata, BITMAP *b, int bx, int by, int maxval, Params *p) {
  //if (bmask && ((int *) bmask->line[by])[bx]) { return INT_MAX; }
  int ans = 0;
  for (int dy = 0; dy < 3; dy++) {
    int *row2 = ((int *) b->line[by+dy])+bx;
    {
      unsigned int c1 = adata[0];
      unsigned int c2 = row2[0];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
      if (ans > maxval) { return ans; }
    }
    {
      unsigned int c1 = adata[1];
      unsigned int c2 = row2[1];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
      if (ans > maxval) { return ans; }
    }
    {
      unsigned int c1 = adata[2];
      unsigned int c2 = row2[2];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
      if (ans > maxval) { return ans; }
    }
    adata += 3;
  }
  return ans;
}

template<>
int fast_patch_dist<4, 0>(int *adata, BITMAP *b, int bx, int by, int maxval, Params *p) {
  //if (bmask && ((int *) bmask->line[by])[bx]) { return INT_MAX; }
  int ans = 0;
  for (int dy = 0; dy < 4; dy++) {
    int *row2 = ((int *) b->line[by+dy])+bx;
    {
      unsigned int c1 = adata[0];
      unsigned int c2 = row2[0];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
      if (ans > maxval) { return ans; }
    }
    {
      unsigned int c1 = adata[1];
      unsigned int c2 = row2[1];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
      if (ans > maxval) { return ans; }
    }
    {
      unsigned int c1 = adata[2];
      unsigned int c2 = row2[2];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
      if (ans > maxval) { return ans; }
    }
    {
      unsigned int c1 = adata[3];
      unsigned int c2 = row2[3];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
      if (ans > maxval) { return ans; }
    }
    adata += 4;
  }
  return ans;
}

template<>
int fast_patch_dist<5, 0>(int *adata, BITMAP *b, int bx, int by, int maxval, Params *p) {
  //if (bmask && ((int *) bmask->line[by])[bx]) { return INT_MAX; }
  int ans = 0;
  for (int dy = 0; dy < 5; dy++) {
    int *row2 = ((int *) b->line[by+dy])+bx;
    {
      unsigned int c1 = adata[0];
      unsigned int c2 = row2[0];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
      if (ans > maxval) { return ans; }
    }
    {
      unsigned int c1 = adata[1];
      unsigned int c2 = row2[1];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
      if (ans > maxval) { return ans; }
    }
    {
      unsigned int c1 = adata[2];
      unsigned int c2 = row2[2];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
      if (ans > maxval) { return ans; }
    }
    {
      unsigned int c1 = adata[3];
      unsigned int c2 = row2[3];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
      if (ans > maxval) { return ans; }
    }
    {
      unsigned int c1 = adata[4];
      unsigned int c2 = row2[4];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
      if (ans > maxval) { return ans; }
    }
    adata += 5;
  }
  return ans;
}

template<>
int fast_patch_dist<6, 0>(int *adata, BITMAP *b, int bx, int by, int maxval, Params *p) {
  //if (bmask && ((int *) bmask->line[by])[bx]) { return INT_MAX; }
  int ans = 0;
  for (int dy = 0; dy < 6; dy++) {
    int *row2 = ((int *) b->line[by+dy])+bx;
    {
      unsigned int c1 = adata[0];
      unsigned int c2 = row2[0];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
      if (ans > maxval) { return ans; }
    }
    {
      unsigned int c1 = adata[1];
      unsigned int c2 = row2[1];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
      if (ans > maxval) { return ans; }
    }
    {
      unsigned int c1 = adata[2];
      unsigned int c2 = row2[2];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
      if (ans > maxval) { return ans; }
    }
    {
      unsigned int c1 = adata[3];
      unsigned int c2 = row2[3];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
      if (ans > maxval) { return ans; }
    }
    {
      unsigned int c1 = adata[4];
      unsigned int c2 = row2[4];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
      if (ans > maxval) { return ans; }
    }
    {
      unsigned int c1 = adata[5];
      unsigned int c2 = row2[5];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
      if (ans > maxval) { return ans; }
    }
    adata += 6;
  }
  return ans;
}

template<>
int fast_patch_dist<7, 0>(int *adata, BITMAP *b, int bx, int by, int maxval, Params *p) {
  //if (bmask && ((int *) bmask->line[by])[bx]) { return INT_MAX; }
  int ans = 0;
  for (int dy = 0; dy < 7; dy++) {
    int *row2 = ((int *) b->line[by+dy])+bx;
    {
      unsigned int c1 = adata[0];
      unsigned int c2 = row2[0];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
      if (ans > maxval) { return ans; }
    }
    {
      unsigned int c1 = adata[1];
      unsigned int c2 = row2[1];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
      if (ans > maxval) { return ans; }
    }
    {
      unsigned int c1 = adata[2];
      unsigned int c2 = row2[2];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
      if (ans > maxval) { return ans; }
    }
    {
      unsigned int c1 = adata[3];
      unsigned int c2 = row2[3];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
      if (ans > maxval) { return ans; }
    }
    {
      unsigned int c1 = adata[4];
      unsigned int c2 = row2[4];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
      if (ans > maxval) { return ans; }
    }
    {
      unsigned int c1 = adata[5];
      unsigned int c2 = row2[5];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
      if (ans > maxval) { return ans; }
    }
    {
      unsigned int c1 = adata[6];
      unsigned int c2 = row2[6];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
      if (ans > maxval) { return ans; }
    }
    adata += 7;
  }
  return ans;
}

template<>
int fast_patch_nobranch<1, 0>(int *adata, BITMAP *b, int bx, int by, Params *p) {
  //if (bmask && ((int *) bmask->line[by])[bx]) { return INT_MAX; }
  int *row2 = ((int *) b->line[by])+bx;
  {
    unsigned int c1 = adata[0];
    unsigned int c2 = row2[0];
    int dr = (c1&255)-(c2&255);
    int dg = ((c1>>8)&255)-((c2>>8)&255);
    int db = (c1>>16)-(c2>>16);
    return dr*dr+dg*dg+db*db;
  }
}

template<>
int fast_patch_nobranch<2, 0>(int *adata, BITMAP *b, int bx, int by, Params *p) {
  //if (bmask && ((int *) bmask->line[by])[bx]) { return INT_MAX; }
  int ans = 0;
  for (int dy = 0; dy < 2; dy++) {
    int *row2 = ((int *) b->line[by+dy])+bx;
    {
      unsigned int c1 = adata[0];
      unsigned int c2 = row2[0];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
    }
    {
      unsigned int c1 = adata[1];
      unsigned int c2 = row2[1];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
    }
    adata += 2;
  }
  return ans;
}

template<>
int fast_patch_nobranch<3, 0>(int *adata, BITMAP *b, int bx, int by, Params *p) {
  //if (bmask && ((int *) bmask->line[by])[bx]) { return INT_MAX; }
  int ans = 0;
  for (int dy = 0; dy < 3; dy++) {
    int *row2 = ((int *) b->line[by+dy])+bx;
    {
      unsigned int c1 = adata[0];
      unsigned int c2 = row2[0];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
    }
    {
      unsigned int c1 = adata[1];
      unsigned int c2 = row2[1];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
    }
    {
      unsigned int c1 = adata[2];
      unsigned int c2 = row2[2];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
    }
    adata += 3;
  }
  return ans;
}

template<>
int fast_patch_nobranch<4, 0>(int *adata, BITMAP *b, int bx, int by, Params *p) {
  //if (bmask && ((int *) bmask->line[by])[bx]) { return INT_MAX; }
  int ans = 0;
  for (int dy = 0; dy < 4; dy++) {
    int *row2 = ((int *) b->line[by+dy])+bx;
    {
      unsigned int c1 = adata[0];
      unsigned int c2 = row2[0];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
    }
    {
      unsigned int c1 = adata[1];
      unsigned int c2 = row2[1];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
    }
    {
      unsigned int c1 = adata[2];
      unsigned int c2 = row2[2];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
    }
    {
      unsigned int c1 = adata[3];
      unsigned int c2 = row2[3];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
    }
    adata += 4;
  }
  return ans;
}

template<>
int fast_patch_nobranch<5, 0>(int *adata, BITMAP *b, int bx, int by, Params *p) {
  //if (bmask && ((int *) bmask->line[by])[bx]) { return INT_MAX; }
  int ans = 0;
  for (int dy = 0; dy < 5; dy++) {
    int *row2 = ((int *) b->line[by+dy])+bx;
    {
      unsigned int c1 = adata[0];
      unsigned int c2 = row2[0];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
    }
    {
      unsigned int c1 = adata[1];
      unsigned int c2 = row2[1];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
    }
    {
      unsigned int c1 = adata[2];
      unsigned int c2 = row2[2];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
    }
    {
      unsigned int c1 = adata[3];
      unsigned int c2 = row2[3];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
    }
    {
      unsigned int c1 = adata[4];
      unsigned int c2 = row2[4];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
    }
    adata += 5;
  }
  return ans;
}

template<>
int fast_patch_nobranch<6, 0>(int *adata, BITMAP *b, int bx, int by, Params *p) {
  //if (bmask && ((int *) bmask->line[by])[bx]) { return INT_MAX; }
  int ans = 0;
  for (int dy = 0; dy < 6; dy++) {
    int *row2 = ((int *) b->line[by+dy])+bx;
    {
      unsigned int c1 = adata[0];
      unsigned int c2 = row2[0];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
    }
    {
      unsigned int c1 = adata[1];
      unsigned int c2 = row2[1];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
    }
    {
      unsigned int c1 = adata[2];
      unsigned int c2 = row2[2];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
    }
    {
      unsigned int c1 = adata[3];
      unsigned int c2 = row2[3];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
    }
    {
      unsigned int c1 = adata[4];
      unsigned int c2 = row2[4];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
    }
    {
      unsigned int c1 = adata[5];
      unsigned int c2 = row2[5];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
    }
    adata += 6;
  }
  return ans;
}

template<>
int fast_patch_nobranch<7, 0>(int *adata, BITMAP *b, int bx, int by, Params *p) {
  //if (bmask && ((int *) bmask->line[by])[bx]) { return INT_MAX; }
  int ans = 0;
  for (int dy = 0; dy < 7; dy++) {
    int *row2 = ((int *) b->line[by+dy])+bx;
    {
      unsigned int c1 = adata[0];
      unsigned int c2 = row2[0];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
    }
    {
      unsigned int c1 = adata[1];
      unsigned int c2 = row2[1];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
    }
    {
      unsigned int c1 = adata[2];
      unsigned int c2 = row2[2];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
    }
    {
      unsigned int c1 = adata[3];
      unsigned int c2 = row2[3];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
    }
    {
      unsigned int c1 = adata[4];
      unsigned int c2 = row2[4];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
    }
    {
      unsigned int c1 = adata[5];
      unsigned int c2 = row2[5];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
    }
    {
      unsigned int c1 = adata[6];
      unsigned int c2 = row2[6];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
    }
    adata += 7;
  }
  return ans;
}

template<>
int patch_dist_ab<1, 0, 0>(Params *p, BITMAP *a, int ax, int ay, BITMAP *b, int bx, int by, int maxval, RegionMasks *region_masks) {
  int *row1 = ((int *) a->line[ay])+ax;
  int *row2 = ((int *) b->line[by])+bx;
  {
    int c1 = row1[0];
    int c2 = row2[0];
    int dr = (c1&255)-(c2&255);
    int dg = ((c1>>8)&255)-((c2>>8)&255);
    int db = (c1>>16)-(c2>>16);
    return dr*dr+dg*dg+db*db;
  }
}

template<>
int patch_dist_ab<2, 0, 0>(Params *p, BITMAP *a, int ax, int ay, BITMAP *b, int bx, int by, int maxval, RegionMasks *region_masks) {
  int ans = 0;
  for (int dy = 0; dy < 2; dy++) {
    int *row1 = ((int *) a->line[ay+dy])+ax;
    int *row2 = ((int *) b->line[by+dy])+bx;
    {
      int c1 = row1[0];
      int c2 = row2[0];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
      if (ans > maxval) { return ans; }
    }
    {
      int c1 = row1[1];
      int c2 = row2[1];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
      if (ans > maxval) { return ans; }
    }
  }
  return ans;
}

template<>
int patch_dist_ab<3, 0, 0>(Params *p, BITMAP *a, int ax, int ay, BITMAP *b, int bx, int by, int maxval, RegionMasks *region_masks) {
  int ans = 0;
  for (int dy = 0; dy < 3; dy++) {
    int *row1 = ((int *) a->line[ay+dy])+ax;
    int *row2 = ((int *) b->line[by+dy])+bx;
    {
      int c1 = row1[0];
      int c2 = row2[0];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
      if (ans > maxval) { return ans; }
    }
    {
      int c1 = row1[1];
      int c2 = row2[1];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
      if (ans > maxval) { return ans; }
    }
    {
      int c1 = row1[2];
      int c2 = row2[2];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
      if (ans > maxval) { return ans; }
    }
  }
  return ans;
}

template<>
int patch_dist_ab<4, 0, 0>(Params *p, BITMAP *a, int ax, int ay, BITMAP *b, int bx, int by, int maxval, RegionMasks *region_masks) {
  int ans = 0;
  for (int dy = 0; dy < 4; dy++) {
    int *row1 = ((int *) a->line[ay+dy])+ax;
    int *row2 = ((int *) b->line[by+dy])+bx;
    {
      int c1 = row1[0];
      int c2 = row2[0];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
      if (ans > maxval) { return ans; }
    }
    {
      int c1 = row1[1];
      int c2 = row2[1];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
      if (ans > maxval) { return ans; }
    }
    {
      int c1 = row1[2];
      int c2 = row2[2];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
      if (ans > maxval) { return ans; }
    }
    {
      int c1 = row1[3];
      int c2 = row2[3];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
      if (ans > maxval) { return ans; }
    }
  }
  return ans;
}

template<>
int patch_dist_ab<5, 0, 0>(Params *p, BITMAP *a, int ax, int ay, BITMAP *b, int bx, int by, int maxval, RegionMasks *region_masks) {
  int ans = 0;
  for (int dy = 0; dy < 5; dy++) {
    int *row1 = ((int *) a->line[ay+dy])+ax;
    int *row2 = ((int *) b->line[by+dy])+bx;
    {
      int c1 = row1[0];
      int c2 = row2[0];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
      if (ans > maxval) { return ans; }
    }
    {
      int c1 = row1[1];
      int c2 = row2[1];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
      if (ans > maxval) { return ans; }
    }
    {
      int c1 = row1[2];
      int c2 = row2[2];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
      if (ans > maxval) { return ans; }
    }
    {
      int c1 = row1[3];
      int c2 = row2[3];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
      if (ans > maxval) { return ans; }
    }
    {
      int c1 = row1[4];
      int c2 = row2[4];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
      if (ans > maxval) { return ans; }
    }
  }
  return ans;
}

template<>
int patch_dist_ab<6, 0, 0>(Params *p, BITMAP *a, int ax, int ay, BITMAP *b, int bx, int by, int maxval, RegionMasks *region_masks) {
  int ans = 0;
  for (int dy = 0; dy < 6; dy++) {
    int *row1 = ((int *) a->line[ay+dy])+ax;
    int *row2 = ((int *) b->line[by+dy])+bx;
    {
      int c1 = row1[0];
      int c2 = row2[0];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
      if (ans > maxval) { return ans; }
    }
    {
      int c1 = row1[1];
      int c2 = row2[1];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
      if (ans > maxval) { return ans; }
    }
    {
      int c1 = row1[2];
      int c2 = row2[2];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
      if (ans > maxval) { return ans; }
    }
    {
      int c1 = row1[3];
      int c2 = row2[3];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
      if (ans > maxval) { return ans; }
    }
    {
      int c1 = row1[4];
      int c2 = row2[4];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
      if (ans > maxval) { return ans; }
    }
    {
      int c1 = row1[5];
      int c2 = row2[5];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
      if (ans > maxval) { return ans; }
    }
  }
  return ans;
}

template<>
int patch_dist_ab<7, 0, 0>(Params *p, BITMAP *a, int ax, int ay, BITMAP *b, int bx, int by, int maxval, RegionMasks *region_masks) {
  int ans = 0;
  for (int dy = 0; dy < 7; dy++) {
    int *row1 = ((int *) a->line[ay+dy])+ax;
    int *row2 = ((int *) b->line[by+dy])+bx;
    {
      int c1 = row1[0];
      int c2 = row2[0];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
      if (ans > maxval) { return ans; }
    }
    {
      int c1 = row1[1];
      int c2 = row2[1];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
      if (ans > maxval) { return ans; }
    }
    {
      int c1 = row1[2];
      int c2 = row2[2];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
      if (ans > maxval) { return ans; }
    }
    {
      int c1 = row1[3];
      int c2 = row2[3];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
      if (ans > maxval) { return ans; }
    }
    {
      int c1 = row1[4];
      int c2 = row2[4];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
      if (ans > maxval) { return ans; }
    }
    {
      int c1 = row1[5];
      int c2 = row2[5];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
      if (ans > maxval) { return ans; }
    }
    {
      int c1 = row1[6];
      int c2 = row2[6];
      int dr = (c1&255)-(c2&255);
      int dg = ((c1>>8)&255)-((c2>>8)&255);
      int db = (c1>>16)-(c2>>16);
      ans += dr*dr+dg*dg+db*db;
      if (ans > maxval) { return ans; }
    }
  }
  return ans;
}
