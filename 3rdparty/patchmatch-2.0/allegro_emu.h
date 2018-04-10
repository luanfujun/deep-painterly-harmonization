
#ifndef _allegro_emu_h
#define _allegro_emu_h

struct BITMAP {
  int w, h;
  unsigned char **line;
  unsigned char *data;
  BITMAP(int ww = 1, int hh = 1): w(ww), h(hh) {}
};

inline int _getpixel32(BITMAP *a, int x, int y) { return ((int *) a->line[y])[x]; }
inline void _putpixel32(BITMAP *a, int x, int y, int c) { ((int *) a->line[y])[x] = c; }

inline int getr32(int c) { return c&255; }
inline int getg32(int c) { return (c>>8)&255; }
inline int getb32(int c) { return (c>>16)&255; }

BITMAP *create_bitmap(int w, int h);
void blit(BITMAP *a, BITMAP *b, int ax, int ay, int bx, int by, int w, int h);
void destroy_bitmap(BITMAP *bmp);
typedef int fixed;
fixed fixmul(fixed a, fixed b);
void clear(BITMAP *bmp);
void clear_to_color(BITMAP *bmp, int c);

/*
unsigned makecol(int r, int g, int b);
int bitmap_mask_color(BITMAP *bmp);
*/

int bitmap_color_depth(BITMAP *bmp);
BITMAP *create_bitmap_ex(int depth, int w, int h);

#endif
