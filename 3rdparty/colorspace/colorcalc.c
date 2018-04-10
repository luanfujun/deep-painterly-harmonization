/**
 * @file colorcalc.c
 * @author Pascal Getreuer 2010 <getreuer@gmail.com>
 *
 * This is a small command line program to demonstrate colorspace.c.  The
 * program accepts an sRGB color as input and transforms it to all the spaces
 * supported by colorspace.c.
 *
 * ==Usage==
 * The syntax for the program is
 *    colorcalc R G B
 * where R, G, B are values between 0 and 1 specifying a color in the gamma-
 * corrected sRGB color space.  For example,
 *    colorcalc 0.5 0.85 0.61
 *
 * ==Compiling Instructions==
 * Compile the files colorcalc.c and colorspace.c with an ANSI C compiler.  The
 * program is compiled with GCC by
 *    gcc colorcalc.c colorspace.c -lm -o colorcalc
 */
#include <stdio.h>
#include <stdlib.h>
#include "colorspace.h"


static void HelpMessage();


int main(int argc, char *argv[])
{
	num R, G, B;
	num D[3];
	
	
	if(argc != 4)
	{
		HelpMessage();
		return 1;
	}
	
	/* Read the input sRGB values */
	R = atof(argv[1]);
	G = atof(argv[2]);
	B = atof(argv[3]);
	
	if(!(0 <= R && R <= 1 && 0 <= G && G <= 1 && 0 <= B && B <= 1))
		printf("\nWarning: Input sRGB values should be between 0 and 1.\n\n");
	
	printf("sRGB         R':%8.3f   G':%8.3f   B':%8.3f\n", R, G, B);
	
	/* Transform sRGB to Y'CbCr */
	Rgb2Ycbcr(&D[0], &D[1], &D[2], R, G, B);
	printf("Y'CbCr       Y':%8.3f   Cb:%8.3f   Cr:%8.3f\n", D[0], D[1], D[2]);
	
	/* Transform sRGB to JPEG-Y'CbCr */
	Rgb2Jpegycbcr(&D[0], &D[1], &D[2], R, G, B);
	printf("JPEG-Y'CbCr  Y':%8.3f   Cb:%8.3f   Cr:%8.3f\n", D[0], D[1], D[2]);
	
	/* Transform sRGB to Y'PbPr */
	Rgb2Ypbpr(&D[0], &D[1], &D[2], R, G, B);
	printf("Y'PbPr       Y':%8.3f   Pb:%8.3f   Pr:%8.3f\n", D[0], D[1], D[2]);
	
	/* Transform sRGB to Y'DbDr */
	Rgb2Ydbdr(&D[0], &D[1], &D[2], R, G, B);
	printf("Y'DbDr       Y':%8.3f   Db:%8.3f   Dr:%8.3f\n", D[0], D[1], D[2]);
	
	/* Transform sRGB to Y'UV */
	Rgb2Yuv(&D[0], &D[1], &D[2], R, G, B);
	printf("Y'UV         Y':%8.3f    U:%8.3f    V:%8.3f\n", D[0], D[1], D[2]);
	
	/* Transform sRGB to Y'IQ */
	Rgb2Yiq(&D[0], &D[1], &D[2], R, G, B);
	printf("Y'IQ         Y':%8.3f    I:%8.3f    Q:%8.3f\n", D[0], D[1], D[2]);
	
	/* Transform sRGB to HSV */
	Rgb2Hsv(&D[0], &D[1], &D[2], R, G, B);
	printf("HSV           H:%8.3f    S:%8.3f    V:%8.3f\n", D[0], D[1], D[2]);
	
	/* Transform sRGB to HSL */
	Rgb2Hsl(&D[0], &D[1], &D[2], R, G, B);
	printf("HSL           H:%8.3f    S:%8.3f    L:%8.3f\n", D[0], D[1], D[2]);
	
	/* Transform sRGB to HSI */
	Rgb2Hsi(&D[0], &D[1], &D[2], R, G, B);
	printf("HSI           H:%8.3f    S:%8.3f    I:%8.3f\n", D[0], D[1], D[2]);
	
	/* Transform sRGB to CIE XYZ */
	Rgb2Xyz(&D[0], &D[1], &D[2], R, G, B);
	printf("XYZ           X:%8.3f    Y:%8.3f    Z:%8.3f\n", D[0], D[1], D[2]);

	/* Transform sRGB to CIE L*a*b */
	Rgb2Lab(&D[0], &D[1], &D[2], R, G, B);
	printf("L*a*b*       L*:%8.3f   a*:%8.3f   b*:%8.3f\n", D[0], D[1], D[2]);
	
	/* Transform sRGB to CIE L*u*v* */
	Rgb2Luv(&D[0], &D[1], &D[2], R, G, B);
	printf("L*u*v*       L*:%8.3f   u*:%8.3f   v*:%8.3f\n", D[0], D[1], D[2]);
	
	/* Transform sRGB to CIE L*C*H* */
	Rgb2Lch(&D[0], &D[1], &D[2], R, G, B);
	printf("L*C*H*       L*:%8.3f   C*:%8.3f   H*:%8.3f\n", D[0], D[1], D[2]);
	
	/* Transform sRGB to CIE CAT02 LMS */
	Rgb2Cat02lms(&D[0], &D[1], &D[2], R, G, B);
	printf("CAT02 LMS     L:%8.3f    M:%8.3f    S:%8.3f\n", D[0], D[1], D[2]);
	
	return 0;
}


/** @brief Print program help message */
static void HelpMessage()
{
	printf("Color calculator, P. Getreuer 2010\n");
	printf("\nSyntax: colorcalc R G B\n\n");
	printf("where R, G, B are values between 0 and 1 specifying a color in\n");
	printf("the gamma-corrected sRGB color space.  The color is transformed\n");
	printf("to all spaces supported by colorspace.c.\n\n");
	printf("Example: colorcalc 0.5 0.85 0.61\n");
}
