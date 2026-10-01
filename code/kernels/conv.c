/* Original teaching reference: NCHW x OIHW grouped convolution.
   FP32, positive sizes/strides/dilations, symmetric non-negative padding.
   Caller owns allocation. Returns zero, or -1 for an invalid contract.
   Size products must fit size_t; callers must check allocation products. */
#include <stddef.h>
int conv2d_nchw(const float *x, const float *w, const float *bias,
                float *out, int B, int C, int H, int W, int O,
                int KH, int KW, int SH, int SW, int PH, int PW,
                int DH, int DW, int groups) {
  if (!x || !w || !out || B<=0 || C<=0 || H<=0 || W<=0 || O<=0 ||
      KH<=0 || KW<=0 || SH<=0 || SW<=0 || DH<=0 || DW<=0 ||
      PH<0 || PW<0 || groups<=0 || C%groups || O%groups) return -1;
  const long long eh=(long long)DH*(KH-1)+1;
  const long long ew=(long long)DW*(KW-1)+1;
  const long long nh=(long long)H+2LL*PH-eh;
  const long long nw=(long long)W+2LL*PW-ew;
  if (nh<0 || nw<0) return -1;
  const size_t OH=(size_t)(nh/SH+1), OW=(size_t)(nw/SW+1);
  const int CG=C/groups, OG=O/groups;
  for (int b=0;b<B;b++) for (int o=0;o<O;o++)
    for (size_t y=0;y<OH;y++) for (size_t z=0;z<OW;z++) {
      float acc=bias ? bias[o] : 0.0f;
      const int group=o/OG;
      for (int ci=0;ci<CG;ci++) for (int ky=0;ky<KH;ky++)
        for (int kx=0;kx<KW;kx++) {
          const long long iy=(long long)y*SH-PH+(long long)ky*DH;
          const long long ix=(long long)z*SW-PW+(long long)kx*DW;
          if (iy>=0 && iy<H && ix>=0 && ix<W) {
            const size_t xi=(((size_t)b*C+group*CG+ci)*H+(size_t)iy)*W+(size_t)ix;
            const size_t wi=(((size_t)o*CG+ci)*KH+ky)*KW+kx;
            acc += x[xi]*w[wi];
          }
        }
      out[(((size_t)b*O+o)*OH+y)*OW+z]=acc;
    }
  return 0;
}
