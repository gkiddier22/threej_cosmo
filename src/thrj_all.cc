#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <experimental/simd>
#include <omp.h>
#include <cmath>
#include <algorithm>

using namespace std;

namespace stdx=std::experimental;
using stdx::native_simd;

constexpr double pi=3.141592653589793238462643383279502884197;
constexpr double inv_4pi=0.25/pi;

using Tsimd = native_simd<double>;
constexpr size_t vlen = Tsimd::size();

template<typename Tsimd> inline Tsimd loadu(const typename Tsimd::value_type *ptr)
  { return Tsimd(ptr, stdx::element_aligned_tag()); }

inline static Tsimd sqr(Tsimd arg) { return arg*arg; }

template<typename Tsimd> inline Tsimd sqrt(Tsimd val)
  {
  Tsimd res;
  for (size_t i=0; i<vlen; ++i) res[i] = sqrt(val[i]);
  return res;
  }

int main(int argc, const char *argv[]) {
  if (argc < 6) {
    fprintf(stderr, "Usage: %s <input_file> <output_file> <lmax> <pol> (TT/EE/TE/EB) <write_output (yes/no)>\n", argv[0]);
    return 1;
  }
  int lmax = atoi(argv[3]);
  string pol = argv[4];
  if ((pol!="TT")&&(pol!="EE")&&(pol!="TE")&&(pol!="EB")) {
    fprintf(stderr, "Error: pol must be 'TT', 'EE', 'TE', or 'EB'.\n");
    return 1;
  }
  string write_output = argv[5];
constexpr size_t safety = 4*vlen;
  vector<double> wl(2*lmax+1+safety);
  FILE *fpw = fopen(argv[1], "rb");
  int nlr = fread(wl.data(), sizeof(double), 2*lmax+1, fpw);
  if (nlr != (2*lmax+1)) {
    printf("Error reading power spectrum.\n");
    fclose(fpw);
    return -1;
  }

  fclose(fpw);
  // safety zone for vectorization
  for (int i=2*lmax+1; i<wl.size(); ++i)
    wl[i] = 0;
  for (size_t i=0; i<wl.size(); ++i)
    wl[i] *= (2*i+1) * inv_4pi;

  vector<double> g(2*lmax+1+safety);
  // factor needed inside the critical loop: 1/(g[i]*(2i+1))
  vector<double> fct(2*lmax+1+safety);
  double gcur = 1.;
  for (size_t i=0; i<g.size(); ++i, gcur*=(i-0.5)/i)
    {
    g[i] = sqrt(gcur);
    fct[i] = sqrt(1./(gcur*(2*i+1)));
    }

  vector<double> m((lmax+1)*(lmax+1));
  double start = omp_get_wtime();

  if (pol=="TT") {
    #pragma omp parallel for schedule(dynamic)
    for (int el1=0; el1<=lmax; ++el1) {
      for (int el2=el1; el2<=lmax; el2+=vlen) {
        Tsimd j3_sum = 0.;
        int el3min = el2-el1;
        int el3max = el2+el1;
        int maxidx = min(el3max, int(nlr-1));

        for (int el3=el3min, ofs=0; el3<=maxidx; el3+=2, ++ofs) {
          Tsimd threej_000 = loadu<Tsimd>(&fct[el2+ofs]) * loadu<Tsimd>(&g[el2-el1+ofs]) * g[ofs] * g[el1-ofs];
          j3_sum += loadu<Tsimd>(&wl[el3])*(threej_000*threej_000);
        }
        for (size_t i=0; i<vlen; ++i) {
          if (el2+i<=lmax) {
            m[el1*(lmax+1)+el2+i] = j3_sum[i] * (2*(el2+i)+1);
            m[(el2+i)*(lmax+1)+el1] = j3_sum[i] * (2*el1+1);
          }
        }
      }
    }
  }

  if (pol=="EE") {
    Tsimd iota;
    for (size_t i=0; i<vlen; ++i) iota[i] = double(i);
    #pragma omp parallel for schedule(dynamic)
    for (int el1=2; el1<=lmax; ++el1) {
      for (int el2=el1; el2<=lmax; el2+=vlen) {
        Tsimd j3_sum = 0.;
        int el3min = el2-el1;
        int el3max = el2+el1;
        int maxidx = min(el3max, int(nlr-1));

        Tsimd el2v = double(el2) + iota;

        Tsimd x_eta_sq = Tsimd(1.)/((el1-1.)*(el1+2.)*(el2v-1.)*el2v);
  
        Tsimd lmbda = sqrt(el1*(el1+1.)*(el2v+1.)*(el2v+2.));
        Tsimd x_lmbda_sq = Tsimd(1.)/(lmbda*lmbda);
        Tsimd lmbda_t1 = lmbda *(1.+2./el1);
        Tsimd lmbda_t2 = lmbda/(((el1+1.)*(el2v+1.)*el1));
        Tsimd x_eta = sqrt(x_eta_sq);

        for (int el3=el3min, ofs=0; el3<=maxidx; el3+=2, ++ofs) {
          Tsimd threej_000 = loadu<Tsimd>(&fct[el2+ofs]) * loadu<Tsimd>(&g[el2-el1+ofs]) * g[ofs] * g[el1-ofs];
          Tsimd Jpmp = 2.*ofs;  // actually scalar
          Tsimd J = Jpmp+2*el2v;
          Tsimd Jmpp = J-2*el1;
          Tsimd el3v = el2v-el1 + Jpmp;
          Tsimd Jppm = J-2*el3v;  // actually scalar

          auto lmbda2 = (J+2.) * (Jppm+1.);
          auto A = lmbda_t1 - lmbda2*lmbda_t2;
          auto B_sq = 0.25 * x_lmbda_sq * lmbda2 * (Jmpp+1.) * (Jmpp+2.) * (J+3.) * (Jppm+2.)  * Jpmp * (Jpmp-1.);
          auto threej_000_2 = loadu<Tsimd>(&fct[el2+1+ofs]) * loadu<Tsimd>(&g[el2+1-el1+ofs]) * g[ofs-1] * g[el1+1-ofs];

          auto tmp1 = A*threej_000;
          auto tmp2 = -sqrt(B_sq)*threej_000_2;

          auto threej_0p2m2 = (tmp1+tmp2)*x_eta;
          j3_sum += loadu<Tsimd>(&wl[el3])*threej_0p2m2*threej_0p2m2;
        }
        for (size_t i=0; i<vlen; ++i) {
          if (el2+i<=lmax) {
            m[el1*(lmax+1)+el2+i] = j3_sum[i] * (2*(el2+i)+1);
            m[(el2+i)*(lmax+1)+el1] = j3_sum[i] * (2*el1+1);
          }
        }
      }
    }
  }

  if (pol=="TE") {
    Tsimd iota;
    for (size_t i=0; i<vlen; ++i) iota[i] = double(i);
    #pragma omp parallel for schedule(dynamic)
    for (int el1=2; el1<=lmax; ++el1) {
      for (int el2=el1; el2<=lmax; el2+=vlen) {
        Tsimd j3_sum = 0.;
        int el3min = el2-el1;
        int el3max = el2+el1;
        int maxidx = min(el3max, int(nlr-1));

        Tsimd el2v = double(el2) + iota;

        Tsimd x_eta_sq = Tsimd(1.)/((el1-1.)*(el1+2.)*(el2v-1.)*el2v);
  
        Tsimd lmbda = sqrt(el1*(el1+1.)*(el2v+1.)*(el2v+2.));
        Tsimd x_lmbda_sq = Tsimd(1.)/(lmbda*lmbda);
        Tsimd lmbda_t1 = lmbda *(1.+2./el1);
        Tsimd lmbda_t2 = lmbda/(((el1+1.)*(el2v+1.)*el1));
        Tsimd x_eta = sqrt(x_eta_sq);

        for (int el3=el3min, ofs=0; el3<=maxidx; el3+=2, ++ofs) {
          Tsimd threej_000 = loadu<Tsimd>(&fct[el2+ofs]) * loadu<Tsimd>(&g[el2-el1+ofs]) * g[ofs] * g[el1-ofs];
          Tsimd Jpmp = 2.*ofs;  // actually scalar
          Tsimd J = Jpmp+2*el2v;
          Tsimd Jmpp = J-2*el1;
          Tsimd el3v = el2v-el1 + Jpmp;
          Tsimd Jppm = J-2*el3v;  // actually scalar

          auto lmbda2 = (J+2.) * (Jppm+1.);
          auto A = lmbda_t1 - lmbda2*lmbda_t2;
          auto B_sq = 0.25 * x_lmbda_sq * lmbda2 * (Jmpp+1.) * (Jmpp+2.) * (J+3.) * (Jppm+2.)  * Jpmp * (Jpmp-1.);
          auto threej_000_2 = loadu<Tsimd>(&fct[el2+1+ofs]) * loadu<Tsimd>(&g[el2+1-el1+ofs]) * g[ofs-1] * g[el1+1-ofs];

          auto tmp1 = A*threej_000;
          auto tmp2 = -sqrt(B_sq)*threej_000_2;

          auto threej_0p2m2 = (tmp1+tmp2)*x_eta;
          j3_sum += loadu<Tsimd>(&wl[el3])*threej_000*threej_0p2m2;
        }
        for (size_t i=0; i<vlen; ++i) {
          if (el2+i<=lmax) {
            m[el1*(lmax+1)+el2+i] = j3_sum[i] * (2*(el2+i)+1);
            m[(el2+i)*(lmax+1)+el1] = j3_sum[i] * (2*el1+1);
          }
        }
      }
    }
  }

  if (pol=="EB") {
    Tsimd iota;
    for (size_t i=0; i<vlen; ++i) iota[i] = double(i);
    #pragma omp parallel for schedule(dynamic)
    for (int el1=2; el1<=lmax; ++el1) {
      for (int el2=el1; el2<=lmax; el2+=vlen) {
        Tsimd j3_sum = 0.;
        int el3min = el2-el1;
        int el3max = el2+el1;
        int maxidx = min(el3max, int(nlr-1));

        Tsimd el2v = double(el2) + iota;

        Tsimd x_eta_sq = Tsimd(1.)/((el1-1.)*(el1+2.)*(el2v-1.)*el2v);
  
        Tsimd termsumsq = (el1+1.) + 2. + 1./(el1+1.);
        Tsimd ebtmp1 = Tsimd(1.)/((el1+1.)*(el2v+2.));
        Tsimd ebtmp2 = Tsimd(1.)/(el1*(el2v+1.));

        for (int el3=el3min+1, ofs=0; el3<=maxidx; el3+=2, ++ofs) {
          // Note the "+1" here, since we are shifted, and J will be odd
          auto el3v = double(el3)+iota;
          auto J = el3v+el1+el2v;
          auto Jmpp = J-2*el1;
          auto Jpmp = J-2*el2v;
          auto Jppm = J-2*el3v;
          auto Lambda_sq = (J+2.)*(Jppm+1.)*(Jmpp+1.)*Jpmp;
  
          auto t1 = loadu<Tsimd>(&g[el2+1-el1+ofs])*g[ofs];
          auto t2 = sqr(loadu<Tsimd>(&fct[el2+1+ofs]) * g[el1-ofs]*t1);
          auto t3 = sqr(loadu<Tsimd>(&fct[el2+2+ofs]) * g[el1+1-ofs]*t1);
          auto term25_sq = t2*(el2v+2.)* termsumsq;
          auto term3_sq = t3*0.25*(J+3.)*(J+4.)*(Jppm+2.)*(Jppm+3.)*ebtmp1;
          auto tmp = term25_sq+term3_sq-2.*sqrt(term25_sq*term3_sq);
          j3_sum += loadu<Tsimd>(&wl[el3])*tmp*Lambda_sq*x_eta_sq*ebtmp2;
        }
        for (size_t i=0; i<vlen; ++i) {
          if (el2+i<=lmax) {
            m[el1*(lmax+1)+el2+i] = j3_sum[i] * (2*(el2+i)+1);
            m[(el2+i)*(lmax+1)+el1] = j3_sum[i] * (2*el1+1);
          }
        }
      }
    }
  }


  printf("Wall-clock time: %g seconds\n", omp_get_wtime() - start);

  if (write_output=="yes") {
    FILE *fpm = fopen(argv[2], "wb");
    if (!fpm) {
      perror("Error opening output file");
      return 1;
    }
    fwrite(m.data(), sizeof(double), (lmax+1)*(lmax+1), fpm);
    fclose(fpm);
    printf("Matrix written to %s.\n", argv[2]);
  }
}
