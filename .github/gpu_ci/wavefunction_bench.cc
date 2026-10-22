// Timing (and output) of the external wavefunction routines of a madmatrix standalone output.
//
// Built by madmatrix_helicity_bench.py in a P* directory of an `output standalone`, with the
// flags of the backend (the generic %.o rule of madmatrix.mk), against the HelAmps_<model>.h
// of that output: once with the helicity as an int, branched on (MadGraph7 before the
// branch-free helicities), once with the helicity as a fptype_amp_sv, one per SIMD lane or GPU
// thread, entering as truth values (after). wavefunction_bench_config.h, written by the
// script, says which (BENCH_PER_LANE_HELICITY) and names the header (BENCH_HELAMPS).
//
// Every routine is called on nevt random momenta, for every helicity it takes:
//   uniform  the events of a SIMD vector (GPU warp) share one helicity, as in the
//            good-helicity loop of sigmaKin;
//   mixed    every event (SIMD lane, GPU thread) has its own helicity, in a random
//            order, as with UMAMI_IN_HELICITY_INDEX: the per-lane helicity on SIMD exists only with
//            BENCH_PER_LANE_HELICITY; scalar and GPU code can always do it (one int per
//            event or thread), but branch on it.
// Output: one line "RESULT <routine> <mode> <ns per call and event> <checksum>" per routine
// and mode, the best of ntrial timings of nrep sweeps over all events and helicities.
// With a dump file, the wavefunctions of every routine, helicity and event (the first ndump
// events, at most 4096), in both modes: double[routine][mode][ihel][ievt][nw6][2] (re, im),
// for the comparison of the two builds.
//
// Usage: wavefunction_bench_<backend>.exe [nevt [nrep [ntrial [dumpfile]]]]

#include "wavefunction_bench_config.h"

#include "mgOnGpuConfig.h"

#include BENCH_HELAMPS
#include "MemoryAccessWavefunctions.h"

#ifdef MGONGPUCPP_GPUIMPL
#include "GpuRuntime.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace madmatrix;

namespace
{
  constexpr int np4 = 4;
  constexpr int nw6 = ALOHAOBJ::nw6;

  // the momentum slots, each with its own kind of momenta
  enum Slot { MASSIVE_F = 0, MASSIVE_V, MASSLESS, PLUS_Z, MINUS_Z, NSLOT };
  constexpr fptype_amp mFermion = 173.;
  constexpr fptype_amp mVector = 91.188;

  // routines: name, slot, mass, helicities
  enum Routine { IXXXXX_M = 0, IXXXXX_0, OXXXXX_M, OXXXXX_0, VXXXXX_M, VXXXXX_0, SXXXXX,
                 IPZXXX, IMZXXX, IXZXXX, OPZXXX, OMZXXX, OXZXXX, NROUTINE };
  const char* const routineNames[NROUTINE] = { "ixxxxx_massive", "ixxxxx_massless", "oxxxxx_massive",
                                               "oxxxxx_massless", "vxxxxx_massive", "vxxxxx_massless",
                                               "sxxxxx", "ipzxxx", "imzxxx", "ixzxxx", "opzxxx", "omzxxx",
                                               "oxzxxx" };
  __host__ __device__ inline int nhelOf( const int routine )
  {
    return routine == SXXXXX ? 1 : routine == VXXXXX_M ? 3 : 2;
  }
  __host__ __device__ inline int helOf( const int routine, const int ihel )
  {
    return routine == SXXXXX ? 0 : routine == VXXXXX_M ? ihel - 1 : 2 * ihel - 1;
  }
  enum Mode { UNIFORM = 0, MIXED, NMODE };
  const char* const modeNames[NMODE] = { "uniform", "mixed" };

  // Momenta: an AOSOA[npagV][NSLOT][np4][neppV] in C++ (one SIMD vector per component), and
  // an SOA[NSLOT][np4][nevt] on the GPU, each thread on one event
  struct BenchMomenta
  {
#ifdef MGONGPUCPP_GPUIMPL
    static __device__ inline const fptype_momenta&
    kernelAccessIp4IparConst( const fptype_momenta* buffer, const int ip4, const int ipar )
    {
      const int nevt = gridDim.x * blockDim.x;
      const int ievt = blockDim.x * blockIdx.x + threadIdx.x;
      return buffer[( ipar * np4 + ip4 ) * nevt + ievt];
    }
#else
    static inline const fptype_momenta_sv&
    kernelAccessIp4IparConst( const fptype_momenta* page, const int ip4, const int ipar )
    {
      return reinterpret_cast<const fptype_momenta_sv*>( page )[ipar * np4 + ip4];
    }
#endif
  };

#ifdef MGONGPUCPP_GPUIMPL
  using W_ACCESS = DeviceAccessWavefunctions;
#else
  using W_ACCESS = HostAccessWavefunctions;
#endif

  // The helicity argument of the routines: one per lane (after), or an int (before)
#if BENCH_PER_LANE_HELICITY
  typedef fptype_amp_sv helarg_t;
#else
  typedef int helarg_t;
#endif

  // One call of a routine on the momenta of one SIMD page (C++) or one event (GPU)
  template<class M_ACCESS>
  __host__ __device__ inline void
  callRoutine( const int routine, const fptype_momenta* momenta, const helarg_t hel, ALOHAOBJ& obj )
  {
    switch( routine )
    {
      case IXXXXX_M: ixxxxx<M_ACCESS, W_ACCESS>( momenta, mFermion, hel, +1, 0, obj, MASSIVE_F ); break;
      case IXXXXX_0: ixxxxx<M_ACCESS, W_ACCESS>( momenta, 0., hel, +1, 0, obj, MASSLESS ); break;
      case OXXXXX_M: oxxxxx<M_ACCESS, W_ACCESS>( momenta, mFermion, hel, +1, 0, obj, MASSIVE_F ); break;
      case OXXXXX_0: oxxxxx<M_ACCESS, W_ACCESS>( momenta, 0., hel, +1, 0, obj, MASSLESS ); break;
      case VXXXXX_M: vxxxxx<M_ACCESS, W_ACCESS>( momenta, mVector, hel, +1, 0, obj, MASSIVE_V ); break;
      case VXXXXX_0: vxxxxx<M_ACCESS, W_ACCESS>( momenta, 0., hel, +1, 0, obj, MASSLESS ); break;
      case SXXXXX: sxxxxx<M_ACCESS, W_ACCESS>( momenta, +1, 0, obj, MASSIVE_V ); break;
      case IPZXXX: ipzxxx<M_ACCESS, W_ACCESS>( momenta, hel, +1, 0, obj, PLUS_Z ); break;
      case IMZXXX: imzxxx<M_ACCESS, W_ACCESS>( momenta, hel, +1, 0, obj, MINUS_Z ); break;
      case IXZXXX: ixzxxx<M_ACCESS, W_ACCESS>( momenta, hel, +1, 0, obj, MASSLESS ); break;
      case OPZXXX: opzxxx<M_ACCESS, W_ACCESS>( momenta, hel, +1, 0, obj, PLUS_Z ); break;
      case OMZXXX: omzxxx<M_ACCESS, W_ACCESS>( momenta, hel, +1, 0, obj, MINUS_Z ); break;
      case OXZXXX: oxzxxx<M_ACCESS, W_ACCESS>( momenta, hel, +1, 0, obj, MASSLESS ); break;
    }
  }

  // the helicity index of an event at step istep of a sweep over the nhel helicities: istep
  // for all (uniform), or shifted by the random offset of the event, 0 to 5 (mixed), so that
  // the sequence of the helicities of the events cannot be predicted
  __host__ __device__ inline int ihelOf( const int mode, const int istep, const int offset, const int nhel )
  {
    return mode == UNIFORM ? istep : ( istep + offset ) % nhel;
  }

#ifdef MGONGPUCPP_GPUIMPL
  // nrep sweeps over the helicities, each thread on its event; the sum of the wavefunctions
  // goes to sink (so that nothing is optimised away), and with dump != nullptr the
  // wavefunctions of the last sweep to dump[ihel][ievt][nw6][2]
  __global__ void
  benchKernel( const int routine, const int mode, const int nrep, const int zero, const fptype_momenta* momenta,
               const int* offsets, fptype_amp* sink, fptype_amp* dump, const int ndump )
  {
    const int ievt = blockDim.x * blockIdx.x + threadIdx.x;
    const int nhel = nhelOf( routine );
    fptype_momenta pvec[np4];
    cxtype_amp w[nw6];
    ALOHAOBJ obj{ pvec, w };
    fptype_amp sum = 0;
    for( int irep = 0; irep < nrep; irep++ )
      for( int istep = 0; istep < nhel; istep++ )
      {
        const int ihel = ihelOf( mode, istep, offsets[ievt], nhel );
        const int hel = helOf( routine, ihel );
        callRoutine<BenchMomenta>( routine, momenta + irep * zero, (helarg_t)hel, obj );
        for( int iw = 0; iw < nw6; iw++ ) sum += cxreal( w[iw] ) + cximag( w[iw] );
        if( dump && irep == nrep - 1 && ievt < ndump )
          for( int iw = 0; iw < nw6; iw++ )
          {
            dump[( ( ihel * ndump + ievt ) * nw6 + iw ) * 2] = cxreal( w[iw] );
            dump[( ( ihel * ndump + ievt ) * nw6 + iw ) * 2 + 1] = cximag( w[iw] );
          }
      }
    sink[ievt] = sum;
  }
#else
  // the same in C++, one SIMD page (neppV events) at a time
  fptype_amp
  benchHost( const int routine, const int mode, const int nrep, const int zero, const int nevt,
             const fptype_momenta* momenta, const int* offsets, fptype_amp* dump, const int ndump )
  {
    const int nhel = nhelOf( routine );
    const int npagV = nevt / neppV;
    fptype_momenta_sv pvec[np4];
    cxtype_amp_sv w[nw6];
    ALOHAOBJ obj{ pvec, w };
    fptype_amp_sv sum{};
    for( int irep = 0; irep < nrep; irep++ )
      for( int ipagV = 0; ipagV < npagV; ipagV++ )
      {
        const fptype_momenta* page = momenta + ipagV * NSLOT * np4 * neppV + irep * zero;
        for( int istep = 0; istep < nhel; istep++ )
        {
#if BENCH_PER_LANE_HELICITY
          helarg_t hel{};
#ifdef MGONGPU_CPPSIMD
          for( int ieppV = 0; ieppV < neppV; ieppV++ )
            hel[ieppV] = helOf( routine, ihelOf( mode, istep, offsets[ipagV * neppV + ieppV], nhel ) );
#else
          hel = helOf( routine, ihelOf( mode, istep, offsets[ipagV], nhel ) );
#endif
#else
          // an int for the whole page: per event in scalar code only (checked in main)
          const helarg_t hel = helOf( routine, ihelOf( mode, istep, offsets[ipagV], nhel ) );
#endif
          callRoutine<BenchMomenta>( routine, page, hel, obj );
          for( int iw = 0; iw < nw6; iw++ ) sum += cxreal( w[iw] ) + cximag( w[iw] );
          if( dump && irep == nrep - 1 && ipagV * neppV < ndump )
            for( int ieppV = 0; ieppV < neppV; ieppV++ )
            {
              const int ievt = ipagV * neppV + ieppV;
              const int ihel = ihelOf( mode, istep, offsets[ievt], nhel );
              for( int iw = 0; iw < nw6; iw++ )
              {
#ifdef MGONGPU_CPPSIMD
                const fptype_amp re = cxreal( w[iw] )[ieppV], im = cximag( w[iw] )[ieppV];
#else
                const fptype_amp re = cxreal( w[iw] ), im = cximag( w[iw] );
#endif
                dump[( ( ihel * ndump + ievt ) * nw6 + iw ) * 2] = re;
                dump[( ( ihel * ndump + ievt ) * nw6 + iw ) * 2 + 1] = im;
              }
            }
        }
      }
    fptype_amp total = 0;
#ifdef MGONGPU_CPPSIMD
    for( int ieppV = 0; ieppV < neppV; ieppV++ ) total += sum[ieppV];
#else
    total = sum;
#endif
    return total;
  }
#endif

  // a random 4-momentum of the given mass, |p| log-uniform in [1, 1000] GeV; axis +1/-1 along
  // the z axis (massless), 0 isotropic
  void randomMomentum( std::mt19937_64& rng, const double mass, const int axis, double p[np4] )
  {
    std::uniform_real_distribution<double> uni( 0., 1. );
    const double pabs = std::exp( std::log( 1000. ) * uni( rng ) );
    double dir[3] = { 0., 0., double( axis ) };
    if( axis == 0 )
    {
      const double cth = 2. * uni( rng ) - 1., sth = std::sqrt( 1. - cth * cth ), phi = 2. * M_PI * uni( rng );
      dir[0] = sth * std::cos( phi );
      dir[1] = sth * std::sin( phi );
      dir[2] = cth;
    }
    p[0] = std::sqrt( pabs * pabs + mass * mass );
    for( int i = 0; i < 3; i++ ) p[i + 1] = pabs * dir[i];
  }
}

int
main( int argc, char** argv )
{
  int nevt = argc > 1 ? std::atoi( argv[1] ) : 4096;
  const int nrep = argc > 2 ? std::atoi( argv[2] ) : 100;
  const int ntrial = argc > 3 ? std::atoi( argv[3] ) : 5;
  const char* dumpFile = argc > 4 ? argv[4] : nullptr;
  // 0, which the compiler cannot know: the momenta of every sweep are at an offset irep * zero,
  // so that the sweeps, all on the same momenta, cannot be hoisted out of the timing loop
  volatile int zeroOpaque = 0;
  const int zero = zeroOpaque;
  volatile double sink = 0;
#ifdef MGONGPUCPP_GPUIMPL
  constexpr int nthr = 256;
  nevt = std::max( nthr, nevt / nthr * nthr );
#else
  nevt = std::max( (int)neppV, nevt / neppV * neppV );
#endif
  const int ndump = std::min( nevt, 4096 ); // a multiple of neppV and of nthr, as nevt
  std::printf( "nevt=%d nrep=%d ntrial=%d neppV=%d per_lane_helicity=%d nw6=%d ndump=%d\n", nevt, nrep, ntrial,
               (int)neppV, (int)BENCH_PER_LANE_HELICITY, nw6, ndump );

  // the momenta, in the layout of BenchMomenta
  std::mt19937_64 rng( 20261022 );
  const double masses[NSLOT] = { mFermion, mVector, 0., 0., 0. };
  const int axes[NSLOT] = { 0, 0, 0, +1, -1 };
  const size_t nmom = (size_t)nevt * NSLOT * np4;
  fptype_momenta* hstMomenta = static_cast<fptype_momenta*>( std::aligned_alloc( 64, ( nmom * sizeof( fptype_momenta ) + 63 ) / 64 * 64 ) );
  for( int ievt = 0; ievt < nevt; ievt++ )
    for( int islot = 0; islot < NSLOT; islot++ )
    {
      double p[np4];
      randomMomentum( rng, masses[islot], axes[islot], p );
      for( int ip4 = 0; ip4 < np4; ip4++ )
#ifdef MGONGPUCPP_GPUIMPL
        hstMomenta[( islot * np4 + ip4 ) * nevt + ievt] = p[ip4];
#else
        hstMomenta[( ( ievt / neppV ) * NSLOT * np4 + islot * np4 + ip4 ) * neppV + ievt % neppV] = p[ip4];
#endif
    }

  // the random helicity offsets of the events (mixed)
  std::vector<int> hstOffsets( nevt );
  for( int& offset : hstOffsets ) offset = std::uniform_int_distribution<int>( 0, 5 )( rng );

  // the dump of one routine and mode: [ihel][ievt][nw6][2]
  const size_t ndumpval = (size_t)3 * ndump * nw6 * 2;
  std::vector<double> hstDumpAll;
  std::vector<fptype_amp> hstDump( ndumpval );
#ifdef MGONGPUCPP_GPUIMPL
  fptype_momenta* devMomenta;
  fptype_amp *devSink, *devDump;
  int* devOffsets;
  gpuMalloc( &devMomenta, nmom * sizeof( fptype_momenta ) );
  gpuMalloc( &devSink, nevt * sizeof( fptype_amp ) );
  gpuMalloc( &devDump, ndumpval * sizeof( fptype_amp ) );
  gpuMemcpy( devMomenta, hstMomenta, nmom * sizeof( fptype_momenta ), gpuMemcpyHostToDevice );
  gpuMalloc( &devOffsets, nevt * sizeof( int ) );
  gpuMemcpy( devOffsets, hstOffsets.data(), nevt * sizeof( int ), gpuMemcpyHostToDevice );
  std::vector<fptype_amp> hstSink( nevt );
#endif

  for( int routine = 0; routine < NROUTINE; routine++ )
    for( int mode = 0; mode < NMODE; mode++ )
    {
      if( routine == SXXXXX && mode == MIXED ) continue; // no helicity
      std::fill( hstDump.begin(), hstDump.end(), 0 );
#if defined MGONGPU_CPPSIMD && !BENCH_PER_LANE_HELICITY
      if( mode == MIXED )
      {
        // one int helicity for all the lanes: no helicity per event
        std::printf( "RESULT %s %s n/a n/a\n", routineNames[routine], modeNames[mode] );
        if( dumpFile ) hstDumpAll.insert( hstDumpAll.end(), ndumpval, std::nan( "" ) );
        continue;
      }
#endif
      double best = 1e300, checksum = 0;
      for( int itrial = 0; itrial <= ntrial; itrial++ ) // trial 0 is a warm-up, and fills the dump
      {
        const int reps = itrial == 0 ? 1 : nrep;
        const auto start = std::chrono::steady_clock::now();
#ifdef MGONGPUCPP_GPUIMPL
        gpuLaunchKernel( benchKernel, nevt / nthr, nthr, routine, mode, reps, zero, devMomenta, devOffsets, devSink,
                         itrial == 0 ? devDump : nullptr, ndump );
        checkGpu( gpuPeekAtLastError() );
        checkGpu( gpuDeviceSynchronize() );
#else
        const double sum = benchHost( routine, mode, reps, zero, nevt, hstMomenta, hstOffsets.data(),
                                      itrial == 0 ? hstDump.data() : nullptr, ndump );
#endif
        const double seconds = std::chrono::duration<double>( std::chrono::steady_clock::now() - start ).count();
#ifdef MGONGPUCPP_GPUIMPL
        gpuMemcpy( hstSink.data(), devSink, nevt * sizeof( fptype_amp ), gpuMemcpyDeviceToHost );
        double sum = 0;
        for( const fptype_amp s : hstSink ) sum += s;
        if( itrial == 0 ) gpuMemcpy( hstDump.data(), devDump, ndumpval * sizeof( fptype_amp ), gpuMemcpyDeviceToHost );
#endif
        sink = sink + sum; // every sweep is used
        if( itrial == 0 )
          checksum = sum;
        else
          best = std::min( best, seconds / ( (double)reps * nevt * nhelOf( routine ) ) );
      }
      std::printf( "RESULT %s %s %.4f %.12e\n", routineNames[routine], modeNames[mode], best * 1e9, checksum );
      if( dumpFile ) hstDumpAll.insert( hstDumpAll.end(), hstDump.begin(), hstDump.end() );
    }
  if( dumpFile )
  {
    FILE* f = std::fopen( dumpFile, "wb" );
    if( !f || std::fwrite( hstDumpAll.data(), sizeof( double ), hstDumpAll.size(), f ) != hstDumpAll.size() )
    {
      std::fprintf( stderr, "cannot write %s\n", dumpFile );
      return 1;
    }
    std::fclose( f );
  }
#ifdef MGONGPUCPP_GPUIMPL
  gpuFree( devMomenta );
  gpuFree( devSink );
  gpuFree( devDump );
  gpuFree( devOffsets );
#endif
  std::free( hstMomenta );
  return 0;
}
