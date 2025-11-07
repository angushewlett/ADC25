//////////////////////////////
// 
// Audio Developer Conference 2025
// "Perfect oscillators in less than one clock cycle"
// Angus F. Hewlett - Nov. 2025
//
// Requirements:
// Intel or AMD x64 CPU with AVX or AVX512, or ARMv8 with NEON
//
// clang CLIO.cpp -mfma -mavx512f -O3 -ffp-contract=fast
// clang CLIO.cpp -mfma -march=haswell -O3 -ffp-contract=fast
// cl.exe CLIO.cpp /O2 /arch:AVX512 /fp:fast /DWIN32
// clang CLIO.cpp -O3 -ffp-contract=fast -lstdc++
//
//////////////////////////////

#include <iostream>
#include <string>
#include <cstdint>
#include <chrono>
#include <array>
#include <algorithm>
#include <thread>

// Intrinsics - NEON or AVX
#if __aarch64__ || __ARM64EC__ || __ARM64__ || _M_ARM64
#include <arm_neon.h>
#else
#include <immintrin.h>
#endif

// Measurements
#if WIN32
#include <intrin.h>
#include <windows.h>
#else
#if __aarch64__ || __ARM64EC__ || __ARM64__ || _M_ARM64
#if defined(__APPLE__)
#include "M1_Cycles.h"
#endif
#else
#include <cpuid.h>
#endif
#endif

using namespace std;

//////////////////////////////
// 
// Absolute bare-bones SIMD wrapper providing a few essential operations.
//
// This implementation is for AVX2/512, if you can't translate this to NEON or AVX512 in half an hour, what are you even doing here?
//
// Full implementation available on request, this is enough to run this demo.
//
//////////////////////////////

#if __aarch64__ || __ARM64EC__ || __ARM64__ || _M_ARM64

class simd_neon
{
    public:
    typedef float32x4_t reg_f;
    static constexpr int voices = 4;

    class vf // vector floating point
    {
        public:
        vf (const float32x4_t& other)
        {
            v = other;
        }

        vf (const vf& other)
        {
            v = other.v;
        }

        vf (float other)
        {
            v = vdupq_n_f32(other);
        }

        vf() {}

        vf operator+(const vf& other) const
        {
            vf result;
            result.v = vaddq_f32(v, other.v);
            return result;
        }

        vf operator-(const vf& other) const
        {
            vf result;
            result.v = vsubq_f32(v, other.v);
            return result;
        }

        vf operator*(const vf& other) const
        {
            vf result;
            result.v = vmulq_f32(v, other.v);
            return result;
        }

        vf operator|(const vf& other) const
        {
            vf result;
            result.v = vorrq_u32(v, other.v);
            return result;
        }

        vf operator+=(const vf& other)
        {
            v = vaddq_f32(v, other.v);
            return *this;
        }

        vf operator-=(const vf& other)
        {
            v = vsubq_f32(v, other.v);
            return *this;
        }

        vf operator*=(const vf& other)
        {
            v = vmulq_f32(v, other.v);
            return *this;
        }

        float horizontal_add()
        {
            return vaddvq_f32(v);
        }

        friend inline vf operator+(float lhs, const vf& rhs)
        {
            vf x = lhs;
            return x + rhs;
        }

        friend inline vf operator-(float lhs, const vf& rhs)
        {
            vf x = lhs;
            return x - rhs;
        }

        float32x4_t v;
    };


    class vi
    {
        public:
        int32x4_t v;
    };

    static inline vf ffloor(vf x)
    {
        x.v = vrndmq_f32(x.v);
        return x;
    }

    static inline vf fsignbit(vf x)
    {
        return vandq_u32(x.v, vdupq_n_u32(0x80000000));
    }

    static inline vf fabs(vf x)
    {
        return vabsq_f32(x.v);
    }

    static inline vf fmax(vf a, vf b)
    {
        return vmaxq_f32(a.v,b.v);
    }

    // 2nd order polynomial approximation of 2^x
    static inline vf fpow2_poly2_nc(vf x)
    {
        vf floor_x = ffloor(x);
        vi ipart;
        ipart.v = vcvtq_s32_f32(floor_x.v);
        ipart.v = vshlq_n_u32(ipart.v, 23);
        vf fpart = x - floor_x;
        // cubic approximation of 2^x in [0, 1]
        vf expfpart = ((fpart *  ((fpart * ((fpart * 0.07944023841053369f) + 0.224494337302845f)) + 0.6960656421638072f))+1.f);
        vf ipartf;
        ipartf.v = vreinterpretq_f32_s32(ipart.v);
        return  ipartf* expfpart;

    }

    // divide using fast reciprocal and a single Newton-Raphson step to improve accuracy
    static inline vf fdiv12_nr(vf a, vf b)
    {
        vf inverse = vrecpeq_f32(b.v);
        vf r1 = inverse * (2.f - (b * inverse));
        // Final division: a / b = a * r1
        return a * r1;
    }

};

using simd = simd_neon;


#else

class simd_avx512
{
    public:

    typedef __m512 reg_f;
    static constexpr int voices = 16;

    class vf // vector floating point
    {
        public:
        vf (const __m512& other)
        {
            v = other;
        }

        vf (const vf& other)
        {
            v = other.v;
        }        

        vf (float other)
        {
            v = _mm512_set1_ps(other);
        }

        vf() {}

        vf operator+(const vf& other) const
        {
            vf result;
            result.v = _mm512_add_ps(v, other.v);
            return result;            
        }

        vf operator-(const vf& other) const
        {
            vf result;
            result.v = _mm512_sub_ps(v, other.v);
            return result;            
        }        

        vf operator*(const vf& other) const
        {
            vf result;
            result.v = _mm512_mul_ps(v, other.v);
            return result;            
        }    

        vf operator|(const vf& other) const
        {
            vf result;
            result.v = _mm512_or_ps(v, other.v);
            return result;            
        }           

        vf operator+=(const vf& other)
        {
            v = _mm512_add_ps(v, other.v);
            return *this;
        }

        vf operator-=(const vf& other)
        {
            v = _mm512_sub_ps(v, other.v);
            return *this;            
        }        

        vf operator*=(const vf& other)
        {
            v = _mm512_mul_ps(v, other.v);
            return *this;            
        }          



        float avx512_sum16(__m512 a)
		{
		    __m512 tmp = _mm512_add_ps(a,_mm512_shuffle_f32x4(a,a,_MM_SHUFFLE(0,0,3,2)));
		    __m128 r = _mm512_castps512_ps128(_mm512_add_ps(tmp,_mm512_shuffle_f32x4(tmp,tmp,_MM_SHUFFLE(0,0,0,1))));
		    r = _mm_hadd_ps(r,r);
		    return _mm_cvtss_f32(_mm_hadd_ps(r,r));
		}

        float horizontal_add()
        {
            return avx512_sum16(v);
        }

        friend inline vf operator+(float lhs, const vf& rhs)
        {
            vf x = lhs;
            return x + rhs;
        }        

        friend inline vf operator-(float lhs, const vf& rhs)
        {
            vf x = lhs;
            return x - rhs;
        }            

        __m512 v;
    };

    class vi
    {
        public:
        __m512i v;

    };

    static inline vf ffloor(vf x)
    {
        x.v = _mm512_roundscale_ps(x.v, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
        return x;
    }

    static inline vf fsignbit(vf x)
    {
        return _mm512_and_ps(x.v, _mm512_castsi512_ps(_mm512_set1_epi32(0x80000000)));
    }    

    static inline vf fabs(vf x)
    {
        return _mm512_andnot_ps(_mm512_set1_ps(-0.0f), x.v);
    }        

    static inline vf fmax(vf a, vf b)
    {
        return _mm512_max_ps(a.v,b.v);
    }    

    // 2nd order polynomial approximation of 2^x
    static inline vf fpow2_poly2_nc(vf x)
    {
        vf floor_x = ffloor(x);
        vi ipart;
        ipart.v = _mm512_cvttps_epi32(floor_x.v);
        ipart.v = _mm512_slli_epi32(ipart.v, 23);
        vf fpart = x - floor_x;
        // cubic approximation of 2^x in [0, 1]
        vf expfpart = ((fpart *  ((fpart * ((fpart * 0.07944023841053369f) + 0.224494337302845f)) + 0.6960656421638072f))+1.f);
        vf ipartf;
        ipartf.v = _mm512_castsi512_ps(ipart.v);
        return  ipartf* expfpart;
    }       

    // divide using fast reciprocal and a single Newton-Raphson step to improve accuracy
    static inline vf fdiv12_nr(vf a, vf b)
    {
        vf inverse;
        inverse.v = _mm512_rcp14_ps(b.v);
        vf muls = (b * inverse * inverse);
        inverse = (inverse + inverse - muls);
        return inverse * a;
    }       
};

class simd_avx256
{
    public:
    typedef __m256 reg_f;
    static constexpr int voices = 8;

    class vf // vector floating point
    {
        public:
        vf (const __m256& other)
        {
            v = other;
        }

        vf (const vf& other)
        {
            v = other.v;
        }        

        vf (float other)
        {
            v = _mm256_set1_ps(other);
        }

        vf() {}

        vf operator+(const vf& other) const
        {
            vf result;
            result.v = _mm256_add_ps(v, other.v);
            return result;            
        }

        vf operator-(const vf& other) const
        {
            vf result;
            result.v = _mm256_sub_ps(v, other.v);
            return result;            
        }        

        vf operator*(const vf& other) const
        {
            vf result;
            result.v = _mm256_mul_ps(v, other.v);
            return result;            
        }    

        vf operator|(const vf& other) const
        {
            vf result;
            result.v = _mm256_or_ps(v, other.v);
            return result;            
        }           

        vf operator+=(const vf& other)
        {
            v = _mm256_add_ps(v, other.v);
            return *this;
        }

        vf operator-=(const vf& other)
        {
            v = _mm256_sub_ps(v, other.v);
            return *this;            
        }        

        vf operator*=(const vf& other)
        {
            v = _mm256_mul_ps(v, other.v);
            return *this;            
        }          

        float horizontal_add()
        {
            __m256 vsum = v;
            __m256 vsum2 = _mm256_permute2f128_ps(vsum , vsum , 1);
            vsum = _mm256_add_ps(vsum, vsum2);
            vsum = _mm256_hadd_ps(vsum, vsum);
            vsum = _mm256_hadd_ps(vsum, vsum);
            return _mm256_cvtss_f32(vsum);
        }

        friend inline vf operator+(float lhs, const vf& rhs)
        {
            vf x = lhs;
            return x + rhs;
        }        

        friend inline vf operator-(float lhs, const vf& rhs)
        {
            vf x = lhs;
            return x - rhs;
        }            

        __m256 v;
    };


    class vi
    {
        public:
        __m256i v;

    };

    static inline vf ffloor(vf x)
    {
        x.v = _mm256_round_ps(x.v, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
        return x;
    }

    static inline vf fsignbit(vf x)
    {
        return _mm256_and_ps(x.v, _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000)));
    }    

    static inline vf fabs(vf x)
    {
        return _mm256_andnot_ps(_mm256_set1_ps(-0.0f), x.v);
    }        

    static inline vf fmax(vf a, vf b)
    {
        return _mm256_max_ps(a.v,b.v);
    }    

    // 2nd order polynomial approximation of 2^x
    static inline vf fpow2_poly2_nc(vf x)
    {
        vf floor_x = ffloor(x);
        vi ipart;
        ipart.v = _mm256_cvttps_epi32(floor_x.v);
        ipart.v = _mm256_slli_epi32(ipart.v, 23);
        vf fpart = x - floor_x;
        // cubic approximation of 2^x in [0, 1]
        vf expfpart = ((fpart *  ((fpart * ((fpart * 0.07944023841053369f) + 0.224494337302845f)) + 0.6960656421638072f))+1.f);
        vf ipartf;
        ipartf.v = _mm256_castsi256_ps(ipart.v);
        return  ipartf* expfpart;

    }       

    // divide using fast reciprocal and a single Newton-Raphson step to improve accuracy
    static inline vf fdiv12_nr(vf a, vf b)
    {
        vf inverse;
        inverse.v = _mm256_rcp_ps(b.v);
        vf muls = (b * inverse * inverse);
        inverse = (inverse + inverse - muls);
        return inverse * a;
    }     

};

using simd = simd_avx256;
#endif



class Oscillator
{
    public:
        using vf = simd::vf;
        vf m_phs = 0.f;
        vf m_inc = 0.001f;

        // Test scaling constants for our window. These are frequency dependent and should be computed per-block.
        static constexpr float k_window_scaling_20 = 20.f;
        static constexpr float k_window_size_m1 = -0.8f;
        static constexpr float k_window_size_inv_20 = 100.f;

        simd::vf Tick()
        {
            
            // Increment phase, round, subtract: 3 instructions
            m_phs += m_inc;
            m_phs -= simd::ffloor(m_phs);  // phase has exceeded transition point by (m_phs - 1)
            
            // Scale to +-1: 1 instruction (fused mul-sub)
            vf saw_naive = ((m_phs * 2) - 1.f);
            
            // Get the sign bit: 1 instruction
            vf saw_sign = simd::fsignbit(saw_naive);
            
            // SAW ONLY version            
            // Generate unsigned piecewise window (at ph=0 & ph=1) scaled up by 20 - absolute, offset, scale, clip: 4 instructions
            vf saw_piecewise_window_scaled = simd::fmax(0.f, simd::fabs(saw_naive) + k_window_size_m1) * k_window_size_inv_20;
            
            // Logistic function denominator: 11 instructions: subtract, 9 for the pow2, add to 1
            vf logistic_denominator = 1.f + simd::fpow2_poly2_nc(k_window_scaling_20 - saw_piecewise_window_scaled);
     
            // Divide, then OR with sign bit: 5 instructions for div, 1 for OR: 6 instructions.
            vf error_func = simd::fdiv12_nr(2.f, logistic_denominator) | saw_sign;
                  
            // subtract error from naive: 1 instruction
            vf saw_out = saw_naive - error_func;
            return saw_out;
        }
};

//
// Test functions to tick 1, 4 or 8 oscillators at a time.
// (Compiler will interleave operations across the four oscs).
//

simd::reg_f TickOsc1(Oscillator& a)
{
    simd::vf result = a.Tick();
    return result.v;
}

simd::reg_f TickOsc2(Oscillator& a, Oscillator& b)
{
    simd::vf result = a.Tick() + b.Tick();
    return result.v;
}

simd::reg_f TickOsc4(Oscillator& a, Oscillator& b, Oscillator& c, Oscillator& d)
{
    simd::vf result = a.Tick() + b.Tick() + c.Tick() + d.Tick();
    return result.v;
}

simd::reg_f TickOsc8(Oscillator& a, Oscillator& b, Oscillator& c, Oscillator& d,Oscillator& e, Oscillator& f, Oscillator& g, Oscillator& h)
{
    simd::vf result = a.Tick() + b.Tick() + c.Tick() + d.Tick() + e.Tick() + f.Tick() + g.Tick() + h.Tick();
    return result.v;
}

simd::reg_f TickOsc16(Oscillator& a0, Oscillator& b0, Oscillator& c0, Oscillator& d0, Oscillator& e0, Oscillator& f0, Oscillator& g0, Oscillator& h0,
                Oscillator& a1, Oscillator& b1, Oscillator& c1, Oscillator& d1, Oscillator& e1, Oscillator& f1, Oscillator& g1, Oscillator& h1
)
{
    simd::vf result = a0.Tick() + b0.Tick() + c0.Tick() + d0.Tick() + e0.Tick() + f0.Tick() + g0.Tick() + h0.Tick()
        +
            a1.Tick() + b1.Tick() + c1.Tick() + d1.Tick() + e1.Tick() + f1.Tick() + g1.Tick() + h1.Tick();
    return result.v;
}

#if WIN32
unsigned long long GetCycleTime()
{
    unsigned long long result;
    QueryThreadCycleTime(GetCurrentThread(), &result);
    return result;
}
#else
#if __APPLE__ && (__aarch64__ || __ARM64EC__ || __ARM64__ || _M_ARM64) // macOS, ARM
static AppleEvents a;
unsigned long long GetCycleTime()
{
    performance_counters p = a.get_counters();
    return (unsigned long long)p.cycles;
}
#else // Linux, intel
unsigned long long GetCycleTime()
{
    return __rdtsc();
}
#endif
#endif


double measureEffectiveFrequencyMhz(std::chrono::milliseconds dur = std::chrono::milliseconds(50))
{
    unsigned long long startCycles = 0, endCycles = 0;
    startCycles = GetCycleTime();
    // QueryThreadCycleTime(GetCurrentThread(), &startCycles);
    auto t0 = std::chrono::high_resolution_clock::now();
    std::this_thread::sleep_for(dur);
    auto t1 = std::chrono::high_resolution_clock::now();
    endCycles = GetCycleTime();
    auto elapsedNs = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    double cycles = double(endCycles - startCycles);
    return (cycles / elapsedNs) * 1000.0; // MHz
}


void cpuid() 
{
    string mModelName;
    for(int i=0x80000002; i<0x80000005; ++i) 
	{
        #if __aarch64__ || __ARM64EC__ || __ARM64__ || _M_ARM64
        mModelName = "Generic ARM";
        #else
        #if WIN32
        int regs[4]; //eax, ebx, ecx, edx;
        __cpuid(regs, i);
        #else
        unsigned int regs[4];
        __get_cpuid(i, &regs[0], &regs[1], &regs[2], &regs[3]);
        #endif
        mModelName += string((const char*)&regs[0], 16);
        #endif
    }
    cout << mModelName << " @ " << measureEffectiveFrequencyMhz() << " MHz\n";
}


int main()
{
    #if __APPLE__ && (__aarch64__ || __ARM64EC__ || __ARM64__ || _M_ARM64)
    a.setup_performance_counters();
    #endif
    cpuid();
    Oscillator o[16];
    for (int i = 0; i < 16; i++) o[i].m_inc = 0.1f + 0.01f*i;

    auto start = std::chrono::high_resolution_clock::now();
    std::array<int, 1000> liTimes;
    std::array<unsigned long long, 1000> liCycles;

    simd::vf accum = 0.f;

	unsigned long long t0, t1;

    // 1000 runs

    for (int i = 0; i < 1000; i++)
    {
        t0 = GetCycleTime();
        for (int j = 0; j < 1000; j++) accum += TickOsc1(o[0]);
        accum *= 0.001f; // reset it to small: carry the dependency but prevent blowups
        t1 = GetCycleTime();
        liCycles[i] = t1 - t0;
    }
    std::sort(liCycles.begin(), liCycles.end());
    std::cout << simd::voices*2 << " voices; "  << "1 ops x 1000 ticks: " << liCycles[250] << " cycles\n";

    for (int i = 0; i < 1000; i++)
    {
        t0 = GetCycleTime();
        for (int j = 0; j < 1000; j++) accum += TickOsc2(o[0],o[1]);
        accum *= 0.001f; // reset it to small: carry the dependency but prevent blowups
        t1 = GetCycleTime();
        liCycles[i] = t1 - t0;
    }
    std::sort(liCycles.begin(), liCycles.end());
    std::cout << simd::voices*2 << " voices; " << "2 ops x 1000 ticks: " << liCycles[250] << " cycles\n";

    for (int i = 0; i < 1000; i++)
    {
        t0 = GetCycleTime();
        for (int j = 0; j < 1000; j++) accum += TickOsc4(o[0],o[1],o[2],o[3]);
        accum *= 0.001f; // reset it to small: carry the dependency but prevent blowups
        t1 = GetCycleTime();
        liCycles[i] = t1 - t0;
    }
    std::sort(liCycles.begin(), liCycles.end());
    std::cout << simd::voices*4 << " voices; " << "4 ops x 1000 ticks: " << liCycles[250] << " cycles\n";

    for (int i = 0; i < 1000; i++)
    {
        t0 = GetCycleTime();
        for (int j = 0; j < 1000; j++) accum += TickOsc8(o[0],o[1],o[2],o[3],o[4],o[5],o[6],o[7]);
        accum *= 0.001f; // reset it to small: carry the dependency but prevent blowups
        t1 = GetCycleTime();
        liCycles[i] = t1 - t0;
    }
    std::sort(liCycles.begin(), liCycles.end());
    std::cout << simd::voices*8 << " voices; " << "8 ops x 1000 ticks: " << liCycles[250] << " cycles\n";

    for (int i = 0; i < 1000; i++)
    {
        t0 = GetCycleTime();
        for (int j = 0; j < 1000; j++) accum += TickOsc16(o[0],o[1],o[2],o[3],o[4],o[5],o[6],o[7],o[8],o[9],o[10],o[11],o[12],o[13],o[14],o[15]);
        accum *= 0.001f; // reset it to small: carry the dependency but prevent blowups
        t1 = GetCycleTime();
        liCycles[i] = t1 - t0;
    }
    std::sort(liCycles.begin(), liCycles.end());
    std::cout << simd::voices*16 << " voices; " << "16 ops x 1000 ticks: " << liCycles[250] << " cycles\n";


    // This is just to create a dependency so the compiler can't opt out of doing the actual processing!
    std::cout << "Output of sum of all oscillators @ end of process: " << accum.horizontal_add() << "\n";
    return 0;
}
