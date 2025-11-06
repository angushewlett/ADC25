//////////////////////////////
// 
// Audio Developer Conference 2025
// "Perfect oscillators in less than one clock cycle"
// Angus F. Hewlett - Nov. 2025
//
// Requirements:
// Intel or AMD x64 CPU with AVX2 
// Windows API (for measurement harness only).
// 
// clang CLIO_avx256.cpp -mfma -mavx2 -O3 -ffp-contract=fast
//
//////////////////////////////

#include <iostream>
#include <string>
#include <cstdint>
#include <chrono>
#include <array>
#include <algorithm>
#include <thread>
#include <immintrin.h>
#include <cpuid.h>
#include <windows.h>

using namespace std;

//////////////////////////////
// 
// Absolute bare-bones SIMD wrapper providing a few essential operations.
// This implementation is for AVX2/256, if you can't translate this to NEON or AVX512 in half an hour, what are you doing here?
// 
//////////////////////////////
class simd
{
    public:

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
        vf ipart = _mm256_cvttps_epi32(floor_x.v);
        ipart.v = _mm256_slli_epi32(ipart.v, 23);
        vf fpart = x - floor_x;
        // cubic approximation of 2^x in [0, 1]
        vf expfpart = ((fpart *  ((fpart * ((fpart * 0.07944023841053369f) + 0.224494337302845f)) + 0.6960656421638072f))+1.f);
        return ipart * expfpart;
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

__m256 TickOsc4(Oscillator& a, Oscillator& b, Oscillator& c, Oscillator& d)
{
    simd::vf result = a.Tick() + b.Tick() + c.Tick() + d.Tick();
    return result.v;
}


__m256 TickOsc8(Oscillator& a, Oscillator& b, Oscillator& c, Oscillator& d,Oscillator& e, Oscillator& f, Oscillator& g, Oscillator& h)
{
    simd::vf result = a.Tick() + b.Tick() + c.Tick() + d.Tick() + e.Tick() + f.Tick() + g.Tick() + h.Tick();
    return result.v;
}



double measureEffectiveFrequencyMhz(std::chrono::milliseconds dur = std::chrono::milliseconds(50))
{
    ULONGLONG startCycles = 0, endCycles = 0;
    QueryThreadCycleTime(GetCurrentThread(), &startCycles);
    auto t0 = std::chrono::high_resolution_clock::now();
    std::this_thread::sleep_for(dur);
    auto t1 = std::chrono::high_resolution_clock::now();
    QueryThreadCycleTime(GetCurrentThread(), &endCycles);
    auto elapsedNs = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    double cycles = double(endCycles - startCycles);
    return (cycles / elapsedNs) * 1000.0; // MHz
}


void cpuid() {
    unsigned int level = 0;
    unsigned int eax, ebx, ecx, edx;

    string mModelName;

    for(int i=0x80000002; i<0x80000005; ++i) {
        __get_cpuid(i, &eax, &ebx, &ecx, &edx);
        mModelName += string((const char*)&eax, 4);
        mModelName += string((const char*)&ebx, 4);
        mModelName += string((const char*)&ecx, 4);
        mModelName += string((const char*)&edx, 4);
    }  
    cout << mModelName << " @ " << measureEffectiveFrequencyMhz() << " MHz\n";
}


int main()
{
    cpuid();
    Oscillator a, b, c, d, e, f, g, h;
    b.m_inc = 0.1f;
    c.m_inc = 0.11f;

    auto start = std::chrono::high_resolution_clock::now();
    std::array<int, 1000> liTimes;
    std::array<unsigned long long, 1000> liCycles;

    simd::vf accum = 0.f;

    // 1000 runs
    for (int i = 0; i < 1000; i++)
    {
	unsigned long long t0, t1;
        auto start1 = std::chrono::high_resolution_clock::now();
        QueryThreadCycleTime(GetCurrentThread(), &t0);
        // Each sub-run is 1000 cycles to give the timer something big enough to measure
        for (int j = 0; j < 1000; j++)
            accum.v += TickOsc4(a,b,c,d);
        accum.v *= 0.001f; // reset it to small: carry the dependency but prevent blowups
        QueryThreadCycleTime(GetCurrentThread(), &t1);
        auto finish1 = std::chrono::high_resolution_clock::now();
        liTimes[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(finish1-start1).count();
        liCycles[i] = t1 - t0;
    }

    auto finish = std::chrono::high_resolution_clock::now();

    std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count() << "ns\n";
    std::sort(liTimes.begin(), liTimes.end());
    std::cout << liTimes[250] << "ns\n";
    std::sort(liCycles.begin(), liCycles.end());
    std::cout << liCycles[250] << "cycles\n";

    // This is just to create a dependency so the compiler can't opt out of doing the actual processing!
    std::cout << "Output of sum of all oscillators @ end of process: " << accum.horizontal_add() << "\n";
    return 0;

}
