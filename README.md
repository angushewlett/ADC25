This repo was created to support my Audio Developer Conference 2025 talk.

It contains a C++ implementation of the CLIO sawtooth branchfree oscillator for AVX/256 which will compile with clang/Windows,
and an AVX/512 version which will compile with MSVC/Windows or clang/Windows.

The code is generally portable but I've had to use some platform-specific shims for measuring CPU cycle counts for this demo.

Other implementations and waveforms are available.

angushewlett@gmail.com is me.
