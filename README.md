This repo was created to support my Audio Developer Conference 2025 talk.

It contains:
- a C++ implementation of the CLIO sawtooth branchfree oscillator
- - AVX/256 version which will compile with clang/Windows,
- - AVX/512 version which will compile with MSVC/Windows or clang/Windows,
- - combined 256/512 version which will compile on clang/mac|linux, clang/Windows or MSVC/Windows

The code is generally portable but I've had to use some platform-specific shims for measuring CPU cycle counts for this demo.

Other implementations and waveforms are available.

angushewlett@gmail.com is me.
