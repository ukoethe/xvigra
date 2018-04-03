/************************************************************************/
/*                                                                      */
/*     Copyright 2017-2018 by Ullrich Koethe                            */
/*                                                                      */
/*    This file is part of the XVIGRA image analysis library.           */
/*                                                                      */
/*    Permission is hereby granted, free of charge, to any person       */
/*    obtaining a copy of this software and associated documentation    */
/*    files (the "Software"), to deal in the Software without           */
/*    restriction, including without limitation the rights to use,      */
/*    copy, modify, merge, publish, distribute, sublicense, and/or      */
/*    sell copies of the Software, and to permit persons to whom the    */
/*    Software is furnished to do so, subject to the following          */
/*    conditions:                                                       */
/*                                                                      */
/*    The above copyright notice and this permission notice shall be    */
/*    included in all copies or substantial portions of the             */
/*    Software.                                                         */
/*                                                                      */
/*    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND    */
/*    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES   */
/*    OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND          */
/*    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT       */
/*    HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,      */
/*    WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING      */
/*    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR     */
/*    OTHER DEALINGS IN THE SOFTWARE.                                   */
/*                                                                      */
/************************************************************************/

#include <benchmark/benchmark.h>
#include <vector>
#include <algorithm>
#include <cstring>
// #include <xvigra/global.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xstrided_view.hpp>
#include <xtensor/xnoalias.hpp>
#include <xvigra/tiny_vector.hpp>

#ifdef XVIGRA_USE_SIMD
#  include <xsimd/xsimd.hpp>
#endif

// #define BENCHMARK_VIGRA

#ifdef BENCHMARK_VIGRA
#  include <vigra/multi_array.hxx>
#endif

// #define BENCHMARK_VIGRA2

#ifdef BENCHMARK_VIGRA2
#  include <vigra2/array_nd.hxx>
#endif

constexpr int SIZE = 1 << 20;

namespace xvigra
{
    // template <class V>
    // void init_memset(benchmark::State& state)
    // {
    //     std::vector<V> data(SIZE);

    //     for (auto _ : state)
    //     {
    //         std::memset(data.data(), 0, SIZE*sizeof(V));
    //         benchmark::DoNotOptimize(data.data());
    //     }
    // }
    // BENCHMARK_TEMPLATE(init_memset, float);

    // template <class V>
    // void init_loop(benchmark::State& state)
    // {
    //     std::vector<V> data(SIZE);

    //     for (auto _ : state)
    //     {
    //         for(int k=0; k<SIZE; ++k)
    //         {
    //             data[k] = 0.0f;
    //         }
    //         benchmark::DoNotOptimize(data.data());
    //     }
    // }
    // BENCHMARK_TEMPLATE(init_loop, float);

    // template <class V>
    // void init_std_fill(benchmark::State& state)
    // {
    //     std::vector<V> data(SIZE);

    //     for (auto _ : state)
    //     {
    //         std::fill(data.begin(), data.end(), 0.0f);
    //         benchmark::DoNotOptimize(data.data());
    //     }
    // }
    // BENCHMARK_TEMPLATE(init_std_fill, float);

    // template <class V>
    // void init_std_copy(benchmark::State& state)
    // {
    //     std::vector<V> data(SIZE), zeros(SIZE, 0.0f);

    //     for (auto _ : state)
    //     {
    //         std::copy(zeros.begin(), zeros.end(), data.begin());
    //         benchmark::DoNotOptimize(zeros.data());
    //         benchmark::DoNotOptimize(data.data());
    //     }
    // }
    // BENCHMARK_TEMPLATE(init_std_copy, float);

    template <class V>
    void xarray_init_std_fill(benchmark::State& state)
    {
        auto data = xt::xarray<V>::from_shape({SIZE});

        for (auto _ : state)
        {
            std::fill(data.begin(), data.end(), V());
            benchmark::DoNotOptimize(data.raw_data());
        }
    }
    BENCHMARK_TEMPLATE(xarray_init_std_fill, float);

    template <class V>
    void xarray_init_assign(benchmark::State& state)
    {
        xt::xarray<V> data = xt::xarray<V>::from_shape({SIZE}),
                      zeros = xt::zeros<V>({SIZE});

        for (auto _ : state)
        {
            data = zeros;
            benchmark::DoNotOptimize(zeros.raw_data());
            benchmark::DoNotOptimize(data.raw_data());
        }
    }
    BENCHMARK_TEMPLATE(xarray_init_assign, float);

    template <class V>
    void xarray_init_zeros(benchmark::State& state)
    {
        auto data = xt::xarray<V>::from_shape({SIZE});

        for (auto _ : state)
        {
            xt::noalias(data) = xt::zeros<V>({SIZE});
            benchmark::DoNotOptimize(data.raw_data());
        }
    }
    BENCHMARK_TEMPLATE(xarray_init_zeros, float);


    template <class V>
    void dynamic_view_init_zeros(benchmark::State& state)
    {
        auto data = xt::xarray<V>::from_shape({SIZE});
        auto view = xt::dynamic_view(data, xt::slice_vector{xt::all()});

        std::cerr << typeid(xt::zeros<V>({SIZE})).name() << "\n";

        for (auto _ : state)
        {
            view = xt::zeros<V>({SIZE});
            // xt::noalias(view) = xt::zeros<V>({SIZE});
            benchmark::DoNotOptimize(data.raw_data());
        }
    }
    BENCHMARK_TEMPLATE(dynamic_view_init_zeros, float);

// #ifdef XVIGRA_USE_SIMD
//     void init_simd_unaligned(benchmark::State& state)
//     {
//         std::vector<float> data(SIZE);

//         for (auto _ : state)
//         {
//             xsimd::batch<float, 8> z(0.0f);
//             float * p = data.data();
//             for(int k=0; k<SIZE; k+=8)
//             {
//                 z.store_unaligned(p + k);
//             }
//             benchmark::DoNotOptimize(data.data());
//         }
//     }
//     BENCHMARK(init_simd_unaligned);

//     void init_simd_aligned(benchmark::State& state)
//     {
//         std::vector<float, xsimd::aligned_allocator<float, XSIMD_DEFAULT_ALIGNMENT>> data(SIZE);

//         for (auto _ : state)
//         {
//             xsimd::batch<float, 8> z(0.0f);
//             float * p = data.data();
//             for(int k=0; k<SIZE; k+=8)
//             {
//                 z.store_aligned(p + k);
//             }
//             benchmark::DoNotOptimize(data.data());
//         }
//     }
//     BENCHMARK(init_simd_aligned);

//     void copy_simd_unaligned(benchmark::State& state)
//     {
//         std::vector<float> data(SIZE), zeros(SIZE, 0.0f);

//         for (auto _ : state)
//         {
//             xsimd::batch<float, 8> d;
//             float * z = zeros.data();
//             float * p = data.data();
//             for(int k=0; k<SIZE; k+=8)
//             {
//                 d.load_unaligned(z + k).store_unaligned(p + k);
//             }
//             benchmark::DoNotOptimize(zeros.data());
//             benchmark::DoNotOptimize(data.data());
//         }
//     }
//     BENCHMARK(copy_simd_unaligned);

//     void copy_simd_aligned(benchmark::State& state)
//     {
//         std::vector<float, xsimd::aligned_allocator<float, XSIMD_DEFAULT_ALIGNMENT>> data(SIZE), zeros(SIZE, 0.0f);

//         for (auto _ : state)
//         {
//             xsimd::batch<float, 8> d;
//             float * z = zeros.data();
//             float * p = data.data();
//             for(int k=0; k<SIZE; k+=8)
//             {
//                 d.load_aligned(z + k).store_aligned(p + k);
//             }
//             benchmark::DoNotOptimize(zeros.data());
//             benchmark::DoNotOptimize(data.data());
//         }
//     }
//     BENCHMARK(copy_simd_aligned);
// #endif

} // namespace xvigra
