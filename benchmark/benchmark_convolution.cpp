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
#include <xvigra/separable_convolution.hpp>

namespace xvigra
{
    auto && kernel = averaging_kernel_1d<float>(1);

    template <class V>
    void simple_averaging_2d_no_simd(benchmark::State& state)
    {
        array_nd<V, 2> data(shape_t<2>{2000,3000}),
                             result(data.shape());

        for (auto _ : state)
        {
            slow_separable_convolution(data, result, kernel, convolution_options().use_simd(false));
            benchmark::DoNotOptimize(result.data());
        }
    }

    BENCHMARK_TEMPLATE(simple_averaging_2d_no_simd, float);

    template <class V>
    void simple_averaging_2d_simd(benchmark::State& state)
    {
        array_nd<V, 2> data(shape_t<2>{2000,3000}),
                             result(data.shape());

        for (auto _ : state)
        {
            slow_separable_convolution(data, result, kernel, convolution_options().use_simd(true));
            benchmark::DoNotOptimize(result.data());
        }
    }

    BENCHMARK_TEMPLATE(simple_averaging_2d_simd, float);

    template <class V>
    void averaging_2d_no_simd(benchmark::State& state)
    {
        array_nd<V, 2> data(shape_t<2>{2000,3000}),
                             result(data.shape());

        for (auto _ : state)
        {
            separable_convolution_functor()(data, result, kernel, convolution_options().use_simd(false));
            benchmark::DoNotOptimize(result.data());
        }
    }

    BENCHMARK_TEMPLATE(averaging_2d_no_simd, float);

    template <class V>
    void averaging_2d_simd(benchmark::State& state)
    {
        array_nd<V, 2> data(shape_t<2>{2000,3000}),
                             result(data.shape());

        for (auto _ : state)
        {
            separable_convolution_functor()(data, result, kernel, convolution_options().use_simd(true));
            benchmark::DoNotOptimize(result.data());
        }
    }

    BENCHMARK_TEMPLATE(averaging_2d_simd, float);

    template <class V>
    void averaging_3d_no_simd(benchmark::State& state)
    {
        array_nd<V, 3> data(shape_t<3>{100,200,300}),
                             result(data.shape());

        for (auto _ : state)
        {
            separable_convolution_functor()(data, result, kernel, convolution_options().use_simd(false));
            benchmark::DoNotOptimize(result.data());
        }
    }

    BENCHMARK_TEMPLATE(averaging_3d_no_simd, float);

    template <class V>
    void averaging_3d_simd(benchmark::State& state)
    {
        array_nd<V, 3> data(shape_t<3>{100,200,300}),
                             result(data.shape());

        for (auto _ : state)
        {
            separable_convolution_functor()(data, result, kernel, convolution_options().use_simd(true));
            benchmark::DoNotOptimize(result.data());
        }
    }

    BENCHMARK_TEMPLATE(averaging_3d_simd, float);

} // namespace xvigra
