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
#include <xvigra/tiny_vector.hpp>

namespace xvigra
{
    template <class V>
    void bm_tiny_vector_loop(benchmark::State& state)
    {
        V a{ 1,2,3,4 };
        V b{ 1,2,3,4 };

        for (auto _ : state)
        {
            V result(dont_init);
            for (index_t i = 0; i < a.size(); ++i)
            {
                result[i] = a[i] + b[i];
            }
            benchmark::DoNotOptimize(result.data());
            benchmark::DoNotOptimize(a.data());
            benchmark::DoNotOptimize(b.data());
        }
    }

    template <class V>
    void bm_tiny_vector_plus(benchmark::State& state)
    {
        V a{ 1,2,3,4 };
        V b{ 1,2,3,4 };

        for (auto _ : state)
        {
            V result = a + b;
            benchmark::DoNotOptimize(result.data());
            benchmark::DoNotOptimize(a.data());
            benchmark::DoNotOptimize(b.data());
        }
    }

    BENCHMARK_TEMPLATE(bm_tiny_vector_loop, tiny_vector<index_t, 4>);
    BENCHMARK_TEMPLATE(bm_tiny_vector_plus, tiny_vector<index_t, 4>);

} // namespace xvigra
