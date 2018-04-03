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
#include <xtensor/xtensor.hpp>
#include <xtensor/xstrided_view.hpp>
#include <xtensor/xnoalias.hpp>
#include <xvigra/global.hpp>

#ifdef BENCHMARK_VIGRA
#  include <vigra/multi_array.hxx>
#endif

#define BENCHMARK_VIGRA2

#ifdef BENCHMARK_VIGRA2
#  include <vigra2/array_nd.hxx>
#endif

static const int SIZE = 1000;

#ifdef BENCHMARK_VIGRA
namespace vigra
{
    using Shape1 = typename MultiArray<1, float>::difference_type;
    using Shape2 = typename MultiArray<2, float>::difference_type;

    template <class V>
    void vigra_iterator(benchmark::State& state)
    {
        MultiArray<2, V> data(Shape2(SIZE, SIZE), 1);
        MultiArray<1, V> res(Shape1(SIZE), 1);

        auto v = data.template bind<0>(SIZE/2);
        for (auto _ : state)
        {
            std::copy(v.begin(), v.end(), res.begin());
            benchmark::DoNotOptimize(res.data());
        }
    }

    // template <class V>
    // void vigra_iterator(benchmark::State& state)
    // {
    //     xt::xtensor<V, 2> data = xt::ones<V>({SIZE,SIZE}),
    //                       res = xt::ones<V>({SIZE});

    //     auto v = xt::view(data, xt::all(), SIZE/2);
    //     for (auto _ : state)
    //     {
    //         std::copy(v.begin(), v.end(), res.begin());
    //         benchmark::DoNotOptimize(res.data());
    //     }
    // }

    template <class V>
    void vigra_loop(benchmark::State& state)
    {
        MultiArray<2, V> data(Shape2(SIZE, SIZE), 1);
        MultiArray<1, V> res(Shape1(SIZE), 1);

        auto v = data.template bind<0>(SIZE/2);
        for (auto _ : state)
        {
            for(int k=0; k<v.shape()[0]; ++k)
            {
                res(k) = v(k);
            }
            benchmark::DoNotOptimize(res.data());
        }
    }

    template <class V>
    void vigra_assign(benchmark::State& state)
    {
        MultiArray<2, V> data(Shape2(SIZE, SIZE), 1);
        MultiArray<1, V> res(Shape1(SIZE), 1);

        auto v = data.template bind<0>(SIZE/2);
        for (auto _ : state)
        {
            res = v;
            benchmark::DoNotOptimize(res.data());
        }
    }

    template <class V>
    void vigra_assign_view(benchmark::State& state)
    {
        MultiArray<2, V> data(Shape2(SIZE, SIZE), 1);
        MultiArray<1, V> res(Shape1(SIZE), 1);

        auto v = data.template bind<0>(SIZE/2);
        auto r = res.subarray(Shape1(0), Shape1(SIZE));
        for (auto _ : state)
        {
            r = v;
            benchmark::DoNotOptimize(r.data());
        }
    }

    // BENCHMARK_TEMPLATE(vigra_dynamic_iterator, float);
    BENCHMARK_TEMPLATE(vigra_iterator, float);
    BENCHMARK_TEMPLATE(vigra_loop, float);
    BENCHMARK_TEMPLATE(vigra_assign, float);
    BENCHMARK_TEMPLATE(vigra_assign_view, float);

} // namespace vigra

#endif // BENCHMARK_VIGRA

#ifdef BENCHMARK_VIGRA2
namespace vigra
{
    template <class V>
    void vigra2_iterator(benchmark::State& state)
    {
        ArrayND<2, V> data(Shape<>{SIZE, SIZE}, 1);
        ArrayND<1, V> res(Shape<>{SIZE}, 1);

        auto v = data.bind(1, SIZE/2);
        for (auto _ : state)
        {
            std::copy(v.begin(), v.end(), res.begin());
            benchmark::DoNotOptimize(res.data());
        }
    }

    template <class V>
    void vigra2_dynamic_iterator(benchmark::State& state)
    {
        ArrayND<runtime_size, V> data(Shape<>{SIZE, SIZE}, 1);
        ArrayND<runtime_size, V> res(Shape<>{SIZE}, 1);

        auto v = data.bind(1, SIZE/2);
        for (auto _ : state)
        {
            std::copy(v.begin(), v.end(), res.begin());
            benchmark::DoNotOptimize(res.data());
        }
    }

    template <class V>
    void vigra2_loop(benchmark::State& state)
    {
        ArrayND<2, V> data(Shape<>{SIZE, SIZE}, 1);
        ArrayND<1, V> res(Shape<>{SIZE}, 1);

        auto v = data.bind(1, SIZE/2);
        for (auto _ : state)
        {
            for(int k=0; k<v.shape()[0]; ++k)
            {
                res(k) = v(k);
            }
            benchmark::DoNotOptimize(res.data());
        }
    }

    template <class V>
    void vigra2_dynamic_loop(benchmark::State& state)
    {
        ArrayND<runtime_size, V> data(Shape<>{SIZE, SIZE}, 1);
        ArrayND<runtime_size, V> res(Shape<>{SIZE}, 1);

        auto v = data.bind(1, SIZE/2);
        for (auto _ : state)
        {
            for(int k=0; k<v.shape()[0]; ++k)
            {
                res(k) = v(k);
            }
            benchmark::DoNotOptimize(res.data());
        }
    }

    template <class V>
    void vigra2_assign(benchmark::State& state)
    {
        ArrayND<2, V> data(Shape<>{SIZE, SIZE}, 1);
        ArrayND<1, V> res(Shape<>{SIZE}, 1);

        auto v = data.bind(1, SIZE/2);
        for (auto _ : state)
        {
            res = v;
            benchmark::DoNotOptimize(res.data());
        }
    }

    template <class V>
    void vigra2_dynamic_assign(benchmark::State& state)
    {
        ArrayND<runtime_size, V> data(Shape<>{SIZE, SIZE}, 1);
        ArrayND<runtime_size, V> res(Shape<>{SIZE}, 1);

        auto v = data.bind(1, SIZE/2);
        for (auto _ : state)
        {
            res = v;
            benchmark::DoNotOptimize(res.data());
        }
    }

    template <class V>
    void vigra2_assign_view(benchmark::State& state)
    {
        ArrayND<2, V> data(Shape<>{SIZE, SIZE}, 1);
        ArrayND<1, V> res(Shape<>{SIZE}, 1);

        auto v = data.bind(1, SIZE/2);
        auto r = res.subarray(Shape<>{0}, Shape<>{SIZE});
        for (auto _ : state)
        {
            r = v;
            benchmark::DoNotOptimize(r.data());
        }
    }

    template <class V>
    void vigra2_dynamic_assign_view(benchmark::State& state)
    {
        ArrayND<runtime_size, V> data(Shape<>{SIZE, SIZE}, 1);
        ArrayND<runtime_size, V> res(Shape<>{SIZE}, 1);

        auto v = data.bind(1, SIZE/2);
        auto r = res.subarray(Shape<>{0}, Shape<>{SIZE});
        for (auto _ : state)
        {
            r = v;
            benchmark::DoNotOptimize(r.data());
        }
    }

    BENCHMARK_TEMPLATE(vigra2_iterator, float);
    BENCHMARK_TEMPLATE(vigra2_dynamic_iterator, float);
    BENCHMARK_TEMPLATE(vigra2_loop, float);
    BENCHMARK_TEMPLATE(vigra2_dynamic_loop, float);
    BENCHMARK_TEMPLATE(vigra2_assign, float);
    BENCHMARK_TEMPLATE(vigra2_dynamic_assign, float);
    BENCHMARK_TEMPLATE(vigra2_assign_view, float);
    BENCHMARK_TEMPLATE(vigra2_dynamic_assign_view, float);

} // namespace vigra
#endif // BENCHMARK_VIGRA2

namespace xvigra
{
    template <class V>
    void xtensor_dynamic_iterator(benchmark::State& state)
    {
        xt::xtensor<V, 2> data = xt::ones<V>({SIZE,SIZE});
        xt::xtensor<V, 1> res = xt::ones<V>({SIZE});

        auto v = xt::dynamic_view(data, xt::slice_vector{xt::all(), SIZE/2});
        for (auto _ : state)
        {
            std::copy(v.begin(), v.end(), res.begin());
            benchmark::DoNotOptimize(res.data());
        }
    }

    template <class V>
    void xtensor_iterator(benchmark::State& state)
    {
        xt::xtensor<V, 2> data = xt::ones<V>({SIZE,SIZE});
        xt::xtensor<V, 1> res = xt::ones<V>({SIZE});

        auto v = xt::view(data, xt::all(), SIZE/2);
        for (auto _ : state)
        {
            std::copy(v.begin(), v.end(), res.begin());
            benchmark::DoNotOptimize(res.data());
        }
    }

    template <class V>
    void xtensor_loop(benchmark::State& state)
    {
        xt::xtensor<V, 2> data = xt::ones<V>({SIZE,SIZE});
        xt::xtensor<V, 1> res = xt::ones<V>({SIZE});

        auto v = xt::dynamic_view(data, xt::slice_vector{xt::all(), SIZE/2});
        for (auto _ : state)
        {
            for(index_t k=0; k<v.shape()[0]; ++k)
            {
                res(k) = v(k);
            }
            benchmark::DoNotOptimize(res.data());
        }
    }

    template <class V>
    void xtensor_assign(benchmark::State& state)
    {
        xt::xtensor<V, 2> data = xt::ones<V>({SIZE,SIZE});
        xt::xtensor<V, 1> res = xt::ones<V>({SIZE});

        auto v = xt::dynamic_view(data, xt::slice_vector{xt::all(), SIZE/2});
        for (auto _ : state)
        {
            res = v;
            benchmark::DoNotOptimize(res.data());
        }
    }

    template <class V>
    void xtensor_assign_view(benchmark::State& state)
    {
        xt::xtensor<V, 2> data = xt::ones<V>({SIZE,SIZE});
        xt::xtensor<V, 1> res = xt::ones<V>({SIZE});

        auto v = xt::view(data, xt::all(), SIZE/2);
        auto r = xt::view(res, xt::all());
        for (auto _ : state)
        {
            r = v;
            benchmark::DoNotOptimize(r.data());
        }
    }

    template <class V>
    void xtensor_assign_dynamic_view(benchmark::State& state)
    {
        xt::xtensor<V, 2> data = xt::ones<V>({SIZE,SIZE});
        xt::xtensor<V, 1> res = xt::ones<V>({SIZE});

        auto v = xt::dynamic_view(data, xt::slice_vector{xt::all(), SIZE/2});
        auto r = xt::dynamic_view(res, xt::slice_vector{xt::all()});

        for (auto _ : state)
        {
            r = v;
            benchmark::DoNotOptimize(r.data());
        }
    }

    template <class V>
    void xtensor_assign_view_noalias(benchmark::State& state)
    {
        xt::xtensor<V, 2> data = xt::ones<V>({SIZE,SIZE});
        xt::xtensor<V, 1> res = xt::ones<V>({SIZE});

        auto v = xt::view(data, xt::all(), SIZE/2);
        auto r = xt::view(res, xt::all());
        for (auto _ : state)
        {
            xt::noalias(r) = v;
            benchmark::DoNotOptimize(r.data());
        }
    }

    template <class V>
    void xtensor_assign_dynamic_view_noalias(benchmark::State& state)
    {
        xt::xtensor<V, 2> data = xt::ones<V>({SIZE,SIZE});
        xt::xtensor<V, 1> res = xt::ones<V>({SIZE});

        auto v = xt::dynamic_view(data, xt::slice_vector{xt::all(), SIZE/2});
        auto r = xt::dynamic_view(res, xt::slice_vector{xt::all()});

        for (auto _ : state)
        {
            xt::noalias(r) = v;
            benchmark::DoNotOptimize(r.data());
        }
    }

    BENCHMARK_TEMPLATE(xtensor_dynamic_iterator, float);
    BENCHMARK_TEMPLATE(xtensor_iterator, float);
    BENCHMARK_TEMPLATE(xtensor_loop, float);
    BENCHMARK_TEMPLATE(xtensor_assign, float);
    BENCHMARK_TEMPLATE(xtensor_assign_view, float);
    BENCHMARK_TEMPLATE(xtensor_assign_dynamic_view, float);
    BENCHMARK_TEMPLATE(xtensor_assign_view_noalias, float);
    BENCHMARK_TEMPLATE(xtensor_assign_dynamic_view_noalias, float);

} // namespace xvigra
