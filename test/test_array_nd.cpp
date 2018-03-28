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

#include <typeinfo>
#include <iostream>
#include <string>
#include <vector>
#include <numeric>
#include "unittest.hpp"
#include <xtensor/xeval.hpp>
#include <xtensor/xinfo.hpp>
#include <xvigra/array_nd.hpp>
// #include <vigra2/gaussians.hxx>

namespace xvigra
{
    constexpr index_t ndim = 3;
    shape_t<> s{4, 3, 2};

#if 1
    using array_nd_types = testing::Types<array_nd<uint8_t, ndim>,
                                          array_nd<int, ndim>,
                                          array_nd<int>,
                                          array_nd<float, ndim>
                                         >;
#else
    using array_nd_types = testing::Types<array_nd<int, ndim>
                                         >;
#endif

    TYPED_TEST_SETUP(array_nd_test, array_nd_types);

    TYPED_TEST(array_nd_test, concepts)
    {
        using A = TypeParam;
        using V = typename A::view_type;
        EXPECT_TRUE(tensor_concept<A>::value);
        EXPECT_TRUE(tensor_concept<A &>::value);
        EXPECT_TRUE(tensor_concept<V>::value);
        EXPECT_TRUE(tensor_concept<V &>::value);
        bool raw_data_api = has_raw_data_api<A>::value;
        EXPECT_TRUE(raw_data_api);
        raw_data_api = has_raw_data_api<V>::value;
        EXPECT_TRUE(raw_data_api);
     }

    TYPED_TEST(array_nd_test, constructor)
    {
        using A = TypeParam;
        using T = typename A::value_type;
        using S = typename A::shape_type;
        using V = typename A::view_type;
        constexpr auto N = A::ndim;

        if(N != runtime_size)
        {
            EXPECT_EQ(N, ndim);
        }

        std::vector<T> data0(prod(s), 0), data1(prod(s));
        std::iota(data1.begin(), data1.end(), 0);

        V v;

        EXPECT_EQ(v.shape(), S());
        EXPECT_EQ(v.strides(), S());
        EXPECT_EQ(v.raw_data(), (T*)0);
        EXPECT_FALSE(v.has_data());

        V v0(s, &data0[0], c_order);
        V v1(s, &data1[0], c_order);

        EXPECT_EQ(v1.shape(), s);
        EXPECT_EQ(v1.shape(1), s[1]);
        EXPECT_EQ(v1.strides(), (S{ 6,2,1 }));
        EXPECT_EQ(v1.strides(1), 2);
        EXPECT_EQ(v1.raw_data(), &data1[0]);
        EXPECT_TRUE(v1.has_data());
        EXPECT_TRUE(v1.is_consecutive());
        EXPECT_FALSE(v1.owns_memory());
        bool memory_range_check = v1.memory_range() == tiny_vector<char*, 2>{(char*)&data1.front(), (char*)(1+&data1.back())};
        EXPECT_TRUE(memory_range_check);

        EXPECT_TRUE(v1 == v1);
        EXPECT_FALSE(v1 != v1);
        EXPECT_TRUE(v1 != v);
        EXPECT_FALSE(v1 == v);

        EXPECT_TRUE(any(v1));
        EXPECT_FALSE(all(v1));

        A a0(s);

        EXPECT_EQ(a0.shape(), s);
        EXPECT_FALSE(any(a0));
        EXPECT_FALSE(all(a0));

        A aa(v1 + 1);

        // c_order iteration
        auto iter1 = v1.begin(), end1 = v1.end();
        int c = 0;
        for(int z=0; z<s[0]; ++z)
        {
            for(int y=0; y<s[1]; ++y)
            {
                for(int x=0; x<s[2]; ++x, ++c, ++iter1)
                {
                    EXPECT_TRUE((v1.is_inside(S{ z,y,x })));
                    EXPECT_FALSE((v1.is_outside(S{ z,y,x })));
                    EXPECT_EQ(v1(z,y,x), c);
                    EXPECT_EQ(v1(c), c);
                    EXPECT_EQ(v1(), 0);
                    EXPECT_EQ((v1[{z,y,x}]), c);
                    EXPECT_EQ(v1[c], c);

                    EXPECT_EQ(aa(z,y,x), c+1);

                    EXPECT_EQ(*iter1, c);
                    EXPECT_FALSE(iter1 == end1);
                    // EXPECT_TRUE(iter1 < end1); // FIXME: xtensor bug
                }
            }
        }
        EXPECT_FALSE(v1.is_inside(S{ -1,-1,-1 }));
        EXPECT_TRUE(v1.is_outside(S{ -1,-1,-1 }));
        EXPECT_TRUE(iter1 == end1);
        // EXPECT_FALSE(iter1 < end1); // FIXME: xtensor bug

        V v2(s, default_axistags(3, false, f_order), &data1[0], f_order);

        EXPECT_EQ(v2.shape(), s);
        EXPECT_EQ(v2.strides(), (S{ 1,4,12 }));
        EXPECT_EQ(v2.raw_data(), &data1[0]);
        EXPECT_TRUE(v2.is_consecutive());
        EXPECT_FALSE(v2.owns_memory());
        EXPECT_EQ(v2.axistags(), (axis_tags<N>{ tags::axis_x, tags::axis_y , tags::axis_z }));

        EXPECT_TRUE(v1 != v2);
        EXPECT_FALSE(v1 == v2);

        auto iter2 = v2.template begin<f_order>(), end2 = v2.template end<f_order>();

        // f_order iteration
        c = 0;
        for(int z=0; z<s[2]; ++z)
        {
            for(int y=0; y<s[1]; ++y)
            {
                for(int x=0; x<s[0]; ++x, ++c, ++iter2)
                {
                    EXPECT_EQ((v2[{x,y,z}]), c);
                    EXPECT_EQ(v2[c], c);
                    EXPECT_EQ(*iter2, c);
                    EXPECT_FALSE(iter2 == end2);
                }
            }
        }
        EXPECT_TRUE(iter2 == end2);

        V v3(s, S{3, 1, 12}, &data1[0]);

        EXPECT_EQ(v3.shape(), s);
        EXPECT_EQ(v3.strides(), (S{ 3,1,12 }));
        EXPECT_EQ(v3.raw_data(), &data1[0]);
        EXPECT_TRUE(v3.is_consecutive());

        c = 0;
        for(int x=0; x<s[2]; ++x)
        {
            for(int z=0; z<s[0]; ++z)
            {
                for(int y=0; y<s[1]; ++y, ++c)
                {
                    EXPECT_EQ((v3[{z,y,x}]), c);
                    EXPECT_EQ(v3[c], c);
                }
            }
        }

        A a1(v3);
        EXPECT_TRUE(a1 == v3);
        A a2(v3, f_order);
        EXPECT_TRUE(a2 == v3);
        A a3(transpose(v3));
        EXPECT_TRUE(transpose(a3) == v3);
        A a4(transpose(v3), f_order);
        EXPECT_TRUE(transpose(a4) == v3);

        auto d = a1.raw_data();
        A a5(std::move(a1));
        EXPECT_FALSE(a1.has_data());
        EXPECT_EQ(a1.shape(), S());
        EXPECT_EQ(a1.size(), 0);
        EXPECT_TRUE(a5 == v3);
        EXPECT_EQ(d, a5.raw_data());

        A a6(s, data1.begin(), data1.end());
        EXPECT_EQ(a6.shape(), s);
        EXPECT_EQ(a6, v1);

        a6 += v1;
        a6 /= 2;
        EXPECT_EQ(a6, v1);

        A a7{{{ 0,  1},
              { 2,  3},
              { 4,  5}},
             {{ 6,  7},
              { 8,  9},
              {10, 11}},
             {{12, 13},
              {14, 15},
              {16, 17}},
             {{18, 19},
              {20, 21},
              {22, 23}}};
        EXPECT_EQ(a7.shape(), s);
        EXPECT_EQ(a7, v1);

        A a8(v0);
        EXPECT_EQ(a8, v0);
        swap(a7, a8);
        EXPECT_EQ(a7, v0);
        EXPECT_EQ(a8, v1);

        T * data = a8.raw_data();
        A a9(std::move(a8));
        EXPECT_EQ(a9.raw_data(), data);
        EXPECT_EQ(a9, v1);
        EXPECT_FALSE(a8.has_data());
        EXPECT_EQ(a8.raw_data(), nullptr);
        EXPECT_EQ(a8.size(), 0);

        swap(v0, v1);
        EXPECT_EQ(v1, a7);
        EXPECT_EQ(v0, a9);
    }

    TYPED_TEST(array_nd_test, assignment)
    {
        using A = TypeParam;
        using T = typename A::value_type;
        // using S = typename A::shape_type;
        using V = typename A::view_type;

        std::vector<T> data0(prod(s), 0), data1(prod(s));
        std::iota(data1.begin(), data1.end(), 0);

        V v0(s, &data1[0]);
        V v1(s, &data0[0]);

        v1.set_channel_axis(2);

        v0.swap_data(v1);
        EXPECT_EQ(v0.raw_data(), &data1[0]);
        EXPECT_EQ(v1.raw_data(), &data0[0]);

        index_t count = 0;
        for (index_t k = 0; k < v0.size(); ++k, ++count)
        {
            EXPECT_EQ(v0[k], 0);
            EXPECT_EQ(v1[k], count);
        }

        v0 = 2;
        for (int k = 0; k < v0.size(); ++k)
        {
            EXPECT_EQ(v0[k], 2);
        }

        v0 = xt::xscalar<T>(1);
        for (int k = 0; k < v0.size(); ++k)
        {
            EXPECT_EQ(v0[k], 1);
        }

        v0 += 2;
        for (int k = 0; k < v0.size(); ++k)
        {
            EXPECT_EQ(v0[k], 3);
        }

        v0 -= 1;
        for (int k = 0; k < v0.size(); ++k)
        {
            EXPECT_EQ(v0[k], 2);
        }

        v0 *= 5;
        for (int k = 0; k < v0.size(); ++k)
        {
            EXPECT_EQ(v0[k], 10);
        }

        v0 /= 10;
        for (int k = 0; k < v0.size(); ++k)
        {
            EXPECT_EQ(v0[k], 1);
        }

        v0 *= v1;
        EXPECT_EQ(v0, v1);

        v0 += v1;
        v0 /= 2;
        EXPECT_EQ(v0, v1);

        v0 += v1;
        v0 -= v1;
        EXPECT_EQ(v0, v1);

        v0 += xt::xscalar<T>(1);
        v0 /= v0;
        for (int k = 0; k < v0.size(); ++k)
        {
            EXPECT_EQ(v0[k], 1);
        }

        v0 -= v0;
        for (int k = 0; k < v0.size(); ++k)
        {
            EXPECT_EQ(v0[k], 0);
        }

        v0 = v1;
        EXPECT_EQ(v0, v1);

        {
            V v;
            EXPECT_FALSE(v.has_data());
            EXPECT_FALSE(v.has_channel_axis());

            v = v1;
            EXPECT_TRUE(v.has_data());
            EXPECT_EQ(v.shape(), v1.shape());
            EXPECT_EQ(v.strides(), v1.strides());
            EXPECT_EQ(v.raw_data(), v1.raw_data());
            EXPECT_EQ(v.channel_axis(), 2);
            EXPECT_TRUE(v.is_consecutive());
            EXPECT_FALSE(v.owns_memory());

            // shape mismatch errors
            EXPECT_THROW(v1 = v1.transpose(), std::runtime_error);
            EXPECT_THROW(v1 = xt::view(v1.transpose(), xt::all()), std::runtime_error);
            EXPECT_THROW(v1 += v1.transpose(), std::runtime_error);
            EXPECT_THROW(v1 -= v1.transpose(), std::runtime_error);
            EXPECT_THROW(v1 *= v1.transpose(), std::runtime_error);
            EXPECT_THROW(v1 /= v1.transpose(), std::runtime_error);
        }
        {
            V v;
            A a(s);
            v = a;
            EXPECT_TRUE(v.has_data());
            EXPECT_EQ(v.shape(), a.shape());
            EXPECT_EQ(v.strides(), a.strides());
            EXPECT_EQ(v.raw_data(), a.raw_data());
            EXPECT_TRUE(v.is_consecutive());
            EXPECT_FALSE(v.owns_memory());
        }
        {
            V v;
            auto a = xt::xarray<T>::from_shape({4,3,2});
            v = a;
            EXPECT_TRUE(v.has_data());
            EXPECT_EQ(v.shape(), shape_t<>(a.shape()));
            EXPECT_EQ(v.strides(), shape_t<>(a.strides()));
            EXPECT_EQ(v.raw_data(), a.raw_data());
            EXPECT_TRUE(v.is_consecutive());
            EXPECT_FALSE(v.owns_memory());
        }
        {
            V v;
            auto a = xt::xarray<uint16_t>::from_shape({4,3,2});
            EXPECT_THROW(v = a, std::runtime_error);       // incompatible pointer types
            EXPECT_THROW(v = v1 + 1, std::runtime_error);  // unevaluated expression
        }

        {
            A a0(s, 0),
              a1(v1);

            T * data = a1.raw_data();
            EXPECT_EQ(a1, v1);
            // assignment only copies data, because shapes are equal
            a1 = a0;
            EXPECT_EQ(a1.raw_data(), data);
            EXPECT_EQ(a1, a0);

            // assignment resizes, because shapes differ
            a1 = v1.transpose();
            EXPECT_NE(a1.raw_data(), data);
            EXPECT_EQ(a1.shape(), reversed(s));
            EXPECT_EQ(a1, v1.transpose());

            a1 = 1;
            EXPECT_EQ(a1.reshape(s), a0.view()+1); // FIXME: a0+1 doesn't compile

            data = a0.raw_data();
            EXPECT_NE(a1.raw_data(), data);
            EXPECT_NE(a1.shape(), a0.shape());

            // move assignment actually moves, because shapes differ
            a1 = std::move(a0);
            EXPECT_FALSE(a0.has_data());
            EXPECT_EQ(a0.size(), 0);
            EXPECT_EQ(a0.raw_data(), nullptr);
            EXPECT_EQ(a1.raw_data(), data);
            EXPECT_TRUE(all(equal(a1.view(), 0))); // FIXME

            A a2(a1);
            EXPECT_NE(a1.raw_data(), a2.raw_data());
            EXPECT_EQ(a1, a2);

            a2 += xt::xscalar<T>(1);
            EXPECT_EQ(a1.view()+1, a2);

            // move assignment only copies data, because shapes are equal
            data = a1.raw_data();
            a1 = std::move(a2);
            EXPECT_EQ(a1.raw_data(), data);
            EXPECT_NE(a1.raw_data(), a2.raw_data());
            EXPECT_EQ(a1, a2);
        }
    }

    TYPED_TEST(array_nd_test, bind)
    {
        using A = TypeParam;
        using T = typename A::value_type;
        using V = typename A::view_type;

        std::vector<T> data1(prod(s));
        std::iota(data1.begin(), data1.end(), 0);

        V v0(s, &data1[0]);

        v0.set_axistags(default_axistags(3, true));
        EXPECT_TRUE(v0.has_channel_axis());
        EXPECT_EQ(v0.channel_axis(), 2);

        auto v1 = v0.bind(0, 2);

        EXPECT_EQ(v1.shape(), (shape_t<2>{3, 2}));
        EXPECT_TRUE(v1.has_channel_axis());
        EXPECT_EQ(v1.channel_axis(), 1);
        EXPECT_EQ(v1.axistags(), (axis_tags<>{tags::axis_x, tags::axis_c}));
        EXPECT_TRUE(v1.is_consecutive());
        EXPECT_FALSE(v1.owns_memory());

        int count = 0;
        for (int i = 0; i < s[0]; ++i)
            for (int j = 0; j < s[1]; ++j)
                for (int k = 0; k < s[2]; ++k, ++count)
                    if (i == 2)
                        EXPECT_EQ((v1[{j, k}]), count);

        auto v2 = v0.bind(1, 1);

        EXPECT_EQ(v2.shape(), (shape_t<2>{4, 2}));
        EXPECT_TRUE(v2.has_channel_axis());
        EXPECT_EQ(v2.channel_axis(), 1);
        EXPECT_EQ(v2.axistags(), (axis_tags<>{ tags::axis_y, tags::axis_c }));
        EXPECT_FALSE(v2.is_consecutive());

        count = 0;
        for (int i = 0; i < s[0]; ++i)
            for (int j = 0; j < s[1]; ++j)
                for (int k = 0; k < s[2]; ++k, ++count)
                    if (j == 1)
                        EXPECT_EQ((v2[{i, k}]), count);

        auto v3 = v0.bind(2, 0);

        EXPECT_EQ(v3.shape(), (shape_t<2>{4, 3}));
        EXPECT_FALSE(v3.has_channel_axis());
        EXPECT_EQ(v3.axistags(), (axis_tags<>{ tags::axis_y, tags::axis_x }));
        EXPECT_FALSE(v3.is_consecutive());

        count = 0;
        for (int i = 0; i < s[0]; ++i)
            for (int j = 0; j < s[1]; ++j)
                for (int k = 0; k < s[2]; ++k, ++count)
                    if (k == 0)
                        EXPECT_EQ((v3[{i, j}]), count);

        auto v4 = v0.bind(0, 3).bind(0, 2).bind(0, 1);
        EXPECT_EQ(v4.shape(), shape_t<1>{1});
        EXPECT_EQ(v4[{0}], (v0[{3, 2, 1}]));
        EXPECT_FALSE(v4.has_channel_axis());
        EXPECT_EQ(v4.axistags(), (axis_tags<>{ tags::axis_unknown }));
        EXPECT_TRUE(v4.is_consecutive());

        auto v5 = v0.bind_left(shape_t<2>{3, 2});
        EXPECT_EQ(v5.shape(), shape_t<1>{2});
        EXPECT_EQ(v5[0], (v0[{3, 2, 0}]));
        EXPECT_EQ(v5[1], (v0[{3, 2, 1}]));
        EXPECT_TRUE(v5.has_channel_axis());
        EXPECT_EQ(v5.channel_axis(), 0);
        EXPECT_EQ(v5.axistags(), (axis_tags<>{ tags::axis_c }));
        EXPECT_TRUE(v0.bind(0, 3).bind(0, 2) == v5);
        EXPECT_TRUE((v0.bind_left(shape_t<>{3, 2}) == v5));
        EXPECT_TRUE((v0.bind_left(shape_t<>{}) == v0));
        EXPECT_TRUE((v0.bind_left(shape_t<0>()) == v0));

        auto v6 = v0.bind_right(shape_t<2>{1, 0});
        EXPECT_EQ(v6.shape(), shape_t<1>{4});
        EXPECT_EQ(v6[0], (v0[{0, 1, 0}]));
        // EXPECT_EQ(v6[1], (v0[{1, 1, 0}]));// FIXME: adapt to changed semantics of operator[]
        // EXPECT_EQ(v6[2], (v0[{2, 1, 0}]));
        // EXPECT_EQ(v6[3], (v0[{3, 1, 0}]));
        EXPECT_FALSE(v6.has_channel_axis());
        EXPECT_EQ(v6.axistags(), (axis_tags<>{ tags::axis_y }));
        EXPECT_TRUE(v0.bind(1,1).bind(1,0) == v6);
        EXPECT_TRUE((v0.bind_right(shape_t<>{1,0}) == v6));
        EXPECT_TRUE((v0.bind_right(shape_t<>{}) == v0));
        EXPECT_TRUE((v0.bind_right(shape_t<0>()) == v0));

        auto v7 = v0.template view<runtime_size>();
        auto id7 = decltype(v7)::internal_dimension;
        EXPECT_EQ(id7, -1);
        EXPECT_TRUE(v7 == v0);

        auto v8 = v0.template view<3>();
        auto id8 = decltype(v8)::internal_dimension;
        EXPECT_EQ(id8, 3);
        EXPECT_TRUE(v8 == v0);

        auto v9 = v0.diagonal();
        EXPECT_EQ(v9.shape(), shape_t<1>{2});
        // EXPECT_EQ(v9[0], (v0[{0, 0, 0}]));
        // EXPECT_EQ(v9[1], (v0[{1, 1, 1}]));  // FIXME: adapt to changed semantics of operator[]
    }

    TYPED_TEST(array_nd_test, subarray)
    {
        using A = TypeParam;
        using T = typename A::value_type;
        using S = typename A::shape_type;
        using V = typename A::view_type;

        std::vector<T> data1(prod(s));
        std::iota(data1.begin(), data1.end(), 0);

        V v0(s, &data1[0]);
        v0.set_channel_axis(2);

        auto v1 = v0.subarray(S{ 0,0,0 }, v0.shape());
        EXPECT_TRUE(v0 == v1);
        EXPECT_EQ(v1.channel_axis(), 2);

        auto v2 = v0.subarray(S{ 1,0,0 }, S{ 3,2,2 } );
        EXPECT_EQ(v2.shape(), (S{ 2,2,2 }));
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                for (int k = 0; k < 2; ++k)
                    EXPECT_EQ((v2[{i, j, k}]), (v0[{i + 1, j, k}]));
        EXPECT_TRUE((v2 == v0.subarray(S{ 1,0,0 }, S{ -1,-1, 2 })));
        EXPECT_TRUE((v2 == v0.subarray(S{ -3,0,0 }, S{ -1,-1, 2 })));

        EXPECT_THROW((v0.subarray(S{ 1,0,0 }, S{ 3,2,4 })), std::runtime_error);
        EXPECT_THROW((v0.subarray(S{ 1,0,0 }, S{ 0,2,2 })), std::runtime_error);
        EXPECT_THROW((v0.subarray(S{ -5,0,0 }, S{ 3,2,2 })), std::runtime_error);
    }

    TYPED_TEST(array_nd_test, channel_axis)
    {
        using A = TypeParam;
        using T = typename A::value_type;
        using S = typename A::shape_type;
        using V = typename A::view_type;
        constexpr auto N = A::ndim;

        std::vector<T> data1(prod(s));
        std::iota(data1.begin(), data1.end(), 0);

        {
            // low-level tests
            S res = detail::permutation_to_order(S{ 6,2,1 }, c_order);
            EXPECT_EQ(res, (S{ 0,1,2 }));
            res = detail::permutation_to_order(S{ 1,4,12 }, c_order);
            EXPECT_EQ(res, (S{ 2,1,0 }));
            res = detail::permutation_to_order(S{ 3, 1, 12 }, c_order);
            EXPECT_EQ(res, (S{ 2,0,1 }));
            res = detail::permutation_to_order(S{ 3,1,0 }, c_order);
            EXPECT_EQ(res, (S{ 2,0,1 }));
            res = detail::permutation_to_order(S{ 3,0,1 }, c_order);
            EXPECT_EQ(res, (S{ 1,0,2 }));
            res = detail::permutation_to_order(S{ 0,1,0 }, c_order);
            EXPECT_EQ(res, (S{ 0,2,1 }));
        }

        {
            V v(s, &data1[0], c_order);
            v.set_channel_axis(2);
            V t = v.transpose(c_order);

            EXPECT_EQ(t.shape(), s);
            EXPECT_EQ(t.strides(), (S{ 6,2,1 }));
            EXPECT_EQ(t.channel_axis(), 2);

            V tt = v.transpose();
            EXPECT_EQ(tt.shape(), reversed(s));
            EXPECT_EQ(tt.strides(), (S{ 1,2,6 }));
            EXPECT_EQ(tt.channel_axis(), 0);

            V v1(reversed(s), &data1[0], f_order);
            EXPECT_TRUE(v1 != v);
            EXPECT_FALSE(v1 == v);
            EXPECT_TRUE(v1 == v.transpose());
            EXPECT_FALSE(v1 != v.transpose());
        }
        {
            V v(s, &data1[0], f_order);
            v.set_channel_axis(2);
            V t = v.transpose(c_order);

            EXPECT_EQ(t.shape(), (S{ 2,3,4 }));
            EXPECT_EQ(t.strides(), (S{ 12,4,1 }));
            EXPECT_EQ(t.channel_axis(), 0);

            t = v.transpose();
            EXPECT_EQ(t.shape(), reversed(s));
            EXPECT_EQ(t.strides(), (S{ 12,4,1 }));
            EXPECT_EQ(t.channel_axis(), 0);
        }
        {
            V v(s, S{ 3, 1, 12 }, &data1[0]);
            v.set_channel_axis(2);
            V t = v.transpose(c_order);

            EXPECT_EQ(t.shape(), (S{ 2,4,3 }));
            EXPECT_EQ(t.strides(), (S{ 12,3,1 }));
            EXPECT_EQ(t.channel_axis(), 0);

            V tt = v.transpose();
            EXPECT_EQ(tt.shape(), reversed(s));
            EXPECT_EQ(tt.strides(), (S{ 12,1,3 }));
            EXPECT_EQ(tt.channel_axis(), 0);
        }
        {
            V v(S{ 4,6,1 }, S{ 6,1,1 }, default_axistags(3), &data1[0]);
            V t = v.transpose(c_order);

            EXPECT_EQ(t.shape(), (S{ 1,4,6 }));
            EXPECT_EQ(t.strides(), (S{ 0,6,1 }));
            EXPECT_EQ(t.axistags(), (axis_tags<N>{tags::axis_x, tags::axis_z, tags::axis_y}));
        }
        {
            V v(S{ 4,1,6 }, S{ 1,1,6 }, &data1[0]);
            v.set_channel_axis(2);
            V t = v.transpose(c_order);

            EXPECT_EQ(t.shape(), (S{ 1,6,4 }));
            EXPECT_EQ(t.strides(), (S{ 0,6,1 }));
            EXPECT_EQ(t.channel_axis(), 1);
        }
        {
            V v(S{ 1,24,1 }, S{ 1,1,1 }, &data1[0]);
            v.set_channel_axis(1);
            V t = v.transpose(c_order);

            EXPECT_EQ(t.shape(), (S{ 1,1,24 }));
            EXPECT_EQ(t.strides(), (S{ 0,0,1 }));
            EXPECT_EQ(t.channel_axis(), 2);
        }
    }

    TYPED_TEST(array_nd_test, slicing)
    {
        using A = TypeParam;
        using T = typename A::value_type;
        using V = typename A::view_type;

        std::vector<T> data1(prod(s));
        std::iota(data1.begin(), data1.end(), 0);
        V v1(s, &data1[0]);

        using namespace slicing;
        {
            xt::slice_vector sv;
            sv.push_back(xt::range(_,_,2));
            auto xv = xt::dynamic_view(v1, sv);
            auto vv = v1.view(slice(_,_,2));
            auto dv = v1.view(slice_vector().push_back(slice(_,_,2)));
            EXPECT_EQ(vv.shape(), (shape_t<>{2,3,2}));
            EXPECT_EQ(dv.shape(), (shape_t<>{2,3,2}));
            EXPECT_EQ(xv, vv);
            EXPECT_EQ(xv, dv);
        }
        {
            xt::slice_vector sv;
            sv.push_back(ellipsis());
            sv.push_back(xt::range(_,_,2));
            auto xv = xt::dynamic_view(v1, sv);
            auto vv = v1.view(ellipsis(), slice(_,_,2));
            auto dv = v1.view(slice_vector().push_back(ellipsis()).push_back(slice(_,_,2)));
            EXPECT_EQ(vv.shape(), (shape_t<>{4,3,1}));
            EXPECT_EQ(dv.shape(), (shape_t<>{4,3,1}));
            EXPECT_EQ(xv, vv);
            EXPECT_EQ(xv, dv);
        }
        {
            xt::slice_vector sv;
            sv.push_back(newaxis());
            sv.push_back(all());
            sv.push_back(xt::range(_,_,2));
            auto xv = xt::dynamic_view(v1, sv);
            auto vv = v1.view(newaxis(), all(), slice(_,_,2));
            auto dv = v1.view(slice_vector().push_back(newaxis()).push_back(all()).push_back(slice(_,_,2)));
            EXPECT_EQ(vv.shape(), (shape_t<>{1,4,2,2}));
            EXPECT_EQ(dv.shape(), (shape_t<>{1,4,2,2}));
            EXPECT_EQ(xv, vv);
            EXPECT_EQ(xv, dv);
        }
        {
            xt::slice_vector sv;
            sv.push_back(1);
            sv.push_back(ellipsis());
            sv.push_back(0);
            auto xv = xt::dynamic_view(v1, sv);
            auto vv = v1.view(1, ellipsis(), 0);
            auto dv = v1.view(slice_vector().push_back(1).push_back(ellipsis()).push_back(0));
            EXPECT_EQ(vv.shape(), (shape_t<>{3}));
            EXPECT_EQ(dv.shape(), (shape_t<>{3}));
            EXPECT_EQ(xv, vv);
            EXPECT_EQ(xv, dv);
        }
    }

    TYPED_TEST(array_nd_test, overlapping_memory)
    {
        using A = TypeParam;
        using T = typename A::value_type;
        using V = typename A::view_type;

        auto a = xt::xarray<T>::from_shape({4,3,2});

        std::vector<T> data1(prod(s));
        std::iota(data1.begin(), data1.end(), 0);
        V v1(s, &data1[0]);

        {
            detail::overlapping_memory_checker m(&v1(), &v1[v1.shape()-1]+1);
            EXPECT_FALSE(m(V()));
            EXPECT_TRUE(m(v1));
            EXPECT_TRUE(m(v1+v1));
            EXPECT_TRUE(m(v1+1));
            EXPECT_FALSE(m(V()+1));
            EXPECT_FALSE(m(a));
            EXPECT_FALSE(m(a+1));
            EXPECT_FALSE(m(xt::view(a, xt::ellipsis())));
            xt::slice_vector sv;
            sv.push_back(xt::ellipsis());
            EXPECT_FALSE(m(xt::dynamic_view(a, sv)));
        }
        {
            auto v2 = v1.bind(0, 0);
            detail::overlapping_memory_checker m(&v2(), &v2[v2.shape()-1]+1);
            EXPECT_FALSE(m(v1.bind(0,1)));
            EXPECT_TRUE(m(v2));
            EXPECT_TRUE(m(v1));
            EXPECT_TRUE(m(v1+v1));
            EXPECT_TRUE(m(v1+1));
            EXPECT_FALSE(m(v1.bind(0,1)+1));
        }
        {
            V v2(s, a.raw_data());
            detail::overlapping_memory_checker m(&v2(), &v2[v2.shape()-1]+1);
            EXPECT_TRUE(m(a));
            EXPECT_TRUE(m(a+1));
            EXPECT_TRUE(m(xt::view(a, xt::ellipsis())));
            xt::slice_vector sv;
            sv.push_back(xt::ellipsis());
            EXPECT_TRUE(m(xt::dynamic_view(a, sv)));
            EXPECT_FALSE(m(v1));
        }
    }

#if 0
    void testOverlappingMemory()
    {
        using namespace tags;
        V v(S{4,2,2}, &data1[0]);

        const int M = (N == runtime_size)
                          ? N
                          : 2;
        auto v1 = v.reshape(shape_t<M>{4, 4}, axis_tags<M>{axis_y, axis_x});
        EXPECT_EQ(v1.axistags(), (axis_tags<M>{axis_y, axis_x}));
        ArrayND<M, int> vs = v1;

        int count = 0;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j, ++count)
                EXPECT_EQ((v1[{i, j}]), count);

        v1 = v1.transpose();
        EXPECT_EQ(v1.axistags(), (axis_tags<M>{axis_y, axis_x}));

        count = 0;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j, ++count)
                EXPECT_EQ((v1[{j, i}]), count);

        auto v2 = v.reshape(shape_t<M>{4, 4}, f_order);
        EXPECT_EQ(v2.axistags(), (axis_tags<M>{axis_unknown, axis_unknown}));

        count = 0;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j, ++count)
                EXPECT_EQ((v2[{i, j}]), count);

        v2 = v2.transpose();

        count = 0;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j, ++count)
                EXPECT_EQ((v2[{j, i}]), count);

        auto v3 = v.flatten();
        EXPECT_EQ(v3.ndim(), 1);
        count = 0;
        for (int i = 0; i < 16; ++i, ++count)
            EXPECT_EQ(v3[i], count);

        using namespace array_detail;
        {
            v1 = vs;
            auto b1 = v1.bind(0, 1),
                 b2 = v1.bind(0, 2);
            EXPECT_EQ(checkMemoryOverlap(b1.memoryRange(), b2.memoryRange()), NoMemoryOverlap);
            b1 = b2;
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    if (i == 1)
                        EXPECT_EQ((vs[{2, j}]), (v1[{i, j}]));
                    else
                        EXPECT_EQ((vs[{i, j}]), (v1[{i, j}]));
        }
        {
            v1 = vs;
            auto b1 = v1.bind(1, 1),
                 b2 = v1.bind(1, 2);
            EXPECT_EQ(checkMemoryOverlap(b1.memoryRange(), b2.memoryRange()), TargetOverlapsLeft);
            b1 = b2;
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    if (j == 1)
                        EXPECT_EQ((vs[{i, 2}]), (v1[{i, j}]));
                    else
                        EXPECT_EQ((vs[{i, j}]), (v1[{i, j}]));
        }
        {
            v1 = vs;
            auto b1 = v1.bind(1, 1),
                 b2 = v1.bind(1, 2);
            EXPECT_EQ(checkMemoryOverlap(b2.memoryRange(), b1.memoryRange()), TargetOverlapsRight);
            b2 = b1;
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    if (j == 2)
                        EXPECT_EQ((vs[{i, 1}]), (v1[{i, j}]));
                    else
                        EXPECT_EQ((vs[{i, j}]), (v1[{i, j}]));
        }
        {
            v1 = vs;
            auto b1 = v1.bind(0, 2),
                 b2 = v1.bind(1, 1);
            EXPECT_EQ(checkMemoryOverlap(b1.memoryRange(), b2.memoryRange()), TargetOverlaps);
            b1 = b2;
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    if (i == 2)
                        EXPECT_EQ((vs[{j, 1}]), (v1[{i, j}]));
                    else
                        EXPECT_EQ((vs[{i, j}]), (v1[{i, j}]));
        }
        {
            v1 = vs;
            auto b1 = v1.bind(0, 1),
                 b2 = v1.bind(1, 2);
            EXPECT_EQ(checkMemoryOverlap(b2.memoryRange(), b1.memoryRange()), TargetOverlaps);
            b2 = b1;
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    if (j == 2)
                        EXPECT_EQ((vs[{1, i}]), (v1[{i, j}]));
                    else
                        EXPECT_EQ((vs[{i, j}]), (v1[{i, j}]));
        }
    }
#endif

    TYPED_TEST(array_nd_test, functions)
    {
        using A = TypeParam;
        using T = typename A::value_type;
        using V = typename A::view_type;
        constexpr auto N = A::ndim;

        std::vector<T> data0(prod(s), 0), data1(prod(s));
        std::iota(data1.begin(), data1.end(), 0);

        V v0(s, &data0[0]);
        V v1(s, &data1[0]);

        EXPECT_FALSE(any(v0));
        EXPECT_TRUE(any(v1));
        EXPECT_FALSE(all(v0));
        EXPECT_FALSE(all(v1));
        v0 += 1;
        v1 += 1;
        EXPECT_TRUE(all(v0));
        EXPECT_TRUE(all(v1));

        EXPECT_EQ(shape_t<2>(minmax(v0)()), (shape_t<2>{1,1}));
        EXPECT_EQ(shape_t<2>(minmax(v1)()), (shape_t<2>{1,24}));
        EXPECT_EQ(sum(v0)(), 24);
        EXPECT_EQ(sum(v1)(), 300);
        EXPECT_EQ(prod(v0)(), 1);
        EXPECT_NEAR(prod(xt::cast<double>(v1))(), 6.204484017332394e+23, 1e-13);
        EXPECT_EQ(norm_sq(v0)(), 24);
        EXPECT_EQ(norm_sq(v1)(), 4900);
        EXPECT_EQ(norm_l0(v1)(), 24.0);
        EXPECT_EQ(norm_l1(v1)(), 300.0);
        EXPECT_EQ(norm_l2(v1)(), 70.0);
        EXPECT_EQ(norm_linf(v1)(), 24.0);

        v1[{0, 0, 0}] = 0;
        EXPECT_EQ(norm_l0(v1)(), 23.0);

        constexpr auto M = (N == runtime_size) ? N : N-1;
        array_nd<T, M> a{{1, 5},
                         {3, 2},
                         {4, 7}};
        auto sc = xt::eval(sum(a.view(), {0}));
        auto sr = xt::eval(sum(a.view(), {1}));

        xt::xarray<T> ec{8, 14},
                      er{6, 5, 11};
        EXPECT_EQ(sc, ec);
        EXPECT_EQ(sr, er);
    }

    TYPED_TEST(array_nd_test, vector_value_type)
    {
        using A = TypeParam;
        using T = typename A::value_type;
        using V = typename A::view_type;
        using W = tiny_vector<T, 3>;
        constexpr auto N = A::ndim;

        std::vector<T> data1(prod(s));
        std::iota(data1.begin(), data1.end(), 0);

        W data[24];
        std::iota(data, data + 24, 0);

        view_nd<W, N> v(s, data);

        int count = 0;
        for (int i = 0; i < s[0]; ++i)
            for (int j = 0; j < s[1]; ++j)
                for (int k = 0; k < s[2]; ++k, ++count)
                {
                    EXPECT_EQ((v[{i, j, k}]), W(3, count));
                    EXPECT_EQ(v[count], W(3, count));
                }

        auto v1 = v.expand_elements(3);
        EXPECT_EQ(v1.dimension(), 4);
        EXPECT_EQ(v1.shape(), s.insert(3, 3));
        EXPECT_EQ(v1.channel_axis(), 3);
        EXPECT_TRUE(v1.is_consecutive());

        v1.bind(3, 1) += 1;
        v1.bind(3, 2) += 2;
        count = 0;
        for (int i = 0; i < s[0]; ++i)
            for (int j = 0; j < s[1]; ++j)
                for (int k = 0; k < s[2]; ++k, ++count)
                {
                    EXPECT_EQ((v[{i, j, k}]), (W{ T(count), T(count+1), T(count+2) }));
                }

        auto v2 = v.bind_channel(0);

        EXPECT_EQ(v2.dimension(), 3);
        EXPECT_EQ(v2.shape(), s);
        EXPECT_FALSE(v2.has_channel_axis());
        EXPECT_FALSE(v2.is_consecutive());

        EXPECT_TRUE(v1.bind(3, 0) == v2);
        EXPECT_TRUE(v1.bind(3, 1) == v.bind_channel(1));
        EXPECT_TRUE(v1.bind_channel(2) == v.bind_channel(2));

        auto v3 = v1.ensure_channel_axis(3);
        EXPECT_EQ(v3.dimension(), v1.dimension());
        EXPECT_EQ(v3.channel_axis(), 3);

        auto v4 = v1.ensure_channel_axis(0);
        EXPECT_EQ(v4.channel_axis(), 0);
        EXPECT_EQ(v4.dimension(), v1.dimension());

        auto v5 = v.ensure_channel_axis(0);
        EXPECT_EQ(v5.channel_axis(), 0);
        EXPECT_EQ(v5.dimension(), v.dimension()+1);

        V vs(s, &data1[0]);
        EXPECT_FALSE(vs.has_channel_axis());
        auto v6 = vs.ensure_channel_axis(3);
        EXPECT_TRUE(v6.has_channel_axis());
        EXPECT_EQ(v6.channel_axis(), 3);
        EXPECT_EQ(v6.dimension(), vs.dimension() + 1);

        view_nd<T, 4> vsized = v6;
        EXPECT_EQ(vsized.channel_axis(), 3);
        EXPECT_EQ(vsized.dimension(), 4);
    }

} // namespace xvigra
