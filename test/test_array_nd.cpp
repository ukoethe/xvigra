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
#include <xvigra/array_nd.hpp>
// #include <vigra2/gaussians.hxx>

namespace xvigra
{
    constexpr index_t ndim = 3;
    shape_t<> s{4, 3, 2};

    using array_nd_types = testing::Types<array_nd<ndim, uint8_t>
                                         >;

    TYPED_TEST_SETUP(array_nd_test, array_nd_types);

    TYPED_TEST(array_nd_test, constructor)
    {
        using A = TypeParam;
        using T = typename A::value_type;
        using S = typename A::shape_type;
        using V = typename A::view_type;

        std::vector<T> data0(prod(s), 0), data1(prod(s));
        std::iota(data1.begin(), data1.end(), 0);

        V v0;

        EXPECT_EQ(v0.shape(), S());
        EXPECT_EQ(v0.strides(), S());
        EXPECT_EQ(v0.data(), (T*)0);

        V v1(s, &data1[0], tags::c_order);

        EXPECT_EQ(v1.shape(), s);
        EXPECT_EQ(v1.strides(), (S{ 6,2,1 }));
        EXPECT_EQ(v1.data(), &data1[0]);
        EXPECT_TRUE(v1.is_consecutive());
        EXPECT_FALSE(v1.owns_memory());
        bool memory_check = v1.memory_range() == tiny_vector<char*, 2>{(char*)&data1.front(), (char*)(1+&data1.back())};
        EXPECT_TRUE(memory_check);

        EXPECT_TRUE(v1 == v1);
        EXPECT_FALSE(v1 != v1);
        EXPECT_TRUE(v1 != v0);
        EXPECT_FALSE(v1 == v0);
        // A a(s);

        // EXPECT_EQ(a.shape(), s);

        // A b(a + 1);
    }

#if 0
template <int N>
struct ArrayNDTest
{
    typedef ArrayND<N, int>                            Array;
    typedef ArrayViewND<N, int>                        View;
    typedef ArrayViewND<N, const int>                  ConstView;
    typedef TinyArray<int, 3>                          Vector;
    typedef ArrayViewND<N, Vector>                     VectorView;
    typedef Shape<N>                                   S;
    typedef ArrayND<(N == runtime_size ? N : 1), int>  Array1D;

    S s{ 4,3,2 };
    std::vector<int> data0, data1;

    ArrayNDTest()
    : data0(prod(s),0)
    , data1(prod(s))
    {
        std::iota(data1.begin(), data1.end(), 0);
    }

    void testConstruction()
    {
        EXPECT_TRUE(ArrayNDConcept<View>::value);
        EXPECT_TRUE(ArrayNDConcept<View &>::value);
        EXPECT_TRUE(ArrayNDConcept<View &&>::value);
        EXPECT_TRUE(ArrayNDConcept<View const &>::value);
        EXPECT_TRUE(ArrayLikeConcept<View>::value);
        EXPECT_TRUE(ArrayLikeConcept<View &>::value);
        EXPECT_TRUE(ArrayLikeConcept<View &&>::value);
        EXPECT_TRUE(ArrayLikeConcept<View const &>::value);
        EXPECT_TRUE(NDimConcept<View>::value);
        EXPECT_TRUE(NDimConcept<View &>::value);
        EXPECT_TRUE(NDimConcept<View &&>::value);
        EXPECT_TRUE(NDimConcept<View const &>::value);
        EXPECT_EQ(N, NDimTraits<View>::value);
        EXPECT_EQ(N, NDimTraits<View &>::value);
        EXPECT_EQ(N, NDimTraits<View &&>::value);
        EXPECT_EQ(N, NDimTraits<View const &>::value);

        View v0;

        EXPECT_EQ(v0.shape(), S());
        EXPECT_EQ(v0.strides(), S());
        EXPECT_EQ(v0.data(), (int*)0);

        View v1(s, &data1[0], tags::c_order);

        EXPECT_EQ(v1.shape(), s);
        EXPECT_EQ(v1.strides(), (S{ 6,2,1 }));
        EXPECT_EQ(v1.data(), &data1[0]);
        EXPECT_TRUE(v1.isConsecutive());
        EXPECT_FALSE(v1.ownsMemory());
        EXPECT_EQ(v1.memoryRange(), (TinyArray<char*, 2>{(char*)&data1.front(), (char*)(1+&data1.back())}));

        EXPECT_TRUE(v1 == v1);
        EXPECT_FALSE(v1 != v1);
        EXPECT_TRUE(v1 != v0);
        EXPECT_FALSE(v1 == v0);

        auto iter1 = v1.begin(tags::c_order), end1 = v1.end(tags::c_order);

        int count = 0;
        for (int i = 0; i < s[0]; ++i)
            for (int j = 0; j < s[1]; ++j)
                for (int k = 0; k < s[2]; ++k, ++count, ++iter1)
                {
                    EXPECT_TRUE((v1.isInside(S{ i,j,k })));
                    EXPECT_FALSE((v1.isOutside(S{ i,j,k })));
                    EXPECT_EQ((v1[{i, j, k}]), count);
                    EXPECT_EQ(v1[count], count);
                    EXPECT_EQ(v1(i, j, k), count);
                    EXPECT_EQ(*iter1, count);
                    EXPECT_FALSE(iter1 == end1);
                }
        EXPECT_FALSE(v1.isInside(S{ -1,-1,-1 }));
        EXPECT_TRUE(v1.isOutside(S{ -1,-1,-1 }));
        EXPECT_TRUE(iter1 == end1);

        View v2(s, defaultAxistags(3, false, F_ORDER), &data1[0], F_ORDER);

        EXPECT_EQ(v2.shape(), s);
        EXPECT_EQ(v2.strides(), (S{ 1,4,12 }));
        EXPECT_EQ(v2.data(), &data1[0]);
        EXPECT_TRUE(v2.isConsecutive());
        EXPECT_FALSE(v2.ownsMemory());
        EXPECT_EQ(v2.axistags(), (AxisTags<N>{ tags::axis_x, tags::axis_y , tags::axis_z }));

        EXPECT_TRUE(v1 != v2);
        EXPECT_FALSE(v1 == v2);

        auto iter2 = v2.begin(F_ORDER), end2 = v2.end(F_ORDER);

        count = 0;
        for (int i = 0; i < s[2]; ++i)
            for (int j = 0; j < s[1]; ++j)
                for (int k = 0; k < s[0]; ++k, ++count, ++iter2)
                {
                    EXPECT_EQ((v2[{k, j, i}]), count);
                    EXPECT_EQ(v2[count], count);
                    EXPECT_EQ(*iter2, count);
                    EXPECT_FALSE(iter2 == end2);
                }
        EXPECT_TRUE(iter2 == end2);

        View v3(s, S{3, 1, 12}, &data1[0]);

        EXPECT_EQ(v3.shape(), s);
        EXPECT_EQ(v3.strides(), (S{ 3,1,12 }));
        EXPECT_EQ(v3.data(), &data1[0]);
        EXPECT_TRUE(v3.isConsecutive());

        auto iter3 = v3.begin(), end3 = v3.end();

        count = 0;
        for (int i = 0; i < s[2]; ++i)
            for (int j = 0; j < s[0]; ++j)
                for (int k = 0; k < s[1]; ++k, ++count, ++iter3)
                {
                    EXPECT_EQ((v3[{j, k, i}]), count);
                    EXPECT_EQ(v3[count], count);
                    EXPECT_EQ(*iter3, count);
                    EXPECT_FALSE(iter3 == end3);
                }
        EXPECT_TRUE(iter3 == end3);

        Array a1(v3);
        EXPECT_TRUE(a1 == v3);
        Array a2(v3, F_ORDER);
        EXPECT_TRUE(a2 == v3);
        Array a3(transpose(v3));
        EXPECT_TRUE(transpose(a3) == v3);
        Array a4(transpose(v3), F_ORDER);
        EXPECT_TRUE(transpose(a4) == v3);

        auto d = a1.data();
        Array a5(std::move(a1));
        EXPECT_FALSE(a1.hasData());
        EXPECT_EQ(a1.shape(), Shape<N>());
        EXPECT_EQ(a1.size(), 0);
        EXPECT_TRUE(a5 == v3);
        EXPECT_EQ(d, a5.data());

        Array1D a6{ 0,1,2,3 };
        EXPECT_EQ(a6.ndim(), 1);
        EXPECT_EQ(a6.size(), 4);
        for (int k = 0; k < 4; ++k)
            EXPECT_EQ(a6(k), k);

        Array a7(s, { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23 });
        EXPECT_EQ(a7.shape(), s);
        EXPECT_TRUE(a7 == v1);
    }

    void testBind()
    {
        View v0(s, &data1[0]);

        v0.setAxistags(defaultAxistags(3, true));
        EXPECT_TRUE(v0.hasChannelAxis());
        EXPECT_EQ(v0.channelAxis(), 2);

        auto v1 = v0.bind(0, 2);

        EXPECT_EQ(v1.shape(), (Shape<2>{3, 2}));
        EXPECT_TRUE(v1.hasChannelAxis());
        EXPECT_EQ(v1.channelAxis(), 1);
        EXPECT_EQ(v1.axistags(), (AxisTags<>{tags::axis_x, tags::axis_c}));
        EXPECT_TRUE(v1.isConsecutive());
        EXPECT_FALSE(v1.ownsMemory());

        int count = 0;
        for (int i = 0; i < s[0]; ++i)
            for (int j = 0; j < s[1]; ++j)
                for (int k = 0; k < s[2]; ++k, ++count)
                    if (i == 2)
                        EXPECT_EQ((v1[{j, k}]), count);

        auto v2 = v0.bind(1, 1);

        EXPECT_EQ(v2.shape(), (Shape<2>{4, 2}));
        EXPECT_TRUE(v2.hasChannelAxis());
        EXPECT_EQ(v2.channelAxis(), 1);
        EXPECT_EQ(v2.axistags(), (AxisTags<>{ tags::axis_y, tags::axis_c }));
        EXPECT_FALSE(v2.isConsecutive());

        count = 0;
        for (int i = 0; i < s[0]; ++i)
            for (int j = 0; j < s[1]; ++j)
                for (int k = 0; k < s[2]; ++k, ++count)
                    if (j == 1)
                        EXPECT_EQ((v2[{i, k}]), count);

        auto v3 = v0.bind(2, 0);

        EXPECT_EQ(v3.shape(), (Shape<2>{4, 3}));
        EXPECT_FALSE(v3.hasChannelAxis());
        EXPECT_EQ(v3.axistags(), (AxisTags<>{ tags::axis_y, tags::axis_x }));
        EXPECT_FALSE(v3.isConsecutive());

        count = 0;
        for (int i = 0; i < s[0]; ++i)
            for (int j = 0; j < s[1]; ++j)
                for (int k = 0; k < s[2]; ++k, ++count)
                    if (k == 0)
                        EXPECT_EQ((v3[{i, j}]), count);

        auto v4 = v0.bind(0, 3).bind(0, 2).bind(0, 1);
        EXPECT_EQ(v4.shape(), Shape<1>(1));
        EXPECT_EQ(v4[{0}], (v0[{3, 2, 1}]));
        EXPECT_FALSE(v4.hasChannelAxis());
        EXPECT_EQ(v4.axistags(), (AxisTags<>{ tags::axis_unknown }));
        EXPECT_TRUE(v4.isConsecutive());

        auto v5 = v0.bindLeft(Shape<2>{3, 2});
        EXPECT_EQ(v5.shape(), Shape<1>(2));
        EXPECT_EQ(v5[0], (v0[{3, 2, 0}]));
        EXPECT_EQ(v5[1], (v0[{3, 2, 1}]));
        EXPECT_TRUE(v5.hasChannelAxis());
        EXPECT_EQ(v5.channelAxis(), 0);
        EXPECT_EQ(v5.axistags(), (AxisTags<>{ tags::axis_c }));
        EXPECT_TRUE(v0.bind(0, 3).bind(0, 2) == v5);
        EXPECT_TRUE((v0.bindLeft(Shape<>{3, 2}) == v5));
        EXPECT_TRUE((v0.bindLeft(Shape<>{}) == v0));
        EXPECT_TRUE((v0.bindLeft(Shape<0>()) == v0));

        auto v6 = v0.bindRight(Shape<2>{1, 0});
        EXPECT_EQ(v6.shape(), Shape<1>(4));
        EXPECT_EQ(v6[0], (v0[{0, 1, 0}]));
        EXPECT_EQ(v6[1], (v0[{1, 1, 0}]));
        EXPECT_EQ(v6[2], (v0[{2, 1, 0}]));
        EXPECT_EQ(v6[3], (v0[{3, 1, 0}]));
        EXPECT_FALSE(v6.hasChannelAxis());
        EXPECT_EQ(v6.axistags(), (AxisTags<>{ tags::axis_y }));
        EXPECT_TRUE(v0.bind(1,1).bind(1,0) == v6);
        EXPECT_TRUE((v0.bindRight(Shape<>{1,0}) == v6));
        EXPECT_TRUE((v0.bindRight(Shape<>{}) == v0));
        EXPECT_TRUE((v0.bindRight(Shape<0>()) == v0));

        auto v7 = v0.template view<runtime_size>();
        EXPECT_EQ(decltype(v7)::actual_dimension, -1);
        EXPECT_TRUE(v7 == v0);

        auto v8 = v0.template view<3>();
        EXPECT_EQ(decltype(v8)::actual_dimension, 3);
        EXPECT_TRUE(v8 == v0);

        auto v9 = v0.diagonal();
        EXPECT_EQ(v9.shape(), Shape<1>(2));
        EXPECT_EQ(v9[0], (v0[{0, 0, 0}]));
        EXPECT_EQ(v9[1], (v0[{1, 1, 1}]));
    }

    void testSubarray()
    {
        View v0(s, &data1[0]);
        v0.setChannelAxis(2);

        auto v1 = v0.subarray(S{ 0,0,0 }, v0.shape());
        EXPECT_TRUE(v0 == v1);
        EXPECT_EQ(v1.channelAxis(), 2);

        auto v2 = v0.subarray(S{ 1,0,0 }, S{ 3,2,2 } );
        EXPECT_EQ(v2.shape(), (S{ 2,2,2 }));
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                for (int k = 0; k < 2; ++k)
                    EXPECT_EQ((v2[{i, j, k}]), (v0[{i + 1, j, k}]));
        EXPECT_TRUE((v2 == v0.subarray(S{ 1,0,0 }, S{ -1,-1, 2 })));
        EXPECT_TRUE((v2 == v0.subarray(S{ -3,0,0 }, S{ -1,-1, 2 })));

        try
        {
            v0.subarray(S{ 1,0,0 }, S{ 3,2,4 });
            failTest("no exception thrown");
        }
        catch (std::exception & c)
        {
            std::string expected("\nPrecondition violation!\nArrayViewND::subarray(): invalid subarray limits.");
            std::string message(c.what());
            EXPECT_TRUE(0 == expected.compare(message.substr(0, expected.size())));
        }

        try
        {
            v0.subarray(S{ 1,0,0 }, S{ 0,2,2 });
            failTest("no exception thrown");
        }
        catch (std::exception & c)
        {
            std::string expected("\nPrecondition violation!\nArrayViewND::subarray(): invalid subarray limits.");
            std::string message(c.what());
            EXPECT_TRUE(0 == expected.compare(message.substr(0, expected.size())));
        }

        try
        {
            v0.subarray(S{ -5,0,0 }, S{ 3,2,2 });
            failTest("no exception thrown");
        }
        catch (std::exception & c)
        {
            std::string expected("\nPrecondition violation!\nArrayViewND::subarray(): invalid subarray limits.");
            std::string message(c.what());
            EXPECT_TRUE(0 == expected.compare(message.substr(0, expected.size())));
        }
    }

    void testTranspose()
    {
        {
            // low-level tests
            S res = detail::permutationToOrder(S{ 6,2,1 }, tags::c_order);
            EXPECT_EQ(res, (S{ 0,1,2 }));
            res = detail::permutationToOrder(S{ 1,4,12 }, tags::c_order);
            EXPECT_EQ(res, (S{ 2,1,0 }));
            res = detail::permutationToOrder(S{ 3, 1, 12 }, tags::c_order);
            EXPECT_EQ(res, (S{ 2,0,1 }));
            res = detail::permutationToOrder(S{ 3,1,0 }, tags::c_order);
            EXPECT_EQ(res, (S{ 2,0,1 }));
            res = detail::permutationToOrder(S{ 3,0,1 }, tags::c_order);
            EXPECT_EQ(res, (S{ 1,0,2 }));
            res = detail::permutationToOrder(S{ 0,1,0 }, tags::c_order);
            EXPECT_EQ(res, (S{ 0,2,1 }));
        }

        {
            View v(s, &data1[0], tags::c_order);
            v.setChannelAxis(2);
            View t = v.transpose(tags::c_order);

            EXPECT_EQ(t.shape(), s);
            EXPECT_EQ(t.strides(), (S{ 6,2,1 }));
            EXPECT_EQ(t.channelAxis(), 2);

            View tt = v.transpose();
            EXPECT_EQ(tt.shape(), reversed(s));
            EXPECT_EQ(tt.strides(), (S{ 1,2,6 }));
            EXPECT_EQ(tt.channelAxis(), 0);

            View v1(reversed(s), &data1[0], F_ORDER);
            EXPECT_TRUE(v1 != v);
            EXPECT_FALSE(v1 == v);
            EXPECT_TRUE(v1 == v.transpose());
            EXPECT_FALSE(v1 != v.transpose());
        }
        {
            View v(s, &data1[0], F_ORDER);
            v.setChannelAxis(2);
            View t = v.transpose(tags::c_order);

            EXPECT_EQ(t.shape(), (S{ 2,3,4 }));
            EXPECT_EQ(t.strides(), (S{ 12,4,1 }));
            EXPECT_EQ(t.channelAxis(), 0);

            t = v.transpose();
            EXPECT_EQ(t.shape(), reversed(s));
            EXPECT_EQ(t.strides(), (S{ 12,4,1 }));
            EXPECT_EQ(t.channelAxis(), 0);
        }
        {
            View v(s, S{ 3, 1, 12 }, &data1[0]);
            v.setChannelAxis(2);
            View t = v.transpose(tags::c_order);

            EXPECT_EQ(t.shape(), (S{ 2,4,3 }));
            EXPECT_EQ(t.strides(), (S{ 12,3,1 }));
            EXPECT_EQ(t.channelAxis(), 0);

            View tt = v.transpose();
            EXPECT_EQ(tt.shape(), reversed(s));
            EXPECT_EQ(tt.strides(), (S{ 12,1,3 }));
            EXPECT_EQ(tt.channelAxis(), 0);
        }
        {
            View v(S{ 4,6,1 }, S{ 6,1,1 }, defaultAxistags(3), &data1[0]);
            View t = v.transpose(tags::c_order);

            EXPECT_EQ(t.shape(), (S{ 1,4,6 }));
            EXPECT_EQ(t.strides(), (S{ 0,6,1 }));
            EXPECT_EQ(t.axistags(), (AxisTags<N>{tags::axis_x, tags::axis_z, tags::axis_y}));
        }
        {
            View v(S{ 4,1,6 }, S{ 1,1,6 }, &data1[0]);
            v.setChannelAxis(2);
            View t = v.transpose(tags::c_order);

            EXPECT_EQ(t.shape(), (S{ 1,6,4 }));
            EXPECT_EQ(t.strides(), (S{ 0,6,1 }));
            EXPECT_EQ(t.channelAxis(), 1);
        }
        {
            View v(S{ 1,24,1 }, S{ 1,1,1 }, &data1[0]);
            v.setChannelAxis(1);
            View t = v.transpose(tags::c_order);

            EXPECT_EQ(t.shape(), (S{ 1,1,24 }));
            EXPECT_EQ(t.strides(), (S{ 0,0,1 }));
            EXPECT_EQ(t.channelAxis(), 2);
        }
    }

    void testAssignment()
    {
        View v0(s, &data1[0]);
        View v1(s, &data0[0]);
        v1.setChannelAxis(2);

        int count = 0;
        for (int k = 0; k < v0.size(); ++k, ++count)
        {
            EXPECT_EQ(v1[k], 0);
            EXPECT_EQ(v0[k], count);
        }

        v0.swapData(v1);
        EXPECT_EQ(v0.data(), &data1[0]);
        EXPECT_EQ(v1.data(), &data0[0]);

        count = 0;
        for (int k = 0; k < v0.size(); ++k, ++count)
        {
            EXPECT_EQ(v0[k], 0);
            EXPECT_EQ(v1[k], count);
        }

        v0.init(2);
        for (int k = 0; k < v0.size(); ++k)
            EXPECT_EQ(v0[k], 2);

        v0 = 1;
        for (int k = 0; k < v0.size(); ++k)
            EXPECT_EQ(v0[k], 1);

        v0 += 2;
        for (int k = 0; k < v0.size(); ++k)
            EXPECT_EQ(v0[k], 3);

        v0 -= 1;
        for (int k = 0; k < v0.size(); ++k)
            EXPECT_EQ(v0[k], 2);

        v0 *= 5;
        for (int k = 0; k < v0.size(); ++k)
            EXPECT_EQ(v0[k], 10);

        v0 /= 10;
        for (int k = 0; k < v0.size(); ++k)
            EXPECT_EQ(v0[k], 1);

        v0 *= v1;
        EXPECT_EQSequence(data1.begin(), data1.end(), data0.begin());

        v0 += v1;
        v0 /= 2;
        EXPECT_EQSequence(data1.begin(), data1.end(), data0.begin());

        v0 += v1;
        v0 -= v1;
        EXPECT_EQSequence(data1.begin(), data1.end(), data0.begin());

        v0 += 1;
        v0 /= v0;
        for (int k = 0; k < v0.size(); ++k)
            EXPECT_EQ(v0[k], 1);

        v0 -= v0;
        for (int k = 0; k < v0.size(); ++k)
            EXPECT_EQ(v0[k], 0);

        v0 = v1;
        EXPECT_EQSequence(data1.begin(), data1.end(), data0.begin());

        View v;
        EXPECT_FALSE(v.isConsecutive());
        EXPECT_EQ(v.channelAxis(), -1);

        v = v1;
        EXPECT_EQ(v.shape(), v1.shape());
        EXPECT_EQ(v.strides(), v1.strides());
        EXPECT_EQ(v.data(), v1.data());
        EXPECT_EQ(v.channelAxis(), 2);
        EXPECT_TRUE(v.isConsecutive());

        try
        {
            v1 = v1.transpose();
            failTest("no exception thrown");
        }
        catch (std::exception & c)
        {
            std::string expected("\nPrecondition violation!\nArrayViewND::operator=(ArrayViewND const &): shape mismatch.");
            std::string message(c.what());
            EXPECT_TRUE(0 == expected.compare(message.substr(0, expected.size())));
        }

        try
        {
            v1 += v1.transpose();
            failTest("no exception thrown");
        }
        catch (std::exception & c)
        {
            std::string expected("\nPrecondition violation!\nArrayViewND::operator+=(ArrayViewND const &): shape mismatch.");
            std::string message(c.what());
            EXPECT_TRUE(0 == expected.compare(message.substr(0, expected.size())));
        }

        try
        {
            v1 -= v1.transpose();
            failTest("no exception thrown");
        }
        catch (std::exception & c)
        {
            std::string expected("\nPrecondition violation!\nArrayViewND::operator-=(ArrayViewND const &): shape mismatch.");
            std::string message(c.what());
            EXPECT_TRUE(0 == expected.compare(message.substr(0, expected.size())));
        }

        try
        {
            v1 *= v1.transpose();
            failTest("no exception thrown");
        }
        catch (std::exception & c)
        {
            std::string expected("\nPrecondition violation!\nArrayViewND::operator*=(ArrayViewND const &): shape mismatch.");
            std::string message(c.what());
            EXPECT_TRUE(0 == expected.compare(message.substr(0, expected.size())));
        }

        try
        {
            v1 /= v1.transpose();
            failTest("no exception thrown");
        }
        catch (std::exception & c)
        {
            std::string expected("\nPrecondition violation!\nArrayViewND::operator/=(ArrayViewND const &): shape mismatch.");
            std::string message(c.what());
            EXPECT_TRUE(0 == expected.compare(message.substr(0, expected.size())));
        }
    }

    void testOverlappingMemory()
    {
        using namespace tags;
        View v(S{4,2,2}, &data1[0]);

        const int M = (N == runtime_size)
                          ? N
                          : 2;
        auto v1 = v.reshape(Shape<M>{4, 4}, AxisTags<M>{axis_y, axis_x});
        EXPECT_EQ(v1.axistags(), (AxisTags<M>{axis_y, axis_x}));
        ArrayND<M, int> vs = v1;

        int count = 0;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j, ++count)
                EXPECT_EQ((v1[{i, j}]), count);

        v1 = v1.transpose();
        EXPECT_EQ(v1.axistags(), (AxisTags<M>{axis_y, axis_x}));

        count = 0;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j, ++count)
                EXPECT_EQ((v1[{j, i}]), count);

        auto v2 = v.reshape(Shape<M>{4, 4}, F_ORDER);
        EXPECT_EQ(v2.axistags(), (AxisTags<M>{axis_unknown, axis_unknown}));

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

    void testFunctions()
    {
        View v0(s, &data0[0]);
        View v1(s, &data1[0]);

        EXPECT_FALSE(v0.any());
        EXPECT_TRUE(v1.any());
        EXPECT_FALSE(v0.all());
        EXPECT_FALSE(v1.all());
        v0 += 1;
        v1 += 1;
        EXPECT_TRUE(v0.all());
        EXPECT_TRUE(v1.all());

        EXPECT_EQ(v0.minmax(), Shape<2>(1,1));
        EXPECT_EQ(v1.minmax(), Shape<2>(1,24));
        EXPECT_EQ(v0.sum(), 24);
        EXPECT_EQ(sum(v0, 0.0), 24);
        EXPECT_EQ(v1.template sum<double>(), 300);
        EXPECT_EQ(sum(v1), 300);
        EXPECT_EQ(v0.prod(), 1);
        EXPECT_EQ(prod(v0), 1);
        EXPECT_EQTolerance(v1.template prod<double>(), 6.204484017332394e+23, 1e-13);
        EXPECT_EQTolerance(prod(v1, 1.0), 6.204484017332394e+23, 1e-13);
        EXPECT_EQ(squaredNorm(v0), 24);
        EXPECT_EQ(squaredNorm(v1), 4900);
        EXPECT_EQ(norm(v1), 70.0);
        EXPECT_EQ(norm(v1, -1), 24.0);
        EXPECT_EQ(norm(v1, 0), 24.0);
        EXPECT_EQ(norm(v1, 1), 300.0);
        EXPECT_EQ(norm(v1, 2), 70.0);

        v1[{0, 0, 0}] = 0;
        EXPECT_EQ(norm(v1, 0), 23.0);


        double data[] = { 1.0, 5.0,
                          3.0, 2.0,
                          4.0, 7.0 };
        ArrayND<2, double> a({ 3,2 }, data);

        double columnSum[] = { 8.0, 14.0 };
        double rowSum[] = { 6.0, 5.0, 11.0 };
        auto as0 = a.sum(tags::axis = 0);
        auto as1 = a.sum(tags::axis = 1);
        EXPECT_EQSequence(columnSum, columnSum + 2, as0.begin());
        EXPECT_EQSequence(rowSum, rowSum + 3, as1.begin());

        double columnMean[] = { 8 / 3.0, 14 / 3.0 };
        double rowMean[] = { 3.0, 2.5, 5.5 };
        auto am0 = a.mean(tags::axis = 0);
        auto am1 = a.mean(tags::axis = 1);
        EXPECT_EQSequence(columnMean, columnMean + 2, am0.begin());
        EXPECT_EQSequence(rowMean, rowMean + 3, am1.begin());
    }

    void testVectorValuetype()
    {
        Vector data[24];
        std::iota(data, data + 24, 0);

        VectorView v(s, data);

        int count = 0;
        for (int i = 0; i < s[0]; ++i)
            for (int j = 0; j < s[1]; ++j)
                for (int k = 0; k < s[2]; ++k, ++count)
                {
                    EXPECT_EQ((v[{i, j, k}]), (Vector{ count, count, count }));
                    EXPECT_EQ(v[count], (Vector{ count, count, count }));
                }

        auto v1 = v.expandElements(3);
        EXPECT_EQ(v1.ndim(), 4);
        EXPECT_EQ(v1.shape(), s.insert(3, 3));
        EXPECT_EQ(v1.channelAxis(), 3);
        EXPECT_TRUE(v1.isConsecutive());

        v1.bind(3, 1) += 1;
        v1.bind(3, 2) += 2;
        count = 0;
        for (int i = 0; i < s[0]; ++i)
            for (int j = 0; j < s[1]; ++j)
                for (int k = 0; k < s[2]; ++k, ++count)
                {
                    EXPECT_EQ((v[{i, j, k}]), (Vector{ count, count+1, count+2 }));
                }

        auto v2 = v.bindChannel(0);

        EXPECT_EQ(v2.ndim(), 3);
        EXPECT_EQ(v2.shape(), s);
        EXPECT_EQ(v2.channelAxis(), tags::axis_missing);
        EXPECT_FALSE(v2.isConsecutive());

        EXPECT_TRUE(v1.bind(3, 0) == v2);
        EXPECT_TRUE(v1.bind(3, 1) == v.bindChannel(1));
        EXPECT_TRUE(v1.bindChannel(2) == v.bindChannel(2));

        auto v3 = v1.ensureChannelAxis(3);
        EXPECT_EQ(v3.ndim(), v1.ndim());
        EXPECT_EQ(v3.channelAxis(), 3);

        auto v4 = v1.ensureChannelAxis(0);
        EXPECT_EQ(v4.channelAxis(), 0);
        EXPECT_EQ(v4.ndim(), v1.ndim());

        auto v5 = v.ensureChannelAxis(0);
        EXPECT_EQ(v5.channelAxis(), 0);
        EXPECT_EQ(v5.ndim(), v.ndim()+1);

        View vs(s, &data1[0]);
        EXPECT_FALSE(vs.hasChannelAxis());
        auto v6 = vs.ensureChannelAxis(3);
        EXPECT_EQ(v6.channelAxis(), 3);
        EXPECT_EQ(v6.ndim(), vs.ndim() + 1);

        ArrayViewND<4, int> vsized = v6;
        EXPECT_EQ(vsized.channelAxis(), 3);
        EXPECT_EQ(vsized.ndim(), 4);
    }

    void testArray()
    {
        using namespace tags;

        Array a(s, defaultAxistags(3), 1);

        EXPECT_EQ(a.ndim(), 3);
        EXPECT_EQ(a.shape(), s);
        EXPECT_TRUE(a.isConsecutive());
        EXPECT_TRUE(a.ownsMemory());
        EXPECT_EQ(a.axistags(), (AxisTags<runtime_size>{ axis_z, axis_y, axis_x }));

        auto a1 = a;
        EXPECT_EQ(a1.ndim(), 3);
        EXPECT_EQ(a1.shape(), s);
        EXPECT_TRUE(a1.isConsecutive());
        EXPECT_TRUE(a1.ownsMemory());
        EXPECT_FALSE(a.data() == a1.data());
        EXPECT_TRUE(a1 == a);

        auto v = a.view();
        EXPECT_EQ(v.ndim(), 3);
        EXPECT_EQ(v.shape(), s);
        EXPECT_TRUE(v.isConsecutive());
        EXPECT_FALSE(v.ownsMemory());
        EXPECT_TRUE(a.data() == v.data());
        EXPECT_TRUE(a == v);
        EXPECT_EQ(v.axistags(), (AxisTags<N>{ axis_z, axis_y, axis_x }));

        Array a2 = v;
        EXPECT_EQ(a2.ndim(), 3);
        EXPECT_EQ(a2.shape(), s);
        EXPECT_TRUE(a2.isConsecutive());
        EXPECT_TRUE(a2.ownsMemory());
        auto d = a2.data();
        EXPECT_FALSE(a.data() == d);
        EXPECT_TRUE(a2 == a);
        EXPECT_EQ(a2.axistags(), (AxisTags<>{ axis_z, axis_y, axis_x }));

        Array a3(s, defaultAxistags(3), &data1[0], F_ORDER);
        EXPECT_EQ(a3.ndim(), 3);
        EXPECT_EQ(a3.shape(), s);
        EXPECT_TRUE(a3.isConsecutive());
        EXPECT_TRUE(a3.ownsMemory());
        EXPECT_EQ(a3.axistags(), (AxisTags<>{ axis_z, axis_y, axis_x }));
        EXPECT_FALSE(a3 == a2);
        EXPECT_FALSE(a3.data() == data1.data());
        EXPECT_EQSequence(data1.begin(), data1.end(), a3.data());

        a2 = a3;
        EXPECT_EQ(a2.ndim(), 3);
        EXPECT_EQ(a2.shape(), s);
        EXPECT_TRUE(a2.isConsecutive());
        EXPECT_TRUE(a2.ownsMemory());
        EXPECT_EQ(a2.axistags(), (AxisTags<>{ axis_z, axis_y, axis_x }));
        EXPECT_EQ(a2.data(), d);
        EXPECT_FALSE(a2 == a);
        EXPECT_TRUE(a2 == a3);

        int count = 0;
        for (int k = 0; k < a.size(); ++k, ++count)
        {
            EXPECT_EQ(a[k], 1);
            EXPECT_EQ(a3[k], count);
        }

        swap(a, a3);

        count = 0;
        for (int k = 0; k < a.size(); ++k, ++count)
        {
            EXPECT_EQ(a3[k], 1);
            EXPECT_EQ(a[k], count);
        }

        auto p = a.data();
        a.resize(reversed(s), AxisTags<N>{axis_y, axis_c, axis_x});
        EXPECT_EQ(a.data(), p);
        EXPECT_EQ(a.shape(), reversed(s));
        EXPECT_EQ(a.axistags(), (AxisTags<N>{axis_y, axis_c, axis_x}));
        EXPECT_TRUE(a.isConsecutive());
        EXPECT_TRUE(a.ownsMemory());

        if (N == runtime_size)
        {
            a.resize(Shape<>{24});
            EXPECT_EQ(a.data(), p);
            EXPECT_EQ(a.shape(), Shape<>{24});
            EXPECT_EQ(a.axistags(), (AxisTags<N>{axis_unknown}));
            EXPECT_TRUE(a.isConsecutive());
            EXPECT_TRUE(a.ownsMemory());
            count = 0;
            for (int k = 0; k < a.size(); ++k, ++count)
            {
                EXPECT_EQ(a[k], count);
            }
        }

        a.resize(S{ 3,4,5 }, AxisTags<runtime_size>{axis_c, axis_x, axis_y}, F_ORDER);
        EXPECT_TRUE(a.data() != p);
        EXPECT_EQ(a.shape(), (S{ 3,4,5 }));
        EXPECT_EQ(a.axistags(), (AxisTags<N>{axis_c, axis_x, axis_y}));
        EXPECT_TRUE(a.isConsecutive());
        EXPECT_TRUE(a.ownsMemory());
        for (int k = 0; k < a.size(); ++k)
        {
            EXPECT_EQ(a[k], 0);
        }

        Array a4(4, 5, 6);
        EXPECT_EQ(a4.shape(), (S{ 4,5,6 }));

        p = a4.data();
        Array a5 = std::move(a4);
        EXPECT_TRUE(a5.hasData());
        EXPECT_EQ(a5.data(), p);
        EXPECT_EQ(a5.shape(), (S{ 4,5,6 }));

        EXPECT_FALSE(a4.hasData());
        EXPECT_EQ(a4.shape(), (S{}));

        a4 = std::move(a5);
        EXPECT_TRUE(a4.hasData());
        EXPECT_EQ(a4.data(), p);
        EXPECT_EQ(a4.shape(), (S{ 4,5,6 }));

        EXPECT_FALSE(a5.hasData());
        EXPECT_EQ(a5.shape(), (S{}));

        using namespace array_math;
        double sigma = 1.0, s2 = 0.5 / sq(sigma);
        Gaussian<> g(sigma);
        ArrayND<2, double> gaussian(s2 / M_PI * exp(-s2*elementwiseSquaredNorm(mgrid<2>({ 7,7 }) - Shape<2>{3, 3})));
        for (auto & c : gaussian.coordinates())
            EXPECT_EQTolerance(gaussian[c], g(c[0] - 3.0)*g(c[1] - 3.0), 1e-15);
    }

    void testIterators()
    {
        Array a1(s);

        std::iota(a1.begin(), a1.end(), 1);
        Array a2(a1), a3(a1);
        a2 += 1;
        a3 -= 1;
        auto iter = makeCoupledIterator(a1, a2.cview(), const_cast<Array const &>(a3));

        EXPECT_TRUE((std::is_same<IteratorND<PointerNDCoupledType<Array, ConstView, Array const>>, decltype(iter)>::value));

        EXPECT_TRUE((std::is_const<typename std::remove_reference<decltype(get<0>(iter))>::type>::value));
        EXPECT_TRUE((!std::is_const<typename std::remove_reference<decltype(get<1>(iter))>::type>::value));
        EXPECT_TRUE((std::is_const<typename std::remove_reference<decltype(get<2>(iter))>::type>::value));
        EXPECT_TRUE((std::is_const<typename std::remove_reference<decltype(get<3>(iter))>::type>::value));

        EXPECT_TRUE(get<0>(iter) == (S{ 0,0,0 }));
        EXPECT_TRUE(&get<1>(iter) == a1.data());
        EXPECT_TRUE(&get<2>(iter) == a2.data());
        EXPECT_TRUE(&get<3>(iter) == a3.data());

        int count = 1;
        for (int i = 0; i < s[0]; ++i)
            for (int j = 0; j < s[1]; ++j)
                for (int k = 0; k < s[2]; ++k, ++count, ++iter)
                {
                    EXPECT_EQ((a1[{i, j, k}]), count);
                    EXPECT_EQ((a2[{i, j, k}]), count + 1);
                    EXPECT_EQ((a3[{i, j, k}]), count - 1);
                    EXPECT_EQ(get<0>(iter), (S{ i,j,k }));
                    EXPECT_EQ(get<1>(iter), count);
                    EXPECT_EQ(get<2>(*iter), count + 1);
                    EXPECT_EQ(get<3>(iter), count - 1);
                    get<1>(iter) = 0;
                }
        EXPECT_FALSE(a1.any());

        auto fiter = makeCoupledIterator(F_ORDER, a2, a3);

        EXPECT_TRUE(get<0>(fiter) == (S{ 0,0,0 }));
        EXPECT_TRUE(&get<1>(fiter) == a2.data());
        EXPECT_TRUE(&get<2>(fiter) == a3.data());

        for (int i = 0; i < s[2]; ++i)
            for (int j = 0; j < s[1]; ++j)
                for (int k = 0; k < s[0]; ++k, ++fiter)
                {
                    EXPECT_EQ(get<0>(fiter), (S{ k, j, i }));
                    EXPECT_EQ(get<1>(fiter), (a2[{k, j, i}]));
                    EXPECT_EQ(get<2>(fiter), (a3[{k, j, i}]));
                    get<1>(fiter) = 0;
                }
        EXPECT_FALSE(a2.any());

        count = 0;
        for (auto & h : iter)
        {
            EXPECT_EQ(get<1>(h), 0);
            EXPECT_EQ(get<2>(h), 0);
            EXPECT_EQ(get<3>(h), count);
            const_cast<int &>(get<3>(h)) = 0;
            ++count;
        }
        EXPECT_EQ(count, 24);
        EXPECT_FALSE(a3.any());
    }
};

struct ArrayNDTestSuite
: public vigra::test_suite
{
    ArrayNDTestSuite()
    : vigra::test_suite("ArrayNDTest")
    {
        addTests<3>();
        addTests<runtime_size>();
    }

    template <int N>
    void addTests()
    {
        add(testCase(&ArrayNDTest<N>::testConstruction));
        add(testCase(&ArrayNDTest<N>::testBind));
        add(testCase(&ArrayNDTest<N>::testTranspose));
        add(testCase(&ArrayNDTest<N>::testAssignment));
        add(testCase(&ArrayNDTest<N>::testOverlappingMemory));
        add(testCase(&ArrayNDTest<N>::testFunctions));
        add(testCase(&ArrayNDTest<N>::testSubarray));
        add(testCase(&ArrayNDTest<N>::testVectorValuetype));
        add(testCase(&ArrayNDTest<N>::testArray));
        add(testCase(&ArrayNDTest<N>::testIterators));
    }
};
#endif

} // namespace xvigra
