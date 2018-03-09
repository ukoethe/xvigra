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

#ifndef XVIGRA_ENABLE_ASSERT
#define XVIGRA_ENABLE_ASSERT
#endif

#include <numeric>
#include <limits>
#include <iostream>
#include "unittest.hpp"
#include <xvigra/tiny_vector.hpp>

namespace xvigra
{
    static const unsigned SIZE = 3;

    using BV = tiny_vector<uint8_t, SIZE>;
    using IV = tiny_vector<int, SIZE>;
    using FV = tiny_vector<float, SIZE>;

    template <class T>
    struct tiny_vector_test_data
    {
        static T data[SIZE];
    };

    template <class T>
    T tiny_vector_test_data<T>::data[SIZE] = {1, 2, 4};

    template <>
    float tiny_vector_test_data<float>::data[SIZE] = { 1.2f, 2.4f, 4.6f};

    template <class T>
    class tiny_vector_test : public testing::Test
    {
    };

    typedef testing::Types<tiny_vector<uint8_t, SIZE>,
                           tiny_vector<int, SIZE>,
                           tiny_vector<float, SIZE>,
                           tiny_vector<int, runtime_size>,         // buffer_size > SIZE
                           tiny_vector<int, runtime_size, int[1]>  // buffer_size < SIZE
                          > tiny_vector_types;

    TYPED_TEST_CASE(tiny_vector_test, tiny_vector_types);

    TYPED_TEST(tiny_vector_test, construction)
    {
        using V = TypeParam;
        using T = typename V::value_type;
        using size_type = typename V::size_type;
        const bool fixed = V::has_fixed_size;

        T * data = tiny_vector_test_data<T>::data;
        int * idata = tiny_vector_test_data<int>::data;

        V v0,
          v1(SIZE, 1),
          v2(idata, idata+SIZE),
          v3(data, data+SIZE);

        EXPECT_EQ(v0.size(), fixed ? SIZE : 0);
        EXPECT_EQ(v1.size(), SIZE);
        EXPECT_EQ(v3.size(), SIZE);
        EXPECT_EQ(v0.empty(), !fixed);
        EXPECT_FALSE(v1.empty());
        EXPECT_FALSE(v3.empty());

        EXPECT_EQ(v3.front(), data[0]);
        EXPECT_EQ(v3.back(), data[SIZE-1]);
        V const & cv3 = v3;
        EXPECT_EQ(cv3.front(), data[0]);
        EXPECT_EQ(cv3.back(), data[SIZE-1]);

        auto v3iter   = v3.begin();
        auto v3citer  = v3.cbegin();
        auto v3riter  = v3.rbegin();
        auto v3criter = v3.crbegin();
        for(size_type k=0; k<v3.size(); ++k, ++v3iter, ++v3citer, ++v3riter, ++v3criter)
        {
            if(fixed)
            {
                EXPECT_EQ(v0[k], 0);
            }
            EXPECT_EQ(v1[k], 1);
            EXPECT_EQ(v3[k], data[k]);
            EXPECT_EQ(v3.at(k), data[k]);
            EXPECT_EQ(*v3iter, data[k]);
            EXPECT_EQ(*v3citer, data[k]);
            EXPECT_EQ(*v3riter, data[SIZE-1-k]);
            EXPECT_EQ(*v3criter, data[SIZE-1-k]);
        }
        EXPECT_EQ(v3iter, v3.end());
        EXPECT_EQ(v3citer, v3.cend());
        EXPECT_EQ(v3riter, v3.rend());
        EXPECT_EQ(v3criter, v3.crend());
        EXPECT_THROW(v3.at(SIZE), std::out_of_range);

        EXPECT_EQ(v3, V(v3));
        EXPECT_EQ(v3, V(v3.begin(), v3.end()));
        if(fixed)
        {
            EXPECT_THROW(V(v3.begin(), v3.begin()+SIZE-1), std::runtime_error);
            EXPECT_THROW((V{1,2}), std::runtime_error);
        }

        if(std::is_integral<T>::value)
        {
            EXPECT_EQ(v3, (V{1, 2, 4}));
        }
        if(fixed)
        {
            EXPECT_EQ(v1, (V{1}));
         }
        else
        {
            EXPECT_EQ((tiny_vector<T, 1>(1,1)), (V{1}));
        }

        V v;
        v.assign(SIZE, 1);
        EXPECT_EQ(v1, v);
        v.assign({1, 2, 4});
        EXPECT_EQ(v2, v);

        v = 1;
        EXPECT_EQ(v, v1);
        v = v3;
        EXPECT_EQ(v, v3);

        V v4(v1), v5(v3);
        swap(v4, v5);
        EXPECT_EQ(v3, v4);
        EXPECT_EQ(v1, v5);

        // testing move constructor and assignment
        v5 = v3.push_back(0).pop_back();
        EXPECT_EQ(v5, v3);
        EXPECT_EQ(V(v3.push_back(0).pop_back()), v3);

        // testing factory functions
        for (int k = 0; k<SIZE; ++k)
        {
            auto uv = V::template unit_vector<SIZE>(k);
            auto vv = V::unit_vector(SIZE, k);
            EXPECT_EQ(uv[k], 1);
            EXPECT_EQ(vv[k], 1);
            uv[k] = 0;
            vv[k] = 0;
            EXPECT_TRUE(uv == 0);
            EXPECT_TRUE(vv == 0);
        }

        V range_ref(SIZE);
        std::iota(range_ref.begin(), range_ref.end(), 0);
        EXPECT_EQ(V::range(SIZE), range_ref);
        EXPECT_EQ(V::range(0, SIZE), range_ref);
        range_ref = range_ref * 2 + 1;
        EXPECT_EQ(V::range(1, 2*SIZE, 2), range_ref);

        auto r = reversed(v3);
        for (int k = 0; k<SIZE; ++k)
            EXPECT_EQ(v3[k], r[SIZE - 1 - k]);

        EXPECT_EQ(transposed(r, V::range(SIZE - 1, -1, -1)), v3);
        EXPECT_EQ(transposed(r), v3);
    }

    TYPED_TEST(tiny_vector_test, subarray)
    {
        using V = TypeParam;
        using A = typename V::template rebind_size<runtime_size>;
        using T = typename V::value_type;

        T * data = tiny_vector_test_data<T>::data;
        V v3(data, data+SIZE);
        V const & cv3 = v3;

        EXPECT_EQ(v3, (v3.template subarray<0, SIZE>()));
        EXPECT_EQ(2u, (v3.template subarray<0, 2>().size()));
        EXPECT_EQ(v3[0], (v3.template subarray<0, 2>()[0]));
        EXPECT_EQ(v3[1], (v3.template subarray<0, 2>()[1]));
        EXPECT_EQ(2u, (v3.template subarray<1, 3>().size()));
        EXPECT_EQ(v3[1], (v3.template subarray<1, 3>()[0]));
        EXPECT_EQ(v3[2], (v3.template subarray<1, 3>()[1]));
        EXPECT_EQ(1u, (v3.template subarray<1, 2>().size()));
        EXPECT_EQ(v3[1], (v3.template subarray<1, 2>()[0]));
        EXPECT_EQ(1u, (v3.subarray(1, 2).size()));
        EXPECT_EQ(v3[1], (v3.subarray(1, 2)[0]));
        EXPECT_EQ(1u, (cv3.template subarray<1, 2>().size()));
        EXPECT_EQ(v3[1], (cv3.template subarray<1, 2>()[0]));
        EXPECT_EQ(1u, (cv3.subarray(1, 2).size()));
        EXPECT_EQ(v3[1], (cv3.subarray(1, 2)[0]));

        A r{ 2,3,4,5 };
        EXPECT_EQ(r, (A{ 2,3,4,5 }));
        EXPECT_EQ(r.subarray(1, 3).size(), 2u);
        EXPECT_EQ(r.subarray(1, 3), (A{ 3,4 }));
        EXPECT_EQ((r.template subarray<1, 3>().size()), 2u);
        EXPECT_EQ((r.template subarray<1, 3>()), (A{ 3,4 }));
    }

    TYPED_TEST(tiny_vector_test, erase_insert)
    {
        using V = TypeParam;
        using V1 = typename V::template rebind_size<SIZE - 1>;
        using T = typename V::value_type;

        T * data = tiny_vector_test_data<T>::data;
        V v3(data, data+SIZE);
        V1 v10(v3.begin(), v3.begin()+SIZE-1);

        EXPECT_EQ(v10, v3.erase(SIZE - 1));
        EXPECT_EQ(v3, v10.insert(SIZE - 1, v3[SIZE - 1]));
        EXPECT_EQ(v10, v3.pop_back());
        EXPECT_EQ(v3, v10.push_back(v3[SIZE - 1]));
        V1 v11(v3.begin() + 1, v3.begin() + SIZE);
        EXPECT_EQ(v11, v3.erase(0));
        EXPECT_EQ(v3, v11.insert(0, v3[0]));
        EXPECT_EQ(v11, v3.pop_front());
        EXPECT_EQ(v3, v11.push_front(v3[0]));
    }

    TYPED_TEST(tiny_vector_test, comparison)
    {
        using V = TypeParam;
        using T = typename V::value_type;
        const bool fixed = V::has_fixed_size;

        T * data = tiny_vector_test_data<T>::data;

        V v0(SIZE, 0),
          v1{1},
          v2(SIZE, 1),
          v3(data, data+SIZE);

        EXPECT_TRUE(v3 == v3);
        EXPECT_EQ(v1 == v2, fixed);
        EXPECT_TRUE(v1 == v1);
        EXPECT_TRUE(v1 == 1);
        EXPECT_TRUE(1 == v1);
        EXPECT_TRUE(v1 != v3);
        EXPECT_TRUE(v2 != v3);
        EXPECT_TRUE(v1 != 0);
        EXPECT_TRUE(0 != v1);
        EXPECT_TRUE(v2 != 0);
        EXPECT_TRUE(0 != v2);

        EXPECT_FALSE(v0 < v0);
        EXPECT_TRUE(v0 < v1);
        EXPECT_TRUE(v0 < v2);
        EXPECT_EQ(v1 < v2, !fixed);
        EXPECT_TRUE(v2 < v3);
        EXPECT_FALSE(v3 < v2);

        if(v1.size() == SIZE)
        {
            EXPECT_TRUE(all_less(v0, v1));
        }
        else
        {
            EXPECT_THROW(all_less(v0, v1), std::runtime_error);
        }
        EXPECT_EQ(all_less(v2, v3), !std::is_integral<T>::value);
        EXPECT_TRUE(all_greater(v2, v0));
        EXPECT_EQ(all_greater(v3, v2), !std::is_integral<T>::value);
        EXPECT_TRUE(all_less_equal(v0, v2));
        EXPECT_TRUE(all_less_equal(0, v0));
        EXPECT_TRUE(all_less_equal(v0, 0));
        EXPECT_TRUE(all_less_equal(v2, v3));
        EXPECT_TRUE(!all_less_equal(v3, v2));
        EXPECT_TRUE(all_greater_equal(v2, v0));
        EXPECT_TRUE(all_greater_equal(v3, v2));
        EXPECT_TRUE(!all_greater_equal(v2, v3));

        EXPECT_TRUE(all_close(v3, v3));
        EXPECT_FALSE(all_close(v2, v3));

        EXPECT_TRUE(!any(v0) && !all(v0) && any(v3) && all(v3));
        v3[0] = 0;
        EXPECT_TRUE(any(v3) && !all(v3));
    }

    TYPED_TEST(tiny_vector_test, ostream)
    {
        using V = TypeParam;
        using T = typename V::value_type;

        T * data = tiny_vector_test_data<T>::data;
        V v3(data, data+SIZE);

        std::ostringstream out;
        out << v3;
        if(std::is_integral<T>::value)
        {
            std::string expected("{1, 2, 4}");
            EXPECT_EQ(expected, out.str());
        }
    }

    TEST(tiny_vector, conversion)
    {
        IV iv{1,2,3},
           iv1(SIZE, 1);
        FV fv{1.1f,2.2f,3.3f},
           fv1{1.0};

        EXPECT_TRUE(iv1 == fv1);
        EXPECT_TRUE(fv1 == iv1);
        EXPECT_TRUE(iv != fv);
        EXPECT_TRUE(fv != iv);
        EXPECT_TRUE(iv == IV(fv));
        iv1 = fv;
        EXPECT_TRUE(iv1 == iv);
    }

    TEST(tiny_vector, interoperability)
    {
        using A = tiny_vector<int, 4>;
        using B = tiny_vector<int>;
        using C = tiny_vector<int, 4, int *>;
        using D = tiny_vector<int, runtime_size, int *>;
        using E = tiny_vector<int, runtime_size, xt::xbuffer_adaptor<int *>>;

        EXPECT_TRUE(tiny_vector_concept<A>::value);
        EXPECT_TRUE(tiny_vector_concept<B>::value);
        EXPECT_TRUE(tiny_vector_concept<C>::value);
        EXPECT_TRUE(tiny_vector_concept<D>::value);
        EXPECT_TRUE(tiny_vector_concept<E>::value);

        static const size_t s = 4;
        std::array<int, s> data{1,2,3,4};
        A a(data.begin());
        B b(a);
        C c(a);
        D d(a);
        E e(data.data(), s);
        std::vector<int> v(data.begin(), data.end());

        EXPECT_EQ(b, a);
        EXPECT_EQ(c, a);
        EXPECT_EQ(d, a);
        EXPECT_EQ(e, a);

        EXPECT_EQ(A(data), a);
        EXPECT_EQ(B(data), a);
        EXPECT_EQ(C(data), a);
        EXPECT_EQ(D(data), a);

        EXPECT_EQ(A(v), a);
        EXPECT_EQ(B(v), a);
        EXPECT_EQ(C(v), a);
        EXPECT_EQ(D(v), a);

        EXPECT_EQ((A{1,2,3,4}), a);
        EXPECT_EQ((B{1,2,3,4}), a);

        data[0] = 0;
        EXPECT_EQ(e, (B{0,2,3,4}));
        EXPECT_EQ((a = e), e);
        EXPECT_EQ((b = e), e);
        EXPECT_EQ((c = e), e);
        EXPECT_EQ((d = e), e);

        EXPECT_EQ((a = 1), A{1});
        EXPECT_EQ((b = 1), A{1});
        EXPECT_EQ((c = 1), A{1});
        EXPECT_EQ((d = 1), A{1});
        EXPECT_EQ((e = 1), A{1});

        EXPECT_EQ((a = v), (A{1,2,3,4}));
        EXPECT_EQ((b = v), a);
        EXPECT_EQ((c = v), a);
        EXPECT_EQ((d = v), a);
        EXPECT_EQ((e = v), a);

        b[s-1] = 5;
        EXPECT_EQ(b, (A{1,2,3,5}));
        EXPECT_EQ((a = b), b);
        EXPECT_EQ((c = b), b);
        EXPECT_EQ((d = b), b);
        EXPECT_EQ((e = b), b);

        data[0] = 0;
        EXPECT_EQ((c = data), (A{0,2,3,5}));
        EXPECT_EQ((a = data), c);
        EXPECT_EQ((b = data), c);
        EXPECT_EQ((d = data), c);
        EXPECT_EQ((e = data), c);

        d.assign(v.begin(), v.end());
        EXPECT_EQ(d, (A{1,2,3,4}));
        a.assign(v.begin(), v.end());
        EXPECT_EQ(a, d);
        b.assign(v.begin(), v.end());
        EXPECT_EQ(b, d);
        c.assign(v.begin(), v.end());
        EXPECT_EQ(c, d);
        e.assign(v.begin(), v.end());
        EXPECT_EQ(e, d);
    }

    TEST(tiny_vector, arithmetic)
    {
        BV bv3{1, 2, 128},
           bv0(SIZE),
           bv1(SIZE, 1);
        IV iv3{1, 2, 128},
           iv0(SIZE),
           iv1(SIZE, 1),
           ivn{-1, -2, -128};
        FV fv3{1.0f, 2.25f, 128.5f},
           fv0(SIZE),
           fv1(SIZE, 1.0f),
           fvn{-1.0f, -2.25f, -128.5f};

        EXPECT_EQ(+iv3, iv3);
        EXPECT_EQ(-iv3, ivn);
        EXPECT_EQ(-fv3, fvn);

        EXPECT_EQ(bv0 + bv1, bv1);
        EXPECT_EQ(bv0 + 1.25, fv1 + 0.25);
        EXPECT_EQ(1.25 + bv0, 0.25 + fv1);
        EXPECT_EQ(bv3 + bv1, (BV{2, 3, 129}));
        EXPECT_EQ(bv3 + bv3, (IV{2, 4, 256}));

        EXPECT_EQ(bv1 - bv1, bv0);
        EXPECT_EQ(bv3 - iv3, bv0);
        EXPECT_EQ(fv3 - fv3, fv0);
        EXPECT_EQ(bv1 - 1.0, fv0);
        EXPECT_EQ(1.0 - bv1, fv0);
        EXPECT_EQ(bv3 - bv1, (BV{0, 1, 127}));
        EXPECT_EQ(bv0 - iv1, -iv1);

        EXPECT_EQ(bv1 * bv1, bv1);
        EXPECT_EQ(bv1 * 1.0, fv1);
        EXPECT_EQ(bv3 * 0.5, (FV{0.5, 1.0, 64.0}));
        EXPECT_EQ(1.0 * bv1, fv1);
        EXPECT_EQ(bv3 * bv3, (IV{1, 4, 16384}));

        EXPECT_EQ(bv3 / bv3, bv1);
        EXPECT_EQ(bv1 / 1.0, fv1);
        EXPECT_EQ(1.0 / bv1, fv1);
        EXPECT_EQ(1.0 / bv3, (FV{1.0, 0.5, 1.0/128.0}));
        EXPECT_EQ(bv3 / 2, (IV{0, 1, 64}));
        EXPECT_EQ(bv3 / 2.0, (FV{0.5, 1.0, 64.0}));
        EXPECT_EQ(fv3 / 2.0, (FV{0.5, 1.125, 64.25}));
        EXPECT_EQ((2.0 * fv3) / 2.0, fv3);

        EXPECT_EQ(bv3 % 2, (BV{1, 0, 0}));
        EXPECT_EQ(iv3 % iv3, iv0);
        EXPECT_EQ(3   % bv3, (BV{0, 1, 3}));
        EXPECT_EQ(iv3 % (iv3 + iv1), iv3);

        BV bvp = (bv3 + bv3)*0.5;
        FV fvp = (fv3 + fv3)*0.5;
        EXPECT_EQ(bvp, bv3);
        EXPECT_EQ(fvp, fv3);
        bvp = 2.0*bv3 - bv3;
        fvp = 2.0*fv3 - fv3;
        EXPECT_EQ(bvp, bv3);
        EXPECT_EQ(fvp, fv3);
    }

    TEST(tiny_vector, algebraic)
    {
        BV bv3{1, 2, 4},
           bv0(SIZE),
           bv1(SIZE, 1);
        IV iv3{1, 2, 4},
           iv0(SIZE),
           iv1(SIZE, 1),
           ivn{-1, -2, -4};
        FV fv3{1.0f, 2.25f, 4.5f},
           fv0(SIZE),
           fv1(SIZE, 1.0f),
           fvn{-1.0f, -2.25f, -4.5f};

        EXPECT_EQ(abs(bv3), bv3);
        EXPECT_EQ(abs(iv3), iv3);
        EXPECT_EQ(abs(fv3), fv3);

        EXPECT_EQ(floor(fv3), (FV{1.0, 2.0, 4.0}));
        EXPECT_EQ(ceil(fv3), (FV{1.0, 3.0, 5.0}));
        EXPECT_EQ(-ceil(-fv3), (FV{1.0, 2.0, 4.0}));
        EXPECT_EQ(round(fv3), (FV{1.0, 2.0, 5.0}));
        EXPECT_EQ(sqrt(fv3*fv3), fv3);
        EXPECT_TRUE(all_close(cbrt(pow(fv3, 3)), fv3));

        tiny_vector<int, 4> src{ 1, 2, -3, -4 }, signs{ 2, -3, 4, -5 };
        EXPECT_EQ(copysign(src, signs), (tiny_vector<int, 4>{1, -2, 3, -4}));

        tiny_vector<double, 3> left{ 3., 5., 8. }, right{ 4., 12., 15. };
        EXPECT_EQ(hypot(left, right), (tiny_vector<double, 3>{5., 13., 17.}));

        EXPECT_EQ(sum(iv3),  7);
        EXPECT_EQ(sum(fv3),  7.75f);
        EXPECT_EQ(prod(iv3), 8);
        EXPECT_EQ(prod(fv3), 10.125f);
        EXPECT_NEAR(mean(iv3), 7.0 / SIZE, 1e-7);
        EXPECT_EQ(cumsum(bv3), (IV{1, 3, 7}));
        EXPECT_EQ(cumprod(bv3), (IV{1, 2, 8}));

        EXPECT_EQ(min(iv3, fv3), (FV{1.0, 2.0, 4.0}));
        EXPECT_EQ(min(3.0, fv3), (FV{1.0, 2.25, 3.0}));
        EXPECT_EQ(min(fv3, 3.0), (FV{1.0, 2.25, 3.0}));
        EXPECT_EQ(min(iv3), 1);
        EXPECT_EQ(min(fv3), 1.0f);
        EXPECT_EQ(max(iv3, fv3), (FV{1.0, 2.25, 4.5}));
        EXPECT_EQ(max(3.0, fv3), (FV{3.0, 3.0, 4.5}));
        EXPECT_EQ(max(fv3, 3.0), (FV{3.0, 3.0, 4.5}));
        EXPECT_EQ(max(iv3), 4);
        EXPECT_EQ(max(fv3), 4.5f);

        EXPECT_EQ(clip_lower(iv3, 0), iv3);
        EXPECT_EQ(clip_lower(iv3, 11), IV{ 11 });
        EXPECT_EQ(clip_upper(iv3, 0), IV{ 0 });
        EXPECT_EQ(clip_upper(iv3, 11), iv3);
        EXPECT_EQ(clip(iv3, 0, 11), iv3);
        EXPECT_EQ(clip(iv3, 11, 12), IV{ 11 });
        EXPECT_EQ(clip(iv3, -1, 0), IV{ 0 });
        EXPECT_EQ(clip(iv3, IV{0 }, IV{11}), iv3);
        EXPECT_EQ(clip(iv3, IV{11}, IV{12}), IV{11});
        EXPECT_EQ(clip(iv3, IV{-1}, IV{0 }), IV{0 });

        EXPECT_EQ(dot(iv3, iv3), 21);
        EXPECT_EQ(dot(fv1, fv3), sum(fv3));

        EXPECT_EQ(cross(bv3, bv3), iv0);
        EXPECT_EQ(cross(iv3, bv3), iv0);
        EXPECT_EQ(cross(fv3, fv3), fv0);
        EXPECT_EQ(cross(fv1, fv3), (FV{ 2.25, -3.5, 1.25 }));
    }

    TEST(tiny_vector, norm)
    {
        using math::sqrt;

        BV bv3{0, 2, 200},
           bv0(SIZE),
           bv1(SIZE, 1);
        IV iv3{1, 2, 2},
           iv0(SIZE),
           iv1(SIZE, 1),
           ivn{-1, -2, -2};
        FV fv3{1.0f, 2.25f, 4.5f},
           fv0(SIZE),
           fv1(SIZE, 1.0f),
           fvn{0.0f, -2.25f, -4.5f};

        EXPECT_EQ(norm_l0(bv0), 0);
        EXPECT_EQ(norm_l0(iv0), 0);
        EXPECT_EQ(norm_l0(fv0), 0);
        EXPECT_EQ(norm_l0(bv3), 2);
        EXPECT_EQ(norm_l0(iv3), 3);
        EXPECT_EQ(norm_l0(fv3), 3);
        EXPECT_EQ(norm_l0(ivn), 3);
        EXPECT_EQ(norm_l0(fvn), 2);

        EXPECT_EQ(norm_l1(bv0), 0);
        EXPECT_EQ(norm_l1(iv0), 0);
        EXPECT_EQ(norm_l1(fv0), 0.0);
        EXPECT_EQ(norm_l1(bv3), 202);
        EXPECT_EQ(norm_l1(iv3), 5);
        EXPECT_EQ(norm_l1(fv3), 7.75);
        EXPECT_EQ(norm_l1(ivn), 5);
        EXPECT_EQ(norm_l1(fvn), 6.75);

        EXPECT_EQ(norm_lp(iv3, 1), 5);
        EXPECT_EQ(norm_lp(fv3, 1), 7.75);
        EXPECT_EQ(norm_lp(ivn, 1), 5);
        EXPECT_EQ(norm_lp(fvn, 1), 6.75);

        EXPECT_EQ(norm_sq(bv0), 0);
        EXPECT_EQ(norm_sq(iv0), 0);
        EXPECT_EQ(norm_sq(fv0), 0.0);
        EXPECT_EQ(norm_sq(bv1), SIZE);
        EXPECT_EQ(norm_sq(iv1), SIZE);
        EXPECT_EQ(norm_sq(fv1), (float)SIZE);
        EXPECT_EQ(norm_sq(bv3), 40004);
        EXPECT_EQ(norm_sq(iv3), 9);
        EXPECT_EQ(norm_sq(fv3), 26.3125);
        EXPECT_EQ(norm_sq(fv3), sum(fv3*fv3));
        EXPECT_EQ(norm_sq(fv3), dot(fv3, fv3));
        EXPECT_EQ(norm_sq(ivn), 9);
        EXPECT_EQ(norm_sq(fvn), 25.3125);

        EXPECT_EQ(norm_l2(iv3), 3.0);
        EXPECT_NEAR(norm_l2(fv3), sqrt(26.3125), 1e-14);

        EXPECT_EQ(norm_lp(iv3, 2), 3.0);

        EXPECT_EQ(norm_linf(bv0), 0);
        EXPECT_EQ(norm_linf(iv0), 0);
        EXPECT_EQ(norm_linf(fv0), 0.0);
        EXPECT_EQ(norm_linf(bv3), 200);
        EXPECT_EQ(norm_linf(iv3), 2);
        EXPECT_EQ(norm_linf(fv3), 4.5);
        EXPECT_EQ(norm_linf(ivn), 2);
        EXPECT_EQ(norm_linf(fvn), 4.5);

        tiny_vector<IV, 3> ivv{ iv3, iv3, iv3 };
        EXPECT_EQ(norm_sq(ivv), 3 * norm_sq(iv3));
        EXPECT_NEAR(norm_l2(ivv), sqrt(3.0*norm_sq(iv3)), 1e-14);
    }

    TEST(tiny_vector, traits)
    {
        EXPECT_TRUE((tiny_vector<int, runtime_size>::may_use_uninitialized_memory));
        EXPECT_FALSE((tiny_vector<IV, runtime_size>::may_use_uninitialized_memory));

        EXPECT_TRUE((std::is_same<IV, promote_type_t<BV>>::value));
        EXPECT_TRUE((std::is_same<tiny_vector<double, 3>, real_promote_type_t<IV>>::value));
        EXPECT_TRUE((std::is_same<tiny_vector<double, 3>, rebind_container_t<IV, double>>::value));

        EXPECT_TRUE((std::is_same<uint64_t, squared_norm_type_t<tiny_vector<int, 1> > >::value));
        EXPECT_TRUE((std::is_same<uint64_t, squared_norm_type_t<tiny_vector<tiny_vector<int, 1>, 1> > >::value));
        EXPECT_TRUE((std::is_same<uint64_t, squared_norm_type_t<tiny_vector<tiny_vector<int, 1>, 1> > >::value));
        EXPECT_TRUE((std::is_same<double, norm_type_t<tiny_vector<int, 1> > >::value));
        EXPECT_TRUE((std::is_same<double, norm_type_t<tiny_vector<tiny_vector<int, 1>, 1> > >::value));
        EXPECT_TRUE((std::is_same<tiny_vector<double, SIZE>, decltype(cos(IV()))>::value));
    }
} // namespace xvigra
