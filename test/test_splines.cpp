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

#include "unittest.hpp"
#include <xvigra/splines.hpp>

namespace xvigra
{
    using spline_types = testing::Types<b_spline<0, double>,
                                        b_spline<1, double>,
                                        b_spline<2, double>,
                                        b_spline<3, double>,
                                        b_spline<4, double>
                                        >;

    TYPED_TEST_SETUP(spline_test, spline_types);

    TYPED_TEST(spline_test, values)
    {
        using BS = TypeParam;
        static constexpr int ORDER = BS::static_order;
        using BSB = b_spline_base<ORDER, double>;

        BS spline;
        BSB spline_base;

        // test principle: spline uses specialized code, spline_base uses a generic algorithm;
        //                 both should give the same results.

        double r = spline.radius();
        EXPECT_EQ(r, spline_base.radius());

        for(int d = 0; d <= ORDER+1; ++d)
        {
            for(double x = -r-0.5; x <= r+0.5; x += 0.5)
            {
                EXPECT_NEAR(spline(x, d), spline_base(x, d), 1e-15);
            }
        }
    }

    TYPED_TEST(spline_test, prefilter_coefficients)
    {
        using BS = TypeParam;
        static constexpr int ORDER = BS::static_order;

        using BSB = b_spline_base<ORDER, double>;

        BS spline;
        BSB spline_base;

        int n = ORDER / 2;
        std::vector<double> const & ps = spline.prefilter_coefficients();
        std::vector<double> const & psb = spline_base.prefilter_coefficients();

        if(n == 0)
        {
            EXPECT_EQ(ps.size(), 0u);
            EXPECT_EQ(psb.size(), 0u);
        }
        else
        {
            std::vector<double> & psb1 = const_cast<std::vector<double> &>(psb);
            std::sort(psb1.begin(), psb1.end());

            for(int i = 0; i < n; ++i)
            {
                EXPECT_NEAR(ps[i], psb[i], 1e-14);
            }
        }
    }

    TYPED_TEST(spline_test, weight_matrix)
    {
        using BS = TypeParam;
        static constexpr int ORDER = BS::static_order;

        using BSB = b_spline_base<ORDER, double>;

        BS spline;
        BSB spline_base;

        int n = ORDER + 1;
        typename BS::weight_matrix_type const & ws = BS::weights();
        typename BSB::weight_matrix_type const & wsb = BSB::weights();

        for(int d = 0; d < n; ++d)
        {
            for(int i = 0; i < n; ++i)
            {
                EXPECT_NEAR(ws[d][i], wsb[d][i], 1e-14);
            }
        }
    }
} // namespace xvigra
